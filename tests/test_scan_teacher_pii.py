import json
import tempfile
import unittest
from pathlib import Path

from src.training.scan_teacher_pii import (
    collect_internal_identifiers,
    scan_directory,
    scan_messages,
    scan_paths,
    scan_trajectory,
    summarize,
    write_outputs,
)


def messages_with_payment_leak(intermediate: bool = True, final: bool = True):
    rows = [
        {"role": "user", "content": "Please exchange both items."},
    ]
    if intermediate:
        rows.append(
            {
                "role": "assistant",
                "content": "I can see you have a credit card on file (credit_card_9513926).",
            }
        )
    rows.append(
        {
            "role": "assistant",
            "content": "Let me look up the order.",
            "tool_calls": [{"id": "c1", "name": "find_order", "arguments": {"order_id": "#W2378156"}}],
        }
    )
    rows.append({"role": "tool", "id": "c1", "content": json.dumps({"order_id": "#W2378156"})})
    if final:
        rows.append(
            {
                "role": "assistant",
                "content": "Refund of $16.63 to your credit card (credit_card_9513926).",
            }
        )
    return rows


class ScanTeacherPiiTests(unittest.TestCase):
    def test_clean_trajectory_passes(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "I can help with your exchange."},
        ]
        hits = scan_messages(messages)
        self.assertEqual(hits, [])

    def test_payment_id_in_intermediate_text_is_hit(self):
        messages = messages_with_payment_leak(final=False)
        hits = scan_messages(messages)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].category, "payment_method_id")
        self.assertEqual(hits[0].identifier, "credit_card_9513926")
        self.assertEqual(hits[0].message_index, 1)

    def test_payment_id_in_final_text_is_hit(self):
        messages = messages_with_payment_leak(intermediate=False)
        hits = scan_messages(messages)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].message_index, 3)

    def test_hallucinated_payment_id_caught_by_safety_net(self):
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "Paying with credit_card_0000000 now.",
            },
        ]
        hits = scan_messages(messages)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].identifier, "credit_card_0000000")
    def test_paypal_id_in_text_is_hit(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Refund via paypal_2433177."},
        ]
        hits = scan_messages(messages)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].category, "payment_method_id")
        self.assertEqual(hits[0].identifier, "paypal_2433177")

    def test_paypal_id_collected_from_payload(self):
        payload = {
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": "Let me check.",
                    "tool_calls": [{"id": "t1", "name": "find_user", "arguments": {}}],
                },
                {
                    "role": "tool",
                    "id": "t1",
                    "content": json.dumps(
                        {"payment_methods": {"paypal_2433177": {"type": "paypal"}}}
                    ),
                },
                {"role": "assistant", "content": "Your refund method is paypal_2433177."},
            ]
        }
        identifiers = collect_internal_identifiers(payload)
        self.assertIn("paypal_2433177", identifiers["payment_method_id"])
        hits = scan_messages(payload["messages"])
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].identifier, "paypal_2433177")

    def test_tool_results_are_never_flagged(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Let me check.", "tool_calls": [{"id": "t1", "name": "find_user", "arguments": {}}]},
            {
                "role": "tool",
                "id": "t1",
                "content": json.dumps(
                    {
                        "user_id": "yusuf_rossi_9620",
                        "email": "yusuf.rossi7301@example.com",
                        "payment_methods": {"credit_card_9513926": {"brand": "mastercard"}},
                    }
                ),
            },
            {"role": "assistant", "content": "Thanks, I have your account."},
        ]
        hits = scan_messages(messages)
        self.assertEqual(hits, [])

    def test_user_id_echoed_in_text_is_hit(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Let me check.", "tool_calls": [{"id": "t1", "name": "find_user", "arguments": {}}]},
            {
                "role": "tool",
                "id": "t1",
                "content": json.dumps({"user_id": "yusuf_rossi_9620"}),
            },
            {"role": "assistant", "content": "Your account id is yusuf_rossi_9620."},
        ]
        hits = scan_messages(messages)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].category, "user_id")
        self.assertEqual(hits[0].identifier, "yusuf_rossi_9620")

    def test_email_echoed_in_text_is_hit(self):
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "I will email yusuf.rossi7301@example.com the receipt.",
            },
        ]
        hits = scan_messages(messages)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].category, "email")

    def test_customer_facing_business_data_not_flagged(self):
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": (
                    "Order #W2378156: keyboard item 7706410293 and "
                    "thermostat item 7747408585 are available."
                ),
            },
        ]
        hits = scan_messages(messages)
        self.assertEqual(hits, [])

    def test_free_text_user_id_lookalike_not_flagged_without_payload_evidence(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "double_click_2024 is not a real id."},
        ]
        hits = scan_messages(messages)
        self.assertEqual(hits, [])

    def test_on_tool_turn_narration_is_marked(self):
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "card credit_card_9513926",
                "tool_calls": [{"id": "t1", "name": "pay", "arguments": {"payment_method_id": "credit_card_9513926"}}],
            },
            {"role": "tool", "id": "t1", "content": "ok"},
        ]
        hits = scan_messages(messages)
        self.assertEqual(len(hits), 1)
        self.assertTrue(hits[0].on_tool_turn)

    def test_supported_payload_formats(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            envelope = {
                "candidate_id": "e0166c34-81d4-4cae-92d0-0d3c9c2b391c",
                "task_id": "0",
                "simulations": [{"candidate_id": None, "task_id": "0", "messages": messages_with_payment_leak(final=False)}],
            }
            source_file = root / "source_c.json"
            source_file.write_text(json.dumps(envelope), encoding="utf-8")
            row = scan_trajectory(envelope, source_file)
            self.assertEqual(row["candidate_id"], "e0166c34-81d4-4cae-92d0-0d3c9c2b391c")
            self.assertEqual(row["task_id"], "0")
            self.assertEqual(row["status"], "HITS")
            corrected = {
                "candidate_id": "c1",
                "task_id": "7",
                "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "done"}],
            }
            corrected_file = root / "corrected_c1.json"
            corrected_file.write_text(json.dumps(corrected), encoding="utf-8")
            row = scan_trajectory(corrected, corrected_file)
            self.assertEqual(row["task_id"], "7")
            self.assertEqual(row["status"], "CLEAN")

    def test_no_assistant_text_raises(self):
        with self.assertRaises(ValueError):
            scan_trajectory(
                {"messages": [{"role": "user", "content": "hi"}]},
                Path("bad.json"),
            )

    def test_directory_scan_reports_errors_for_non_trajectories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "traj.json").write_text(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "hi"},
                            {"role": "assistant", "content": "ok"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (root / "manifest.json").write_text('{"not": "a trajectory"}', encoding="utf-8")
            rows = scan_directory(root)
            self.assertEqual(summarize(rows)["errors"], 1)
            self.assertEqual(summarize(rows)["clean"], 1)

    def test_write_outputs_writes_report_and_hits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source_c.json"
            source.write_text(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "hi"},
                            {"role": "assistant", "content": "card credit_card_9513926"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            rows = scan_paths([source])
            out = root / "scan_out"
            write_outputs(rows, out)
            report = json.loads((out / "pii_scan_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["summary"]["hits"], 1)
            hits = (out / "pii_hits.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(hits), 1)
            with self.assertRaises(FileExistsError):
                write_outputs(rows, out)

    def test_scan_jsonl_scans_each_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jsonl_file = root / "candidate_trajectories.jsonl"
            lines = [
                json.dumps({"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]}),
                json.dumps({"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "card credit_card_1234567"}]}),
                json.dumps({"messages": [{"role": "user", "content": "hi"}]}),
            ]
            jsonl_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
            rows = scan_paths([jsonl_file])
            self.assertEqual([r["status"] for r in rows], ["CLEAN", "HITS", "ERROR"])
            self.assertEqual(rows[1]["source_line"], 1)
            self.assertTrue(rows[1]["candidate_id"].endswith("#1"))

    def test_collect_internal_identifiers_from_payment_methods_keys(self):
        payload = {
            "user_id": "ivan_hernandez_6923",
            "payment_methods": {"credit_card_3095586": {"type": "card"}, "gift_card_9368765": {"type": "gift"}},
            "email": "ivan.hernandez@example.com",
            "product_id": "7706410293",
            "order_id": "#W2378156",
        }
        ids = collect_internal_identifiers(payload)
        self.assertEqual(ids["user_id"], {"ivan_hernandez_6923"})
        self.assertEqual(ids["payment_method_id"], {"credit_card_3095586", "gift_card_9368765"})
        self.assertEqual(ids["email"], {"ivan.hernandez@example.com"})


if __name__ == "__main__":
    unittest.main()
