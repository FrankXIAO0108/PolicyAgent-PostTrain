from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.training.assemble_scripted_replay_corrections import (
    build_correction_payload,
    validate_replay_evidence,
)
from src.training.correction_validation import sha256


class AssembleScriptedReplayCorrectionsTests(unittest.TestCase):
    def test_replay_evidence_rejects_tool_result_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            corrected = Path(temp_dir) / "corrected.json"
            corrected.write_text("{}\n", encoding="utf-8")
            replay = {
                "status": "COMPLETED",
                "task_id": "38",
                "spec": {"sha256_lf": "SPEC"},
                "result": {"replay_seed_matches_spec": True},
                "tool_result_mismatches": 1,
                "protocol": {
                    "tool_result_pairs_ok": True,
                    "mixed_messages": 0,
                },
                "corrected_messages": {"sha256": sha256(corrected)},
            }

            with self.assertRaisesRegex(ValueError, "tool-result mismatches"):
                validate_replay_evidence(
                    replay,
                    corrected,
                    expected_task_id="38",
                    expected_spec_sha256_lf="SPEC",
                )

    def test_payload_binds_replay_and_run_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            policy = root / "policy.md"
            policy.write_text("policy\n", encoding="utf-8")
            source = root / "source.json"
            source.write_text("{}\n", encoding="utf-8")
            replay_path = root / "replay.json"
            replay_path.write_text("{}\n", encoding="utf-8")
            run_path = root / "run.json"
            run_path.write_text("{}\n", encoding="utf-8")
            spec = {
                "task_id": "38",
                "author_id": "pipeline-author",
                "source": {"sha256": sha256(source)},
                "policy": {"sha256": sha256(policy)},
                "change_log": [{"category": "test", "reason": "test"}],
            }

            payload = build_correction_payload(
                spec=spec,
                source_path=source,
                policy_path=policy,
                messages=[
                    {"role": "user", "content": "help"},
                    {"role": "assistant", "content": "done"},
                ],
                replay_path=replay_path,
                replay={"result": {"reward": 0}, "state": {"hash": "x"}},
                run_manifest_path=run_path,
                run_manifest_sha256=sha256(run_path),
            )

            self.assertEqual(payload["generation_mode"], "ENVIRONMENT_REPLAY")
            self.assertEqual(payload["replay_manifest"]["sha256"], sha256(replay_path))
            self.assertEqual(
                payload["replay_evidence"]["run_manifest"]["sha256"],
                sha256(run_path),
            )
            self.assertEqual(payload["system_policy"], "policy\n")


if __name__ == "__main__":
    unittest.main()
