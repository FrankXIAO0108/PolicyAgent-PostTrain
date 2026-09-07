import json
from copy import deepcopy
from pathlib import Path

import pytest

from src.evaluation.task44_reward_evidence import (
    identity_evidence,
    confirmation_evidence,
)
from src.training.run_retail_agentic_grpo import (
    validate_config_and_split,
    validate_optimization_contract,
)


def tool(name, args, result, cid):
    return [
        {
            "role": "assistant",
            "tool_calls": [{"id": cid, "name": name, "arguments": args}],
        },
        {
            "role": "tool",
            "id": cid,
            "name": name,
            "error": False,
            "content": result if isinstance(result, str) else json.dumps(result),
        },
    ]


def fixture():
    before = {
        "order_id": "o1",
        "user_id": "u1",
        "payment_history": [
            {
                "transaction_type": "payment",
                "payment_method_id": "gift_card_1234",
                "amount": 153.23,
            }
        ],
    }
    after = deepcopy(before)
    after["payment_history"].append(
        {
            "transaction_type": "refund",
            "payment_method_id": "gift_card_1234",
            "amount": 17.98999999999998,
        }
    )
    return (
        tool("find_user_id_by_name_zip", {}, "u1", "auth")
        + tool("get_order_details", {"order_id": "o1"}, before, "order")
        + [
            {
                "role": "assistant",
                "content": "Order o1: replace 9190635437 with 5320792178. Refund $17.99 to gift card ending in 1234. Would you like to proceed?",
            },
            {"role": "user", "content": "Yes, please go ahead."},
        ]
        + tool(
            "modify_pending_order_items",
            {
                "order_id": "o1",
                "item_ids": ["9190635437"],
                "new_item_ids": ["5320792178"],
                "payment_method_id": "gift_card_1234",
            },
            after,
            "write",
        )
    )


def identity(messages):
    return identity_evidence(
        messages, {"required_user_id": "u1", "required_order_ids": ["o1"]}
    )


def check(messages):
    return confirmation_evidence(messages)["checks"][0]


def test_order_ownership_is_sufficient_without_profile():
    assert identity(fixture())["complete"]


@pytest.mark.parametrize(
    "mutation", ["wrong_owner", "no_auth", "error", "duplicate", "late", "wrong_order"]
)
def test_identity_rejects_invalid_or_late_evidence(mutation):
    m = fixture()
    if mutation == "wrong_owner":
        d = json.loads(m[3]["content"])
        d["user_id"] = "other"
        m[3]["content"] = json.dumps(d)
    elif mutation == "wrong_order":
        d = json.loads(m[3]["content"])
        d["order_id"] = "other"
        m[3]["content"] = json.dumps(d)
    elif mutation == "no_auth":
        m = m[2:]
    elif mutation == "error":
        m[1]["error"] = True
    elif mutation == "duplicate":
        m.insert(4, deepcopy(m[3]))
    else:
        m = m[2:] + m[:2]
    assert not identity(m)["complete"]


@pytest.mark.parametrize(
    "ask",
    [
        "Would you like to proceed?",
        "Would you like me to proceed?",
        "Let me confirm the details before proceeding.",
        "I'll now confirm the details and ask for your approval before proceeding.",
        "Before proceeding, I need to confirm this change.",
    ],
)
def test_equivalent_confirmation_and_order_backed_payment_alias(ask):
    m = fixture()
    m[4]["content"] = m[4]["content"].split("Would")[0] + ask
    assert check(m)["verified_verdict"] == "PASS"


@pytest.mark.parametrize(
    "reply",
    [
        "No, do not proceed.",
        "Yes, but don't proceed.",
        "I cannot confirm this.",
        "Wait, not yet.",
    ],
)
def test_refusal_is_not_authorization(reply):
    m = fixture()
    m[5]["content"] = reply
    assert check(m)["verified_verdict"] == "FAIL"


@pytest.mark.parametrize(
    "reply",
    [
        "Yes, but change to another lamp.",
        "Yes, instead use 7453605304.",
        "Yes, also add a keyboard.",
        "Yes, replace it with 7453605304.",
        "Sounds wonderful.",
    ],
)
def test_changed_or_unknown_reply_is_review_not_pass(reply):
    m = fixture()
    m[5]["content"] = reply
    assert check(m)["verified_verdict"] == "REVIEW"


def test_refund_mismatch_is_failure_even_if_final_answer_corrects_it():
    m = fixture()
    m[4]["content"] = m[4]["content"].replace("17.99", "18.99")
    m += [{"role": "assistant", "content": "Sorry, the refund is actually $17.99."}]
    assert check(m)["refund_binding"]["verdict"] == "FAIL"
    assert check(m)["verified_verdict"] == "FAIL"


def test_user_confirmed_wrong_amount_is_detected():
    m = fixture()
    m[5]["content"] = "Yes, I confirm the $18.99 refund."
    assert check(m)["verified_verdict"] == "FAIL"


def test_prices_and_balance_are_not_refund_claims():
    m = fixture()
    m[4]["content"] += " Old price $153.23, new price $135.24, balance $34.99."
    assert check(m)["verified_verdict"] == "PASS"


@pytest.mark.parametrize(
    "mutation",
    [
        "no_amount",
        "missing_result",
        "wrong_payment",
        "changed_after_yes",
        "no_yes",
        "unbound_order",
    ],
)
def test_incomplete_binding_never_passes(mutation):
    m = fixture()
    if mutation == "no_amount":
        m[4]["content"] = m[4]["content"].replace(
            "Refund $17.99", "Refund the difference"
        )
    elif mutation == "missing_result":
        m.pop()
    elif mutation == "wrong_payment":
        d = json.loads(m[-1]["content"])
        d["payment_history"][-1]["payment_method_id"] = "other"
        m[-1]["content"] = json.dumps(d)
    elif mutation == "changed_after_yes":
        m.insert(
            6, {"role": "assistant", "content": "I will use another payment method."}
        )
    elif mutation == "no_yes":
        m.pop(5)
    else:
        m[3]["id"] = "unbound"
    assert check(m)["verified_verdict"] != "PASS"


def test_v2_score_entrypoint_checks_amount_instead_of_trusting_legacy_pass():
    from src.evaluation.staged_reward_shadow import score_rollout

    root = Path(__file__).resolve().parents[1]
    path = (
        root
        / "configs/retail_agentic_qwen3_4b_task44_staged_v6_n4_50step_c8192_v2.json"
    )
    spec = json.loads(path.read_text(encoding="utf-8"))["reward"]["staged_reward_spec"]
    spec["tasks"]["44"]["identity_link"] = {
        "required_user_id": "u1",
        "required_order_ids": ["o1"],
    }
    m = fixture()
    m[4]["content"] = m[4]["content"].replace("17.99", "18.99")
    raw = {"task_id": "44", "messages": m, "evidence_sha256": "bound"}
    ev = {
        "task_id": "44",
        "evidence_sha256": "bound",
        "tool_trace": [],
        "completion": {},
        "terminal_evaluator": {
            "reward": 1,
            "user_stopped": True,
            "tau2": {"environment": {"reward": 1}},
            "action_progress": {
                "matches": [
                    {
                        "action_id": "44_4",
                        "matched": True,
                        "matched_call_index": 2,
                        "name": "modify_pending_order_items",
                    }
                ]
            },
        },
    }
    score = score_rollout(
        raw,
        ev,
        spec,
        {"checks": [{"confirmed": True, "parameter_binding": {"verdict": "PASS"}}]},
    )
    assert score["components"]["confirmation_binding"]["verdict"] == "FAIL"
    assert score["staged_reward"] <= 0.15
    from types import SimpleNamespace
    from src.rl.retail_agentic_env import tiered_terminal_process_reward

    converted = [
        SimpleNamespace(
            **{
                **message,
                "tool_calls": [
                    SimpleNamespace(**call) for call in message.get("tool_calls") or []
                ],
            }
        )
        for message in m
    ]
    online = tiered_terminal_process_reward(
        task_id="44",
        messages=converted,
        action_progress=ev["terminal_evaluator"]["action_progress"],
        environment_payload={"reward": 1},
        communication_payload={},
        environment_state_reward=1,
        user_stopped=True,
        completion={},
        staged_reward_spec=spec,
    )
    assert online["components"]["confirmation_binding"]["verdict"] == "FAIL"
    assert online["staged_reward"] == score["staged_reward"]
    assert (
        score["confirmation_evidence"]["checks"][0]["refund_binding"]["reason"]
        == "CONFIRMED_REFUND_CONFLICT"
    )


def test_v2_only_changes_reward_evidence_version():
    root = Path(__file__).resolve().parents[1]
    stem = "configs/retail_agentic_qwen3_4b_task44_staged_v6_n4_50step_c8192_"
    old = json.loads((root / (stem + "v1.json")).read_text(encoding="utf-8"))
    newpath = root / (stem + "v2.json")
    new = json.loads(newpath.read_text(encoding="utf-8"))
    spec = new["reward"]["staged_reward_spec"]
    assert spec.pop("evidence_rules_version") == "task44_evidence_v2"
    spec["spec_id"] = old["reward"]["staged_reward_spec"]["spec_id"]
    assert new == old
    valid = validate_config_and_split(newpath)["config"]
    assert (
        validate_optimization_contract(valid, selected_task_count=1)[
            "expected_rollouts"
        ]
        == 200
    )
