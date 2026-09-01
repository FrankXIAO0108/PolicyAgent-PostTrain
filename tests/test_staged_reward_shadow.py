from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from src.evaluation.staged_reward_shadow import (
    _grounded_value_contains,
    _sha256,
    _strict_contains,
    build_report,
    score_rollout,
)


SPEC = {
    "reward": {
        "terminal_complete_success_score": 1.0,
        "terminal_incomplete_communication_cap": 0.9,
        "complete_success_efficiency_penalty_cap": 0.02,
        "nonterminal_component_weights": {
            "identity_link": 0.1,
            "target_evidence": 0.2,
            "required_write_progress": 0.5,
            "grounded_communication": 0.1,
        },
        "normalize_active_nonterminal_weights_to": 0.9,
        "no_correct_write_cap": 0.35,
        "tool_error_penalty_each": 0.03,
        "tool_error_penalty_cap": 0.09,
        "repeated_call_penalty_each": 0.02,
        "repeated_call_penalty_cap": 0.06,
        "limit_reached_penalty": 0.1,
        "unexpected_write_hard_cap": 0.0,
        "minimum": 0.0,
        "maximum": 1.0,
    },
    "tasks": {
        "1": {
            "identity_link": {
                "required_user_id": "u1",
                "required_order_ids": ["o1"],
            },
            "target_evidence_calls": [
                {
                    "evidence_id": "order",
                    "name": "get_order_details",
                    "arguments": {"order_id": "o1"},
                }
            ],
            "required_write_action_ids": ["1_write"],
            "communication_source": (
                "terminal_evaluator.tau2.communication.communicate_checks"
            ),
        }
    },
}


def call(name: str, arguments: dict, content: str, *, error: bool = False) -> dict:
    return {
        "name": name,
        "arguments": arguments,
        "result": {"content": content, "error": error},
    }


def fixtures(*, write: bool, terminal: bool = False) -> tuple[dict, dict]:
    messages = [
        {"role": "assistant", "content": None},
        {"role": "tool", "content": "u1", "error": False},
        {"role": "tool", "content": json.dumps({"orders": ["o1"]}), "error": False},
        {"role": "tool", "content": "tracking X1", "error": False},
        {"role": "assistant", "content": "The tracking number is X1."},
    ]
    if write:
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "write-1",
                            "name": "write_tool",
                            "arguments": {"order_id": "o1"},
                            "requestor": "assistant",
                        }
                    ],
                },
                {
                    "role": "tool",
                    "id": "write-1",
                    "name": "write_tool",
                    "content": "updated",
                    "error": False,
                    "requestor": "assistant",
                },
            ]
        )
    raw = {
        "task_id": "1",
        "messages": messages,
        "evidence_sha256": "BOUND",
    }
    evidence = {
        "task_id": "1",
        "evidence_sha256": "BOUND",
        "tool_trace": [
            call("find_user_id_by_name_zip", {}, "u1"),
            call("get_user_details", {"user_id": "u1"}, json.dumps({"orders": ["o1"]})),
            call("get_order_details", {"order_id": "o1"}, "tracking X1"),
        ],
        "completion": {
            "customer_turn_limit_reached": False,
            "tool_call_limit_reached": False,
        },
        "terminal_evaluator": {
            "reward": 1.0 if terminal else 0.0,
            "user_stopped": terminal,
            "action_progress": {
                "matches": [
                    {
                        "action_id": "1_write",
                        "name": "write_tool",
                        "matched": write,
                        "matched_call_index": 0 if write else None,
                    }
                ],
                "unexpected_write_count": 0,
            },
            "tau2": {
                "communication": {
                    "communicate_checks": [{"info": "X1", "met": True}]
                }
            },
        },
    }
    return raw, evidence


def v6_spec() -> dict:
    spec = deepcopy(SPEC)
    spec["reward"].update(
        {
            "composition_mode": "hierarchical_state_authorization_review_v6",
            "additive_component_weights": {
                "environment_state": 0.35,
                "interaction_complete": 0.10,
                "identity_link": 0.08,
                "target_evidence": 0.12,
                "write_authorization": 0.20,
                "post_write_communication": 0.15,
            },
            "no_verified_write_cap": 0.25,
            "authorization_review_value": 0.5,
            "authorization_review_cap": 0.75,
            "authorization_fail_hard_cap": 0.15,
        }
    )
    return spec


def confirmation_fixture(*, verdict: str = "PASS", confirmed: bool = True) -> dict:
    return {
        "write_count": 1,
        "confirmed_write_count": int(confirmed),
        "diagnostic_version": "v3_prompt_bound_confirmation_scope",
        "checks": [
            {
                "confirmed": confirmed,
                "parameter_binding": {"verdict": verdict},
            }
        ],
    }


class StagedRewardShadowTests(unittest.TestCase):
    def test_short_token_does_not_match_inside_unrelated_word(self) -> None:
        self.assertFalse(_strict_contains("Please provide your email.", "IL"))
        self.assertTrue(_strict_contains("Chicago, IL 60621", "IL"))

    def test_grounded_numeric_value_tolerates_serialized_float_noise(self) -> None:
        self.assertTrue(
            _grounded_value_contains(
                '{"transaction_type":"refund","amount":17.98999999999998}',
                "17.99",
            )
        )
        self.assertFalse(_grounded_value_contains("refund amount 17.98", "17.99"))
        self.assertFalse(
            _grounded_value_contains("unrelated natural-language value", "17.99")
        )

    def test_no_required_action_match_process_credit_is_capped(self) -> None:
        raw, evidence = fixtures(write=False)
        result = score_rollout(raw, evidence, SPEC)
        self.assertEqual(result["staged_reward"], 0.35)
        self.assertTrue(result["no_verified_write_cap_applied"])

    def test_required_action_match_raises_nonterminal_reward_above_cap(self) -> None:
        raw, evidence = fixtures(write=True)
        result = score_rollout(raw, evidence, SPEC)
        self.assertGreater(result["staged_reward"], 0.35)

    def test_action_match_without_bound_result_fails_closed(self) -> None:
        raw, evidence = fixtures(write=True)
        raw = deepcopy(raw)
        raw["messages"] = raw["messages"][:-1]
        with self.assertRaisesRegex(ValueError, "binding is ambiguous"):
            score_rollout(raw, evidence, SPEC)

    def test_duplicate_required_action_evidence_fails_closed(self) -> None:
        raw, evidence = fixtures(write=True)
        evidence = deepcopy(evidence)
        evidence["terminal_evaluator"]["action_progress"]["matches"].append(
            deepcopy(
                evidence["terminal_evaluator"]["action_progress"]["matches"][0]
            )
        )
        with self.assertRaisesRegex(ValueError, "must be unique"):
            score_rollout(raw, evidence, SPEC)

    def test_tool_reported_error_does_not_receive_write_credit(self) -> None:
        raw, evidence = fixtures(write=True)
        raw = deepcopy(raw)
        raw["messages"][-1]["error"] = True
        result = score_rollout(raw, evidence, SPEC)
        self.assertEqual(
            result["components"]["required_write_progress"]["value"], 0.0
        )
        self.assertEqual(result["staged_reward"], 0.35)

    def test_solvable_task_premature_transfer_is_hard_capped(self) -> None:
        raw, evidence = fixtures(write=False)
        raw = deepcopy(raw)
        evidence = deepcopy(evidence)
        raw["messages"].extend(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "transfer-1",
                            "name": "transfer_to_human_agents",
                            "arguments": {},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "id": "transfer-1",
                    "name": "transfer_to_human_agents",
                    "content": "Transfer successful",
                    "error": False,
                },
            ]
        )
        evidence["tool_trace"].append(
            call("transfer_to_human_agents", {}, "Transfer successful")
        )
        spec = deepcopy(SPEC)
        spec["tasks"]["1"]["premature_transfer_rule"] = {
            "enabled": True,
            "hard_cap": 0.0,
        }
        result = score_rollout(raw, evidence, spec)
        self.assertTrue(result["premature_transfer"])
        self.assertEqual(result["staged_reward"], 0.0)

    def test_terminal_success_is_one_when_no_penalty_applies(self) -> None:
        raw, evidence = fixtures(write=True, terminal=True)
        result = score_rollout(raw, evidence, SPEC)
        self.assertEqual(result["staged_reward"], 1.0)

    def test_additive_v3_does_not_use_terminal_override(self) -> None:
        raw, evidence = fixtures(write=True, terminal=True)
        spec = deepcopy(SPEC)
        spec["reward"].update(
            {
                "composition_mode": "additive_terminal_process_v3",
                "additive_component_weights": {
                    "terminal_environment": 0.45,
                    "identity_link": 0.08,
                    "target_evidence": 0.12,
                    "required_write_progress": 0.25,
                    "grounded_communication": 0.10,
                },
            }
        )
        result = score_rollout(raw, evidence, spec)
        self.assertEqual(result["staged_reward"], 1.0)
        self.assertFalse(result["terminal_override_applied"])
        self.assertEqual(
            result["composition_mode"], "additive_terminal_process_v3"
        )

    def test_additive_v3_terminal_success_cannot_hide_missing_communication(self) -> None:
        raw, evidence = fixtures(write=True, terminal=True)
        raw = deepcopy(raw)
        raw["messages"][4]["content"] = "The request is complete."
        spec = deepcopy(SPEC)
        spec["reward"].update(
            {
                "composition_mode": "additive_terminal_process_v3",
                "additive_component_weights": {
                    "terminal_environment": 0.45,
                    "identity_link": 0.08,
                    "target_evidence": 0.12,
                    "required_write_progress": 0.25,
                    "grounded_communication": 0.10,
                },
            }
        )
        result = score_rollout(raw, evidence, spec)
        self.assertEqual(result["staged_reward"], 0.9)
        self.assertFalse(result["terminal_override_applied"])

    def test_additive_v3_terminal_success_without_write_is_hard_capped(self) -> None:
        raw, evidence = fixtures(write=False, terminal=True)
        spec = deepcopy(SPEC)
        spec["reward"].update(
            {
                "composition_mode": "additive_terminal_process_v3",
                "additive_component_weights": {
                    "terminal_environment": 0.45,
                    "identity_link": 0.08,
                    "target_evidence": 0.12,
                    "required_write_progress": 0.25,
                    "grounded_communication": 0.10,
                },
            }
        )
        result = score_rollout(raw, evidence, spec)
        self.assertEqual(result["staged_reward"], 0.35)
        self.assertTrue(result["no_verified_write_cap_applied"])

    def test_additive_v4_confirmation_changes_success_ordering(self) -> None:
        raw, evidence = fixtures(write=True, terminal=True)
        spec = deepcopy(SPEC)
        spec["reward"].update(
            {
                "composition_mode": "additive_terminal_process_confirmation_v4",
                "additive_component_weights": {
                    "terminal_environment": 0.40,
                    "identity_link": 0.08,
                    "target_evidence": 0.12,
                    "required_write_progress": 0.20,
                    "grounded_communication": 0.10,
                    "confirmation_binding": 0.10,
                },
            }
        )
        passed = {
            "write_count": 1,
            "confirmed_write_count": 1,
            "diagnostic_version": "v3_prompt_bound_confirmation_scope",
            "checks": [
                {
                    "confirmed": True,
                    "parameter_binding": {"verdict": "PASS"},
                }
            ],
        }
        review = deepcopy(passed)
        review["checks"][0]["parameter_binding"]["verdict"] = "REVIEW"

        passed_score = score_rollout(
            raw, evidence, spec, confirmation_diagnostic=passed
        )
        review_score = score_rollout(
            raw, evidence, spec, confirmation_diagnostic=review
        )

        self.assertEqual(passed_score["staged_reward"], 1.0)
        self.assertEqual(review_score["staged_reward"], 0.9)
        self.assertGreater(passed_score["staged_reward"], review_score["staged_reward"])

    def test_additive_v4_fails_closed_without_bound_confirmation(self) -> None:
        raw, evidence = fixtures(write=True, terminal=True)
        spec = deepcopy(SPEC)
        spec["reward"].update(
            {
                "composition_mode": "additive_terminal_process_confirmation_v4",
                "additive_component_weights": {
                    "terminal_environment": 0.40,
                    "identity_link": 0.08,
                    "target_evidence": 0.12,
                    "required_write_progress": 0.20,
                    "grounded_communication": 0.10,
                    "confirmation_binding": 0.10,
                },
            }
        )

        with self.assertRaisesRegex(ValueError, "requires bound confirmation"):
            score_rollout(raw, evidence, spec)

    def test_hierarchical_v5_ranks_authorized_write_above_no_write_above_unauthorized_write(self) -> None:
        spec = deepcopy(SPEC)
        spec["reward"].update(
            {
                "composition_mode": "hierarchical_state_authorization_v5",
                "additive_component_weights": {
                    "environment_state": 0.35,
                    "interaction_complete": 0.10,
                    "identity_link": 0.08,
                    "target_evidence": 0.12,
                    "authorized_write": 0.20,
                    "post_write_communication": 0.15,
                },
                "no_verified_write_cap": 0.25,
                "unauthorized_write_hard_cap": 0.15,
            }
        )
        authorized = {
            "write_count": 1,
            "confirmed_write_count": 1,
            "diagnostic_version": "v3_prompt_bound_confirmation_scope",
            "checks": [
                {"confirmed": True, "parameter_binding": {"verdict": "PASS"}}
            ],
        }
        unauthorized = deepcopy(authorized)
        unauthorized["checks"][0]["parameter_binding"]["verdict"] = "REVIEW"

        raw_write, evidence_write = fixtures(write=True)
        raw_write["messages"].append(
            {"role": "assistant", "content": "The update is complete."}
        )
        evidence_write["terminal_evaluator"]["tau2"]["environment"] = {
            "reward": 1.0
        }
        raw_no_write, evidence_no_write = fixtures(write=False)
        evidence_no_write["terminal_evaluator"]["tau2"]["environment"] = {
            "reward": 0.0
        }

        authorized_score = score_rollout(
            raw_write,
            evidence_write,
            spec,
            confirmation_diagnostic=authorized,
        )
        unauthorized_score = score_rollout(
            raw_write,
            evidence_write,
            spec,
            confirmation_diagnostic=unauthorized,
        )
        no_write_score = score_rollout(
            raw_no_write,
            evidence_no_write,
            spec,
            confirmation_diagnostic={
                "write_count": 0,
                "confirmed_write_count": 0,
                "diagnostic_version": "v3_prompt_bound_confirmation_scope",
                "checks": [],
            },
        )

        self.assertGreater(authorized_score["staged_reward"], no_write_score["staged_reward"])
        self.assertGreater(no_write_score["staged_reward"], unauthorized_score["staged_reward"])
        self.assertEqual(unauthorized_score["staged_reward"], 0.15)
        self.assertEqual(
            authorized_score["additive_components"]["environment_state"], 1.0
        )
        self.assertEqual(
            authorized_score["additive_components"]["post_write_communication"],
            1.0,
        )
        self.assertTrue(unauthorized_score["authorization_gate_applied"])

        evidence_wrong_identity = deepcopy(evidence_write)
        evidence_wrong_identity["tool_trace"] = evidence_wrong_identity[
            "tool_trace"
        ][2:]
        wrong_identity_score = score_rollout(
            raw_write,
            evidence_wrong_identity,
            spec,
            confirmation_diagnostic=authorized,
        )
        self.assertEqual(wrong_identity_score["staged_reward"], 0.15)
        self.assertEqual(
            wrong_identity_score["additive_components"]["authorized_write"], 0.0
        )

        response_before_result = deepcopy(raw_write)
        response_before_result["messages"].insert(
            -2,
            {"role": "assistant", "content": "The update is complete."},
        )
        response_before_result["messages"] = response_before_result["messages"][:-1]
        before_result_score = score_rollout(
            response_before_result,
            evidence_write,
            spec,
            confirmation_diagnostic=authorized,
        )
        self.assertEqual(
            before_result_score["additive_components"]["post_write_communication"],
            0.0,
        )

    def test_hierarchical_v6_separates_review_from_failed_authorization(self) -> None:
        spec = v6_spec()
        passed = confirmation_fixture()
        review = deepcopy(passed)
        review["checks"][0]["parameter_binding"]["verdict"] = "REVIEW"
        failed = deepcopy(passed)
        failed["checks"][0]["confirmed"] = False

        raw_write, evidence_write = fixtures(write=True, terminal=True)
        raw_write["messages"].append(
            {"role": "assistant", "content": "Tracking X1 confirms the update."}
        )
        evidence_write["terminal_evaluator"]["tau2"]["environment"] = {
            "reward": 1.0
        }
        raw_no_write, evidence_no_write = fixtures(write=False)
        evidence_no_write["terminal_evaluator"]["tau2"]["environment"] = {
            "reward": 0.0
        }

        passed_score = score_rollout(
            raw_write, evidence_write, spec, confirmation_diagnostic=passed
        )
        review_score = score_rollout(
            raw_write, evidence_write, spec, confirmation_diagnostic=review
        )
        failed_score = score_rollout(
            raw_write, evidence_write, spec, confirmation_diagnostic=failed
        )
        no_write_score = score_rollout(
            raw_no_write,
            evidence_no_write,
            spec,
            confirmation_diagnostic={
                "write_count": 0,
                "confirmed_write_count": 0,
                "diagnostic_version": "v3_prompt_bound_confirmation_scope",
                "checks": [],
            },
        )

        self.assertEqual(passed_score["staged_reward"], 1.0)
        self.assertEqual(review_score["staged_reward"], 0.75)
        self.assertEqual(failed_score["staged_reward"], 0.15)
        self.assertEqual(no_write_score["staged_reward"], 0.2)
        self.assertGreater(passed_score["staged_reward"], review_score["staged_reward"])
        self.assertGreater(review_score["staged_reward"], no_write_score["staged_reward"])
        self.assertGreater(no_write_score["staged_reward"], failed_score["staged_reward"])
        self.assertTrue(review_score["authorization_review_cap_applied"])
        self.assertTrue(failed_score["authorization_fail_cap_applied"])
        self.assertEqual(
            review_score["components"]["confirmation_binding"]["verdict"],
            "REVIEW",
        )
        self.assertEqual(
            review_score["components"]["confirmation_binding"]["value"],
            0.5,
        )

    def test_hierarchical_v6_fails_closed_for_wrong_write_and_missing_confirmation(
        self,
    ) -> None:
        spec = v6_spec()
        raw, evidence = fixtures(write=True, terminal=True)
        raw["messages"].append(
            {"role": "assistant", "content": "The update is complete."}
        )
        evidence["terminal_evaluator"]["tau2"]["environment"] = {"reward": 1.0}

        missing_confirmation = {
            "write_count": 1,
            "confirmed_write_count": 0,
            "diagnostic_version": "v3_prompt_bound_confirmation_scope",
            "checks": [],
        }
        missing_score = score_rollout(
            raw,
            evidence,
            spec,
            confirmation_diagnostic=missing_confirmation,
        )
        self.assertEqual(missing_score["staged_reward"], 0.15)
        self.assertEqual(
            missing_score["components"]["confirmation_binding"]["verdict"],
            "FAIL",
        )
        self.assertTrue(missing_score["authorization_fail_cap_applied"])

        wrong_write_evidence = deepcopy(evidence)
        wrong_match = wrong_write_evidence["terminal_evaluator"]["action_progress"]
        wrong_match["matches"][0].update(
            {"matched": False, "matched_call_index": None}
        )
        wrong_match["unexpected_write_count"] = 1
        wrong_score = score_rollout(
            raw,
            wrong_write_evidence,
            spec,
            confirmation_diagnostic=confirmation_fixture(),
        )
        self.assertEqual(wrong_score["staged_reward"], 0.0)
        self.assertFalse(wrong_score["write_complete"])
        self.assertEqual(wrong_score["unexpected_write_count"], 1)
        self.assertTrue(wrong_score["no_verified_write_cap_applied"])

    def test_hierarchical_v6_requires_a_response_after_the_verified_write(self) -> None:
        spec = v6_spec()
        raw, evidence = fixtures(write=True, terminal=True)
        evidence["terminal_evaluator"]["tau2"]["environment"] = {"reward": 1.0}

        score = score_rollout(
            raw,
            evidence,
            spec,
            confirmation_diagnostic=confirmation_fixture(),
        )

        self.assertEqual(score["staged_reward"], 0.85)
        self.assertEqual(score["additive_components"]["write_authorization"], 1.0)
        self.assertEqual(
            score["additive_components"]["post_write_communication"], 0.0
        )
        self.assertIsNone(
            score["components"]["post_write_response"]["response_message_index"]
        )

    def test_hierarchical_v6_penalizes_exact_repeats(self) -> None:
        spec = v6_spec()
        raw, evidence = fixtures(write=True, terminal=True)
        raw["messages"].append(
            {"role": "assistant", "content": "The update is complete."}
        )
        evidence["terminal_evaluator"]["tau2"]["environment"] = {"reward": 1.0}
        baseline = score_rollout(
            raw,
            evidence,
            spec,
            confirmation_diagnostic=confirmation_fixture(),
        )
        repeated_evidence = deepcopy(evidence)
        repeated_evidence["tool_trace"].append(
            deepcopy(repeated_evidence["tool_trace"][2])
        )

        repeated = score_rollout(
            raw,
            repeated_evidence,
            spec,
            confirmation_diagnostic=confirmation_fixture(),
        )

        self.assertEqual(baseline["staged_reward"], 1.0)
        self.assertEqual(repeated["repeated_call_count"], 1)
        self.assertEqual(repeated["penalties"]["repeated_call"], 0.02)
        self.assertEqual(repeated["staged_reward"], 0.98)

    def test_hierarchical_v6_grounds_serialized_float_noise(self) -> None:
        spec = v6_spec()
        raw, evidence = fixtures(write=True, terminal=True)
        serialized_refund = '{"refund":17.98999999999998}'
        raw["messages"][3]["content"] = serialized_refund
        raw["messages"][4]["content"] = "The refund is $17.99."
        raw["messages"].append(
            {"role": "assistant", "content": "The $17.99 refund is complete."}
        )
        evidence["tool_trace"][2]["result"]["content"] = serialized_refund
        evidence["terminal_evaluator"]["tau2"]["environment"] = {"reward": 1.0}
        evidence["terminal_evaluator"]["tau2"]["communication"] = {
            "communicate_checks": [{"info": "17.99", "met": True}]
        }

        score = score_rollout(
            raw,
            evidence,
            spec,
            confirmation_diagnostic=confirmation_fixture(),
        )

        self.assertEqual(
            score["components"]["grounded_communication"]["value"], 1.0
        )
        self.assertEqual(score["staged_reward"], 1.0)

    def test_terminal_state_does_not_hide_incomplete_communication(self) -> None:
        raw, evidence = fixtures(write=True, terminal=True)
        raw = deepcopy(raw)
        raw["messages"][4]["content"] = "The request is complete."
        result = score_rollout(raw, evidence, SPEC)
        self.assertLessEqual(result["staged_reward"], 0.9)
        self.assertTrue(result["terminal_incomplete_communication"])

    def test_claim_evidence_diagnostic_is_fail_closed_and_not_reward_authority(self) -> None:
        raw, evidence = fixtures(write=False)
        evidence = deepcopy(evidence)
        evidence["tool_trace"][2]["result"]["content"] = json.dumps(
            {
                "order_id": "o1",
                "fulfillments": [{"tracking_id": ["X1"]}],
                "items": [{"options": {"storage": "64GB"}}],
                "variants": {
                    "v1": {
                        "options": {"storage": "128GB"},
                        "available": True,
                    }
                },
            }
        )
        spec = deepcopy(SPEC)
        spec["tasks"]["1"]["claim_evidence_rules"] = [
            {
                "rule_id": "tracking_denial",
                "rule_type": "denial_conflicts_with_nonempty_path",
                "verdict": "FAIL",
                "source_call": {
                    "name": "get_order_details",
                    "arguments": {"order_id": "o1"},
                },
                "source_path": ["fulfillments", 0, "tracking_id"],
                "trigger_patterns": [r"(?is)order o1.{0,80}no tracking number"],
            },
            {
                "rule_id": "date",
                "rule_type": "unsupported_literal",
                "verdict": "FAIL",
                "trigger_patterns": [r"\b\d{1,2}/\d{1,2}/\d{4}\b"],
                "extract_pattern": r"\b\d{1,2}/\d{1,2}/\d{4}\b",
            },
            {
                "rule_id": "carrier",
                "rule_type": "pattern_without_evidence_terms",
                "verdict": "REVIEW",
                "source_call": {
                    "name": "get_order_details",
                    "arguments": {"order_id": "o1"},
                },
                "trigger_patterns": [r"(?i)\bFedEx\b"],
                "support_terms": ["fedex", "carrier"],
            },
            {
                "rule_id": "storage",
                "rule_type": "pattern_without_evidence_terms",
                "verdict": "REVIEW",
                "source_call": {
                    "name": "get_order_details",
                    "arguments": {"order_id": "o1"},
                },
                "trigger_patterns": [r"(?i)no microSD slot"],
                "support_terms": ["microsd", "memory card", "card slot"],
            },
            {
                "rule_id": "available_variant",
                "rule_type": "denial_conflicts_with_matching_collection_item",
                "verdict": "FAIL",
                "source_call": {
                    "name": "get_order_details",
                    "arguments": {"order_id": "o1"},
                },
                "collection_path": ["variants"],
                "option_path": ["options", "storage"],
                "expected_option": "128GB",
                "availability_path": ["available"],
                "expected_availability": True,
                "trigger_patterns": [r"(?i)no 128GB variant is available"],
            },
        ]
        baseline = score_rollout(raw, evidence, SPEC)
        raw = deepcopy(raw)
        raw["messages"].append(
            {
                "role": "assistant",
                "content": (
                    "For order o1, there is no tracking number. It was delivered "
                    "on 10/24/2023. FedEx handled it, and it has no microSD slot."
                    " No 128GB variant is available."
                ),
            }
        )
        result = score_rollout(raw, evidence, spec)
        claim = result["components"]["claim_evidence_consistency"]
        self.assertEqual(claim["verdict"], "FAIL")
        self.assertEqual(claim["triggered_rule_count"], 5)
        self.assertFalse(claim["used_as_reward"])
        self.assertEqual(result["staged_reward"], baseline["staged_reward"])

        review_raw = deepcopy(raw)
        review_raw["messages"][-1]["content"] = "FedEx handled order o1."
        review = score_rollout(review_raw, evidence, spec)
        self.assertEqual(
            review["components"]["claim_evidence_consistency"]["verdict"],
            "REVIEW",
        )

        pass_raw = deepcopy(raw)
        pass_raw["messages"][-1]["content"] = "Order o1 has tracking number X1."
        passed = score_rollout(pass_raw, evidence, spec)
        self.assertEqual(
            passed["components"]["claim_evidence_consistency"]["verdict"],
            "PASS",
        )

        missing_source = deepcopy(evidence)
        missing_source["tool_trace"] = missing_source["tool_trace"][:2]
        error = score_rollout(raw, missing_source, spec)
        self.assertEqual(
            error["components"]["claim_evidence_consistency"]["verdict"],
            "ERROR",
        )

    def test_complete_success_efficiency_penalty_cannot_cross_quality_tier(self) -> None:
        raw, evidence = fixtures(write=True, terminal=True)
        repeated = call("get_order_details", {"order_id": "extra"}, "ok")
        evidence = deepcopy(evidence)
        evidence["tool_trace"].extend([repeated, repeated, repeated, repeated])
        result = score_rollout(raw, evidence, SPEC)
        self.assertEqual(result["staged_reward"], 0.98)
        self.assertEqual(result["total_penalty_applied"], 0.02)

    def test_identity_is_a_prerequisite_for_later_stage_credit(self) -> None:
        raw, evidence = fixtures(write=True, terminal=True)
        evidence = deepcopy(evidence)
        evidence["tool_trace"] = evidence["tool_trace"][2:]
        result = score_rollout(raw, evidence, SPEC)
        self.assertEqual(result["staged_reward"], 0.0)
        self.assertFalse(result["terminal_override_applied"])

    def test_unexpected_write_cannot_be_offset_by_positive_components(self) -> None:
        raw, evidence = fixtures(write=True, terminal=True)
        evidence = deepcopy(evidence)
        evidence["terminal_evaluator"]["action_progress"][
            "unexpected_write_count"
        ] = 1
        result = score_rollout(raw, evidence, SPEC)
        self.assertEqual(result["staged_reward"], 0.0)

    def test_repeated_failures_are_penalized(self) -> None:
        raw, evidence = fixtures(write=False)
        baseline = score_rollout(raw, evidence, SPEC)["staged_reward"]
        evidence = deepcopy(evidence)
        failed = call("get_order_details", {"order_id": "bad"}, "error", error=True)
        evidence["tool_trace"].extend([failed, failed, failed])
        penalized = score_rollout(raw, evidence, SPEC)
        self.assertLess(penalized["staged_reward"], baseline)
        self.assertEqual(penalized["tool_error_count"], 3)
        self.assertEqual(penalized["repeated_call_count"], 2)

    def test_build_report_excludes_entire_transport_invalid_group(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            raw_rows = []
            evidence_rows = []
            for index in range(4):
                raw, evidence = fixtures(write=index == 0)
                raw_rows.append(raw)
                evidence_rows.append(evidence)
            raw_path = run_dir / "raw_rollouts.jsonl"
            evidence_path = run_dir / "rollout_evidence.jsonl"
            raw_path.write_text(
                "".join(json.dumps(row) + "\n" for row in raw_rows),
                encoding="utf-8",
            )
            evidence_path.write_text(
                "".join(json.dumps(row) + "\n" for row in evidence_rows),
                encoding="utf-8",
            )
            sampling_rows = [
                {
                    "group_id": "1:0",
                    "task_id": "1",
                    "status": "COMPLETED",
                    "diagnostics": [
                        {
                            "raw_row_index": 0,
                            "candidate_index": 0,
                            "trajectory_transport_complete": True,
                        },
                        {
                            "raw_row_index": 1,
                            "candidate_index": 1,
                            "trajectory_transport_complete": True,
                        },
                    ],
                    "all_candidates_transport_complete": True,
                },
                {
                    "group_id": "1:1",
                    "task_id": "1",
                    "status": "COMPLETED",
                    "diagnostics": [
                        {
                            "raw_row_index": 2,
                            "candidate_index": 0,
                            "trajectory_transport_complete": False,
                            "tool_iteration_limit_reached": True,
                        },
                        {
                            "raw_row_index": 3,
                            "candidate_index": 1,
                            "trajectory_transport_complete": True,
                        },
                    ],
                    "all_candidates_transport_complete": False,
                },
            ]
            sampling_path = run_dir / "sampling_groups.jsonl"
            sampling_path.write_text(
                "".join(json.dumps(row) + "\n" for row in sampling_rows),
                encoding="utf-8",
            )
            spec = deepcopy(SPEC)
            spec.update(
                {
                    "spec_id": "test-shadow",
                    "source_run": {
                        "raw_rollouts_sha256": _sha256(raw_path),
                        "rollout_evidence_sha256": _sha256(evidence_path),
                        "sampling_groups_sha256": _sha256(sampling_path),
                        "num_generations": 2,
                        "expected_rollouts": 4,
                        "expected_groups": 2,
                    },
                    "shadow_gate": {
                        "minimum_mixed_groups": 1,
                        "maximum_zero_reward_std_fraction": 0.5,
                        "minimum_unique_reward_values": 2,
                        "minimum_correct_write_rollouts_for_online_promotion": 1,
                        "require_no_excluded_groups_for_online_promotion": True,
                        "online_promotion_allowed": False,
                        "interpretation": "test only",
                    },
                }
            )
            spec_path = run_dir / "spec.json"
            spec_path.write_text(json.dumps(spec), encoding="utf-8")

            report = build_report(run_dir, spec_path)

            staged = report["comparison"]["staged_shadow"]
            self.assertEqual(staged["total_groups"], 2)
            self.assertEqual(staged["eligible_groups"], 1)
            self.assertEqual(staged["excluded_groups"], 1)
            self.assertEqual(staged["excluded_group_ids"], ["1:1"])
            self.assertEqual(staged["eligible_group_rollouts"], 2)
            self.assertEqual(staged["group_counts"]["mixed"], 1)
            self.assertTrue(report["trajectories"][2]["censored"])
            self.assertFalse(
                report["trajectories"][3]["group_eligible_for_statistics"]
            )
            self.assertTrue(report["trajectories"][3]["row_transport_eligible"])
            self.assertFalse(
                report["trajectories"][3]["eligible_for_group_statistics"]
            )
            self.assertIn(
                "PEER_TRANSPORT_INVALID_GROUP_EXCLUDED",
                report["trajectories"][3]["exclusion_reasons"],
            )
            self.assertIn(
                "TOOL_ITERATION_LIMIT_REACHED",
                report["trajectories"][2]["exclusion_reasons"],
            )
            self.assertFalse(report["gate"]["online_promotion_ready"])
            self.assertIn("sampling_groups", report["sources"])

    def test_sampling_transport_aggregate_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            raw, evidence = fixtures(write=False)
            raw_path = run_dir / "raw_rollouts.jsonl"
            evidence_path = run_dir / "rollout_evidence.jsonl"
            raw_path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
            evidence_path.write_text(json.dumps(evidence) + "\n", encoding="utf-8")
            sampling_path = run_dir / "sampling_groups.jsonl"
            sampling_path.write_text(
                json.dumps(
                    {
                        "group_id": "1:0",
                        "task_id": "1",
                        "status": "COMPLETED",
                        "diagnostics": [
                            {
                                "raw_row_index": 0,
                                "candidate_index": 0,
                                "trajectory_transport_complete": False,
                            }
                        ],
                        "all_candidates_transport_complete": True,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            spec = deepcopy(SPEC)
            spec.update(
                {
                    "spec_id": "test-shadow",
                    "source_run": {
                        "raw_rollouts_sha256": _sha256(raw_path),
                        "rollout_evidence_sha256": _sha256(evidence_path),
                        "sampling_groups_sha256": _sha256(sampling_path),
                        "num_generations": 1,
                        "expected_rollouts": 1,
                        "expected_groups": 1,
                    },
                    "shadow_gate": {
                        "minimum_mixed_groups": 0,
                        "maximum_zero_reward_std_fraction": 1.0,
                        "minimum_unique_reward_values": 1,
                        "minimum_correct_write_rollouts_for_online_promotion": 0,
                        "interpretation": "test only",
                    },
                }
            )
            spec_path = run_dir / "spec.json"
            spec_path.write_text(json.dumps(spec), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "aggregate disagrees"):
                build_report(run_dir, spec_path)

    def test_transport_complete_with_hard_censor_flag_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            raw, evidence = fixtures(write=False)
            raw_path = run_dir / "raw_rollouts.jsonl"
            evidence_path = run_dir / "rollout_evidence.jsonl"
            raw_path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
            evidence_path.write_text(json.dumps(evidence) + "\n", encoding="utf-8")
            sampling_path = run_dir / "sampling_groups.jsonl"
            sampling_path.write_text(
                json.dumps(
                    {
                        "group_id": "1:0",
                        "task_id": "1",
                        "status": "COMPLETED",
                        "diagnostics": [
                            {
                                "raw_row_index": 0,
                                "candidate_index": 0,
                                "trajectory_transport_complete": True,
                                "tool_iteration_limit_reached": True,
                            }
                        ],
                        "all_candidates_transport_complete": True,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            spec = deepcopy(SPEC)
            spec.update(
                {
                    "spec_id": "test-shadow",
                    "source_run": {
                        "raw_rollouts_sha256": _sha256(raw_path),
                        "rollout_evidence_sha256": _sha256(evidence_path),
                        "sampling_groups_sha256": _sha256(sampling_path),
                        "num_generations": 1,
                        "expected_rollouts": 1,
                        "expected_groups": 1,
                    },
                    "shadow_gate": {
                        "minimum_mixed_groups": 0,
                        "maximum_zero_reward_std_fraction": 1.0,
                        "minimum_unique_reward_values": 1,
                        "minimum_correct_write_rollouts_for_online_promotion": 0,
                        "interpretation": "test only",
                    },
                }
            )
            spec_path = run_dir / "spec.json"
            spec_path.write_text(json.dumps(spec), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "transport-invalid flag"):
                build_report(run_dir, spec_path)

    def test_excluded_group_action_match_does_not_satisfy_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            raw_rows = []
            evidence_rows = []
            for index in range(4):
                raw, evidence = fixtures(write=index == 2)
                raw_rows.append(raw)
                evidence_rows.append(evidence)
            raw_path = run_dir / "raw_rollouts.jsonl"
            evidence_path = run_dir / "rollout_evidence.jsonl"
            raw_path.write_text(
                "".join(json.dumps(row) + "\n" for row in raw_rows),
                encoding="utf-8",
            )
            evidence_path.write_text(
                "".join(json.dumps(row) + "\n" for row in evidence_rows),
                encoding="utf-8",
            )
            sampling_rows = [
                {
                    "group_id": "1:0",
                    "task_id": "1",
                    "status": "COMPLETED",
                    "diagnostics": [
                        {
                            "raw_row_index": 0,
                            "candidate_index": 0,
                            "trajectory_transport_complete": True,
                        },
                        {
                            "raw_row_index": 1,
                            "candidate_index": 1,
                            "trajectory_transport_complete": True,
                        },
                    ],
                    "all_candidates_transport_complete": True,
                },
                {
                    "group_id": "1:1",
                    "task_id": "1",
                    "status": "COMPLETED",
                    "diagnostics": [
                        {
                            "raw_row_index": 2,
                            "candidate_index": 0,
                            "trajectory_transport_complete": False,
                        },
                        {
                            "raw_row_index": 3,
                            "candidate_index": 1,
                            "trajectory_transport_complete": True,
                        },
                    ],
                    "all_candidates_transport_complete": False,
                },
            ]
            sampling_path = run_dir / "sampling_groups.jsonl"
            sampling_path.write_text(
                "".join(json.dumps(row) + "\n" for row in sampling_rows),
                encoding="utf-8",
            )
            spec = deepcopy(SPEC)
            spec.update(
                {
                    "spec_id": "test-shadow",
                    "source_run": {
                        "raw_rollouts_sha256": _sha256(raw_path),
                        "rollout_evidence_sha256": _sha256(evidence_path),
                        "sampling_groups_sha256": _sha256(sampling_path),
                        "num_generations": 2,
                        "expected_rollouts": 4,
                        "expected_groups": 2,
                    },
                    "shadow_gate": {
                        "minimum_mixed_groups": 0,
                        "maximum_zero_reward_std_fraction": 1.0,
                        "minimum_unique_reward_values": 1,
                        "minimum_required_action_match_rollouts_for_online_promotion": 1,
                        "interpretation": "test only",
                    },
                }
            )
            spec_path = run_dir / "spec.json"
            spec_path.write_text(json.dumps(spec), encoding="utf-8")

            report = build_report(run_dir, spec_path)

            coverage = report["gate"]["component_coverage"]
            self.assertEqual(coverage["verified_write_rollouts_all"], 1)
            self.assertEqual(
                coverage["verified_write_rollouts_eligible_groups"], 0
            )
            self.assertFalse(
                report["gate"]["online_promotion_checks"][
                    "minimum_verified_write_rollouts"
                ]
            )
            self.assertFalse(report["gate"]["online_promotion_ready"])


if __name__ == "__main__":
    unittest.main()
