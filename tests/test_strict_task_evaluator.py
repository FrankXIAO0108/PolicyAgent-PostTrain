from __future__ import annotations

from src.evaluation.strict_task_evaluator import aggregate_strict_task_evaluation
from src.evaluation.task_rubric import (
    AtomicVerdict,
    CapabilityGroup,
    PredicateResult,
    parse_task_rubric,
)


def rubric():
    return parse_task_rubric(
        {
            "schema_version": "retail-strict-task-rubric-v1",
            "task_id": "fixture",
            "domain": "retail",
            "required_capability_groups": [
                "evaluation_integrity",
                "final_state_correctness",
                "protocol_compliance",
            ],
            "predicates": [
                {
                    "predicate_id": "integrity",
                    "predicate_type": "tool_result_success",
                    "capability_group": "evaluation_integrity",
                    "required": True,
                    "parameters": {},
                },
                {
                    "predicate_id": "state",
                    "predicate_type": "field_equals",
                    "capability_group": "final_state_correctness",
                    "required": True,
                    "parameters": {},
                },
                {
                    "predicate_id": "protocol_a",
                    "predicate_type": "one_tool_call_per_turn",
                    "capability_group": "protocol_compliance",
                    "required": True,
                    "parameters": {},
                },
                {
                    "predicate_id": "protocol_b",
                    "predicate_type": "message_tool_exclusivity",
                    "capability_group": "protocol_compliance",
                    "required": True,
                    "parameters": {},
                },
                {
                    "predicate_id": "optional",
                    "predicate_type": "final_claim_matches_state",
                    "capability_group": "intent_alignment",
                    "required": False,
                    "parameters": {},
                },
            ],
        }
    )


def result(
    predicate_id: str,
    group: CapabilityGroup,
    verdict: AtomicVerdict,
) -> PredicateResult:
    return PredicateResult(
        predicate_id=predicate_id,
        capability_group=group,
        verdict=verdict,
        reason=f"fixture {verdict.value}",
        error="fixture infrastructure error" if verdict is AtomicVerdict.ERROR else None,
    )


def all_pass_results() -> list[PredicateResult]:
    return [
        result("integrity", CapabilityGroup.EVALUATION_INTEGRITY, AtomicVerdict.PASS),
        result("state", CapabilityGroup.FINAL_STATE_CORRECTNESS, AtomicVerdict.PASS),
        result("protocol_a", CapabilityGroup.PROTOCOL_COMPLIANCE, AtomicVerdict.PASS),
        result("protocol_b", CapabilityGroup.PROTOCOL_COMPLIANCE, AtomicVerdict.PASS),
    ]


def group_map(evaluation):
    return {item.capability_group: item for item in evaluation.capability_groups}


def test_all_required_groups_pass_with_fixed_group_output() -> None:
    tau2 = {"reward": 0.0, "source": "frozen"}
    evaluation = aggregate_strict_task_evaluation(
        rubric(), all_pass_results(), tau2_result=tau2
    )
    groups = group_map(evaluation)
    assert len(groups) == len(CapabilityGroup) == 7
    assert evaluation.evaluation_valid is True
    assert evaluation.strict_pass is True
    assert evaluation.diagnostic_passed_groups == 3
    assert evaluation.diagnostic_total_groups == 3
    assert evaluation.diagnostic_score == 1.0
    assert evaluation.diagnostic_score_authoritative is False
    assert evaluation.tau2_result is tau2
    assert groups[CapabilityGroup.INTENT_ALIGNMENT].required is False
    assert (
        groups[CapabilityGroup.INTENT_ALIGNMENT].verdict
        is AtomicVerdict.NOT_APPLICABLE
    )


def test_required_fail_is_model_failure_but_evaluation_remains_valid() -> None:
    results = all_pass_results()
    results[1] = result(
        "state", CapabilityGroup.FINAL_STATE_CORRECTNESS, AtomicVerdict.FAIL
    )
    evaluation = aggregate_strict_task_evaluation(
        rubric(), results, tau2_result={"reward": 1.0}
    )
    state = group_map(evaluation)[CapabilityGroup.FINAL_STATE_CORRECTNESS]
    assert evaluation.evaluation_valid is True
    assert evaluation.strict_pass is False
    assert state.counts_as_model_failure is True
    assert evaluation.diagnostic_score == 2 / 3
    assert evaluation.tau2_result == {"reward": 1.0}


def test_review_and_not_applicable_never_become_strict_pass() -> None:
    for verdict in (AtomicVerdict.REVIEW, AtomicVerdict.NOT_APPLICABLE):
        results = all_pass_results()
        results[1] = result(
            "state", CapabilityGroup.FINAL_STATE_CORRECTNESS, verdict
        )
        evaluation = aggregate_strict_task_evaluation(
            rubric(), results, tau2_result=None
        )
        assert evaluation.evaluation_valid is True
        assert evaluation.strict_pass is False
        assert (
            group_map(evaluation)[CapabilityGroup.FINAL_STATE_CORRECTNESS].verdict
            is verdict
        )


def test_atomic_error_invalidates_evaluation_without_becoming_model_failure() -> None:
    results = all_pass_results()
    results[0] = result(
        "integrity", CapabilityGroup.EVALUATION_INTEGRITY, AtomicVerdict.ERROR
    )
    evaluation = aggregate_strict_task_evaluation(
        rubric(), results, tau2_result={"reward": 0.0}
    )
    integrity = group_map(evaluation)[CapabilityGroup.EVALUATION_INTEGRITY]
    assert evaluation.evaluation_valid is False
    assert evaluation.strict_pass is False
    assert integrity.verdict is AtomicVerdict.ERROR
    assert integrity.counts_as_model_failure is False
    assert "fixture infrastructure error" in " ".join(evaluation.issues)


def test_missing_required_result_invalidates_only_its_group() -> None:
    results = [item for item in all_pass_results() if item.predicate_id != "state"]
    evaluation = aggregate_strict_task_evaluation(
        rubric(), results, tau2_result=None
    )
    groups = group_map(evaluation)
    assert evaluation.evaluation_valid is False
    assert groups[CapabilityGroup.FINAL_STATE_CORRECTNESS].verdict is AtomicVerdict.ERROR
    assert groups[CapabilityGroup.PROTOCOL_COMPLIANCE].verdict is AtomicVerdict.PASS
    assert "missing required predicate result: state" in evaluation.issues


def test_duplicate_and_unknown_results_invalidate_evaluation() -> None:
    duplicate = all_pass_results() + [all_pass_results()[0]]
    duplicate_evaluation = aggregate_strict_task_evaluation(
        rubric(), duplicate, tau2_result=None
    )
    assert duplicate_evaluation.evaluation_valid is False
    assert "duplicate predicate result: integrity" in duplicate_evaluation.issues

    unknown = all_pass_results() + [
        result("unknown", CapabilityGroup.EVIDENCE_CONSISTENCY, AtomicVerdict.PASS)
    ]
    unknown_evaluation = aggregate_strict_task_evaluation(
        rubric(), unknown, tau2_result=None
    )
    assert unknown_evaluation.evaluation_valid is False
    assert "unknown predicate result: unknown" in unknown_evaluation.issues


def test_group_mismatch_is_evaluation_error_not_model_failure() -> None:
    results = all_pass_results()
    results[1] = result("state", CapabilityGroup.PROTOCOL_COMPLIANCE, AtomicVerdict.PASS)
    evaluation = aggregate_strict_task_evaluation(
        rubric(), results, tau2_result=None
    )
    state = group_map(evaluation)[CapabilityGroup.FINAL_STATE_CORRECTNESS]
    assert evaluation.evaluation_valid is False
    assert state.verdict is AtomicVerdict.ERROR
    assert state.counts_as_model_failure is False


def test_optional_predicate_cannot_change_strict_result_or_denominator() -> None:
    baseline = aggregate_strict_task_evaluation(
        rubric(), all_pass_results(), tau2_result=None
    )
    with_optional_failure = aggregate_strict_task_evaluation(
        rubric(),
        all_pass_results()
        + [result("optional", CapabilityGroup.INTENT_ALIGNMENT, AtomicVerdict.FAIL)],
        tau2_result=None,
    )
    assert baseline.strict_pass is True
    assert with_optional_failure.strict_pass is True
    assert with_optional_failure.diagnostic_total_groups == 3


def test_tau2_result_is_preserved_independently_of_strict_result() -> None:
    tau2_success = {"reward": 1.0, "nested": {"db_reward": 1.0}}
    strict_failure_results = all_pass_results()
    strict_failure_results[2] = result(
        "protocol_a", CapabilityGroup.PROTOCOL_COMPLIANCE, AtomicVerdict.FAIL
    )
    evaluation = aggregate_strict_task_evaluation(
        rubric(), strict_failure_results, tau2_result=tau2_success
    )
    assert evaluation.strict_pass is False
    assert evaluation.tau2_result is tau2_success
    assert evaluation.tau2_result == {
        "reward": 1.0,
        "nested": {"db_reward": 1.0},
    }
