from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

from src.evaluation.task_rubric import (
    AtomicVerdict,
    CapabilityGroup,
    PredicateResult,
    TaskRubric,
)


FIXED_CAPABILITY_GROUPS = tuple(CapabilityGroup)


@dataclass(frozen=True, slots=True)
class CapabilityGroupResult:
    capability_group: CapabilityGroup
    required: bool
    verdict: AtomicVerdict
    predicate_ids: tuple[str, ...]
    verdict_counts: tuple[tuple[AtomicVerdict, int], ...]

    @property
    def counts_as_pass(self) -> bool:
        return self.verdict is AtomicVerdict.PASS

    @property
    def counts_as_model_failure(self) -> bool:
        return self.verdict is AtomicVerdict.FAIL


@dataclass(frozen=True, slots=True)
class StrictTaskEvaluation:
    task_id: str
    evaluation_valid: bool
    strict_pass: bool
    capability_groups: tuple[CapabilityGroupResult, ...]
    diagnostic_passed_groups: int
    diagnostic_total_groups: int
    diagnostic_score: float
    diagnostic_score_authoritative: bool
    tau2_result: Any
    issues: tuple[str, ...]


def _aggregate_verdict(results: Sequence[PredicateResult]) -> AtomicVerdict:
    verdicts = {result.verdict for result in results}
    if AtomicVerdict.ERROR in verdicts:
        return AtomicVerdict.ERROR
    if AtomicVerdict.FAIL in verdicts:
        return AtomicVerdict.FAIL
    if AtomicVerdict.REVIEW in verdicts:
        return AtomicVerdict.REVIEW
    if AtomicVerdict.NOT_APPLICABLE in verdicts:
        return AtomicVerdict.NOT_APPLICABLE
    return AtomicVerdict.PASS


def aggregate_strict_task_evaluation(
    rubric: TaskRubric,
    predicate_results: Sequence[PredicateResult],
    *,
    tau2_result: Any,
) -> StrictTaskEvaluation:
    """Aggregate deterministic predicate results without changing Tau2 output.

    The rubric fixes the required-group denominator. Missing, duplicate, unknown,
    or group-mismatched evidence invalidates evaluation rather than becoming a
    model failure. Optional predicates never change strict-pass semantics.
    """

    required_groups = set(rubric.required_capability_groups)
    specs_by_id = {spec.predicate_id: spec for spec in rubric.predicates}
    results_by_id: dict[str, PredicateResult] = {}
    invalid_groups: set[CapabilityGroup] = set()
    issues: list[str] = []

    for index, result in enumerate(predicate_results):
        if not isinstance(result, PredicateResult):
            issues.append(f"predicate_results[{index}] is not a PredicateResult")
            invalid_groups.update(required_groups)
            continue
        spec = specs_by_id.get(result.predicate_id)
        if spec is None:
            issues.append(f"unknown predicate result: {result.predicate_id}")
            invalid_groups.update(required_groups)
            continue
        if result.predicate_id in results_by_id:
            issues.append(f"duplicate predicate result: {result.predicate_id}")
            invalid_groups.add(spec.capability_group)
            continue
        results_by_id[result.predicate_id] = result
        if result.capability_group is not spec.capability_group:
            issues.append(
                f"predicate {result.predicate_id} capability group mismatch: "
                f"expected {spec.capability_group.value}, observed "
                f"{result.capability_group.value}"
            )
            invalid_groups.add(spec.capability_group)

    group_results: list[CapabilityGroupResult] = []
    for group in FIXED_CAPABILITY_GROUPS:
        is_required = group in required_groups
        required_specs = tuple(
            spec
            for spec in rubric.predicates
            if spec.required and spec.capability_group is group
        )
        if not is_required:
            group_results.append(
                CapabilityGroupResult(
                    capability_group=group,
                    required=False,
                    verdict=AtomicVerdict.NOT_APPLICABLE,
                    predicate_ids=(),
                    verdict_counts=(),
                )
            )
            continue

        missing = [
            spec.predicate_id
            for spec in required_specs
            if spec.predicate_id not in results_by_id
        ]
        if missing:
            invalid_groups.add(group)
            issues.extend(f"missing required predicate result: {item}" for item in missing)

        observed = tuple(
            results_by_id[spec.predicate_id]
            for spec in required_specs
            if spec.predicate_id in results_by_id
        )
        counts = Counter(result.verdict for result in observed)
        if group in invalid_groups or not required_specs:
            verdict = AtomicVerdict.ERROR
        else:
            verdict = _aggregate_verdict(observed)
        if verdict is AtomicVerdict.ERROR:
            for result in observed:
                if result.verdict is AtomicVerdict.ERROR:
                    issues.append(
                        f"predicate {result.predicate_id} evaluation error: {result.error}"
                    )
        group_results.append(
            CapabilityGroupResult(
                capability_group=group,
                required=True,
                verdict=verdict,
                predicate_ids=tuple(spec.predicate_id for spec in required_specs),
                verdict_counts=tuple(
                    (candidate, counts[candidate])
                    for candidate in AtomicVerdict
                    if counts[candidate]
                ),
            )
        )

    required_results = tuple(result for result in group_results if result.required)
    evaluation_valid = all(
        result.verdict is not AtomicVerdict.ERROR for result in required_results
    )
    strict_pass = evaluation_valid and all(
        result.verdict is AtomicVerdict.PASS for result in required_results
    )
    passed = sum(
        result.verdict is AtomicVerdict.PASS for result in required_results
    )
    total = len(required_results)
    diagnostic_score = passed / total if total else 0.0
    return StrictTaskEvaluation(
        task_id=rubric.task_id,
        evaluation_valid=evaluation_valid,
        strict_pass=strict_pass,
        capability_groups=tuple(group_results),
        diagnostic_passed_groups=passed,
        diagnostic_total_groups=total,
        diagnostic_score=diagnostic_score,
        diagnostic_score_authoritative=False,
        tau2_result=tau2_result,
        issues=tuple(dict.fromkeys(issues)),
    )
