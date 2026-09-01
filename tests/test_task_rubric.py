from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.task_rubric import (
    AtomicVerdict,
    CapabilityGroup,
    PredicateResult,
    RubricSchemaError,
    SCHEMA_VERSION,
    load_task_rubric,
    parse_task_rubric,
)


def _valid_rubric() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": "24",
        "domain": "retail",
        "description": "Schema fixture only; it does not define business truth.",
        "required_capability_groups": [
            "final_state_correctness",
            "required_task_execution",
        ],
        "predicates": [
            {
                "predicate_id": "final_state",
                "predicate_type": "field_equals",
                "capability_group": "final_state_correctness",
                "required": True,
                "parameters": {"path": ["orders", "example"]},
            },
            {
                "predicate_id": "required_action",
                "predicate_type": "required_action",
                "capability_group": "required_task_execution",
                "required": True,
                "parameters": {"name": "placeholder"},
            },
            {
                "predicate_id": "diagnostic_only",
                "predicate_type": "tool_result_success",
                "capability_group": "evidence_consistency",
                "required": False,
                "parameters": {},
            },
        ],
        "metadata": {"fixture": True},
    }


class TaskRubricSchemaTests(unittest.TestCase):
    def test_valid_schema_loads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "task_24.json"
            path.write_text(json.dumps(_valid_rubric()), encoding="utf-8")
            rubric = load_task_rubric(path)

        self.assertEqual(rubric.task_id, "24")
        self.assertEqual(rubric.domain, "retail")
        self.assertEqual(len(rubric.predicates), 3)
        self.assertEqual(
            rubric.required_capability_groups[0],
            CapabilityGroup.FINAL_STATE_CORRECTNESS,
        )

    def test_invalid_schema_fails_fast(self) -> None:
        payload = _valid_rubric()
        payload["unexpected"] = True
        with self.assertRaisesRegex(RubricSchemaError, "unknown keys"):
            parse_task_rubric(payload)

        payload = _valid_rubric()
        payload["predicates"][0]["capability_group"] = "invented_group"
        with self.assertRaisesRegex(RubricSchemaError, "unsupported capability"):
            parse_task_rubric(payload)

        payload = _valid_rubric()
        payload["predicates"][0]["required"] = False
        with self.assertRaisesRegex(RubricSchemaError, "lack a required predicate"):
            parse_task_rubric(payload)

    def test_duplicate_predicate_ids_are_rejected(self) -> None:
        payload = _valid_rubric()
        payload["predicates"][1]["predicate_id"] = "final_state"
        with self.assertRaisesRegex(RubricSchemaError, "duplicate predicate_id"):
            parse_task_rubric(payload)

    def test_review_is_not_counted_as_pass(self) -> None:
        result = PredicateResult(
            predicate_id="needs_review",
            capability_group=CapabilityGroup.INTENT_ALIGNMENT,
            verdict=AtomicVerdict.REVIEW,
            reason="The available evidence is ambiguous.",
        )
        self.assertFalse(result.counts_as_pass)
        self.assertFalse(result.counts_as_model_failure)

    def test_error_is_not_counted_as_model_failure(self) -> None:
        result = PredicateResult(
            predicate_id="missing_artifact",
            capability_group=CapabilityGroup.EVALUATION_INTEGRITY,
            verdict=AtomicVerdict.ERROR,
            reason="The evaluation could not be completed.",
            error="returned_results.json is missing",
        )
        self.assertFalse(result.counts_as_pass)
        self.assertFalse(result.counts_as_model_failure)
        self.assertTrue(result.verdict.invalidates_evaluation)

    def test_non_error_result_cannot_carry_error_detail(self) -> None:
        with self.assertRaisesRegex(RubricSchemaError, "Only ERROR"):
            PredicateResult(
                predicate_id="state",
                capability_group=CapabilityGroup.FINAL_STATE_CORRECTNESS,
                verdict=AtomicVerdict.FAIL,
                reason="Observed state differs from expected state.",
                error="not an infrastructure error",
            )

    def test_result_rejects_unparsed_enum_strings(self) -> None:
        with self.assertRaisesRegex(RubricSchemaError, "AtomicVerdict"):
            PredicateResult(
                predicate_id="state",
                capability_group=CapabilityGroup.FINAL_STATE_CORRECTNESS,
                verdict="PASS",  # type: ignore[arg-type]
                reason="Direct construction must not bypass enum validation.",
            )

    def test_schema_rejects_non_finite_json_numbers(self) -> None:
        payload = _valid_rubric()
        payload["predicates"][0]["parameters"] = {"threshold": float("nan")}
        with self.assertRaisesRegex(RubricSchemaError, "NaN or Infinity"):
            parse_task_rubric(payload)


if __name__ == "__main__":
    unittest.main()
