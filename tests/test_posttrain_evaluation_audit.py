import unittest

from src.training.audit_posttrain_evaluations import (
    audit_evaluation,
    is_strict_json_object,
)


class PosttrainEvaluationAuditTests(unittest.TestCase):
    def test_strict_json_requires_entire_completion(self) -> None:
        self.assertTrue(is_strict_json_object('{"tool":"get_order","arguments":{}}'))
        self.assertFalse(
            is_strict_json_object('{"tool":"get_order","arguments":{}} trailing')
        )
        self.assertFalse(
            is_strict_json_object(',{"tool":"get_order","arguments":{}}')
        )
        self.assertFalse(is_strict_json_object('[{"tool":"get_order"}]'))

    def test_audit_reports_extraction_gap(self) -> None:
        report = audit_evaluation(
            {
                "rows": [
                    {
                        "scenario_id": "strict",
                        "completion": '{"tool":"get_order","arguments":{}}',
                        "valid_json": True,
                    },
                    {
                        "scenario_id": "trailing",
                        "completion": '{"tool":"get_order","arguments":{}} text',
                        "valid_json": True,
                    },
                ]
            }
        )
        self.assertEqual(report["extractable_json_rate"], 1.0)
        self.assertEqual(report["strict_json_object_rate"], 0.5)
        self.assertEqual(report["format_gap_count"], 1)
        self.assertEqual(report["format_gap_scenarios"], ["trailing"])


if __name__ == "__main__":
    unittest.main()
