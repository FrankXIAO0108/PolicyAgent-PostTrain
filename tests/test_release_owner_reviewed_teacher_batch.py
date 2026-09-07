from __future__ import annotations

import json
import unittest

from src.training.release_owner_reviewed_teacher_batch import (
    HARD_ENTITY_KEYS,
    REPORTED_ENTITY_KEYS,
    entity_groups,
)


class OwnerReviewedTeacherReleaseTests(unittest.TestCase):
    def test_entities_are_extracted_from_json_tool_content(self) -> None:
        payload = {
            "messages": [
                {
                    "role": "tool",
                    "content": json.dumps(
                        {
                            "user_id": "u1",
                            "order_id": "o1",
                            "product_id": "p1",
                        }
                    ),
                }
            ]
        }
        self.assertEqual(
            entity_groups(payload, HARD_ENTITY_KEYS),
            {"user_id:u1", "order_id:o1"},
        )
        self.assertEqual(
            entity_groups(payload, REPORTED_ENTITY_KEYS),
            {"product_id:p1"},
        )

    def test_product_is_not_a_hard_split_key(self) -> None:
        payload = {"product_id": "p1", "order_id": "o1"}
        self.assertEqual(
            entity_groups(payload, HARD_ENTITY_KEYS), {"order_id:o1"}
        )

    def test_invalid_json_string_is_ignored(self) -> None:
        self.assertEqual(
            entity_groups({"content": "{not-json"}, HARD_ENTITY_KEYS), set()
        )


if __name__ == "__main__":
    unittest.main()
