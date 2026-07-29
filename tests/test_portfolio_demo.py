from __future__ import annotations

import unittest
from pathlib import Path

from src.portfolio_demo import build_portfolio_demo, render_markdown


class PortfolioDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.demo = build_portfolio_demo(cls.root)

    def test_demo_is_bound_to_frozen_evidence(self) -> None:
        self.assertEqual(self.demo["frozen_scope"]["task_count"], 20)
        self.assertEqual(
            self.demo["frozen_scope"]["failure_task_ids"],
            ["59", "95", "98", "107"],
        )
        self.assertEqual(
            [case["task_id"] for case in self.demo["cases"]],
            ["95", "98", "107"],
        )
        self.assertTrue(
            all(
                len(source["sha256"]) == 64
                for source in self.demo["evidence"].values()
            )
        )

    def test_demo_preserves_interpretation_boundaries(self) -> None:
        self.assertFalse(
            self.demo["post_training_status"]["sft_completed"]
        )
        self.assertFalse(
            self.demo["post_training_status"]["rlhf_or_grpo_completed"]
        )
        rendered = render_markdown(self.demo)
        self.assertIn("不是未见任务泛化性能", rendered)
        self.assertIn("独立人工金标为 0", rendered)
        self.assertIn("Task | 业务问题", rendered)


if __name__ == "__main__":
    unittest.main()
