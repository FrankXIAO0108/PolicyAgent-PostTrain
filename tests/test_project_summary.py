from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.project_summary import build_project_summary, render_markdown


class ProjectSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.demo = build_project_summary(cls.root)

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
        self.assertEqual(
            set(self.demo["evidence"]),
            {"evaluation_report", "guard_audit", "comparison"},
        )
        for source in self.demo["evidence"].values():
            with self.subTest(path=source["path"]):
                self.assertEqual(
                    source["sha256"],
                    hashlib.sha256(
                        (self.root / source["path"]).read_bytes()
                    ).hexdigest().upper(),
                )

    def test_demo_preserves_interpretation_boundaries(self) -> None:
        self.assertEqual(
            self.demo["schema_version"], "policy-agent-project-summary-v2.0"
        )
        self.assertNotIn("post_training_status", self.demo)
        self.assertEqual(
            self.demo["demo_scope"]["current_training_status"], "not_assessed"
        )
        self.assertEqual(
            self.demo["demo_scope"]["current_training_status_reference"],
            "TECHNICAL_REPORT.md",
        )
        rendered = render_markdown(self.demo)
        self.assertIn("不是未见任务泛化性能", rendered)
        self.assertIn("冻结 Baseline/Guard 离线示例", rendered)
        self.assertIn("不判断当前后训练状态", rendered)
        self.assertIn("[技术报告](TECHNICAL_REPORT.md)", rendered)
        self.assertNotIn("GRPO：未运行", rendered)
        self.assertIn("Task | 业务问题", rendered)

    def test_demo_needs_only_three_frozen_inputs_and_standard_library(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for source in self.demo["evidence"].values():
                destination = root / source["path"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(self.root / source["path"], destination)
            self.assertEqual(build_project_summary(root), self.demo)
            result = subprocess.run(
                [
                    sys.executable,
                    "-S",
                    "-X",
                    "utf8",
                    "-m",
                    "src.project_summary",
                    "--project-root",
                    str(root),
                    "--format",
                    "json",
                ],
                cwd=self.root,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            self.assertEqual(json.loads(result.stdout), self.demo)


if __name__ == "__main__":
    unittest.main()
