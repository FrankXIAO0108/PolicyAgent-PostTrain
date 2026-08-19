import json
import os
import tempfile
import unittest
from pathlib import Path

from src.training.run_teacher_eval import (
    DEFAULT_CONFIG,
    SCOPE_PREFIX,
    entity_overlap,
    validate_config,
)

LOCAL_TAU2_ROOT = Path(r"D:\tau2-bench")
REQUIRES_TAU2 = LOCAL_TAU2_ROOT.joinpath("data/tau2/domains/retail/tasks.json").is_file()


def load_real_config() -> dict:
    return json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8-sig"))


class TeacherEvalConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        if not REQUIRES_TAU2:
            self.skipTest("local tau2 checkout not available")
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.addCleanup(self.temp_dir.cleanup)
        self.old_env = os.environ.get("POLICYAGENT_TAU2_ROOT")
        os.environ["POLICYAGENT_TAU2_ROOT"] = str(LOCAL_TAU2_ROOT)

    def tearDown(self) -> None:
        if self.old_env is None:
            os.environ.pop("POLICYAGENT_TAU2_ROOT", None)
        else:
            os.environ["POLICYAGENT_TAU2_ROOT"] = self.old_env

    def write(self, payload: dict) -> Path:
        path = self.root / "config.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path

    def test_real_frozen_config_validates(self):
        validated = validate_config(DEFAULT_CONFIG)
        self.assertEqual(len(validated["task_ids"]), 30)
        self.assertEqual(len(validated["model_runs"]), 2)
        self.assertEqual(
            [run["name"] for run in validated["model_runs"]], ["base", "sft"]
        )
        sources = {
            row["source"] for row in validated["task_rows"]
        }
        self.assertEqual(sources, {"train_candidates", "test_clean"})
        self.assertEqual(
            sum(1 for row in validated["task_rows"] if row["source"] == "train_candidates"),
            13,
        )
        self.assertEqual(
            sum(1 for row in validated["task_rows"] if row["source"] == "test_clean"),
            17,
        )
        self.assertEqual(validated["num_trials"], 1)
        self.assertEqual(validated["seed"], 20260818)

    def test_rejects_wrong_scope(self):
        payload = load_real_config()
        payload["scope"] = "WRONG"
        with self.assertRaises(ValueError):
            validate_config(self.write(payload))

    def test_rejects_unfrozen_status(self):
        payload = load_real_config()
        payload["status"] = "DRAFT"
        with self.assertRaises(ValueError):
            validate_config(self.write(payload))

    def test_rejects_duplicate_tasks(self):
        payload = load_real_config()
        payload["tasks"].append(dict(payload["tasks"][0]))
        with self.assertRaises(ValueError):
            validate_config(self.write(payload))

    def test_rejects_unknown_source(self):
        payload = load_real_config()
        payload["tasks"][0]["source"] = "mystery"
        with self.assertRaises(ValueError):
            validate_config(self.write(payload))

    def test_rejects_non_llm_agent(self):
        payload = load_real_config()
        payload["agent"]["implementation"] = "audited_teacher_llm_agent"
        with self.assertRaises(ValueError):
            validate_config(self.write(payload))

    def test_rejects_nonzero_agent_temperature(self):
        payload = load_real_config()
        payload["agent"]["temperature"] = 0.5
        with self.assertRaises(ValueError):
            validate_config(self.write(payload))

    def test_rejects_wrong_evaluation_type(self):
        payload = load_real_config()
        payload["evaluation"]["type"] = "ALL"
        with self.assertRaises(ValueError):
            validate_config(self.write(payload))

    def test_rejects_task_outside_expected_split(self):
        payload = load_real_config()
        payload["tasks"][0]["task_id"] = "58"  # a teacher task in the official train split
        with self.assertRaises(ValueError):
            validate_config(self.write(payload))

    def test_entity_gate_catches_teacher_task(self):
        # Task 58 belongs to the teacher pool; its definition text contains
        # teacher-pool entities and must be rejected by the entity gate.
        payload = load_real_config()
        payload["tasks"][0] = {
            "task_id": "58",
            "source": "train_candidates",
            "reason": "leak check",
        }
        with self.assertRaises(ValueError) as context:
            validate_config(self.write(payload))
        self.assertIn("overlaps teacher training entities", str(context.exception))

    def test_entity_overlap_detects_value_in_text(self):
        payload = {"id": "99", "text": "service customer ivan_hernandez_6923 today"}
        hits = entity_overlap(payload, ["ivan_hernandez_6923", "#W9999999"])
        self.assertEqual(hits, ["ivan_hernandez_6923"])

    def test_entity_overlap_empty_when_clean(self):
        payload = {"id": "99", "text": "no shared entities here"}
        self.assertEqual(entity_overlap(payload, ["ivan_hernandez_6923"]), [])


if __name__ == "__main__":
    unittest.main()
