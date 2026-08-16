import copy
import json
import tempfile
import unittest
from pathlib import Path

from src.training.run_tau2_teacher_pilot_layer1 import (
    DEFAULT_CONFIG,
    SCOPE,
    trial_specs,
    validate_config,
)


def load_real_config() -> dict:
    return json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8-sig"))


class Layer1ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def write(self, payload: dict) -> Path:
        path = self.root / "config.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path

    def test_real_frozen_config_validates(self):
        validated = validate_config(DEFAULT_CONFIG)
        self.assertEqual(len(validated["task_ids"]), 8)
        self.assertEqual(validated["candidates_per_task"], 4)
        self.assertEqual(validated["temperature_ladder"], [0.2, 0.4, 0.6, 0.8])
        self.assertEqual(validated["base_seed"], 20260815)

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

    def test_rejects_task_outside_rl_train(self):
        payload = load_real_config()
        payload["tasks"][0] = {"task_id": "999", "stratum": "x", "reason": "y"}
        with self.assertRaises(ValueError):
            validate_config(self.write(payload))

    def test_rejects_overlap_with_rl_validation(self):
        payload = load_real_config()
        payload["tasks"][0] = {"task_id": "2", "stratum": "x", "reason": "y"}
        with self.assertRaises(ValueError):
            validate_config(self.write(payload))

    def test_rejects_overlap_with_development_audit(self):
        payload = load_real_config()
        payload["tasks"][0] = {"task_id": "1", "stratum": "x", "reason": "y"}
        with self.assertRaises(ValueError):
            validate_config(self.write(payload))

    def test_rejects_duplicate_tasks(self):
        payload = load_real_config()
        payload["tasks"].append(copy.deepcopy(payload["tasks"][0]))
        with self.assertRaises(ValueError):
            validate_config(self.write(payload))

    def test_rejects_ladder_length_mismatch(self):
        payload = load_real_config()
        payload["generation"]["agent"]["temperature_ladder"] = [0.2, 0.4]
        with self.assertRaises(ValueError):
            validate_config(self.write(payload))

    def test_rejects_nonzero_user_temperature(self):
        payload = load_real_config()
        payload["generation"]["user"]["temperature"] = 0.5
        with self.assertRaises(ValueError):
            validate_config(self.write(payload))

    def test_rejects_hidden_gold_visibility(self):
        payload = load_real_config()
        payload["teacher_visibility"]["evaluation_criteria"] = True
        with self.assertRaises(ValueError):
            validate_config(self.write(payload))

    def test_rejects_llm_judge(self):
        payload = load_real_config()
        payload["offline_evaluation"]["llm_judge_used"] = True
        with self.assertRaises(ValueError):
            validate_config(self.write(payload))


class TrialSpecsTests(unittest.TestCase):
    def test_trial_specs_map_ladder_to_distinct_seeds(self):
        validated = validate_config(DEFAULT_CONFIG)
        specs = trial_specs(validated)
        self.assertEqual(len(specs), 4)
        self.assertEqual([s["temperature"] for s in specs], [0.2, 0.4, 0.6, 0.8])
        self.assertEqual([s["seed"] for s in specs], [20260815, 20260816, 20260817, 20260818])
        self.assertEqual(len({s["seed"] for s in specs}), 4)


if __name__ == "__main__":
    unittest.main()
