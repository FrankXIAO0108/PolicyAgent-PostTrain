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


class ResultsRepairTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.config = load_real_config()
        self.generation = self.config["generation"]
        self.commit = "ddb9cc8"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_build_results_dict_has_full_schema(self):
        from src.training.run_tau2_teacher_pilot_layer1 import build_results_dict

        simulations = [
            {"id": "sim-a", "policy": "policy text", "seed": 1},
            {"id": "sim-b", "policy": "policy text", "seed": 2},
        ]
        tasks = [{"id": "7", "description": "task"}]
        built = build_results_dict(simulations, tasks, self.generation, self.commit)
        self.assertEqual(built["info"]["git_commit"], self.commit)
        self.assertEqual(built["info"]["num_trials"], 4)
        self.assertEqual(built["info"]["environment_info"]["domain_name"], "retail")
        self.assertEqual(built["info"]["environment_info"]["policy"], "policy text")
        self.assertEqual(built["info"]["agent_info"]["implementation"], "audited_teacher_llm_agent")
        self.assertEqual(built["tasks"], tasks)
        self.assertEqual(built["simulations"], simulations)
        self.assertNotIn("simulation_index", built)

    def test_build_results_dict_rejects_empty_simulations(self):
        from src.training.run_tau2_teacher_pilot_layer1 import build_results_dict

        with self.assertRaises(ValueError):
            build_results_dict([], [], self.generation, self.commit)

    def test_repair_results_rebuilds_incomplete_files(self):
        from src.training.run_tau2_teacher_pilot_layer1 import repair_results

        run_dir = self.root / "run"
        task_dir = run_dir / "private_evaluation" / "task_7"
        task_dir.mkdir(parents=True)
        (task_dir / "task_snapshot.json").write_text(
            json.dumps({"id": "7", "description": "task"}), encoding="utf-8"
        )
        (task_dir / "returned_results.json").write_text(
            json.dumps({"simulations": [{"id": "sim-a", "policy": "p", "seed": 1}]}),
            encoding="utf-8",
        )
        rebuilt = repair_results(run_dir, self.config, self.commit)
        self.assertEqual(rebuilt, 1)
        payload = json.loads((task_dir / "returned_results.json").read_text(encoding="utf-8"))
        self.assertIn("info", payload)
        self.assertIn("tasks", payload)
        self.assertEqual(payload["tasks"], [{"id": "7", "description": "task"}])

    def test_repair_results_skips_complete_files(self):
        from src.training.run_tau2_teacher_pilot_layer1 import repair_results

        run_dir = self.root / "run"
        task_dir = run_dir / "private_evaluation" / "task_7"
        task_dir.mkdir(parents=True)
        (task_dir / "returned_results.json").write_text(
            json.dumps({"info": {}, "tasks": [], "simulations": [{"id": "s"}]}),
            encoding="utf-8",
        )
        rebuilt = repair_results(run_dir, self.config, self.commit)
        self.assertEqual(rebuilt, 0)

    def test_finalize_rejects_non_started_manifest(self):
        from src.training.run_tau2_teacher_pilot_layer1 import finalize

        run_dir = self.root / "run"
        run_dir.mkdir()
        (run_dir / "run_manifest.json").write_text(
            json.dumps({"status": "COMPLETED"}), encoding="utf-8"
        )
        with self.assertRaises(ValueError):
            finalize(run_dir)

    def test_finalize_rejects_config_hash_mismatch(self):
        from src.training.run_tau2_teacher_pilot_layer1 import finalize

        run_dir = self.root / "run"
        run_dir.mkdir()
        (run_dir / "run_manifest.json").write_text(
            json.dumps(
                {
                    "status": "STARTED",
                    "bindings": {
                        "config_path": "configs/retail_tau2_teacher_pilot_layer1_v1.json",
                        "config_sha256": "WRONGHASH",
                    },
                    "project": {"commit": self.commit},
                    "task_ids": ["7"],
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaises(ValueError):
            finalize(run_dir)


if __name__ == "__main__":
    unittest.main()
