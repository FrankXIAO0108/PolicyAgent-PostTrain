import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.training.prepare_sft_plateau_plan import prepare
from src.training.run_retail_agentic_grpo import sha256


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


class PrepareSftPlateauPlanTests(unittest.TestCase):
    def fixture(self, root: Path):
        data_dir = root / "private" / "merged"
        dataset = [
            {"candidate_id": "a", "task_id": "1", "split": "TRAIN"},
            {"candidate_id": "b", "task_id": "1", "split": "TRAIN"},
            {"candidate_id": "c", "task_id": "2", "split": "TRAIN"},
            {"candidate_id": "d", "task_id": "3", "split": "TRAIN"},
            {"candidate_id": "e", "task_id": "4", "split": "TRAIN"},
            {"candidate_id": "v", "task_id": "9", "split": "VALIDATION"},
        ]
        split_plan = [
            {"task_id": str(task), "split": "VALIDATION" if task == 9 else "TRAIN"}
            for task in (1, 2, 3, 4, 9)
        ]
        dataset_path = data_dir / "dataset.jsonl"
        split_path = data_dir / "split.jsonl"
        write_jsonl(dataset_path, dataset)
        write_jsonl(split_path, split_plan)
        manifest = {
            "scope": "TEACHER_TRAJECTORY_SFT",
            "claims": {"business_improvement_claim_allowed": False},
            "files": {
                "sft_dataset": {
                    "path": "dataset.jsonl",
                    "sha256": sha256(dataset_path),
                    "rows": len(dataset),
                },
                "split_plan": {
                    "path": "split.jsonl",
                    "sha256": sha256(split_path),
                    "rows": len(split_plan),
                },
            },
            "leakage_and_quality_checks": {"passed": True},
        }
        write_json(data_dir / "manifest.json", manifest)
        source_config = {
            "scope": "TEACHER_TRAJECTORY_SFT",
            "data_dir": "private/merged",
            "seed": 10,
            "sft": {"max_steps": 80},
        }
        config_path = root / "configs" / "source.json"
        write_json(config_path, source_config)
        protocol = {
            "scope": "TEACHER_TRAJECTORY_SFT_PLATEAU",
            "source_config": "configs/source.json",
            "source_data_dir": "private/merged",
            "existing_full_run": "private/existing_run",
            "selection_seed": 99,
            "optimization_steps": [20, 40, 80],
            "data_fractions": [0.25, 0.5, 1.0],
            "full_reference_steps": 80,
            "stability_seeds": [10, 11, 12],
            "claims": {"sft_plateau_claim_allowed_before_stage_2": False},
        }
        protocol_path = root / "configs" / "protocol.json"
        write_json(protocol_path, protocol)
        write_json(
            root / "private" / "existing_run" / "run_manifest.json",
            {
                "status": "COMPLETED",
                "bindings": {
                    "config_sha256": "REMOTE_LF_CONFIG_HASH",
                    "data_manifest_sha256": sha256(data_dir / "manifest.json"),
                },
            },
        )
        return protocol_path, data_dir

    def test_builds_nested_equal_epoch_plan_and_keeps_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            protocol_path, _ = self.fixture(root)
            output = root / "private" / "plan"
            with patch("src.training.prepare_sft_plateau_plan.REPO_ROOT", root):
                result = prepare(protocol_path, output)

            self.assertEqual(result["status"], "PREPARED_NOT_RUN")
            self.assertFalse(result["claims"]["sft_plateau_established"])
            variants = {row["variant_id"]: row for row in result["variants"]}
            self.assertEqual(set(variants), {
                "opt_full_s20", "opt_full_s40", "opt_full_s80",
                "data_25_equal_epoch", "data_50_equal_epoch",
                "stability_full_s80_seed11", "stability_full_s80_seed12",
            })
            self.assertTrue(variants["opt_full_s80"]["reuse_existing_full_run"])
            self.assertEqual(
                variants["opt_full_s80"]["existing_run_bound_config_sha256"],
                "REMOTE_LF_CONFIG_HASH",
            )
            self.assertEqual(
                variants["opt_full_s80"]["config_path"], "configs/source.json"
            )
            generated_config = json.loads(
                (output / "configs" / "opt_full_s20.json").read_text()
            )
            self.assertFalse(generated_config["artifacts"]["save_merged_model"])
            quarter = variants["data_25_equal_epoch"]
            half = variants["data_50_equal_epoch"]
            self.assertLessEqual(quarter["train_tasks"], half["train_tasks"])
            self.assertLess(quarter["max_steps"], half["max_steps"])
            for name in ("data_25_equal_epoch", "data_50_equal_epoch"):
                manifest = json.loads(
                    (output / "data" / name / "manifest.json").read_text()
                )
                self.assertEqual(manifest["counts"]["VALIDATION"], 1)
                self.assertTrue(
                    manifest["plateau_subset"]["validation_rows_unchanged"]
                )
            self.assertEqual(
                sum(command["stage"] == "STAGE_1" for command in result["commands"]),
                4,
            )

    def test_refuses_nonempty_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            protocol_path, _ = self.fixture(root)
            output = root / "private" / "plan"
            output.mkdir(parents=True)
            (output / "keep.txt").write_text("keep")
            with patch("src.training.prepare_sft_plateau_plan.REPO_ROOT", root):
                with self.assertRaises(FileExistsError):
                    prepare(protocol_path, output)

    def test_fails_on_source_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            protocol_path, data_dir = self.fixture(root)
            (data_dir / "dataset.jsonl").write_text("{}\n", encoding="utf-8")
            with patch("src.training.prepare_sft_plateau_plan.REPO_ROOT", root):
                with self.assertRaisesRegex(ValueError, "binding mismatch"):
                    prepare(protocol_path, root / "private" / "plan")


if __name__ == "__main__":
    unittest.main()
