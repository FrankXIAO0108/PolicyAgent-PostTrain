"""Prepare frozen evaluation configs only; does not run GPU/API evaluation."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.run_retail_agentic_grpo import load_json, save_json, sha256  # noqa: E402
from src.training.rollout_diagnostics import validate_sampling_request  # noqa: E402
from src.training.dirhash import directory_sha256  # noqa: E402

EVAL_SEEDS = tuple(range(2026090611, 2026090619))
GREEDY_SEED = 2026090619


def evaluation_configs(training: dict, model: dict, arm: str) -> dict:
    configs = {}
    for index, seed in enumerate((*EVAL_SEEDS, GREEDY_SEED)):
        greedy = index == len(EVAL_SEEDS)
        c = deepcopy(training)
        c["execution_mode"] = "ROLLOUT_DIAGNOSTIC"
        c.pop("engineering_acceptance", None)
        c["claims"] = {**c["claims"], "parameter_update_allowed": False,
                       "development_training_only": False, "development_probe_only": True,
                       "paired_eval_arm": arm, "paired_eval_contract": "task113-sft30-v7.1-acc8-s50-v1"}
        c["model"] = deepcopy(model)
        c["model_loading"]["mode"] = "qwen3_bf16_inference_v1"
        c["seed"] = seed
        n = 1 if greedy else 4
        c["grpo"].update(max_steps=1, learning_rate=0, beta=0, num_generations=n,
                         gradient_accumulation_steps=n, steps_per_generation=n,
                         temperature=1.0 if greedy else 0.8, gradient_checkpointing=False,
                         log_completions=False, warmup_steps=0)
        c["sampling"] = ({"mode": "TRUE_GREEDY", "do_sample": False} if greedy else {
            "mode": "STOCHASTIC_GROUP_SAMPLING", "contract_version": "fixed-n4-single-group-v1",
            "do_sample": True, "temperature": 0.8, "top_p": 1, "top_k": 0,
        })
        c["diagnostic"] = {
            "strict_opening_manifest_binding": True, "expected_rollouts": n,
            "expected_tasks": 1, "expected_rollouts_per_task": n,
            "groups_per_task": 1, "group_size": n, "weight_update_expected": False,
            "task_selection_uses_reward": False, "trainer_max_steps_unused": True,
        }
        validate_sampling_request(c, 8192, 1)
        configs["greedy" if greedy else f"n4_{index + 1:02}"] = c
    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--arm", choices=["SFT", "GRPO"], required=True)
    parser.add_argument("--training-run", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Use a new eval config directory")
    training = load_json(args.training_config)
    model = deepcopy(training["model"])
    if args.arm == "GRPO":
        if args.training_run is None:
            parser.error("GRPO evaluation requires the completed --training-run")
        manifest = load_json(args.training_run / "run_manifest.json")
        optimization = load_json(args.training_run / "optimization_evidence.json")
        if manifest["status"] != "COMPLETED" or optimization["status"] != "PASSED":
            raise ValueError("Cannot evaluate an unaccepted training run as completed")
        if load_json(args.training_run / "config.json") != training:
            raise ValueError("Completed run differs from frozen training config")
        merged = manifest["artifacts"]["merged_model"]
        model.update(name_or_path=merged["path"], expected_sha256=merged["sha256"],
                     source_stage="GRPO_TASK113_SFT30_STAGED_V7_1_STEP50")
    actual = directory_sha256(Path(model["name_or_path"]))
    if actual != model["expected_sha256"]:
        raise ValueError("Evaluation model hash mismatch")
    configs = evaluation_configs(training, model, args.arm)
    commands = []
    for name, config in configs.items():
        path = args.output_dir / f"{name}.json"
        save_json(path, config)
        commands.append([sys.executable, "-m", "src.training.run_retail_agentic_grpo",
                         "--config", str(path), "--output-dir", str(args.output_dir / "results" / name),
                         "--allow-dirty", "--sample-only", "--completion-budget", "8192",
                         "--groups-per-task", "1"])
    save_json(args.output_dir / "eval_commands.json", {
        "status": "CONFIGS_PREPARED_NOT_EXECUTED", "arm": args.arm,
        "expected_rollouts": 33, "commands": commands,
        "training_config_sha256": sha256(args.training_config), "model_sha256": actual,
        "external_api_called": False, "gpu_used": False,
    })
    print(f"Prepared {args.arm}: 1 greedy + 8 x n4 = 33 rollouts; not executed")


if __name__ == "__main__":
    main()
