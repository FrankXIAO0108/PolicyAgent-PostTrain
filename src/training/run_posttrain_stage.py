from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from src.training.run_posttrain_engineering_smoke import (
    REPO_ROOT,
    _evaluate,
    _lora_config,
    _merge_adapter,
    _precision,
    _save_json,
    _supported_config,
    _train_metrics,
    correct_arguments_reward,
    correct_tool_reward,
    directory_sha256,
    load_json,
    load_jsonl,
    sha256,
    valid_json_reward,
    validate_inputs,
)


STAGE_ORDER = ("base", "sft", "dpo", "grpo")


def _load_runtime(config: dict[str, Any]) -> dict[str, Any]:
    try:
        import accelerate
        import bitsandbytes
        import datasets
        import peft
        import torch
        import transformers
        import trl
        from datasets import Dataset
        from peft import LoraConfig, PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        from trl import DPOConfig, DPOTrainer, GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError("Missing cloud training dependency.") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; cloud training is not allowed to fall back to CPU.")
    seed = int(config["seed"])
    random.seed(seed)
    set_seed(seed)
    dtype, bf16, fp16 = _precision(torch, config["precision"])
    model_spec = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_spec["name_or_path"],
        revision=model_spec["revision"],
        trust_remote_code=bool(model_spec["trust_remote_code"]),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return {
        "torch": torch,
        "Dataset": Dataset,
        "LoraConfig": LoraConfig,
        "PeftModel": PeftModel,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "SFTConfig": SFTConfig,
        "SFTTrainer": SFTTrainer,
        "DPOConfig": DPOConfig,
        "DPOTrainer": DPOTrainer,
        "GRPOConfig": GRPOConfig,
        "GRPOTrainer": GRPOTrainer,
        "tokenizer": tokenizer,
        "dtype": dtype,
        "bf16": bf16,
        "fp16": fp16,
        "versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "trl": trl.__version__,
            "datasets": datasets.__version__,
            "peft": peft.__version__,
            "accelerate": accelerate.__version__,
            "bitsandbytes": bitsandbytes.__version__,
            "cuda_available": True,
            "cuda_version": torch.version.cuda,
            "gpu_names": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
            "effective_dtype": str(dtype),
        },
    }


def _progress_path(output_dir: Path) -> Path:
    return output_dir / "progress_manifest.json"


def _write_progress(output_dir: Path, progress: dict[str, Any]) -> None:
    _save_json(_progress_path(output_dir), progress)


def _load_progress(output_dir: Path, preflight: dict[str, Any]) -> dict[str, Any]:
    path = _progress_path(output_dir)
    if not path.is_file():
        raise FileNotFoundError(f"Missing prior-stage progress manifest: {path}")
    progress = load_json(path)
    bindings = progress.get("bindings", {})
    expected = {
        "config_sha256": preflight["config_sha256"],
        "data_manifest_sha256": preflight["data_manifest_sha256"],
        "git_commit": preflight["git_commit"],
    }
    actual = {
        "config_sha256": bindings.get("config_sha256"),
        "data_manifest_sha256": bindings.get("data_manifest_sha256"),
        "git_commit": progress.get("git", {}).get("commit"),
    }
    if actual != expected:
        raise RuntimeError(f"Stage bindings changed: actual={actual}, expected={expected}")
    return progress


def _save_log_history(path: Path, history: list[dict[str, Any]]) -> dict[str, Any]:
    text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n"
        for row in history
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))
    return {"path": str(path.resolve()), "sha256": sha256(path), "rows": len(history)}


def _checkpoint_record(trainer_dir: Path) -> dict[str, Any]:
    checkpoints = [
        path
        for path in trainer_dir.glob("checkpoint-*")
        if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit()
    ]
    if not checkpoints:
        raise FileNotFoundError(f"No resumable checkpoint in {trainer_dir}")
    checkpoint = max(
        checkpoints, key=lambda path: int(path.name.removeprefix("checkpoint-"))
    )
    return {"path": str(checkpoint.resolve()), "sha256": directory_sha256(checkpoint)}


def _stage_record(
    *,
    stage: str,
    adapter: Path,
    merged: Path,
    trainer_dir: Path,
    metrics: dict[str, Any],
    log_history: dict[str, Any],
) -> dict[str, Any]:
    return {
        "stage": stage.upper(),
        "status": "COMPLETED",
        "artifact_path": str(adapter.resolve()),
        "artifact_sha256": directory_sha256(adapter),
        "adapter": {
            "path": str(adapter.resolve()),
            "sha256": directory_sha256(adapter),
        },
        "merged_model": {
            "path": str(merged.resolve()),
            "sha256": directory_sha256(merged),
        },
        "checkpoint": _checkpoint_record(trainer_dir),
        "loss_history": log_history,
        "train_metrics": metrics,
    }


def _common_args(config: dict[str, Any], runtime: dict[str, Any], stage: str) -> dict[str, Any]:
    spec = config[stage]
    return {
        "logging_steps": 1,
        "logging_first_step": True,
        "save_strategy": "steps",
        "save_steps": int(spec["max_steps"]),
        "save_total_limit": 1,
        "report_to": "none",
        "seed": int(config["seed"]),
        "data_seed": int(config["seed"]),
        "bf16": runtime["bf16"],
        "fp16": runtime["fp16"],
        "gradient_checkpointing": False,
        "remove_unused_columns": False,
    }


def _evaluate_stage(
    model_path: str,
    rows: list[dict[str, Any]],
    output_path: Path,
    config: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    model_spec = config["model"]
    return _evaluate(
        model_path,
        rows,
        output_path,
        config["evaluation"],
        runtime["dtype"],
        runtime["torch"],
        runtime["AutoModelForCausalLM"],
        runtime["AutoTokenizer"],
        bool(model_spec["trust_remote_code"]),
    )


def run_stage(
    *, stage: str, preflight: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    config = preflight["config"]
    data_dir: Path = preflight["data_dir"]
    if stage == "base":
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        if not output_dir.is_dir():
            raise FileNotFoundError(f"Run directory does not exist: {output_dir}")

    runtime = _load_runtime(config)
    model_spec = config["model"]
    model_id = model_spec["name_or_path"]
    trust_remote_code = bool(model_spec["trust_remote_code"])
    holdout_rows = load_jsonl(data_dir / "holdout.jsonl")

    if stage == "base":
        progress = {
            "schema_version": "posttrain-engineering-smoke-progress-v1",
            "scope": config["scope"],
            "status": "IN_PROGRESS",
            "claims": config["claims"],
            "git": {
                "commit": preflight["git_commit"],
                "branch": preflight["git_branch"],
                "dirty_at_start": preflight["git_dirty_at_start"],
            },
            "bindings": {
                "config_path": preflight["config_path"],
                "config_sha256": preflight["config_sha256"],
                "data_manifest_path": str(preflight["data_manifest_path"]),
                "data_manifest_sha256": preflight["data_manifest_sha256"],
                "model_name_or_path": model_id,
                "model_revision": model_spec["revision"],
                "seed": int(config["seed"]),
            },
            "environment": runtime["versions"],
            "completed_stages": [],
            "stage_records": {},
            "holdout_evaluations": {},
            "formal_retail_readiness_gate_opened": False,
            "business_improvement_claim_allowed": False,
        }
        _save_json(output_dir / "environment.json", runtime["versions"])
        progress["holdout_evaluations"]["base"] = _evaluate_stage(
            model_id,
            holdout_rows,
            output_dir / "evaluation_base.json",
            config,
            runtime,
        )
        progress["completed_stages"].append("base")
        _write_progress(output_dir, progress)
        return progress

    progress = _load_progress(output_dir, preflight)
    required_previous = STAGE_ORDER[STAGE_ORDER.index(stage) - 1]
    if required_previous not in progress.get("completed_stages", []):
        raise RuntimeError(f"Stage {stage} requires completed stage {required_previous}.")
    if stage in progress.get("completed_stages", []):
        raise RuntimeError(f"Stage already completed: {stage}")

    tokenizer = runtime["tokenizer"]
    common = _common_args(config, runtime, stage)
    lora = _lora_config(config, runtime["LoraConfig"])

    if stage == "sft":
        rows = load_jsonl(data_dir / "sft.jsonl")
        source_model = model_id
        adapter = output_dir / "sft_adapter"
        merged = output_dir / "sft_merged"
        trainer_dir = output_dir / "sft_trainer"
        args = _supported_config(
            runtime["SFTConfig"],
            {
                **common,
                **config["sft"],
                "output_dir": str(trainer_dir),
                "max_length": config["max_length"],
                "completion_only_loss": True,
            },
        )
        model = runtime["AutoModelForCausalLM"].from_pretrained(
            source_model,
            revision=model_spec["revision"],
            dtype=runtime["dtype"],
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
        )
        trainer = runtime["SFTTrainer"](
            model=model,
            args=args,
            train_dataset=runtime["Dataset"].from_list(rows),
            processing_class=tokenizer,
            peft_config=lora,
        )
    elif stage == "dpo":
        rows = load_jsonl(data_dir / "dpo.jsonl")
        source_model = str(output_dir / "sft_merged")
        adapter = output_dir / "dpo_adapter"
        merged = output_dir / "dpo_merged"
        trainer_dir = output_dir / "dpo_trainer"
        args = _supported_config(
            runtime["DPOConfig"],
            {
                **common,
                **config["dpo"],
                "output_dir": str(trainer_dir),
                "max_length": config["max_length"],
            },
        )
        model = runtime["AutoModelForCausalLM"].from_pretrained(
            source_model,
            dtype=runtime["dtype"],
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
        )
        trainer = runtime["DPOTrainer"](
            model=model,
            ref_model=None,
            args=args,
            train_dataset=runtime["Dataset"].from_list(rows),
            processing_class=tokenizer,
            peft_config=lora,
        )
    else:
        rows = load_jsonl(data_dir / "grpo.jsonl")
        source_model = str(output_dir / "dpo_merged")
        adapter = output_dir / "grpo_adapter"
        merged = output_dir / "grpo_merged"
        trainer_dir = output_dir / "grpo_trainer"
        args = _supported_config(
            runtime["GRPOConfig"],
            {
                **common,
                **config["grpo"],
                "output_dir": str(trainer_dir),
                "max_prompt_length": config["max_length"],
                "reward_weights": config["grpo"]["reward_weights"],
            },
        )
        model = runtime["AutoModelForCausalLM"].from_pretrained(
            source_model,
            dtype=runtime["dtype"],
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
        )
        trainer = runtime["GRPOTrainer"](
            model=model,
            args=args,
            train_dataset=runtime["Dataset"].from_list(rows),
            reward_funcs=[
                valid_json_reward,
                correct_tool_reward,
                correct_arguments_reward,
            ],
            processing_class=tokenizer,
            peft_config=lora,
        )

    started = time.time()
    train_result = trainer.train()
    metrics = _train_metrics(train_result, time.time() - started)
    trainer.save_state()
    trainer.save_model(adapter)
    log_record = _save_log_history(
        output_dir / "logs" / f"{stage}_log_history.jsonl",
        list(trainer.state.log_history),
    )
    del trainer, model
    runtime["torch"].cuda.empty_cache()
    _merge_adapter(
        source_model,
        adapter,
        merged,
        tokenizer,
        runtime["dtype"],
        runtime["AutoModelForCausalLM"],
        runtime["PeftModel"],
        trust_remote_code,
    )
    progress["stage_records"][stage] = _stage_record(
        stage=stage,
        adapter=adapter,
        merged=merged,
        trainer_dir=trainer_dir,
        metrics=metrics,
        log_history=log_record,
    )
    progress["holdout_evaluations"][stage] = _evaluate_stage(
        str(merged),
        holdout_rows,
        output_dir / f"evaluation_{stage}.json",
        config,
        runtime,
    )
    progress["completed_stages"].append(stage)
    _write_progress(output_dir, progress)

    if stage == "grpo":
        manifest = {
            key: value
            for key, value in progress.items()
            if key not in {"completed_stages", "stage_records"}
        }
        manifest["schema_version"] = "posttrain-engineering-smoke-run-v1"
        manifest["status"] = "COMPLETED"
        manifest["stages"] = [
            progress["stage_records"][name] for name in ("sft", "dpo", "grpo")
        ]
        _save_json(output_dir / "run_manifest.json", manifest)
        return manifest
    return progress


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGE_ORDER, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "posttrain_engineering_smoke_v1.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    preflight = validate_inputs(args.config.resolve(), args.allow_dirty)
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "stage": args.stage,
                    "config_sha256": preflight["config_sha256"],
                    "data_manifest_sha256": preflight["data_manifest_sha256"],
                    "git_commit": preflight["git_commit"],
                    "git_dirty_at_start": preflight["git_dirty_at_start"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    output_dir = args.output_dir.resolve()
    try:
        result = run_stage(stage=args.stage, preflight=preflight, output_dir=output_dir)
    except Exception as exc:
        if output_dir.exists():
            _save_json(
                output_dir / f"failure_{args.stage}.json",
                {
                    "schema_version": "posttrain-engineering-stage-failure-v1",
                    "stage": args.stage,
                    "status": "FAILED",
                    "git_commit": preflight["git_commit"],
                    "config_sha256": preflight["config_sha256"],
                    "data_manifest_sha256": preflight["data_manifest_sha256"],
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
