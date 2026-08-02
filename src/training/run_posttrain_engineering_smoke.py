from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_DATA_FILES = ("sft.jsonl", "dpo.jsonl", "grpo.jsonl", "holdout.jsonl")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(file_path.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest().upper()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def validate_inputs(config_path: Path, allow_dirty: bool) -> dict[str, Any]:
    config = load_json(config_path)
    if config.get("scope") != "ISOLATED_ENGINEERING_SMOKE":
        raise ValueError("Config scope must be ISOLATED_ENGINEERING_SMOKE")
    claims = config.get("claims", {})
    if claims.get("formal_retail_gate_unchanged") is not True:
        raise ValueError("Smoke must leave the formal Retail gate unchanged")
    if claims.get("business_improvement_claim_allowed") is not False:
        raise ValueError("Smoke cannot authorize a business-improvement claim")

    data_dir = REPO_ROOT / config["data_dir"]
    data_manifest_path = data_dir / "manifest.json"
    data_manifest = load_json(data_manifest_path)
    if data_manifest.get("leakage_checks", {}).get("passed") is not True:
        raise ValueError("Synthetic data leakage check did not pass")
    if data_manifest.get("contains_tau2_frozen_tasks") is not False:
        raise ValueError("Engineering smoke data must exclude frozen tau2 tasks")
    for name in REQUIRED_DATA_FILES:
        path = data_dir / name
        key = path.stem
        expected = data_manifest["files"][key]["sha256"]
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"Data hash mismatch: {path}: {actual} != {expected}")

    commit = git_value("rev-parse", "HEAD")
    status = git_value("status", "--porcelain")
    if status and not allow_dirty:
        raise RuntimeError(
            "Refusing to train from a dirty worktree. Commit the execution package first "
            "or pass --allow-dirty for a non-release diagnostic."
        )
    return {
        "config": config,
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256(config_path),
        "data_dir": data_dir,
        "data_manifest_path": data_manifest_path,
        "data_manifest_sha256": sha256(data_manifest_path),
        "git_commit": commit,
        "git_branch": git_value("branch", "--show-current"),
        "git_dirty_at_start": bool(status),
    }


def _supported_config(config_class: type, values: dict[str, Any]) -> Any:
    supported = set(inspect.signature(config_class).parameters)
    return config_class(**{key: value for key, value in values.items() if key in supported})


def _precision(torch: Any, requested: str) -> tuple[Any, bool, bool]:
    cuda = torch.cuda.is_available()
    if requested == "bf16" and cuda and torch.cuda.is_bf16_supported():
        return torch.bfloat16, True, False
    if requested in {"bf16", "fp16"} and cuda:
        return torch.float16, False, True
    return torch.float32, False, False


def _lora_config(config: dict[str, Any], LoraConfig: type) -> Any:
    spec = config["lora"]
    return LoraConfig(
        r=spec["r"],
        lora_alpha=spec["alpha"],
        lora_dropout=spec["dropout"],
        target_modules=spec["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )


def _normalize_completion(completion: Any) -> str:
    if isinstance(completion, str):
        return completion.strip()
    if isinstance(completion, list) and completion:
        tail = completion[-1]
        if isinstance(tail, dict):
            return str(tail.get("content", "")).strip()
    return str(completion).strip()


def _parse_action(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    if start < 0:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    return value


def valid_json_reward(completions: list[Any], **_: Any) -> list[float]:
    return [float(_parse_action(_normalize_completion(item)) is not None) for item in completions]


def correct_tool_reward(
    completions: list[Any], expected_action: list[str], **_: Any
) -> list[float]:
    rewards: list[float] = []
    for completion, expected_text in zip(completions, expected_action):
        predicted = _parse_action(_normalize_completion(completion)) or {}
        expected = json.loads(expected_text)
        rewards.append(float(predicted.get("tool") == expected.get("tool")))
    return rewards


def correct_arguments_reward(
    completions: list[Any], expected_action: list[str], **_: Any
) -> list[float]:
    rewards: list[float] = []
    for completion, expected_text in zip(completions, expected_action):
        predicted = _parse_action(_normalize_completion(completion)) or {}
        expected = json.loads(expected_text)
        rewards.append(float(predicted.get("arguments") == expected.get("arguments")))
    return rewards


def _train_metrics(train_result: Any, elapsed_seconds: float) -> dict[str, Any]:
    metrics = dict(getattr(train_result, "metrics", {}) or {})
    metrics["elapsed_seconds_wall"] = elapsed_seconds
    return metrics


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _merge_adapter(
    base_or_merged_path: str,
    adapter_path: Path,
    output_path: Path,
    tokenizer: Any,
    dtype: Any,
    AutoModelForCausalLM: type,
    PeftModel: type,
    trust_remote_code: bool,
) -> None:
    model = AutoModelForCausalLM.from_pretrained(
        base_or_merged_path,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, str(adapter_path))
    merged = model.merge_and_unload()
    merged.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)


def _evaluate(
    model_path: str,
    rows: list[dict[str, Any]],
    output_path: Path,
    eval_config: dict[str, Any],
    dtype: Any,
    torch: Any,
    AutoModelForCausalLM: type,
    AutoTokenizer: type,
    trust_remote_code: bool,
) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    results: list[dict[str, Any]] = []
    for row in rows:
        encoded = tokenizer(row["prompt"], return_tensors="pt").to(device)
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=eval_config["max_new_tokens"],
                do_sample=eval_config["do_sample"],
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        completion_ids = generated[0, encoded["input_ids"].shape[1] :]
        completion = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        predicted = _parse_action(completion)
        expected = json.loads(row["expected_action"])
        results.append(
            {
                "scenario_id": row["scenario_id"],
                "category": row["category"],
                "completion": completion,
                "valid_json": predicted is not None,
                "tool_match": bool(predicted and predicted.get("tool") == expected["tool"]),
                "arguments_match": bool(
                    predicted and predicted.get("arguments") == expected["arguments"]
                ),
                "exact_action_match": predicted == expected,
            }
        )
    count = len(results)
    metrics = {
        "rows": count,
        "valid_json_rate": sum(item["valid_json"] for item in results) / count,
        "tool_match_rate": sum(item["tool_match"] for item in results) / count,
        "arguments_match_rate": sum(item["arguments_match"] for item in results) / count,
        "exact_action_match_rate": sum(item["exact_action_match"] for item in results)
        / count,
    }
    payload = {"model_path": model_path, "metrics": metrics, "rows": results}
    _save_json(output_path, payload)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def _stage_record(path: Path, stage: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": stage,
        "status": "COMPLETED",
        "artifact_path": str(path),
        "artifact_sha256": directory_sha256(path),
        "train_metrics": metrics,
    }


def run(preflight: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    try:
        import accelerate
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
        raise RuntimeError(
            "Training dependencies are missing. Install requirements-posttrain-smoke.txt "
            "inside the rented server environment."
        ) from exc

    config = preflight["config"]
    data_dir: Path = preflight["data_dir"]
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config["seed"])
    random.seed(seed)
    set_seed(seed)
    model_spec = config["model"]
    model_id = model_spec["name_or_path"]
    trust_remote_code = bool(model_spec["trust_remote_code"])
    dtype, bf16, fp16 = _precision(torch, config["precision"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=model_spec["revision"],
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sft_rows = load_jsonl(data_dir / "sft.jsonl")
    dpo_rows = load_jsonl(data_dir / "dpo.jsonl")
    grpo_rows = load_jsonl(data_dir / "grpo.jsonl")
    holdout_rows = load_jsonl(data_dir / "holdout.jsonl")
    common = {
        "logging_steps": 1,
        "save_strategy": "no",
        "report_to": "none",
        "seed": seed,
        "data_seed": seed,
        "bf16": bf16,
        "fp16": fp16,
        "gradient_checkpointing": False,
        "remove_unused_columns": False,
    }
    versions = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "trl": trl.__version__,
        "datasets": datasets.__version__,
        "peft": peft.__version__,
        "accelerate": accelerate.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_names": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
        "effective_dtype": str(dtype),
    }
    _save_json(output_dir / "environment.json", versions)

    evaluations: dict[str, Any] = {}
    evaluations["base"] = _evaluate(
        model_id,
        holdout_rows,
        output_dir / "evaluation_base.json",
        config["evaluation"],
        dtype,
        torch,
        AutoModelForCausalLM,
        AutoTokenizer,
        trust_remote_code,
    )

    sft_adapter = output_dir / "sft_adapter"
    sft_merged = output_dir / "sft_merged"
    sft_args = _supported_config(
        SFTConfig,
        {
            **common,
            **config["sft"],
            "output_dir": str(output_dir / "sft_trainer"),
            "max_length": config["max_length"],
            "completion_only_loss": True,
        },
    )
    sft_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=model_spec["revision"],
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )
    sft_trainer = SFTTrainer(
        model=sft_model,
        args=sft_args,
        train_dataset=Dataset.from_list(sft_rows),
        processing_class=tokenizer,
        peft_config=_lora_config(config, LoraConfig),
    )
    started = time.time()
    sft_result = sft_trainer.train()
    sft_metrics = _train_metrics(sft_result, time.time() - started)
    sft_trainer.save_model(sft_adapter)
    del sft_trainer, sft_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _merge_adapter(
        model_id,
        sft_adapter,
        sft_merged,
        tokenizer,
        dtype,
        AutoModelForCausalLM,
        PeftModel,
        trust_remote_code,
    )
    evaluations["sft"] = _evaluate(
        str(sft_merged),
        holdout_rows,
        output_dir / "evaluation_sft.json",
        config["evaluation"],
        dtype,
        torch,
        AutoModelForCausalLM,
        AutoTokenizer,
        trust_remote_code,
    )

    dpo_adapter = output_dir / "dpo_adapter"
    dpo_merged = output_dir / "dpo_merged"
    dpo_args = _supported_config(
        DPOConfig,
        {
            **common,
            **config["dpo"],
            "output_dir": str(output_dir / "dpo_trainer"),
            "max_length": config["max_length"],
        },
    )
    dpo_model = AutoModelForCausalLM.from_pretrained(
        sft_merged,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )
    dpo_trainer = DPOTrainer(
        model=dpo_model,
        ref_model=None,
        args=dpo_args,
        train_dataset=Dataset.from_list(dpo_rows),
        processing_class=tokenizer,
        peft_config=_lora_config(config, LoraConfig),
    )
    started = time.time()
    dpo_result = dpo_trainer.train()
    dpo_metrics = _train_metrics(dpo_result, time.time() - started)
    dpo_trainer.save_model(dpo_adapter)
    del dpo_trainer, dpo_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _merge_adapter(
        str(sft_merged),
        dpo_adapter,
        dpo_merged,
        tokenizer,
        dtype,
        AutoModelForCausalLM,
        PeftModel,
        trust_remote_code,
    )
    evaluations["dpo"] = _evaluate(
        str(dpo_merged),
        holdout_rows,
        output_dir / "evaluation_dpo.json",
        config["evaluation"],
        dtype,
        torch,
        AutoModelForCausalLM,
        AutoTokenizer,
        trust_remote_code,
    )

    grpo_adapter = output_dir / "grpo_adapter"
    grpo_merged = output_dir / "grpo_merged"
    grpo_args = _supported_config(
        GRPOConfig,
        {
            **common,
            **config["grpo"],
            "output_dir": str(output_dir / "grpo_trainer"),
            "max_prompt_length": config["max_length"],
            "reward_weights": config["grpo"]["reward_weights"],
        },
    )
    grpo_model = AutoModelForCausalLM.from_pretrained(
        dpo_merged,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )
    grpo_trainer = GRPOTrainer(
        model=grpo_model,
        args=grpo_args,
        train_dataset=Dataset.from_list(grpo_rows),
        reward_funcs=[valid_json_reward, correct_tool_reward, correct_arguments_reward],
        processing_class=tokenizer,
        peft_config=_lora_config(config, LoraConfig),
    )
    started = time.time()
    grpo_result = grpo_trainer.train()
    grpo_metrics = _train_metrics(grpo_result, time.time() - started)
    grpo_trainer.save_model(grpo_adapter)
    del grpo_trainer, grpo_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _merge_adapter(
        str(dpo_merged),
        grpo_adapter,
        grpo_merged,
        tokenizer,
        dtype,
        AutoModelForCausalLM,
        PeftModel,
        trust_remote_code,
    )
    evaluations["grpo"] = _evaluate(
        str(grpo_merged),
        holdout_rows,
        output_dir / "evaluation_grpo.json",
        config["evaluation"],
        dtype,
        torch,
        AutoModelForCausalLM,
        AutoTokenizer,
        trust_remote_code,
    )

    stages = [
        _stage_record(sft_adapter, "SFT", sft_metrics),
        _stage_record(dpo_adapter, "DPO", dpo_metrics),
        _stage_record(grpo_adapter, "GRPO", grpo_metrics),
    ]
    manifest = {
        "schema_version": "posttrain-engineering-smoke-run-v1",
        "scope": config["scope"],
        "status": "COMPLETED",
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
            "seed": seed,
        },
        "environment": versions,
        "stages": stages,
        "holdout_evaluations": evaluations,
        "formal_retail_readiness_gate_opened": False,
        "business_improvement_claim_allowed": False,
    }
    _save_json(output_dir / "run_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
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
        printable = {**preflight, "config": preflight["config"]}
        printable["data_dir"] = str(preflight["data_dir"])
        printable["data_manifest_path"] = str(preflight["data_manifest_path"])
        print(json.dumps(printable, ensure_ascii=False, indent=2))
        return
    output_dir = args.output_dir.resolve()
    try:
        manifest = run(preflight, output_dir)
    except Exception as exc:
        if output_dir.exists():
            _save_json(
                output_dir / "failure_manifest.json",
                {
                    "schema_version": "posttrain-engineering-smoke-failure-v1",
                    "scope": preflight["config"]["scope"],
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
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
