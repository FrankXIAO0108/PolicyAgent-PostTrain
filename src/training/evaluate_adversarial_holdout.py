from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
STAGES = ("base", "sft", "dpo", "grpo")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(file_path.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest().upper()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def parse_model_spec(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("模型参数必须是 stage=path_or_model_id")
    stage, model = value.split("=", 1)
    if stage not in STAGES or not model.strip():
        raise argparse.ArgumentTypeError(f"无效模型参数: {value}")
    return stage, model.strip()


def _extract_action(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    if start < 0:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _strict_action(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text.strip())
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _schema_valid(action: dict[str, Any] | None, allowed_tools: set[str]) -> bool:
    if action is None or set(action) != {"tool", "arguments"}:
        return False
    return isinstance(action["tool"], str) and action["tool"] in allowed_tools and isinstance(
        action["arguments"], dict
    )


def score_completion(
    row: dict[str, Any], completion: str, allowed_tools: set[str]
) -> dict[str, Any]:
    extracted = _extract_action(completion)
    strict = _strict_action(completion)
    expected = json.loads(row["expected_action"])
    tool_match = bool(extracted and extracted.get("tool") == expected.get("tool"))
    arguments_match = bool(
        extracted and extracted.get("arguments") == expected.get("arguments")
    )
    exact = extracted == expected
    return {
        "scenario_id": row["scenario_id"],
        "category": row["category"],
        "completion": completion,
        "extractable_json": extracted is not None,
        "strict_json_object": strict is not None,
        "schema_valid": _schema_valid(strict, allowed_tools),
        "registered_tool": bool(extracted and extracted.get("tool") in allowed_tools),
        "tool_match": tool_match,
        "arguments_match": arguments_match,
        "exact_action_match": exact,
        "strict_exact_action_match": strict == expected,
    }


METRIC_FIELDS = (
    "extractable_json",
    "strict_json_object",
    "schema_valid",
    "registered_tool",
    "tool_match",
    "arguments_match",
    "exact_action_match",
    "strict_exact_action_match",
)


def aggregate_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("评测结果不能为空")

    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        count = len(group)
        summary: dict[str, Any] = {"rows": count}
        for field in METRIC_FIELDS:
            summary[f"{field}_rate"] = sum(bool(row[field]) for row in group) / count
        summary["format_gap_count"] = sum(
            row["extractable_json"] and not row["strict_json_object"] for row in group
        )
        return summary

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)
    return {
        "overall": summarize(rows),
        "by_category": {category: summarize(grouped[category]) for category in sorted(grouped)},
    }


def validate_inputs(
    config_path: Path,
    model_specs: dict[str, str],
    allow_dirty: bool,
) -> dict[str, Any]:
    config = load_json(config_path)
    if config.get("scope") != "ISOLATED_EVALUATION_ONLY":
        raise ValueError("Config scope must be ISOLATED_EVALUATION_ONLY")
    if set(model_specs) != set(STAGES):
        missing = sorted(set(STAGES) - set(model_specs))
        extra = sorted(set(model_specs) - set(STAGES))
        raise ValueError(f"模型阶段必须恰好为 {STAGES}; missing={missing}, extra={extra}")

    holdout_dir = (REPO_ROOT / config["holdout_dir"]).resolve()
    manifest_path = holdout_dir / "manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("scope") != "ISOLATED_EVALUATION_ONLY":
        raise ValueError("Holdout scope mismatch")
    if manifest.get("training_use_prohibited") is not True:
        raise ValueError("Holdout must be prohibited from training")
    if manifest.get("business_metric_claim_allowed") is not False:
        raise ValueError("Holdout cannot authorize a business claim")
    if manifest.get("leakage_checks", {}).get("passed") is not True:
        raise ValueError("Holdout leakage checks did not pass")

    holdout_path = holdout_dir / manifest["file"]["path"]
    actual_hash = sha256(holdout_path)
    if actual_hash != manifest["file"]["sha256"]:
        raise ValueError("Holdout hash mismatch")
    rows = load_jsonl(holdout_path)
    if len(rows) != manifest["rows"]:
        raise ValueError("Holdout row count mismatch")
    allowed_tools = set(manifest.get("allowed_tools", []))
    if not allowed_tools:
        raise ValueError("Holdout allowed tool registry is empty")
    for row in rows:
        expected = json.loads(row["expected_action"])
        if not _schema_valid(expected, allowed_tools):
            raise ValueError(f"Invalid expected action schema: {row['scenario_id']}")
        if row.get("training_use_prohibited") is not True:
            raise ValueError(f"Training prohibition missing: {row['scenario_id']}")

    status = git_value("status", "--porcelain")
    if status and not allow_dirty:
        raise RuntimeError("Refusing evaluation from a dirty worktree; commit first or use --allow-dirty")
    return {
        "config": config,
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "holdout_manifest_path": str(manifest_path),
        "holdout_manifest_sha256": sha256(manifest_path),
        "holdout_path": str(holdout_path),
        "holdout_sha256": actual_hash,
        "rows": rows,
        "allowed_tools": allowed_tools,
        "model_specs": model_specs,
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_branch": git_value("branch", "--show-current"),
        "git_dirty_at_start": bool(status),
    }


def _model_binding(model: str, revision: str) -> dict[str, Any]:
    path = Path(model).expanduser()
    if path.is_dir():
        resolved = path.resolve()
        return {
            "source": "local_directory",
            "path": str(resolved),
            "sha256": directory_sha256(resolved),
        }
    return {"source": "model_id", "name_or_path": model, "revision": revision}


def _runtime(config: dict[str, Any]) -> dict[str, Any]:
    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    except ImportError as exc:
        raise RuntimeError("Missing torch/transformers evaluation dependencies") from exc

    requested = config["precision"]
    if requested == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif requested in {"bf16", "fp16"} and torch.cuda.is_available():
        dtype = torch.float16
    else:
        dtype = torch.float32
    return {
        "torch": torch,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "set_seed": set_seed,
        "dtype": dtype,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "effective_dtype": str(dtype),
        },
    }


def evaluate_model(
    model_path: str,
    rows: list[dict[str, Any]],
    allowed_tools: set[str],
    config: dict[str, Any],
    runtime: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    torch = runtime["torch"]
    tokenizer = runtime["AutoTokenizer"].from_pretrained(
        model_path,
        revision=config["model_revision"],
        trust_remote_code=config["trust_remote_code"],
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = runtime["AutoModelForCausalLM"].from_pretrained(
        model_path,
        revision=config["model_revision"],
        dtype=runtime["dtype"],
        trust_remote_code=config["trust_remote_code"],
        low_cpu_mem_usage=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    scored: list[dict[str, Any]] = []
    for row in rows:
        encoded = tokenizer(row["prompt"], return_tensors="pt").to(device)
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=config["evaluation"]["max_new_tokens"],
                do_sample=config["evaluation"]["do_sample"],
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        completion_ids = generated[0, encoded["input_ids"].shape[1] :]
        completion = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        scored.append(score_completion(row, completion, allowed_tools))
    metrics = aggregate_scores(scored)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics, scored


def run(preflight: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = preflight["config"]
    runtime = _runtime(config)
    runtime["set_seed"](int(config["seed"]))
    save_json(output_dir / "environment.json", runtime["versions"])

    evaluations: dict[str, Any] = {}
    model_bindings: dict[str, Any] = {}
    for stage in STAGES:
        model_path = preflight["model_specs"][stage]
        binding = _model_binding(model_path, config["model_revision"])
        model_bindings[stage] = binding
        metrics, rows = evaluate_model(
            model_path,
            preflight["rows"],
            preflight["allowed_tools"],
            config,
            runtime,
        )
        payload = {
            "schema_version": "posttrain-adversarial-evaluation-stage-v2",
            "stage": stage,
            "model": binding,
            "metrics": metrics["overall"],
            "by_category": metrics["by_category"],
            "rows": rows,
        }
        save_json(output_dir / f"evaluation_{stage}.json", payload)
        evaluations[stage] = metrics
        save_json(
            output_dir / "progress_manifest.json",
            {
                "status": "RUNNING",
                "completed_stages": list(evaluations),
                "evaluations": evaluations,
                "model_bindings": model_bindings,
            },
        )

    manifest = {
        "schema_version": "posttrain-adversarial-evaluation-run-v2",
        "scope": "ISOLATED_EVALUATION_ONLY",
        "status": "COMPLETED",
        "git": {
            "commit": preflight["git_commit"],
            "branch": preflight["git_branch"],
            "dirty_at_start": preflight["git_dirty_at_start"],
        },
        "bindings": {
            "config_path": preflight["config_path"],
            "config_sha256": preflight["config_sha256"],
            "holdout_manifest_path": preflight["holdout_manifest_path"],
            "holdout_manifest_sha256": preflight["holdout_manifest_sha256"],
            "holdout_path": preflight["holdout_path"],
            "holdout_sha256": preflight["holdout_sha256"],
            "models": model_bindings,
        },
        "environment": runtime["versions"],
        "evaluations": evaluations,
        "formal_retail_readiness_gate_opened": False,
        "business_improvement_claim_allowed": False,
    }
    save_json(output_dir / "run_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "posttrain_adversarial_eval_v2.json",
    )
    parser.add_argument("--model", action="append", type=parse_model_spec, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    model_specs = dict(args.model)
    preflight = validate_inputs(args.config.resolve(), model_specs, args.allow_dirty)
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "config_sha256": preflight["config_sha256"],
                    "holdout_sha256": preflight["holdout_sha256"],
                    "rows": len(preflight["rows"]),
                    "models": model_specs,
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
        result = run(preflight, output_dir)
    except Exception as exc:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_json(
            output_dir / "failure.json",
            {
                "schema_version": "posttrain-adversarial-evaluation-failure-v2",
                "status": "FAILED",
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(),
                "config_sha256": preflight["config_sha256"],
                "holdout_sha256": preflight["holdout_sha256"],
            },
        )
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
