from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

from src.training.run_retail_agentic_grpo import (
    REPO_ROOT,
    build_retail_system_prompt,
    directory_sha256,
    load_json,
    load_jsonl,
    save_json,
    sha256,
    validate_upstream_checkout,
)


TOOL_NAMES = (
    "calculate",
    "cancel_pending_order",
    "exchange_delivered_order_items",
    "find_user_id_by_email",
    "find_user_id_by_name_zip",
    "get_item_details",
    "get_order_details",
    "get_product_details",
    "get_user_details",
    "list_all_product_types",
    "modify_pending_order_address",
    "modify_pending_order_items",
    "modify_pending_order_payment",
    "modify_user_address",
    "respond_to_user",
    "return_delivered_order_items",
    "transfer_to_human_agents",
)


def git_value(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def parse_tool_call(text: str) -> dict[str, Any] | None:
    start = text.find("<tool_call>")
    end = text.find("</tool_call>")
    if start < 0 or end <= start:
        return None
    payload = text[start + len("<tool_call>") : end].strip()
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    if not isinstance(parsed.get("name"), str) or not isinstance(
        parsed.get("arguments"), dict
    ):
        return None
    return parsed


def tool_call_completion(call: dict[str, Any]) -> str:
    return (
        "<tool_call>\n"
        + json.dumps(call, ensure_ascii=False, separators=(",", ":"))
        + "\n</tool_call>"
    )


def validate_inputs(config_path: Path, allow_dirty: bool) -> dict[str, Any]:
    config = load_json(config_path)
    if config.get("scope") != "ISOLATED_TOOL_PROTOCOL_WARMUP":
        raise ValueError("Tool SFT scope mismatch")
    claims = config.get("claims", {})
    if claims.get("formal_retail_gate_unchanged") is not True:
        raise ValueError("Tool SFT must leave the formal Retail gate unchanged")
    if claims.get("business_improvement_claim_allowed") is not False:
        raise ValueError("Tool SFT config cannot carry a business claim")
    quantization = config.get("quantization", {})
    if quantization != {
        "enabled": True,
        "mode": "4bit_nf4",
        "double_quant": True,
    }:
        raise ValueError("Tool SFT runner only supports the frozen 4-bit NF4 setup")
    data_dir = (REPO_ROOT / config["data_dir"]).resolve()
    manifest_path = data_dir / "manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("scope") != config["scope"]:
        raise ValueError("Tool SFT data scope mismatch")
    if manifest["claims"].get("business_improvement_claim_allowed") is not False:
        raise ValueError("Tool SFT data cannot carry a business claim")
    if manifest["leakage_checks"].get("passed") is not True:
        raise ValueError("Tool SFT leakage checks failed")
    for key, filename in (("sft", "sft.jsonl"), ("holdout", "holdout.jsonl")):
        path = data_dir / filename
        record = manifest["files"][key]
        if sha256(path) != record["sha256"] or len(load_jsonl(path)) != int(
            record["rows"]
        ):
            raise ValueError(f"Tool SFT {key} binding mismatch")
    upstream = validate_upstream_checkout(
        config["upstream"]["commit"],
        config["upstream"].get("source_package_sha256"),
        config["upstream"].get("required_files"),
    )
    model_path = Path(config["model"]["name_or_path"]).expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    model_hash = directory_sha256(model_path)
    if model_hash != config["model"]["expected_sha256"]:
        raise ValueError("Tool SFT starting model hash mismatch")
    dirty = bool(git_value("status", "--porcelain"))
    if dirty and not allow_dirty:
        raise RuntimeError("Commit Tool SFT inputs before running")
    return {
        "config": config,
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "data_dir": data_dir,
        "data_manifest_path": str(manifest_path),
        "data_manifest_sha256": sha256(manifest_path),
        "model_path": str(model_path),
        "model_sha256": model_hash,
        "upstream": upstream,
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_branch": git_value("branch", "--show-current"),
        "git_dirty_at_start": dirty,
    }


def load_runtime(preflight: dict[str, Any]) -> dict[str, Any]:
    import accelerate
    import bitsandbytes
    import datasets
    import peft
    import torch
    import transformers
    import trl
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        set_seed,
    )
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Tool SFT")
    config = preflight["config"]
    set_seed(int(config["seed"]))
    tokenizer = AutoTokenizer.from_pretrained(preflight["model_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return {
        "torch": torch,
        "Dataset": Dataset,
        "LoraConfig": LoraConfig,
        "PeftModel": PeftModel,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "SFTConfig": SFTConfig,
        "SFTTrainer": SFTTrainer,
        "tokenizer": tokenizer,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "transformers": transformers.__version__,
            "trl": trl.__version__,
            "datasets": datasets.__version__,
            "peft": peft.__version__,
            "accelerate": accelerate.__version__,
            "bitsandbytes": bitsandbytes.__version__,
        },
    }


def build_tools() -> list[Any]:
    from src.rl.retail_agentic_env import RetailAgenticEnvironment

    environment = RetailAgenticEnvironment()
    return [getattr(environment, name) for name in TOOL_NAMES]


def render_rows(
    rows: list[dict[str, Any]], tokenizer: Any, tools: list[Any]
) -> list[dict[str, Any]]:
    system = build_retail_system_prompt()
    rendered: list[dict[str, Any]] = []
    for row in rows:
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": row["user_message"]},
            ],
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
        )
        rendered.append(
            {
                "prompt": prompt,
                "completion": tool_call_completion(row["expected_call"]),
                "scenario_id": row["scenario_id"],
                "category": row["category"],
                "expected_call": json.dumps(
                    row["expected_call"], ensure_ascii=False, sort_keys=True
                ),
                "argument_policy": row["argument_policy"],
            }
        )
    return rendered


def evaluate(
    model_path: str,
    rows: list[dict[str, Any]],
    output_path: Path,
    preflight: dict[str, Any],
    runtime: dict[str, Any],
    tools: list[Any],
) -> dict[str, Any]:
    torch = runtime["torch"]
    tokenizer = runtime["AutoTokenizer"].from_pretrained(model_path)
    rendered = render_rows(rows, tokenizer, tools)
    model = runtime["AutoModelForCausalLM"].from_pretrained(
        model_path, dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to("cuda")
    model.eval()
    results = []
    for source, item in zip(rows, rendered, strict=True):
        encoded = tokenizer(item["prompt"], return_tensors="pt").to("cuda")
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                max_new_tokens=int(preflight["config"]["evaluation"]["max_new_tokens"]),
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(
            output[0, encoded["input_ids"].shape[1] :], skip_special_tokens=False
        ).strip()
        predicted = parse_tool_call(completion)
        expected = source["expected_call"]
        tool_match = bool(predicted and predicted["name"] == expected["name"])
        required_keys = set(expected["arguments"])
        arguments = predicted.get("arguments", {}) if predicted else {}
        schema_match = tool_match and required_keys <= set(arguments) and all(
            arguments[key] not in (None, "") for key in required_keys
        )
        exact_match = tool_match and arguments == expected["arguments"]
        argument_match = exact_match if source["argument_policy"] == "exact" else schema_match
        results.append(
            {
                "scenario_id": source["scenario_id"],
                "category": source["category"],
                "completion": completion,
                "contains_tool_call_tag": "<tool_call>" in completion,
                "valid_tool_call": predicted is not None,
                "tool_match": tool_match,
                "argument_match": argument_match,
                "exact_match": exact_match,
            }
        )
    count = len(results)
    metrics = {
        "rows": count,
        "valid_tool_call_rate": sum(row["valid_tool_call"] for row in results) / count,
        "malformed_tool_call_rate": sum(
            row["contains_tool_call_tag"] and not row["valid_tool_call"]
            for row in results
        )
        / count,
        "tool_match_rate": sum(row["tool_match"] for row in results) / count,
        "argument_match_rate": sum(row["argument_match"] for row in results) / count,
        "ordinary_text_rate": sum(
            not row["contains_tool_call_tag"] for row in results
        )
        / count,
    }
    save_json(output_path, {"model_path": model_path, "metrics": metrics, "rows": results})
    del model
    torch.cuda.empty_cache()
    return metrics


def run(preflight: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime = load_runtime(preflight)
    config = preflight["config"]
    tokenizer = runtime["tokenizer"]
    tools = build_tools()
    train_rows = load_jsonl(preflight["data_dir"] / "sft.jsonl")
    holdout_rows = load_jsonl(preflight["data_dir"] / "holdout.jsonl")
    rendered_train = render_rows(train_rows, tokenizer, tools)
    prompt_lengths = [
        len(tokenizer(row["prompt"], add_special_tokens=False)["input_ids"])
        for row in rendered_train
    ]
    sequence_lengths = [
        len(
            tokenizer(
                row["prompt"] + row["completion"], add_special_tokens=False
            )["input_ids"]
        )
        for row in rendered_train
    ]
    if max(sequence_lengths) > int(config["max_length"]):
        raise RuntimeError("Rendered Tool SFT sequence exceeds max_length")
    save_json(output_dir / "environment.json", runtime["versions"])
    base_metrics = evaluate(
        preflight["model_path"],
        holdout_rows,
        output_dir / "evaluation_base.json",
        preflight,
        runtime,
        tools,
    )
    torch = runtime["torch"]
    quant = runtime["BitsAndBytesConfig"](
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = runtime["AutoModelForCausalLM"].from_pretrained(
        preflight["model_path"],
        quantization_config=quant,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    spec = config["sft"]
    args = runtime["SFTConfig"](
        output_dir=str(output_dir / "trainer"),
        max_steps=int(spec["max_steps"]),
        learning_rate=float(spec["learning_rate"]),
        per_device_train_batch_size=int(spec["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(spec["gradient_accumulation_steps"]),
        max_length=int(config["max_length"]),
        completion_only_loss=True,
        logging_steps=1,
        save_strategy="steps",
        save_steps=int(spec["max_steps"]),
        save_total_limit=1,
        report_to="none",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=int(config["seed"]),
        data_seed=int(config["seed"]),
    )
    lora_spec = config["lora"]
    lora = runtime["LoraConfig"](
        r=int(lora_spec["r"]),
        lora_alpha=int(lora_spec["alpha"]),
        lora_dropout=float(lora_spec["dropout"]),
        target_modules=list(lora_spec["target_modules"]),
        task_type="CAUSAL_LM",
    )
    trainer = runtime["SFTTrainer"](
        model=model,
        args=args,
        train_dataset=runtime["Dataset"].from_list(rendered_train),
        processing_class=tokenizer,
        peft_config=lora,
    )
    started = time.time()
    result = trainer.train()
    adapter_dir = output_dir / "tool_sft_adapter"
    trainer.save_model(adapter_dir)
    save_json(output_dir / "train_metrics.json", result.metrics)
    save_json(output_dir / "log_history.json", trainer.state.log_history)
    del trainer, model
    torch.cuda.empty_cache()
    base = runtime["AutoModelForCausalLM"].from_pretrained(
        preflight["model_path"], dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    merged = runtime["PeftModel"].from_pretrained(
        base, str(adapter_dir)
    ).merge_and_unload()
    merged_dir = output_dir / "tool_sft_merged"
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    del base, merged
    sft_metrics = evaluate(
        str(merged_dir),
        holdout_rows,
        output_dir / "evaluation_sft.json",
        preflight,
        runtime,
        tools,
    )
    targets = config["evaluation"]
    protocol_warmup_gate_passed = (
        sft_metrics["valid_tool_call_rate"]
        >= float(targets["target_valid_tool_call_rate"])
        and sft_metrics["tool_match_rate"]
        >= float(targets["target_tool_match_rate"])
        and sft_metrics["argument_match_rate"]
        >= float(targets["target_argument_match_rate"])
        and sft_metrics["ordinary_text_rate"]
        <= float(targets["target_ordinary_text_rate_max"])
    )
    manifest = {
        "schema_version": "retail-tool-protocol-sft-run-v1",
        "scope": config["scope"],
        "status": "COMPLETED",
        "git": {
            "commit": preflight["git_commit"],
            "branch": preflight["git_branch"],
            "dirty_at_start": preflight["git_dirty_at_start"],
        },
        "bindings": {
            "config_path": preflight["config_path"],
            "config_sha256": preflight["config_sha256"],
            "data_manifest_path": preflight["data_manifest_path"],
            "data_manifest_sha256": preflight["data_manifest_sha256"],
            "starting_model": preflight["model_path"],
            "starting_model_sha256": preflight["model_sha256"],
            "upstream": preflight["upstream"],
        },
        "data": {
            "train_rows": len(train_rows),
            "holdout_rows": len(holdout_rows),
            "max_rendered_prompt_tokens": max(prompt_lengths),
            "max_rendered_sequence_tokens": max(sequence_lengths),
        },
        "evaluations": {"base": base_metrics, "sft": sft_metrics},
        "protocol_warmup_gate": {
            "passed": protocol_warmup_gate_passed,
            "targets": targets,
        },
        "artifacts": {
            "adapter": {"path": str(adapter_dir), "sha256": directory_sha256(adapter_dir)},
            "merged_model": {"path": str(merged_dir), "sha256": directory_sha256(merged_dir)},
        },
        "formal_retail_readiness_gate_opened": False,
        "business_improvement_claim_allowed": False,
        "agentic_grpo_allowed": False,
    }
    save_json(output_dir / "run_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "retail_tool_sft_qwen3_4b_v1.json",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    preflight = validate_inputs(args.config.resolve(), args.allow_dirty)
    if args.preflight_only:
        runtime = load_runtime(preflight)
        tools = build_tools()
        rendered = render_rows(
            load_jsonl(preflight["data_dir"] / "holdout.jsonl")[:1],
            runtime["tokenizer"],
            tools,
        )[0]
        print(
            json.dumps(
                {
                    "status": "PREFLIGHT_PASSED",
                    "config_sha256": preflight["config_sha256"],
                    "data_manifest_sha256": preflight["data_manifest_sha256"],
                    "model_sha256": preflight["model_sha256"],
                    "tool_count": len(tools),
                    "rendered_prompt_tokens": len(
                        runtime["tokenizer"](
                            rendered["prompt"], add_special_tokens=False
                        )["input_ids"]
                    ),
                    "runtime": runtime["versions"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.output_dir is None:
        parser.error("--output-dir is required")
    try:
        print(json.dumps(run(preflight, args.output_dir.resolve()), ensure_ascii=False, indent=2))
    except Exception as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        save_json(
            args.output_dir / "failure_manifest.json",
            {
                "schema_version": "retail-tool-protocol-sft-failure-v1",
                "status": "FAILED",
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(),
                "config_sha256": preflight["config_sha256"],
            },
        )
        raise


if __name__ == "__main__":
    main()
