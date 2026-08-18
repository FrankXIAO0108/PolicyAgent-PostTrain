"""Teacher trajectory multi-turn SFT runner.

Trains Qwen3-4B-Instruct (LoRA + 4-bit NF4) on the released teacher SFT
dataset (``teacher_sft_release`` output). Every assistant turn in the frozen
trajectories is a supervised target; user/tool/system turns are masked.

The dataset is the ``SECOND_REVIEWED`` teacher pool (see
docs/04_数据治理与后训练/2026-08-18_教师轨迹合并SFT数据划分与门禁.md), not an
``ADJUDICATED`` policy-gold pool. This runner therefore:
- fails closed on hash/row binding of the data manifest;
- reports Base vs SFT validation loss on the entity-disjoint VALIDATION split;
- does not claim any formal Retail gate or business improvement.

Scope is intentionally separate from the isolated tool-protocol warmup runner.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from src.training.run_retail_agentic_grpo import (
    REPO_ROOT,
    directory_sha256,
    load_json,
    load_jsonl,
    save_json,
    sha256,
    validate_upstream_checkout,
)
from src.training.run_retail_tool_sft import TOOL_NAMES, build_tools

SCOPE = "TEACHER_TRAJECTORY_SFT"
SUPPORTED_SCOPES = {SCOPE}


def git_value(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def to_chat_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize released teacher messages to OpenAI-style chat messages.

    The released format uses positional tool results (no ``tool_call_id`` on
    the ``tool`` role), so the id of the preceding assistant tool call is
    attached for chat-template compatibility. Tool-call ``arguments`` are
    serialized to the JSON string form expected by the model template.
    """
    chat: list[dict[str, Any]] = []
    pending_call_id: str | None = None
    for message in messages:
        role = str(message.get("role", ""))
        calls = message.get("tool_calls") or []
        if role == "assistant" and calls:
            if len(calls) != 1:
                raise ValueError("teacher assistant turn must carry exactly one tool call")
            call = calls[0]
            arguments = call.get("arguments")
            if not isinstance(arguments, dict):
                raise ValueError("teacher tool call arguments must be an object")
            pending_call_id = str(call["id"])
            chat.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": pending_call_id,
                            "type": "function",
                            "function": {
                                "name": str(call["name"]),
                                "arguments": json.dumps(
                                    arguments, ensure_ascii=False, separators=(",", ":")
                                ),
                            },
                        }
                    ],
                }
            )
        elif role == "assistant":
            chat.append({"role": "assistant", "content": message.get("content") or ""})
        elif role == "user":
            chat.append({"role": "user", "content": message.get("content") or ""})
        elif role == "tool":
            chat.append(
                {
                    "role": "tool",
                    "content": message.get("content") or "",
                    "tool_call_id": pending_call_id,
                }
            )
            pending_call_id = None
        elif role == "system":
            chat.append({"role": "system", "content": message.get("content") or ""})
        else:
            raise ValueError(f"unsupported message role {role!r}")
    if not any(message.get("role") == "assistant" for message in chat):
        raise ValueError("no assistant turns found in trajectory")
    return chat


def build_chat(row: dict[str, Any]) -> list[dict[str, Any]]:
    system_policy = str(row.get("system_policy", "")).strip()
    if not system_policy:
        raise ValueError("row is missing system_policy")
    return [{"role": "system", "content": system_policy}] + to_chat_messages(
        row["messages"]
    )


def tokenize_row(
    tokenizer: Any,
    chat: list[dict[str, Any]],
    tools: list[Any],
    max_length: int,
) -> dict[str, list[int]]:
    try:
        encoded = tokenizer.apply_chat_template(
            chat,
            tools=tools,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
            add_generation_prompt=False,
            truncation=False,
        )
    except TypeError as error:
        raise RuntimeError(
            "tokenizer.apply_chat_template does not support "
            "return_assistant_tokens_mask; upgrade transformers"
        ) from error
    input_ids = [int(value) for value in encoded["input_ids"]]
    assistant_mask = [bool(value) for value in encoded["assistant_tokens_mask"]]
    if not any(assistant_mask):
        raise ValueError("tokenized sequence has no assistant tokens")
    if len(input_ids) > max_length:
        raise ValueError(
            f"trajectory exceeds max_length ({len(input_ids)} > {max_length}); "
            "raise max_length or split the row"
        )
    labels = [token if keep else -100 for token, keep in zip(input_ids, assistant_mask)]
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def _assistant_nll(
    logits: Any, labels: Any, torch: Any
) -> tuple[float, int]:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    valid = shift_labels != -100
    if not bool(valid.any()):
        return 0.0, 0
    nll = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1))[valid.view(-1)],
        shift_labels.view(-1)[valid.view(-1)],
        reduction="sum",
    ).item()
    return float(nll), int(valid.sum().item())


def evaluate_validation(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    tools: list[Any],
    max_length: int,
    torch: Any,
    device: Any,
) -> dict[str, Any]:
    total_nll = 0.0
    total_tokens = 0
    per_row: list[dict[str, Any]] = []
    for row in rows:
        batch = tokenize_row(
            tokenizer, build_chat(row), tools, int(max_length)
        )
        inputs = {
            key: torch.tensor(value).unsqueeze(0).to(device)
            for key, value in batch.items()
        }
        with torch.inference_mode():
            output = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"],
            )
        loss = float(output.loss.item())
        nll, count = _assistant_nll(output.logits, inputs["labels"], torch)
        per_row.append(
            {
                "candidate_id": row["candidate_id"],
                "task_id": row["task_id"],
                "loss": loss,
                "assistant_tokens": count,
                "assistant_nll": nll,
            }
        )
        total_nll += nll
        total_tokens += count
    mean_loss = total_nll / total_tokens if total_tokens else float("inf")
    return {
        "rows": len(rows),
        "assistant_tokens": total_tokens,
        "mean_assistant_loss": mean_loss,
        "perplexity": math.exp(mean_loss) if math.isfinite(mean_loss) else None,
        "per_row": per_row,
    }


def validate_inputs(config_path: Path, allow_dirty: bool) -> dict[str, Any]:
    config = load_json(config_path)
    if config.get("scope") not in SUPPORTED_SCOPES:
        raise ValueError("Teacher SFT scope mismatch")
    claims = config.get("claims", {})
    if claims.get("formal_retail_gate_unchanged") is not True:
        raise ValueError("Teacher SFT must leave the formal Retail gate unchanged")
    if claims.get("business_improvement_claim_allowed") is not False:
        raise ValueError("Teacher SFT config cannot carry a business claim")
    quantization = config.get("quantization", {})
    if quantization != {
        "enabled": True,
        "mode": "4bit_nf4",
        "double_quant": True,
    }:
        raise ValueError("Teacher SFT runner only supports the frozen 4-bit NF4 setup")
    data_dir = (REPO_ROOT / config["data_dir"]).resolve()
    manifest_path = data_dir / "manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("scope") != config["scope"]:
        raise ValueError("Teacher SFT data scope mismatch")
    if manifest["claims"].get("business_improvement_claim_allowed") is not False:
        raise ValueError("Teacher SFT data cannot carry a business claim")
    checks = manifest.get("leakage_and_quality_checks") or {}
    if checks.get("passed") is not True:
        raise ValueError("Teacher SFT data quality/leakage checks failed")
    for key, record in manifest["files"].items():
        path = data_dir / record["path"]
        if sha256(path) != record["sha256"] or len(load_jsonl(path)) != int(
            record["rows"]
        ):
            raise ValueError(f"Teacher SFT {key} binding mismatch")
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
        raise ValueError("Teacher SFT starting model hash mismatch")
    dirty = bool(git_value("status", "--porcelain"))
    if dirty and not allow_dirty:
        raise RuntimeError("Commit Teacher SFT inputs before running")
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
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        set_seed,
    )
    import trl
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Teacher SFT")
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
        "DataCollatorForSeq2Seq": DataCollatorForSeq2Seq,
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


def run(preflight: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime = load_runtime(preflight)
    torch = runtime["torch"]
    tokenizer = runtime["tokenizer"]
    tools = build_tools()
    config = preflight["config"]
    max_length = int(config["max_length"])

    rows = load_jsonl(preflight["data_dir"] / "sft_dataset.jsonl")
    train_rows = [row for row in rows if row.get("split") == "TRAIN"]
    val_rows = [row for row in rows if row.get("split") == "VALIDATION"]
    if not train_rows or not val_rows:
        raise ValueError("Teacher SFT requires both TRAIN and VALIDATION rows")
    train_batches = [
        tokenize_row(tokenizer, build_chat(row), tools, max_length)
        for row in train_rows
    ]
    val_batches = [
        tokenize_row(tokenizer, build_chat(row), tools, max_length)
        for row in val_rows
    ]
    max_sequence = max(
        len(batch["input_ids"]) for batch in [*train_batches, *val_batches]
    )
    save_json(output_dir / "environment.json", runtime["versions"])

    device = torch.device("cuda")
    base_model = runtime["AutoModelForCausalLM"].from_pretrained(
        preflight["model_path"],
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    base_model.eval()
    base_metrics = evaluate_validation(
        base_model, tokenizer, val_rows, tools, max_length, torch, device
    )
    save_json(output_dir / "evaluation_base.json", base_metrics)
    del base_model
    torch.cuda.empty_cache()

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
        max_length=max_length,
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
        train_dataset=runtime["Dataset"].from_list(train_batches),
        data_collator=runtime["DataCollatorForSeq2Seq"](
            tokenizer, padding=True, label_pad_token_id=-100
        ),
        processing_class=tokenizer,
        peft_config=lora,
    )
    started = time.time()
    result = trainer.train()
    adapter_dir = output_dir / "teacher_sft_adapter"
    trainer.save_model(adapter_dir)
    save_json(output_dir / "train_metrics.json", result.metrics)
    save_json(output_dir / "log_history.json", trainer.state.log_history)
    del trainer, model
    torch.cuda.empty_cache()

    base = runtime["AutoModelForCausalLM"].from_pretrained(
        preflight["model_path"], dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    merged = runtime["PeftModel"].from_pretrained(base, str(adapter_dir)).merge_and_unload()
    merged_dir = output_dir / "teacher_sft_merged"
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    del base, merged
    torch.cuda.empty_cache()

    sft_model = runtime["AutoModelForCausalLM"].from_pretrained(
        str(merged_dir), dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(device)
    sft_model.eval()
    sft_metrics = evaluate_validation(
        sft_model, tokenizer, val_rows, tools, max_length, torch, device
    )
    save_json(output_dir / "evaluation_sft.json", sft_metrics)
    del sft_model
    torch.cuda.empty_cache()

    base_loss = base_metrics["mean_assistant_loss"]
    sft_loss = sft_metrics["mean_assistant_loss"]
    gate_passed = (
        math.isfinite(base_loss)
        and math.isfinite(sft_loss)
        and sft_loss < base_loss
    )
    manifest = {
        "schema_version": "retail-teacher-sft-run-v1",
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
            "rows": len(rows),
            "train_rows": len(train_rows),
            "validation_rows": len(val_rows),
            "max_sequence_tokens": max_sequence,
        },
        "training": {
            "elapsed_seconds": round(time.time() - started, 2),
            "train_metrics": result.metrics,
        },
        "evaluations": {"base": base_metrics, "sft": sft_metrics},
        "teacher_sft_gate": {
            "passed": gate_passed,
            "base_mean_assistant_loss": base_loss,
            "sft_mean_assistant_loss": sft_loss,
        },
        "artifacts": {
            "adapter": {
                "path": str(adapter_dir),
                "sha256": directory_sha256(adapter_dir),
            },
            "merged_model": {
                "path": str(merged_dir),
                "sha256": directory_sha256(merged_dir),
            },
        },
        "formal_retail_readiness_gate_opened": False,
        "business_improvement_claim_allowed": False,
        "notes": [
            "SECOND_REVIEWED teacher pool; not ADJUDICATED policy gold.",
            "Validation split is entity-disjoint from training (merged pool plan v1).",
        ],
    }
    save_json(output_dir / "run_manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "train_rows": len(train_rows),
                "validation_rows": len(val_rows),
                "base_mean_assistant_loss": base_loss,
                "sft_mean_assistant_loss": sft_loss,
                "gate_passed": gate_passed,
            },
            ensure_ascii=False,
        )
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Teacher trajectory multi-turn SFT runner."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    preflight = validate_inputs(args.config, args.allow_dirty)
    print(json.dumps({"status": "VALIDATED", "config_sha256": preflight["config_sha256"]}))
    if args.validate_only:
        return
    if args.output_dir is None:
        parser.error("--output-dir is required unless --validate-only")
    run(preflight, args.output_dir)


if __name__ == "__main__":
    main()
