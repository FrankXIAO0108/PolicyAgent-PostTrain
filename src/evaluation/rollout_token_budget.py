"""Count complete protocol-bridge TRAIN transcripts without model/API execution.

This is a capacity reference, NOT a capability evaluation of the RL checkpoint.
The exact checkpoint directory and frozen dataset hashes are required by the CLI.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys


def count_trajectory(chat: list[dict], render, eos_id: int) -> dict:
    if len(chat) < 3 or [x["role"] for x in chat[:2]] != ["system", "user"]:
        raise ValueError("Expected a system policy and frozen user opening")
    prefix = list(render(chat[:2], True))
    prompt_tokens = len(prefix)
    model_tokens = observation_tokens = 0
    max_observation = 0
    turns = []
    for index, message in enumerate(chat[2:], 2):
        role = message["role"]
        if role not in {"assistant", "tool"}:
            raise ValueError("Only protocol-bridged assistant/tool turns are accepted")
        if role == "assistant" and index > 2 and chat[index - 1]["role"] == "assistant":
            raise ValueError(
                "Consecutive assistant turns: missing tool result or invalid continuation"
            )
        if role == "tool" and (
            chat[index - 1]["role"] != "assistant"
            or not chat[index - 1].get("tool_calls")
        ):
            raise ValueError("Tool result has no preceding atomic call")
        if (
            role == "assistant"
            and message.get("tool_calls")
            and len(message["tool_calls"]) != 1
        ):
            raise ValueError("Capacity reference requires atomic tool calls")
        ids = list(render(chat[: index + 1], role == "tool"))
        if role == "assistant":
            positions = [i for i, token in enumerate(ids) if token == eos_id]
            if not positions or positions[-1] < len(prefix):
                raise ValueError("Assistant turn has no new EOS")
            ids = ids[: positions[-1] + 1]  # same EOS boundary as the TRL tool loop
        if ids[: len(prefix)] != prefix:
            raise ValueError(
                "Template is not prefix-preserving; refusing approximate token counts"
            )
        delta = len(ids) - len(prefix)
        if role == "assistant":
            model_tokens += delta
        else:
            observation_tokens += delta
            max_observation = max(max_observation, delta)
        turns.append(
            {"message_index": index, "role": role, "tokens_with_template": delta}
        )
        prefix = ids
    if chat[-1]["role"] != "assistant" or chat[-1].get("tool_calls"):
        raise ValueError("Capacity reference must contain the final model response")
    if prompt_tokens + model_tokens + observation_tokens != len(prefix):
        raise AssertionError("Token accounting failed")
    return {
        "prompt_tokens": prompt_tokens,
        "model_tokens": model_tokens,
        "observation_tokens_with_template": observation_tokens,
        "completion_tokens": model_tokens + observation_tokens,
        "total_tokens": len(prefix),
        "max_single_observation_tokens": max_observation,
        "turns": turns,
    }


def summarize_lengths(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("No eligible complete TRAIN trajectories")
    fields = [
        "prompt_tokens",
        "model_tokens",
        "observation_tokens_with_template",
        "completion_tokens",
        "total_tokens",
        "max_single_observation_tokens",
    ]
    stats = {}
    for field in fields:
        values = sorted(row[field] for row in rows)
        stats[field] = {
            "min": values[0],
            "max": values[-1],
            "mean": sum(values) / len(values),
            "p95_nearest_rank": values[math.ceil(0.95 * len(values)) - 1],
        }
    # Engineering reference only: cover all observed training trajectories plus
    # 25% headroom. No claim that future trajectories fit or that GPU can train it.
    reference = math.ceil(stats["completion_tokens"]["max"] * 1.25 / 1024) * 1024
    return {
        "rows": len(rows),
        "statistics": stats,
        "reference_budget_with_25pct_headroom": reference,
        "candidate_budgets": {
            str(b): {
                "reference_rows_exceeding": sum(
                    r["completion_tokens"] > b for r in rows
                )
            }
            for b in (1024, 4096, 8192, 16384)
        },
        "gpu_feasibility_verified": False,
        "sft_capability_assessed": False,
        "budget_approved": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    from src.training.run_retail_agentic_grpo import (
        REPO_ROOT,
        load_json,
        load_jsonl,
        sha256,
        save_json,
    )
    from src.training.dirhash import directory_sha256
    from src.training.rollout_diagnostics import verify_trl_source
    from src.training.run_teacher_sft import build_chat
    from src.rl.retail_agentic_env import RetailAgenticEnvironment
    from transformers import AutoTokenizer
    from trl import GRPOTrainer
    from trl.chat_template_utils import (
        add_response_schema,
        get_training_chat_template,
        is_chat_template_prefix_preserving,
    )

    if args.output_dir.exists():
        raise FileExistsError("Use a new output directory")
    runtime = verify_trl_source(GRPOTrainer)
    config = load_json(args.config)
    model = Path(config["model"]["name_or_path"])
    if (
        not model.is_dir()
        or directory_sha256(model) != config["model"]["expected_sha256"].upper()
    ):
        raise ValueError("Exact checkpoint directory/hash mismatch")
    manifest_path = args.data_dir / "manifest.json"
    manifest = load_json(manifest_path)
    # Bind the source manifest of the bridge dataset, not an arbitrary lookalike.
    binding_path = REPO_ROOT / config["model"]["training_data_manifest_path"]
    if sha256(binding_path) != config["model"]["training_data_manifest_sha256"].upper():
        raise ValueError("Training-data lineage binding mismatch")
    binding = load_json(binding_path)
    dataset_path = args.data_dir / "sft_dataset.jsonl"
    expected = manifest["files"]["sft_dataset"]["sha256"].upper()
    if sha256(dataset_path) != expected:
        raise ValueError("Dataset hash mismatch")
    if sha256(manifest_path) != sha256(binding_path):
        raise ValueError("Capacity manifest differs from the checkpoint lineage")
    if expected != binding["files"]["sft_dataset"]["sha256"].upper():
        raise ValueError("Dataset digest differs from the checkpoint lineage")
    dataset = load_jsonl(dataset_path)
    if len(dataset) != manifest["files"]["sft_dataset"]["rows"]:
        raise ValueError("Dataset row count mismatch")
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    if not getattr(tokenizer, "response_template", None) and not getattr(
        tokenizer, "response_schema", None
    ):
        tokenizer = add_response_schema(tokenizer)
    template = (
        None
        if is_chat_template_prefix_preserving(tokenizer)
        else get_training_chat_template(tokenizer)
    )
    # Inspect bound methods without constructing/resetting a tau2 environment.
    env = object.__new__(RetailAgenticEnvironment)
    tools = [
        fn
        for name, fn in inspect.getmembers(env, predicate=inspect.ismethod)
        if not name.startswith("_") and name not in {"reset", "get_reward"}
    ]

    def render(chat, generation_prompt):
        return tokenizer.apply_chat_template(
            chat,
            tools=tools,
            tokenize=True,
            return_dict=False,
            add_generation_prompt=generation_prompt,
            chat_template=template,
        )

    rows = []
    for row in dataset:
        if row["split"] != "TRAIN":
            continue  # validation/test content must not set the training budget
        rows.append(
            {
                "candidate_id": row["candidate_id"],
                "task_id": str(row["task_id"]),
                **count_trajectory(build_chat(row), render, tokenizer.eos_token_id),
            }
        )
    report = {
        "scope": "TRAIN_CAPACITY_REFERENCE_ONLY",
        "runtime": runtime,
        "config_sha256": sha256(args.config),
        "checkpoint_sha256": config["model"]["expected_sha256"],
        "dataset_sha256": expected,
        "manifest_sha256": sha256(manifest_path),
        "template_sha256": hashlib.sha256(
            str(template or tokenizer.chat_template).encode()
        )
        .hexdigest()
        .upper(),
        "summary": summarize_lengths(rows),
        "trajectories": rows,
        "external_api_called": False,
        "model_loaded": False,
        "limitations": [
            "Teacher trajectories are not capability evidence for the current checkpoint.",
            "No budget or GPU feasibility is approved by this reference report.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    save_json(args.output_dir / "command.json", {"argv": sys.argv})
    save_json(args.output_dir / "report.json", report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
