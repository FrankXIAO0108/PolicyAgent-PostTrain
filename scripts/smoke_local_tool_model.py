from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.training.run_retail_agentic_grpo import (
        check_runtime,
        check_tool_template,
        directory_sha256,
        sha256,
    )

    model_path = args.model_path.expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    runtime = check_runtime()
    runtime["tool_template"] = check_tool_template(str(model_path))
    manifest_path = model_path / "MODEL_MANIFEST.json"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_order_details",
                "description": "Read the current state of one customer order.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "The exact order identifier.",
                        }
                    },
                    "required": ["order_id"],
                },
            },
        }
    ]
    messages = [
        {
            "role": "system",
            "content": (
                "You are a tool-using customer-service agent. Use exactly one "
                "tool call when a lookup is requested."
            ),
        },
        {
            "role": "user",
            "content": (
                "You must call get_order_details for order W2378156. "
                "Do not answer with ordinary text."
            ),
        },
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(rendered, return_tensors="pt").to("cuda")
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to("cuda")
    model.eval()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    completion = tokenizer.decode(
        output_ids[0, inputs["input_ids"].shape[1] :],
        skip_special_tokens=False,
    ).strip()
    payload = {
        "schema_version": "policyagent-tool-model-smoke-v1",
        "status": "PASSED" if completion else "FAILED",
        "model_path": str(model_path),
        "model_manifest_sha256": (
            sha256(manifest_path) if manifest_path.is_file() else None
        ),
        "model_directory_sha256": directory_sha256(model_path),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "runtime": runtime,
        "prompt_tokens": int(inputs["input_ids"].shape[1]),
        "completion_tokens": int(output_ids.shape[1] - inputs["input_ids"].shape[1]),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "tool_call_marker_present": "<tool_call>" in completion,
        "requested_tool_name_present": "get_order_details" in completion,
        "requested_order_id_present": "W2378156" in completion,
        "completion": completion,
    }
    if payload["status"] != "PASSED":
        raise RuntimeError("Model produced an empty tool-call smoke completion")
    if args.output:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
