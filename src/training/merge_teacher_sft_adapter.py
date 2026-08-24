"""Merge one selected Teacher-SFT LoRA adapter without retraining.

The merge is intentionally separate from checkpoint selection.  It consumes a
completed, hash-bound ``run_teacher_sft`` artifact and writes one full model
plus a merge manifest.  It does not run evaluation or open any Retail claim
gate.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.training.run_retail_agentic_grpo import (
    REPO_ROOT,
    directory_sha256,
    load_json,
    save_json,
    sha256,
)


def git_value(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate_source(
    source_run_dir: Path, expected_manifest_sha256: str
) -> dict[str, Any]:
    source_run_dir = source_run_dir.expanduser().resolve()
    manifest_path = source_run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    actual_manifest_sha256 = sha256(manifest_path)
    if actual_manifest_sha256 != expected_manifest_sha256.upper():
        raise ValueError("Teacher-SFT source manifest hash mismatch")

    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != "retail-teacher-sft-run-v1":
        raise ValueError("Teacher-SFT source manifest schema mismatch")
    if manifest.get("scope") != "TEACHER_TRAJECTORY_SFT":
        raise ValueError("Teacher-SFT source scope mismatch")
    if manifest.get("status") != "COMPLETED":
        raise ValueError("Teacher-SFT source run is not completed")
    if manifest.get("git", {}).get("dirty_at_start") is not False:
        raise ValueError("Teacher-SFT source run did not start from a clean tree")
    if manifest.get("teacher_sft_gate", {}).get("passed") is not True:
        raise ValueError("Teacher-SFT source loss gate did not pass")
    if manifest.get("business_improvement_claim_allowed") is not False:
        raise ValueError("Teacher-SFT source carries an unsupported business claim")

    adapter_dir = source_run_dir / "teacher_sft_adapter"
    if not adapter_dir.is_dir():
        raise FileNotFoundError(adapter_dir)
    expected_adapter_sha256 = manifest.get("artifacts", {}).get("adapter", {}).get(
        "sha256"
    )
    if not expected_adapter_sha256:
        raise ValueError("Teacher-SFT source adapter hash is missing")
    actual_adapter_sha256 = directory_sha256(adapter_dir)
    if actual_adapter_sha256 != str(expected_adapter_sha256).upper():
        raise ValueError("Teacher-SFT source adapter hash mismatch")

    base_dir = Path(manifest.get("bindings", {}).get("starting_model", ""))
    base_dir = base_dir.expanduser().resolve()
    if not base_dir.is_dir():
        raise FileNotFoundError(base_dir)
    expected_base_sha256 = manifest.get("bindings", {}).get(
        "starting_model_sha256"
    )
    if not expected_base_sha256:
        raise ValueError("Teacher-SFT source base-model hash is missing")
    actual_base_sha256 = directory_sha256(base_dir)
    if actual_base_sha256 != str(expected_base_sha256).upper():
        raise ValueError("Teacher-SFT source base-model hash mismatch")

    return {
        "source_run_dir": source_run_dir,
        "source_manifest_path": manifest_path,
        "source_manifest_sha256": actual_manifest_sha256,
        "source_manifest": manifest,
        "adapter_dir": adapter_dir,
        "adapter_sha256": actual_adapter_sha256,
        "base_dir": base_dir,
        "base_sha256": actual_base_sha256,
    }


def run(
    preflight: dict[str, Any], output_dir: Path, *, git_dirty_at_start: bool
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    import peft
    import torch
    import transformers
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(
        str(preflight["base_dir"]),
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(preflight["base_dir"]))
    merged = PeftModel.from_pretrained(
        base, str(preflight["adapter_dir"])
    ).merge_and_unload()
    merged_dir = output_dir / "merged_model"
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    merged_sha256 = directory_sha256(merged_dir)

    source_manifest = preflight["source_manifest"]
    manifest = {
        "schema_version": "teacher-sft-selected-merge-v1",
        "scope": "TEACHER_TRAJECTORY_SFT_SELECTED_MERGE",
        "status": "COMPLETED",
        "git": {
            "commit": git_value("rev-parse", "HEAD"),
            "branch": git_value("branch", "--show-current"),
            "dirty_at_start": git_dirty_at_start,
        },
        "source": {
            "run_dir": str(preflight["source_run_dir"]),
            "run_manifest_path": str(preflight["source_manifest_path"]),
            "run_manifest_sha256": preflight["source_manifest_sha256"],
            "run_git_commit": source_manifest["git"]["commit"],
            "config_sha256": source_manifest["bindings"]["config_sha256"],
            "data_manifest_sha256": source_manifest["bindings"][
                "data_manifest_sha256"
            ],
            "base_model": {
                "path": str(preflight["base_dir"]),
                "sha256": preflight["base_sha256"],
            },
            "adapter": {
                "path": str(preflight["adapter_dir"]),
                "sha256": preflight["adapter_sha256"],
            },
        },
        "output": {
            "merged_model": {
                "path": str(merged_dir),
                "sha256": merged_sha256,
            }
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "command": sys.argv,
        "formal_retail_readiness_gate_opened": False,
        "business_improvement_claim_allowed": False,
        "notes": [
            "Checkpoint selection used development validation only.",
            "Merge creates a deployment/evaluation artifact; it is not new training evidence.",
        ],
    }
    save_json(output_dir / "merge_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge one hash-bound Teacher-SFT adapter without retraining."
    )
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    git_dirty_at_start = bool(git_value("status", "--porcelain"))
    if git_dirty_at_start and not args.allow_dirty:
        raise RuntimeError("Commit Teacher-SFT merge inputs before running")
    preflight = validate_source(
        args.source_run_dir, args.expected_source_manifest_sha256
    )
    print(
        json.dumps(
            {
                "status": "VALIDATED",
                "source_manifest_sha256": preflight["source_manifest_sha256"],
                "adapter_sha256": preflight["adapter_sha256"],
                "base_sha256": preflight["base_sha256"],
            },
            ensure_ascii=False,
        )
    )
    if args.validate_only:
        return
    if args.output_dir is None:
        parser.error("--output-dir is required unless --validate-only")
    run(preflight, args.output_dir, git_dirty_at_start=git_dirty_at_start)


if __name__ == "__main__":
    main()
