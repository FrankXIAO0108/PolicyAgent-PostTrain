"""Prepare a frozen, nested Teacher-SFT scaling plan without training.

The planner separates two questions that are otherwise easily confounded:

* optimization scaling: full data at multiple optimizer-step budgets;
* data scaling: nested train-task subsets with approximately equal epochs.

Validation rows are copied unchanged into every data-scale variant. The tool
only prepares hashed private datasets/configs and commands; it never launches
GPU training or tau2 evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from src.training.run_retail_agentic_grpo import REPO_ROOT, load_json, load_jsonl, sha256


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _stable_task_order(task_ids: set[str], seed: int) -> list[str]:
    def key(task_id: str) -> tuple[str, str]:
        digest = hashlib.sha256(f"{seed}:{task_id}".encode("utf-8")).hexdigest()
        return digest, task_id

    return sorted(task_ids, key=key)


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("plateau output must remain inside the repository") from exc


def _validate_source(
    source_config_path: Path, source_data_dir: Path
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    config = load_json(source_config_path)
    manifest_path = source_data_dir / "manifest.json"
    manifest = load_json(manifest_path)
    if config.get("scope") != "TEACHER_TRAJECTORY_SFT":
        raise ValueError("source config is not a Teacher SFT config")
    if (REPO_ROOT / str(config.get("data_dir", ""))).resolve() != source_data_dir.resolve():
        raise ValueError("source config and source_data_dir do not match")
    if manifest.get("scope") != config["scope"]:
        raise ValueError("source data scope mismatch")
    if manifest.get("claims", {}).get("business_improvement_claim_allowed") is not False:
        raise ValueError("source data must forbid business-improvement claims")
    if manifest.get("leakage_and_quality_checks", {}).get("passed") is not True:
        raise ValueError("source data quality/leakage gate is not passed")

    bound: dict[str, list[dict[str, Any]]] = {}
    for name in ("sft_dataset", "split_plan"):
        record = manifest.get("files", {}).get(name)
        if not isinstance(record, dict):
            raise ValueError(f"source manifest is missing {name}")
        path = source_data_dir / str(record["path"])
        rows = load_jsonl(path)
        if sha256(path) != str(record["sha256"]) or len(rows) != int(record["rows"]):
            raise ValueError(f"source {name} binding mismatch")
        bound[name] = rows

    dataset = bound["sft_dataset"]
    split_plan = bound["split_plan"]
    train = [row for row in dataset if row.get("split") == "TRAIN"]
    validation = [row for row in dataset if row.get("split") == "VALIDATION"]
    if not train or not validation or len(train) + len(validation) != len(dataset):
        raise ValueError("source dataset must contain only non-empty TRAIN/VALIDATION")
    return config, manifest, dataset, split_plan


def _variant_config(
    source: dict[str, Any], data_dir: Path, seed: int, max_steps: int
) -> dict[str, Any]:
    config = json.loads(json.dumps(source))
    config["data_dir"] = _repo_relative(data_dir)
    config["seed"] = seed
    config["sft"]["max_steps"] = max_steps
    config["plateau_protocol"] = {
        "scope": "TEACHER_TRAJECTORY_SFT_PLATEAU",
        "development_only": True,
    }
    return config


def _write_data_variant(
    *,
    variant_dir: Path,
    source_manifest: dict[str, Any],
    dataset: list[dict[str, Any]],
    split_plan: list[dict[str, Any]],
    selected_train_tasks: set[str],
    fraction: float,
) -> tuple[Path, int, int]:
    validation_tasks = {
        str(row["task_id"]) for row in dataset if row.get("split") == "VALIDATION"
    }
    included_tasks = selected_train_tasks | validation_tasks
    selected_rows = [
        row
        for row in dataset
        if row.get("split") == "VALIDATION"
        or str(row.get("task_id")) in selected_train_tasks
    ]
    selected_plan = [
        row for row in split_plan if str(row.get("task_id")) in included_tasks
    ]
    dataset_path = variant_dir / "sft_dataset.jsonl"
    split_path = variant_dir / "split_plan.jsonl"
    _save_jsonl(dataset_path, selected_rows)
    _save_jsonl(split_path, selected_plan)
    train_rows = sum(row.get("split") == "TRAIN" for row in selected_rows)
    validation_rows = sum(row.get("split") == "VALIDATION" for row in selected_rows)
    manifest = {
        "schema_version": "tau2-teacher-sft-plateau-data-manifest-v1",
        "scope": "TEACHER_TRAJECTORY_SFT",
        "claims": dict(source_manifest["claims"]),
        "files": {
            "sft_dataset": {
                "path": "sft_dataset.jsonl",
                "sha256": sha256(dataset_path),
                "rows": len(selected_rows),
            },
            "split_plan": {
                "path": "split_plan.jsonl",
                "sha256": sha256(split_path),
                "rows": len(selected_plan),
            },
        },
        "counts": {"TRAIN": train_rows, "VALIDATION": validation_rows},
        "leakage_and_quality_checks": {
            "passed": True,
            "entity_leakage_across_splits": 0,
            "pii_hits": 0,
            "derived_by_nested_train_task_subsetting": True,
            "source_release_gate_ready": True,
        },
        "plateau_subset": {
            "train_task_fraction": fraction,
            "selected_train_task_ids": sorted(selected_train_tasks),
            "validation_rows_unchanged": True,
        },
    }
    manifest_path = variant_dir / "manifest.json"
    _save_json(manifest_path, manifest)
    return manifest_path, train_rows, validation_rows


def prepare(protocol_path: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = load_json(protocol_path)
    if protocol.get("scope") != "TEACHER_TRAJECTORY_SFT_PLATEAU":
        raise ValueError("plateau protocol scope mismatch")
    if protocol.get("claims", {}).get("sft_plateau_claim_allowed_before_stage_2") is not False:
        raise ValueError("protocol must forbid premature plateau claims")

    source_config_path = (REPO_ROOT / protocol["source_config"]).resolve()
    source_data_dir = (REPO_ROOT / protocol["source_data_dir"]).resolve()
    source_config, source_manifest, dataset, split_plan = _validate_source(
        source_config_path, source_data_dir
    )
    existing_run_dir = (REPO_ROOT / protocol["existing_full_run"]).resolve()
    existing_run_manifest_path = existing_run_dir / "run_manifest.json"
    existing_run = load_json(existing_run_manifest_path)
    if existing_run.get("status") != "COMPLETED":
        raise ValueError("existing full Teacher SFT run is not completed")
    if (
        existing_run.get("bindings", {}).get("data_manifest_sha256")
        != sha256(source_data_dir / "manifest.json")
    ):
        raise ValueError("existing full run data binding mismatch")
    train_rows = [row for row in dataset if row["split"] == "TRAIN"]
    validation_rows = [row for row in dataset if row["split"] == "VALIDATION"]
    train_tasks = {str(row["task_id"]) for row in train_rows}
    order = _stable_task_order(train_tasks, int(protocol["selection_seed"]))
    full_steps = int(protocol["full_reference_steps"])
    base_seed = int(source_config["seed"])
    variants: list[dict[str, Any]] = []

    # Same data, different optimization budgets.
    for steps in protocol["optimization_steps"]:
        steps = int(steps)
        variant_id = f"opt_full_s{steps}"
        reuse_existing = steps == full_steps
        if reuse_existing:
            config_path = source_config_path
        else:
            config = _variant_config(source_config, source_data_dir, base_seed, steps)
            config_path = output_dir / "configs" / f"{variant_id}.json"
            _save_json(config_path, config)
        variant = {
                "variant_id": variant_id,
                "axis": "OPTIMIZATION_STEPS",
                "train_task_fraction": 1.0,
                "train_tasks": len(train_tasks),
                "train_rows": len(train_rows),
                "validation_rows": len(validation_rows),
                "max_steps": steps,
                "seed": base_seed,
                "config_path": _repo_relative(config_path),
                "config_sha256": sha256(config_path),
                "reuse_existing_full_run": reuse_existing,
            }
        if reuse_existing:
            variant["existing_run_manifest_path"] = _repo_relative(
                existing_run_manifest_path
            )
            variant["existing_run_manifest_sha256"] = sha256(
                existing_run_manifest_path
            )
            variant["existing_run_bound_config_sha256"] = existing_run[
                "bindings"
            ]["config_sha256"]
            variant["existing_run_bound_data_manifest_sha256"] = existing_run[
                "bindings"
            ]["data_manifest_sha256"]
        variants.append(variant)

    # Nested data subsets, with step counts proportional to selected rows so
    # that each candidate sees approximately the same number of epochs.
    for fraction_value in protocol["data_fractions"]:
        fraction = float(fraction_value)
        if not 0 < fraction <= 1:
            raise ValueError("data fractions must be in (0, 1]")
        if fraction == 1.0:
            continue  # identical to opt_full_s{full_steps}
        task_count = max(1, math.ceil(len(order) * fraction))
        selected_tasks = set(order[:task_count])
        selected_train_rows = [
            row for row in train_rows if str(row["task_id"]) in selected_tasks
        ]
        steps = max(1, round(full_steps * len(selected_train_rows) / len(train_rows)))
        percent = round(fraction * 100)
        variant_id = f"data_{percent:02d}_equal_epoch"
        data_dir = output_dir / "data" / variant_id
        manifest_path, selected_count, validation_count = _write_data_variant(
            variant_dir=data_dir,
            source_manifest=source_manifest,
            dataset=dataset,
            split_plan=split_plan,
            selected_train_tasks=selected_tasks,
            fraction=fraction,
        )
        config = _variant_config(source_config, data_dir, base_seed, steps)
        config_path = output_dir / "configs" / f"{variant_id}.json"
        _save_json(config_path, config)
        variants.append(
            {
                "variant_id": variant_id,
                "axis": "DATA_SCALE_EQUAL_EPOCH",
                "train_task_fraction": fraction,
                "train_tasks": len(selected_tasks),
                "train_rows": selected_count,
                "validation_rows": validation_count,
                "max_steps": steps,
                "seed": base_seed,
                "data_manifest_path": _repo_relative(manifest_path),
                "data_manifest_sha256": sha256(manifest_path),
                "config_path": _repo_relative(config_path),
                "config_sha256": sha256(config_path),
                "reuse_existing_full_run": False,
            }
        )

    # Stability configs are conditional stage-2 work; generating them is free,
    # but the plan explicitly does not authorize launching them yet.
    for seed_value in protocol["stability_seeds"]:
        seed = int(seed_value)
        if seed == base_seed:
            continue
        variant_id = f"stability_full_s{full_steps}_seed{seed}"
        config = _variant_config(source_config, source_data_dir, seed, full_steps)
        config_path = output_dir / "configs" / f"{variant_id}.json"
        _save_json(config_path, config)
        variants.append(
            {
                "variant_id": variant_id,
                "axis": "STABILITY_SEED",
                "train_task_fraction": 1.0,
                "train_tasks": len(train_tasks),
                "train_rows": len(train_rows),
                "validation_rows": len(validation_rows),
                "max_steps": full_steps,
                "seed": seed,
                "config_path": _repo_relative(config_path),
                "config_sha256": sha256(config_path),
                "conditional_on_stage_1": True,
                "reuse_existing_full_run": False,
            }
        )

    commands = []
    for variant in variants:
        if variant.get("reuse_existing_full_run"):
            continue
        commands.append(
            {
                "variant_id": variant["variant_id"],
                "stage": "STAGE_2_CONDITIONAL"
                if variant.get("conditional_on_stage_1")
                else "STAGE_1",
                "command": (
                    "python -m src.training.run_teacher_sft "
                    f"--config {variant['config_path']} "
                    f"--output-dir <RUN_ROOT>/{variant['variant_id']}"
                ),
            }
        )

    selection_path = output_dir / "train_task_order.json"
    _save_json(
        selection_path,
        {"selection_seed": protocol["selection_seed"], "ordered_task_ids": order},
    )
    result = {
        "schema_version": "retail-teacher-sft-plateau-plan-v1",
        "status": "PREPARED_NOT_RUN",
        "protocol_path": _repo_relative(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "source": {
            "config_path": _repo_relative(source_config_path),
            "config_sha256": sha256(source_config_path),
            "data_manifest_path": _repo_relative(source_data_dir / "manifest.json"),
            "data_manifest_sha256": sha256(source_data_dir / "manifest.json"),
            "train_tasks": len(train_tasks),
            "train_rows": len(train_rows),
            "validation_rows": len(validation_rows),
        },
        "existing_full_run": {
            "manifest_path": _repo_relative(existing_run_manifest_path),
            "manifest_sha256": sha256(existing_run_manifest_path),
            "bound_config_sha256": existing_run["bindings"]["config_sha256"],
            "bound_data_manifest_sha256": existing_run["bindings"][
                "data_manifest_sha256"
            ],
            "local_source_config_sha256": sha256(source_config_path),
            "config_hash_note": (
                "The historical cloud run binds the LF config bytes; the local "
                "Windows working tree may use CRLF. Both hashes are retained."
            ),
        },
        "selection_order_path": _repo_relative(selection_path),
        "selection_order_sha256": sha256(selection_path),
        "variants": variants,
        "commands": commands,
        "claims": {
            "gpu_training_started": False,
            "tau2_evaluation_started": False,
            "sft_plateau_established": False,
            "rl_justified": False,
        },
    }
    _save_json(output_dir / "plan_manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Teacher-SFT plateau plan")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.protocol.resolve(), args.output_dir.resolve())
    print(
        json.dumps(
            {
                "status": result["status"],
                "variants": len(result["variants"]),
                "stage_1_commands": sum(
                    command["stage"] == "STAGE_1" for command in result["commands"]
                ),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
