from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from src.evaluation.final_holdout_rebuild_cost import task_stratum


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "retail-teacher-expansion-150-plan-v1.0.0"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def git_commit(path: Path) -> str:
    return subprocess.run(
        ["git", "-c", f"safe.directory={path.as_posix()}", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def build_plan(
    *,
    protocol: dict[str, Any],
    tasks: list[dict[str, Any]],
    split: dict[str, Any],
    current_sft: list[dict[str, Any]],
    development: dict[str, Any],
    final_holdout: dict[str, Any],
    template: dict[str, Any],
    task_split_path: str,
) -> dict[str, Any]:
    if protocol.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Expansion protocol schema version mismatch")
    task_by_id = {str(task["id"]): task for task in tasks}
    upstream_train = {str(task_id) for task_id in split.get("train") or []}
    development_ids = {str(row["task_id"]) for row in development.get("tasks") or []}
    final_ids = {str(task_id) for task_id in final_holdout.get("task_ids") or []}
    if development_ids & final_ids:
        raise ValueError("Development evaluation overlaps the sealed final holdout")
    eligible_ids = upstream_train - development_ids
    if eligible_ids & final_ids:
        raise ValueError("Expansion pool overlaps the sealed final holdout")
    missing = eligible_ids - set(task_by_id)
    if missing:
        raise ValueError(f"Eligible tasks missing from tasks.json: {missing}")

    current_task_ids = {str(row["task_id"]) for row in current_sft}
    if current_task_ids & final_ids:
        raise ValueError("Current SFT data overlaps the sealed final holdout")
    task_rows = [
        {
            "task_id": task_id,
            "stratum": task_stratum(task_by_id[task_id]),
            "reason": "Upstream train task excluded from frozen development and final evaluation.",
            "already_present_in_current_sft": task_id in current_task_ids,
        }
        for task_id in sorted(eligible_ids, key=int)
    ]
    wave_a = protocol["wave_a"]
    candidates_per_task = int(wave_a["candidates_per_task"])
    candidate_count = len(task_rows) * candidates_per_task
    retained_rate = float(protocol["retention_evidence"]["observed_retained_rate"])
    expected_retained = round(candidate_count * retained_rate)
    wave_b_max = len(task_rows) * int(
        protocol["wave_b"]["maximum_additional_candidates_per_task"]
    )
    expected_wave_b = round(wave_b_max * retained_rate)

    generated_config = copy.deepcopy(template)
    generated_config.update(
        {
            "schema_version": "retail-tau2-teacher-expansion-150-wave-a-v1",
            "status": "FROZEN",
            "scope": "TAU2_GROUNDED_TEACHER_DATA_ENGINEERING_PILOT_EXPANSION_150_WAVE_A",
            "purpose": "Generate Wave-A candidates for the 53-to-150 owner-reviewed SFT expansion.",
            "task_split": task_split_path,
            "task_subset": "generation_pool",
            "tasks": [
                {
                    key: value
                    for key, value in row.items()
                    if key != "already_present_in_current_sft"
                }
                for row in task_rows
            ],
        }
    )
    generated_config["generation"]["agent"]["temperature_ladder"] = list(
        wave_a["temperature_ladder"]
    )
    generated_config["generation"]["candidates_per_task"] = candidates_per_task
    generated_config["generation"]["seed"] = int(wave_a["seed"])
    generated_config["claims"]["official_test_reserved"] = True
    generated_config["known_limitations"] = [
        "Wave A is candidate generation only; every released row still requires deterministic checks and owner review.",
        "The 0.53 retained rate is a planning estimate from the 32-candidate Layer-1 pilot, not a guaranteed yield.",
        "Actual trajectory entity groups determine the later TRAIN/VALIDATION split; no split is assigned before generation.",
        "The teacher and user simulator share a model family, so same-source bias remains disclosed.",
    ]
    smoke_task_ids = [str(task_id) for task_id in wave_a["smoke_task_ids"]]
    smoke_task_set = set(smoke_task_ids)
    eligible_task_set = {row["task_id"] for row in task_rows}
    if len(smoke_task_ids) != len(smoke_task_set):
        raise ValueError("Wave-A smoke task IDs must be unique")
    if not smoke_task_set or not smoke_task_set <= eligible_task_set:
        raise ValueError("Wave-A smoke tasks must be a non-empty eligible subset")
    smoke_config = copy.deepcopy(generated_config)
    smoke_config.update(
        {
            "schema_version": "retail-tau2-teacher-expansion-150-wave-a-smoke-v1",
            "scope": (
                "TAU2_GROUNDED_TEACHER_DATA_ENGINEERING_PILOT_"
                "EXPANSION_150_WAVE_A_SMOKE"
            ),
            "purpose": (
                "Validate the Wave-A environment and persistence path on a small "
                "bound task subset before the 61-task run."
            ),
            "tasks": [
                row
                for row in generated_config["tasks"]
                if row["task_id"] in smoke_task_set
            ],
        }
    )
    smoke_config["tasks"].sort(key=lambda row: smoke_task_ids.index(row["task_id"]))

    task_split = {
        "schema_version": "retail-teacher-expansion-150-task-split-v1",
        "splits": {
            "generation_pool": [row["task_id"] for row in task_rows],
            "rl_validation": [],
            "development_audit": sorted(development_ids, key=int),
            "sealed_final_holdout": sorted(final_ids, key=int),
        },
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PREPARED_NOT_RUN",
        "summary": {
            "upstream_train_task_count": len(upstream_train),
            "excluded_development_task_count": len(upstream_train & development_ids),
            "eligible_task_count": len(task_rows),
            "eligible_already_present_in_current_sft": sum(
                row["already_present_in_current_sft"] for row in task_rows
            ),
            "eligible_new_to_current_sft": sum(
                not row["already_present_in_current_sft"] for row in task_rows
            ),
            "current_sft_rows": len(current_sft),
            "wave_a_candidate_count": candidate_count,
            "wave_a_expected_retained_rows": expected_retained,
            "wave_a_projected_total_rows": len(current_sft) + expected_retained,
            "wave_b_max_candidate_count": wave_b_max,
            "wave_b_expected_retained_rows": expected_wave_b,
            "wave_a_plus_b_projected_total_rows": len(current_sft)
            + expected_retained
            + expected_wave_b,
        },
        "tasks": task_rows,
        "wave_a_smoke_config": smoke_config,
        "wave_a_generation_config": generated_config,
        "task_split": task_split,
        "gates": {
            "development_overlap_count": len(eligible_ids & development_ids),
            "final_holdout_overlap_count": len(eligible_ids & final_ids),
            "external_api_called": False,
            "gpu_training_started": False,
            "passed": True,
        },
    }


def prepare(
    *, protocol_path: Path, upstream_root: Path, output_dir: Path
) -> dict[str, Any]:
    protocol_path = protocol_path.resolve()
    upstream_root = upstream_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    protocol = load_json(protocol_path)
    if protocol.get("status") != "FROZEN":
        raise ValueError("Expansion protocol must be FROZEN")
    if git_commit(upstream_root) != protocol["upstream"]["commit"]:
        raise ValueError("Upstream commit mismatch")

    paths = {
        "tasks": upstream_root / protocol["upstream"]["tasks_path"],
        "split": upstream_root / protocol["upstream"]["split_path"],
        "current_sft": REPO_ROOT / protocol["current_sft"]["path"],
        "development_evaluation": REPO_ROOT
        / protocol["development_evaluation"]["path"],
        "final_holdout": REPO_ROOT / protocol["final_holdout"]["path"],
        "generation_template": REPO_ROOT / protocol["generation_template"]["path"],
        "retention_evidence": REPO_ROOT / protocol["retention_evidence"]["path"],
    }
    expected = {
        "tasks": protocol["upstream"]["tasks_sha256"],
        "split": protocol["upstream"]["split_sha256"],
        "current_sft": protocol["current_sft"]["sha256"],
        "development_evaluation": protocol["development_evaluation"]["sha256"],
        "final_holdout": protocol["final_holdout"]["sha256"],
        "generation_template": protocol["generation_template"]["sha256"],
        "retention_evidence": protocol["retention_evidence"]["sha256"],
    }
    for name, path in paths.items():
        if sha256(path) != str(expected[name]).upper():
            raise ValueError(f"{name} hash mismatch")

    current_sft = load_jsonl(paths["current_sft"])
    if len(current_sft) != int(protocol["current_sft"]["rows"]):
        raise ValueError("Current SFT row count mismatch")
    try:
        split_rel = str(
            (output_dir / "task_split.json").relative_to(REPO_ROOT)
        ).replace("\\", "/")
    except ValueError as exc:
        raise ValueError(
            "Output directory must be inside the project repository"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    plan = build_plan(
        protocol=protocol,
        tasks=load_json(paths["tasks"]),
        split=load_json(paths["split"]),
        current_sft=current_sft,
        development=load_json(paths["development_evaluation"]),
        final_holdout=load_json(paths["final_holdout"]),
        template=load_json(paths["generation_template"]),
        task_split_path=split_rel,
    )
    outputs = {
        "expansion_plan.json": {
            key: value
            for key, value in plan.items()
            if key
            not in {
                "wave_a_smoke_config",
                "wave_a_generation_config",
                "task_split",
            }
        },
        "wave_a_smoke_config.json": plan["wave_a_smoke_config"],
        "wave_a_generation_config.json": plan["wave_a_generation_config"],
        "task_split.json": plan["task_split"],
    }
    for name, value in outputs.items():
        (output_dir / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "PREPARED_NOT_RUN",
        "summary": plan["summary"],
        "gates": plan["gates"],
        "outputs": {name: sha256(output_dir / name) for name in outputs},
        "protocol_sha256": sha256(protocol_path),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "external_api_called": False,
    }
    (output_dir / "plan_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPO_ROOT / "configs/retail_teacher_expansion_150_v1.json",
    )
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                protocol_path=args.protocol,
                upstream_root=args.upstream_root,
                output_dir=args.output,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
