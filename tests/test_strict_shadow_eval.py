from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.evaluation.run_strict_shadow_eval import (
    _normalise_messages,
    default_source_matrix,
    evaluate_source_artifact,
    write_shadow_report,
)


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_result(path: Path, *, task_id: str, reward_info, termination_reason="user_stop") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "tasks": [{"id": task_id}],
                "simulations": [
                    {
                        "task_id": task_id,
                        "termination_reason": termination_reason,
                        "reward_info": reward_info,
                        "messages": [
                            {"role": "assistant", "content": None, "tool_calls": [
                                {"id": "call-1", "name": "find_user_id_by_name_zip", "arguments": {}}
                            ]},
                            {"role": "tool", "id": "call-1", "content": "{}", "error": False},
                            {"role": "assistant", "content": "done", "tool_calls": None},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_message_normalisation_preserves_tool_result_success() -> None:
    messages = _normalise_messages(
        [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "x", "name": "lookup", "arguments": {"id": "1"}}
            ]},
            {"role": "tool", "id": "x", "content": "{}", "error": False},
        ]
    )
    assert messages[0]["tool_calls"][0] == {
        "id": "x",
        "name": "lookup",
        "arguments": {"id": "1"},
    }
    assert messages[1]["tool_call_id"] == "x"
    assert messages[1]["success"] is True


def test_shadow_evaluation_records_hash_and_never_modifies_source(tmp_path: Path) -> None:
    source = tmp_path / "returned_results.json"
    write_result(source, task_id="24", reward_info={"reward": 1.0})
    before = sha256(source)
    row = evaluate_source_artifact(
        task_id="24",
        stage="base",
        source_path=source,
        rubric_path=ROOT / "configs/evaluation/retail_strict_v1/task_24.json",
    )
    assert sha256(source) == before == row["source_sha256"]
    assert row["source_unchanged"] is True
    assert row["tau2_result"] == {"reward": 1.0}
    assert row["strict_evaluation"] is not None
    assert row["strict_evaluation"]["evaluation_valid"] is False
    assert row["strict_evaluation"]["strict_pass"] is False
    assert row["status"] == "EVALUATED_EVIDENCE_INCOMPLETE"


def test_reward_info_none_is_separate_infrastructure_failure(tmp_path: Path) -> None:
    source = tmp_path / "returned_results.json"
    write_result(
        source,
        task_id="24",
        reward_info=None,
        termination_reason="infrastructure_error",
    )
    row = evaluate_source_artifact(
        task_id="24",
        stage="sft",
        source_path=source,
        rubric_path=ROOT / "configs/evaluation/retail_strict_v1/task_24.json",
    )
    assert row["status"] == "INFRASTRUCTURE_FAILURE"
    assert row["tau2_result"] is None
    assert row["strict_evaluation"] is None
    assert "not counted as a model failure" in row["reason"]


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_shadow_report(ROOT, output)
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_default_matrix_records_missing_and_protocol_incompatible_cells() -> None:
    matrix = default_source_matrix(ROOT)
    assert matrix["base"]["21"] is None
    assert matrix["sft"]["21"] is None
    assert matrix["sft"]["107"] is not None
    assert "rerun" in str(matrix["sft"]["107"])
    assert set(matrix["rl"].values()) == {None}
