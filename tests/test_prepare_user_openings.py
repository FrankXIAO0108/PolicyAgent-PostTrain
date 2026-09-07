from pathlib import Path

import pytest

from src.rl.prepare_user_openings import REPO_ROOT, manifest_path, resolve_task_ids


SPLIT = {
    "splits": {
        "rl_train": ["0", "91", "113"],
        "rl_validation": ["6", "8"],
    }
}


def test_resolve_exact_task_ids_preserves_requested_order() -> None:
    assert resolve_task_ids(SPLIT, "rl_train", ["113", "0"], None) == [
        "113",
        "0",
    ]


def test_resolve_exact_task_ids_rejects_outside_subset() -> None:
    with pytest.raises(ValueError, match="outside rl_train"):
        resolve_task_ids(SPLIT, "rl_train", ["6"], None)


def test_resolve_exact_task_ids_rejects_limit_combination() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        resolve_task_ids(SPLIT, "rl_train", ["113"], 1)


def test_manifest_path_is_repo_relative_inside_repo() -> None:
    path = REPO_ROOT / "data" / "retail_agentic_rl_v2" / "opening.jsonl"
    assert manifest_path(path) == "data/retail_agentic_rl_v2/opening.jsonl"


def test_manifest_path_keeps_absolute_path_outside_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # tmp_path may itself be inside the checkout when --basetemp is explicit.
    monkeypatch.setattr("src.rl.prepare_user_openings.REPO_ROOT", tmp_path / "repo")
    outside = tmp_path / "outside" / "opening.jsonl"
    assert Path(manifest_path(outside)) == outside.resolve()
