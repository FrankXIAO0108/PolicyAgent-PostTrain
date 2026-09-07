"""Local artifact regression: do not run cloud inference or require torch."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from src.training.rollout_diagnostics import validate_sampling_request

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "_local_private_runs/tr0905/s100"
NAMES = ("task44_greedy", "task44_n4", "task113_greedy", "task113_n4")
pytestmark = pytest.mark.skipif(
    not (BASE / "probe_v2/launch_probe.py").exists(),
    reason="Private SFT development probe artifacts not present",
)


def read(version, name):
    return json.loads((BASE / version / (name + ".json")).read_text(encoding="utf-8-sig"))


@pytest.mark.parametrize("name", NAMES)
def test_frozen_old_config_reproduces_missing_claim(name):
    with pytest.raises(ValueError, match="claims.rl_train_task_only"):
        validate_sampling_request(read("probe_v1", name), 8192, 1)


@pytest.mark.parametrize("name", NAMES)
def test_new_config_changes_only_required_claim(name):
    old = read("probe_v1", name)
    new = read("probe_v2", name)
    old["claims"]["rl_train_task_only"] = True
    assert old == new
    contract = validate_sampling_request(new, 8192, 1)
    assert contract["actual_num_generations"] == (1 if name.endswith("greedy") else 4)
    assert new["claims"]["parameter_update_allowed"] is False


def run_preflight(config_dir=None):
    command = [sys.executable, str(BASE / "probe_v2/launch_probe.py"), "--validate-only"]
    if config_dir is not None:
        command += ["--config-dir", str(config_dir)]
    return subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=30)


def test_actual_launcher_validates_all_four_without_gpu():
    result = run_preflight()
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["expected_rollouts"] == 10
    assert set(report["checks"]) == set(NAMES)
    assert report["external_api_called"] is False
    assert report["gpu_used"] is False
    assert report["remote_model_runtime_verified"] is False


def test_last_config_failure_is_caught_before_runtime(tmp_path):
    for name in NAMES:
        config = read("probe_v2", name)
        if name == NAMES[-1]:
            del config["claims"]["rl_train_task_only"]
        (tmp_path / (name + ".json")).write_text(json.dumps(config), encoding="utf-8")
    result = run_preflight(tmp_path)
    assert result.returncode != 0
    assert "claims.rl_train_task_only" in result.stderr
    assert "Missing existing simulator credential" not in result.stderr
    assert "No module named 'torch'" not in result.stderr


@pytest.mark.parametrize("change", ["lr", "budget", "count"])
def test_rejects_accidental_sampling_scope_change(change):
    config = read("probe_v2", "task44_n4")
    if change == "lr":
        config["grpo"]["learning_rate"] = 1e-6
    elif change == "budget":
        config["grpo"]["max_completion_length"] = 4096
    else:
        config["diagnostic"]["expected_rollouts"] = 8
    with pytest.raises(ValueError):
        validate_sampling_request(config, 8192, 1)


@pytest.mark.parametrize("name", NAMES)
def test_new_config_retains_verified_data_bindings(name):
    from src.training.run_retail_agentic_grpo import (
        load_json, load_jsonl, sha256, validate_opening_contract,
        validate_sft_manifest_binding,
    )
    config = read("probe_v2", name)
    data = config["data"]
    split_path = ROOT / data["task_split"]
    split = load_json(split_path)
    assert set(data["task_ids"]) <= set(split["splits"][data["train_subset"]])
    validate_sft_manifest_binding(config["model"], split)
    opening_path = ROOT / data["openings"]
    validate_opening_contract(
        data=data, split_sha256=sha256(split_path), openings_path=opening_path,
        openings_manifest=load_json(ROOT / data["openings_manifest"]),
        openings=load_jsonl(opening_path), expected_ordered=data["task_ids"], strict=True,
    )
