"""No paid calls: bounded batch and no-fallback assertions."""

import json

import pytest

from scripts.run_task44_recovered_semantics import execute


@pytest.mark.parametrize("failed_index", [None, 0, 1, 2, 3])
def test_once_per_frozen_row_no_retry_or_fallback(tmp_path, failed_index):
    calls = []
    rows = [
        {
            "rule_only_reward": 0.9,
            "scoring_input": {"base": {}, "raw": {"index": i}, "spec": {}},
        }
        for i in range(4)
    ]

    def scorer(base, raw, spec, **kwargs):
        calls.append(raw["index"])
        if raw["index"] == failed_index:
            raise TimeoutError("mock; must never become a numeric reward")
        return {"offline_reward": 0.2 + raw["index"] * 0.1}

    results = execute(rows, "policy", tmp_path, scorer=scorer)
    assert calls == [0, 1, 2, 3]
    assert len(results) == len(list(tmp_path.glob("result_*.json"))) == 4
    for i, row in enumerate(results):
        assert row["training_eligible"] is False
        assert json.loads((tmp_path / f"result_{i}.json").read_text()) == row
        if i == failed_index:
            assert row["hybrid_reward"] is None and row["status"] == "SCORING_ERROR"
        else:
            assert row["status"] == "READY"
            assert row["details"]["used_as_training_reward"] is False
