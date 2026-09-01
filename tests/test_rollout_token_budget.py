from copy import deepcopy

import pytest

from src.evaluation.rollout_token_budget import count_trajectory, summarize_lengths


CHAT = [
    {"role": "system"},
    {"role": "user"},
    {"role": "assistant", "tool_calls": [{"id": "a"}]},
    {"role": "tool"},
    {"role": "assistant"},
]


def render(chat, generation_prompt):
    # EOS-boundary behavior: model ends at EOS, tool suffix supplies next header.
    tokens = [1, 2, 3]
    for message in chat[2:]:
        tokens += [10, 99] if message["role"] == "assistant" else [20, 21, 22, 23]
    return tokens


def test_accounts_for_tools_and_model_without_truncation():
    row = count_trajectory(CHAT, render, 99)
    assert row["prompt_tokens"] == 3
    assert row["model_tokens"] == 4
    assert row["observation_tokens_with_template"] == 4
    assert row["total_tokens"] == 11
    assert row["completion_tokens"] == 8
    assert row["max_single_observation_tokens"] == 4


def test_rejects_nonprefix_template():
    def broken(chat, generation_prompt):
        return ([7] if len(chat) > 3 else []) + render(chat, generation_prompt)

    with pytest.raises(ValueError, match="prefix-preserving"):
        count_trajectory(CHAT, broken, 99)


def test_rejects_incomplete_reference():
    with pytest.raises(ValueError, match="final model response"):
        count_trajectory(CHAT[:-1], render, 99)


def test_rejects_tool_without_call():
    chat = deepcopy(CHAT)
    chat[2].pop("tool_calls")
    with pytest.raises(ValueError, match="preceding atomic"):
        count_trajectory(chat, render, 99)
    with pytest.raises(ValueError, match="missing tool result"):
        count_trajectory(CHAT[:3] + CHAT[4:], render, 99)


def test_reference_is_not_approval_or_capability_claim():
    report = summarize_lengths([count_trajectory(CHAT, render, 99)])
    assert report["reference_budget_with_25pct_headroom"] == 1024
    assert not report["budget_approved"]
    assert not report["sft_capability_assessed"]
    assert not report["gpu_feasibility_verified"]
    with pytest.raises(ValueError, match="No eligible"):
        summarize_lengths([])
