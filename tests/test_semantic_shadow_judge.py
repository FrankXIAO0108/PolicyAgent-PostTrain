import copy
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from src.evaluation import semantic_shadow_judge as judge


@pytest.fixture
def raw():
    return {
        "task_id": "44",
        "reward": 1,
        "quality": "GOLD",
        "messages": [
            {"role": "assistant", "content": "Refund $17.99 to gift card?"},
            {"role": "user", "content": "Yes, please."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "name": "modify_pending_order_items",
                        "arguments": {"order_id": "O1"},
                    }
                ],
            },
            {"role": "tool", "content": "Refunded $17.99", "id": "t1"},
            {"role": "assistant", "content": "Refunded $17.99"},
        ],
    }


@pytest.fixture
def packet(raw):
    return judge.build_packet(raw, "Confirm before writing.", row=1)


@pytest.fixture
def review(packet):
    return {
        "trajectory_sha256": packet["trajectory_sha256"],
        "write_reviews": [
            {
                "write_message_index": 2,
                "verdict": "PASS",
                "reason": "已确认",
                "evidence": [
                    {"message_index": 0, "quote": "Refund $17.99"},
                    {"message_index": 1, "quote": "Yes, please."},
                ],
            }
        ],
        "factual_consistency": {
            "verdict": "PASS",
            "reason": "回复与工具一致",
            "evidence": [
                {"message_index": 3, "quote": "Refunded $17.99"},
                {"message_index": 4, "quote": "Refunded $17.99"},
            ],
        },
        "limitations": "Not independently calibrated",
    }


def test_packet_is_score_blind_and_retains_tools(raw, packet):
    raw["reward"] = 0
    raw["quality"] = "REJECT"
    assert packet == judge.build_packet(raw, "Confirm before writing.", row=1)
    assert "reward" not in packet and "quality" not in packet
    assert packet["messages"][2]["tool_calls"] == raw["messages"][2]["tool_calls"]
    assert len(packet["messages"]) == len(raw["messages"])


def test_valid_review_and_uncertainty(review, packet):
    assert judge.validate_review(json.dumps(review), packet) == review
    for item in [*review["write_reviews"], review["factual_consistency"]]:
        item.update(verdict="UNCERTAIN", evidence=[])
    judge.validate_review(json.dumps(review), packet)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r.update(trajectory_sha256="wrong"),
        lambda r: r.update(quality="gold"),
        lambda r: r.update(write_reviews=[]),
        lambda r: r["write_reviews"].append(copy.deepcopy(r["write_reviews"][0])),
        lambda r: r["write_reviews"][0].update(write_message_index=4),
        lambda r: r["write_reviews"][0].update(verdict="GOLD"),
        lambda r: r["write_reviews"][0].update(reason=""),
        lambda r: r["write_reviews"][0].update(evidence=[]),
        lambda r: r["write_reviews"][0]["evidence"].pop(),
        lambda r: r["write_reviews"][0]["evidence"][0].update(message_index=4),
        lambda r: r["write_reviews"][0]["evidence"][0].update(message_index=True),
        lambda r: r["write_reviews"][0]["evidence"][0].update(quote="Invented quote"),
        lambda r: r["factual_consistency"]["evidence"].pop(),
    ],
)
def test_fail_closed_on_invalid_review(mutation, review, packet):
    mutation(review)
    with pytest.raises(ValueError):
        judge.validate_review(json.dumps(review), packet)


def test_duplicate_json_key(packet):
    with pytest.raises(ValueError, match="Duplicate"):
        judge.validate_review('{"a":1,"a":2}', packet)


@pytest.fixture
def config(tmp_path, raw):
    source, policy = tmp_path / "raw.jsonl", tmp_path / "policy.md"
    source.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    policy.write_text("Confirm before writing.", encoding="utf-8")
    return {
        "mode": "SHADOW_ONLY",
        "used_as_training_reward": False,
        "raw_rollouts": str(source),
        "raw_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "policy_path": str(policy),
        "policy_sha256": hashlib.sha256(policy.read_bytes()).hexdigest(),
        "rows": [1],
        "max_requests": 1,
        "max_request_chars": 120000,
        "max_output_tokens": 4096,
        "timeout_seconds": 60,
        "judge": {
            "model": "deepseek-v4-flash",
            "provider": "deepseek",
            "approved_host": "api.deepseek.com",
            "api_key_env": "SHADOW_JUDGE_API_KEY",
            "base_url_env": "SHADOW_JUDGE_BASE_URL",
        },
    }


@pytest.mark.parametrize(
    "url",
    [
        "http://api.deepseek.com",
        "https://evil.example",
        "https://api.deepseek.com@evil.example",
        "https://api.deepseek.com?x=1",
        "https://api.deepseek.com:9999",
        "https://api.deepseek.com/unapproved",
    ],
)
def test_endpoint_restricted(config, monkeypatch, url):
    monkeypatch.setenv("SHADOW_JUDGE_BASE_URL", url)
    monkeypatch.setenv("SHADOW_JUDGE_API_KEY", "test-only")
    with pytest.raises(ValueError):
        judge.validate_endpoint(config)


def test_chosen_model_allowed_but_no_fallback(config, monkeypatch):
    monkeypatch.setenv("SHADOW_JUDGE_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("SHADOW_JUDGE_API_KEY", "test-only")
    assert judge.validate_endpoint(config) == "https://api.deepseek.com"
    config["judge"]["model"] = "Qwen3-4B-Instruct-2507"
    with pytest.raises(ValueError):
        judge.validate_endpoint(config)


def run_cli(tmp_path, config, monkeypatch, execute=False):
    path, output = tmp_path / "config.json", tmp_path / "out"
    path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["semantic_shadow_judge", "--config", str(path), "--output-dir", str(output)]
        + (["--execute-api"] if execute else []),
    )
    judge.main()
    return output


def test_prepare_never_calls_api(config, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("API client must not be constructed in dry run")

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=forbidden))
    output = run_cli(tmp_path, config, monkeypatch)
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["external_api_called"] is False
    assert manifest["request_count"] == 1
    with pytest.raises(FileExistsError):
        judge.main()


def test_api_failure_is_error_not_model_fail(config, tmp_path, monkeypatch):
    monkeypatch.setenv("SHADOW_JUDGE_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("SHADOW_JUDGE_API_KEY", "test-only")

    def failure(**kwargs):
        assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
        raise TimeoutError("Do not persist credential-containing exception text")

    def client(**kwargs):
        assert kwargs["max_retries"] == 0
        return SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=failure))
        )

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=client))
    with pytest.raises(RuntimeError):
        run_cli(tmp_path, config, monkeypatch, execute=True)
    manifest = json.loads((tmp_path / "out/run_manifest.json").read_text())
    assert manifest["status"] == "ERROR"
    assert manifest["error_type"] == "TimeoutError"
    assert manifest["used_as_training_reward"] is False
    assert not list((tmp_path / "out").glob("*_review.json"))


def test_source_hash_mismatch(config, tmp_path, monkeypatch):
    config["raw_sha256"] = "bad"
    with pytest.raises(ValueError, match="hash"):
        run_cli(tmp_path, config, monkeypatch)
