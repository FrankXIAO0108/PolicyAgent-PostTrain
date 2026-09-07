"""CPU/mock-only contract tests; no paid API, GPU, or training improvement claim."""

import hashlib
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_task44_partial_semantics import scoring_fixture
from src.evaluation import task44_hybrid_reward as hybrid

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/retail_agentic_qwen3_4b_task44_hybrid_n4_50step_v1.json"


def case():
    base, raw, old, spec = scoring_fixture()
    base["staged_reward"] = 0.75
    settings = json.loads(CONFIG.read_text(encoding="utf-8"))["reward"][
        "staged_reward_spec"
    ]["semantic_assistance"]
    spec["semantic_assistance"] = {
        **settings,
        "policy_sha256": hashlib.sha256(b"policy").hexdigest(),
    }
    packet = hybrid.candidate_packet(raw, "policy")
    answer = {
        "trajectory_sha256": packet["trajectory_sha256"],
        "candidate_table_sha256": packet["candidate_table_sha256"],
        "authorizations": old["authorizations"],
        "candidate_results": [],
    }
    for c in packet["candidates"]:
        claims = []
        for kind in c["required_kinds"]:
            claims.append(
                {
                    "kind": kind,
                    "item_id": "old" if kind == "cheapest" else None,
                    "value": True if kind == "cheapest" else "17.99",
                }
            )
        answer["candidate_results"].append(
            {
                "candidate_id": c["candidate_id"],
                "status": "EXTRACTED" if claims else "NOT_IN_SCOPE",
                "claims": claims,
            }
        )
    return base, raw, spec, packet, answer


def response(answer):
    return {
        "choices": [
            {"finish_reason": "stop", "message": {"content": json.dumps(answer)}}
        ],
        "usage": {"total_tokens": 123},
    }


def test_online_offline_cached_parity_and_no_second_api(tmp_path):
    base, raw, spec, packet, answer = case()
    calls = []

    def request(p, s):
        assert list(tmp_path.glob("semantic_cache/*/request.json"))
        calls.append(1)
        assert p == packet
        return response(answer)

    first = hybrid.hybrid_score(
        base, raw, spec, policy="policy", directory=tmp_path, request=request
    )
    cached = hybrid.hybrid_score(
        base, raw, spec, policy="policy", directory=tmp_path, request=request
    )
    offline = hybrid.score_candidate_response(
        base, raw, spec, packet, json.dumps(answer)
    )
    assert (
        first["offline_reward"]
        == cached["offline_reward"]
        == offline["offline_reward"]
        == 0.85
    )
    assert first["used_as_training_reward"] and cached["cache_hit"] and calls == [1]
    assert base["staged_reward"] == 0.75


@pytest.mark.parametrize(
    "mutate",
    [
        lambda a: a["candidate_results"].pop(0),
        lambda a: a["candidate_results"][0].update(candidate_id="invented"),
        lambda a: a.update(candidate_table_sha256="wrong"),
        lambda a: a.update(trajectory_sha256="wrong"),
        lambda a: a.update(reward=1),
        lambda a: a["candidate_results"][0].update(status="NOT_IN_SCOPE", claims=[]),
        lambda a: a["candidate_results"][0].update(status="UNCERTAIN", claims=[]),
        lambda a: a["candidate_results"][0]["claims"][0].update(item_id="invented"),
    ],
)
def test_bad_or_unknown_semantics_cannot_release_score(mutate):
    base, raw, spec, packet, answer = case()
    mutate(answer)
    with pytest.raises((ValueError, hybrid.SemanticScoringError)):
        hybrid.score_candidate_response(base, raw, spec, packet, json.dumps(answer))


def availability_case():
    base, raw, spec, _, answer = case()
    product = json.loads(raw["messages"][3]["content"])
    product["variants"]["old"]["options"] = {
        "color": "white",
        "brightness": "high",
        "power": "USB",
    }
    product["variants"]["new"]["options"] = {
        "color": "black",
        "brightness": "medium",
        "power": "AC",
    }
    raw["messages"][3]["content"] = json.dumps(product)
    # Regression for the row194 failure mechanism, without hardcoding its IDs.
    raw["messages"][4]["content"] = (
        "1. **white, high, USB** - $153.23 (not available)\nReplace old with new in o1; refund $17.99 to gift_card_1. Agree?"
    )
    packet = hybrid.candidate_packet(raw, "policy")
    answer.update(
        trajectory_sha256=packet["trajectory_sha256"],
        candidate_table_sha256=packet["candidate_table_sha256"],
    )
    answer["candidate_results"] = []
    for c in packet["candidates"]:
        claims = [
            {
                "kind": k,
                "item_id": "old" if k == "availability" else None,
                "value": False if k == "availability" else "17.99",
            }
            for k in c["required_kinds"]
        ]
        answer["candidate_results"].append(
            {
                "candidate_id": c["candidate_id"],
                "status": "EXTRACTED" if claims else "NOT_IN_SCOPE",
                "claims": claims,
            }
        )
    return base, raw, spec, packet, answer


def test_194_style_false_availability_cannot_earn_full_communication():
    base, raw, spec, packet, answer = availability_case()
    result = hybrid.score_candidate_response(
        base, raw, spec, packet, json.dumps(answer)
    )
    assert result["additive_components"]["post_write_communication"] == 0
    assert result["offline_reward"] == 0.85


@pytest.mark.parametrize("mode", ["omit", "flip", "wrong_entity"])
def test_194_style_omission_or_substitution_blocks_update(mode):
    base, raw, spec, packet, answer = availability_case()
    row = next(
        r
        for r in answer["candidate_results"]
        if any(c["kind"] == "availability" for c in r["claims"])
    )
    if mode == "omit":
        row.update(status="NOT_IN_SCOPE", claims=[])
    elif mode == "flip":
        row["claims"][0]["value"] = True
    else:
        row["claims"][0]["item_id"] = "new"
    with pytest.raises(ValueError):
        hybrid.score_candidate_response(base, raw, spec, packet, json.dumps(answer))


def test_later_correction_does_not_remove_earlier_candidate():
    rows = hybrid.candidates(
        [
            {
                "role": "assistant",
                "content": "old is cheapest. Actually new is cheapest. Refund $17.99.\n"
                + "x" * 300,
            }
        ]
    )
    assert [r["required_kinds"] for r in rows] == [
        ["cheapest"],
        ["cheapest"],
        ["refund_amount"],
        [],
    ]
    assert len(rows[-1]["text"]) == 300


def test_no_claims_trajectory_not_given_fake_communication_credit():
    base, raw, spec, _, _ = case()
    raw["messages"] = [{"role": "assistant", "content": "Hello."}]
    base["write_complete"] = False
    base["confirmation_evidence"]["checks"] = []
    base["additive_components"]["post_write_communication"] = 0
    packet = hybrid.candidate_packet(raw, "policy")
    answer = {
        "trajectory_sha256": packet["trajectory_sha256"],
        "candidate_table_sha256": packet["candidate_table_sha256"],
        "authorizations": [],
        "candidate_results": [
            {
                "candidate_id": packet["candidates"][0]["candidate_id"],
                "status": "NOT_IN_SCOPE",
                "claims": [],
            }
        ],
    }
    result = hybrid.score_candidate_response(
        base, raw, spec, packet, json.dumps(answer)
    )
    assert result["additive_components"]["post_write_communication"] == 0
    assert result["offline_reward"] <= 0.25


@pytest.mark.parametrize("failure", ["timeout", "invalid_json", "truncated"])
def test_request_failure_preserves_evidence_and_never_retries(tmp_path, failure):
    base, raw, spec, _, answer = case()
    calls = []

    def request(*args):
        calls.append(1)
        if failure == "timeout":
            raise TimeoutError("mock")
        result = response(answer)
        if failure == "invalid_json":
            result["choices"][0]["message"]["content"] = "not json"
        else:
            result["choices"][0]["finish_reason"] = "length"
        return result

    for _ in range(2):
        with pytest.raises((hybrid.SemanticScoringError, FileExistsError)):
            hybrid.hybrid_score(
                base, raw, spec, policy="policy", directory=tmp_path, request=request
            )
    assert calls == [1]
    assert list(tmp_path.glob("semantic_cache/*/request.json"))
    assert list(tmp_path.glob("semantic_cache/*/error.json"))


@pytest.mark.parametrize(
    "restriction", ["identity", "unexpected", "rule_fail", "refund_175"]
)
def test_hybrid_retains_hard_caps(restriction):
    base, raw, spec, packet, answer = case()
    if restriction == "identity":
        base["components"]["identity_link"]["value"] = 0
    elif restriction == "unexpected":
        base["unexpected_write_count"] = 1
    elif restriction == "rule_fail":
        base["confirmation_evidence"]["checks"][0]["verified_verdict"] = "FAIL"
    else:
        raw["messages"][4]["content"] = raw["messages"][4]["content"].replace(
            "17.99", "18.99"
        )
        answer["authorizations"][0]["parameters"]["refund_amount"] = "18.99"
        answer["authorizations"][0]["evidence"][0]["quote"] = "refund $18.99"
        packet = hybrid.candidate_packet(raw, "policy")
        answer.update(
            trajectory_sha256=packet["trajectory_sha256"],
            candidate_table_sha256=packet["candidate_table_sha256"],
        )
        for row in answer["candidate_results"]:
            for claim in row["claims"]:
                if claim["kind"] == "refund_amount" and row["candidate_id"].startswith(
                    "m4:"
                ):
                    claim["value"] = "18.99"
    assert (
        hybrid.score_candidate_response(base, raw, spec, packet, json.dumps(answer))[
            "offline_reward"
        ]
        <= 0.15
    )


@pytest.mark.parametrize("bad_row", [None, 0, 1, 2, 3])
def test_real_guard_hook_scores_whole_group_before_backward(
    tmp_path, monkeypatch, bad_row
):
    torch = pytest.importorskip(
        "torch", reason="CPU backward contract requires PyTorch"
    )
    from src.training.rollout_diagnostics import make_guarded_grpo_trainer

    monkeypatch.setenv("POLICYAGENT_ROLLOUT_LOG", str(tmp_path / "rollouts.jsonl"))
    monkeypatch.setenv("POLICYAGENT_REWARD_CONFIG_JSON", "{}")
    order = []

    class Environment:
        _runtime_stop_signals = []
        _user_stopped = True

        def __init__(self, index):
            self.index, self.cached = index, None

        def _set_trainer_completion_telemetry(self, payload):
            self.telemetry = payload

        def _blocking_transport_reasons(self):
            return []

        def _uses_semantic_reward(self):
            return True

        def _semantic_pending_snapshot(self):
            return {"task_id": "44", "messages": [], "completion": self.telemetry}

        def get_reward(self):
            if self.cached is not None:
                return self.cached
            assert list(tmp_path.glob("semantic_group_*.json"))
            order.append(f"score{self.index}")
            if self.index == bad_row:
                raise hybrid.SemanticScoringError("mock reviewer outage")
            self.cached = [0.75, 0.85, 1.0, 0.15][self.index]
            return self.cached

    class Native:
        def __init__(self):
            self.model = SimpleNamespace(
                config=SimpleNamespace(max_position_embeddings=100)
            )
            self._is_vlm = False
            self.state = SimpleNamespace(global_step=0)
            self.max_completion_length = 10
            self.max_tool_calling_iterations = 32
            self._tokenizer = SimpleNamespace(
                eos_token_id=99, decode=lambda ids, **kw: str(ids)
            )
            self.environments = [Environment(i) for i in range(4)]

        def _generate(self, prompts):
            ids = [[5, 99]] * 4
            trace = self._policyagent_stop_trace
            trace.generated(trace.start(prompts), prompts, ids, 99)
            return (
                prompts,
                ids,
                [[1, 1]] * 4,
                [[{"role": "assistant", "content": "done"}]] * 4,
            )

    trainer = make_guarded_grpo_trainer(Native, lambda _: None)()
    parameter = torch.tensor(1.0, requires_grad=True)
    optimizer = torch.optim.SGD([parameter], lr=0.1)

    def train_double():
        trainer._generate([[1]] * 4)
        assert [e.get_reward() for e in trainer.environments] == [0.75, 0.85, 1.0, 0.15]
        order.append("advantages")
        (parameter**2).backward()
        order.append("backward")
        optimizer.step()
        order.append("update")

    if bad_row is None:
        train_double()
        assert order == [
            "score0",
            "score1",
            "score2",
            "score3",
            "advantages",
            "backward",
            "update",
        ]
        assert parameter.item() != 1
    else:
        with pytest.raises(hybrid.SemanticScoringError):
            train_double()
        assert parameter.grad is None and parameter.item() == 1
        pending = next(
            p
            for p in tmp_path.glob("semantic_group_*.json")
            if ".blocked." not in p.name
        )
        assert len(json.loads(pending.read_text())["rows"]) == 4
        assert list(tmp_path.glob("*.blocked.json"))


def test_new_config_validates_without_key_and_preserves_training_parameters(
    monkeypatch,
):
    from src.training.run_retail_agentic_grpo import (
        validate_config_and_split,
        validate_optimization_contract,
    )

    monkeypatch.delenv("SHADOW_JUDGE_API_KEY", raising=False)
    config = validate_config_and_split(CONFIG)["config"]
    baseline = json.loads(
        (
            ROOT
            / "configs/retail_agentic_qwen3_4b_task44_staged_v6_n4_50step_c8192_v2.json"
        ).read_text()
    )
    assert config["grpo"] == {**baseline["grpo"], "num_iterations": 1}
    assert config["model"] == baseline["model"]
    assert (
        validate_optimization_contract(config, selected_task_count=1)[
            "expected_rollouts"
        ]
        == 200
    )
    assert (
        config["reward"]["llm_judge_used"] is False
    )  # no global judge; explicit partial extraction only


def test_actual_environment_reward_chain_and_cached_native_consumption(
    tmp_path, monkeypatch
):
    from src.rl import retail_agentic_env as runtime

    runtime._ensure_tau2_importable()
    from tau2.evaluator.evaluator_env import EnvironmentEvaluator
    from tau2.evaluator.evaluator_communicate import CommunicateEvaluator

    _, raw, _, _, template = case()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    monkeypatch.setenv("POLICYAGENT_ROLLOUT_LOG", str(tmp_path / "rollouts.jsonl"))
    messages = []
    for m in raw["messages"]:
        messages.append(
            SimpleNamespace(
                **{
                    **m,
                    "tool_calls": [
                        SimpleNamespace(**c) for c in m.get("tool_calls", [])
                    ],
                }
            )
        )
    env = object.__new__(runtime.RetailAgenticEnvironment)
    env._require_ready = lambda: None
    env._messages = messages
    env._reward_config = config["reward"]
    env._last_reward_info = None
    env._require_transport_complete = False
    env._tool_iteration_limit_as_terminal_failure = False
    env._evaluator = None
    env._rollout_stage = runtime.FULL_TASK_STAGE
    env._environment_factory = None
    env._task = SimpleNamespace(
        id="44", evaluation_criteria=SimpleNamespace(communicate_info=["17.99"])
    )
    env._user_stopped = True
    env._customer_turns, env._max_customer_turns = 2, 8
    env._tool_counter, env._max_tool_calls = 3, 24
    env._policy_findings = []
    persisted, calls = [], []
    env._persist_rollout = persisted.append
    monkeypatch.setattr(
        EnvironmentEvaluator,
        "calculate_reward",
        lambda **kw: SimpleNamespace(
            reward=1.0, model_dump=lambda **kw: {"reward": 1.0}
        ),
    )
    monkeypatch.setattr(
        CommunicateEvaluator,
        "calculate_reward",
        lambda **kw: SimpleNamespace(
            communicate_checks=[SimpleNamespace(met=True)],
            model_dump=lambda **kw: {"communicate_checks": [{"met": True}]},
        ),
    )
    monkeypatch.setattr(
        runtime,
        "one_to_one_action_progress",
        lambda *args: {
            "recall": 1,
            "matches": [
                {
                    "action_id": "44_4",
                    "matched": True,
                    "matched_call_index": 2,
                    "name": "modify_pending_order_items",
                }
            ],
            "unexpected_write_count": 0,
        },
    )

    def request(packet, settings):
        calls.append(1)
        answer = copy.deepcopy(template)
        answer.update(
            trajectory_sha256=packet["trajectory_sha256"],
            candidate_table_sha256=packet["candidate_table_sha256"],
        )
        return response(answer)

    monkeypatch.setattr(hybrid, "call_extractor", request)
    result = env.get_reward()
    assert env.get_reward() == result and calls == [1] and len(persisted) == 1
    assert persisted[0]["llm_semantic_assistance_used"] is True
    assert persisted[0]["semantic_assistance"]["used_as_training_reward"] is True
    assert persisted[0]["hybrid_additive_components"]["post_write_communication"] == 0
    assert result <= 0.15  # this small fixture deliberately has no authentication


def test_known_zero_cap_does_not_block_on_score_irrelevant_unknown():
    base, raw, spec, packet, answer = case()
    base["unexpected_write_count"] = 1
    answer["candidate_results"][0].update(status="UNCERTAIN", claims=[])
    result = hybrid.score_candidate_response(
        base, raw, spec, packet, json.dumps(answer)
    )
    assert result["offline_reward"] == 0 and result["score_invariant_uncertainty"]


def test_amount_before_refund_word_is_also_required():
    c = hybrid.candidates(
        [{"role": "assistant", "content": "The $17.99 refund is complete."}]
    )
    assert c[0]["required_kinds"] == ["refund_amount"]


@pytest.mark.parametrize(
    "text",
    [
        "Let me analyze the available desk lamp variants to find the cheapest one.",
        "I will compare available product options to identify the least expensive option.",
        "Looking at the available variants for the Desk Lamp (product ID 123456), the available ones are:",
        "Available options:",
    ],
)
def test_closed_context_grammar_does_not_force_variant_assertions(text):
    row = hybrid.candidates([{"role": "assistant", "content": text}])[0]
    assert hybrid.context_only_reason(text)
    assert row["required_kinds"] == []


@pytest.mark.parametrize(
    "text",
    [
        "Let me confirm old is available.",
        "Let me analyze the available desk lamp variants to find the cheapest one, old is cheapest.",
        "Available options: old is available.",
        "old is unavailable in these options:",
        "Looking at the available variants for the Desk Lamp (item ID 123456), the available ones are:",
        "Looking at the available variants for the Desk Lamp, all are cheapest:",
        "The available desk lamp is $135.24.",
        "old is cheapest.",
    ],
)
def test_context_prefix_cannot_hide_actual_assertions(text):
    assert hybrid.context_only_reason(text) is None
    assert any(
        c["required_kinds"]
        for c in hybrid.candidates([{"role": "assistant", "content": text}])
    )


@pytest.mark.parametrize("status", ["EXTRACTED", "NOT_IN_SCOPE", "UNCERTAIN"])
def test_context_override_is_audited_without_skipping_following_false_fact(status):
    base, raw, spec, _, _ = availability_case()
    intro = "Let me analyze the available desk lamp variants to find the cheapest one."
    original = raw["messages"][4]["content"]
    raw["messages"][4]["content"] = intro + "\n" + original
    packet = hybrid.candidate_packet(raw, "policy")
    answer = {
        "trajectory_sha256": packet["trajectory_sha256"],
        "candidate_table_sha256": packet["candidate_table_sha256"],
        "authorizations": [],
        "candidate_results": [],
    }
    # Keep the original, real authorization references (substring still present).
    _, _, _, _, prior = availability_case()
    answer["authorizations"] = prior["authorizations"]
    for c in packet["candidates"]:
        if c["text"] == intro:
            claims = [{"kind": "availability", "item_id": None, "value": True}]
            row = {
                "candidate_id": c["candidate_id"],
                "status": status,
                "claims": claims if status == "EXTRACTED" else [],
            }
        else:
            claims = [
                {
                    "kind": k,
                    "item_id": "old" if k == "availability" else None,
                    "value": False if k == "availability" else "17.99",
                }
                for k in c["required_kinds"]
            ]
            row = {
                "candidate_id": c["candidate_id"],
                "status": "EXTRACTED" if claims else "NOT_IN_SCOPE",
                "claims": claims,
            }
        answer["candidate_results"].append(row)
    result = hybrid.score_candidate_response(
        base, raw, spec, packet, json.dumps(answer)
    )
    assert result["offline_reward"] == 0.85
    assert result["additive_components"]["post_write_communication"] == 0
    assert result["context_decisions"][0]["extractor_status"] == status
    assert result["unresolved_candidate_ids"] == []


def test_null_variant_on_real_assertion_still_rejected():
    _, _, _, packet, answer = availability_case()
    claim = next(
        c
        for r in answer["candidate_results"]
        for c in r["claims"]
        if c["kind"] == "availability"
    )
    claim["item_id"] = None
    with pytest.raises(ValueError):
        hybrid.validate_candidate_extraction(json.dumps(answer), packet)


def test_forged_context_candidate_cannot_override_bound_source():
    _, _, _, packet, answer = availability_case()
    packet["candidates"][0]["text"] = "Available options:"
    with pytest.raises(ValueError, match="content hash"):
        hybrid.validate_candidate_extraction(json.dumps(answer), packet)
    packet["candidate_table_sha256"] = hybrid.digest(packet["candidates"])
    answer["candidate_table_sha256"] = packet["candidate_table_sha256"]
    with pytest.raises(ValueError, match="source span"):
        hybrid.validate_candidate_extraction(json.dumps(answer), packet)


def test_scoring_inputs_preserved_before_api_failure(tmp_path):
    base, raw, spec, _, _ = case()

    def request(*args):
        saved = json.loads(
            next(tmp_path.glob("semantic_cache/*/request.json")).read_text()
        )
        assert saved["scoring_input"] == {"base": base, "raw": raw, "spec": spec}
        raise TimeoutError("mock")

    with pytest.raises(hybrid.SemanticScoringError):
        hybrid.hybrid_score(
            base, raw, spec, policy="policy", directory=tmp_path, request=request
        )
