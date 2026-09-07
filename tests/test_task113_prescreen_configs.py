from pathlib import Path
import json

from src.training.rollout_diagnostics import validate_sampling_request
from src.training.run_retail_agentic_grpo import validate_config_and_split


PROJECT = Path(__file__).resolve().parents[1]
GREEDY = PROJECT / "configs" / "retail_agentic_qwen3_4b_task113_greedy_prescreen_v1.json"
PASSK = PROJECT / "configs" / "retail_agentic_qwen3_4b_task113_passk_n4_prescreen_v1.json"
ONLINE = (
    PROJECT
    / "configs"
    / "retail_agentic_qwen3_4b_task113_staged_claim_v7_grpo_10step_n4_v1.json"
)
ONLINE_OPENING = (
    PROJECT
    / "data"
    / "retail_agentic_rl_v2"
    / "task113_frozen_opening_from_prescreen_v1.jsonl"
)
PAIRED_SFT = (
    PROJECT
    / "configs"
    / "retail_agentic_qwen3_4b_task113_frozen_sft_post_eval_n4_v1.json"
)
PAIRED_GRPO = (
    PROJECT
    / "configs"
    / "retail_agentic_qwen3_4b_task113_frozen_grpo_post_eval_n4_v1.json"
)
UPSTREAM_GUIDELINES_SHA256 = (
    "740A29DFA64D7BC08EEA3BF7493575B914A63F744ACBAF7F199EE07EDDAF72D3"
)


def test_task113_prescreen_configs_are_frozen_no_update_contracts() -> None:
    greedy = validate_config_and_split(GREEDY)["config"]
    passk = validate_config_and_split(PASSK)["config"]

    for config in (greedy, passk):
        assert config["execution_mode"] == "ROLLOUT_DIAGNOSTIC"
        assert config["data"]["train_subset"] == "rl_train"
        assert config["data"]["task_ids"] == ["113"]
        assert config["claims"]["sft_task_unseen_according_to_bound_split"] is True
        assert config["grpo"]["learning_rate"] == 0
        assert config["grpo"]["beta"] == 0
        assert config["upstream"]["required_files"] == {
            "data/tau2/user_simulator/simulation_guidelines.md": (
                UPSTREAM_GUIDELINES_SHA256
            )
        }
        assert config["model"] == greedy["model"]
        assert config["reward"] == greedy["reward"]
        assert config["rollout"] == greedy["rollout"]


def test_task113_sampling_contracts_are_exact_greedy_and_n4() -> None:
    greedy = validate_config_and_split(GREEDY)["config"]
    passk = validate_config_and_split(PASSK)["config"]

    greedy_contract = validate_sampling_request(greedy, 8192, 1)
    passk_contract = validate_sampling_request(passk, 8192, 1)

    assert greedy_contract["mode"] == "TRUE_GREEDY"
    assert greedy_contract["actual_num_generations"] == 1
    assert passk_contract["mode"] == "STOCHASTIC_GROUP_SAMPLING"
    assert passk_contract["actual_num_generations"] == 4


def test_task113_online_config_is_ten_step_claim_v7_contract() -> None:
    result = validate_config_and_split(ONLINE)
    config = result["config"]

    assert config["execution_mode"] == "OPTIMIZE"
    assert config["data"]["task_ids"] == ["113"]
    assert config["grpo"]["max_steps"] == 10
    assert config["grpo"]["num_generations"] == 4
    assert config["grpo"]["beta"] == 0.02
    assert config["engineering_acceptance"]["expected_rollouts"] == 40
    assert config["engineering_acceptance"]["expected_groups"] == 10
    staged = config["reward"]["staged_reward_spec"]
    assert staged["reward"]["composition_mode"] == (
        "hierarchical_state_authorization_claim_v7"
    )
    assert staged["tasks"]["113"]["claim_evidence_rules"]


def test_task113_recovered_opening_matches_frozen_prescreen_message() -> None:
    row = json.loads(ONLINE_OPENING.read_text(encoding="utf-8"))

    assert row["task_id"] == "113"
    assert row["user_seed"] == 20260902
    assert row["initial_user_message"] == (
        "Hi, I need some help with my account. I’d like to cancel all my pending "
        "orders, please."
    )
    assert row["hidden_user_scenario_persisted"] is False


def test_task113_paired_eval_changes_only_checkpoint_binding() -> None:
    sft = json.loads(PAIRED_SFT.read_text(encoding="utf-8"))
    grpo = json.loads(PAIRED_GRPO.read_text(encoding="utf-8"))

    assert sft["claims"]["paired_eval_contract"] == (
        "task113-sft-vs-staged-v7-n4-v1"
    )
    assert grpo["claims"]["paired_eval_contract"] == (
        "task113-sft-vs-staged-v7-n4-v1"
    )
    assert sft["claims"]["paired_eval_arm"] == "SFT"
    assert grpo["claims"]["paired_eval_arm"] == "GRPO"

    sft_comparable = dict(sft)
    grpo_comparable = dict(grpo)
    sft_comparable["model"] = None
    grpo_comparable["model"] = None
    sft_comparable["claims"] = dict(sft_comparable["claims"])
    grpo_comparable["claims"] = dict(grpo_comparable["claims"])
    sft_comparable["claims"]["paired_eval_arm"] = None
    grpo_comparable["claims"]["paired_eval_arm"] = None

    assert sft_comparable == grpo_comparable
    assert sft["model"]["expected_sha256"] == (
        "0A2E06C9BCA6082F3FE6723EC54A46D4CE1D37A8C6DD6B9BC116BBA4BAB16576"
    )
    assert grpo["model"]["expected_sha256"] == (
        "D1DF1E0E3ABDDEB716455EAADDB77BBDBD1543A65BA249C179D5C58B85D84ABE"
    )
    assert sft["sampling"] == grpo["sampling"]
    assert sft["reward"] == grpo["reward"]
    assert sft["data"] == grpo["data"]
