from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import inspect
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from src.training.dirhash import directory_sha256

from src.rl.user_simulator_fail_fast import (
    SYSTEM_FAILURE_LOG_ENV,
    UserSimulatorSystemFailure,
    probe_user_simulator_api,
    sanitize_error_message,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def user_simulator_binding(model: str, raw_args: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_args)
    except json.JSONDecodeError as exc:
        raise ValueError("POLICYAGENT_USER_LLM_ARGS_JSON is invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("POLICYAGENT_USER_LLM_ARGS_JSON must be a JSON object")

    sensitive_fragments = ("key", "token", "secret", "password", "credential")

    def redact(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): (
                    "<REDACTED>"
                    if any(part in str(key).lower() for part in sensitive_fragments)
                    else redact(item)
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [redact(item) for item in value]
        return value

    safe_args = redact(parsed)
    canonical = json.dumps(
        safe_args, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "model": model,
        "llm_args": safe_args,
        "llm_args_sha256": hashlib.sha256(canonical).hexdigest().upper(),
        "seed_source": "per_opening_user_seed",
    }


def git_value(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_upstream_files(
    root: Path, expected_files: dict[str, str] | None
) -> dict[str, str]:
    verified: dict[str, str] = {}
    for relative_path, expected_sha256 in (expected_files or {}).items():
        path = (root / relative_path).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise FileNotFoundError(f"Required tau2 file missing: {relative_path}")
        actual_sha256 = sha256(path)
        if actual_sha256 != expected_sha256.upper():
            raise ValueError(f"Required tau2 file hash mismatch: {relative_path}")
        verified[relative_path] = actual_sha256
    return verified


def validate_upstream_checkout(
    expected_commit: str,
    expected_package_sha256: str | None = None,
    expected_files: dict[str, str] | None = None,
) -> dict[str, Any]:
    root_value = os.environ.get("POLICYAGENT_TAU2_ROOT")
    if not root_value:
        raise RuntimeError("Set POLICYAGENT_TAU2_ROOT to the pinned tau2 checkout")
    root = Path(root_value).expanduser().resolve()
    if (root / ".git").exists():
        result = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={root.as_posix()}",
                "rev-parse",
                "HEAD",
            ],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        actual_commit = result.stdout.strip()
        if result.returncode == 0:
            if actual_commit != expected_commit:
                raise ValueError(
                    f"tau2 checkout mismatch: {actual_commit} != {expected_commit}"
                )
            return {
                "path": str(root),
                "commit": actual_commit,
                "verification_method": "git_head",
                "required_file_sha256": _validate_upstream_files(
                    root, expected_files
                ),
            }

    marker_path = root / "PINNED_UPSTREAM_COMMIT.txt"
    transfer_manifest_path = root / "TRANSFER_MANIFEST.json"
    if not marker_path.is_file() or not transfer_manifest_path.is_file():
        raise FileNotFoundError(
            f"tau2 requires a valid Git HEAD or transfer evidence under {root}"
        )
    marker_commit = marker_path.read_text(encoding="utf-8").strip()
    transfer = load_json(transfer_manifest_path)
    if marker_commit != expected_commit or transfer.get("commit") != expected_commit:
        raise ValueError("Transferred tau2 commit binding mismatch")
    if not expected_package_sha256:
        raise ValueError("Transferred tau2 requires source_package_sha256 in config")
    archive_path = Path(str(transfer["source_package_path"])).expanduser().resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"Transferred tau2 source package missing: {archive_path}")
    actual_package_sha256 = sha256(archive_path)
    if actual_package_sha256 != expected_package_sha256:
        raise ValueError("Transferred tau2 source package hash mismatch")
    required_paths = (root / "src", root / "data" / "tau2" / "domains" / "retail")
    if any(not path.is_dir() for path in required_paths):
        raise FileNotFoundError("Transferred tau2 checkout lacks source or Retail data")
    return {
        "path": str(root),
        "commit": marker_commit,
        "verification_method": "commit_marker_and_source_package_sha256",
        "source_package_path": str(archive_path),
        "source_package_sha256": actual_package_sha256,
        "transfer_manifest_sha256": sha256(transfer_manifest_path),
        "required_file_sha256": _validate_upstream_files(root, expected_files),
    }


def validate_config_and_split(config_path: Path) -> dict[str, Any]:
    config = load_json(config_path)
    if config.get("scope") != "ISOLATED_AGENTIC_RL_ENGINEERING":
        raise ValueError("Agentic RL config scope mismatch")
    from src.training.rollout_diagnostics import validate_generation_safety

    validate_generation_safety(config)
    from src.rl.retail_agentic_env import (
        DEFAULT_REWARD_CONFIG,
        FULL_TASK_STAGE,
        is_tiered_reward_config,
        SUPPORTED_ROLLOUT_STAGES,
        TERMINAL_ONLY_REWARD_CONFIG,
    )


    reward = config["reward"]
    tiered_reward = is_tiered_reward_config(reward)
    if (
        reward not in (DEFAULT_REWARD_CONFIG, TERMINAL_ONLY_REWARD_CONFIG)
        and not tiered_reward
    ):
        raise ValueError(
            "Reward config differs from the implemented frozen specifications"
        )
    execution_mode = config.get("execution_mode", "OPTIMIZE")
    if execution_mode not in {"OPTIMIZE", "ROLLOUT_DIAGNOSTIC"}:
        raise ValueError(f"Unsupported execution_mode: {execution_mode}")
    rollout_stage = config.get("rollout", {}).get("stage", FULL_TASK_STAGE)
    if rollout_stage not in SUPPORTED_ROLLOUT_STAGES:
        raise ValueError(f"Unsupported rollout.stage: {rollout_stage}")
    if (
        reward == TERMINAL_ONLY_REWARD_CONFIG or tiered_reward
    ) and rollout_stage != FULL_TASK_STAGE:
        raise ValueError("terminal or tiered reward requires rollout.stage=FULL_TASK")
    if execution_mode == "ROLLOUT_DIAGNOSTIC":
        if float(config["grpo"]["learning_rate"]) != 0.0:
            raise ValueError("ROLLOUT_DIAGNOSTIC requires learning_rate=0")
        if float(config["grpo"]["beta"]) != 0.0:
            raise ValueError("ROLLOUT_DIAGNOSTIC requires beta=0")
        expected_rollouts = int(config["diagnostic"]["expected_rollouts"])
        actual_rollouts = int(config["grpo"]["max_steps"]) * int(
            config["grpo"]["num_generations"]
        )
        if expected_rollouts != actual_rollouts:
            raise ValueError(
                "diagnostic.expected_rollouts must equal max_steps*num_generations"
            )
        if int(config["diagnostic"]["expected_tasks"]) != int(
            config["data"]["max_tasks"]
        ):
            raise ValueError("diagnostic.expected_tasks must equal data.max_tasks")
    quantization = config.get("quantization", {"enabled": False})
    if bool(quantization.get("enabled", False)) and quantization.get(
        "mode"
    ) != "4bit_nf4":
        raise ValueError("Only 4bit_nf4 quantization is supported")

    data = config["data"]
    split_path = (REPO_ROOT / data["task_split"]).resolve()
    if not split_path.is_file():
        raise FileNotFoundError(f"Missing frozen task split: {split_path}")
    split = load_json(split_path)
    if split["upstream"]["commit"] != config["upstream"]["commit"]:
        raise ValueError("Upstream commit binding mismatch")
    if split["leakage_checks"].get("passed") is not True:
        raise ValueError("Task split leakage checks are not passing")
    sft_manifest_binding = None
    if config["model"].get("source_stage") in {"SFT", "SFT_PROTOCOL_BRIDGE"}:
        sft_manifest_binding = validate_sft_manifest_binding(config["model"], split)
        if (
            split["leakage_checks"].get(
                "rl_validation_sft_task_overlap_count"
            )
            != 0
        ):
            raise ValueError("SFT-seen task is present in RL validation")
    selected = selected_task_ids(config, split)
    validate_optimization_contract(config, selected_task_count=len(selected))
    if tiered_reward:
        spec_task_ids = set(reward["staged_reward_spec"]["tasks"])
        if set(selected) != spec_task_ids:
            raise ValueError(
                "Tiered reward task scope differs from the selected task IDs"
            )
    upstream_checkout = validate_upstream_checkout(
        config["upstream"]["commit"],
        config["upstream"].get("source_package_sha256"),
        config["upstream"].get("required_files"),
    )
    return {
        "config": config,
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "split": split,
        "split_path": str(split_path),
        "split_sha256": sha256(split_path),
        "sft_manifest_binding": sft_manifest_binding,
        "upstream_checkout": upstream_checkout,
    }


def selected_task_ids(
    config: dict[str, Any], split: dict[str, Any]
) -> list[str]:
    """Resolve an ordered diagnostic subset without depending on split ordering."""
    data = config["data"]
    subset_ids = [str(task_id) for task_id in split["splits"][data["train_subset"]]]
    configured = data.get("task_ids")
    if configured is not None:
        selected = [str(task_id) for task_id in configured]
        if not selected or len(selected) != len(set(selected)):
            raise ValueError("data.task_ids must be non-empty and unique")
        outside = sorted(set(selected) - set(subset_ids), key=int)
        if outside:
            raise ValueError(
                f"Configured task IDs are outside {data['train_subset']}: {outside}"
            )
        max_tasks = data.get("max_tasks")
        if max_tasks is not None and int(max_tasks) != len(selected):
            raise ValueError("data.max_tasks must equal len(data.task_ids)")
        return selected
    max_tasks = data.get("max_tasks")
    if max_tasks is not None:
        max_tasks = int(max_tasks)
        if max_tasks <= 0:
            raise ValueError("data.max_tasks must be positive when configured")
        subset_ids = subset_ids[:max_tasks]
    return subset_ids


def validate_sft_manifest_binding(
    model_config: dict[str, Any], split: dict[str, Any]
) -> dict[str, Any]:
    """Bind a direct or derived SFT manifest to the split's source manifest."""

    split_source_sha256 = str(
        split.get("source", {}).get("sft_data_manifest_sha256") or ""
    ).upper()
    if not split_source_sha256:
        raise ValueError("RL split has no SFT source manifest binding")

    expected_training_sha256 = str(
        model_config.get("training_data_manifest_sha256") or ""
    ).upper()
    manifest_path_value = model_config.get("training_data_manifest_path")
    if not manifest_path_value:
        if expected_training_sha256 != split_source_sha256:
            raise ValueError("RL split is not bound to the SFT training manifest")
        return {
            "binding_type": "DIRECT",
            "training_data_manifest_sha256": expected_training_sha256,
            "source_manifest_sha256": split_source_sha256,
        }

    manifest_path = (REPO_ROOT / str(manifest_path_value)).resolve()
    if not manifest_path.is_relative_to(REPO_ROOT) or not manifest_path.is_file():
        raise FileNotFoundError(f"Missing derived SFT manifest: {manifest_path}")
    actual_training_sha256 = sha256(manifest_path)
    if actual_training_sha256 != expected_training_sha256:
        raise ValueError("Derived SFT training manifest hash mismatch")
    manifest = load_json(manifest_path)
    manifest_source_sha256 = str(
        manifest.get("source", {}).get("manifest_sha256") or ""
    ).upper()
    configured_source_sha256 = str(
        model_config.get("training_data_source_manifest_sha256") or ""
    ).upper()
    if not configured_source_sha256 or configured_source_sha256 != split_source_sha256:
        raise ValueError("Derived SFT source binding differs from the RL split")
    if manifest_source_sha256 != split_source_sha256:
        raise ValueError("Derived SFT manifest does not reference the RL split source")
    return {
        "binding_type": "DERIVED",
        "training_data_manifest_path": str(manifest_path),
        "training_data_manifest_sha256": actual_training_sha256,
        "source_manifest_sha256": split_source_sha256,
    }


def validate_terminal_task_eligibility(
    task_ids: list[str], tasks: list[Any]
) -> dict[str, Any]:
    """Reject tasks whose outcome cannot be identified from final environment state."""

    from src.guards.retail_pre_action import WRITE_TOOLS

    by_id = {str(task.id): task for task in tasks}
    missing = sorted(set(task_ids) - set(by_id), key=int)
    if missing:
        raise ValueError(f"Terminal-only task IDs are missing upstream: {missing}")
    eligible: list[str] = []
    ineligible: list[str] = []
    bases_by_task: dict[str, list[str]] = {}
    expected_writes_by_task: dict[str, list[str]] = {}
    for task_id in task_ids:
        criteria = getattr(by_id[task_id], "evaluation_criteria", None)
        bases = {
            str(getattr(item, "value", item)).lower()
            for item in (getattr(criteria, "reward_basis", None) or [])
        }
        bases_by_task[task_id] = sorted(bases)
        expected_writes = sorted(
            {
                str(getattr(action, "name", ""))
                for action in (getattr(criteria, "actions", None) or [])
                if str(getattr(action, "name", "")) in WRITE_TOOLS
            }
        )
        expected_writes_by_task[task_id] = expected_writes
        if bases & {"db", "env_assertion"} and expected_writes:
            eligible.append(task_id)
        else:
            ineligible.append(task_id)
    if ineligible:
        raise ValueError(
            "Terminal-only reward requires DB or ENV_ASSERTION basis and at "
            "least one expected state-changing action; "
            f"ineligible task IDs: {ineligible}"
        )
    return {
        "status": "TERMINAL_TASKS_ELIGIBLE",
        "task_ids": eligible,
        "reward_basis_by_task": bases_by_task,
        "expected_writes_by_task": expected_writes_by_task,
    }


def validate_opening_contract(
    *,
    data: dict[str, Any],
    split_sha256: str,
    openings_path: Path,
    openings_manifest: dict[str, Any],
    openings: list[dict[str, Any]],
    expected_ordered: list[str],
    strict: bool,
) -> None:
    if openings_manifest["task_split_sha256"] != split_sha256:
        raise ValueError("Opening manifest is bound to another task split")
    if openings_manifest["output_sha256"] != sha256(openings_path):
        raise ValueError("Opening utterance hash mismatch")

    if strict:
        manifest_task_ids = [str(value) for value in openings_manifest.get("task_ids") or []]
        if manifest_task_ids != expected_ordered:
            raise ValueError("Opening manifest task_ids differ from frozen task scope")
        if int(openings_manifest.get("rows", -1)) != len(openings):
            raise ValueError("Opening manifest row count differs from opening data")
        if str(openings_manifest.get("subset")) != str(data["train_subset"]):
            raise ValueError("Opening manifest subset differs from frozen task subset")
        if str(openings_manifest.get("task_split_path")) != str(data["task_split"]):
            raise ValueError("Opening manifest task split path differs from config")
        if str(openings_manifest.get("output_path")) != str(data["openings"]):
            raise ValueError("Opening manifest output path differs from config")

    row_ids = [str(row["task_id"]) for row in openings]
    coverage_matches = (
        row_ids == expected_ordered
        if strict
        else set(row_ids) == set(expected_ordered)
    )
    if not coverage_matches:
        missing = sorted(set(expected_ordered) - set(row_ids), key=int)
        extra = sorted(set(row_ids) - set(expected_ordered), key=int)
        raise ValueError(
            f"Opening coverage mismatch: missing={missing}, extra={extra}, "
            f"ordered_rows={row_ids}"
        )
    if any(row.get("hidden_user_scenario_persisted") is not False for row in openings):
        raise ValueError("Opening data may contain hidden user scenario content")


def validate_inputs(config_path: Path, allow_dirty: bool) -> dict[str, Any]:
    validated = validate_config_and_split(config_path)
    config = validated["config"]
    split = validated["split"]
    data = config["data"]
    openings_path = (REPO_ROOT / data["openings"]).resolve()
    openings_manifest_path = (REPO_ROOT / data["openings_manifest"]).resolve()
    for path in (openings_path, openings_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing frozen Agentic RL input: {path}. Generate user openings first."
            )

    openings_manifest = load_json(openings_manifest_path)
    openings = load_jsonl(openings_path)
    expected_ordered = selected_task_ids(config, split)
    validate_opening_contract(
        data=data,
        split_sha256=validated["split_sha256"],
        openings_path=openings_path,
        openings_manifest=openings_manifest,
        openings=openings,
        expected_ordered=expected_ordered,
        strict=config.get("diagnostic", {}).get("strict_opening_manifest_binding")
        is True,
    )

    terminal_task_eligibility = None
    from src.rl.retail_agentic_env import TERMINAL_ONLY_REWARD_CONFIG

    if config["reward"] == TERMINAL_ONLY_REWARD_CONFIG:
        from tau2.registry import registry

        terminal_task_eligibility = validate_terminal_task_eligibility(
            expected_ordered,
            registry.get_tasks_loader("retail")(),
        )

    dirty = bool(git_value("status", "--porcelain"))
    if dirty and not allow_dirty:
        raise RuntimeError("Commit the run inputs before training or pass --allow-dirty")
    model_path = Path(config["model"]["name_or_path"]).expanduser()
    if not model_path.is_dir():
        raise FileNotFoundError(
            f"SFT starting model not found: {model_path}. Do not silently fall back."
        )
    actual_model_hash = directory_sha256(model_path)
    if actual_model_hash != config["model"]["expected_sha256"]:
        raise ValueError("SFT starting model hash mismatch")
    return {
        **validated,
        "openings_path": str(openings_path),
        "openings_sha256": sha256(openings_path),
        "openings_manifest_path": str(openings_manifest_path),
        "openings_manifest_sha256": sha256(openings_manifest_path),
        "openings": openings,
        "model_path": str(model_path.resolve()),
        "model_sha256": actual_model_hash,
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_branch": git_value("branch", "--show-current"),
        "git_dirty_at_start": dirty,
        "terminal_task_eligibility": terminal_task_eligibility,
    }


def check_runtime() -> dict[str, Any]:
    import torch
    import transformers
    import trl
    from trl import GRPOTrainer

    parameters = inspect.signature(GRPOTrainer.__init__).parameters
    if "environment_factory" not in parameters:
        raise RuntimeError(
            f"TRL {trl.__version__} lacks GRPO environment_factory support"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for the Agentic GRPO run")
    from packaging.version import Version

    if Version(transformers.__version__) < Version("5.2.0"):
        raise RuntimeError("Agentic tool rollout requires transformers>=5.2")
    try:
        import jmespath  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("TRL tool rollout requires jmespath") from exc
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "trl": trl.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "environment_factory_supported": True,
    }


def check_tool_template(model_path: str) -> dict[str, Any]:
    """Fail before training if the starting tokenizer cannot render tools."""

    from transformers import AutoTokenizer

    def probe_tool(value: str) -> str:
        """Return a probe value.

        Args:
            value: Probe text.

        Returns:
            The unchanged probe text.
        """

        return value

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Call the probe tool."}],
        tools=[probe_tool],
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(rendered, str) or not rendered.strip():
        raise RuntimeError("Tokenizer tool-call chat template rendered empty output")
    if "probe_tool" not in rendered or "value" not in rendered:
        raise RuntimeError("Tokenizer chat template omitted the probe tool schema")
    return {
        "tokenizer_class": type(tokenizer).__name__,
        "chat_template_present": bool(tokenizer.chat_template),
        "tool_schema_rendered": True,
        "rendered_character_count": len(rendered),
    }


def environment_only_preflight(config_path: Path) -> dict[str, Any]:
    """Exercise tau2 state reset, a real tool, and reward without GPU/API use."""

    validated = validate_config_and_split(config_path)
    from tau2.data_model.message import UserMessage

    from src.rl.retail_agentic_env import RetailAgenticEnvironment
    config = validated["config"]
    os.environ["POLICYAGENT_REWARD_CONFIG_JSON"] = json.dumps(
        config["reward"], ensure_ascii=False, sort_keys=True
    )
    os.environ["POLICYAGENT_MAX_CUSTOMER_TURNS"] = str(
        config["rollout"]["max_customer_turns"]
    )
    os.environ["POLICYAGENT_MAX_TOOL_CALLS"] = str(
        config["rollout"]["max_tool_calls"]
    )
    os.environ["POLICYAGENT_ROLLOUT_STAGE"] = str(
        config["rollout"].get("stage", "FULL_TASK")
    )

    class _PreflightUser:
        def generate_next_message(self, message, state):
            del message
            return UserMessage(role="user", content="Yes, I confirm."), state

    def user_factory(environment, task, messages, seed):
        del environment, task, messages, seed
        return _PreflightUser(), {}

    task_id = selected_task_ids(config, validated["split"])[0]
    environment = RetailAgenticEnvironment(user_factory=user_factory)
    environment.reset(
        task_id=task_id,
        initial_user_message="I need help with my order.",
        task_split="train",
        user_seed=int(config["seed"]),
    )
    product_types = json.loads(environment.list_all_product_types())
    reward = environment.get_reward()
    return {
        "status": "ENVIRONMENT_PREFLIGHT_PASSED",
        "task_id": task_id,
        "real_retail_tool_result_count": len(product_types),
        "programmatic_reward": reward,
        "reward_info": environment._last_reward_info,
        "config_sha256": validated["config_sha256"],
        "split_sha256": validated["split_sha256"],
        "upstream_checkout": validated["upstream_checkout"],
        "external_api_called": False,
        "gpu_required": False,
    }


def wrap_retail_policy_for_agentic_protocol(
    policy: str,
    rollout_stage: str = "FULL_TASK",
) -> str:
    """Wrap a frozen Retail policy with the tool-mediated dialogue contract."""

    policy = str(policy).strip()
    if not policy:
        raise ValueError("Retail policy must not be empty")
    from src.rl.retail_agentic_env import (
        FULL_TASK_STAGE,
        IDENTITY_AUTHENTICATION_STAGE,
        SUPPORTED_ROLLOUT_STAGES,
    )

    if rollout_stage not in SUPPORTED_ROLLOUT_STAGES:
        raise ValueError(f"Unsupported rollout stage: {rollout_stage}")
    stage_contract = ""
    if rollout_stage == IDENTITY_AUTHENTICATION_STAGE:
        stage_contract = (
            "\n\n<stage_contract>\n"
            "This rollout trains only the identity-authentication prefix of the "
            "task. Obtain any missing identity information from the customer via "
            "respond_to_user. Then call exactly one appropriate identity lookup "
            "tool: find_user_id_by_email or find_user_id_by_name_zip. After the "
            "tool returns a user ID, stop immediately with a short non-tool "
            "response. Do not call get_user_details, order, product, or any "
            "state-changing tool.\n"
            "</stage_contract>"
        )
    elif rollout_stage != FULL_TASK_STAGE:
        raise AssertionError("validated rollout stage was not handled")
    return (
        "You are a customer-service agent. Follow the Retail policy below. "
        "Use exactly one tool call at a time. Every customer-facing message, "
        "including clarification and confirmation, MUST be sent through the "
        "respond_to_user tool so the customer can reply. Never expose hidden task "
        "data. When the issue is resolved and the customer has stopped, finish with "
        "a short non-tool response.\n\n<policy>\n"
        + policy
        + "\n</policy>"
        + stage_contract
    )


def build_retail_system_prompt(rollout_stage: str = "FULL_TASK") -> str:
    from tau2.registry import registry

    environment = registry.get_env_constructor("retail")()
    policy = environment.get_policy()
    return wrap_retail_policy_for_agentic_protocol(policy, rollout_stage)


def build_dataset(preflight: dict[str, Any]):
    from datasets import Dataset

    rollout_stage = preflight["config"].get("rollout", {}).get(
        "stage", "FULL_TASK"
    )
    system = build_retail_system_prompt(rollout_stage)
    rows = []
    for opening in preflight["openings"]:
        rows.append(
            {
                "prompt": [
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": opening["initial_user_message"],
                    },
                ],
                "task_id": str(opening["task_id"]),
                "initial_user_message": opening["initial_user_message"],
                "task_split": "train",
                "user_seed": int(opening["user_seed"]),
            }
        )
    return Dataset.from_list(rows)


def validate_activation_precision(config: dict[str, Any]) -> bool:
    """Opt in to the audited Qwen3/NF4 activation path; never change old runs."""
    if "activation_precision" not in config:
        return False
    if config["activation_precision"] != {"mode": "qwen3_nf4_bf16_outputs_v1"}:
        raise ValueError("Unsupported activation_precision configuration")
    quantization = config.get("quantization", {})
    if (
        config.get("precision") != "bf16"
        or quantization.get("enabled") is not True
        or quantization.get("mode") != "4bit_nf4"
        or config.get("grpo", {}).get("use_vllm") is not False
    ):
        raise ValueError("BF16 activation mitigation requires BF16 Qwen3/NF4 without vLLM")
    if config.get("execution_mode", "OPTIMIZE") == "OPTIMIZE" and config.get(
        "generation_safety"
    ) != {"mode": "eos_finished_rows_v1"}:
        raise ValueError("Training precision mitigation requires the finished-row guard")
    return True


def validate_model_loading(config: dict[str, Any]) -> str | None:
    """Opt in to an explicit non-quantized BF16 loading path."""
    setting = config.get("model_loading")
    if setting is None:
        return None
    if setting not in (
        {"mode": "qwen3_bf16_lora_v1"},
        {"mode": "qwen3_bf16_inference_v1"},
    ):
        raise ValueError("Unsupported model_loading configuration")
    quantization = config.get("quantization", {})
    if (
        config.get("precision") != "bf16"
        or quantization.get("enabled") is not False
        or config.get("grpo", {}).get("use_vllm") is not False
        or config.get("generation_safety") != {"mode": "eos_finished_rows_v1"}
    ):
        raise ValueError(
            "Qwen3 BF16 loading requires non-quantized BF16, the finished-row "
            "guard, and no vLLM"
        )
    mode = setting["mode"]
    if mode == "qwen3_bf16_inference_v1" and config.get("execution_mode") != (
        "ROLLOUT_DIAGNOSTIC"
    ):
        raise ValueError("Qwen3 BF16 inference loading is diagnostic-only")
    return mode


def validate_optimization_contract(
    config: dict[str, Any], *, selected_task_count: int | None = None
) -> dict[str, Any] | None:
    """Validate the explicitly opted-in engineering-closure arithmetic."""
    if config.get("execution_mode", "OPTIMIZE") != "OPTIMIZE":
        return None
    acceptance = config.get("engineering_acceptance")
    if not isinstance(acceptance, dict):
        return None
    if acceptance.get("contract_version") != "grpo-engineering-closure-v1":
        return None

    grpo = config["grpo"]
    max_steps = int(grpo["max_steps"])
    batch_size = int(grpo["per_device_train_batch_size"])
    grad_accumulation = int(grpo["gradient_accumulation_steps"])
    num_generations = int(grpo["num_generations"])
    steps_per_generation = int(grpo["steps_per_generation"])
    if min(
        max_steps,
        batch_size,
        grad_accumulation,
        num_generations,
        steps_per_generation,
    ) <= 0:
        raise ValueError("GRPO optimization contract values must be positive")
    generation_batch_size = batch_size * steps_per_generation
    if generation_batch_size % num_generations:
        raise ValueError(
            "per_device_train_batch_size * steps_per_generation must be divisible "
            "by num_generations"
        )
    if acceptance.get("steps_per_generation_equals_gradient_accumulation_required"):
        if steps_per_generation != grad_accumulation:
            raise ValueError(
                "Engineering closure requires steps_per_generation == "
                "gradient_accumulation_steps"
            )
    if acceptance.get("one_prompt_group_per_optimizer_step"):
        if generation_batch_size != num_generations:
            raise ValueError(
                "Engineering closure requires exactly one GRPO prompt group per step"
            )
    if float(grpo["learning_rate"]) <= 0.0:
        raise ValueError("OPTIMIZE requires a positive learning rate")
    if acceptance.get("kl_reference_required") and float(grpo["beta"]) <= 0.0:
        raise ValueError("Engineering closure requires beta > 0 for KL reference")
    transport_complete_required = acceptance.get(
        "transport_complete_groups_required", False
    )
    if type(transport_complete_required) is not bool:
        raise ValueError(
            "transport_complete_groups_required must be an explicit bool"
        )

    expected_steps = int(acceptance["expected_optimizer_steps"])
    expected_rollouts = int(acceptance["expected_rollouts"])
    expected_groups = int(acceptance["expected_groups"])
    if expected_steps != max_steps:
        raise ValueError("expected_optimizer_steps differs from grpo.max_steps")
    if expected_groups != max_steps:
        raise ValueError("expected_groups differs from the one-group-per-step contract")
    if expected_rollouts != max_steps * num_generations:
        raise ValueError("expected_rollouts differs from max_steps * num_generations")
    configured_task_pool_size = int(acceptance["configured_task_pool_size"])
    task_ids = list((config.get("data") or {}).get("task_ids") or [])
    if configured_task_pool_size != len(task_ids):
        raise ValueError("configured_task_pool_size differs from data.task_ids")
    if selected_task_count is not None and configured_task_pool_size != selected_task_count:
        raise ValueError("Configured task pool differs from selected/opening task count")
    return {
        "status": "VALIDATED",
        "generation_batch_size": generation_batch_size,
        "steps_per_generation": steps_per_generation,
        "expected_optimizer_steps": expected_steps,
        "expected_groups": expected_groups,
        "expected_rollouts": expected_rollouts,
        "configured_task_pool_size": configured_task_pool_size,
        "kl_reference_required": bool(acceptance.get("kl_reference_required")),
    }


def trainable_parameter_fingerprint(model: Any) -> dict[str, Any]:
    """Hash the exact bytes of trainable tensors before and after optimization."""
    import torch

    digest = hashlib.sha256()
    tensor_count = 0
    numel = 0
    nonfinite_count = 0
    tensors: dict[str, dict[str, Any]] = {}
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        if not parameter.requires_grad:
            continue
        value = parameter.detach().contiguous()
        shape = list(value.shape)
        dtype = str(value.dtype)
        raw = value.view(torch.uint8).cpu().numpy().tobytes()
        tensor_nonfinite = int((~torch.isfinite(value)).sum().item())
        tensor_sha256 = hashlib.sha256(raw).hexdigest().upper()
        digest.update(name.encode("utf-8"))
        digest.update(dtype.encode("ascii"))
        digest.update(json.dumps(shape).encode("ascii"))
        digest.update(raw)
        nonfinite_count += tensor_nonfinite
        tensors[name] = {
            "sha256": tensor_sha256,
            "shape": shape,
            "dtype": dtype,
            "numel": value.numel(),
            "all_finite": tensor_nonfinite == 0,
            "nonfinite_count": tensor_nonfinite,
        }
        tensor_count += 1
        numel += value.numel()
    if tensor_count == 0:
        raise RuntimeError("No trainable parameters found for optimization evidence")
    return {
        "sha256": digest.hexdigest().upper(),
        "tensor_count": tensor_count,
        "numel": numel,
        "all_finite": nonfinite_count == 0,
        "nonfinite_count": nonfinite_count,
        "tensors": tensors,
    }


def compare_trainable_fingerprints(
    starting: dict[str, Any], final: dict[str, Any]
) -> dict[str, Any]:
    starting_tensors = dict(starting.get("tensors") or {})
    final_tensors = dict(final.get("tensors") or {})
    starting_names = set(starting_tensors)
    final_names = set(final_tensors)
    shared_names = sorted(starting_names & final_names)
    metadata_mismatches = [
        name
        for name in shared_names
        if (
            starting_tensors[name].get("shape"),
            starting_tensors[name].get("dtype"),
        )
        != (
            final_tensors[name].get("shape"),
            final_tensors[name].get("dtype"),
        )
    ]
    changed_names = [
        name
        for name in shared_names
        if starting_tensors[name].get("sha256")
        != final_tensors[name].get("sha256")
    ]
    return {
        "tensor_keys_match": starting_names == final_names,
        "missing_from_final": sorted(starting_names - final_names),
        "added_in_final": sorted(final_names - starting_names),
        "tensor_metadata_match": not metadata_mismatches,
        "metadata_mismatch_names": metadata_mismatches,
        "changed_tensor_count": len(changed_names),
        "changed_tensor_names": changed_names,
        "parameter_change_detected": bool(changed_names),
    }


def summarize_training_log(log_history: list[dict[str, Any]]) -> dict[str, Any]:
    """Bind actual finite optimizer diagnostics instead of inferring from config."""
    keys = ("loss", "grad_norm", "reward", "reward_std", "kl")
    values: dict[str, list[float]] = {key: [] for key in keys}
    nonfinite: list[dict[str, Any]] = []
    for row_index, row in enumerate(log_history):
        for key in keys:
            if row.get(key) is None:
                continue
            try:
                value = float(row[key])
            except (TypeError, ValueError):
                nonfinite.append(
                    {"row_index": row_index, "key": key, "value": repr(row[key])}
                )
                continue
            if not math.isfinite(value):
                nonfinite.append(
                    {"row_index": row_index, "key": key, "value": repr(value)}
                )
            else:
                values[key].append(value)
    return {
        "all_monitored_values_finite": not nonfinite,
        "nonfinite_values": nonfinite,
        "series": values,
        "finite_gradients_recorded": bool(values["grad_norm"])
        and not any(row["key"] == "grad_norm" for row in nonfinite),
        "kl_curve_available": bool(values["kl"])
        and all(math.isfinite(value) for value in values["kl"]),
    }


def adapter_weights_artifact(adapter_dir: Path) -> dict[str, Any]:
    candidates = [
        adapter_dir / "adapter_model.safetensors",
        adapter_dir / "adapter_model.bin",
    ]
    files = [path for path in candidates if path.is_file()]
    if len(files) != 1:
        raise RuntimeError(
            f"Expected exactly one adapter weights file in {adapter_dir}, found {len(files)}"
        )
    path = files[0]
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def run(preflight: dict[str, Any], output_dir: Path, *, sample_only: bool = False,
        completion_budget: int | None = None, groups_per_task: int = 1) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = deepcopy(preflight["config"])
    if config.get("sampling") and not sample_only:
        raise ValueError("Explicit sampling diagnostic config requires --sample-only")
    from src.training.rollout_diagnostics import validate_generation_safety

    guarded_generation = validate_generation_safety(config)
    bf16_activations = validate_activation_precision(config)
    model_loading_mode = validate_model_loading(config)
    optimization_contract = validate_optimization_contract(
        config, selected_task_count=len(preflight["openings"])
    )
    generation_runtime = None
    sampling_runtime = None
    sampling_contract = None
    if sample_only:
        from trl import GRPOTrainer as AuditedTrainer
        from src.training.rollout_diagnostics import verify_trl_source, validate_sampling_request

        sampling_contract = validate_sampling_request(
            config, completion_budget, groups_per_task
        )
        sampling_runtime = verify_trl_source(AuditedTrainer)
        config["grpo"]["max_completion_length"] = completion_budget
        config["sampling"] = {
            **sampling_contract,
            "configured_temperature": config["grpo"]["temperature"],
        }
        config["diagnostic"]["expected_rollouts"] = (
            len(preflight["openings"]) * groups_per_task * config["grpo"]["num_generations"]
        )
        # API calls must not precede source/budget validation.
    if guarded_generation:
        from trl import GRPOTrainer as AuditedTrainer
        from src.training import rollout_diagnostics

        generation_runtime = {
            **rollout_diagnostics.verify_trl_source(AuditedTrainer),
            "mode": config["generation_safety"]["mode"],
            "adapter_source_sha256": sha256(Path(rollout_diagnostics.__file__)),
            "row_scope": "generation_local",
        }
    os.environ["POLICYAGENT_REWARD_CONFIG_JSON"] = json.dumps(
        config["reward"], ensure_ascii=False, sort_keys=True
    )
    os.environ["POLICYAGENT_MAX_CUSTOMER_TURNS"] = str(
        config["rollout"]["max_customer_turns"]
    )
    os.environ["POLICYAGENT_MAX_TOOL_CALLS"] = str(
        config["rollout"]["max_tool_calls"]
    )
    os.environ["POLICYAGENT_ROLLOUT_STAGE"] = str(
        config["rollout"].get("stage", "FULL_TASK")
    )
    os.environ["POLICYAGENT_REQUIRE_TRANSPORT_COMPLETE_GROUPS"] = (
        "1"
        if bool(
            (config.get("engineering_acceptance") or {}).get(
                "transport_complete_groups_required", False
            )
        )
        else "0"
    )
    rollout_log = output_dir / "raw_rollouts.jsonl"
    os.environ["POLICYAGENT_ROLLOUT_LOG"] = str(rollout_log)
    rollout_evidence_log = output_dir / "rollout_evidence.jsonl"
    os.environ["POLICYAGENT_ROLLOUT_EVIDENCE_LOG"] = str(rollout_evidence_log)
    system_failure_log = output_dir / "system_failures.jsonl"
    os.environ[SYSTEM_FAILURE_LOG_ENV] = str(system_failure_log)
    generation_log = output_dir / "generation_events.jsonl"
    precision_log = output_dir / "precision_events.jsonl"
    runtime = check_runtime()
    if model_loading_mode:
        if not runtime["bf16_supported"]:
            raise RuntimeError("Qwen3 BF16 loading requires actual BF16 support")
        runtime["model_loading"] = {
            **config["model_loading"],
            "base_weight_dtype": "torch.bfloat16",
            "quantized": False,
            "peft_adapter_applied": not sample_only,
        }
    if optimization_contract is not None:
        runtime["optimization_contract"] = optimization_contract
    if bf16_activations:
        if not runtime["bf16_supported"]:
            raise RuntimeError("BF16 activation mitigation requires actual BF16 support")
        from src.training import rollout_diagnostics

        runtime["activation_precision"] = {
            **config["activation_precision"],
            "scope": "actor_and_reference_all_forwards_including_checkpoint_recompute",
            "generation_autocast": "bf16_per_generate_call",
            "source_sha256": sha256(Path(rollout_diagnostics.__file__)),
        }
    runtime["tool_template"] = check_tool_template(preflight["model_path"])
    if guarded_generation:
        runtime["generation_safety"] = generation_runtime
        save_json(output_dir / "config.json", config)
        save_json(output_dir / "command.json", {
            "executable": sys.executable,
            "argv": sys.argv,
            "config_sha256": preflight["config_sha256"],
            "runner_source_sha256": sha256(Path(__file__)),
        })
    save_json(output_dir / "environment.json", runtime)

    import torch
    from peft import LoraConfig, PeftModel
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        set_seed,
    )
    from trl import GRPOConfig, GRPOTrainer

    from src.rl.retail_agentic_env import RetailAgenticEnvironment

    grpo = config["grpo"]
    lora = config["lora"]
    execution_mode = config.get("execution_mode", "OPTIMIZE")
    optimization_enabled = execution_mode == "OPTIMIZE"
    set_seed(int(config["seed"]))
    dataset = build_dataset(preflight)
    bf16 = config["precision"] == "bf16" and runtime["bf16_supported"]
    quantization = config.get("quantization", {"enabled": False})
    model_init_kwargs: dict[str, Any] | None = None
    if bool(quantization.get("enabled", False)):
        if quantization.get("mode") != "4bit_nf4":
            raise ValueError("Only 4bit_nf4 quantization is supported")
        compute_dtype = torch.bfloat16 if bf16 else torch.float16
        model_init_kwargs = {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=bool(
                    quantization.get("double_quant", True)
                ),
            ),
            "torch_dtype": compute_dtype,
        }
    elif model_loading_mode:
        # Transformers v5 otherwise defaults a string-loaded model to FP32 in
        # GRPOTrainer, which is both a different experiment and unsafe on 24 GB.
        model_init_kwargs = {"dtype": torch.bfloat16}
    training_args_kwargs = dict(
        output_dir=str(output_dir / "trainer"),
        max_steps=int(grpo["max_steps"]),
        learning_rate=float(grpo["learning_rate"]),
        per_device_train_batch_size=int(grpo["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(grpo["gradient_accumulation_steps"]),
        steps_per_generation=int(
            grpo.get("steps_per_generation", grpo["gradient_accumulation_steps"])
        ),
        num_generations=(
            int(sampling_contract["trl_constructor_num_generations"])
            if sample_only
            else int(grpo["num_generations"])
        ),
        max_completion_length=int(grpo["max_completion_length"]),
        max_tool_calling_iterations=int(
            config["rollout"]["max_tool_calling_iterations"]
        ),
        temperature=float(grpo["temperature"]),
        beta=float(grpo["beta"]),
        loss_type=str(grpo["loss_type"]),
        use_vllm=bool(grpo["use_vllm"]),
        logging_steps=int(grpo["logging_steps"]),
        save_steps=int(grpo["save_steps"]),
        save_strategy="steps" if optimization_enabled else "no",
        save_total_limit=2,
        log_completions=bool(grpo.get("log_completions", False)),
        num_completions_to_print=int(grpo.get("num_completions_to_print", 0)),
        gradient_checkpointing=bool(grpo.get("gradient_checkpointing", False)),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        model_init_kwargs=model_init_kwargs,
        report_to="none",
        bf16=bf16,
        fp16=not bf16,
        seed=int(config["seed"]),
    )
    if sample_only:
        if sampling_contract["mode"] == "TRUE_GREEDY":
            training_args_kwargs["generation_kwargs"] = {
                "do_sample": False,
                "num_beams": 1,
                "num_return_sequences": 1,
            }
        elif sampling_contract["mode"] == "STOCHASTIC_GROUP_SAMPLING":
            if all(
                name in sampling_contract for name in ("temperature", "top_p", "top_k")
            ):
                training_args_kwargs["generation_kwargs"] = {
                    "do_sample": True,
                    "temperature": float(sampling_contract["temperature"]),
                    "top_p": float(sampling_contract["top_p"]),
                    "top_k": int(sampling_contract["top_k"]),
                    "num_beams": 1,
                    "num_return_sequences": 1,
                }
            else:
                # Preserve old sampling-only diagnostics whose decode knobs were
                # not explicitly frozen. They remain ineligible for S7 claims.
                training_args_kwargs["generation_kwargs"] = {"do_sample": True}
        else:  # pragma: no cover - validate_sampling_request owns this contract.
            raise ValueError(f"Unsupported sampling mode: {sampling_contract['mode']}")
    training_args = GRPOConfig(**training_args_kwargs)
    peft_config = None
    if not sample_only:
        peft_config = LoraConfig(
            r=int(lora["r"]),
            lora_alpha=int(lora["alpha"]),
            lora_dropout=float(lora["dropout"]),
            target_modules=list(lora["target_modules"]),
            task_type="CAUSAL_LM",
        )
    # Local imports, template and argument constructors must fail before any
    # paid API probe. Model loading still follows the probe to fail fast on auth.
    user_model = os.environ.get("POLICYAGENT_USER_MODEL", "").strip()
    simulator_binding = user_simulator_binding(
        user_model, os.environ.get("POLICYAGENT_USER_LLM_ARGS_JSON", "{}")
    )
    api_preflight = probe_user_simulator_api(model=user_model)
    save_json(output_dir / "user_simulator_preflight.json", api_preflight)
    runtime["user_simulator"] = {
        **simulator_binding,
        "preflight_status": api_preflight.get("status"),
        "external_api_called": api_preflight.get("external_api_called"),
    }
    save_json(
        output_dir / "run_state.json",
        {
            "schema_version": "retail-agentic-grpo-state-v1",
            "status": "STARTED",
            "started_at_unix": time.time(),
            "config_sha256": preflight["config_sha256"],
            "split_sha256": preflight["split_sha256"],
            "openings_sha256": preflight["openings_sha256"],
            "starting_model_sha256": preflight["model_sha256"],
        },
    )
    trainer_class = GRPOTrainer

    def emit_generation(event):
        with generation_log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")

    if sample_only:
        from src.training.rollout_diagnostics import make_sampling_trainer

        sampling_factory_kwargs = {
            "expected_do_sample": sampling_contract["do_sample"]
        }
        if bf16_activations:
            sampling_factory_kwargs["bf16_generation"] = True
        trainer_class = (
            make_sampling_trainer(GRPOTrainer, **sampling_factory_kwargs)
        )
    elif guarded_generation:
        from src.training.rollout_diagnostics import make_guarded_grpo_trainer

        trainer_class = (
            make_guarded_grpo_trainer(GRPOTrainer, emit_generation, bf16_generation=True)
            if bf16_activations else make_guarded_grpo_trainer(GRPOTrainer, emit_generation)
        )
    trainer = trainer_class(
        model=preflight["model_path"],
        args=training_args,
        train_dataset=dataset,
        reward_funcs=[],
        environment_factory=RetailAgenticEnvironment,
        peft_config=peft_config,
    )
    if sample_only:
        from src.training.rollout_diagnostics import bind_sampling_runtime

        if getattr(trainer.model, "peft_config", None) is not None:
            raise RuntimeError("Exact-checkpoint pure sampling gained a PEFT adapter")
        sampling_runtime = {
            **sampling_runtime,
            "decode_contract": bind_sampling_runtime(trainer, sampling_contract),
            "direct_merged_checkpoint_inference": peft_config is None,
        }
    starting_adapter_dir = output_dir / "agentic_grpo_starting_adapter"
    starting_trainable_fingerprint = None
    starting_adapter_weights = None
    if guarded_generation and not sample_only:
        # Preserve the actual initialized LoRA tensors, not merely a nonzero LR
        # or gradient log, so updates can be checked against the final adapter.
        trainer.save_model(str(starting_adapter_dir))
        starting_adapter_weights = adapter_weights_artifact(starting_adapter_dir)
        starting_trainable_fingerprint = trainable_parameter_fingerprint(
            trainer.model
        )
    precision_context = nullcontext()
    if bf16_activations:
        from src.training.rollout_diagnostics import qwen3_nf4_bf16_activation_context

        def emit_precision(event):
            with precision_log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")

        precision_context = qwen3_nf4_bf16_activation_context(
            (trainer.model, getattr(trainer, "ref_model", None)), emit_precision
        )
    # The activation policy must also cover old/reference log-probs and backward
    # recomputation. Do not place one long autocast context around optimizer steps.
    with precision_context:
        if sample_only:
            from src.training.rollout_diagnostics import run_pure_sampling

            sampled_result = run_pure_sampling(trainer, list(dataset), config, preflight, output_dir,
                                              groups_per_task, {**runtime, "sampling_adapter": sampling_runtime})
        else:
            result = trainer.train()
    if sample_only:
        if bf16_activations:
            # Bind only after precision-context exit; otherwise its cleanup event
            # would change a file whose earlier hash was already in the manifest.
            sampled_result["artifacts"]["precision_events.jsonl"] = {
                "path": str(precision_log), "sha256": sha256(precision_log)
            }
            save_json(output_dir / "run_manifest.json", sampled_result)
            state = load_json(output_dir / "run_state.json")
            state["run_manifest_sha256"] = sha256(output_dir / "run_manifest.json")
            save_json(output_dir / "run_state.json", state)
        return sampled_result
    log_history = list(trainer.state.log_history)
    save_json(output_dir / "train_metrics.json", result.metrics)
    save_json(output_dir / "log_history.json", log_history)
    training_log_evidence = summarize_training_log(log_history)
    optimization_evidence = None
    final_trainable_fingerprint = None
    if guarded_generation:
        final_trainable_fingerprint = trainable_parameter_fingerprint(trainer.model)
    adapter_dir = output_dir / "agentic_grpo_adapter"
    if optimization_enabled:
        trainer.save_model(str(adapter_dir))
    if guarded_generation:
        final_adapter_weights = adapter_weights_artifact(adapter_dir)
        acceptance = config["engineering_acceptance"]
        trainable_comparison = compare_trainable_fingerprints(
            starting_trainable_fingerprint, final_trainable_fingerprint
        )
        actual_rollouts = load_jsonl(rollout_log) if rollout_log.is_file() else []
        actual_task_ids = sorted(
            {str(row.get("task_id")) for row in actual_rollouts}
        )
        expected_task_ids = sorted(str(value) for value in config["data"]["task_ids"])
        actual_global_step = int(trainer.state.global_step)
        reward_stds = training_log_evidence["series"]["reward_std"]
        criteria = {
            "optimizer_steps_match": actual_global_step
            == int(acceptance["expected_optimizer_steps"]),
            "rollout_count_matches": len(actual_rollouts)
            == int(acceptance["expected_rollouts"]),
            "task_set_matches": actual_task_ids == expected_task_ids,
            "trainable_tensor_keys_match": trainable_comparison[
                "tensor_keys_match"
            ],
            "trainable_tensor_metadata_match": trainable_comparison[
                "tensor_metadata_match"
            ],
            "trainable_parameters_changed": trainable_comparison[
                "parameter_change_detected"
            ],
            "starting_trainable_parameters_finite": bool(
                starting_trainable_fingerprint["all_finite"]
            ),
            "final_trainable_parameters_finite": bool(
                final_trainable_fingerprint["all_finite"]
            ),
            "monitored_training_log_values_finite": bool(
                training_log_evidence["all_monitored_values_finite"]
            ),
            "finite_gradients_recorded": (
                not acceptance.get("finite_gradients_required")
                or training_log_evidence["finite_gradients_recorded"]
            ),
            "kl_curve_recorded": (
                not acceptance.get("kl_reference_required")
                or training_log_evidence["kl_curve_available"]
            ),
            "minimum_nonzero_reward_std_groups_met": sum(
                not math.isclose(value, 0.0, abs_tol=1e-12)
                for value in reward_stds
            )
            >= int(acceptance["minimum_nonzero_reward_std_groups"]),
        }
        optimization_evidence = {
            "schema_version": "retail-agentic-grpo-optimization-evidence-v2",
            "status": "PASSED" if all(criteria.values()) else "FAILED",
            "expected_optimizer_steps": int(
                acceptance["expected_optimizer_steps"]
            ),
            "actual_optimizer_steps": actual_global_step,
            "expected_rollouts": int(acceptance["expected_rollouts"]),
            "actual_rollouts": len(actual_rollouts),
            "expected_task_ids": expected_task_ids,
            "actual_task_ids": actual_task_ids,
            "starting_adapter_weights": starting_adapter_weights,
            "final_adapter_weights": final_adapter_weights,
            "starting_trainable_parameters": starting_trainable_fingerprint,
            "final_trainable_parameters": final_trainable_fingerprint,
            "trainable_parameter_comparison": trainable_comparison,
            "trainable_parameter_change_detected": criteria[
                "trainable_parameters_changed"
            ],
            "training_log": training_log_evidence,
            "criteria": criteria,
        }
        save_json(output_dir / "optimization_evidence.json", optimization_evidence)
    del trainer
    torch.cuda.empty_cache()

    merged_dir = output_dir / "agentic_grpo_merged"
    if optimization_enabled:
        dtype = torch.bfloat16 if bf16 else torch.float16
        base_model = AutoModelForCausalLM.from_pretrained(
            preflight["model_path"], dtype=dtype, low_cpu_mem_usage=True
        )
        merged = PeftModel.from_pretrained(
            base_model, str(adapter_dir)
        ).merge_and_unload()
        merged.save_pretrained(merged_dir, safe_serialization=True)
        tokenizer = AutoTokenizer.from_pretrained(preflight["model_path"])
        tokenizer.save_pretrained(merged_dir)

    if not rollout_log.is_file() or not rollout_log.read_text(encoding="utf-8").strip():
        raise RuntimeError("Agentic GRPO completed without a raw rollout artifact")
    if not rollout_evidence_log.is_file() or not rollout_evidence_log.read_text(
        encoding="utf-8"
    ).strip():
        raise RuntimeError("Agentic GRPO completed without rollout evidence sidecars")
    rollout_rows = len(rollout_log.read_text(encoding="utf-8").splitlines())
    evidence_rows = len(
        rollout_evidence_log.read_text(encoding="utf-8").splitlines()
    )
    if evidence_rows != rollout_rows:
        raise RuntimeError(
            f"Rollout evidence count mismatch: {evidence_rows} != {rollout_rows}"
        )
    if not optimization_enabled:
        expected_rollouts = int(config["diagnostic"]["expected_rollouts"])
        if rollout_rows != expected_rollouts:
            raise RuntimeError(
                f"Rollout count mismatch: {rollout_rows} != {expected_rollouts}"
            )
    artifacts = {
        "raw_rollouts": {
            "path": str(rollout_log),
            "sha256": sha256(rollout_log),
            "rows": rollout_rows,
        },
        "rollout_evidence": {
            "path": str(rollout_evidence_log),
            "sha256": sha256(rollout_evidence_log),
            "rows": evidence_rows,
        },
        "train_metrics": {
            "path": str(output_dir / "train_metrics.json"),
            "sha256": sha256(output_dir / "train_metrics.json"),
        },
        "log_history": {
            "path": str(output_dir / "log_history.json"),
            "sha256": sha256(output_dir / "log_history.json"),
        },
    }
    if guarded_generation:
        artifacts["starting_adapter"] = {
            "path": str(starting_adapter_dir),
            "sha256": directory_sha256(starting_adapter_dir),
        }
        for name, path in {
            "generation_events": generation_log,
            "config_snapshot": output_dir / "config.json",
            "command": output_dir / "command.json",
            "optimization_evidence": output_dir / "optimization_evidence.json",
        }.items():
            if not path.is_file():
                raise RuntimeError(f"Guarded GRPO missing required artifact: {name}")
            artifacts[name] = {"path": str(path), "sha256": sha256(path)}
    if bf16_activations:
        artifacts["precision_events"] = {
            "path": str(precision_log), "sha256": sha256(precision_log)
        }
    completion_dir = output_dir / "trainer" / "completions"
    if completion_dir.is_dir():
        completion_files = list(completion_dir.glob("*.parquet"))
        artifacts["completion_logs"] = {
            "path": str(completion_dir),
            "sha256": directory_sha256(completion_dir),
            "files": len(completion_files),
            "bytes": sum(path.stat().st_size for path in completion_files),
        }
    system_failure_rows = 0
    if system_failure_log.is_file():
        system_failure_rows = len(
            system_failure_log.read_text(encoding="utf-8").splitlines()
        )
        artifacts["system_failures"] = {
            "path": str(system_failure_log),
            "sha256": sha256(system_failure_log),
            "rows": system_failure_rows,
        }
    if optimization_enabled:
        artifacts.update(
            {
                "adapter": {
                    "path": str(adapter_dir),
                    "sha256": directory_sha256(adapter_dir),
                },
                "merged_model": {
                    "path": str(merged_dir),
                    "sha256": directory_sha256(merged_dir),
                },
            }
        )
    manifest = {
        "schema_version": "retail-agentic-grpo-run-v2",
        "scope": "ISOLATED_AGENTIC_RL_ENGINEERING",
        "status": "COMPLETED",
        "execution_mode": execution_mode,
        "optimization_enabled": optimization_enabled,
        "git": {
            "commit": preflight["git_commit"],
            "branch": preflight["git_branch"],
            "dirty_at_start": preflight["git_dirty_at_start"],
        },
        "bindings": {
            "config_path": preflight["config_path"],
            "config_sha256": preflight["config_sha256"],
            "task_split_path": preflight["split_path"],
            "task_split_sha256": preflight["split_sha256"],
            "openings_path": preflight["openings_path"],
            "openings_sha256": preflight["openings_sha256"],
            "starting_model": preflight["model_path"],
            "starting_model_sha256": preflight["model_sha256"],
            "sft_manifest_binding": preflight.get("sft_manifest_binding"),
            "upstream_checkout": preflight["upstream_checkout"],
        },
        "environment": runtime,
        "artifacts": artifacts,
        "reward": config["reward"],
        "rollout": config["rollout"],
        "quantization": quantization,
        "optimization_evidence": optimization_evidence,
        "user_simulator_preflight": api_preflight,
        "system_failure_count": system_failure_rows,
        "formal_retail_readiness_gate_opened": False,
        "business_improvement_claim_allowed": False,
    }
    save_json(output_dir / "run_manifest.json", manifest)
    save_json(
        output_dir / "run_state.json",
        {
            "schema_version": "retail-agentic-grpo-state-v1",
            "status": "COMPLETED",
            "completed_at_unix": time.time(),
            "run_manifest_sha256": sha256(output_dir / "run_manifest.json"),
        },
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "retail_agentic_grpo_v1.json",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--environment-only-preflight", action="store_true")
    parser.add_argument(
        "--user-simulator-api-preflight-only", action="store_true"
    )
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--sample-only", action="store_true",
                        help="Reuse audited TRL generation without train/loss/backward")
    parser.add_argument("--completion-budget", type=int,
                        help="Explicit cumulative budget for --sample-only; never inferred")
    parser.add_argument("--groups-per-task", type=int, default=1)
    args = parser.parse_args()
    if sum((args.sample_only, args.preflight_only, args.environment_only_preflight,
            args.user_simulator_api_preflight_only)) > 1:
        parser.error("Select only one sampling or preflight mode")
    if args.sample_only:
        if args.completion_budget is None or args.completion_budget <= 0 or args.groups_per_task <= 0:
            parser.error("--sample-only requires positive --completion-budget and --groups-per-task")
        if args.preflight_only or args.environment_only_preflight or args.user_simulator_api_preflight_only:
            parser.error("--sample-only cannot be combined with preflight modes")
    elif args.completion_budget is not None or args.groups_per_task != 1:
        parser.error("Sampling overrides require --sample-only")
    if args.environment_only_preflight:
        print(
            json.dumps(
                environment_only_preflight(args.config.resolve()),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.user_simulator_api_preflight_only:
        user_model = os.environ.get("POLICYAGENT_USER_MODEL", "").strip()
        print(
            json.dumps(
                probe_user_simulator_api(model=user_model),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    preflight = validate_inputs(args.config.resolve(), args.allow_dirty)
    if args.preflight_only:
        runtime = check_runtime()
        runtime["tool_template"] = check_tool_template(preflight["model_path"])
        print(
            json.dumps(
                {
                    "status": "GPU_PREFLIGHT_PASSED",
                    "config_sha256": preflight["config_sha256"],
                    "split_sha256": preflight["split_sha256"],
                    "openings_sha256": preflight["openings_sha256"],
                    "rows": len(preflight["openings"]),
                    "starting_model_sha256": preflight["model_sha256"],
                    "runtime": runtime,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.output_dir is None:
        parser.error("--output-dir is required unless a preflight mode is selected")
    output_dir = args.output_dir.resolve()
    # Reserve ownership outside the failure handler. A rejected old directory
    # must not receive even a failure_manifest.json from this invocation.
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        result = run(preflight, output_dir, sample_only=args.sample_only,
                     completion_budget=args.completion_budget, groups_per_task=args.groups_per_task)
    except Exception as exc:
        output_dir.mkdir(parents=True, exist_ok=True)
        is_user_simulator_failure = isinstance(exc, UserSimulatorSystemFailure)
        failure_manifest_path = output_dir / "failure_manifest.json"
        partial_artifacts = {}
        for name in (
            "raw_rollouts.jsonl",
            "rollout_evidence.jsonl",
            "generation_events.jsonl",
            "sampling_groups.jsonl",
            "effective_config.json",
            "command.json",
            "user_simulator_preflight.json",
        ):
            path = output_dir / name
            if not path.is_file():
                continue
            artifact = {
                "path": str(path),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            if path.suffix == ".jsonl":
                artifact["rows"] = sum(
                    bool(line.strip())
                    for line in path.read_text(encoding="utf-8").splitlines()
                )
            partial_artifacts[name] = artifact
        save_json(
            failure_manifest_path,
            {
                "schema_version": "retail-agentic-grpo-failure-v1",
                "scope": "ISOLATED_AGENTIC_RL_ENGINEERING",
                "status": "FAILED",
                "failure_domain": (
                    "SYSTEM" if is_user_simulator_failure else "UNCLASSIFIED_ENGINEERING"
                ),
                "failure_stage": (
                    "USER_SIMULATOR"
                    if is_user_simulator_failure
                    else "RUNNER_OR_DEPENDENCY"
                ),
                "failure_category": (
                    exc.category if is_user_simulator_failure else type(exc).__name__
                ),
                "abort_run": (
                    exc.abort_run if is_user_simulator_failure else True
                ),
                "training_eligible": False,
                "reward_eligible": False,
                "exception_type": type(exc).__name__,
                "exception_message": sanitize_error_message(str(exc)),
                "traceback": sanitize_error_message(traceback.format_exc()),
                "config_sha256": preflight["config_sha256"],
                "split_sha256": preflight["split_sha256"],
                "openings_sha256": preflight["openings_sha256"],
                "starting_model_sha256": preflight["model_sha256"],
                "partial_artifacts": partial_artifacts,
            },
        )
        save_json(
            output_dir / "run_state.json",
            {
                "schema_version": "retail-agentic-grpo-state-v1",
                "status": "FAILED",
                "failed_at_unix": time.time(),
                "failure_manifest_sha256": sha256(failure_manifest_path),
            },
        )
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
