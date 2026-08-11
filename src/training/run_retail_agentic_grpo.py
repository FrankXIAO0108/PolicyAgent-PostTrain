from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest().upper()


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
    from src.rl.retail_agentic_env import DEFAULT_REWARD_CONFIG

    reward = config["reward"]
    if reward != DEFAULT_REWARD_CONFIG:
        raise ValueError(
            "Frozen Agentic RL v1 reward config differs from the implemented spec"
        )
    execution_mode = config.get("execution_mode", "OPTIMIZE")
    if execution_mode not in {"OPTIMIZE", "ROLLOUT_DIAGNOSTIC"}:
        raise ValueError(f"Unsupported execution_mode: {execution_mode}")
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
        "upstream_checkout": upstream_checkout,
    }


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
    split_path = Path(validated["split_path"])
    if openings_manifest["task_split_sha256"] != validated["split_sha256"]:
        raise ValueError("Opening manifest is bound to another task split")
    if openings_manifest["output_sha256"] != sha256(openings_path):
        raise ValueError("Opening utterance hash mismatch")

    subset = data["train_subset"]
    expected_ordered = list(split["splits"][subset])
    max_tasks = data.get("max_tasks")
    if max_tasks is not None:
        max_tasks = int(max_tasks)
        if max_tasks <= 0:
            raise ValueError("data.max_tasks must be positive when configured")
        expected_ordered = expected_ordered[:max_tasks]
    expected_ids = set(expected_ordered)
    row_ids = {str(row["task_id"]) for row in openings}
    if row_ids != expected_ids:
        raise ValueError(
            f"Opening coverage mismatch: missing={sorted(expected_ids-row_ids)}, "
            f"extra={sorted(row_ids-expected_ids)}"
        )
    if any(row.get("hidden_user_scenario_persisted") is not False for row in openings):
        raise ValueError("Opening data may contain hidden user scenario content")

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

    class _PreflightUser:
        def generate_next_message(self, message, state):
            del message
            return UserMessage(role="user", content="Yes, I confirm."), state

    def user_factory(environment, task, messages, seed):
        del environment, task, messages, seed
        return _PreflightUser(), {}

    task_id = str(validated["split"]["splits"]["rl_train"][0])
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


def build_retail_system_prompt() -> str:
    from tau2.registry import registry

    environment = registry.get_env_constructor("retail")()
    policy = environment.get_policy()
    return (
        "You are a customer-service agent. Follow the Retail policy below. "
        "Use exactly one tool call at a time. Every customer-facing message, "
        "including clarification and confirmation, MUST be sent through the "
        "respond_to_user tool so the customer can reply. Never expose hidden task "
        "data. When the issue is resolved and the customer has stopped, finish with "
        "a short non-tool response.\n\n<policy>\n"
        + policy
        + "\n</policy>"
    )


def build_dataset(preflight: dict[str, Any]):
    from datasets import Dataset

    system = build_retail_system_prompt()
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


def run(preflight: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = preflight["config"]
    os.environ["POLICYAGENT_REWARD_CONFIG_JSON"] = json.dumps(
        config["reward"], ensure_ascii=False, sort_keys=True
    )
    os.environ["POLICYAGENT_MAX_CUSTOMER_TURNS"] = str(
        config["rollout"]["max_customer_turns"]
    )
    os.environ["POLICYAGENT_MAX_TOOL_CALLS"] = str(
        config["rollout"]["max_tool_calls"]
    )
    rollout_log = output_dir / "raw_rollouts.jsonl"
    os.environ["POLICYAGENT_ROLLOUT_LOG"] = str(rollout_log)
    runtime = check_runtime()
    runtime["tool_template"] = check_tool_template(preflight["model_path"])
    save_json(output_dir / "environment.json", runtime)
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
    training_args = GRPOConfig(
        output_dir=str(output_dir / "trainer"),
        max_steps=int(grpo["max_steps"]),
        learning_rate=float(grpo["learning_rate"]),
        per_device_train_batch_size=int(grpo["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(grpo["gradient_accumulation_steps"]),
        num_generations=int(grpo["num_generations"]),
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
    peft_config = LoraConfig(
        r=int(lora["r"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        target_modules=list(lora["target_modules"]),
        task_type="CAUSAL_LM",
    )
    trainer = GRPOTrainer(
        model=preflight["model_path"],
        args=training_args,
        train_dataset=dataset,
        reward_funcs=[],
        environment_factory=RetailAgenticEnvironment,
        peft_config=peft_config,
    )
    result = trainer.train()
    save_json(output_dir / "train_metrics.json", result.metrics)
    save_json(output_dir / "log_history.json", trainer.state.log_history)
    adapter_dir = output_dir / "agentic_grpo_adapter"
    if optimization_enabled:
        trainer.save_model(str(adapter_dir))
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
    rollout_rows = len(rollout_log.read_text(encoding="utf-8").splitlines())
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
        "train_metrics": {
            "path": str(output_dir / "train_metrics.json"),
            "sha256": sha256(output_dir / "train_metrics.json"),
        },
        "log_history": {
            "path": str(output_dir / "log_history.json"),
            "sha256": sha256(output_dir / "log_history.json"),
        },
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
            "upstream_checkout": preflight["upstream_checkout"],
        },
        "environment": runtime,
        "artifacts": artifacts,
        "reward": config["reward"],
        "rollout": config["rollout"],
        "quantization": quantization,
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
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    if args.environment_only_preflight:
        print(
            json.dumps(
                environment_only_preflight(args.config.resolve()),
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
    try:
        result = run(preflight, output_dir)
    except Exception as exc:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_json(
            output_dir / "failure_manifest.json",
            {
                "schema_version": "retail-agentic-grpo-failure-v1",
                "scope": "ISOLATED_AGENTIC_RL_ENGINEERING",
                "status": "FAILED",
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(),
                "config_sha256": preflight["config_sha256"],
                "split_sha256": preflight["split_sha256"],
                "openings_sha256": preflight["openings_sha256"],
                "starting_model_sha256": preflight["model_sha256"],
            },
        )
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
