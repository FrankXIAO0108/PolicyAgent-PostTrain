"""Version-bound generation protection and pure sampling around native TRL.

No third-party loop is copied or patched. A private control-flow exception stops
_generate_and_score_completions immediately AFTER _generate and BEFORE scoring
forward passes. EOS-complete rows receive a terminal PAD distribution; active
rows retain strict numerical checks. Only the audited text/sync runtime is supported.
"""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
import base64
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
from threading import Lock
from typing import Any
from weakref import WeakSet


TRL_VERSION = "1.9.0"
TRANSFORMERS_VERSION = "5.14.1"
TRL_SOURCE_SHA256 = "655D9AD98549290FD32381174BFCA5CAB9F849B839966EDC9A4D3ABE67DD97BC"
TRANSFORMERS_SOURCE_SHA256 = (
    "EC9E5BCE8B654D5EF8169DDF595BE18A06A95C87568FCD909D37D88255FFB8F7"
)
_SAMPLING_OWNERS = WeakSet()
_SAMPLING_OWNERS_LOCK = Lock()
_PRECISION_OWNERS = WeakSet()
_PRECISION_OWNERS_LOCK = Lock()


@contextmanager
def _exclusive_sampling_model(owner):
    """Reserve a model/trainer identity before mutating per-generation state."""
    with _SAMPLING_OWNERS_LOCK:
        if owner in _SAMPLING_OWNERS:
            raise RuntimeError(
                "Concurrent or nested sampling on the same object is forbidden"
            )
        _SAMPLING_OWNERS.add(owner)
    try:
        yield
    finally:
        with _SAMPLING_OWNERS_LOCK:
            _SAMPLING_OWNERS.discard(owner)


@contextmanager
def qwen3_nf4_bf16_activation_context(models, emit):
    """Explicit whole-training activation policy, not an inference-only patch.

    The caller must keep this context around generation, old/current/reference
    scoring AND backward (including checkpoint recomputation), and supply any
    separate reference model. Optional absent references are ignored. NF4 and
    mixed-precision configuration remain the runner's explicit responsibility.
    This context changes only embedding/norm output tensors; it does not change
    parameters, grad mode, train/eval mode, autocast, or the native training loop.
    """
    import torch

    cores = []
    for model in models:
        if model is None:
            continue
        seen = set()
        while True:
            if id(model) in seen:
                raise ValueError("Cyclic training precision model wrapper")
            seen.add(id(model))
            wrapped = getattr(model, "module", None)
            if isinstance(wrapped, torch.nn.Module):
                model = wrapped
                continue
            getter = getattr(model, "get_base_model", None)
            base = getter() if callable(getter) else model
            if base is model:
                break
            model = base
        if (
            not isinstance(model, torch.nn.Module)
            or getattr(getattr(model, "config", None), "model_type", None) != "qwen3"
            or getattr(model, "is_loaded_in_4bit", False) is not True
        ):
            raise ValueError("BF16 activation policy requires Qwen3 loaded in 4bit")
        if all(core is not model for core in cores):
            cores.append(model)
    if not cores:
        raise ValueError("Training precision requires at least one model")

    targets = []
    for core in cores:
        embeddings = [m for m in core.modules() if isinstance(m, torch.nn.Embedding)]
        norms = [m for m in core.modules() if type(m).__name__ == "Qwen3RMSNorm"]
        if not embeddings or not norms:
            raise ValueError("Qwen3 embedding/RMSNorm precision targets are missing")
        for module in embeddings + norms:
            if all(target is not module for target in targets):
                targets.append(module)
    owners = cores + [m for m in targets if all(m is not c for c in cores)]
    # Separate from generation ownership: guarded generation must be permitted
    # inside this lifetime. Reserve ALL cores/targets atomically before mutation.
    with _PRECISION_OWNERS_LOCK:
        if any(owner in _PRECISION_OWNERS for owner in owners):
            raise RuntimeError("Concurrent or nested training precision is forbidden")
        for owner in owners:
            _PRECISION_OWNERS.add(owner)

    details = {
        "mode": "qwen3_nf4_bf16_outputs_v1",
        "scope": "whole_training_including_reference_and_checkpoint_recomputation",
        "core_count": len(cores),
        "embedding_modules": sum(isinstance(m, torch.nn.Embedding) for m in targets),
        "rmsnorm_modules": sum(type(m).__name__ == "Qwen3RMSNorm" for m in targets),
        "parameters_modified": False,
        "grad_mode_modified": False,
    }
    handles = []
    failure = None

    def cast_output(module, arguments, output):
        if not isinstance(output, torch.Tensor) or not output.is_floating_point():
            raise TypeError("Unexpected Qwen3 embedding/RMSNorm output contract")
        return output.to(dtype=torch.bfloat16)

    try:
        for target in targets:
            handles.append(target.register_forward_hook(cast_output))
        emit({"event": "training_precision_enter", **details, "hooks": len(handles)})
        yield details
    except BaseException as exc:
        failure = exc
        raise
    finally:
        try:
            for handle in reversed(handles):
                handle.remove()
        finally:
            with _PRECISION_OWNERS_LOCK:
                for owner in owners:
                    _PRECISION_OWNERS.discard(owner)
        try:
            emit(
                {
                    "event": "training_precision_exit",
                    **details,
                    "hooks_removed": len(handles),
                    "status": "ERROR" if failure is not None else "COMPLETED",
                }
            )
        except Exception as logging_error:
            if failure is None:
                raise
            failure.add_note(
                f"Training precision exit logging unavailable: {type(logging_error).__name__}"
            )


def verify_trl_source(trainer_class: type) -> dict[str, str]:
    import trl

    path = Path(inspect.getfile(trainer_class))
    digest = hashlib.sha256(path.read_bytes()).hexdigest().upper()
    if trl.__version__ != TRL_VERSION or digest != TRL_SOURCE_SHA256:
        raise RuntimeError(
            "Generation protection requires the audited TRL version AND source hash"
        )
    import transformers

    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise RuntimeError(
            "Generation protection requires the audited transformers parser version"
        )
    from transformers.generation.utils import GenerationMixin

    generation_path = Path(inspect.getfile(GenerationMixin))
    generation_digest = hashlib.sha256(generation_path.read_bytes()).hexdigest().upper()
    if generation_digest != TRANSFORMERS_SOURCE_SHA256:
        raise RuntimeError(
            "Generation protection requires the audited transformers source hash"
        )
    return {
        "version": trl.__version__,
        "source_sha256": digest,
        "transformers_version": transformers.__version__,
        "transformers_source_sha256": generation_digest,
    }


def validate_sampling_request(
    config: dict, budget: int | None, groups: int
) -> dict[str, Any]:
    """Validate and return the effective pure-sampling contract.

    A missing sampling block preserves the historical two-candidate stochastic
    diagnostic. True greedy is opt-in only: an inert/near-zero temperature is
    never accepted as evidence that sampling was disabled.
    """
    if config.get("execution_mode") != "ROLLOUT_DIAGNOSTIC":
        raise ValueError("Pure sampling requires ROLLOUT_DIAGNOSTIC")
    if config["grpo"]["learning_rate"] != 0 or config["grpo"]["beta"] != 0:
        raise ValueError("Pure sampling must not use optimization or KL-loss settings")
    if type(budget) is not int or budget <= 0 or type(groups) is not int or groups <= 0:
        raise ValueError("Explicit positive budget and group count required")
    if config["grpo"]["use_vllm"]:
        raise ValueError("Pure sampling supports only the transformers backend")
    configured_task_ids = config.get("data", {}).get("task_ids")
    if (
        not isinstance(configured_task_ids, list)
        or not configured_task_ids
        or len({str(task_id) for task_id in configured_task_ids})
        != len(configured_task_ids)
    ):
        raise ValueError("Pure sampling requires a non-empty unique frozen task scope")
    normalized_task_ids = [str(task_id) for task_id in configured_task_ids]
    configured_max_tasks = config.get("data", {}).get("max_tasks")
    if configured_max_tasks is not None and int(configured_max_tasks) != len(
        configured_task_ids
    ):
        raise ValueError("Pure sampling task scope differs from data.max_tasks")
    legacy_task_scope = set(normalized_task_ids) == {"43", "72"} and len(
        normalized_task_ids
    ) == 2
    if not legacy_task_scope:
        if config.get("diagnostic", {}).get("strict_opening_manifest_binding") is not True:
            raise ValueError(
                "Non-legacy pure sampling requires strict opening manifest binding"
            )
        sampling_subset = config.get("data", {}).get("train_subset")
        if sampling_subset not in {"development_audit", "rl_train"}:
            raise ValueError(
                "Non-legacy pure sampling remains restricted to "
                "development_audit or explicitly bound rl_train"
            )
        if (
            sampling_subset == "rl_train"
            and config.get("claims", {}).get("rl_train_task_only") is not True
        ):
            raise ValueError(
                "rl_train pure sampling requires claims.rl_train_task_only=true"
            )
    if config["rollout"]["stage"] != "FULL_TASK":
        raise ValueError("Sampling requires the full-task protocol")
    requested = config.get("sampling")
    if requested is not None:
        diagnostic = config.get("diagnostic", {})
        num_generations = int(config["grpo"]["num_generations"])
        expected_per_task = groups * num_generations
        expected_total = len(normalized_task_ids) * expected_per_task
        bindings = {
            "expected_tasks": len(normalized_task_ids),
            "expected_rollouts_per_task": expected_per_task,
            "expected_rollouts": expected_total,
        }
        for name, expected in bindings.items():
            if int(diagnostic.get(name, -1)) != expected:
                raise ValueError(
                    f"Pure sampling diagnostic.{name} differs from frozen task shape"
                )
        optional_bindings = {
            "groups_per_task": groups,
            "group_size": num_generations,
        }
        for name, expected in optional_bindings.items():
            if name in diagnostic and int(diagnostic[name]) != expected:
                raise ValueError(
                    f"Pure sampling diagnostic.{name} differs from frozen task shape"
                )
    if requested is None:
        if config["grpo"]["num_generations"] != 2:
            raise ValueError(
                "Historical stochastic sampling requires exactly two candidates"
            )
        return {
            "mode": "STOCHASTIC_GROUP_SAMPLING",
            "do_sample": True,
            "actual_num_generations": 2,
            "trl_constructor_num_generations": 2,
            "groups_per_task": groups,
            "trainer_max_steps_unused": True,
        }
    stochastic_request = {
        "mode": "STOCHASTIC_GROUP_SAMPLING",
        "do_sample": True,
        "temperature": 0.8,
        "top_p": 1.0,
        "top_k": 0,
    }
    adaptive_request = {
        **stochastic_request,
        "contract_version": "adaptive-quality-gate-v1",
        "sample_schedule": [4, 6, 8],
        "minimum_quality_positive": 1,
    }
    fixed_n4_request = {
        **stochastic_request,
        "contract_version": "fixed-n4-single-group-v1",
    }
    if requested == fixed_n4_request:
        if int(config["grpo"]["num_generations"]) != 4:
            raise ValueError("Fixed n4 sampling requires exactly four candidates")
        if config["grpo"].get("temperature") != requested["temperature"]:
            raise ValueError(
                "Fixed n4 sampling temperature must match grpo.temperature"
            )
        if budget != int(config["grpo"]["max_completion_length"]):
            raise ValueError(
                "Fixed n4 sampling completion budget must match the frozen config"
            )
        if groups != 1:
            raise ValueError("Fixed n4 sampling requires one same-prompt group")
        return {
            **stochastic_request,
            "contract_version": requested["contract_version"],
            "actual_num_generations": 4,
            "trl_constructor_num_generations": 4,
            "groups_per_task": 1,
            "trainer_max_steps_unused": True,
        }
    if requested == adaptive_request:
        num_generations = int(config["grpo"]["num_generations"])
        if num_generations not in requested["sample_schedule"]:
            raise ValueError(
                "Adaptive quality sampling num_generations must be one of 4, 6, or 8"
            )
        if config["grpo"].get("temperature") != requested["temperature"]:
            raise ValueError(
                "Adaptive quality sampling temperature must match grpo.temperature"
            )
        if budget != int(config["grpo"]["max_completion_length"]):
            raise ValueError(
                "Adaptive quality sampling completion budget must match the frozen config"
            )
        if groups != 1:
            raise ValueError("Adaptive quality sampling requires one same-prompt group")
        return {
            **stochastic_request,
            "contract_version": requested["contract_version"],
            "sample_schedule": list(requested["sample_schedule"]),
            "minimum_quality_positive": requested["minimum_quality_positive"],
            "actual_num_generations": num_generations,
            "trl_constructor_num_generations": num_generations,
            "groups_per_task": 1,
            "trainer_max_steps_unused": True,
        }
    if requested == stochastic_request and all(
        type(requested[key]) is expected_type
        for key, expected_type in (
            ("mode", str),
            ("do_sample", bool),
            ("temperature", float),
            ("top_p", float),
            ("top_k", int),
        )
    ):
        if config["grpo"]["num_generations"] != 2:
            raise ValueError(
                "S7 stochastic sampling requires exactly two candidates per group"
            )
        if config["grpo"].get("temperature") != requested["temperature"]:
            raise ValueError(
                "S7 stochastic sampling temperature must match grpo.temperature"
            )
        if budget != int(config["grpo"]["max_completion_length"]):
            raise ValueError(
                "S7 stochastic completion budget must match the frozen config exactly"
            )
        if groups != 2:
            raise ValueError("S7 stochastic pass@k requires two groups per task")
        return {
            **stochastic_request,
            "actual_num_generations": 2,
            "trl_constructor_num_generations": 2,
            "groups_per_task": 2,
            "trainer_max_steps_unused": True,
        }
    if requested != {"mode": "TRUE_GREEDY", "do_sample": False}:
        raise ValueError(
            "Sampling must be the exact S7 stochastic contract or "
            "sampling.mode=TRUE_GREEDY with sampling.do_sample=false"
        )
    if config["grpo"]["num_generations"] != 1:
        raise ValueError("True greedy pass@1 requires exactly one generated trajectory")
    if budget != int(config["grpo"]["max_completion_length"]):
        raise ValueError(
            "S6 true greedy completion budget must match the frozen config exactly"
        )
    if groups != 1:
        raise ValueError("S6 true greedy baseline requires one trajectory per task")
    return {
        "mode": "TRUE_GREEDY",
        "do_sample": False,
        "actual_num_generations": 1,
        # GRPOConfig rejects num_generations < 2 even though this diagnostic
        # exits before advantages, scoring, loss or backward. The constructor
        # placeholder is replaced on the sampling-only trainer before rollout.
        # TRL 1.9 also requires the constructor generation batch to be divisible
        # by that placeholder, so use two constructor-only generation steps.
        "trl_constructor_num_generations": 2,
        "trl_constructor_steps_per_generation": 2,
        "groups_per_task": 1,
        "trainer_max_steps_unused": True,
    }


def bind_sampling_runtime(trainer, contract: dict[str, Any]) -> dict[str, Any]:
    """Fail closed unless the instantiated trainer matches the decode contract."""

    expected_do_sample = contract["do_sample"]
    generation_config = getattr(trainer, "generation_config", None)
    generation_kwargs = getattr(trainer, "generation_kwargs", None)
    if generation_config is None or not isinstance(generation_kwargs, dict):
        raise RuntimeError("Sampling trainer has no auditable generation configuration")
    if getattr(generation_config, "do_sample", None) is not expected_do_sample:
        raise RuntimeError("Effective GenerationConfig does not match sampling contract")
    if generation_kwargs.get("do_sample") is not expected_do_sample:
        raise RuntimeError("Effective generation kwargs do not match sampling contract")
    if contract["mode"] == "STOCHASTIC_GROUP_SAMPLING" and all(
        key in contract for key in ("temperature", "top_p", "top_k")
    ):
        for name in ("temperature", "top_p", "top_k"):
            expected = contract[name]
            if getattr(generation_config, name, None) != expected:
                raise RuntimeError(
                    f"Effective GenerationConfig {name} does not match sampling contract"
                )
            if generation_kwargs.get(name) != expected:
                raise RuntimeError(
                    f"Effective generation kwargs {name} does not match sampling contract"
                )
        trainer_args = getattr(trainer, "args", None)
        if (
            getattr(trainer, "num_generations", None)
            != contract["trl_constructor_num_generations"]
            or (
                trainer_args is not None
                and getattr(trainer_args, "num_generations", None)
                != contract["trl_constructor_num_generations"]
            )
        ):
            raise RuntimeError("Stochastic generation count does not match sampling contract")
    if contract["mode"] == "TRUE_GREEDY":
        if (
            generation_kwargs.get("num_beams") != 1
            or generation_kwargs.get("num_return_sequences") != 1
        ):
            raise RuntimeError("True greedy requires one beam and one returned sequence")
        if getattr(trainer, "num_generations", None) != 2:
            raise RuntimeError("Unexpected TRL constructor generation placeholder")
        trainer.num_generations = 1
        trainer_args = getattr(trainer, "args", None)
        if trainer_args is not None:
            if getattr(trainer_args, "num_generations", None) != 2:
                raise RuntimeError("Unexpected trainer-args generation placeholder")
            trainer_args.num_generations = 1
    if getattr(trainer, "num_generations", None) != contract["actual_num_generations"]:
        raise RuntimeError("Effective rollout count does not match sampling contract")
    return {
        **contract,
        "effective_do_sample": getattr(generation_config, "do_sample"),
        "effective_num_generations": trainer.num_generations,
        **(
            {
                f"effective_{name}": getattr(generation_config, name)
                for name in ("temperature", "top_p", "top_k")
            }
            if contract["mode"] == "STOCHASTIC_GROUP_SAMPLING"
            and all(name in contract for name in ("temperature", "top_p", "top_k"))
            else {}
        ),
        "temperature_is_decode_authority": bool(expected_do_sample),
    }


class _Generated(BaseException):
    """Not an error: never let the parent continue into logprob/loss computation."""

    def __init__(self, output: tuple):
        self.output = output


class _FatalToolError(BaseException):
    """Escape TRL's catch-Exception tool handler for infrastructure failures."""

    def __init__(self, cause: Exception):
        self.cause = cause


class BudgetTrace:
    """Exact token accounting; indices are explicit (identical prompts are safe)."""

    def __init__(self, budget: int, context_limit: int, emit):
        if budget <= 0 or context_limit <= 0:
            raise ValueError("Token limits must be positive")
        self.budget = budget
        self.context_limit = context_limit
        self.emit = emit
        self.rows: list[dict[str, Any]] = []
        self.suffix_queue: deque[int] = deque()
        self.next_generation: list[tuple[int, list[int]]] = []

    def start(self, prompts: list[list[int]]) -> None:
        if self.rows or not prompts:
            raise ValueError("Trace must start exactly once with a non-empty group")
        # TRL generates up to B tokens each turn BEFORE clipping to the remaining
        # budget. Conservatively reserve that transient generation headroom too.
        if any(len(p) + 2 * self.budget > self.context_limit for p in prompts):
            raise ValueError(
                "Context cannot hold prompt + cumulative budget + generation headroom"
            )
        self.rows = [
            {
                "prompt_ids": list(p),
                "retained": [],
                "flags": [],
                "tool_batches": 0,
                "generated_tokens": 0,
                "observations": 0,
            }
            for p in prompts
        ]

    def flag(self, index: int, code: str) -> None:
        if code not in self.rows[index]["flags"]:
            self.rows[index]["flags"].append(code)

    def tool_called(self, index: int) -> None:
        if not self.suffix_queue or self.suffix_queue[-1] != index:
            self.suffix_queue.append(index)
            self.rows[index]["tool_batches"] += 1

    def suffix(self, ids: list[int], messages: list[dict]) -> None:
        if not self.suffix_queue:
            raise RuntimeError("Unbound tool suffix: runtime contract changed")
        index = self.suffix_queue.popleft()
        row = self.rows[index]
        next_ids = row["prompt_ids"] + row["retained"] + list(ids)
        over_budget = len(row["retained"]) + len(ids) > self.budget
        over_context = len(next_ids) >= self.context_limit
        row["observations"] += len(ids)
        if over_budget:
            self.flag(index, "TOOL_RESULT_BUDGET_EXCEEDED")
        if over_context:
            self.flag(index, "CONTEXT_LIMIT")
        self.emit(
            {
                "event": "tool_result",
                "row": index,
                "messages": deepcopy(messages),
                "suffix_ids": list(ids),
                "retained_before": len(row["retained"]),
                "would_rollback": over_budget or over_context,
            }
        )
        if not over_budget and not over_context:
            self.next_generation.append((index, next_ids))

    def generated(
        self,
        indices: list[int],
        prompts: list[list[int]],
        outputs: list[list[int]],
        eos_id: int,
    ) -> None:
        for index, prompt, output in zip(indices, prompts, outputs, strict=True):
            row = self.rows[index]
            suffix = prompt[len(row["prompt_ids"]) :]
            remaining = self.budget - len(suffix)
            kept = list(output[: max(0, remaining)])
            row["retained"] = suffix + kept
            row["generated_tokens"] += len(output)
            if len(output) > remaining or not kept or kept[-1] != eos_id:
                self.flag(index, "MODEL_GENERATION_LIMIT")
            self.emit(
                {
                    "event": "generation",
                    "row": index,
                    "input_ids": list(prompt),
                    "generated_ids": list(output),
                    "retained_generated_ids": kept,
                }
            )

    def summarize(
        self, output: tuple, environments: list, tokenizer, iterations: int
    ) -> list[dict]:
        prompt_ids, completion_ids, masks, completions = output[:4]
        result = []
        for i, (row, ids, mask, messages, env) in enumerate(
            zip(
                self.rows, completion_ids, masks, completions, environments, strict=True
            )
        ):
            if (
                ids != row["retained"]
                or prompt_ids[i] != row["prompt_ids"]
                or len(mask) != len(ids)
            ):
                raise RuntimeError(
                    "TRL retained tokens differ from the budget observer"
                )
            last = messages[-1] if messages else {}
            ended = last.get("role") == "assistant" and not last.get("tool_calls")
            # Mark suspicious markup, never attempt an alternative tool parser.
            if ended and "<tool_call>" in str(last.get("content") or ""):
                self.flag(i, "PARSE_OR_PROTOCOL_ERROR")
            if last.get("tool_calls"):
                self.flag(
                    i,
                    "TOOL_ITERATION_LIMIT"
                    if row["tool_batches"] >= iterations
                    else "UNRESOLVED_TOOL_CALL",
                )
            user_stopped = bool(env._user_stopped)
            priority = [
                "USER_API_ERROR",
                "OOM",
                "TOOL_RESULT_BUDGET_EXCEEDED",
                "CONTEXT_LIMIT",
                "MODEL_GENERATION_LIMIT",
                "CUSTOMER_TURN_LIMIT",
                "TOOL_CALL_LIMIT",
                "PARSE_OR_PROTOCOL_ERROR",
                "TOOL_ITERATION_LIMIT",
                "UNRESOLVED_TOOL_CALL",
            ]
            reason = next(
                (x for x in priority if x in row["flags"]),
                "USER_STOP_AND_MODEL_END"
                if user_stopped and ended
                else "MODEL_END_BEFORE_USER_STOP",
            )
            result.append(
                {
                    "row": i,
                    "stop_reason": reason,
                    "flags": row["flags"],
                    "user_stopped": user_stopped,
                    "model_ended": ended,
                    "prompt_tokens": len(prompt_ids[i]),
                    "completion_tokens": len(ids),
                    "model_tokens_retained": sum(mask),
                    "observation_tokens_retained": len(ids) - sum(mask),
                    "model_tokens_generated_before_clipping": row["generated_tokens"],
                    "observation_tokens_returned": row["observations"],
                    "decoded_completion": tokenizer.decode(
                        ids, skip_special_tokens=False
                    ),
                    "training_eligible": False,
                    "eligibility_scope": "DIAGNOSTIC_ONLY_NOT_RELEASED",
                    # Ordinary tool errors may be observed and recovered;
                    # they are not missing transport evidence.
                    "tool_exception_observed": "TOOL_EXCEPTION" in row["flags"],
                    "trajectory_transport_complete": not any(
                        flag in priority for flag in row["flags"]
                    ),
                }
            )
        return result


class GuardedTrajectoryTrace:
    """Observe native multi-turn GRPO termination without changing its outputs."""

    def __init__(self, budget: int, context_limit: int, emit):
        if budget <= 0 or context_limit <= 0:
            raise ValueError("Token limits must be positive")
        self.budget = budget
        self.context_limit = context_limit
        self.emit = emit
        self.rows: list[dict[str, Any]] = []
        self.suffix_queue: deque[int] = deque()
        self.next_generation: list[tuple[int, list[int]]] = []
        self.generation_count = 0
        self.binding_error: str | None = None

    def start(self, prompts: list[list[int]]) -> list[int]:
        if self.rows or not prompts:
            raise ValueError("Guarded trace must start exactly once")
        self.rows = [
            {
                "prompt_ids": list(prompt),
                "retained": [],
                "flags": [],
                "tool_batches": 0,
                "generated_tokens": 0,
                "observation_tokens": 0,
            }
            for prompt in prompts
        ]
        return list(range(len(prompts)))

    def bind_generation(self, prompts: list[list[int]]) -> list[int] | None:
        if not self.rows:
            return self.start(prompts)
        if self.binding_error is not None:
            return None
        expected = self.next_generation
        self.next_generation = []
        if [prompt for _, prompt in expected] != prompts:
            raise RuntimeError("Guarded generation-to-rollout binding mismatch")
        return [index for index, _ in expected]

    def flag(self, index: int, code: str) -> None:
        if code not in self.rows[index]["flags"]:
            self.rows[index]["flags"].append(code)

    def tool_called(self, index: int) -> None:
        if not self.suffix_queue or self.suffix_queue[-1] != index:
            self.suffix_queue.append(index)
            self.rows[index]["tool_batches"] += 1

    def suffix(self, ids: list[int], *_ignored_messages) -> None:
        if not self.suffix_queue:
            self.binding_error = "Unbound guarded tool suffix"
            self.emit(
                {
                    "event": "telemetry_binding_error",
                    "reason": self.binding_error,
                }
            )
            return
        index = self.suffix_queue.popleft()
        row = self.rows[index]
        next_ids = row["prompt_ids"] + row["retained"] + list(ids)
        completion_size = len(row["retained"]) + len(ids)
        over_budget = completion_size > self.budget
        over_context = len(next_ids) >= self.context_limit
        row["observation_tokens"] += len(ids)
        if over_budget:
            self.flag(index, "TOOL_RESULT_BUDGET_EXCEEDED")
        if over_context:
            self.flag(index, "CONTEXT_LIMIT")
        self.emit(
            {
                "event": "tool_suffix_observed",
                "row": index,
                "suffix_tokens": len(ids),
                "completion_tokens_before": len(row["retained"]),
                "would_rollback": over_budget or over_context,
            }
        )
        if not over_budget and not over_context:
            self.next_generation.append((index, next_ids))

    def generated(
        self,
        indices: list[int],
        prompts: list[list[int]],
        outputs: list[list[int]],
        eos_id: int,
    ) -> None:
        for index, prompt, output in zip(indices, prompts, outputs, strict=True):
            row = self.rows[index]
            if prompt[: len(row["prompt_ids"])] != row["prompt_ids"]:
                raise RuntimeError("Guarded generation prompt prefix mismatch")
            prefix = prompt[len(row["prompt_ids"]) :]
            remaining = self.budget - len(prefix)
            kept = list(output[: max(0, remaining)])
            row["retained"] = prefix + kept
            row["generated_tokens"] += len(output)
            eos_observed = bool(output) and output[-1] == eos_id
            clipped = len(output) > max(0, remaining)
            exhausted_without_eos = (
                len(row["retained"]) >= self.budget and not eos_observed
            )
            if clipped or exhausted_without_eos:
                self.flag(index, "COMPLETION_BUDGET_EXHAUSTED")
                self.emit({
                    "event": "censored_generation",
                    "row": index,
                    "generated_token_ids": list(output),
                    "retained_token_count": len(kept),
                    "training_eligible": False,
                })
            self.emit(
                {
                    "event": "generation_stop_observed",
                    "row": index,
                    "raw_generated_tokens": len(output),
                    "retained_generated_tokens": len(kept),
                    "cumulative_completion_tokens": len(row["retained"]),
                    "eos_observed": eos_observed,
                    "clipped_to_remaining_budget": clipped,
                }
            )

    def summarize(
        self, output: tuple, environments: list, tokenizer, iterations: int
    ) -> list[dict[str, Any]]:
        if self.binding_error is not None:
            raise RuntimeError(self.binding_error)
        prompt_ids, completion_ids, masks, completions = output[:4]
        if len(environments) != len(self.rows):
            raise RuntimeError("Guarded environment-to-rollout binding mismatch")
        if masks is None:
            masks = [[1] * len(ids) for ids in completion_ids]
        result = []
        eos_id = tokenizer.eos_token_id
        for index, (row, ids, mask, messages, environment) in enumerate(
            zip(
                self.rows,
                completion_ids,
                masks,
                completions,
                environments,
                strict=True,
            )
        ):
            if (
                list(ids) != row["retained"]
                or list(prompt_ids[index]) != row["prompt_ids"]
                or len(mask) != len(ids)
            ):
                raise RuntimeError("Native GRPO output differs from guarded observer")
            for signal in getattr(environment, "_runtime_stop_signals", []):
                self.flag(index, signal)
            last = messages[-1] if messages else {}
            ended = last.get("role") == "assistant" and not last.get("tool_calls")
            final_eos = bool(ids) and ids[-1] == eos_id
            if last.get("tool_calls"):
                self.flag(
                    index,
                    "TOOL_ITERATION_LIMIT"
                    if row["tool_batches"] >= iterations
                    else "UNRESOLVED_TOOL_CALL",
                )
            priority = [
                "TOOL_RESULT_BUDGET_EXCEEDED",
                "CONTEXT_LIMIT",
                "COMPLETION_BUDGET_EXHAUSTED",
                "TOOL_ITERATION_LIMIT",
                "UNRESOLVED_TOOL_CALL",
            ]
            reason = next((code for code in priority if code in row["flags"]), None)
            if reason is None:
                if final_eos:
                    reason = (
                        "USER_STOP_AND_MODEL_EOS"
                        if bool(environment._user_stopped)
                        else "MODEL_EOS_BEFORE_USER_STOP"
                    )
                else:
                    reason = "MODEL_END_WITHOUT_EOS"
                    self.flag(index, reason)
            model_tokens = sum(int(value) for value in mask)
            completion_budget_exhausted = any(
                code in row["flags"]
                for code in (
                    "TOOL_RESULT_BUDGET_EXCEEDED",
                    "COMPLETION_BUDGET_EXHAUSTED",
                )
            )
            result.append(
                {
                    "row": index,
                    "stop_reason": reason,
                    "stop_flags": list(row["flags"]),
                    "model_ended": ended,
                    "model_eos_observed": final_eos,
                    "completion_token_budget_exhausted": completion_budget_exhausted,
                    "context_limit_reached": "CONTEXT_LIMIT" in row["flags"],
                    "tool_iteration_limit_reached": (
                        "TOOL_ITERATION_LIMIT" in row["flags"]
                    ),
                    "unresolved_tool_call": "UNRESOLVED_TOOL_CALL" in row["flags"],
                    "framework_loop_abnormal_end": False,
                    "prompt_tokens": len(prompt_ids[index]),
                    "completion_tokens": len(ids),
                    "model_tokens_retained": model_tokens,
                    "observation_tokens_retained": len(ids) - model_tokens,
                    "model_completion_truncated": (
                        "COMPLETION_BUDGET_EXHAUSTED" in row["flags"]
                    ),
                    "model_completion_truncation_source": ("guarded_grpo_trainer_v1"),
                    "stop_reason_source": "guarded_grpo_trainer_v1",
                }
            )
        return result


def _failure_read(reader):
    """An evidence failure must not replace the model's numerical exception."""
    try:
        return reader()
    except Exception as exc:
        return {"status": "unavailable", "exception_type": type(exc).__name__}


def _failure_tensor(value, *, values=False):
    import torch

    if not torch.is_tensor(value):
        return {"status": "unavailable", "reason": "not_a_tensor"}
    record = {
        "status": "metadata_only",
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
    }
    if values:
        # Full input/position vectors are small; a dense 4D attention mask is not.
        if value.ndim > 2 or value.numel() > 131072:
            record["reason"] = "value_capture_size_limit"
        else:
            record.update(status="available", values=value.detach().cpu().tolist())
    return record


def _failure_cache(cache):
    """Read cache metadata only, after forward; never clone KV values or reset it."""
    if cache is None:
        return {"status": "none", "type": "NoneType", "layers": []}
    record = {
        "status": "available",
        "type": f"{type(cache).__module__}.{type(cache).__qualname__}",
        "observation_time": "after_forward_before_sampling",
        "tensor_values_saved": False,
    }
    if isinstance(cache, (tuple, list)):
        layers = cache
        record["representation"] = "legacy_key_value_pairs"
    else:
        layers = getattr(cache, "layers", None)
        record["representation"] = "cache_layers"
    record["sequence_length"] = (
        _failure_read(lambda: int(cache.get_seq_length()))
        if hasattr(cache, "get_seq_length")
        else {"status": "unavailable", "reason": "no_sequence_length_method"}
    )
    if not isinstance(layers, (list, tuple)) or len(layers) > 512:
        record["layers"] = {
            "status": "unavailable",
            "reason": "unsupported_cache_layout",
        }
        return record
    record["layers"] = []
    for index, layer in enumerate(layers):

        def metadata(layer=layer):
            if isinstance(layer, (tuple, list)) and len(layer) == 2:
                key, value = layer
            else:
                key, value = (
                    getattr(layer, "keys", None),
                    getattr(layer, "values", None),
                )
            key_meta, value_meta = _failure_tensor(key), _failure_tensor(value)
            return {
                "index": index,
                "key": key_meta,
                "value": value_meta,
                # This is stored tensor capacity, not necessarily logical cache length.
                "stored_sequence_dimension": key_meta.get("shape", [None])[-2]
                if len(key_meta.get("shape", [])) >= 2
                else None,
            }

        record["layers"].append(_failure_read(metadata))
    return record


def _native_sample_state(core):
    """Borrow allowlisted locals from the verified native frame; never retain it."""
    frame = inspect.currentframe()
    try:
        for _ in range(128):
            if frame is None:
                break
            module_name = frame.f_globals.get("__name__")
            if (
                module_name == "transformers.generation.utils"
                and frame.f_code.co_name == "_sample"
            ):
                module = sys.modules.get(module_name)
                mixin = getattr(module, "GenerationMixin", None)
                method = inspect.unwrap(getattr(mixin, "_sample", None))
                if (
                    module is not None
                    and module.__dict__ is frame.f_globals
                    and getattr(method, "__code__", None) is frame.f_code
                    and Path(module.__file__).resolve()
                    == Path(frame.f_code.co_filename).resolve()
                    and frame.f_locals.get("self") is core
                ):
                    return {
                        "status": "verified_frame",
                        "source": {
                            "module": module_name,
                            "function": "_sample",
                            "line": frame.f_lineno,
                        },
                        "input_ids": frame.f_locals.get("input_ids"),
                        "unfinished_sequences": frame.f_locals.get(
                            "unfinished_sequences"
                        ),
                        "generation_config": frame.f_locals.get("generation_config"),
                        "logits_processor": frame.f_locals.get("logits_processor"),
                        "has_eos_stopping_criteria": frame.f_locals.get(
                            "has_eos_stopping_criteria"
                        ),
                    }
            frame = frame.f_back
        return {"status": "unavailable", "reason": "no_verified_native_sample_frame"}
    finally:
        # Do not retain a stack frame (and all tensors reachable through it).
        del frame


def _failure_generation_state(core, rows):
    """Serialize only prefix/mask evidence, not config, processors or other locals."""
    import torch

    state = _native_sample_state(core)
    if state["status"] != "verified_frame":
        return state
    result = {"status": state["status"], "source": state["source"], "rows": list(rows)}
    for key, source, ndim, reason in (
        ("full_prefix", "input_ids", 2, "prefix_batch_contract"),
        (
            "unfinished_sequences",
            "unfinished_sequences",
            1,
            "unfinished_batch_contract",
        ),
    ):
        value = state[source]
        if (
            torch.is_tensor(value)
            and value.ndim == ndim
            and value.shape[0] == len(rows)
        ):
            result[key] = _failure_read(
                lambda value=value: _failure_tensor(value, values=True)
            )
        else:
            result[key] = {"status": "unavailable", "reason": reason}
    return result


class _FinishedRowPad:
    """Terminal processor: isolate proven EOS rows, never repair active scores.

    The native loop, full batch shape, RNG calls and KV layout remain intact.
    EOS means this generation ended, not that the business task succeeded.
    """

    def __init__(
        self, core, prompt_ids, rows, generation_config, emit, generation_index
    ):
        self.core, self.prompts, self.rows = (
            core,
            [list(p) for p in prompt_ids],
            list(rows),
        )
        self.config, self.emit, self.generation_index = (
            generation_config,
            emit,
            generation_index,
        )
        self.width = max(map(len, self.prompts))
        self.calls = self.nonfinite_steps = 0
        self.reported_rows = set()
        if (
            len(self.prompts) != len(self.rows)
            or len(set(self.rows)) != len(self.rows)
            or generation_config.num_beams != 1
            or generation_config.num_return_sequences != 1
            or generation_config.remove_invalid_values
            or getattr(generation_config, "guidance_scale", None) not in (None, 1.0)
        ):
            raise ValueError("Unsupported finished-row isolation configuration")
        # Use the resolved runtime IDs, not tokenizer guesses (PAD may equal EOS).
        self.eos = generation_config._eos_token_tensor
        self.pad = int(generation_config._pad_token_tensor.item())
        if (
            self.eos is None
            or self.eos.ndim != 1
            or self.eos.numel() == 0
            or self.pad < 0
        ):
            raise ValueError("Explicit runtime EOS and PAD are required")

    def finished(self, input_ids=None):
        import torch

        state = _native_sample_state(self.core)
        prefix, unfinished = state.get("input_ids"), state.get("unfinished_sequences")
        processors = state.get("logits_processor")
        if (
            state["status"] != "verified_frame"
            or state.get("generation_config") is not self.config
            or state.get("has_eos_stopping_criteria") is not True
            or not processors
            or processors[-1] is not self
            or not torch.is_tensor(prefix)
            or prefix.ndim != 2
            or prefix.shape != (len(self.rows), self.width + self.calls)
            or (input_ids is not None and input_ids is not prefix)
            or not torch.is_tensor(unfinished)
            or unfinished.shape != (len(self.rows),)
            or unfinished.device != prefix.device
            or not bool(((unfinished == 0) | (unfinished == 1)).all())
        ):
            raise RuntimeError("Unverified native finished-row binding")
        if self.calls == 0:
            expected = torch.tensor(
                [[self.pad] * (self.width - len(p)) + p for p in self.prompts],
                device=prefix.device,
                dtype=prefix.dtype,
            )
            if not torch.equal(prefix, expected):
                raise RuntimeError("Finished-row prompt binding mismatch")
        finished = unfinished == 0
        if bool(finished.any()):
            suffix = prefix[finished, self.width :]
            if suffix.shape[1] == 0:
                raise RuntimeError("Finished row lacks generated EOS")
            eos = torch.isin(suffix, self.eos.to(suffix.device))
            first = eos.to(torch.long).argmax(dim=1)
            after = torch.arange(suffix.shape[1], device=suffix.device) > first[:, None]
            if not bool(eos.any(dim=1).all()) or bool(
                ((suffix != self.pad) & after).any()
            ):
                raise RuntimeError("Finished row lacks EOS followed only by PAD")
        return finished

    def __call__(self, input_ids, scores):
        import torch

        finished = self.finished(input_ids)
        if (
            scores.ndim != 2
            or scores.shape[0] != len(self.rows)
            or self.pad >= scores.shape[1]
        ):
            raise RuntimeError("Unexpected terminal processor score shape")
        active_scores = scores[~finished]
        # Top-k/p legitimately create -inf entries, but NaN, +inf and an empty
        # finite support must never reach softmax/multinomial on an active row.
        if (
            bool(torch.isnan(active_scores).any())
            or bool(torch.isposinf(active_scores).any())
            or not bool(torch.isfinite(active_scores).any(dim=1).all())
        ):
            failure = FloatingPointError("INVALID_ACTIVE_SAMPLING_SCORES")
            try:
                self.emit(
                    {
                        "event": "invalid_active_sampling_scores",
                        "generation_index": self.generation_index,
                        "processor_step": self.calls,
                        "active_rows": [
                            row
                            for row, done in zip(
                                self.rows, finished.tolist(), strict=True
                            )
                            if not done
                        ],
                        "finite_counts": torch.isfinite(active_scores)
                        .sum(dim=1)
                        .tolist(),
                        "score_width": scores.shape[1],
                    }
                )
            except Exception as exc:
                failure.add_note(
                    f"Sampling-score evidence unavailable: {type(exc).__name__}"
                )
            raise failure
        if bool(finished.any()):
            scores = scores.clone()
            scores[finished] = -torch.inf
            scores[finished, self.pad] = 0
            ended = {
                row
                for row, flag in zip(self.rows, finished.tolist(), strict=True)
                if flag
            }
            new_rows = ended - self.reported_rows
            if new_rows:
                self.emit(
                    {
                        "event": "finished_rows_padding",
                        "generation_index": self.generation_index,
                        "rows": sorted(new_rows),
                        "generated_prefix_length": self.calls,
                        "pad_token_id": self.pad,
                        "business_success_implied": False,
                    }
                )
                self.reported_rows.update(new_rows)
        self.calls += 1
        return scores


def _nonfinite_forward_context(core, arguments, kwargs, rows):
    bound = _failure_read(
        lambda: inspect.signature(core.forward)
        .bind_partial(*arguments, **kwargs)
        .arguments
    )
    inputs = {}
    for name in ("input_ids", "attention_mask", "position_ids", "cache_position"):
        value = kwargs.get(name, bound.get(name))
        inputs[name] = _failure_read(
            lambda value=value: _failure_tensor(value, values=True)
        )
    cache = kwargs.get("past_key_values", bound.get("past_key_values"))
    return {
        "forward_inputs": inputs,
        "past_key_values": _failure_read(lambda: _failure_cache(cache)),
        "generation_state": _failure_read(
            lambda: _failure_generation_state(core, rows)
        ),
    }


def _generate_with_finished_row_guard(
    core,
    prompt_ids,
    indices,
    generation_index,
    emit,
    native_generate,
    images,
    multimodal_fields,
    expected_max_new_tokens=None,
):
    """Guard only native generation; never enclose scoring or backward."""
    import torch

    if getattr(core, "_policyagent_finished_row_owner", None) is not None:
        raise RuntimeError(
            "Concurrent or nested generation on the same model is forbidden"
        )
    # Capture before native generation: after a device-side assert CUDA
    # state may be unreadable. These snapshots never reset the RNG.
    device = next(core.parameters()).device
    rng = {
        "encoding": "base64_uint8",
        "cpu": base64.b64encode(bytes(torch.get_rng_state().tolist())).decode("ascii"),
        "cuda": {},
    }
    if device.type == "cuda":
        rng["cuda"][str(device)] = base64.b64encode(
            bytes(torch.cuda.get_rng_state(device).tolist())
        ).decode("ascii")
    emit(
        {
            "event": "generation_start",
            "generation_index": generation_index,
            "rows": indices,
            "input_ids": [list(p) for p in prompt_ids],
            "rng_before": rng,
            "raw_logits_guard": True,
        }
    )
    forward_count = 0
    current_arguments, current_kwargs = (), {}
    isolation = None
    original_builder = core._get_logits_processor
    missing = object()
    original_instance_builder = core.__dict__.get("_get_logits_processor", missing)

    @wraps(original_builder)
    def build_processors(*args, **kwargs):
        nonlocal isolation
        if isolation is not None:
            raise RuntimeError("Multiple native generations in one bound turn")
        bound = inspect.signature(original_builder).bind(*args, **kwargs).arguments
        if expected_max_new_tokens is not None and (
            bound["generation_config"].max_new_tokens != expected_max_new_tokens
        ):
            raise RuntimeError("Effective generation budget differs from requested cap")
        if bound["input_ids_seq_length"] != max(map(len, prompt_ids)):
            raise RuntimeError("Native generation prompt width mismatch")
        processors = original_builder(*args, **kwargs)
        isolation = _FinishedRowPad(
            core,
            prompt_ids,
            indices,
            bound["generation_config"],
            emit,
            generation_index,
        )
        # HF merges caller processors BEFORE temperature/top-p/etc. Wrap
        # the completed builder instead so this is the terminal check.
        result = type(processors)(processors)
        result.append(isolation)
        return result

    def remember_forward_inputs(module, arguments, kwargs):
        nonlocal current_arguments, current_kwargs
        # References only. CPU copies happen solely on a numerical failure.
        current_arguments, current_kwargs = arguments, kwargs

    def check_raw_logits(module, arguments, result):
        nonlocal forward_count
        logits = getattr(result, "logits", None)
        if logits is None or logits.ndim != 3 or logits.shape[0] != len(indices):
            raise RuntimeError("Unexpected raw-logits contract in guarded generation")
        last = logits[:, -1, :]
        finite_counts = torch.isfinite(last).sum(dim=-1).tolist()
        step = forward_count
        forward_count += 1
        if any(count != last.shape[-1] for count in finite_counts):
            failure = FloatingPointError("NONFINITE_RAW_MODEL_LOGITS")
            ignored = False
            if isolation is not None:
                try:
                    finished = isolation.finished().tolist()
                    ignored = all(
                        count == last.shape[-1] or done
                        for count, done in zip(finite_counts, finished, strict=True)
                    )
                except Exception as exc:
                    # Missing/unverifiable state cannot relax the guard.
                    failure.add_note(
                        f"Finished-row binding unavailable: {type(exc).__name__}"
                    )
            event = {
                "event": "finished_row_nonfinite_logits"
                if ignored
                else "nonfinite_raw_logits",
                "generation_index": generation_index,
                "forward_index": step,
                "rows": indices,
                "logits_shape": list(logits.shape),
                "finite_counts": finite_counts,
            }
            # Full context once for an isolated row fault, always for an
            # active-row fault. Avoid quadratic prefix/KV metadata logs.
            if not ignored or isolation.nonfinite_steps == 0:
                event["failure_context"] = _failure_read(
                    lambda: _nonfinite_forward_context(
                        core, current_arguments, current_kwargs, indices
                    )
                )
            if ignored:
                event.update(
                    disposition="EOS_ROW_ONLY_PAD_PENDING",
                    business_success_implied=False,
                )
            try:
                emit(event)
            except Exception as exc:
                failure.add_note(f"Evidence logging unavailable: {type(exc).__name__}")
                raise failure from exc
            if ignored:
                isolation.nonfinite_steps += 1
                return  # Native terminal processor still MUST run.
            # Never repair an active row or retry. This is an
            # infrastructure failure, not a reward-zero trajectory.
            raise failure

    handles = []
    try:
        core._policyagent_finished_row_owner = build_processors
        core._get_logits_processor = build_processors
        handles.append(
            core.register_forward_pre_hook(remember_forward_inputs, with_kwargs=True)
        )
        handles.append(core.register_forward_hook(check_raw_logits))
        output, logprobs = native_generate(prompt_ids, images, multimodal_fields)
        if forward_count == 0:
            raise RuntimeError("Raw-logits guard did not observe model forward")
        if isolation is None:
            raise RuntimeError("Terminal finished-row processor was not installed")
        if isolation.calls != forward_count:
            raise RuntimeError("Terminal processor did not observe every forward")
    except Exception as exc:
        try:
            emit(
                {
                    "event": "generation_error",
                    "rows": indices,
                    "generation_index": generation_index,
                    "observed_forward_count": forward_count,
                    "exception_type": type(exc).__name__,
                }
            )
        except Exception as logging_error:
            exc.add_note(
                f"Generation logging unavailable: {type(logging_error).__name__}"
            )
        raise
    finally:
        for handle in handles:
            handle.remove()
        if original_instance_builder is missing:
            del core._get_logits_processor
        else:
            core._get_logits_processor = original_instance_builder
        del core._policyagent_finished_row_owner
        current_arguments, current_kwargs = (), {}
    return output, logprobs


def validate_generation_safety(config: dict) -> bool:
    """Opt in explicitly; historical configs keep their original runtime."""
    if "generation_safety" not in config:
        return False
    if config["generation_safety"] != {"mode": "eos_finished_rows_v1"}:
        raise ValueError("Unsupported generation_safety configuration")
    execution_mode = config.get("execution_mode", "OPTIMIZE")
    if execution_mode not in {"OPTIMIZE", "ROLLOUT_DIAGNOSTIC"}:
        raise ValueError(
            "generation_safety requires OPTIMIZE or ROLLOUT_DIAGNOSTIC"
        )
    if config.get("grpo", {}).get("use_vllm") is not False:
        raise ValueError("Training generation_safety requires transformers, not vLLM")
    return True


def _bf16_native_generation(trainer, core, native_generate):
    """Enter AMP for one generation only; never cache weights across updates."""

    @wraps(native_generate)
    def generate(*args, **kwargs):
        import torch

        with trainer.accelerator.autocast():
            if next(core.parameters()).device.type == "cuda" and (
                not torch.is_autocast_enabled("cuda")
                or torch.get_autocast_dtype("cuda") != torch.bfloat16
            ):
                raise RuntimeError("Generation precision requires actual BF16 autocast")
            return native_generate(*args, **kwargs)

    return generate


@contextmanager
def _remaining_batch_generation_budget(trainer, trace, indices, prompts, emit):
    """Cap shared generation at the largest remaining trajectory budget.

    Do not use the minimum: that would censor other active candidates early.
    Per-row trimming and transport rejection remain native/observer decisions.
    """
    if trace is None or indices is None:
        yield
        return
    remaining = []
    for index, prompt in zip(indices, prompts, strict=True):
        original = trace.rows[index]["prompt_ids"]
        if prompt[:len(original)] != original:
            raise RuntimeError("Remaining-budget prompt prefix mismatch")
        remaining.append(trace.budget - (len(prompt) - len(original)))
    if not remaining or min(remaining) <= 0:
        raise RuntimeError("Generation requested with no remaining trajectory budget")
    original_config = trainer.generation_config
    original_kwargs = trainer.generation_kwargs
    configured = original_config.max_new_tokens
    if "max_new_tokens" in original_kwargs:
        configured = min(configured, original_kwargs["max_new_tokens"])
    limit = min(configured, max(remaining))
    try:
        trainer.generation_config = deepcopy(original_config)
        trainer.generation_kwargs = dict(original_kwargs)
        trainer.generation_config.max_new_tokens = limit
        trainer.generation_kwargs["max_new_tokens"] = limit
        emit({"event": "remaining_generation_budget", "rows": indices,
              "remaining_tokens": remaining, "max_new_tokens": limit})
        yield
    finally:
        trainer.generation_config = original_config
        trainer.generation_kwargs = original_kwargs


def make_guarded_grpo_trainer(base_class: type, emit, *, bf16_generation=False) -> type:
    """Keep native training/KL intact and protect only generation forwards."""
    if type(bf16_generation) is not bool:
        raise ValueError("bf16_generation must be an explicit bool")

    class GuardedGRPOTrainer(base_class):
        def _generate(self, prompts):
            if getattr(self, "_policyagent_stop_trace", None) is not None:
                raise RuntimeError("Nested guarded trajectory generation is forbidden")
            group_index = getattr(self, "_policyagent_rollout_group_index", 0)
            self._policyagent_rollout_group_index = group_index + 1

            def record(event):
                emit(
                    {
                        **event,
                        "row_scope": "rollout_group",
                        "rollout_group_index": group_index,
                        "optimizer_step": getattr(
                            getattr(self, "state", None), "global_step", 0
                        ),
                    }
                )

            model_config = self.model.config
            if self._is_vlm:
                model_config = model_config.text_config
            trace = GuardedTrajectoryTrace(
                self.max_completion_length,
                model_config.max_position_embeddings,
                record,
            )
            self._policyagent_stop_trace = trace
            try:
                output = super()._generate(prompts)
                report = trace.summarize(
                    output,
                    self.environments,
                    self._tokenizer,
                    self.max_tool_calling_iterations,
                )
                for row, environment in zip(report, self.environments, strict=True):
                    environment._set_trainer_completion_telemetry(
                        {key: value for key, value in row.items() if key != "row"}
                    )
                    record({"event": "rollout_stop", **row})
                blocking = [
                    environment._blocking_transport_reasons()
                    if hasattr(environment, "_blocking_transport_reasons") else []
                    for environment in self.environments
                ]
                if any(blocking):
                    # Save the WHOLE group before the first reward call can
                    # raise. Separate quarantine records cannot seed a resume.
                    for index, environment in enumerate(self.environments):
                        environment._persist_rejected_rollout(trainer_evidence={
                            "optimizer_step": getattr(self.state, "global_step", 0),
                            "rollout_group_index": group_index,
                            "row": index,
                            "group_blocking_reasons": blocking,
                            "prompt_token_ids": output[0][index],
                            "completion_token_ids": output[1][index],
                            "completion_text": self._tokenizer.decode(
                                output[1][index], skip_special_tokens=False
                            ),
                            "trainer_messages": output[3][index],
                        })
                if not any(blocking):
                    from src.evaluation.task44_hybrid_reward import prepare_semantic_group

                    prepare_semantic_group(
                        self.environments, output,
                        optimizer_step=getattr(getattr(self, "state", None), "global_step", 0),
                        group_index=group_index,
                    )
                return output
            except Exception as exc:
                try:
                    record(
                        {
                            "event": "rollout_group_abnormal_end",
                            "framework_loop_abnormal_end": True,
                            "exception_type": type(exc).__name__,
                            "business_failure_implied": False,
                        }
                    )
                except Exception as logging_error:
                    exc.add_note(
                        "Stop telemetry logging unavailable: "
                        f"{type(logging_error).__name__}"
                    )
                raise
            finally:
                del self._policyagent_stop_trace

        def _generate_single_turn(self, prompt_ids, images, multimodal_fields):
            if (
                self.accelerator.num_processes != 1
                or self.use_vllm
                or self._is_vlm
                or self.use_transformers_continuous_batching
            ):
                raise ValueError(
                    "Only single-process text/transformers generation is audited"
                )
            core = (
                self.model.get_base_model()
                if hasattr(self.model, "get_base_model")
                else self.model
            )
            with _exclusive_sampling_model(self), _exclusive_sampling_model(core):
                index = getattr(self, "_policyagent_generation_index", 0)
                self._policyagent_generation_index = index + 1
                trace = getattr(self, "_policyagent_stop_trace", None)
                original_indices = (
                    trace.bind_generation(prompt_ids)
                    if trace is not None
                    else list(range(len(prompt_ids)))
                )

                def record(event):
                    # TRL may continue only a subset of candidates. These are
                    # turn-local indices, never original task/rollout identities.
                    emit(
                        {
                            **event,
                            "row_scope": "generation_local",
                            "optimizer_step": getattr(
                                getattr(self, "state", None), "global_step", 0
                            ),
                        }
                    )

                native_generate = super()._generate_single_turn
                if bf16_generation:
                    native_generate = _bf16_native_generation(
                        self, core, native_generate
                    )
                with _remaining_batch_generation_budget(
                    self, trace, original_indices, prompt_ids, record
                ):
                    output, logprobs = _generate_with_finished_row_guard(
                        core,
                        prompt_ids,
                        list(range(len(prompt_ids))),
                        index,
                        record,
                        native_generate,
                        images,
                        multimodal_fields,
                        expected_max_new_tokens=self.generation_config.max_new_tokens,
                    )
                if trace is not None and original_indices is not None:
                    trace.generated(
                        original_indices,
                        prompt_ids,
                        output,
                        self._tokenizer.eos_token_id,
                    )
                return output, logprobs

        def _tool_call_loop(self, *args, **kwargs):
            trace = getattr(self, "_policyagent_stop_trace", None)
            if trace is None:
                return super()._tool_call_loop(*args, **kwargs)
            originals = self._sync_tool_dicts

            class ObservedMembership(dict):
                def __init__(self, index, mapping):
                    self.index = index
                    super().__init__(mapping)

                def __contains__(self, name):
                    trace.tool_called(self.index)
                    return super().__contains__(name)

            self._sync_tool_dicts = [
                ObservedMembership(index, mapping)
                for index, mapping in enumerate(originals)
            ]
            try:
                return super()._tool_call_loop(*args, **kwargs)
            finally:
                self._sync_tool_dicts = originals

        def _get_tool_suffix_ids(self, tool_messages):
            ids = super()._get_tool_suffix_ids(tool_messages)
            trace = getattr(self, "_policyagent_stop_trace", None)
            if trace is not None:
                trace.suffix(ids)
            return ids

    return GuardedGRPOTrainer


def make_sampling_trainer(
    base_class: type, *, bf16_generation=False, expected_do_sample=True
) -> type:
    """Factory keeps the module importable in local CPU-only unit tests."""
    if type(bf16_generation) is not bool:
        raise ValueError("bf16_generation must be an explicit bool")
    if type(expected_do_sample) is not bool:
        raise ValueError("expected_do_sample must be an explicit bool")

    class SamplingTrainer(base_class):
        def train(self, *args, **kwargs):
            raise RuntimeError("Pure sampling forbids train()")

        def create_optimizer(self, *args, **kwargs):
            raise RuntimeError("Pure sampling forbids optimizer creation")

        def compute_loss(self, *args, **kwargs):
            raise RuntimeError("Pure sampling forbids loss computation")

        def sample_group(self, inputs: list[dict], emit):
            with _exclusive_sampling_model(self):
                return self._policyagent_sample_group(inputs, emit)

        def _policyagent_sample_group(self, inputs, emit):
            import torch

            if self.accelerator.num_processes != 1 or self.use_vllm or self._is_vlm:
                raise ValueError(
                    "Only single-process text/transformers sampling is audited"
                )
            if self.use_transformers_continuous_batching:
                raise ValueError("Continuous batching is not audited")
            if self.optimizer is not None or self.lr_scheduler is not None:
                raise RuntimeError("Unexpected optimizer state")
            if (
                len(inputs) != self.num_generations
                or len({str(x["task_id"]) for x in inputs}) != 1
            ):
                raise ValueError("One complete same-task group is required")
            if any(x != inputs[0] for x in inputs):
                raise ValueError("All candidates must share the exact reset inputs")
            if (
                getattr(getattr(self, "generation_config", None), "do_sample", None)
                is not expected_do_sample
                or not isinstance(getattr(self, "generation_kwargs", None), dict)
                or self.generation_kwargs.get("do_sample") is not expected_do_sample
            ):
                raise RuntimeError("Effective decode mode differs from sampling contract")
            self.model.requires_grad_(False)
            self.model.eval()
            self._budget_trace = GuardedTrajectoryTrace(
                self.max_completion_length,
                self.model.config.max_position_embeddings,
                emit,
            )
            try:
                with torch.inference_mode():
                    # Parent handles pool allocation, tool bindings and reset.
                    # Our _generate exits before its scoring/forward path.
                    self._generate_and_score_completions(deepcopy(inputs))
            except _Generated as done:
                report = self._budget_trace.summarize(
                    done.output,
                    self.environments,
                    self._tokenizer,
                    self.max_tool_calling_iterations,
                )
                for row, environment in zip(
                    report, self.environments, strict=True
                ):
                    environment._set_trainer_completion_telemetry(
                        {key: value for key, value in row.items() if key != "row"}
                    )
                    row.update(
                        {
                            "training_eligible": False,
                            "eligibility_scope": "DIAGNOSTIC_ONLY_NOT_RELEASED",
                            "trajectory_transport_complete": not any(
                                row[name]
                                for name in (
                                    "completion_token_budget_exhausted",
                                    "context_limit_reached",
                                    "tool_iteration_limit_reached",
                                    "unresolved_tool_call",
                                    "framework_loop_abnormal_end",
                                )
                            ),
                        }
                    )
                if any(p.grad is not None for p in self.model.parameters()):
                    raise RuntimeError("Unexpected parameter gradient in pure sampling")
                return done.output, report
            except _FatalToolError as failure:
                raise failure.cause
            raise RuntimeError("TRL bypassed the audited generation boundary")

        def _generate(self, prompts):
            raise _Generated(super()._generate(prompts))

        def _generate_single_turn(self, prompt_ids, images, multimodal_fields):
            # PEFT.generate calls the core model, not PEFT.forward. Reject nested
            # use before consuming the current turn's candidate-index bindings.
            core = (
                self.model.get_base_model()
                if hasattr(self.model, "get_base_model")
                else self.model
            )
            with _exclusive_sampling_model(core):
                return self._policyagent_generate_single_turn(
                    core, prompt_ids, images, multimodal_fields
                )

        def _policyagent_generate_single_turn(
            self, core, prompt_ids, images, multimodal_fields
        ):
            if getattr(core, "_policyagent_finished_row_owner", None) is not None:
                raise RuntimeError(
                    "Concurrent or nested generation on the same model is forbidden"
                )
            trace = self._budget_trace
            if not trace.rows and any(
                len(prompt) + 2 * trace.budget > trace.context_limit
                for prompt in prompt_ids
            ):
                raise ValueError(
                    "Context cannot hold prompt + cumulative budget + generation headroom"
                )
            indices = trace.bind_generation(prompt_ids)
            if indices is None:
                raise RuntimeError("Sampling generation-to-rollout binding unavailable")
            generation_index = getattr(trace, "generation_count", 0)
            trace.generation_count = generation_index + 1
            try:
                native_generate = super()._generate_single_turn
                if bf16_generation:
                    native_generate = _bf16_native_generation(
                        self, core, native_generate
                    )
                output, logprobs = _generate_with_finished_row_guard(
                    core,
                    prompt_ids,
                    indices,
                    generation_index,
                    trace.emit,
                    native_generate,
                    images,
                    multimodal_fields,
                )
            except Exception as exc:
                for i in indices:
                    trace.flag(
                        i,
                        "OOM"
                        if type(exc).__name__ == "OutOfMemoryError"
                        else "GENERATION_ERROR",
                    )
                raise
            trace.generated(indices, prompt_ids, output, self._tokenizer.eos_token_id)
            return output, logprobs

        def _tool_call_loop(self, *args, **kwargs):
            if any(self._async_tool_dicts):
                raise ValueError("Async tools are not audited")
            originals = self._sync_tool_dicts

            def observed(index, name, function):
                @wraps(function)
                def call(**arguments):
                    trace = self._budget_trace
                    trace.tool_called(index)
                    trace.emit(
                        {
                            "event": "tool_call",
                            "row": index,
                            "name": name,
                            "arguments": arguments,
                        }
                    )
                    try:
                        return function(**arguments)
                    except Exception as exc:
                        from src.rl.user_simulator_fail_fast import (
                            UserSimulatorSystemFailure,
                        )

                        env = self.environments[index]
                        code = "TOOL_EXCEPTION"
                        if isinstance(exc, UserSimulatorSystemFailure):
                            code = "USER_API_ERROR"
                        elif type(exc).__name__ == "OutOfMemoryError":
                            code = "OOM"
                        elif (
                            name == "respond_to_user"
                            and env._customer_turns >= env._max_customer_turns
                        ):
                            code = "CUSTOMER_TURN_LIMIT"
                        elif env._tool_counter >= env._max_tool_calls:
                            code = "TOOL_CALL_LIMIT"
                        if code in {
                            "CUSTOMER_TURN_LIMIT",
                            "TOOL_CALL_LIMIT",
                        }:
                            trace.flag(index, code)
                        trace.emit(
                            {
                                "event": "tool_error",
                                "row": index,
                                "code": code,
                                "exception_type": type(exc).__name__,
                            }
                        )
                        if code in {"USER_API_ERROR", "OOM"}:
                            raise _FatalToolError(exc) from None
                        raise

                return call

            class ObservedTools(dict):
                # Unknown names must still produce the same TRL error result,
                # but need a row binding before _get_tool_suffix_ids observes it.
                def __init__(self, index, mapping):
                    self.index = index
                    super().__init__(
                        {
                            name: observed(index, name, fn)
                            for name, fn in mapping.items()
                        }
                    )

                def __contains__(self, name):
                    return True

                def __missing__(self, name):
                    def missing(**arguments):
                        raise ValueError(f"Tool {name} not found.")

                    return observed(self.index, name, missing)

            self._sync_tool_dicts = [
                ObservedTools(i, mapping) for i, mapping in enumerate(originals)
            ]
            try:
                return super()._tool_call_loop(*args, **kwargs)
            finally:
                self._sync_tool_dicts = originals

        def _get_tool_suffix_ids(self, tool_messages):
            ids = super()._get_tool_suffix_ids(tool_messages)
            self._budget_trace.suffix(ids)
            return ids

    return SamplingTrainer


def run_pure_sampling(
    trainer,
    dataset: list[dict],
    config: dict,
    preflight: dict,
    output_dir: Path,
    groups_per_task: int,
    runtime: dict,
) -> dict:
    """Persist every group, including interrupted groups, without changing reward."""
    import torch
    from src.training.run_retail_agentic_grpo import save_json, sha256

    events_path = output_dir / "generation_events.jsonl"
    groups_path = output_dir / "sampling_groups.jsonl"
    if events_path.exists() or groups_path.exists():
        raise FileExistsError("Refusing to overwrite sampling logs")

    def append(path, row):
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    save_json(output_dir / "effective_config.json", config)
    save_json(
        output_dir / "command.json",
        {
            "argv": sys.argv,
            "groups_per_task": groups_per_task,
            "parameter_update_requested": False,
            "decode_contract": deepcopy(config["sampling"]),
        },
    )
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    completed = []
    group_size = config["grpo"]["num_generations"]
    for repeat in range(groups_per_task):
        for example in dataset:
            group_id = f"{example['task_id']}:{repeat}"

            def emit(event):
                append(
                    events_path,
                    {"group_id": group_id, "task_id": str(example["task_id"]), **event},
                )

            try:
                generated, diagnostics = trainer.sample_group(
                    [deepcopy(example) for _ in range(group_size)], emit
                )
                # Original programmatic reward and raw DB/trajectory persistence.
                # Budget-invalid rewards remain observational, never training eligible.
                rewards = [env.get_reward() for env in trainer.environments]
                for index, row in enumerate(diagnostics):
                    row["raw_row_index"] = len(completed) * group_size + index
                    row["candidate_index"] = index
                record = {
                    "group_id": group_id,
                    "task_id": str(example["task_id"]),
                    "user_seed": example["user_seed"],
                    "status": "COMPLETED",
                    "prompt": example["prompt"],
                    "completions": generated[3],
                    "rewards_observational_only": rewards,
                    "diagnostics": diagnostics,
                    "all_candidates_transport_complete": all(
                        d["trajectory_transport_complete"] for d in diagnostics
                    ),
                    "training_eligible": False,
                }
                append(groups_path, record)
                completed.append(record)
            except Exception as exc:
                # No raw exception text (may contain credentials or user data).
                emit({"event": "group_failed", "exception_type": type(exc).__name__})
                for env in trainer.environments or []:
                    if getattr(env, "_environment", None) is not None:
                        env._persist_rollout(
                            {
                                "reward": None,
                                "failure_type": type(exc).__name__,
                                "training_eligible": False,
                            }
                        )
                append(
                    groups_path,
                    {
                        "group_id": group_id,
                        "status": "FAILED",
                        "exception_type": type(exc).__name__,
                        "training_eligible": False,
                    },
                )
                raise

    def read_jsonl(path: Path) -> list[dict[str, Any]]:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    raw_path = output_dir / "raw_rollouts.jsonl"
    evidence_path = output_dir / "rollout_evidence.jsonl"
    raw_rows = read_jsonl(raw_path)
    evidence_rows = read_jsonl(evidence_path)
    expected_rollouts = len(completed) * group_size
    if len(raw_rows) != expected_rollouts or len(evidence_rows) != expected_rollouts:
        raise RuntimeError("Pure sampling raw/evidence rollout count mismatch")
    expected_tasks = {str(task_id) for task_id in config["data"]["task_ids"]}
    if {str(row.get("task_id")) for row in raw_rows} != expected_tasks:
        raise RuntimeError("Pure sampling raw rollout task coverage mismatch")
    if {str(row.get("task_id")) for row in evidence_rows} != expected_tasks:
        raise RuntimeError("Pure sampling evidence task coverage mismatch")
    for index, (raw, evidence) in enumerate(
        zip(raw_rows, evidence_rows, strict=True)
    ):
        expected_group = completed[index // group_size]
        expected_task_id = str(expected_group["task_id"])
        if (
            str(raw.get("task_id")) != expected_task_id
            or str(evidence.get("task_id")) != expected_task_id
            or raw.get("user_seed") != expected_group["user_seed"]
            or evidence.get("user_seed") != expected_group["user_seed"]
        ):
            raise RuntimeError("Pure sampling rollout identity binding mismatch")
        if raw.get("hidden_user_scenario_persisted") is not False or evidence.get(
            "hidden_user_scenario_persisted"
        ) is not False:
            raise RuntimeError("Pure sampling artifact persisted hidden user scenario")
        tool_call_count = raw.get("tool_calls")
        if (
            not isinstance(raw.get("messages"), list)
            or type(tool_call_count) is not int
            or tool_call_count < 0
        ):
            raise RuntimeError("Pure sampling raw trajectory evidence is incomplete")
        expected_initial_user = next(
            (
                message.get("content")
                for message in expected_group["prompt"]
                if message.get("role") == "user"
            ),
            None,
        )
        actual_initial_user = next(
            (
                message.get("content")
                for message in raw["messages"]
                if message.get("role") == "user"
            ),
            None,
        )
        if expected_initial_user is None or actual_initial_user != expected_initial_user:
            raise RuntimeError("Pure sampling initial user message binding mismatch")
        if raw.get("reward") is None:
            raise RuntimeError("Pure sampling observational reward is missing")
        required_evidence = {
            "initial_state": dict,
            "final_state": dict,
            "state_diff": list,
            "state_hashes": dict,
            "tool_trace": list,
            "terminal_evaluator": dict,
        }
        if any(
            not isinstance(evidence.get(name), expected_type)
            for name, expected_type in required_evidence.items()
        ):
            raise RuntimeError("Pure sampling evidence sidecar is incomplete")
        if raw.get("completion") != evidence.get("completion"):
            raise RuntimeError("Raw/evidence completion telemetry mismatch")
        completion = raw.get("completion") or {}
        if completion.get("stop_reason_source") != "guarded_grpo_trainer_v1":
            raise RuntimeError("Pure sampling rollout lacks S5K stop telemetry")
        claimed = evidence.get("evidence_sha256")
        canonical = dict(evidence)
        canonical.pop("evidence_sha256", None)
        digest = hashlib.sha256(
            json.dumps(
                canonical,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest().upper()
        if claimed != digest or raw.get("evidence_sha256") != claimed:
            raise RuntimeError(f"Pure sampling evidence hash mismatch at row {index}")

    artifacts = {}
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
        artifact = {"sha256": sha256(path), "path": str(path)}
        if path.suffix == ".jsonl":
            artifact["rows"] = len(read_jsonl(path))
        artifacts[name] = artifact
    # Bind actual execution sources as well as commit: the worktree may be dirty.
    source_paths = [
        Path(__file__),
        Path(__file__).with_name("run_retail_agentic_grpo.py"),
    ]
    source_paths += sorted((Path(__file__).parents[1] / "rl").glob("*.py"))
    source_paths += sorted((Path(__file__).parents[1] / "guards").glob("*.py"))
    source_paths.append(
        Path(__file__).parents[1] / "evaluation" / "staged_reward_shadow.py"
    )
    manifest = {
        "schema_version": "retail-pure-sampling-v1",
        "status": "COMPLETED",
        "execution_mode": "PURE_SAMPLING",
        "sampling_mode": config["sampling"]["mode"],
        "optimization_enabled": False,
        "backward_called": False,
        "optimizer_created": False,
        "loss_computed": False,
        "training_eligible": False,
        "training_memory_feasibility_verified": False,
        "config_sha256": preflight["config_sha256"],
        "starting_model_sha256": preflight["model_sha256"],
        "task_split_sha256": preflight["split_sha256"],
        "openings_sha256": preflight["openings_sha256"],
        "upstream": preflight["upstream_checkout"],
        "git": {
            "commit": preflight["git_commit"],
            "dirty": preflight["git_dirty_at_start"],
        },
        "execution_sources": {
            str(p.relative_to(Path(__file__).parents[2])): sha256(p)
            for p in source_paths
        },
        "runtime": runtime,
        "groups": len(completed),
        "rollouts": len(completed) * group_size,
        "effective_task_ids": sorted(expected_tasks, key=int),
        "decode_contract": deepcopy(config["sampling"]),
        "elapsed_seconds": time.perf_counter() - start,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "artifacts": artifacts,
        "business_improvement_claim_allowed": False,
    }
    save_json(output_dir / "run_manifest.json", manifest)
    save_json(
        output_dir / "run_state.json",
        {
            "status": "COMPLETED",
            "mode": config["sampling"]["mode"],
            "run_manifest_sha256": sha256(output_dir / "run_manifest.json"),
        },
    )
    return manifest
