from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from src.guards.retail_pre_action import (
    WRITE_TOOLS,
    ToolProposal,
    context_from_messages,
    evaluate_retail_actions,
)


REWARD_CONFIG_ENV = "POLICYAGENT_REWARD_CONFIG_JSON"
ROLLOUT_LOG_ENV = "POLICYAGENT_ROLLOUT_LOG"
DEFAULT_REWARD_CONFIG: dict[str, Any] = {
    "process_reward_mode": "one_to_one_required_action_progress",
    "environment_state_action_progress_gate": "multiply",
    "environment_state_weight": 0.70,
    "required_action_weight": 0.20,
    "communication_weight": 0.10,
    "tool_error_penalty_each": 0.05,
    "tool_error_penalty_cap": 0.20,
    "repeated_call_penalty_each": 0.03,
    "repeated_call_penalty_cap": 0.15,
    "unexpected_write_penalty_each": 0.05,
    "unexpected_write_penalty_cap": 0.20,
    "unfinished_interaction_penalty": 0.10,
    "llm_judge_used": False,
    "policy_guard_used_as_reward": False,
    "confirmation_signal_used_as_reward": False,
}


def load_reward_config() -> dict[str, Any]:
    """Load the frozen reward specification supplied by the training runner."""

    raw = os.environ.get(REWARD_CONFIG_ENV)
    configured = json.loads(raw) if raw else {}
    if not isinstance(configured, dict):
        raise RuntimeError(f"{REWARD_CONFIG_ENV} must contain a JSON object")
    unknown = sorted(set(configured) - set(DEFAULT_REWARD_CONFIG))
    if unknown:
        raise RuntimeError(f"Unknown Agentic RL reward keys: {unknown}")
    reward = {**DEFAULT_REWARD_CONFIG, **configured}
    if reward["process_reward_mode"] != "one_to_one_required_action_progress":
        raise RuntimeError("Unsupported Agentic RL process_reward_mode")
    if reward["environment_state_action_progress_gate"] != "multiply":
        raise RuntimeError("Unsupported environment-state action-progress gate")
    for key in (
        "environment_state_weight",
        "required_action_weight",
        "communication_weight",
        "tool_error_penalty_each",
        "tool_error_penalty_cap",
        "repeated_call_penalty_each",
        "repeated_call_penalty_cap",
        "unexpected_write_penalty_each",
        "unexpected_write_penalty_cap",
        "unfinished_interaction_penalty",
    ):
        reward[key] = float(reward[key])
        if reward[key] < 0:
            raise RuntimeError(f"Reward field {key} must be non-negative")
    if sum(
        reward[key]
        for key in (
            "environment_state_weight",
            "required_action_weight",
            "communication_weight",
        )
    ) <= 0:
        raise RuntimeError("At least one positive reward component weight is required")
    for key in (
        "llm_judge_used",
        "policy_guard_used_as_reward",
        "confirmation_signal_used_as_reward",
    ):
        if reward[key] is not False:
            raise RuntimeError(f"Agentic RL v1 requires {key}=false")
    return reward


def _tool_signature(name: str, arguments: dict[str, Any]) -> str:
    return f"{name}:{json.dumps(arguments, ensure_ascii=False, sort_keys=True)}"


def gate_environment_state_reward(
    environment_reward: float, action_recall: float | None
) -> tuple[float, float]:
    """Prevent a satisfied initial DB state from rewarding a no-op trajectory."""

    gate = 1.0 if action_recall is None else action_recall
    return environment_reward * gate, gate


def one_to_one_action_progress(task: Any, messages: list[Any]) -> dict[str, Any]:
    """Measure required-action progress without reusing one call twice.

    Tau2's benchmark evaluator intentionally checks whether each golden action is
    present. For process shaping we need stricter credit accounting: once a model
    tool call satisfies one expected action, it cannot satisfy another duplicate.
    """

    criteria = task.evaluation_criteria
    expected = (
        [
            action
            for action in (criteria.actions or [])
            if action.requestor == "assistant"
        ]
        if criteria is not None
        else []
    )
    predicted = [
        call
        for message in messages
        if getattr(message, "role", None) == "assistant"
        for call in (getattr(message, "tool_calls", None) or [])
        if getattr(call, "requestor", "assistant") == "assistant"
    ]
    unused = set(range(len(predicted)))
    matches: list[dict[str, Any]] = []
    for action in expected:
        match_index = next(
            (
                index
                for index in sorted(unused)
                if action.compare_with_tool_call(predicted[index])
            ),
            None,
        )
        matched = match_index is not None
        if matched:
            unused.remove(match_index)
        matches.append(
            {
                "action_id": action.action_id,
                "name": action.name,
                "matched": matched,
                "matched_call_index": match_index,
            }
        )

    expected_exact = Counter(
        _tool_signature(action.name, action.arguments) for action in expected
    )
    predicted_exact = Counter(
        _tool_signature(call.name, call.arguments) for call in predicted
    )
    duplicate_excess = sum(
        max(0, count - max(1, expected_exact.get(signature, 0)))
        for signature, count in predicted_exact.items()
    )
    matched_count = sum(item["matched"] for item in matches)
    unexpected_writes = [
        {
            "name": predicted[index].name,
            "arguments": dict(predicted[index].arguments),
        }
        for index in sorted(unused)
        if predicted[index].name in WRITE_TOOLS
    ]
    return {
        "expected_count": len(expected),
        "predicted_count": len(predicted),
        "matched_count": matched_count,
        "recall": matched_count / len(expected) if expected else None,
        "matches": matches,
        "duplicate_excess_count": duplicate_excess,
        "unexpected_write_count": len(unexpected_writes),
        "unexpected_writes": unexpected_writes,
    }


_CONFIRM_QUESTION = re.compile(
    r"\b(confirm|confirmation|yes\s*/\s*no|proceed|go ahead)\b|确认|是否继续",
    re.IGNORECASE,
)
_AFFIRMATIVE = re.compile(
    r"\b(yes|confirm(?:ed)?|proceed|go ahead|sure|do it)\b|确认|同意|继续",
    re.IGNORECASE,
)


def confirmation_diagnostics(messages: list[Any]) -> dict[str, Any]:
    """Conservatively detect explicit confirmation before Retail write calls.

    This signal is diagnostic-only until validated against independent human gold.
    """

    confirmed_after = -1
    last_write = -1
    checks: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        role = getattr(message, "role", None)
        content = str(getattr(message, "content", "") or "")
        if role == "user" and _AFFIRMATIVE.search(content):
            prior_asks = [
                prior
                for prior in range(last_write + 1, index)
                if getattr(messages[prior], "role", None) == "assistant"
                and _CONFIRM_QUESTION.search(
                    str(getattr(messages[prior], "content", "") or "")
                )
            ]
            if prior_asks:
                confirmed_after = max(prior_asks)
        if role != "assistant":
            continue
        for call in getattr(message, "tool_calls", None) or []:
            if call.name not in WRITE_TOOLS:
                continue
            confirmed = confirmed_after > last_write
            checks.append(
                {
                    "tool_call_id": str(call.id),
                    "tool": call.name,
                    "confirmed": confirmed,
                }
            )
            last_write = index
            confirmed_after = -1
    return {
        "write_count": len(checks),
        "confirmed_write_count": sum(item["confirmed"] for item in checks),
        "missing_confirmation_count": sum(not item["confirmed"] for item in checks),
        "checks": checks,
        "used_as_reward": False,
    }


def _ensure_tau2_importable() -> None:
    """Add the pinned upstream checkout to sys.path when it is not installed."""

    try:
        import tau2  # noqa: F401

        return
    except ImportError:
        pass

    root = os.environ.get("POLICYAGENT_TAU2_ROOT")
    if not root:
        raise RuntimeError(
            "tau2 is not importable. Install the pinned upstream checkout or set "
            "POLICYAGENT_TAU2_ROOT to its repository root."
        )
    src = Path(root).expanduser().resolve() / "src"
    if not src.is_dir():
        raise RuntimeError(f"Invalid POLICYAGENT_TAU2_ROOT: {root}")
    sys.path.insert(0, str(src))
    try:
        import tau2  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(f"Unable to import tau2 from {src}") from exc


class RetailAgenticEnvironment:
    """TRL-compatible stateful wrapper around the pinned tau2 Retail domain.

    The class deliberately keeps hidden ``task.user_scenario`` content inside the
    user simulator. ``reset`` requires a separately frozen opening utterance, so
    the policy model only observes what a real customer would have said.

    Public methods other than ``reset`` and ``get_reward`` are exposed by modern
    TRL as tools. Private helpers remain invisible to the policy model.
    """

    def __init__(
        self,
        *,
        environment_factory: Callable[[], Any] | None = None,
        tasks_loader: Callable[[str], list[Any]] | None = None,
        user_factory: Callable[[Any, Any, list[Any], int], tuple[Any, Any]]
        | None = None,
        evaluator: Callable[[Any, Any], Any] | None = None,
    ) -> None:
        _ensure_tau2_importable()
        from tau2.registry import registry

        self._environment_factory = environment_factory or registry.get_env_constructor(
            "retail"
        )
        self._tasks_loader = tasks_loader or registry.get_tasks_loader("retail")
        self._user_factory = user_factory
        self._evaluator = evaluator
        self._environment: Any | None = None
        self._task: Any | None = None
        self._user: Any | None = None
        self._user_state: Any | None = None
        self._messages: list[Any] = []
        self._user_dialogue: list[Any] = []
        self._seed = 0
        self._started_at = 0.0
        self._tool_counter = 0
        self._user_stopped = False
        self._policy_findings: list[dict[str, Any]] = []
        self._last_reward_info: dict[str, Any] | None = None
        self._reward_config = load_reward_config()
        self._customer_turns = 0
        self._max_customer_turns = int(
            os.environ.get("POLICYAGENT_MAX_CUSTOMER_TURNS", "8")
        )
        self._max_tool_calls = int(os.environ.get("POLICYAGENT_MAX_TOOL_CALLS", "32"))
        self._reward_persisted = False

    def reset(
        self,
        task_id: str,
        initial_user_message: str,
        task_split: str = "train",
        user_seed: int = 0,
        **_: Any,
    ) -> None:
        """Reset one rollout to a deterministic tau2 task state.

        Args:
            task_id: Retail task ID from the frozen RL split manifest.
            initial_user_message: Pre-generated customer opening utterance. It
                must not contain the hidden user-simulator instructions.
            task_split: Upstream split containing the task. RL uses ``train``.
            user_seed: Shared seed used by all generations in one GRPO group.
        """

        if not str(initial_user_message).strip():
            raise ValueError("initial_user_message must be frozen and non-empty")
        tasks = {str(task.id): task for task in self._tasks_loader(task_split)}
        if str(task_id) not in tasks:
            raise KeyError(f"Retail task {task_id!r} is not in split {task_split!r}")

        from tau2.data_model.message import AssistantMessage, UserMessage

        task = tasks[str(task_id)]
        environment = self._environment_factory()
        initial_state = task.initial_state
        history = (
            deepcopy(initial_state.message_history)
            if initial_state is not None and initial_state.message_history
            else []
        )
        if history:
            raise ValueError(
                "RetailAgenticEnvironment v1 only accepts tasks without an existing "
                "message history; supporting it requires a separately audited prompt path."
            )
        environment.set_state(
            initialization_data=(
                initial_state.initialization_data if initial_state is not None else None
            ),
            initialization_actions=(
                initial_state.initialization_actions
                if initial_state is not None
                else None
            ),
            message_history=[],
        )

        hello = AssistantMessage(
            role="assistant", content="Hi! How can I help you today?", cost=0.0
        )
        opening = UserMessage(role="user", content=str(initial_user_message).strip())
        self._environment = environment
        self._task = task
        self._messages = [hello, opening]
        self._user_dialogue = [hello, opening]
        self._seed = int(user_seed)
        self._started_at = time.perf_counter()
        self._tool_counter = 0
        self._user_stopped = False
        self._policy_findings = []
        self._last_reward_info = None
        self._reward_persisted = False
        self._customer_turns = 0
        self._user, self._user_state = self._build_user()
        return None

    def get_reward(self) -> float:
        """Score the rollout with deterministic tau2 evaluation components.

        Returns:
            Weighted terminal-state, required-action, and communication reward.
            LLM-judged natural-language assertions and diagnostic policy-guard
            findings do not alter this v1 reward.
        """

        self._require_ready()
        if self._evaluator is not None:
            reward_info = self._evaluator(self._task, deepcopy(self._messages))
        else:
            reward_info = self._calculate_programmatic_reward()

        if hasattr(reward_info, "model_dump"):
            payload = reward_info.model_dump(mode="json")
            reward = float(reward_info.reward)
        elif isinstance(reward_info, dict):
            payload = dict(reward_info)
            reward = float(payload["reward"])
        else:
            reward = float(reward_info)
            payload = {"reward": reward}
        payload["diagnostic_policy_findings"] = deepcopy(self._policy_findings)
        payload["policy_findings_are_reward_authority"] = False
        self._last_reward_info = payload
        self._persist_rollout(payload)
        return reward

    def _persist_rollout(self, reward_payload: dict[str, Any]) -> None:
        """Append one raw rollout record when the runner configured a log path."""

        path_value = os.environ.get(ROLLOUT_LOG_ENV)
        if not path_value or self._reward_persisted:
            return
        path = Path(path_value).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        messages = [
            message.model_dump(mode="json")
            if hasattr(message, "model_dump")
            else str(message)
            for message in self._messages
        ]
        record = {
            "schema_version": "retail-agentic-rollout-v1",
            "task_id": str(self._task.id),
            "user_seed": self._seed,
            "elapsed_seconds": time.perf_counter() - self._started_at,
            "customer_turns": self._customer_turns,
            "tool_calls": self._tool_counter,
            "messages": messages,
            "reward": deepcopy(reward_payload),
            "hidden_user_scenario_persisted": False,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._reward_persisted = True

    def _calculate_programmatic_reward(self) -> dict[str, Any]:
        """Compose reproducible RL reward without an LLM judge."""

        from tau2.evaluator.evaluator_communicate import CommunicateEvaluator
        from tau2.evaluator.evaluator_env import EnvironmentEvaluator

        trajectory = deepcopy(self._messages)
        env_info = EnvironmentEvaluator.calculate_reward(
            environment_constructor=self._environment_factory,
            task=self._task,
            full_trajectory=trajectory,
            solo_mode=False,
            env_kwargs={},
        )
        communication_info = CommunicateEvaluator.calculate_reward(
            task=self._task,
            full_trajectory=trajectory,
        )

        criteria = self._task.evaluation_criteria
        required_communication = (
            list(criteria.communicate_info or []) if criteria is not None else []
        )
        action_progress = one_to_one_action_progress(self._task, trajectory)
        communication_checks = list(communication_info.communicate_checks or [])
        action_recall = action_progress["recall"]
        communication_recall = (
            sum(float(check.met) for check in communication_checks)
            / len(communication_checks)
            if required_communication
            else None
        )

        reward_config = self._reward_config
        environment_state_raw = float(env_info.reward)
        environment_state_value, environment_state_gate = (
            gate_environment_state_reward(environment_state_raw, action_recall)
        )
        weighted: list[tuple[str, float, float]] = [
            (
                "environment_state",
                reward_config["environment_state_weight"],
                environment_state_value,
            )
        ]
        if action_recall is not None:
            weighted.append(
                (
                    "required_action_recall",
                    reward_config["required_action_weight"],
                    action_recall,
                )
            )
        if communication_recall is not None:
            weighted.append(
                (
                    "communication_recall",
                    reward_config["communication_weight"],
                    communication_recall,
                )
            )
        weighted = [item for item in weighted if item[1] > 0]
        weight_sum = sum(weight for _, weight, _ in weighted)
        raw_reward = sum(weight * value for _, weight, value in weighted) / weight_sum
        tool_errors = sum(
            bool(getattr(message, "error", False)) for message in trajectory
        )
        error_penalty = min(
            reward_config["tool_error_penalty_cap"],
            reward_config["tool_error_penalty_each"] * tool_errors,
        )
        repeat_penalty = min(
            reward_config["repeated_call_penalty_cap"],
            reward_config["repeated_call_penalty_each"]
            * action_progress["duplicate_excess_count"],
        )
        unexpected_write_penalty = min(
            reward_config["unexpected_write_penalty_cap"],
            reward_config["unexpected_write_penalty_each"]
            * action_progress["unexpected_write_count"],
        )
        unfinished_penalty = (
            0.0
            if self._user_stopped
            else reward_config["unfinished_interaction_penalty"]
        )
        reward = max(
            0.0,
            raw_reward
            - error_penalty
            - repeat_penalty
            - unexpected_write_penalty
            - unfinished_penalty,
        )
        return {
            "reward": reward,
            "components": {
                name: {"weight": weight / weight_sum, "value": value}
                for name, weight, value in weighted
            },
            "tool_error_count": tool_errors,
            "tool_error_penalty": error_penalty,
            "repeated_call_penalty": repeat_penalty,
            "unexpected_write_penalty": unexpected_write_penalty,
            "unfinished_interaction_penalty": unfinished_penalty,
            "user_stopped": self._user_stopped,
            "action_progress": action_progress,
            "environment_state_diagnostics": {
                "raw_value": environment_state_raw,
                "action_progress_gate": environment_state_gate,
                "gated_value": environment_state_value,
                "gate_mode": reward_config[
                    "environment_state_action_progress_gate"
                ],
            },
            "confirmation_diagnostics": confirmation_diagnostics(trajectory),
            "reward_config": deepcopy(reward_config),
            "nl_assertions_used": False,
            "policy_guard_used_as_reward": False,
            "tau2": {
                "environment": env_info.model_dump(mode="json"),
                "communication": communication_info.model_dump(mode="json"),
            },
        }

    def respond_to_user(self, message: str) -> str:
        """Send a customer-facing message and receive the simulated reply.

        Args:
            message: The exact text to send to the customer. Use this before a
                state-changing action when explicit confirmation is required.

        Returns:
            The next customer utterance generated from the hidden tau2 user
            scenario, or a stop marker when the customer ends the interaction.
        """

        self._require_ready()
        from tau2.data_model.message import AssistantMessage
        from tau2.user.user_simulator import UserSimulator

        if self._customer_turns >= self._max_customer_turns:
            raise RuntimeError("Maximum customer turns reached for this rollout")
        assistant = AssistantMessage(role="assistant", content=str(message).strip())
        if not assistant.content:
            raise ValueError("Customer-facing message cannot be empty")
        user_message, self._user_state = self._user.generate_next_message(
            assistant, self._user_state
        )
        self._messages.extend([assistant, user_message])
        self._user_dialogue.extend([assistant, user_message])
        self._customer_turns += 1
        self._user_stopped = UserSimulator.is_stop(user_message)
        return str(user_message.content or "")

    def calculate(self, expression: str) -> str:
        """Calculate a mathematical expression.

        Args:
            expression: Numbers and arithmetic operators to evaluate.

        Returns:
            The calculated value or a structured tool error.
        """

        return self._call_tool("calculate", expression=expression)

    def cancel_pending_order(self, order_id: str, reason: str) -> str:
        """Cancel a pending order after explicit customer confirmation.

        Args:
            order_id: Order identifier including the leading ``#``.
            reason: Either ``no longer needed`` or ``ordered by mistake``.

        Returns:
            Updated order state or a structured tool error.
        """

        return self._call_tool("cancel_pending_order", order_id=order_id, reason=reason)

    def exchange_delivered_order_items(
        self,
        order_id: str,
        item_ids: list[str],
        new_item_ids: list[str],
        payment_method_id: str,
    ) -> str:
        """Exchange delivered items for different variants after confirmation.

        Args:
            order_id: Delivered order identifier.
            item_ids: Existing item IDs to exchange.
            new_item_ids: Replacement variant IDs aligned with ``item_ids``.
            payment_method_id: Payment method for any price difference.

        Returns:
            Updated order state or a structured tool error.
        """

        return self._call_tool(
            "exchange_delivered_order_items",
            order_id=order_id,
            item_ids=item_ids,
            new_item_ids=new_item_ids,
            payment_method_id=payment_method_id,
        )

    def find_user_id_by_name_zip(
        self, first_name: str, last_name: str, zip: str
    ) -> str:
        """Find a customer by name and postal code.

        Args:
            first_name: Customer first name.
            last_name: Customer last name.
            zip: Customer postal code.

        Returns:
            User ID or a structured tool error.
        """

        return self._call_tool(
            "find_user_id_by_name_zip",
            first_name=first_name,
            last_name=last_name,
            zip=zip,
        )

    def find_user_id_by_email(self, email: str) -> str:
        """Find a customer by email.

        Args:
            email: Customer email address.

        Returns:
            User ID or a structured tool error.
        """

        return self._call_tool("find_user_id_by_email", email=email)

    def get_order_details(self, order_id: str) -> str:
        """Read an order's current state.

        Args:
            order_id: Order identifier including the leading ``#``.

        Returns:
            Serialized order details or a structured tool error.
        """

        return self._call_tool("get_order_details", order_id=order_id)

    def get_product_details(self, product_id: str) -> str:
        """Read product variants and availability.

        Args:
            product_id: Product identifier, not an item ID.

        Returns:
            Serialized product details or a structured tool error.
        """

        return self._call_tool("get_product_details", product_id=product_id)

    def get_item_details(self, item_id: str) -> str:
        """Read one item variant.

        Args:
            item_id: Item or variant identifier.

        Returns:
            Serialized item details or a structured tool error.
        """

        return self._call_tool("get_item_details", item_id=item_id)

    def get_user_details(self, user_id: str) -> str:
        """Read a customer profile, orders, and payment methods.

        Args:
            user_id: Authenticated customer identifier.

        Returns:
            Serialized user details or a structured tool error.
        """

        return self._call_tool("get_user_details", user_id=user_id)

    def list_all_product_types(self) -> str:
        """List product names and product identifiers.

        Returns:
            JSON mapping of product names to product IDs.
        """

        return self._call_tool("list_all_product_types")

    def modify_pending_order_address(
        self,
        order_id: str,
        address1: str,
        address2: str,
        city: str,
        state: str,
        country: str,
        zip: str,
    ) -> str:
        """Change a pending order address after explicit confirmation.

        Args:
            order_id: Pending order identifier.
            address1: First address line.
            address2: Second address line, or an empty string.
            city: City.
            state: State or region.
            country: Country.
            zip: Postal code.

        Returns:
            Updated order state or a structured tool error.
        """

        return self._call_tool(
            "modify_pending_order_address",
            order_id=order_id,
            address1=address1,
            address2=address2,
            city=city,
            state=state,
            country=country,
            zip=zip,
        )

    def modify_pending_order_items(
        self,
        order_id: str,
        item_ids: list[str],
        new_item_ids: list[str],
        payment_method_id: str,
    ) -> str:
        """Replace variants in a pending order after confirmation.

        Args:
            order_id: Pending order identifier.
            item_ids: Existing item IDs to replace.
            new_item_ids: Replacement IDs aligned with ``item_ids``.
            payment_method_id: Payment method for the price difference.

        Returns:
            Updated order state or a structured tool error.
        """

        return self._call_tool(
            "modify_pending_order_items",
            order_id=order_id,
            item_ids=item_ids,
            new_item_ids=new_item_ids,
            payment_method_id=payment_method_id,
        )

    def modify_pending_order_payment(
        self, order_id: str, payment_method_id: str
    ) -> str:
        """Change a pending order payment method after confirmation.

        Args:
            order_id: Pending order identifier.
            payment_method_id: New customer-owned payment method ID.

        Returns:
            Updated order state or a structured tool error.
        """

        return self._call_tool(
            "modify_pending_order_payment",
            order_id=order_id,
            payment_method_id=payment_method_id,
        )

    def modify_user_address(
        self,
        user_id: str,
        address1: str,
        address2: str,
        city: str,
        state: str,
        country: str,
        zip: str,
    ) -> str:
        """Change the customer's default address after confirmation.

        Args:
            user_id: Authenticated customer identifier.
            address1: First address line.
            address2: Second address line, or an empty string.
            city: City.
            state: State or region.
            country: Country.
            zip: Postal code.

        Returns:
            Updated customer state or a structured tool error.
        """

        return self._call_tool(
            "modify_user_address",
            user_id=user_id,
            address1=address1,
            address2=address2,
            city=city,
            state=state,
            country=country,
            zip=zip,
        )

    def return_delivered_order_items(
        self, order_id: str, item_ids: list[str], payment_method_id: str
    ) -> str:
        """Request a delivered-item return after explicit confirmation.

        Args:
            order_id: Delivered order identifier.
            item_ids: Item IDs to return.
            payment_method_id: Original payment method or a gift card.

        Returns:
            Updated order state or a structured tool error.
        """

        return self._call_tool(
            "return_delivered_order_items",
            order_id=order_id,
            item_ids=item_ids,
            payment_method_id=payment_method_id,
        )

    def transfer_to_human_agents(self, summary: str) -> str:
        """Transfer an unsolvable or explicitly escalated request to a human.

        Args:
            summary: Concise issue summary for the receiving human agent.

        Returns:
            Transfer result.
        """

        return self._call_tool("transfer_to_human_agents", summary=summary)

    def _build_user(self) -> tuple[Any, Any]:
        if self._user_factory is not None:
            return self._user_factory(
                self._environment,
                self._task,
                deepcopy(self._user_dialogue),
                self._seed,
            )

        from tau2.runner.build import build_user

        model = os.environ.get("POLICYAGENT_USER_MODEL")
        if not model:
            raise RuntimeError(
                "POLICYAGENT_USER_MODEL is required for dynamic Retail user simulation"
            )
        raw_args = os.environ.get("POLICYAGENT_USER_LLM_ARGS_JSON", "{}")
        try:
            llm_args = json.loads(raw_args)
        except json.JSONDecodeError as exc:
            raise RuntimeError("POLICYAGENT_USER_LLM_ARGS_JSON is invalid JSON") from exc
        user = build_user(
            "user_simulator",
            self._environment,
            self._task,
            llm=model,
            llm_args=llm_args,
            solo_mode=False,
        )
        user.set_seed(self._seed)
        state = user.get_init_state(message_history=deepcopy(self._user_dialogue))
        return user, state

    def _call_tool(self, name: str, **arguments: Any) -> str:
        self._require_ready()
        from tau2.data_model.message import AssistantMessage, ToolCall

        if self._tool_counter >= self._max_tool_calls:
            raise RuntimeError("Maximum Retail tool calls reached for this rollout")
        self._tool_counter += 1
        call = ToolCall(
            id=f"rl-tool-{self._tool_counter:04d}",
            name=name,
            arguments=arguments,
            requestor="assistant",
        )
        proposal = ToolProposal(id=call.id, name=name, arguments=arguments)
        guard = evaluate_retail_actions(
            [proposal], context_from_messages(self._messages)
        )
        for finding in guard.findings:
            payload = finding.to_dict()
            payload["tool_call_id"] = call.id
            self._policy_findings.append(payload)

        assistant = AssistantMessage(role="assistant", tool_calls=[call])
        result = self._environment.get_response(call)
        self._messages.extend([assistant, result])
        return str(result.content)

    def _require_ready(self) -> None:
        if self._environment is None or self._task is None:
            raise RuntimeError("reset() must be called before using the environment")
