from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tau2.agent.llm_agent import LLMAgent, LLMAgentStateType
from tau2.data_model.message import AssistantMessage, SystemMessage
from tau2.utils.llm_utils import generate

from src.guards.retail_pre_action import (
    GuardDecision,
    ToolProposal,
    context_from_messages,
    evaluate_retail_actions,
)


class GuardedLLMAgent(LLMAgent):
    """Tau2-compatible LLM agent with a deterministic pre-execution boundary.

    This initial adapter enforces rules that can be decided from a proposed
    message alone. Rich context is covered by the pure guard and offline audit;
    live context hydration will be added before a paid rerun.
    """

    def __init__(
        self,
        *args: Any,
        guard_mode: str = "enforce",
        guard_max_retries: int = 1,
        guard_trace_path: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.guard_mode = guard_mode
        self.guard_max_retries = guard_max_retries
        self.guard_trace_path = (
            Path(guard_trace_path).resolve() if guard_trace_path else None
        )

    def _record_guard_event(self, payload: dict[str, Any]) -> None:
        if self.guard_trace_path is None:
            return
        self.guard_trace_path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "schema_version": "guard-live-trace-v1.0.0",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        with self.guard_trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    @staticmethod
    def _tool_proposals(message: AssistantMessage) -> list[ToolProposal]:
        return [
            ToolProposal(
                id=call.id,
                name=call.name,
                arguments=call.arguments,
            )
            for call in message.tool_calls or []
        ]

    def _generate_next_message(
        self,
        message: Any,
        state: LLMAgentStateType,
    ) -> AssistantMessage:
        proposal = super()._generate_next_message(message, state)
        if self.guard_mode != "enforce":
            return proposal

        context = context_from_messages(list(state.messages))
        for retry_index in range(self.guard_max_retries + 1):
            result = evaluate_retail_actions(
                self._tool_proposals(proposal),
                context,
                assistant_content=str(proposal.content or ""),
            )
            self._record_guard_event(
                {
                    "event": "proposal_evaluated",
                    "retry_index": retry_index,
                    "allowed": result.allowed,
                    "decision": result.decision.value,
                    "tool_proposals": [
                        {
                            "name": item.name,
                            "arguments": item.arguments,
                        }
                        for item in self._tool_proposals(proposal)
                    ],
                    "blocking_findings": [
                        finding.to_dict()
                        for finding in result.findings
                        if finding.blocking
                    ],
                }
            )
            if result.allowed:
                return proposal
            if result.decision in {
                GuardDecision.REQUIRE_CONFIRMATION,
                GuardDecision.BLOCK,
                GuardDecision.TRANSFER,
            }:
                break
            if retry_index >= self.guard_max_retries:
                break

            feedback = {
                "status": "rejected_before_execution",
                "violations": [
                    finding.to_dict()
                    for finding in result.findings
                    if finding.blocking
                ],
                "instruction": (
                    "Generate a corrected next turn. Do not claim that rejected "
                    "tools ran. Use at most one tool call and request missing scope "
                    "confirmation when required."
                ),
            }
            proposal = generate(
                model=self.llm,
                tools=self.tools,
                messages=[
                    *state.system_messages,
                    *state.messages,
                    SystemMessage(
                        role="system",
                        content=(
                            "<guard_feedback>"
                            f"{json.dumps(feedback, ensure_ascii=False)}"
                            "</guard_feedback>"
                        ),
                    ),
                ],
                call_name="guard_retry",
                **self.llm_args,
            )

        reasons = "; ".join(
            finding.message for finding in result.findings if finding.blocking
        )
        if result.decision == GuardDecision.REQUIRE_CONFIRMATION:
            return AssistantMessage.text(
                "Before I proceed, I need your explicit confirmation because the "
                f"available tool has a broader effect: {reasons}"
            )
        if result.decision == GuardDecision.TRANSFER:
            return AssistantMessage.text(
                "I cannot safely complete this request with the available tools. "
                "I need to transfer you to a human agent."
            )
        return AssistantMessage.text(
            "I need to pause before executing that action because a safety check "
            f"failed: {reasons} No rejected action was executed."
        )


def create_guarded_llm_agent(tools: Any, domain_policy: str, **kwargs: Any):
    llm_args = dict(kwargs.get("llm_args") or {})
    guard_mode = str(llm_args.pop("guard_mode", "enforce"))
    guard_max_retries = int(llm_args.pop("guard_max_retries", 1))
    guard_trace_path = llm_args.pop("guard_trace_path", None)
    return GuardedLLMAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=llm_args,
        guard_mode=guard_mode,
        guard_max_retries=guard_max_retries,
        guard_trace_path=guard_trace_path,
    )


def register_guarded_llm_agent(name: str = "guarded_llm_agent") -> None:
    from tau2.registry import registry

    if registry.get_agent_factory(name) is None:
        registry.register_agent_factory(create_guarded_llm_agent, name)
