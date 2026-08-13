from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tau2.agent.llm_agent import LLMAgent, LLMAgentStateType
from tau2.data_model.message import MultiToolMessage, UserMessage
from tau2.utils.llm_utils import generate


PROMPT_AUDIT_LOG_ENV = "POLICYAGENT_TEACHER_PROMPT_AUDIT_LOG"
FORBIDDEN_GOLD_MARKERS = (
    "<resolution_steps>",
    "evaluation_criteria",
    "action_id",
    "gold_database_state",
    "nl_assertions",
    "reward_basis",
)


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


class AuditedTeacherLLMAgent(LLMAgent):
    """Normal tau2 LLMAgent with an evidence-only outbound request audit.

    The generation call is intentionally equivalent to upstream LLMAgent. The
    wrapper records the exact request payload in a private artifact and scans
    structural gold markers before the request is sent. It never injects task
    evaluation criteria into model messages.
    """

    def __init__(self, *args: Any, audit_task_id: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.audit_task_id = str(audit_task_id)
        self.audit_turn = 0
        self.audit_seed: int | None = None

    def set_seed(self, seed: int):
        self.audit_seed = int(seed)
        return super().set_seed(seed)

    def _append_audit(self, payload: dict[str, Any]) -> None:
        value = os.environ.get(PROMPT_AUDIT_LOG_ENV, "").strip()
        if not value:
            raise RuntimeError(f"{PROMPT_AUDIT_LOG_ENV} must be configured")
        path = Path(value).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

    def _generate_next_message(
        self, message: Any, state: LLMAgentStateType
    ):
        if isinstance(message, UserMessage) and message.is_audio:
            raise ValueError("User message cannot be audio. Use VoiceLLMAgent instead.")
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)
        messages = state.system_messages + state.messages
        serialized_messages = [_dump(item) for item in messages]
        serialized_tools = [_dump(tool.openai_schema) for tool in self.tools]
        searchable = json.dumps(serialized_messages, ensure_ascii=False).lower()
        marker_hits = [marker for marker in FORBIDDEN_GOLD_MARKERS if marker in searchable]
        request = {
            "model": self.llm,
            "messages": serialized_messages,
            "tools": serialized_tools,
            "llm_args": self.llm_args,
        }
        assistant_message = generate(
            model=self.llm,
            tools=self.tools,
            messages=messages,
            call_name="agent_response",
            **self.llm_args,
        )
        raw = getattr(assistant_message, "raw_data", None) or {}
        response_model = raw.get("model") if isinstance(raw, dict) else None
        fingerprint = (
            raw.get("system_fingerprint") if isinstance(raw, dict) else None
        )
        self._append_audit(
            {
                "schema_version": "teacher-outbound-prompt-audit-v1",
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "task_id": self.audit_task_id,
                "seed": self.audit_seed,
                "turn": self.audit_turn,
                "agent_class": type(self).__name__,
                "upstream_behavior_base": "tau2.agent.llm_agent.LLMAgent",
                "request_sha256": _sha256_json(request),
                "system_prompt_sha256": _sha256_json(serialized_messages[0]),
                "tool_schema_sha256": _sha256_json(serialized_tools),
                "forbidden_gold_marker_hits": marker_hits,
                "gold_visibility_check_passed": not marker_hits,
                "private_request_evidence": request,
                "response": {
                    "requested_model": self.llm,
                    "reported_model": response_model,
                    "system_fingerprint": fingerprint,
                    "assistant_message_sha256": _sha256_json(_dump(assistant_message)),
                },
            }
        )
        self.audit_turn += 1
        return assistant_message


def create_audited_teacher_agent(tools: Any, domain_policy: str, **kwargs: Any):
    task = kwargs.get("task")
    if task is None:
        raise ValueError("Audited teacher agent requires the current task for audit ID")
    return AuditedTeacherLLMAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
        audit_task_id=str(task.id),
    )


def register_audited_teacher_agent(
    name: str = "audited_teacher_llm_agent",
) -> None:
    from tau2.registry import registry

    if registry.get_agent_factory(name) is None:
        registry.register_agent_factory(create_audited_teacher_agent, name)
