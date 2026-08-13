from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tau2.data_model.message import AssistantMessage, UserMessage

from src.agents.audited_teacher_agent import (
    PROMPT_AUDIT_LOG_ENV,
    AuditedTeacherLLMAgent,
)


class AuditedTeacherAgentTests(unittest.TestCase):
    def test_records_exact_request_hash_and_passes_clean_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompt_audit.jsonl"
            agent = AuditedTeacherLLMAgent(
                tools=[],
                domain_policy="Authenticate the user before reading an order.",
                llm="teacher/model",
                llm_args={"temperature": 0.4},
                audit_task_id="57",
            )
            agent.set_seed(123)
            state = agent.get_init_state()
            response = AssistantMessage.text("How may I help?")
            with patch.dict(
                "os.environ", {PROMPT_AUDIT_LOG_ENV: str(path)}, clear=False
            ), patch("src.agents.audited_teacher_agent.generate", return_value=response):
                actual = agent._generate_next_message(
                    UserMessage(role="user", content="Where is my order?"), state
                )
            self.assertEqual(actual.content, "How may I help?")
            row = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(row["task_id"], "57")
            self.assertEqual(row["seed"], 123)
            self.assertTrue(row["gold_visibility_check_passed"])
            self.assertEqual(row["forbidden_gold_marker_hits"], [])
            self.assertEqual(row["private_request_evidence"]["model"], "teacher/model")
            self.assertNotIn("evaluation_criteria", json.dumps(row["private_request_evidence"]))

    def test_structural_gold_marker_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompt_audit.jsonl"
            agent = AuditedTeacherLLMAgent(
                tools=[],
                domain_policy="<resolution_steps>cancel order</resolution_steps>",
                llm="teacher/model",
                audit_task_id="113",
            )
            state = agent.get_init_state()
            with patch.dict(
                "os.environ", {PROMPT_AUDIT_LOG_ENV: str(path)}, clear=False
            ), patch(
                "src.agents.audited_teacher_agent.generate",
                return_value=AssistantMessage.text("ok"),
            ):
                agent._generate_next_message(UserMessage(role="user", content="hi"), state)
            row = json.loads(path.read_text(encoding="utf-8"))
            self.assertFalse(row["gold_visibility_check_passed"])
            self.assertIn("<resolution_steps>", row["forbidden_gold_marker_hits"])


if __name__ == "__main__":
    unittest.main()
