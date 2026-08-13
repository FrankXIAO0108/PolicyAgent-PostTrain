from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.rl.user_simulator_fail_fast import (
    SYSTEM_FAILURE_LOG_ENV,
    UserSimulatorSystemFailure,
    classify_user_simulator_exception,
    generate_with_fail_fast,
    probe_user_simulator_api,
    sanitize_error_message,
)
from src.training import run_retail_agentic_grpo as agentic_runner


class _HttpError(RuntimeError):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class UserSimulatorFailFastTests(unittest.TestCase):
    def test_authentication_error_is_fatal_and_redacted(self) -> None:
        fake_key = "sk-" + "secret-value"
        error = _HttpError(f"Authentication Fails for {fake_key}", 401)
        classification = classify_user_simulator_exception(error)
        self.assertEqual(classification.category, "AUTHENTICATION_FAILED")
        self.assertFalse(classification.retryable)
        self.assertNotIn("secret-value", sanitize_error_message(str(error)))

    def test_transient_error_retries_then_succeeds(self) -> None:
        attempts = []

        def generate(message, state):
            del message
            attempts.append(1)
            if len(attempts) < 3:
                raise TimeoutError("temporary timeout")
            return SimpleNamespace(content="Here is the customer reply."), state + 1

        reply, state = generate_with_fail_fast(
            generate,
            object(),
            0,
            task_id="7",
            user_seed=11,
            retry_delays_seconds=(0.0, 0.0),
        )
        self.assertEqual(reply.content, "Here is the customer reply.")
        self.assertEqual(state, 1)
        self.assertEqual(len(attempts), 3)

    def test_empty_response_is_recorded_and_aborts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "system_failures.jsonl"
            with patch.dict(os.environ, {SYSTEM_FAILURE_LOG_ENV: str(path)}):
                with self.assertRaises(UserSimulatorSystemFailure) as raised:
                    generate_with_fail_fast(
                        lambda message, state: (SimpleNamespace(content=""), state),
                        object(),
                        {},
                        task_id="20",
                        user_seed=99,
                    )
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "SYSTEM_FAILURE")
            self.assertEqual(payload["task_id"], "20")
            self.assertEqual(raised.exception.category, "INVALID_RESPONSE")

    def test_preflight_passes_without_persisting_response_or_key(self) -> None:
        captured = {}

        def completion(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message="OK")])

        result = probe_user_simulator_api(
            model="deepseek/deepseek-v4-flash", completion_fn=completion
        )
        self.assertEqual(result["status"], "PASSED")
        self.assertEqual(result["max_tokens"], 1)
        self.assertNotIn("messages", result)
        self.assertEqual(captured["max_tokens"], 1)

    def test_preflight_failure_writes_system_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "system_failures.jsonl"

            def completion(**kwargs):
                del kwargs
                fake_key = "sk-" + "do-not-store-this"
                raise _HttpError(f"invalid api key {fake_key}", 401)

            with patch.dict(os.environ, {SYSTEM_FAILURE_LOG_ENV: str(path)}):
                with self.assertRaises(UserSimulatorSystemFailure):
                    probe_user_simulator_api(model="deepseek/test", completion_fn=completion)
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("do-not-store-this", text)
            payload = json.loads(text)
            self.assertEqual(payload["stage"], "USER_SIMULATOR_PREFLIGHT")
            self.assertEqual(payload["category"], "AUTHENTICATION_FAILED")

    def test_runner_probes_api_before_runtime_or_gpu_loading(self) -> None:
        preflight = {
            "config": {
                "reward": {},
                "rollout": {"max_customer_turns": 2, "max_tool_calls": 4},
            }
        }
        failure = UserSimulatorSystemFailure(
            "AUTHENTICATION_FAILED",
            "invalid api key " + "sk-" + "never-persist-this",
            attempts=1,
            abort_run=True,
        )
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"POLICYAGENT_USER_MODEL": "deepseek/test"}, clear=False
        ), patch.object(
            agentic_runner, "probe_user_simulator_api", side_effect=failure
        ) as probe, patch.object(agentic_runner, "check_runtime") as runtime:
            with self.assertRaises(UserSimulatorSystemFailure):
                agentic_runner.run(preflight, Path(directory) / "run")
        probe.assert_called_once_with(model="deepseek/test")
        runtime.assert_not_called()


if __name__ == "__main__":
    unittest.main()
