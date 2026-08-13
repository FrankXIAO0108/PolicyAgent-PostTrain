from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


SYSTEM_FAILURE_LOG_ENV = "POLICYAGENT_SYSTEM_FAILURE_LOG"
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_RETRY_DELAYS_SECONDS = (1.0, 2.0)


@dataclass(frozen=True)
class FailureClassification:
    category: str
    retryable: bool
    abort_run: bool


class UserSimulatorSystemFailure(RuntimeError):
    """A non-agent failure that invalidates the current rollout or run."""

    def __init__(
        self,
        category: str,
        message: str,
        *,
        attempts: int,
        abort_run: bool,
    ) -> None:
        super().__init__(f"{category}: {sanitize_error_message(message)}")
        self.category = category
        self.attempts = attempts
        self.abort_run = abort_run


def sanitize_error_message(message: str) -> str:
    """Remove credentials and keep failure artifacts bounded."""

    value = str(message)
    value = re.sub(r"sk-[A-Za-z0-9_-]{8,}", "sk-<redacted>", value)
    value = re.sub(
        r"(?i)(api[_ -]?key\s*[:=]\s*)[^\s,;}]+", r"\1<redacted>", value
    )
    return value[:1000]


def classify_user_simulator_exception(exc: BaseException) -> FailureClassification:
    status = getattr(exc, "status_code", None)
    message = f"{type(exc).__name__}: {exc}".lower()
    if "empty response" in message or "returned no choices" in message:
        return FailureClassification("INVALID_RESPONSE", False, True)
    if status in (401, 403) or any(
        token in message
        for token in ("authentication", "unauthorized", "invalid api key")
    ):
        return FailureClassification("AUTHENTICATION_FAILED", False, True)
    if status == 402 or any(
        token in message
        for token in ("insufficient balance", "insufficient quota", "payment required")
    ):
        return FailureClassification("QUOTA_OR_BALANCE_EXHAUSTED", False, True)
    if status == 429 or "rate limit" in message or "ratelimit" in message:
        return FailureClassification("RATE_LIMITED", True, True)
    if status is not None and int(status) >= 500:
        return FailureClassification("UPSTREAM_SERVER_ERROR", True, True)
    if any(
        token in message
        for token in ("timeout", "timed out", "connection", "temporarily unavailable")
    ):
        return FailureClassification("UPSTREAM_NETWORK_ERROR", True, True)
    return FailureClassification("UPSTREAM_REQUEST_FAILED", False, True)


def _append_failure(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _failure_log_path() -> Path | None:
    value = os.environ.get(SYSTEM_FAILURE_LOG_ENV)
    return Path(value).expanduser().resolve() if value else None


def generate_with_fail_fast(
    generate_fn: Callable[[Any, Any], tuple[Any, Any]],
    message: Any,
    state: Any,
    *,
    task_id: str,
    user_seed: int,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    retry_delays_seconds: tuple[float, ...] = DEFAULT_RETRY_DELAYS_SECONDS,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> tuple[Any, Any]:
    """Call the dynamic user with bounded retries and fail closed."""

    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    failures: list[dict[str, Any]] = []
    for attempt in range(1, max_attempts + 1):
        try:
            user_message, next_state = generate_fn(message, state)
            content = str(getattr(user_message, "content", "") or "").strip()
            if not content:
                raise ValueError("User simulator returned an empty response")
            return user_message, next_state
        except Exception as exc:
            classification = classify_user_simulator_exception(exc)
            failures.append(
                {
                    "attempt": attempt,
                    "category": classification.category,
                    "exception_type": type(exc).__name__,
                    "message": sanitize_error_message(str(exc)),
                    "retryable": classification.retryable,
                }
            )
            should_retry = classification.retryable and attempt < max_attempts
            if should_retry:
                delay_index = min(attempt - 1, len(retry_delays_seconds) - 1)
                if retry_delays_seconds:
                    sleep_fn(retry_delays_seconds[delay_index])
                continue
            payload = {
                "schema_version": "user-simulator-system-failure-v1",
                "status": "SYSTEM_FAILURE",
                "stage": "USER_SIMULATOR_RUNTIME",
                "task_id": str(task_id),
                "user_seed": int(user_seed),
                "category": classification.category,
                "attempts": failures,
                "abort_run": classification.abort_run,
            }
            _append_failure(_failure_log_path(), payload)
            raise UserSimulatorSystemFailure(
                classification.category,
                str(exc),
                attempts=attempt,
                abort_run=classification.abort_run,
            ) from None
    raise AssertionError("unreachable")


def probe_user_simulator_api(
    *,
    model: str,
    completion_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Perform a minimal paid request before loading the local GPU model."""

    if not model.strip():
        raise UserSimulatorSystemFailure(
            "CONFIGURATION_ERROR", "User simulator model is empty", attempts=0, abort_run=True
        )
    if completion_fn is None:
        from litellm import completion

        completion_fn = completion
    started = time.perf_counter()
    try:
        response = completion_fn(
            model=model,
            messages=[{"role": "user", "content": "Reply with OK."}],
            temperature=0.0,
            max_tokens=1,
            timeout=45,
        )
        choices = getattr(response, "choices", None)
        if not choices and isinstance(response, dict):
            choices = response.get("choices")
        if not choices:
            raise ValueError("User simulator preflight returned no choices")
    except Exception as exc:
        classification = classify_user_simulator_exception(exc)
        payload = {
            "schema_version": "user-simulator-system-failure-v1",
            "status": "SYSTEM_FAILURE",
            "stage": "USER_SIMULATOR_PREFLIGHT",
            "category": classification.category,
            "exception_type": type(exc).__name__,
            "message": sanitize_error_message(str(exc)),
            "abort_run": True,
        }
        _append_failure(_failure_log_path(), payload)
        raise UserSimulatorSystemFailure(
            classification.category,
            str(exc),
            attempts=1,
            abort_run=True,
        ) from None
    return {
        "schema_version": "user-simulator-api-preflight-v1",
        "status": "PASSED",
        "model": model,
        "max_tokens": 1,
        "elapsed_seconds": time.perf_counter() - started,
        "external_api_called": True,
        "credential_persisted": False,
    }
