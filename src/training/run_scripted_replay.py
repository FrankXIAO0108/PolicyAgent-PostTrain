from __future__ import annotations

'''ENVIRONMENT_REPLAY driver for the 2026-08-21 correction-target batch.

Replays the 7 frozen correction-target specs (36/38/59/76/101/105/107) as
true environment replays: a scripted policy-compliant agent follows the spec
plan while the frozen user_simulator (temperature 0.0, same seed) and the
real tau2 Retail environment generate every user reply and tool result.
Tool results are never hand-written.

Usage (validate-only, no API calls):
    python -m src.training.run_scripted_replay --spec-dir ... --validate-only

Full run (needs a user-simulator LLM, e.g. DeepSeek API on the remote box):
    python -m src.training.run_scripted_replay --spec-dir ... --output-dir ... \
        --llm-user deepseek/deepseek-chat --path-remap OLD=NEW
'''

import argparse
import hashlib
import json
import os
import random
import sys
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.training.run_tau2_teacher_trajectory_smoke import (
    REPO_ROOT,
    git_value,
    sha256,
    write_json,
)

AGENT_NAME = 'scripted_policy_compliant_agent_v1'
SCOPE = 'CORRECTION_TARGET_REPLAY_20260821'
SCHEMA_TARGET = 'retail-confirmation-correction-target-v1'
SCHEMA_MANIFEST = 'retail-confirmation-correction-targets-manifest-v1'
DEFAULT_SPEC_DIR = (
    REPO_ROOT / '_local_private_runs' / 'correction_targets_20260821'
)
UPSTREAM_COMMIT = '58e5e1ace69302e6982d27014569c03e0ffccdd2'
STOP_TOKEN = '###STOP###'
GREETING = 'Hi! How can I help you today?'
ABORT_TEXT = (
    'I was unable to confirm an allowed cancellation reason, so I did not '
    'cancel the order. The order remains unchanged. Is there anything else '
    'I can help you with?'
)

ALLOWED_TOOLS = {
    'calculate',
    'cancel_pending_order',
    'exchange_delivered_order_items',
    'find_user_id_by_name_zip',
    'find_user_id_by_email',
    'get_order_details',
    'get_product_details',
    'get_item_details',
    'get_user_details',
    'list_all_product_types',
    'modify_pending_order_address',
    'modify_pending_order_items',
    'modify_pending_order_payment',
    'modify_user_address',
    'return_delivered_order_items',
    'transfer_to_human_agents',
}
SUPPORTED_OPS = {
    'keep',
    'remove',
    'assistant_text',
    'assistant_text_line_edit',
    'tool_call',
    'user_reply_expected',
    'branch_on_user_reply',
}
STEP_EMIT_TEXT = 'EMIT_TEXT'
STEP_EMIT_TOOL_CALL = 'EMIT_TOOL_CALL'
STEP_EXPECT_USER = 'EXPECT_USER'
STEP_EXPECT_TOOL_RESULT = 'EXPECT_TOOL_RESULT'
STEP_EXPECT_BRANCH = 'EXPECT_BRANCH'


def sha256_lf(path: Path) -> str:
    '''Hash with BOM stripped and CRLF normalized to LF (manifest semantics).'''
    data = path.read_bytes()
    if data.startswith(b'\xef\xbb\xbf'):
        data = data[3:]
    data = data.replace(b'\r\n', b'\n')
    return hashlib.sha256(data).hexdigest().upper()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8-sig'))


def remap_path(raw: str, remaps: list[tuple[str, str]]) -> str:
    '''Apply OLD=NEW prefix remaps (case-insensitive) to an absolute path.'''
    for old, new in remaps:
        if raw.lower().startswith(old.lower()):
            return new + raw[len(old):]
    return raw


@dataclass
class ReplayStep:
    kind: str
    source: str  # keep | insert | edit | retry | abort | implicit
    frozen_index: Optional[int] = None
    text: Optional[str] = None
    tool_name: Optional[str] = None
    tool_arguments: Optional[dict] = field(default=None)
    tool_call_id: Optional[str] = None
    frozen_user_index: Optional[int] = None
    frozen_result_index: Optional[int] = None
    branch_rules: Optional[list[dict]] = field(default=None)
    branch_fallback: Optional[str] = None
    retry_text: Optional[str] = None
    note: str = ''


def _plan_ops(spec: dict[str, Any]) -> list[dict[str, Any]]:
    ops: list[dict[str, Any]] = []
    for entry in spec.get('plan', []):
        op = entry.get('op')
        if op not in SUPPORTED_OPS:
            raise ValueError(f'unsupported plan op {op!r}')
        if op in ('keep', 'remove'):
            events = entry.get('events')
            if not (
                isinstance(events, list)
                and len(events) == 2
                and int(events[0]) <= int(events[1])
            ):
                raise ValueError(f'{op} events must be an inclusive [start, end]')
            ops.append(
                {
                    'kind': op,
                    'start': int(events[0]),
                    'end': int(events[1]),
                    'entry': entry,
                }
            )
        elif op == 'assistant_text_line_edit':
            ops.append(
                {
                    'kind': op,
                    'event_index': int(entry['event_index']),
                    'entry': entry,
                }
            )
        else:
            ops.append(
                {
                    'kind': op,
                    'anchor': int(entry['after_event']),
                    'entry': entry,
                }
            )
    return ops


def _check_alternation(steps: list[ReplayStep]) -> list[str]:
    '''Validate the corrected-trajectory message sequence alternation.

    Every emit_tool_call is followed by an environment-generated tool result
    (implicit EXPECT_TOOL_RESULT). expect_user/expect_branch must follow an
    assistant text; emit_tool_call may follow a user reply, a branch, or a
    previous tool result. No two assistant emits may be consecutive.
    '''
    issues: list[str] = []
    sequence: list[str] = []
    for i, step in enumerate(steps):
        sequence.append(step.kind)
        if step.kind == STEP_EMIT_TOOL_CALL:
            nxt = steps[i + 1].kind if i + 1 < len(steps) else None
            if nxt != STEP_EXPECT_TOOL_RESULT:
                sequence.append(STEP_EXPECT_TOOL_RESULT)
    previous: Optional[str] = STEP_EMIT_TEXT  # the orchestrator pre-emits the greeting
    for kind in sequence:
        if kind == STEP_EMIT_TOOL_CALL:
            if previous not in (
                None,
                STEP_EXPECT_USER,
                STEP_EXPECT_BRANCH,
                STEP_EXPECT_TOOL_RESULT,
            ):
                issues.append(f'emit_tool_call after invalid predecessor {previous}')
        elif kind in (STEP_EXPECT_USER, STEP_EXPECT_BRANCH):
            if previous != STEP_EMIT_TEXT:
                issues.append(f'{kind} after {previous}, expected EMIT_TEXT')
        elif kind == STEP_EXPECT_TOOL_RESULT:
            if previous != STEP_EMIT_TOOL_CALL:
                issues.append(
                    f'expect_tool_result after {previous}, expected EMIT_TOOL_CALL'
                )
        elif kind == STEP_EMIT_TEXT:
            if previous in (STEP_EMIT_TEXT, STEP_EMIT_TOOL_CALL):
                issues.append('consecutive assistant emits without an expect step')
        previous = kind
    return issues

def _derive_seed(seed_source: int) -> int:
    '''Replicate the batch runner trial-seed derivation for num_trials=1.'''
    random.seed(seed_source)
    return random.randint(0, 1000000)


def build_steps(
    spec: dict[str, Any], frozen: list[dict[str, Any]]
) -> tuple[Optional[list[ReplayStep]], list[str], list[str]]:
    '''Expand a spec plan into a linear list of replay steps.

    keep/remove reference frozen event indexes; anchored ops (assistant_text,
    tool_call, user_reply_expected, branch_on_user_reply) fire right after
    their anchor event is consumed, in plan order. Removed events (and their
    orphaned tool results) produce no steps. Returns (steps, errors, warnings).
    '''
    n = len(frozen)
    errors: list[str] = []
    warnings: list[str] = []
    ops = _plan_ops(spec)
    cover: dict[int, list[dict[str, Any]]] = {}
    for op in ops:
        if op['kind'] in ('keep', 'remove'):
            for i in range(op['start'], op['end'] + 1):
                cover.setdefault(i, []).append(op)
        elif op['kind'] == 'assistant_text_line_edit':
            cover.setdefault(op['event_index'], []).append(op)

    for i in range(n):
        if frozen[i].get('role') != 'assistant':
            continue
        ops_i = cover.get(i, [])
        if not ops_i:
            errors.append(
                f'event {i}: assistant event not covered by any keep/remove/line_edit'
            )
        elif len(ops_i) > 1:
            errors.append(f'event {i}: assistant event covered by {len(ops_i)} ops')

    if frozen[0].get('role') != 'assistant' or (
        (frozen[0].get('content') or '').strip() != GREETING
    ):
        errors.append('event 0 must be the assistant greeting')
    c0 = cover.get(0, [])
    if not c0 or c0[0]['kind'] != 'keep':
        errors.append('event 0 must be covered by keep')

    for op in ops:
        entry = op['entry']
        kind = op['kind']
        if kind == 'assistant_text_line_edit':
            i = op['event_index']
            if not (0 <= i < n) or frozen[i].get('role') != 'assistant':
                errors.append(f'line_edit event_index {i} out of range or not assistant')
                continue
            content = frozen[i].get('content') or ''
            old = entry.get('old', '')
            new = entry.get('new', '')
            if not old or old not in content:
                errors.append(f'line_edit event {i}: old text not found in frozen content')
            if not new:
                errors.append(f'line_edit event {i}: new text empty')
            continue
        if kind in ('keep', 'remove'):
            continue
        anchor = op['anchor']
        if not (0 <= anchor < n):
            errors.append(f'{kind} after_event {anchor} out of range')
        if kind == 'tool_call':
            tool = entry.get('tool')
            if tool not in ALLOWED_TOOLS:
                errors.append(f'tool_call uses unknown tool {tool!r}')
            if not isinstance(entry.get('arguments'), dict):
                errors.append(f'tool_call {tool} arguments must be an object')
        elif kind == 'branch_on_user_reply':
            if entry.get('fallback') != 'ask_once_more_then_abort':
                errors.append(f"branch fallback {entry.get('fallback')!r} unsupported")
            if not entry.get('retry_text'):
                errors.append('branch retry_text missing')
            for rule in entry.get('rules') or []:
                if not rule.get('contains') or not rule.get('set_reason'):
                    errors.append('branch rule needs contains and set_reason')
    if errors:
        return None, errors, warnings

    steps: list[ReplayStep] = []
    anchored: dict[int, list[dict[str, Any]]] = {}
    for op in ops:
        if op['kind'] in ('keep', 'remove', 'assistant_text_line_edit'):
            continue
        anchored.setdefault(op['anchor'], []).append(op)

    for i in range(n):
        msg = frozen[i]
    for i in range(n):
        msg = frozen[i]
        role = msg.get('role')
        ops_i = cover.get(i, [])
        op0 = ops_i[0] if ops_i else None
        if role == 'assistant':
            if op0 is not None and op0['kind'] != 'remove':
                kind0 = op0['kind']
                if kind0 == 'keep':
                    if i == 0:
                        # The greeting is pre-emitted by the orchestrator; the
                        # agent must not re-emit it. Event 0 is still
                        # coverage-checked above.
                        pass
                    else:
                        tool_calls = msg.get('tool_calls') or []
                        if tool_calls:
                            tc = tool_calls[0]
                            if msg.get('content'):
                                warnings.append(
                                    f'event {i}: frozen assistant message mixes text and '
                                    'tool call; replay emits the tool call only '
                                    '(protocol requires no mixing)'
                                )
                            steps.append(
                                ReplayStep(
                                    kind=STEP_EMIT_TOOL_CALL,
                                    source='keep',
                                    frozen_index=i,
                                    tool_name=tc.get('name'),
                                    tool_arguments=dict(tc.get('arguments') or {}),
                                    tool_call_id=tc.get('id') or None,
                                    note='replay frozen tool call',
                                )
                            )
                        else:
                            steps.append(
                                ReplayStep(
                                    kind=STEP_EMIT_TEXT,
                                    source='keep',
                                    frozen_index=i,
                                    text=msg.get('content') or '',
                                    note='replay frozen assistant text',
                                )
                            )
                else:  # assistant_text_line_edit
                    entry = op0['entry']
                    content = msg.get('content') or ''
                    steps.append(
                        ReplayStep(
                            kind=STEP_EMIT_TEXT,
                            source='edit',
                            frozen_index=i,
                            text=content.replace(entry['old'], entry['new']),
                            note='line-edited frozen assistant text',
                        )
                    )
        elif role == 'user':
            if op0 is not None and op0['kind'] == 'keep':
                steps.append(
                    ReplayStep(
                        kind=STEP_EXPECT_USER,
                        source='keep',
                        frozen_index=i,
                        frozen_user_index=i,
                        note='frozen user reply (prefix check)',
                    )
                )
        elif role == 'tool':
            if op0 is not None and op0['kind'] == 'keep':
                steps.append(
                    ReplayStep(
                        kind=STEP_EXPECT_TOOL_RESULT,
                        source='keep',
                        frozen_index=i,
                        frozen_result_index=i,
                        note='frozen tool result (soft compare)',
                    )
                )
        # Anchored inserts are emitted after the frozen event they anchor to,
        # even when that event itself is removed (the insert takes the
        # removed event's position in the corrected trajectory).
        anchored_ops = anchored.get(i, [])
        for k, op in enumerate(anchored_ops):
            entry = op['entry']
            kind0 = op['kind']
            if (
                kind0 == 'user_reply_expected'
                and k + 1 < len(anchored_ops)
                and anchored_ops[k + 1]['kind'] == 'branch_on_user_reply'
            ):
                # The branch consumes the same user reply; a plain
                # expectation directly before it is redundant.
                warnings.append(
                    f'event {i}: user_reply_expected directly before '
                    'branch_on_user_reply is subsumed by the branch'
                )
                continue
            if kind0 == 'assistant_text':
                steps.append(
                    ReplayStep(kind=STEP_EMIT_TEXT, source='insert', text=entry['text'])
                )
            elif kind0 == 'tool_call':
                steps.append(
                    ReplayStep(
                        kind=STEP_EMIT_TOOL_CALL,
                        source='insert',
                        tool_name=entry['tool'],
                        tool_arguments=dict(entry.get('arguments') or {}),
                    )
                )
            elif kind0 == 'user_reply_expected':
                steps.append(ReplayStep(kind=STEP_EXPECT_USER, source='expect'))
            elif kind0 == 'branch_on_user_reply':
                steps.append(
                    ReplayStep(
                        kind=STEP_EXPECT_BRANCH,
                        source='branch',
                        branch_rules=list(entry.get('rules') or []),
                        branch_fallback=entry.get('fallback'),
                        retry_text=entry.get('retry_text'),
                    )
                )

    seen_branch = False
    for st in steps:
        text = st.text or ''
        args_json = json.dumps(st.tool_arguments or {}, ensure_ascii=False)
        if '{user_reason}' in text or '{user_reason}' in args_json:
            if not seen_branch:
                errors.append(
                    'a step uses {user_reason} before any branch_on_user_reply'
                )
        if st.kind == STEP_EXPECT_BRANCH:
            seen_branch = True

    errors.extend(_check_alternation(steps))
    if errors:
        return None, errors, warnings
    return steps, [], warnings


def validate_spec(
    entry: dict[str, Any],
    spec_dir: Path,
    remaps: list[tuple[str, str]],
    expected_commit: str,
    derived_seed: int,
    manifest_decisions_sha256: str,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    name = str(entry['name'])
    spec_path = (spec_dir / name).resolve()
    if not spec_path.is_file():
        return {
            'name': name,
            'task_id': str(entry.get('task_id')),
            'errors': [f'spec file missing: {spec_path}'],
            'warnings': [],
        }
    if sha256_lf(spec_path) != str(entry.get('sha256')).upper():
        errors.append('spec LF-normalized sha256 does not match manifest')
    spec = load_json(spec_path)
    if spec.get('schema_version') != SCHEMA_TARGET:
        errors.append(f"schema_version {spec.get('schema_version')!r} != {SCHEMA_TARGET}")
    if str(spec.get('task_id')) != str(entry.get('task_id')):
        errors.append('task_id mismatch vs manifest')
    if spec.get('run_name') != entry.get('run_name'):
        errors.append('run_name mismatch vs manifest')
    if spec.get('simulation_id') != entry.get('simulation_id'):
        errors.append('simulation_id mismatch vs manifest')
    if spec.get('generation_mode') != 'ENVIRONMENT_REPLAY':
        errors.append(f"generation_mode {spec.get('generation_mode')!r} != ENVIRONMENT_REPLAY")
    replay = spec.get('replay') or {}
    if replay.get('environment') != 'tau2-retail':
        errors.append(f"replay.environment {replay.get('environment')!r} != tau2-retail")
    if replay.get('upstream_commit') != expected_commit:
        errors.append(f"replay.upstream_commit {replay.get('upstream_commit')!r} != {expected_commit}")
    if replay.get('user_implementation') != 'user_simulator':
        errors.append(f"replay.user_implementation {replay.get('user_implementation')!r} != user_simulator")
    if float(replay.get('user_temperature')) != 0.0:
        errors.append(f"replay.user_temperature {replay.get('user_temperature')!r} != 0.0")
    if replay.get('agent_implementation') != AGENT_NAME:
        errors.append(f"replay.agent_implementation {replay.get('agent_implementation')!r} != {AGENT_NAME}")
    replay_seed = int(replay.get('seed') or -1)
    if replay_seed != derived_seed:
        errors.append(f'replay.seed {replay_seed} != derived trial seed {derived_seed}')

    source = spec.get('source') or {}
    source_path = Path(remap_path(str(source.get('path') or ''), remaps))
    frozen: list[dict[str, Any]] = []
    if not source_path.is_file():
        errors.append(f'source missing: {source_path}')
    else:
        if sha256(source_path) != str(source.get('sha256')).upper():
            errors.append('source raw sha256 mismatch vs spec')
        try:
            source_results = load_json(source_path)
        except Exception as exc:
            source_results = None
            errors.append(f'source unreadable: {exc}')
        if source_results:
            tasks = source_results.get('tasks') or []
            sims = source_results.get('simulations') or []
            if len(tasks) != 1 or len(sims) != 1:
                errors.append('source must contain exactly one task and one simulation')
            else:
                if str(tasks[0].get('id')) != str(spec.get('task_id')):
                    errors.append('source task id does not match spec task_id')
                if str(sims[0].get('seed')) != str(replay_seed):
                    errors.append('source simulation seed does not match replay.seed')
                frozen = sims[0].get('messages') or []
                if not frozen:
                    errors.append('source simulation has no messages')
                else:
                    steps, step_errors, step_warnings = build_steps(spec, frozen)
                    if step_errors:
                        errors.extend(f'plan: {e}' for e in step_errors)
                    else:
                        counts: dict[str, int] = {}
                        for st in steps:
                            counts[st.kind] = counts.get(st.kind, 0) + 1
                        warnings.extend(
                            f'{name}: {w}' for w in step_warnings
                        )

    policy = spec.get('policy') or {}
    policy_path = Path(remap_path(str(policy.get('path') or ''), remaps))
    if not policy_path.is_file():
        errors.append(f'policy missing: {policy_path}')
    elif sha256(policy_path) != str(policy.get('sha256')).upper():
        errors.append('policy raw sha256 mismatch vs spec')
    if policy.get('upstream_commit') != expected_commit:
        errors.append(f"policy.upstream_commit {policy.get('upstream_commit')!r} != {expected_commit}")

    decisions = spec.get('decisions_binding') or {}
    decisions_path = Path(remap_path(str(decisions.get('path') or ''), remaps))
    if not decisions_path.is_file():
        errors.append(f'decisions missing: {decisions_path}')
    elif sha256(decisions_path) != str(decisions.get('sha256')).upper():
        errors.append('decisions raw sha256 mismatch vs spec')
    if decisions.get('sha256') != manifest_decisions_sha256:
        errors.append('decisions binding does not match manifest decisions_binding')
    if not spec.get('human_decisions'):
        errors.append('human_decisions empty')
    if not spec.get('change_log'):
        errors.append('change_log empty')

    return {
        'name': name,
        'task_id': str(spec.get('task_id')),
        'run_name': spec.get('run_name'),
        'simulation_id': spec.get('simulation_id'),
        'spec_path': str(spec_path),
        'source_path': str(source_path),
        'frozen_message_count': len(frozen),
        'errors': errors,
        'warnings': warnings,
    }


def validate_spec_dir(
    spec_dir: Path,
    remaps: list[tuple[str, str]],
    seed_source: int,
    expected_commit: str,
) -> dict[str, Any]:
    errors: list[str] = []
    manifest_path = spec_dir / 'manifest.json'
    if not manifest_path.is_file():
        errors.append(f'manifest missing: {manifest_path}')
        return {
            'spec_dir': str(spec_dir),
            'manifest_sha256_lf': None,
            'seed_source': seed_source,
            'derived_seed': None,
            'upstream_commit': expected_commit,
            'specs': [],
            'errors': errors,
        }
    manifest = load_json(manifest_path)
    if manifest.get('schema_version') != SCHEMA_MANIFEST:
        errors.append(f"manifest schema_version {manifest.get('schema_version')!r}")
    manifest_decisions = manifest.get('decisions_binding') or {}
    manifest_decisions_sha256 = str(manifest_decisions.get('sha256') or '').upper()
    specs = manifest.get('specs') or []
    if not specs:
        errors.append('manifest specs empty')
    derived_seed = _derive_seed(seed_source)
    records = [
        validate_spec(
            entry,
            spec_dir,
            remaps,
            expected_commit,
            derived_seed,
            manifest_decisions_sha256,
        )
        for entry in specs
    ]
    return {
        'spec_dir': str(spec_dir),
        'manifest_sha256_lf': sha256_lf(manifest_path),
        'seed_source': seed_source,
        'derived_seed': derived_seed,
        'upstream_commit': expected_commit,
        'specs': records,
        'errors': errors,
    }
# --- tau2 runtime (lazy; only needed for full runs) -------------------------

try:
    from tau2.agent.base_agent import HalfDuplexAgent
    from tau2.data_model.message import (
        AssistantMessage,
        MultiToolMessage,
        ToolCall,
        ToolMessage,
        UserMessage,
    )

    _TAU2_AVAILABLE = True
except Exception:  # pragma: no cover - validate-only environments
    _TAU2_AVAILABLE = False


_TASK_CONTEXT: dict[str, dict[str, Any]] = {}


def _json_equiv(a: str, b: str) -> bool:
    try:
        return json.loads(a) == json.loads(b)
    except Exception:
        return a.strip() == b.strip()


def _fill_text(text: str, user_reason: Optional[str], notes: list[str]) -> str:
    if '{user_reason}' in text:
        if user_reason is None:
            notes.append('{user_reason} emitted without a branch match')
        text = text.replace('{user_reason}', user_reason or '')
    return text


def _fill_arguments(
    arguments: dict[str, Any], user_reason: Optional[str], notes: list[str]
) -> dict[str, Any]:
    def walk(value: Any) -> Any:
        if isinstance(value, str):
            if '{user_reason}' in value:
                if user_reason is None:
                    notes.append('{user_reason} emitted without a branch match')
                return value.replace('{user_reason}', user_reason or '')
            return value
        if isinstance(value, dict):
            return {k: walk(v) for k, v in value.items()}
        if isinstance(value, list):
            return [walk(v) for v in value]
        return value

    return walk(arguments)


if _TAU2_AVAILABLE:

    class ScriptedPolicyCompliantAgent(HalfDuplexAgent):
        '''Follows the spec plan; user replies and tool results come from the
        frozen user_simulator and the real environment.'''

        def __init__(
            self,
            tools: list[Any],
            domain_policy: str,
            *,
            task_id: str,
            steps: list[ReplayStep],
            frozen: list[dict[str, Any]],
            trace_path: Path,
            spec_name: str,
        ) -> None:
            super().__init__(tools, domain_policy)
            self.task_id = task_id
            self.steps = steps
            self.frozen = frozen
            self.trace_path = trace_path
            self.spec_name = spec_name

        def get_init_state(self, message_history: Optional[list[Any]] = None) -> dict:
            return {
                'idx': 0,
                'pending_tool_result': False,
                'branch_retried': False,
                'user_reason': None,
                'aborted': False,
                'trace': [],
                'prefix_mismatches': [],
                'result_mismatches': [],
                'branch': {'evaluated': False, 'matched': False, 'aborted': False},
            }

        @staticmethod
        def _describe_incoming(message: Any) -> dict[str, Any]:
            if isinstance(message, ToolMessage):
                return {
                    'kind': 'tool',
                    'tool_call_id': message.id,
                    'error': bool(message.error),
                    'content_preview': (message.content or '')[:200],
                }
            if isinstance(message, MultiToolMessage):
                return {'kind': 'multi_tool', 'count': len(message.tool_messages)}
            if isinstance(message, UserMessage):
                return {
                    'kind': 'user',
                    'content_preview': (message.content or '')[:200],
                }
            return {'kind': type(message).__name__}

        def _record(
            self,
            state: dict,
            incoming: dict[str, Any],
            emitted: Any,
            notes: list[str],
            step_index: int,
            emitted_source: str = '',
        ) -> None:
            if emitted.tool_calls:
                call = emitted.tool_calls[0]
                emitted_info = {
                    'kind': 'tool_call',
                    'tool_name': call.name,
                    'tool_call_id': call.id,
                    'arguments': call.arguments,
                }
            else:
                emitted_info = {
                    'kind': 'stop' if (emitted.content or '') == STOP_TOKEN else 'text',
                    'content_preview': (emitted.content or '')[:200],
                }
            state['trace'].append(
                {
                    'incoming': incoming,
                    'step_index': step_index,
                    'emitted_source': emitted_source or None,
                    'emitted': emitted_info,
                    'user_reason': state.get('user_reason'),
                    'notes': list(notes),
                }
            )

        def _flush_trace(self, state: dict) -> None:
            write_json(
                self.trace_path,
                {
                    'schema_version': 'retail-correction-replay-agent-trace-v1',
                    'task_id': self.task_id,
                    'spec': self.spec_name,
                    'calls': state['trace'],
                    'summary': {
                        'prefix_mismatches': state['prefix_mismatches'],
                        'result_mismatches': state['result_mismatches'],
                        'branch': state['branch'],
                    },
                },
            )

        def _stop_message(self) -> Any:
            return AssistantMessage.text(STOP_TOKEN)

        @classmethod
        def is_stop(cls, message: Any) -> bool:
            # The tau2 base participant defaults to False; without this
            # override the replay would not terminate on the final
            # ###STOP### message and would spin until max_steps.
            return bool((getattr(message, 'content', None) or '') == STOP_TOKEN)

        def _check_user_prefix(
            self, message: Any, step: ReplayStep, state: dict, notes: list[str]
        ) -> None:
            frozen_content = (self.frozen[step.frozen_user_index].get('content') or '')
            replayed_content = message.content or ''
            if frozen_content.strip() != replayed_content.strip():
                state['prefix_mismatches'].append(
                    {
                        'frozen_index': step.frozen_user_index,
                        'frozen': frozen_content[:200],
                        'replayed': replayed_content[:200],
                    }
                )
                notes.append(f'prefix user mismatch at event {step.frozen_user_index}')

        def _consume_tool_result(
            self, message: Any, step: ReplayStep, state: dict, notes: list[str]
        ) -> None:
            frozen_content = (self.frozen[step.frozen_result_index].get('content') or '')
            replayed_content = message.content or ''
            if not _json_equiv(frozen_content, replayed_content):
                state['result_mismatches'].append(
                    {
                        'frozen_index': step.frozen_result_index,
                        'frozen': frozen_content[:200],
                        'replayed': replayed_content[:200],
                    }
                )
                notes.append(f'tool result mismatch at event {step.frozen_result_index}')

        def _evaluate_branch(
            self, rules: list[dict], reply: str
        ) -> Optional[str]:
            for rule in rules:
                needle = rule.get('contains')
                if needle and needle in reply:
                    return rule.get('set_reason')
            return None

        def generate_next_message(
            self, message: Any, state: dict
        ) -> tuple[Any, dict]:
            steps = self.steps
            notes: list[str] = []
            incoming = self._describe_incoming(message)
            kind = incoming['kind']
            idx = state['idx']

            if state['aborted'] or idx >= len(steps):
                msg = self._stop_message()
                self._record(state, incoming, msg, notes, idx, emitted_source='stop')
                self._flush_trace(state)
                return msg, state

            if state['pending_tool_result']:
                if kind != 'tool':
                    notes.append(f'expected tool result, got {kind}')
                state['pending_tool_result'] = False
                step = steps[idx] if idx < len(steps) else None
                if step is not None and step.kind == STEP_EXPECT_TOOL_RESULT:
                    self._consume_tool_result(message, step, state, notes)
                    idx = state['idx'] = idx + 1
                else:
                    notes.append(
                        'tool result for inserted/replaced call (not matched to frozen)'
                    )
            elif idx < len(steps) and steps[idx].kind == STEP_EXPECT_USER:
                if kind != 'user':
                    notes.append(f'expected user reply, got {kind}')
                step = steps[idx]
                if step.frozen_user_index is not None:
                    self._check_user_prefix(message, step, state, notes)
                idx = state['idx'] = idx + 1
            elif idx < len(steps) and steps[idx].kind == STEP_EXPECT_BRANCH:
                if kind != 'user':
                    notes.append(f'expected user reply for branch, got {kind}')
                step = steps[idx]
                reply = message.content or ''
                matched = self._evaluate_branch(step.branch_rules or [], reply)
                if matched is not None:
                    state['user_reason'] = matched
                    state['branch_retried'] = False
                    state['branch'] = {
                        'evaluated': True,
                        'matched': True,
                        'aborted': False,
                        'user_reason': matched,
                    }
                    idx = state['idx'] = idx + 1
                    notes.append(f'branch matched reason={matched!r}')
                elif not state['branch_retried']:
                    state['branch_retried'] = True
                    retry_text = _fill_text(
                        step.retry_text or '', state['user_reason'], notes
                    )
                    msg = AssistantMessage.text(retry_text)
                    self._record(state, incoming, msg, notes, idx, emitted_source='retry')
                    self._flush_trace(state)
                    return msg, state
                else:
                    state['aborted'] = True
                    state['branch'] = {
                        'evaluated': True,
                        'matched': False,
                        'aborted': True,
                    }
                    state['idx'] = len(steps) + 1
                    msg = AssistantMessage.text(ABORT_TEXT)
                    self._record(state, incoming, msg, notes, idx, emitted_source='abort')
                    self._flush_trace(state)
                    return msg, state
            else:
                notes.append(f'unexpected {kind} message at step {idx}')

            while True:
                if idx >= len(steps):
                    msg = self._stop_message()
                    self._record(state, incoming, msg, notes, idx, emitted_source='stop')
                    self._flush_trace(state)
                    return msg, state
                step = steps[idx]
                if step.kind in (STEP_EXPECT_USER, STEP_EXPECT_BRANCH):
                    notes.append(f'internal: paused at {step.kind} without incoming')
                    idx = state['idx'] = idx + 1
                    continue
                if step.kind == STEP_EXPECT_TOOL_RESULT:
                    notes.append(
                        'internal: expect_tool_result without a pending tool result'
                    )
                    idx = state['idx'] = idx + 1
                    continue
                if step.kind == STEP_EMIT_TEXT:
                    text = _fill_text(step.text or '', state['user_reason'], notes)
                    msg = AssistantMessage.text(text)
                    idx = state['idx'] = idx + 1
                    self._record(state, incoming, msg, notes, idx, emitted_source=step.source)
                    self._flush_trace(state)
                    return msg, state
                if step.kind == STEP_EMIT_TOOL_CALL:
                    arguments = _fill_arguments(
                        step.tool_arguments or {}, state['user_reason'], notes
                    )
                    call = ToolCall(
                        id=step.tool_call_id or f'replay-tool-{uuid.uuid4().hex[:12]}',
                        name=step.tool_name or '',
                        arguments=arguments,
                    )
                    msg = AssistantMessage.text(content=None, tool_calls=[call])
                    idx = state['idx'] = idx + 1
                    state['pending_tool_result'] = True
                    self._record(state, incoming, msg, notes, idx, emitted_source=step.source)
                    self._flush_trace(state)
                    return msg, state
                notes.append(f'internal: unknown step kind {step.kind}')
                idx = state['idx'] = idx + 1

        def stop(self, message: Optional[Any] = None, state: Optional[dict] = None) -> None:
            if state is not None:
                self._flush_trace(state)


    def _agent_factory(
        tools: list[Any],
        domain_policy: str,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
        task: Any = None,
        **kwargs: Any,
    ) -> ScriptedPolicyCompliantAgent:
        task_id = str(getattr(task, 'id', ''))
        ctx = _TASK_CONTEXT.get(task_id)
        if ctx is None:
            raise RuntimeError(f'no scripted replay context for task {task_id}')
        return ScriptedPolicyCompliantAgent(
            tools,
            domain_policy,
            task_id=task_id,
            steps=ctx['steps'],
            frozen=ctx['frozen'],
            trace_path=ctx['trace_path'],
            spec_name=ctx['spec_name'],
        )

def _corrected_message_checks(messages: list[dict[str, Any]]) -> dict[str, Any]:
    '''Protocol checks over the replayed corrected trajectory.'''
    tool_calls = 0
    results_ok = True
    mixed = 0
    bad_pairs: list[str] = []
    for i, msg in enumerate(messages):
        tool_calls_list = msg.get('tool_calls') or []
        if msg.get('role') == 'assistant' and tool_calls_list:
            if msg.get('content'):
                mixed += 1
            for call in tool_calls_list:
                tool_calls += 1
                call_id = call.get('id')
                nxt = messages[i + 1] if i + 1 < len(messages) else None
                if not (
                    nxt
                    and nxt.get('role') == 'tool'
                    and nxt.get('id') == call_id
                ):
                    results_ok = False
                    bad_pairs.append(
                        f'assistant event {i} tool call {call.get("name")} '
                        f'has no matching tool result'
                    )
    return {
        'assistant_tool_calls': tool_calls,
        'tool_result_pairs_ok': results_ok,
        'bad_pairs': bad_pairs,
        'mixed_messages': mixed,
    }


def run_one(
    rec: dict[str, Any],
    output_dir: Path,
    seed_source: int,
    llm_user: str,
    bindings: dict[str, Any],
) -> dict[str, Any]:
    if not _TAU2_AVAILABLE:
        raise RuntimeError('tau2 runtime is not importable; cannot run replays')
    spec = load_json(Path(rec['spec_path']))
    task_id = rec['task_id']
    task_dir = output_dir / f'task_{task_id}'
    task_dir.mkdir(parents=True, exist_ok=False)

    from src.evaluation.replay_evaluator import replay_results_artifact
    from tau2.evaluator.evaluator import EvaluationType
    from tau2.registry import registry
    from tau2.run import get_tasks, run_tasks

    if registry.get_agent_factory(AGENT_NAME) is None:
        registry.register_agent_factory(_agent_factory, AGENT_NAME)

    tasks = get_tasks('retail', task_ids=[task_id])
    if len(tasks) != 1 or str(tasks[0].id) != task_id:
        raise RuntimeError(f'Unable to resolve exactly one Retail task {task_id}')

    source_results = load_json(Path(rec['source_path']))
    frozen = source_results['simulations'][0].get('messages') or []
    steps, step_errors, step_warnings = build_steps(spec, frozen)
    if step_errors:
        raise ValueError(f'plan no longer validates: {step_errors}')
    _TASK_CONTEXT[task_id] = {
        'steps': steps,
        'frozen': frozen,
        'trace_path': task_dir / 'agent_trace.json',
        'spec_name': rec['name'],
    }

    results = run_tasks(
        domain='retail',
        tasks=tasks,
        agent=AGENT_NAME,
        user='user_simulator',
        llm_agent=None,
        llm_args_agent=None,
        llm_user=llm_user,
        llm_args_user={'temperature': 0.0},
        num_trials=1,
        max_steps=120,
        max_errors=5,
        save_dir=task_dir / 'tau2_artifacts',
        console_display=True,
        evaluation_type=EvaluationType.ALL,
        max_concurrency=1,
        seed=seed_source,
        log_level='INFO',
        verbose_logs=True,
        max_retries=0,
        auto_review=False,
        save_to=task_dir / 'returned_results.json',
    )

    sim = results.simulations[0]
    reward = float(sim.reward_info.reward)
    termination = str(sim.termination_reason)
    messages = [m.model_dump(mode='json') for m in sim.messages]
    corrected = {
        'schema_version': 'retail-correction-replay-corrected-messages-v1',
        'task_id': task_id,
        'run_name': spec.get('run_name'),
        'simulation_id': sim.id,
        'seed': sim.seed,
        'messages': messages,
    }
    corrected_path = task_dir / 'corrected_messages.json'
    write_json(corrected_path, corrected)

    trace_path = task_dir / 'agent_trace.json'
    trace = load_json(trace_path) if trace_path.is_file() else {}
    trace_summary = trace.get('summary') or {}
    branch = trace_summary.get('branch') or {}
    prefix_mismatches = trace_summary.get('prefix_mismatches') or []
    result_mismatches = trace_summary.get('result_mismatches') or []

    tau2_root = os.environ.get('POLICYAGENT_TAU2_ROOT')
    if not tau2_root:
        raise RuntimeError('Set POLICYAGENT_TAU2_ROOT for the state comparison')
    original = replay_results_artifact(Path(rec['source_path']), tau2_root=tau2_root)
    corrected_replay = replay_results_artifact(
        task_dir / 'returned_results.json', tau2_root=tau2_root
    )
    state = {
        'original_db_hash': original.agent_hash,
        'corrected_db_hash': corrected_replay.agent_hash,
        'original_user_db_hash': original.agent_user_hash,
        'corrected_user_db_hash': corrected_replay.agent_user_hash,
        'original_db_match_gold': original.db_match,
        'corrected_db_match_gold': corrected_replay.db_match,
        'corrected_state_equals_original': (
            corrected_replay.agent_hash == original.agent_hash
            and corrected_replay.agent_user_hash == original.agent_user_hash
        ),
    }

    protocol = _corrected_message_checks(messages)
    replay_manifest = {
        'schema_version': 'retail-correction-replay-task-manifest-v1',
        'status': 'COMPLETED',
        'task_id': task_id,
        'run_name': spec.get('run_name'),
        'spec': {
            'name': rec['name'],
            'path': rec['spec_path'],
            'sha256_lf': sha256_lf(Path(rec['spec_path'])),
            'sha256_raw': sha256(Path(rec['spec_path'])),
        },
        'source': {
            'path': rec['source_path'],
            'sha256_raw': sha256(Path(rec['source_path'])),
        },
        'bindings': bindings,
        'replay': {
            'environment': 'tau2-retail',
            'upstream_commit': bindings.get('upstream', {}).get('commit'),
            'seed': sim.seed,
            'seed_source': seed_source,
            'user_implementation': 'user_simulator',
            'user_temperature': 0.0,
            'agent_implementation': AGENT_NAME,
        },
        'result': {
            'reward': reward,
            'success': reward == 1.0,
            'termination_reason': termination,
            'num_messages': len(messages),
            'simulation_id': sim.id,
            'replay_seed_matches_spec': int(sim.seed) == int(
                (spec.get('replay') or {}).get('seed')
            ),
        },
        'branch': branch,
        'prefix_user_mismatches': len(prefix_mismatches),
        'tool_result_mismatches': len(result_mismatches),
        'protocol': protocol,
        'state': state,
        'corrected_messages': {
            'path': str(corrected_path),
            'sha256': sha256(corrected_path),
        },
        'agent_trace': {'path': str(trace_path), 'calls': len(trace.get('calls') or [])},
        'plan_warnings': step_warnings,
    }
    write_json(task_dir / 'replay_manifest.json', replay_manifest)
    return {
        'task_id': task_id,
        'status': 'COMPLETED',
        'reward': reward,
        'success': reward == 1.0,
        'termination_reason': termination,
        'num_messages': len(messages),
        'branch': branch,
        'prefix_user_mismatches': len(prefix_mismatches),
        'tool_result_mismatches': len(result_mismatches),
        'corrected_state_equals_original': state['corrected_state_equals_original'],
        'corrected_db_match_gold': state['corrected_db_match_gold'],
        'protocol': protocol,
        'replay_manifest_sha256': sha256(task_dir / 'replay_manifest.json'),
    }


def run(
    validated: dict[str, Any],
    output_dir: Path,
    seed_source: int,
    llm_user: str,
    allow_dirty: bool,
) -> dict[str, Any]:
    from src.training.run_retail_agentic_grpo import validate_upstream_checkout

    dirty = bool(git_value(REPO_ROOT, 'status', '--porcelain'))
    if dirty and not allow_dirty:
        raise RuntimeError('Commit the frozen replay inputs or pass --allow-dirty')
    if output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=False)

    upstream = validate_upstream_checkout(validated['upstream_commit'])
    bindings = {
        'spec_dir': validated['spec_dir'],
        'manifest_sha256_lf': validated['manifest_sha256_lf'],
        'seed_source': validated['seed_source'],
        'derived_trial_seed': validated['derived_seed'],
        'upstream_commit': validated['upstream_commit'],
        'upstream': upstream,
        'llm_user': llm_user,
    }
    manifest = {
        'schema_version': 'retail-correction-replay-run-v1',
        'status': 'STARTED',
        'started_at': datetime.now(timezone.utc).isoformat(),
        'scope': SCOPE,
        'project': {
            'commit': git_value(REPO_ROOT, 'rev-parse', 'HEAD'),
            'branch': git_value(REPO_ROOT, 'branch', '--show-current'),
            'dirty_at_start': dirty,
        },
        'bindings': bindings,
        'task_ids': [rec['task_id'] for rec in validated['specs']],
        'evaluation': {
            'type': 'ALL',
            'num_trials': 1,
            'max_steps': 120,
            'max_errors': 5,
            'max_concurrency': 1,
            'max_retries': 0,
        },
        'claims': {
            'official_metric': False,
            'tool_results_environment_generated': True,
            'user_replies_regenerated_by_frozen_simulator': True,
            'reward_not_a_gate': True,
        },
    }
    write_json(output_dir / 'run_manifest.json', manifest)

    per_task: list[dict[str, Any]] = []
    for rec in validated['specs']:
        task_id = rec['task_id']
        try:
            per_task.append(run_one(rec, output_dir, seed_source, llm_user, bindings))
        except Exception as exc:  # noqa: BLE001 - recorded as a task failure
            per_task.append(
                {
                    'task_id': task_id,
                    'status': 'FAILED',
                    'exception_type': type(exc).__name__,
                    'message': str(exc),
                    'traceback': traceback.format_exc(),
                }
            )

    failures = [row for row in per_task if row['status'] != 'COMPLETED']
    summary = {
        'schema_version': 'retail-correction-replay-summary-v1',
        'status': 'COMPLETED_WITH_FAILURES' if failures else 'COMPLETED',
        'task_count': len(per_task),
        'completed_tasks': len(per_task) - len(failures),
        'failures': failures,
        'tasks': per_task,
    }
    write_json(output_dir / 'replay_summary.json', summary)
    manifest.update(
        {
            'status': summary['status'],
            'completed_at': datetime.now(timezone.utc).isoformat(),
            'task_count': len(per_task),
            'completed_tasks': summary['completed_tasks'],
            'failures': failures,
            'summary_sha256': sha256(output_dir / 'replay_summary.json'),
        }
    )
    write_json(output_dir / 'run_manifest.json', manifest)
    return manifest


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description='ENVIRONMENT_REPLAY driver for the correction-target batch.'
    )
    parser.add_argument('--spec-dir', default=str(DEFAULT_SPEC_DIR))
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--output-dir', help='required unless --validate-only')
    parser.add_argument('--seed-source', type=int, default=20260818)
    parser.add_argument('--smoke-task', help='run a single task id (smoke) instead of the full batch')
    parser.add_argument('--llm-user', default='deepseek/deepseek-chat')
    parser.add_argument('--path-remap', action='append', default=[], metavar='OLD=NEW')
    parser.add_argument('--upstream-commit', default=UPSTREAM_COMMIT)
    parser.add_argument('--allow-dirty', action='store_true')
    args = parser.parse_args(argv)

    remaps: list[tuple[str, str]] = []
    for item in args.path_remap:
        if '=' not in item:
            raise SystemExit(f'--path-remap must be OLD=NEW, got {item!r}')
        old, new = item.split('=', 1)
        remaps.append((old, new))

    validated = validate_spec_dir(
        Path(args.spec_dir), remaps, args.seed_source, args.upstream_commit
    )
    all_errors = list(validated['errors'])
    for rec in validated['specs']:
        all_errors.extend(f"{rec['name']}: {e}" for e in rec['errors'])
    if all_errors:
        print(
            json.dumps(
                {
                    'status': 'INVALID',
                    'error_count': len(all_errors),
                    'errors': all_errors,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        raise SystemExit(1)

    if args.smoke_task:
        rows = [rec for rec in validated['specs'] if str(rec['task_id']) == args.smoke_task]
        if not rows:
            raise SystemExit(f'unknown --smoke-task {args.smoke_task!r}; not in the batch')
        validated['specs'] = rows
        print(f'SMOKE_TASK={args.smoke_task}')

    if args.validate_only:
        print(
            json.dumps(
                {
                    'status': 'VALIDATED',
                    'schema_version': 'retail-correction-replay-validation-v1',
                    'spec_dir': validated['spec_dir'],
                    'manifest_sha256_lf': validated['manifest_sha256_lf'],
                    'upstream_commit': validated['upstream_commit'],
                    'seed_source': validated['seed_source'],
                    'derived_trial_seed': validated['derived_seed'],
                    'specs': [
                        {
                            'name': rec['name'],
                            'task_id': rec['task_id'],
                            'run_name': rec['run_name'],
                            'simulation_id': rec['simulation_id'],
                            'frozen_message_count': rec['frozen_message_count'],
                            'warnings': rec['warnings'],
                        }
                        for rec in validated['specs']
                    ],
                    'external_api_called': False,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    if not args.output_dir:
        raise SystemExit('--output-dir is required (unless --validate-only)')
    manifest = run(
        validated,
        Path(args.output_dir),
        args.seed_source,
        args.llm_user,
        args.allow_dirty,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()