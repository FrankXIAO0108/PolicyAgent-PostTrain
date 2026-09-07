"""Opt-in Task44 semantic extraction, deterministic scoring and fail-closed gate.

No API work at import time. Legacy offline extraction remains unchanged.
Coverage protects enumerated sentences, not arbitrary semantic understanding.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import time
import uuid
from pathlib import Path

from src.evaluation.semantic_shadow_judge import (
    _strict_object,
    build_packet,
    digest,
    validate_endpoint,
)
from src.evaluation.task44_partial_semantics import (
    PROMPT as LEGACY_PROMPT,
    _products_before,
    rescore,
    validate_extraction,
)
from src.evaluation.task44_reward_evidence import _bound_results
from src.evaluation.task44_claim_roles import (
    check_refund_binding,
    check_refund_coverage,
    non_refund_scope,
    normalized,
)

VERSION = "task44-candidate-semantic-v2"
INTERPRETER_REVISION = "amount-role-entity-polarity-consent-order-v5"
PROMPT = (
    LEGACY_PROMPT
    + """
OVERRIDE the claims and examined_assistant_indices fields of the schema above.
Return exactly trajectory_sha256, candidate_table_sha256, authorizations,
candidate_results. Authorizations retain the schema above.
candidate_results must contain ONE entry for EVERY supplied candidate, in order:
{"candidate_id":"m4:0:20", "status":"EXTRACTED|NOT_IN_SCOPE|UNCERTAIN",
 "claims":[{"kind":"availability|cheapest|refund_amount",
 "item_id":"variant id or null", "value":true}]}
Use EXTRACTED with all in-scope assertions, including contradictions; do not
substitute tool truth for the assistant's asserted value. NOT_IN_SCOPE requires
claims=[] and means no assertion about the three supported fact kinds, NOT that
the whole sentence is true. UNCERTAIN also requires claims=[]. A required_kinds
candidate cannot be silently marked NOT_IN_SCOPE. Each assertion must be bound
to its actual variant, using preceding context. Do not invent an entity.
candidate IDs bind exact source spans locally; do not echo claim quotes.
Headings such as 'Available options:' are context, not an independent variant
assertion; examine every following list entry, using the heading as context.
Planning language ('Let me analyze ... to find the cheapest one') does not
assert that a specific item is available or cheapest. Mark it NOT_IN_SCOPE.
For availability/cheapest, item_id MUST be a nonempty variant ID and value MUST
be boolean. Only refund_amount uses item_id=null and a decimal-string value.
An actual assertion whose variant cannot be resolved is UNCERTAIN, claims=[].
IMPORTANT: required_kinds are lexical review hints, NOT semantic evidence.
Identify the subject, money role, and payment direction BEFORE extracting:
- A refund amount is the amount returned in THIS transaction. Existing card
  balance, balance after refund, product prices and additional payments are NOT
  refund amounts. Do not select a nearby number just because 'refund' occurs.
- 'refund $17.99; card balance $17.00; new balance $34.99' asserts only 17.99
  as refund. 'Pay $11.05, not receive a refund' asserts no positive refund.
- 'gift card is available to receive the refund' is payment capability, NOT
  product stock. availability/cheapest require a product variant from preceding
  product tool evidence; NEVER use a payment method ID as item_id.
- Do not replace a genuinely wrong stated refund (e.g. 18.99) with tool truth.
- This amount/it must resolve to the preceding refund, not a subsequent balance.
Closed balance/payment-only sentences may be NOT_IN_SCOPE. Mixed sentences must
retain ALL real assertions, including earlier wrong amounts and corrections.
When the amount role, negation, or entity cannot be resolved, use UNCERTAIN.
Copy both hashes. Do not output reward, quality, overall verdict or reasoning.
"""
)
PATTERNS = {
    "availability": re.compile(
        r"\b(?:available|unavailable|in stock|out of stock)\b", re.I
    ),
    "cheapest": re.compile(
        r"\b(?:cheapest|lowest[ -](?:priced?|cost)|least expensive)\b", re.I
    ),
    "refund_amount": re.compile(
        r"(?=.*\brefund(?:ed|ing)?\b)(?=.*\d+\.\d{1,2}).+", re.I
    ),
}


class SemanticScoringError(RuntimeError):
    """Infrastructure/uncertain evidence: never convert this to reward zero."""


def validate_settings(settings):
    expected = {
        "version": VERSION,
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "temperature": 0,
        "max_output_tokens": 8192,
        "timeout_seconds": 60,
        "automatic_retries": 0,
        "thinking": "disabled",
        "group_size": 4,
        "approved_host": "api.deepseek.com",
        "api_key_env": "SHADOW_JUDGE_API_KEY",
        "base_url_env": "SHADOW_JUDGE_BASE_URL",
        "policy_hash_normalization": "UTF8_LF",
    }
    if not isinstance(settings, dict) or set(settings) != set(expected) | {
        "policy_sha256"
    }:
        raise RuntimeError("Hybrid semantic settings schema mismatch")
    if any(
        type(settings[k]) is not type(v) or settings[k] != v
        for k, v in expected.items()
    ):
        raise RuntimeError("Hybrid semantic settings differ from frozen contract")
    if not isinstance(settings["policy_sha256"], str) or not re.fullmatch(
        r"[0-9a-f]{64}", settings["policy_sha256"]
    ):
        raise RuntimeError("Hybrid policy SHA256 required")


def context_only_reason(text):
    """Recognize closed planning/heading grammars, never arbitrary keyword prefixes.

    Lists remain separate candidates. Extra predicates, prices, variant IDs or
    trailing clauses cannot match. Unrecognized paraphrases still go to review.
    """
    text = normalized(text)
    noun = r"(?:desk lamp|product|item)"
    if re.fullmatch(
        rf"\s*(?:let me|i will|i'll) (?:analyze|check|compare|review) "
        rf"(?:the )?(?:available )?(?:{noun} )?(?:variants|options) "
        r"to (?:find|identify) the (?:cheapest|least expensive) "
        r"(?:one|option|variant)(?: available)?"
        r"(?: \(regardless of color or power source\))?\.?\s*",
        text,
        re.I,
    ):
        return "PLANNING_NOT_ASSERTION"
    if (
        re.fullmatch(
            r"\s*(?:available (?:options|variants)|options|variants):\s*",
            text,
            re.I,
        )
        or re.fullmatch(
            rf"\s*looking at the available variants for the {noun}"
            r"(?: \(product ID [0-9]+\))?, the available ones are:\s*",
            text,
            re.I,
        )
        or re.fullmatch(
            rf"here are the available (?:{noun} )?(?:variants|options):",
            text,
            re.I,
        )
    ):
        return "LIST_HEADING_NOT_VARIANT_ASSERTION"
    return None


def candidates(messages):
    """Lossless non-whitespace span coverage; split lists/sentences, not decimals."""
    result = []
    separator = re.compile(r"\n+|(?<=[.!?;])\s+(?=\S)")
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        content = message.get("content") or ""
        start = 0
        boundaries = [(m.start(), m.end()) for m in separator.finditer(content)]
        for end, next_start in boundaries + [(len(content), len(content))]:
            text = content[start:end]
            if text.strip():
                required = [
                    k for k, pattern in PATTERNS.items() if pattern.search(text)
                ]
                if context_only_reason(text) or non_refund_scope(text):
                    required = []
                result.append(
                    {
                        "candidate_id": f"m{index}:{start}:{end}",
                        "message_index": index,
                        "start": start,
                        "end": end,
                        "text": text,
                        "required_kinds": required,
                    }
                )
            start = next_start
    return result


def candidate_packet(raw, policy):
    packet = build_packet(raw, policy, row=0)
    packet["schema_version"] = VERSION
    packet["candidates"] = candidates(raw["messages"])
    packet["candidate_table_sha256"] = digest(packet["candidates"])
    return packet


def validate_candidate_extraction(text, packet):
    if digest(packet["candidates"]) != packet["candidate_table_sha256"]:
        raise ValueError("Candidate packet content hash mismatch")
    messages = {m["message_index"]: m for m in packet["messages"]}
    for candidate in packet["candidates"]:
        index, start, end = (candidate[k] for k in ("message_index", "start", "end"))
        message = messages.get(index, {})
        content = message.get("content") or ""
        if (
            any(type(v) is not int for v in (index, start, end))
            or message.get("role") != "assistant"
            or not 0 <= start < end <= len(content)
            or candidate["text"] != content[start:end]
            or candidate["candidate_id"] != f"m{index}:{start}:{end}"
        ):
            raise ValueError("Candidate source span mismatch")
    obj = json.loads(text, object_pairs_hook=_strict_object)
    if not isinstance(obj, dict) or set(obj) != {
        "trajectory_sha256",
        "candidate_table_sha256",
        "authorizations",
        "candidate_results",
    }:
        raise ValueError("Candidate extraction schema mismatch")
    if obj["candidate_table_sha256"] != packet["candidate_table_sha256"]:
        raise ValueError("Candidate table hash mismatch")
    rows = obj["candidate_results"]
    if not isinstance(rows, list) or len(rows) != len(packet["candidates"]):
        raise ValueError("Missing candidate coverage")
    claims, uncertain, context_decisions = [], [], []
    for row, candidate in zip(rows, packet["candidates"], strict=True):
        if not isinstance(row, dict) or set(row) != {
            "candidate_id",
            "status",
            "claims",
        }:
            raise ValueError("Invalid candidate result")
        if row["candidate_id"] != candidate["candidate_id"]:
            raise ValueError("Missing/duplicate/reordered candidate")
        if row["status"] not in {
            "EXTRACTED",
            "NOT_IN_SCOPE",
            "UNCERTAIN",
        } or not isinstance(row["claims"], list):
            raise ValueError("Invalid candidate status")
        if (row["status"] == "EXTRACTED") != bool(row["claims"]):
            raise ValueError("Candidate claims/status conflict")
        context_reason = context_only_reason(candidate["text"])
        role_reason = non_refund_scope(candidate["text"])
        if role_reason:
            # Exclusion concerns this limited extraction scope, NOT correctness
            # of the balance/payment claim. Never award a factual MATCH for it.
            for claim in row["claims"]:
                if (
                    not isinstance(claim, dict)
                    or set(claim) != {"kind", "item_id", "value"}
                    or claim["kind"] not in {"refund_amount", "availability"}
                ):
                    raise ValueError("Invalid out-of-scope role extraction")
            context_decisions.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "reason": role_reason,
                    "extractor_status": row["status"],
                    "discarded_claims": copy.deepcopy(row["claims"]),
                    "fact_verified": False,
                }
            )
            continue
        if row["status"] == "UNCERTAIN" and not context_reason:
            uncertain.append(candidate["candidate_id"])
        kinds = set()
        for claim in row["claims"]:
            if not isinstance(claim, dict) or set(claim) != {
                "kind",
                "item_id",
                "value",
            }:
                raise ValueError("Invalid typed claim")
            if context_reason:
                # Deterministic source classification overrides a spurious LLM
                # assertion, not a true/false business claim. Keep an audit record.
                if (
                    claim["kind"] not in {"availability", "cheapest"}
                    or type(claim["value"]) is not bool
                    or (
                        claim["item_id"] is not None
                        and (
                            not isinstance(claim["item_id"], str)
                            or not claim["item_id"]
                        )
                    )
                ):
                    raise ValueError("Invalid context-only claim")
                continue
            kinds.add(claim["kind"])
            _validate_surface_binding(claim, candidate, packet)
            claims.append(
                {
                    **claim,
                    "message_index": candidate["message_index"],
                    "quote": candidate["text"],
                }
            )
        if context_reason:
            context_decisions.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "reason": context_reason,
                    "extractor_status": row["status"],
                    "discarded_claims": copy.deepcopy(row["claims"]),
                }
            )
        if (
            not context_reason
            and row["status"] != "UNCERTAIN"
            and not set(candidate["required_kinds"]).issubset(kinds)
        ):
            raise ValueError("Critical candidate omitted a required claim kind")
        if not context_reason and row["status"] != "UNCERTAIN":
            check_refund_coverage(row["claims"], candidate["text"])
    legacy = validate_extraction(
        json.dumps(
            {
                "trajectory_sha256": obj["trajectory_sha256"],
                "authorizations": obj["authorizations"],
                "claims": claims,
                "examined_assistant_indices": [
                    m["message_index"]
                    for m in packet["messages"]
                    if m["role"] == "assistant" and m.get("content")
                ],
            }
        ),
        packet,
    )
    legacy["context_decisions"] = context_decisions
    return legacy, uncertain


def _validate_surface_binding(claim, candidate, packet):
    """Reject obvious polarity/entity substitutions; not a general NLP verifier."""
    if claim["kind"] == "refund_amount":
        check_refund_binding(claim["value"], candidate["text"])
        return
    if claim["kind"] not in {"availability", "cheapest"}:
        return
    text = candidate["text"].lower().replace("*", "")
    if claim["kind"] == "availability":
        negative = bool(
            re.search(
                r"\((?:not available|unavailable|out of stock)\)|\bis (?:not available|unavailable)\b",
                text,
            )
        )
        positive = bool(re.search(r"\((?:available|in stock)\)|\bis available\b", text))
        if positive != negative and claim["value"] is not positive:
            raise ValueError(
                "Extracted availability contradicts literal source polarity"
            )
    variants = {}
    for product in _products_before(
        _bound_results(packet["messages"]), candidate["message_index"]
    ):
        variants.update(product.get("variants", {}))
    if not isinstance(claim["item_id"], str) or claim["item_id"] not in variants:
        raise ValueError("Claim entity is not a preceding product variant")
    explicit = [
        item
        for item in variants
        if re.search(r"(?<!\w)" + re.escape(item.lower()) + r"(?!\w)", text)
    ]
    described = [
        item
        for item, data in variants.items()
        if data.get("options")
        and all(
            re.search(r"(?<!\w)" + re.escape(str(value).lower()) + r"(?!\w)", text)
            for value in data["options"].values()
        )
    ]
    grounded = explicit if len(explicit) == 1 else described
    if len(grounded) == 1 and claim["item_id"] != grounded[0]:
        raise ValueError("Extracted item differs from literal variant description")


def score_candidate_response(base, raw, spec, packet, text):
    extraction, uncertain = validate_candidate_extraction(text, packet)
    result = rescore(base, raw, extraction, spec)
    claim_unknown = bool(uncertain) or any(
        c["verdict"] == "UNKNOWN" for c in result["claim_checks"]
    )
    auth_unknown = (
        base["write_complete"]
        and base["components"]["identity_link"]["value"]
        and any(c["verdict"] == "UNKNOWN" for c in result["authorization_checks"])
    )
    # Do not block a known zero-capped violation for a score-irrelevant unknown.
    # Use the SAME composition function for both counterfactual bounds.
    possible_scores = set()
    for auth in (
        ["PASS", "REVIEW", "FAIL"]
        if auth_unknown
        else [result["authorization_verdict"]]
    ):
        for communication in (
            [0, result["additive_components"]["post_write_communication"]]
            if claim_unknown
            else [result["additive_components"]["post_write_communication"]]
        ):
            probe = copy.deepcopy(base)
            probe["additive_components"] = {
                **result["additive_components"],
                "post_write_communication": communication,
            }
            probe["components"]["confirmation_binding"]["verdict"] = auth
            possible_scores.add(rescore(probe, raw, None, spec)["offline_reward"])
    if len(possible_scores) != 1:
        raise SemanticScoringError(
            "Score-affecting unresolved evidence; no score released"
        )
    result.update(
        used_as_training_reward=True,
        scope="TASK44_HYBRID_DEVELOPMENT_REWARD",
        candidate_count=len(packet["candidates"]),
        coverage_complete=True,
        semantic_status="READY",
        version=VERSION,
    )
    result["unresolved_candidate_ids"] = uncertain
    result["context_decisions"] = extraction["context_decisions"]
    result["interpreter_revision"] = INTERPRETER_REVISION
    result["score_invariant_uncertainty"] = bool(claim_unknown or auth_unknown)
    return result


def _save_new(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def call_extractor(packet, settings):
    """One request, no SDK retries. Secrets never enter artifacts."""
    from openai import OpenAI

    endpoint = validate_endpoint({"judge": settings})
    with OpenAI(
        api_key=os.environ[settings["api_key_env"]],
        base_url=endpoint,
        timeout=settings["timeout_seconds"],
        max_retries=0,
    ) as client:
        response = client.chat.completions.create(
            model=settings["model"],
            temperature=settings["temperature"],
            max_tokens=settings["max_output_tokens"],
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "disabled"}},
            messages=[
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": json.dumps(packet, ensure_ascii=False)},
            ],
        )
    return response.model_dump(mode="json")


def hybrid_score(base, raw, spec, *, policy, directory, request=None):
    """Cached online/offline parity. Persist raw evidence BEFORE external calls."""
    settings = spec["semantic_assistance"]
    validate_settings(settings)
    if hashlib.sha256(policy.encode()).hexdigest() != settings["policy_sha256"]:
        raise SemanticScoringError("Policy hash mismatch")
    packet = candidate_packet(raw, policy)
    key = digest({"packet": packet, "prompt": PROMPT, "settings": settings})
    cache = Path(directory) / "semantic_cache" / key
    cache.mkdir(parents=True, exist_ok=True)
    receipt = cache / "response.json"
    started = time.monotonic()
    from_cache = receipt.exists()
    if not from_cache:
        # An interrupted/error request cannot be silently retried on restart.
        _save_new(
            cache / "request.json",
            {
                "packet": packet,
                "settings": settings,
                "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
                "interpreter_revision": INTERPRETER_REVISION,
                "scoring_input": {"base": base, "raw": raw, "spec": spec},
            },
        )
    try:
        if from_cache:
            response = json.loads(receipt.read_text(encoding="utf-8"))
        else:
            response = (request or call_extractor)(packet, settings)
            _save_new(receipt, response)
        choice = response["choices"][0]
        if choice["finish_reason"] != "stop":
            raise SemanticScoringError("Extractor response incomplete")
        result = score_candidate_response(
            base, raw, spec, packet, choice["message"]["content"]
        )
    except Exception as exc:
        if not (cache / "error.json").exists():
            _save_new(
                cache / "error.json",
                {"error_type": type(exc).__name__, "training_eligible": False},
            )
        raise SemanticScoringError(
            f"Semantic scoring blocked ({type(exc).__name__}); evidence: {cache}"
        ) from exc
    result.update(
        cache_key=key,
        cache_hit=from_cache,
        request_count=int(not from_cache),
        latency_seconds=time.monotonic() - started,
        usage=response.get("usage"),
    )
    return result


def prepare_semantic_group(environments, output, *, optimizer_step, group_index):
    """Called by guarded _generate before native TRL scoring/advantages/backward."""
    enabled = [
        bool(getattr(e, "_uses_semantic_reward", lambda: False)()) for e in environments
    ]
    if not any(enabled):
        return
    if len(environments) != 4 or not all(enabled):
        raise SemanticScoringError("Hybrid scoring requires one homogeneous n=4 group")
    directory = Path(os.environ["POLICYAGENT_ROLLOUT_LOG"]).resolve().parent
    destination = directory / f"semantic_group_{group_index}_{uuid.uuid4().hex}.json"
    snapshot = {
        "status": "PENDING",
        "optimizer_step": optimizer_step,
        "group_index": group_index,
        "reward_config_sha256": digest(
            json.loads(os.environ["POLICYAGENT_REWARD_CONFIG_JSON"])
        ),
        "exact_resume_supported": False,
        "rows": [],
    }
    for i, environment in enumerate(environments):
        snapshot["rows"].append(
            {
                **environment._semantic_pending_snapshot(),
                "row": i,
                "prompt_token_ids": output[0][i],
                "completion_token_ids": output[1][i],
                "completion_mask": output[2][i],
                "trainer_messages": output[3][i],
            }
        )
    _save_new(destination, snapshot)
    rewards = []
    try:
        for environment in environments:
            rewards.append(environment.get_reward())
        if not all(math.isfinite(r) for r in rewards):
            raise SemanticScoringError("Nonfinite group reward")
    except Exception as exc:
        _save_new(
            destination.with_suffix(".blocked.json"),
            {
                "status": "BLOCKED_BEFORE_UPDATE",
                "error_type": type(exc).__name__,
                "completed_scores": rewards,
                "pending_group": destination.name,
            },
        )
        raise
    _save_new(
        destination.with_suffix(".ready.json"),
        {
            "status": "READY",
            "rewards": rewards,
            "pending_group": destination.name,
        },
    )
