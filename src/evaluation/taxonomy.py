from __future__ import annotations

from typing import Any


_ROOT_CAUSE_MAP = {
    "golden_mismatch": "dataset_alignment_error",
    "dataset_issue": "dataset_alignment_error",
    "variant_understanding_failure": "variant_error",
    "wrong_variant": "variant_error",
    "scope_confirmation_failure": "scope_error",
    "wrong_payment_method": "payment_error",
    "policy_violation": "policy_error",
    "missing_action": "missing_action",
    "communication_omission": "communication_error",
}


def _ordered_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def build_three_layer_taxonomy(
    *,
    official_signal: dict[str, Any],
    detailed_root_causes: list[dict[str, Any]],
    state_diff: dict[str, Any],
) -> dict[str, Any]:
    """Separate benchmark signals, causal diagnoses, and business impact.

    Tau2 itself emits reward components, not the root-cause labels below. The
    normalized labels are deterministic downstream diagnoses and therefore
    stay in a separate layer.
    """

    signals: list[str] = []
    if official_signal.get("db_match") is False:
        signals.append("db_mismatch")
    if official_signal.get("nl_match") is False:
        signals.append("nl_failure")

    primary_roots: list[str] = []
    secondary_roots: list[str] = []
    for cause in detailed_root_causes:
        normalized = _ROOT_CAUSE_MAP.get(str(cause.get("code", "")))
        if not normalized:
            continue
        if cause.get("caused_official_failure", True):
            primary_roots.append(normalized)
        else:
            secondary_roots.append(normalized)

    primary_roots = _ordered_unique(primary_roots)
    secondary_roots = [
        root
        for root in _ordered_unique(secondary_roots)
        if root not in primary_roots
    ]
    roots = primary_roots + secondary_roots

    # A raw DB field mismatch is an observed consequence, not automatically a
    # root cause. Use coarse flags only as a fallback when attribution produced
    # no evidence-backed diagnosis.
    if not roots:
        flags = state_diff.get("flags", {})
        if flags.get("wrong_address"):
            roots.append("address_error")
        if flags.get("wrong_refund"):
            roots.append("refund_error")
        if flags.get("wrong_status"):
            roots.append("status_error")
        if flags.get("extra_cancel") or flags.get("extra_exchange"):
            roots.append("scope_error")
        if flags.get("missing_cancel") or flags.get("missing_exchange"):
            roots.append("missing_action")
    impacts: list[str] = []
    if "dataset_alignment_error" in roots:
        impacts.append("benchmark_data_risk")
    if "variant_error" in roots:
        impacts.append("wrong_product_selection")
    if "scope_error" in roots:
        impacts.append("overbroad_or_incorrect_order_effect")
    if "payment_error" in roots:
        impacts.append("wrong_refund_or_charge_destination")
    if "policy_error" in roots:
        impacts.append("policy_risk")
    if "missing_action" in roots:
        impacts.append("incomplete_customer_request")
    if "communication_error" in roots:
        impacts.append("incomplete_customer_communication")
    if "address_error" in roots:
        impacts.append("wrong_shipping_destination")
    if "refund_error" in roots:
        impacts.append("incorrect_refund_state")
    if "status_error" in roots:
        impacts.append("incorrect_order_status")

    return {
        "official_signal": signals,
        "root_cause": roots,
        "primary_causal_root_cause": primary_roots,
        "secondary_findings": secondary_roots,
        "business_impact": _ordered_unique(impacts),
        "quarantine_recommended": "dataset_alignment_error" in roots,
    }
