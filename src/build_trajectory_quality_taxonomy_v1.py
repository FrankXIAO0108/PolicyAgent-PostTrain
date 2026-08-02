"""
Build Trajectory Quality Taxonomy v1
and SFT Data Eligibility Rules v1.

Source experiment
-----------------
Retail Prompt Base / Trial-1

Frozen raw baseline:
16 / 20 = 80%

This artifact combines:

Failure audits:
- 59
- 98
- 95
- 107

Priority Reward=1 quality audits:
- 16
- 28
- 46
- 21

Important
---------
Only 8 / 20 trajectories have received detailed human audit here.

Therefore:

- this file MUST NOT label the other 12 successful trajectories as Gold;
- Raw Reward is not a training label;
- Action Match Rate is not a trajectory-quality score;
- the frozen raw baseline remains 16/20 = 80%.

Offline only.
No API calls.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(
    r"D:\PolicyAgent-PostTrain"
)

RUN_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "20260722_110504_retail_baseline20_trial1_deepseek"
)

DOCS_DIR = (
    PROJECT_ROOT
    / "docs"
)

JSON_OUTPUT = (
    RUN_DIR
    / "trajectory_quality_taxonomy_v1.json"
)

CSV_OUTPUT = (
    RUN_DIR
    / "trajectory_training_eligibility_v1.csv"
)

RULES_OUTPUT = (
    RUN_DIR
    / "sft_data_eligibility_rules_v1.json"
)

MD_OUTPUT = (
    DOCS_DIR
    / "02_评测与失败分析"
    / "2026-07-22_轨迹质量分类标准_v1.md"
)


# =============================================================================
# Frozen metadata
# =============================================================================

BASELINE = {
    "experiment":
        "Retail Prompt Base / Trial-1",

    "task_count":
        20,

    "raw_success_count":
        16,

    "raw_failure_count":
        4,

    "raw_success_rate":
        0.80,

    "detailed_human_audit_count":
        8,

    "audited_reward_0_count":
        4,

    "audited_reward_1_count":
        4,

    "unaudited_reward_1_count":
        12,

    "important_note":
        (
            "Only 8 trajectories have detailed human audit. "
            "The remaining 12 Reward=1 trajectories are UNREVIEWED, "
            "not automatically Gold."
        ),
}


UPSTREAM = {
    "repository":
        "sierra-research/tau2-bench",

    "frozen_commit":
        "58e5e1ace69302e6982d27014569c03e0ffccdd2",

    "task21_source_verification":
        {
            "status":
                "SOURCE_VERIFIED",

            "finding":
                (
                    "modify_pending_order_items reuses the stale final "
                    "'variant' variable during the second mutation loop, "
                    "causing earlier modified items to receive the final "
                    "target variant's price/options in multi-item updates."
                ),

            "local_source":
                (
                    r"D:\tau2-bench\src\tau2"
                    r"\domains\retail\tools.py"
                ),
        },
}


# =============================================================================
# Audited trajectories
# =============================================================================

TRAJECTORIES: list[dict[str, Any]] = [

    # =========================================================================
    # Reward = 1
    # =========================================================================

    {
        "task_id":
            "16",

        "raw_reward":
            1.0,

        "action_match":
            "8/9",

        "trajectory_role":
            "POSITIVE",

        "quality_bucket":
            "GOLD",

        "primary_label":
            "OUTCOME_CORRECT_PROCESS_CORRECT",

        "secondary_labels": [
            "GOOD_ENTITY_DISAMBIGUATION",
            "MULTI_GOAL_COMPLETION",
            "SAFE_DESTRUCTIVE_ACTION_CONFIRMATION",
            "CLAIM_ACTION_CONSISTENCY",
            "EQUIVALENT_REASONING_PATH_MISMATCH",
        ],

        "outcome_correct":
            True,

        "policy_compliant":
            True,

        "tool_arguments_correct":
            True,

        "post_tool_state_integrity":
            "PASS",

        "claim_state_consistency":
            "PASS",

        "authorized_scope_safe":
            True,

        "benchmark_suspect":
            False,

        "environment_bug":
            False,

        "raw_positive_sft":
            "YES",

        "corrected_positive_sft":
            "NOT_REQUIRED",

        "negative_training":
            "NO",

        "preference_training":
            "POSITIVE_CANDIDATE",

        "verifier_training":
            "POSITIVE_REFERENCE",

        "status":
            "DIRECT_GOLD_ELIGIBLE",

        "summary":
            (
                "High-quality multi-goal trajectory. "
                "The only missing action match is an explicit calculator call; "
                "the final arithmetic result is correct and all critical "
                "reads/writes, confirmations, entity disambiguation, and "
                "claims are consistent."
            ),
    },

    {
        "task_id":
            "28",

        "raw_reward":
            1.0,

        "action_match":
            "10/11",

        "trajectory_role":
            "POSITIVE",

        "quality_bucket":
            "GOLD",

        "primary_label":
            "OUTCOME_CORRECT_PROCESS_CORRECT",

        "secondary_labels": [
            "SAFE_ACTION_SCOPE_HANDLING",
            "GOOD_ENTITY_DISAMBIGUATION",
            "CONDITIONAL_INTENT_FOLLOWING",
            "ORDERED_MULTI_TOOL_EXECUTION",
            "MULTI_ORDER_MULTI_ITEM_PLANNING",
            "CLAIM_ACTION_CONSISTENCY",
            "EQUIVALENT_REASONING_PATH_MISMATCH",
        ],

        "outcome_correct":
            True,

        "policy_compliant":
            True,

        "tool_arguments_correct":
            True,

        "post_tool_state_integrity":
            "PASS",

        "claim_state_consistency":
            "PASS",

        "authorized_scope_safe":
            True,

        "benchmark_suspect":
            False,

        "environment_bug":
            False,

        "raw_positive_sft":
            "YES",

        "corrected_positive_sft":
            "NOT_REQUIRED",

        "negative_training":
            "NO",

        "preference_training":
            "POSITIVE_CANDIDATE",

        "verifier_training":
            "HIGH_VALUE_POSITIVE_SCOPE_SAMPLE",

        "status":
            "DIRECT_GOLD_ELIGIBLE",

        "summary":
            (
                "High-quality scope-safe trajectory. "
                "The Agent correctly refuses to expand a single-item "
                "cancellation request into whole-order cancellation, "
                "while completing all five returns in the requested order."
            ),
    },

    {
        "task_id":
            "46",

        "raw_reward":
            1.0,

        "action_match":
            "4/7",

        "trajectory_role":
            "POSITIVE_NEEDS_CLEANUP",

        "quality_bucket":
            "SILVER",

        "primary_label":
            "OUTCOME_CORRECT_PROCESS_IMPERFECT",

        "secondary_labels": [
            "STATIC_PATH_ACTION_CHECK_MISMATCH",
            "EQUIVALENT_REASONING_PATH_MISMATCH",
            "ENTITY_DISAMBIGUATION_WEAKNESS",
            "USER_CONSTRAINT_GROUNDING_FAILURE",
            "EVALUATOR_COVERAGE_GAP",
        ],

        "outcome_correct":
            True,

        "policy_compliant":
            True,

        "tool_arguments_correct":
            True,

        "post_tool_state_integrity":
            "PASS",

        "claim_state_consistency":
            "CORE_PASS_CONTEXT_PARTIAL_FAIL",

        "authorized_scope_safe":
            True,

        "benchmark_suspect":
            "ACTION_PATH_ONLY",

        "environment_bug":
            False,

        "raw_positive_sft":
            "NO",

        "corrected_positive_sft":
            "YES",

        "negative_training":
            "NO_AS_WHOLE_TRAJECTORY",

        "preference_training":
            "YES_AFTER_CLEANUP",

        "verifier_training":
            "YES",

        "status":
            "CORRECT_BEFORE_GOLD",

        "summary":
            (
                "Final business state and refund amounts are correct, "
                "but the Agent initially guesses the wrong vacuum entity "
                "and later forgets the user's stated email constraint. "
                "Static action mismatches themselves are mostly harmless "
                "reference-path differences."
            ),
    },

    {
        "task_id":
            "21",

        "raw_reward":
            1.0,

        "action_match":
            "11/12",

        "trajectory_role":
            "ENVIRONMENT_BUG_CONTAMINATED",

        "quality_bucket":
            "SUSPECT",

        "primary_label":
            "SOURCE_VERIFIED_TOOL_STATE_CORRUPTION",

        "secondary_labels": [
            "POST_TOOL_STATE_VERIFICATION_FAILURE",
            "CLAIM_TOOL_RESULT_INCONSISTENCY",
            "EVALUATOR_COVERAGE_GAP",
            "IDENTIFIER_TYPE_RECOVERY",
            "DYNAMIC_MULTI_GOAL_HANDLING",
            "EQUIVALENT_REASONING_PATH_MISMATCH",
        ],

        "outcome_correct":
            "EVALUATOR_PASS_BUT_STATE_INTEGRITY_INVALID",

        "policy_compliant":
            True,

        "tool_arguments_correct":
            True,

        "post_tool_state_integrity":
            "FAIL_SOURCE_VERIFIED_TOOL_BUG",

        "claim_state_consistency":
            "FAIL",

        "authorized_scope_safe":
            True,

        "benchmark_suspect":
            True,

        "environment_bug":
            True,

        "environment_bug_source_verified":
            True,

        "raw_positive_sft":
            "NO",

        "corrected_positive_sft":
            "RERUN_AFTER_ENV_FIX",

        "negative_training":
            "NO_AS_AGENT_FAILURE_WHOLE_TRAJECTORY",

        "preference_training":
            "SEGMENT_LEVEL_ONLY",

        "verifier_training":
            "VERY_HIGH_VALUE",

        "status":
            "EXCLUDE_RAW_HOLD_FOR_ENV_FIX",

        "summary":
            (
                "Agent planning and tool arguments are correct, but the "
                "source-verified modify_pending_order_items bug corrupts "
                "multi-item state. The Agent then fails to validate the "
                "returned state before claiming success. Reward=1 fails "
                "to expose the corrupted environment state."
            ),
    },

    # =========================================================================
    # Reward = 0
    # =========================================================================

    {
        "task_id":
            "59",

        "raw_reward":
            0.0,

        "action_match":
            "3/5",

        "trajectory_role":
            "EXCLUDED_ALIGNMENT_CASE",

        "quality_bucket":
            "EXCLUDED",

        "primary_label":
            "USER_SIMULATOR_GOLD_MISMATCH",

        "secondary_labels": [
            "STATIC_GOLD_FINAL_INTENT_MISMATCH",
            "BENCHMARK_ALIGNMENT_FAILURE",
        ],

        "outcome_correct":
            "CONVERSATION_CONSISTENT_BUT_GOLD_MISMATCH",

        "policy_compliant":
            True,

        "tool_arguments_correct":
            "CONSISTENT_WITH_FINAL_USER_INTENT",

        "post_tool_state_integrity":
            "PASS",

        "claim_state_consistency":
            "PASS",

        "authorized_scope_safe":
            True,

        "benchmark_suspect":
            True,

        "environment_bug":
            False,

        "raw_positive_sft":
            "NO_RAW_BENCHMARK_LABEL",

        "corrected_positive_sft":
            "ONLY_AFTER_ADJUDICATION",

        "negative_training":
            "NO",

        "preference_training":
            "NO_AS_RAW_PAIR",

        "verifier_training":
            "ALIGNMENT_AUDIT_ONLY",

        "status":
            "EXCLUDE_FROM_RAW_TRAINING_POOL",

        "summary":
            (
                "User Simulator final authorized intent conflicts with "
                "Static Golden. Reward=0 must not be treated as an Agent "
                "negative sample."
            ),
    },

    {
        "task_id":
            "98",

        "raw_reward":
            0.0,

        "action_match":
            "1/3",

        "trajectory_role":
            "MIXED",

        "quality_bucket":
            "MIXED",

        "primary_label":
            "CLAIM_ACTION_INCONSISTENCY",

        "secondary_labels": [
            "DYNAMIC_INTENT_STATIC_GOLD_MISMATCH",
            "ACTION_SCOPE_CONFIRMATION_FAILURE",
            "EVALUATOR_COVERAGE_GAP",
        ],

        "outcome_correct":
            False,

        "policy_compliant":
            "PARTIAL_FAIL",

        "tool_arguments_correct":
            "MIXED",

        "post_tool_state_integrity":
            "TOOL_SUCCESS",

        "claim_state_consistency":
            "FAIL",

        "authorized_scope_safe":
            False,

        "benchmark_suspect":
            True,

        "environment_bug":
            False,

        "raw_positive_sft":
            "NO",

        "corrected_positive_sft":
            "SEGMENT_LEVEL",

        "negative_training":
            "SEGMENT_LEVEL_ONLY",

        "preference_training":
            "YES_AFTER_RELABELING",

        "verifier_training":
            "VERY_HIGH_VALUE",

        "status":
            "SEGMENT_AND_RELABEL",

        "summary":
            (
                "Contains both benchmark payment-method mismatch and real "
                "Agent failures: whole-order scope was not explicitly "
                "authorized and the final claim misrepresented the actual "
                "refund/action scope."
            ),
    },

    {
        "task_id":
            "95",

        "raw_reward":
            0.0,

        "action_match":
            "0/2",

        "trajectory_role":
            "VALID_NEGATIVE",

        "quality_bucket":
            "NEGATIVE",

        "primary_label":
            "ENVIRONMENT_STATE_SEMANTICS_MISUNDERSTANDING",

        "secondary_labels": [
            "FALSE_CAPABILITY_BOUNDARY_DETECTION",
            "PREMATURE_ESCALATION",
            "INCOMPLETE_MULTI_GOAL_EXECUTION",
        ],

        "outcome_correct":
            False,

        "policy_compliant":
            True,

        "tool_arguments_correct":
            "NO_CRITICAL_WRITE_EXECUTED",

        "post_tool_state_integrity":
            "NOT_APPLICABLE",

        "claim_state_consistency":
            "FALSE_CAPABILITY_CLAIM",

        "authorized_scope_safe":
            True,

        "benchmark_suspect":
            False,

        "environment_bug":
            False,

        "raw_positive_sft":
            "NO",

        "corrected_positive_sft":
            "YES",

        "negative_training":
            "YES",

        "preference_training":
            "YES",

        "verifier_training":
            "VERY_HIGH_VALUE",

        "status":
            "VALID_CORRECTION_CANDIDATE",

        "summary":
            (
                "Agent confuses one available Variant with one physical "
                "inventory unit, incorrectly concludes the second exchange "
                "is impossible, and prematurely escalates."
            ),
    },

    {
        "task_id":
            "107",

        "raw_reward":
            0.0,

        "action_match":
            "1/2",

        "trajectory_role":
            "VALID_NEGATIVE_WITH_TOOL_GAP",

        "quality_bucket":
            "NEGATIVE",

        "primary_label":
            "POLICY_GROUNDING_FAILURE",

        "secondary_labels": [
            "POLICY_TOOL_ENFORCEMENT_GAP",
            "INVALID_ACTION_ALLOWED_BY_TOOL",
            "NL_EVALUATOR_COVERAGE_GAP",
        ],

        "outcome_correct":
            False,

        "policy_compliant":
            False,

        "tool_arguments_correct":
            "POLICY_INVALID_BUT_TOOL_ACCEPTED",

        "post_tool_state_integrity":
            "TOOL_EXECUTED_AS_REQUESTED",

        "claim_state_consistency":
            "TOOL_PASS_POLICY_FAIL",

        "authorized_scope_safe":
            True,

        "benchmark_suspect":
            False,

        "environment_bug":
            "ENFORCEMENT_GAP",

        "raw_positive_sft":
            "NO",

        "corrected_positive_sft":
            "YES",

        "negative_training":
            "YES",

        "preference_training":
            "YES",

        "verifier_training":
            "VERY_HIGH_VALUE",

        "status":
            "VALID_CORRECTION_CANDIDATE",

        "summary":
            (
                "Agent performs same-variant exchange despite Retail Policy "
                "requiring a different product option. The exchange tool "
                "fails to enforce the rule and accepts the invalid action."
            ),
    },
]


# =============================================================================
# SFT eligibility rules
# =============================================================================

SFT_RULES: dict[str, Any] = {

    "version":
        "v1",

    "core_principles": [

        {
            "rule_id":
                "R01",

            "rule":
                "Raw Reward is not a training label.",

            "reason":
                (
                    "Reward=0 may be benchmark/evaluator noise, and Reward=1 "
                    "may contain hidden process, policy, or state-integrity defects."
                ),
        },

        {
            "rule_id":
                "R02",

            "rule":
                "Action Match Rate is not a trajectory-quality score.",

            "reason":
                (
                    "Equivalent reasoning paths can fail action matching, "
                    "while high action match can coexist with hidden state corruption."
                ),
        },

        {
            "rule_id":
                "R03",

            "rule":
                "Only human/verifier-audited GOLD trajectories enter raw SFT Gold.",

            "reason":
                (
                    "Unaudited Reward=1 trajectories remain UNREVIEWED."
                ),
        },

        {
            "rule_id":
                "R04",

            "rule":
                "SILVER trajectories require correction before positive SFT.",

            "reason":
                (
                    "Outcome may be correct while process defects would be "
                    "copied by imitation training."
                ),
        },

        {
            "rule_id":
                "R05",

            "rule":
                "SUSPECT or environment-bug-contaminated trajectories are held out.",

            "reason":
                (
                    "Do not train on corrupted observations/states before adjudication "
                    "or environment repair."
                ),
        },

        {
            "rule_id":
                "R06",

            "rule":
                "Valid Agent failures may be used only with explicit failure labels "
                "and corrected target trajectories.",

            "reason":
                (
                    "Negative trajectory alone is insufficient for standard SFT."
                ),
        },

        {
            "rule_id":
                "R07",

            "rule":
                "Mixed badcases must be segmented and relabeled.",

            "reason":
                (
                    "A single trajectory can contain both correct behavior and "
                    "benchmark noise or real failures."
                ),
        },

        {
            "rule_id":
                "R08",

            "rule":
                "Benchmark-alignment cases are excluded from raw negative pools.",

            "reason":
                (
                    "Training against an incorrect/static label can teach the "
                    "Agent to ignore final explicit user intent."
                ),
        },

        {
            "rule_id":
                "R09",

            "rule":
                "User Authorized Scope must cover Tool Effect Scope.",

            "reason":
                (
                    "A technically executable write is unsafe when its actual "
                    "effect exceeds what the user explicitly authorized."
                ),
        },

        {
            "rule_id":
                "R10",

            "rule":
                "Tool success is not sufficient evidence of business success.",

            "reason":
                (
                    "Tool output must be checked against policy, expected state, "
                    "and user intent."
                ),
        },

        {
            "rule_id":
                "R11",

            "rule":
                "Post-tool state must be verified before final success claims.",

            "reason":
                (
                    "Task 21 demonstrates that a tool can return a corrupted "
                    "state while Reward still equals 1."
                ),
        },

        {
            "rule_id":
                "R12",

            "rule":
                "Final Claim must match Tool Result and Final State.",

            "reason":
                (
                    "False or stale success claims are production-critical "
                    "even when a tool call itself succeeded."
                ),
        },

        {
            "rule_id":
                "R13",

            "rule":
                (
                    "Latest explicit user intent must be tracked subject to "
                    "Policy and Tool constraints."
                ),

            "reason":
                (
                    "Dynamic conversations can legitimately diverge from a "
                    "static pre-generated action path."
                ),
        },

        {
            "rule_id":
                "R14",

            "rule":
                "Equivalent correct computation paths should not be penalized as failures.",

            "reason":
                (
                    "Missing a specific calculator call is not a correctness "
                    "failure when the verified result is mathematically correct."
                ),
        },
    ],

    "gold_gate": {

        "required": [
            "Outcome correct",
            "Critical write arguments correct",
            "Policy compliant",
            "User authorized scope respected",
            "No unresolved environment/tool corruption",
            "Post-tool state consistent with intended state",
            "Final claim consistent with tool result/final state",
            "No high-risk hidden process failure",
        ],

        "not_required": [
            "100% exact Static Action Match",
            "Identical reasoning path to one Golden trajectory",
            "Mandatory calculator call when equivalent verified arithmetic is correct",
        ],
    },

    "buckets": {

        "GOLD":
            "Direct positive SFT eligible after audit.",

        "SILVER":
            "Outcome correct but requires trajectory cleanup/correction.",

        "SUSPECT":
            "Hold. Requires adjudication, source verification, or environment repair.",

        "NEGATIVE":
            (
                "Valid Agent failure. Use for corrected SFT, preference, "
                "or verifier construction; not as naive positive imitation."
            ),

        "MIXED":
            "Segment and relabel before any training use.",

        "EXCLUDED":
            "Do not use raw trajectory as training signal.",
    },
}


# =============================================================================
# Helpers
# =============================================================================

def normalize(
    value: Any,
) -> str:

    if value is True:
        return "YES"

    if value is False:
        return "NO"

    if value is None:
        return "UNRESOLVED"

    return str(
        value
    )


def write_json(
    path: Path,
    data: Any,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


# =============================================================================
# Validation
# =============================================================================

def validate() -> None:

    assert BASELINE[
        "task_count"
    ] == 20

    assert BASELINE[
        "raw_success_count"
    ] == 16

    assert BASELINE[
        "raw_failure_count"
    ] == 4

    assert BASELINE[
        "detailed_human_audit_count"
    ] == 8

    assert len(
        TRAJECTORIES
    ) == 8

    task_ids = [
        item[
            "task_id"
        ]
        for item in TRAJECTORIES
    ]

    assert len(
        set(
            task_ids
        )
    ) == 8

    expected_ids = {
        "16",
        "28",
        "46",
        "21",
        "59",
        "98",
        "95",
        "107",
    }

    assert set(
        task_ids
    ) == expected_ids

    by_id = {
        item[
            "task_id"
        ]:
            item
        for item in TRAJECTORIES
    }

    assert (
        by_id[
            "16"
        ][
            "quality_bucket"
        ]
        == "GOLD"
    )

    assert (
        by_id[
            "28"
        ][
            "quality_bucket"
        ]
        == "GOLD"
    )

    assert (
        by_id[
            "46"
        ][
            "quality_bucket"
        ]
        == "SILVER"
    )

    assert (
        by_id[
            "21"
        ][
            "quality_bucket"
        ]
        == "SUSPECT"
    )

    assert (
        by_id[
            "21"
        ][
            "environment_bug_source_verified"
        ]
        is True
    )

    assert (
        by_id[
            "59"
        ][
            "quality_bucket"
        ]
        == "EXCLUDED"
    )

    assert (
        by_id[
            "95"
        ][
            "quality_bucket"
        ]
        == "NEGATIVE"
    )

    assert (
        by_id[
            "107"
        ][
            "quality_bucket"
        ]
        == "NEGATIVE"
    )


# =============================================================================
# CSV
# =============================================================================

def build_csv() -> None:

    fields = [
        "task_id",
        "raw_reward",
        "action_match",
        "trajectory_role",
        "quality_bucket",
        "primary_label",
        "outcome_correct",
        "policy_compliant",
        "tool_arguments_correct",
        "post_tool_state_integrity",
        "claim_state_consistency",
        "authorized_scope_safe",
        "benchmark_suspect",
        "environment_bug",
        "raw_positive_sft",
        "corrected_positive_sft",
        "negative_training",
        "preference_training",
        "verifier_training",
        "status",
    ]

    CSV_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with CSV_OUTPUT.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for item in TRAJECTORIES:

            writer.writerow({
                field:
                    normalize(
                        item.get(
                            field
                        )
                    )
                for field in fields
            })


# =============================================================================
# Markdown
# =============================================================================

def build_markdown() -> str:

    bucket_counts = Counter(
        item[
            "quality_bucket"
        ]
        for item in TRAJECTORIES
    )

    lines: list[str] = [

        "# Trajectory Quality Taxonomy v1",
        "",
        "## 1. Scope",
        "",
        "Frozen experiment:",
        "",
        "- Retail Prompt Base / Trial-1",
        "- Raw Baseline: 16/20 = 80%",
        "- Detailed Human Audit: 8/20",
        "- Audited Reward=0: 4/4",
        "- Audited Reward=1: 4/16",
        "- Remaining Reward=1 trajectories unreviewed: 12",
        "",
        "> Important: the 12 unaudited Reward=1 trajectories are UNREVIEWED,",
        "> not automatically Gold.",
        "",
        "---",
        "",
        "## 2. Audited trajectory taxonomy",
        "",
        "| Task | Raw Reward | Action | Bucket | Primary Label | Raw Positive SFT |",
        "|---|---:|---|---|---|---|",
    ]

    for item in TRAJECTORIES:

        lines.append(
            "| "
            + item[
                "task_id"
            ]
            + " | "
            + str(
                item[
                    "raw_reward"
                ]
            )
            + " | "
            + item[
                "action_match"
            ]
            + " | "
            + item[
                "quality_bucket"
            ]
            + " | "
            + item[
                "primary_label"
            ]
            + " | "
            + item[
                "raw_positive_sft"
            ]
            + " |"
        )

    lines.extend([
        "",
        "### Bucket counts among audited 8",
        "",
    ])

    for bucket in sorted(
        bucket_counts
    ):

        lines.append(
            f"- {bucket}: "
            f"{bucket_counts[bucket]}"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Key audited examples",
        "",
        "### GOLD — Task 16",
        "",
        "Multi-goal completion, correct entity disambiguation,",
        "safe destructive-action confirmation, and claim-state consistency.",
        "",
        "The single action mismatch is only a missing explicit calculator path.",
        "",
        "### GOLD — Task 28",
        "",
        "Strong positive example for action-scope safety.",
        "",
        "The user only authorizes cancelling one item, while the available",
        "cancellation action would cancel the whole pending order.",
        "",
        "The Agent correctly refuses to expand the destructive scope.",
        "",
        "### SILVER — Task 46",
        "",
        "Outcome and critical tool writes are correct, but the trajectory",
        "contains entity-disambiguation weakness and user-context grounding loss.",
        "",
        "Correct before positive imitation training.",
        "",
        "### SUSPECT — Task 21",
        "",
        "The Agent's tool arguments are correct, but the upstream",
        "modify_pending_order_items implementation corrupts multi-item state.",
        "",
        "The Agent then fails to verify the returned state before claiming success.",
        "",
        "This is both:",
        "",
        "- a source-verified Tool / Environment bug;",
        "- a useful Post-Tool State Verification badcase.",
        "",
        "Do not use the raw trajectory as Gold.",
        "",
        "### EXCLUDED — Task 59",
        "",
        "User Simulator final intent conflicts with Static Golden.",
        "",
        "Do not convert Reward=0 directly into a negative training label.",
        "",
        "### MIXED — Task 98",
        "",
        "Contains both benchmark-alignment noise and real Agent scope/claim failures.",
        "",
        "Must be segmented and relabeled.",
        "",
        "### NEGATIVE — Task 95",
        "",
        "Valid Agent failure:",
        "",
        "Variant identity is incorrectly interpreted as physical inventory count,",
        "causing false capability-boundary detection and premature escalation.",
        "",
        "### NEGATIVE — Task 107",
        "",
        "Valid Policy Grounding failure plus Tool Enforcement Gap.",
        "",
        "---",
        "",
        "## 4. SFT Data Eligibility Rules v1",
        "",
    ])

    for rule in SFT_RULES[
        "core_principles"
    ]:

        lines.extend([
            f"### {rule['rule_id']} — {rule['rule']}",
            "",
            rule[
                "reason"
            ],
            "",
        ])

    lines.extend([
        "---",
        "",
        "## 5. Gold Gate v1",
        "",
        "A trajectory may enter raw SFT Gold only if all required gates pass:",
        "",
    ])

    for requirement in SFT_RULES[
        "gold_gate"
    ][
        "required"
    ]:

        lines.append(
            f"- {requirement}"
        )

    lines.extend([
        "",
        "The following are NOT required for Gold:",
        "",
    ])

    for item in SFT_RULES[
        "gold_gate"
    ][
        "not_required"
    ]:

        lines.append(
            f"- {item}"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 6. Current training-pool decision",
        "",
        "### Direct raw positive SFT",
        "",
        "- Task 16",
        "- Task 28",
        "",
        "### Correct before positive SFT",
        "",
        "- Task 46",
        "",
        "### Hold / exclude raw because of environment integrity",
        "",
        "- Task 21",
        "",
        "### Valid corrected-negative / preference candidates",
        "",
        "- Task 95",
        "- Task 107",
        "",
        "### Segment and relabel",
        "",
        "- Task 98",
        "",
        "### Exclude raw benchmark label",
        "",
        "- Task 59",
        "",
        "---",
        "",
        "## 7. Critical conclusion",
        "",
        "The current audited evidence rejects both naive rules:",
        "",
        "```text",
        "Reward=1 -> Gold",
        "Reward=0 -> Negative",
        "```",
        "",
        "and:",
        "",
        "```text",
        "higher Action Match -> higher trajectory quality",
        "```",
        "",
        "The actual training-quality gate must evaluate:",
        "",
        "```text",
        "Outcome",
        "  -> Policy Compliance",
        "  -> User Authorized Scope",
        "  -> Tool Arguments",
        "  -> Tool Result / State Integrity",
        "  -> Claim-State Consistency",
        "  -> Process Quality",
        "```",
        "",
        "---",
        "",
        "## 8. Boundary of this version",
        "",
        "This taxonomy is based on only 8 deeply audited trajectories.",
        "",
        "It must not be generalized to all 20 tasks yet.",
        "",
        "The remaining 12 Reward=1 trajectories remain:",
        "",
        "`UNREVIEWED_SUCCESS`",
        "",
        "until automated verifier checks and/or targeted human audit are applied.",
        "",
    ])

    return "\n".join(
        lines
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    validate()

    artifact = {
        "version":
            "v1",

        "artifact_name":
            "Trajectory Quality Taxonomy",

        "baseline":
            BASELINE,

        "upstream":
            UPSTREAM,

        "trajectories":
            TRAJECTORIES,
    }

    write_json(
        JSON_OUTPUT,
        artifact,
    )

    write_json(
        RULES_OUTPUT,
        SFT_RULES,
    )

    build_csv()

    DOCS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MD_OUTPUT.write_text(
        build_markdown(),
        encoding="utf-8",
    )

    bucket_counts = Counter(
        item[
            "quality_bucket"
        ]
        for item in TRAJECTORIES
    )

    print(
        "TRAJECTORY_QUALITY_TAXONOMY_V1_CREATED"
    )

    print(
        "RAW_BASELINE = 16/20 = 80%"
    )

    print(
        "DETAILED_AUDIT = 8/20"
    )

    print(
        "UNREVIEWED_REWARD1 = 12"
    )

    for bucket in sorted(
        bucket_counts
    ):

        print(
            "BUCKET",
            bucket,
            "=",
            bucket_counts[
                bucket
            ],
        )

    print(
        "DIRECT_GOLD_TASKS = 16,28"
    )

    print(
        "SILVER_TASKS = 46"
    )

    print(
        "SUSPECT_TASKS = 21"
    )

    print(
        "VALID_NEGATIVE_TASKS = 95,107"
    )

    print(
        "MIXED_TASKS = 98"
    )

    print(
        "EXCLUDED_TASKS = 59"
    )

    print(
        "JSON =",
        JSON_OUTPUT,
    )

    print(
        "CSV =",
        CSV_OUTPUT,
    )

    print(
        "RULES =",
        RULES_OUTPUT,
    )

    print(
        "MARKDOWN =",
        MD_OUTPUT,
    )

    print(
        "TRAJECTORY_QUALITY_TAXONOMY_V1_OK"
    )


if __name__ == "__main__":
    main()
