"""
Build Failure Taxonomy v2 and Training Data Eligibility Matrix v2.

Background
----------
Source experiment:
Retail Prompt Base / Trial-1

Frozen raw baseline:
16 / 20 = 80%

Failure tasks:
59, 98, 95, 107

v2 change
---------
Task 107 was previously marked unresolved.

After Policy / Tool source-code verification against the frozen upstream
tau2-bench commit, Task 107 is reclassified as:

VALID_AGENT_FAILURE
+
POLICY_GROUNDING_FAILURE
+
POLICY_TOOL_ENFORCEMENT_GAP

Important
---------
This script does NOT modify the frozen raw baseline score.

Human-audited labels are used only for:
- failure taxonomy
- training-data eligibility
- verifier design
- SFT / preference-data construction

Offline only.
No API calls.
"""

from __future__ import annotations

import csv
import json
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
    / "failure_taxonomy_v2.json"
)

CSV_OUTPUT = (
    RUN_DIR
    / "training_data_eligibility_matrix_v2.csv"
)

MD_OUTPUT = (
    DOCS_DIR
    / "02_评测与失败分析"
    / "2026-07-22_失败分类体系_v2.md"
)

DIFF_OUTPUT = (
    DOCS_DIR
    / "02_评测与失败分析"
    / "2026-07-22_失败分类体系_v1到v2变更说明.md"
)

V1_JSON_PATH = (
    RUN_DIR
    / "failure_taxonomy_v1.json"
)


# =============================================================================
# Frozen experiment metadata
# =============================================================================

BASELINE = {
    "experiment_role":
        "Prompt Base / Trial-1",

    "domain":
        "retail",

    "task_count":
        20,

    "business_success_count":
        16,

    "business_failure_count":
        4,

    "raw_success_rate":
        0.80,

    "raw_failure_task_ids": [
        "59",
        "98",
        "95",
        "107",
    ],

    "raw_metric_policy":
        (
            "Human audit must not retroactively modify "
            "the frozen raw benchmark result."
        ),
}


UPSTREAM = {
    "repository":
        "sierra-research/tau2-bench",

    "frozen_commit":
        "58e5e1ace69302e6982d27014569c03e0ffccdd2",

    "retail_policy_source":
        r"D:\tau2-bench\data\tau2\domains\retail\policy.md",

    "retail_tools_source":
        r"D:\tau2-bench\src\tau2\domains\retail\tools.py",
}


# =============================================================================
# Failure Taxonomy v2
# =============================================================================

FAILURES: list[dict[str, Any]] = [

    # -------------------------------------------------------------------------
    # Task 59
    # -------------------------------------------------------------------------

    {
        "task_id":
            "59",

        "raw_reward":
            0.0,

        "raw_db_reward":
            0.0,

        "raw_nl_reward":
            0.0,

        "human_audit_category":
            "BENCHMARK_ALIGNMENT_FAILURE",

        "primary_label":
            "USER_SIMULATOR_GOLD_MISMATCH",

        "secondary_labels": [
            "STATIC_GOLD_FINAL_INTENT_MISMATCH",
            "EVALUATOR_SPECIFICATION_ALIGNMENT_ISSUE",
        ],

        "real_agent_failure":
            False,

        "benchmark_suspect":
            True,

        "evaluator_suspect":
            True,

        "tool_enforcement_gap":
            False,

        "claim_action_consistency":
            "PASS",

        "summary":
            (
                "The User Simulator explicitly authorized cancellation of "
                "#W2702727 and no change to #W8268610, while Static Golden "
                "expected cancellation of #W8268610 and modification of "
                "#W2702727. The Agent followed the final explicit user intent."
            ),

        "first_divergence_point":
            (
                "The generated user trajectory diverged from Static Golden "
                "before the Agent made the disputed write action."
            ),

        "negative_sft_eligibility":
            "EXCLUDE",

        "corrected_sft_eligibility":
            "POSSIBLE_AFTER_ADJUDICATION",

        "preference_eligibility":
            "EXCLUDE_AS_RAW_PAIR",

        "verifier_eligibility":
            "YES",

        "rl_eligibility":
            "NO",

        "training_status":
            "EXCLUDE_FROM_RAW_NEGATIVE_POOL",

        "training_priority":
            "AUDIT_ONLY",

        "recommended_use": [
            "Benchmark audit",
            "User Simulator / Golden consistency checking",
            "Final authorized intent tracking",
            "Reward-label-noise detection",
        ],
    },

    # -------------------------------------------------------------------------
    # Task 98
    # -------------------------------------------------------------------------

    {
        "task_id":
            "98",

        "raw_reward":
            0.0,

        "raw_db_reward":
            0.0,

        "raw_nl_reward":
            1.0,

        "human_audit_category":
            "MIXED_BADCASE",

        "primary_label":
            "CLAIM_ACTION_INCONSISTENCY",

        "secondary_labels": [
            "DYNAMIC_INTENT_STATIC_GOLD_MISMATCH",
            "ACTION_SCOPE_CONFIRMATION_FAILURE",
            "EVALUATOR_COVERAGE_GAP",
        ],

        "real_agent_failure":
            True,

        "benchmark_suspect":
            True,

        "evaluator_suspect":
            True,

        "tool_enforcement_gap":
            False,

        "claim_action_consistency":
            "FAIL_ON_CANCELLATION",

        "summary":
            (
                "All three write tools executed successfully. "
                "Two exchange actions mismatched Static Golden because the Agent "
                "used the payment method explicitly confirmed by the user. "
                "Separately, the Agent made a real production-risk error: "
                "the cancellation tool cancelled the entire order and refunded "
                "$1058.79, but the Agent described it as cancelling only the "
                "skateboard with a $202.13 refund."
            ),

        "first_divergence_point":
            (
                "Benchmark divergence begins at payment-method confirmation. "
                "The first genuine Agent failure occurs after the cancellation "
                "Tool result, when the Agent misstates the actual action scope "
                "and refund amount."
            ),

        "negative_sft_eligibility":
            "SEGMENT_ONLY",

        "corrected_sft_eligibility":
            "YES_AFTER_SEGMENTATION",

        "preference_eligibility":
            "YES_AFTER_RELABELING",

        "verifier_eligibility":
            "YES_HIGH_VALUE",

        "rl_eligibility":
            "NOT_YET",

        "training_status":
            "SEGMENT_AND_RELABEL",

        "training_priority":
            "HIGH",

        "recommended_use": [
            "Action-scope verifier",
            "Post-tool state grounding",
            "Claim-action consistency verifier",
            "Final authorized intent tracking",
        ],
    },

    # -------------------------------------------------------------------------
    # Task 95
    # -------------------------------------------------------------------------

    {
        "task_id":
            "95",

        "raw_reward":
            0.0,

        "raw_db_reward":
            0.0,

        "raw_nl_reward":
            0.0,

        "human_audit_category":
            "VALID_AGENT_FAILURE",

        "primary_label":
            "ENVIRONMENT_STATE_SEMANTICS_MISUNDERSTANDING",

        "secondary_labels": [
            "FALSE_CAPABILITY_BOUNDARY_DETECTION",
            "PREMATURE_ESCALATION",
            "INCOMPLETE_MULTI_GOAL_EXECUTION",
        ],

        "real_agent_failure":
            True,

        "benchmark_suspect":
            False,

        "evaluator_suspect":
            False,

        "tool_enforcement_gap":
            False,

        "claim_action_consistency":
            "FALSE_CAPABILITY_CLAIM",

        "summary":
            (
                "The Agent correctly found target Variant 9844888101 but "
                "misinterpreted one matching Variant as one physical inventory "
                "unit. It incorrectly concluded that only one of two laptops "
                "could use that Variant and prematurely escalated to a human."
            ),

        "first_divergence_point":
            (
                "After observing item_id=9844888101 with available=true, "
                "the Agent incorrectly inferred that only one physical unit "
                "was available."
            ),

        "negative_sft_eligibility":
            "YES_WITH_CORRECTED_TARGET",

        "corrected_sft_eligibility":
            "YES",

        "preference_eligibility":
            "YES",

        "verifier_eligibility":
            "YES_HIGH_VALUE",

        "rl_eligibility":
            "NOT_YET",

        "training_status":
            "ELIGIBLE_AFTER_CORRECTION",

        "training_priority":
            "VERY_HIGH",

        "recommended_use": [
            "SFT badcase correction",
            "Environment schema grounding",
            "Premature-escalation verifier",
            "Multi-goal completeness verifier",
            "Preference pair construction",
        ],
    },

    # -------------------------------------------------------------------------
    # Task 107
    # -------------------------------------------------------------------------

    {
        "task_id":
            "107",

        "raw_reward":
            0.0,

        "raw_db_reward":
            0.0,

        "raw_nl_reward":
            1.0,

        "human_audit_category":
            "VALID_AGENT_FAILURE",

        "primary_label":
            "POLICY_GROUNDING_FAILURE",

        "secondary_labels": [
            "POLICY_TOOL_ENFORCEMENT_GAP",
            "INVALID_ACTION_ALLOWED_BY_TOOL",
            "NL_EVALUATOR_COVERAGE_GAP",
        ],

        "real_agent_failure":
            True,

        "benchmark_suspect":
            False,

        "evaluator_suspect":
            "PARTIAL_COVERAGE_GAP",

        "tool_enforcement_gap":
            True,

        "claim_action_consistency":
            "TOOL_PASS_POLICY_FAIL",

        "summary":
            (
                "The Agent incorrectly treated same-variant exchange "
                "1615379700 -> 1615379700 as Policy-compliant. "
                "Retail Policy requires exchange to a different product option. "
                "The Tool nevertheless accepted the invalid same-item exchange, "
                "revealing a Policy-Tool Enforcement Gap."
            ),

        "first_divergence_point":
            (
                "The Agent states that because the same Hiking Boots Variant "
                "is available, it can exchange the item for the same item. "
                "This violates the Retail exchange Policy requirement that the "
                "replacement use a different product option."
            ),

        "source_verified":
            True,

        "source_verification": {
            "upstream_commit":
                UPSTREAM[
                    "frozen_commit"
                ],

            "policy_evidence":
                (
                    "Retail exchange Policy requires a replacement item "
                    "from the same product but with different product options."
                ),

            "tool_evidence":
                (
                    "exchange_delivered_order_items does not enforce "
                    "old_item_id != new_item_id, while another related "
                    "pending-order modification path explicitly contains "
                    "same-item validation."
                ),

            "policy_source":
                UPSTREAM[
                    "retail_policy_source"
                ],

            "tool_source":
                UPSTREAM[
                    "retail_tools_source"
                ],
        },

        "negative_sft_eligibility":
            "YES",

        "corrected_sft_eligibility":
            "YES",

        "preference_eligibility":
            "YES",

        "verifier_eligibility":
            "YES_VERY_HIGH_VALUE",

        "rl_eligibility":
            "NOT_YET",

        "training_status":
            "ELIGIBLE_AFTER_CORRECTION",

        "training_priority":
            "VERY_HIGH",

        "recommended_use": [
            "Policy-grounding SFT",
            "Policy-aware preference pair",
            "Pre-tool Policy verifier",
            "Policy-Tool consistency testing",
            "Tool guardrail regression test",
        ],
    },
]


# =============================================================================
# Helpers
# =============================================================================

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


# =============================================================================
# Validation
# =============================================================================

def validate() -> None:

    assert (
        BASELINE[
            "task_count"
        ]
        == 20
    )

    assert (
        BASELINE[
            "business_success_count"
        ]
        == 16
    )

    assert (
        BASELINE[
            "business_failure_count"
        ]
        == 4
    )

    assert (
        BASELINE[
            "raw_success_rate"
        ]
        == 0.80
    )

    task_ids = [
        item[
            "task_id"
        ]
        for item
        in FAILURES
    ]

    assert task_ids == [
        "59",
        "98",
        "95",
        "107",
    ]

    assert len(
        set(
            task_ids
        )
    ) == 4

    task59 = next(
        item
        for item in FAILURES
        if item[
            "task_id"
        ] == "59"
    )

    assert (
        task59[
            "real_agent_failure"
        ]
        is False
    )

    task95 = next(
        item
        for item in FAILURES
        if item[
            "task_id"
        ] == "95"
    )

    assert (
        task95[
            "human_audit_category"
        ]
        == "VALID_AGENT_FAILURE"
    )

    task107 = next(
        item
        for item in FAILURES
        if item[
            "task_id"
        ] == "107"
    )

    assert (
        task107[
            "human_audit_category"
        ]
        == "VALID_AGENT_FAILURE"
    )

    assert (
        task107[
            "primary_label"
        ]
        == "POLICY_GROUNDING_FAILURE"
    )

    assert (
        task107[
            "tool_enforcement_gap"
        ]
        is True
    )

    assert (
        task107[
            "source_verified"
        ]
        is True
    )


# =============================================================================
# CSV
# =============================================================================

def build_csv() -> None:

    fields = [
        "task_id",
        "raw_reward",
        "human_audit_category",
        "primary_label",
        "real_agent_failure",
        "benchmark_suspect",
        "evaluator_suspect",
        "tool_enforcement_gap",
        "claim_action_consistency",
        "negative_sft_eligibility",
        "corrected_sft_eligibility",
        "preference_eligibility",
        "verifier_eligibility",
        "rl_eligibility",
        "training_status",
        "training_priority",
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

        for item in FAILURES:

            row = {}

            for field in fields:

                row[
                    field
                ] = normalize(
                    item.get(
                        field
                    )
                )

            writer.writerow(
                row
            )


# =============================================================================
# Markdown report
# =============================================================================

def build_markdown() -> str:

    lines: list[str] = [
        "# Failure Taxonomy v2 + Training Data Eligibility Matrix v2",
        "",
        "## 1. 冻结实验结果",
        "",
        "- 实验：Retail Prompt Base / Trial-1",
        "- 总任务数：20",
        "- Raw Success：16",
        "- Raw Failure：4",
        "- Raw Success Rate：80%",
        "",
        "> Failure Taxonomy 的人工审计不会回改冻结的 Raw Baseline。",
        "> 原始结果始终保留为 16/20 = 80%。",
        "",
        "---",
        "",
        "## 2. Failure Taxonomy v2",
        "",
        "| Task | Raw | 人工审计分类 | 主标签 | 真实 Agent Failure | Benchmark 可疑 | Tool Enforcement Gap |",
        "|---|---:|---|---|---|---|---|",
    ]

    for item in FAILURES:

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
                "human_audit_category"
            ]
            + " | "
            + item[
                "primary_label"
            ]
            + " | "
            + normalize(
                item[
                    "real_agent_failure"
                ]
            )
            + " | "
            + normalize(
                item[
                    "benchmark_suspect"
                ]
            )
            + " | "
            + normalize(
                item[
                    "tool_enforcement_gap"
                ]
            )
            + " |"
        )

    lines.extend([
        "",
        "### 当前结构",
        "",
        "```text",
        "4 个 Raw Reward=0",
        "",
        "├── Benchmark Alignment Failure",
        "│   └── Task 59",
        "│",
        "├── Mixed Badcase",
        "│   └── Task 98",
        "│",
        "└── Valid Agent Failure",
        "    ├── Task 95：Environment Semantics",
        "    └── Task 107：Policy Grounding",
        "```",
        "",
        "---",
        "",
        "## 3. Training Data Eligibility Matrix v2",
        "",
        "| Task | 负 SFT | 修正 SFT | Preference | Verifier | RL | 状态 |",
        "|---|---|---|---|---|---|---|",
    ])

    for item in FAILURES:

        lines.append(
            "| "
            + item[
                "task_id"
            ]
            + " | "
            + item[
                "negative_sft_eligibility"
            ]
            + " | "
            + item[
                "corrected_sft_eligibility"
            ]
            + " | "
            + item[
                "preference_eligibility"
            ]
            + " | "
            + item[
                "verifier_eligibility"
            ]
            + " | "
            + item[
                "rl_eligibility"
            ]
            + " | "
            + item[
                "training_status"
            ]
            + " |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. 各任务最终裁决",
        "",
    ])

    for item in FAILURES:

        lines.extend([
            f"### Task {item['task_id']}",
            "",
            f"**分类：** `{item['human_audit_category']}`",
            "",
            f"**主标签：** `{item['primary_label']}`",
            "",
            "**结论：**",
            "",
            item[
                "summary"
            ],
            "",
            "**第一次偏离点：**",
            "",
            item[
                "first_divergence_point"
            ],
            "",
            "**推荐用途：**",
            "",
        ])

        for usage in item[
            "recommended_use"
        ]:

            lines.append(
                f"- {usage}"
            )

        lines.extend([
            "",
            "---",
            "",
        ])

    lines.extend([
        "## 5. Task 107 源码核验后的关键结论",
        "",
        "Task 107 在 v1 中属于待核验案例。",
        "",
        "源码核验后确认：",
        "",
        "```text",
        "Retail Policy",
        "要求 exchange 使用 different product option",
        "",
        "        ↓",
        "",
        "Agent",
        "却执行 old_item_id == new_item_id",
        "",
        "        ↓",
        "",
        "exchange_delivered_order_items",
        "没有阻止这一非法动作",
        "",
        "        ↓",
        "",
        "Tool 返回成功",
        "",
        "        ↓",
        "",
        "NL Evaluator",
        "只检查两个订单是否发生 exchange",
        "没有检查 Policy compliance",
        "```",
        "",
        "因此 Task 107 是一个完整的四层失配案例：",
        "",
        "```text",
        "Policy Rule",
        "    ↓",
        "Agent Policy Grounding",
        "    ↓",
        "Tool Enforcement",
        "    ↓",
        "Evaluator Coverage",
        "```",
        "",
        "最终标签：",
        "",
        "- `POLICY_GROUNDING_FAILURE`",
        "- `POLICY_TOOL_ENFORCEMENT_GAP`",
        "- `INVALID_ACTION_ALLOWED_BY_TOOL`",
        "- `NL_EVALUATOR_COVERAGE_GAP`",
        "",
        "---",
        "",
        "## 6. 当前最优先的训练样本",
        "",
        "### 第一优先级：Task 95",
        "",
        "训练能力：",
        "",
        "- Environment Schema Understanding",
        "- Variant / Inventory 语义区分",
        "- Capability Boundary Judgment",
        "- Avoid Premature Escalation",
        "- Multi-goal Completion",
        "",
        "### 第一优先级：Task 107",
        "",
        "训练能力：",
        "",
        "- Policy Grounding",
        "- Policy-aware Tool Calling",
        "- Conditional Intent / Fallback Handling",
        "- Pre-tool Policy Verification",
        "- Tool Enforcement Gap Detection",
        "",
        "### Verifier 高价值样本：Task 98",
        "",
        "重点：",
        "",
        "```text",
        "User Authorized Scope",
        "→ Tool Actual Scope",
        "→ Tool Result",
        "→ Final DB State",
        "→ Agent Final Claim",
        "```",
        "",
        "### 排除普通负样本池：Task 59",
        "",
        "原因：",
        "",
        "Simulator / Static Golden 与最终用户意图发生冲突。",
        "",
        "---",
        "",
        "## 7. 当前后训练结论",
        "",
        "当前证据支持的路线是：",
        "",
        "```text",
        "Prompt Base",
        "    ↓",
        "Trajectory Audit",
        "    ↓",
        "Failure Taxonomy",
        "    ↓",
        "Training Data Cleaning",
        "    ↓",
        "Corrected SFT",
        "    +",
        "Verifier",
        "    ↓",
        "重新评测",
        "    ↓",
        "再决定是否需要 Preference / RL",
        "```",
        "",
        "目前没有足够证据支持直接跳到 RL。",
        "",
        "RL 必须由 SFT / Verifier 后仍持续存在的系统性失败来证明必要性。",
        "",
        "---",
        "",
        "## 8. 版本边界",
        "",
        "Failure Taxonomy v2 仅来自当前冻结的 20-task Trial-1。",
        "",
        "它不能代表整个 tau2-bench 的全局失败分布。",
        "",
        "后续仍需要：",
        "",
        "- 对成功轨迹做质量审计；",
        "- 做多 Trial 稳定性实验；",
        "- 扩展训练数据；",
        "- 重新运行 Base / SFT / Verifier 对照实验。",
        "",
    ])

    return "\n".join(
        lines
    )


# =============================================================================
# v1 -> v2 diff report
# =============================================================================

def build_diff_markdown() -> str:

    v1_exists = (
        V1_JSON_PATH.exists()
    )

    lines = [
        "# Failure Taxonomy v1 → v2 变更记录",
        "",
        f"v1 JSON 是否存在：`{v1_exists}`",
        "",
        "## 核心变更",
        "",
        "只有 Task 107 的最终裁决发生实质变化。",
        "",
        "### v1",
        "",
        "```text",
        "UNRESOLVED_POLICY_TOOL_SEMANTICS",
        "CONDITIONAL_BRANCH_STATIC_GOLD_MISMATCH",
        "HOLD_UNTIL_POLICY_VERIFIED",
        "```",
        "",
        "### 源码核验",
        "",
        f"冻结上游 commit：`{UPSTREAM['frozen_commit']}`",
        "",
        "核验对象：",
        "",
        f"- `{UPSTREAM['retail_policy_source']}`",
        f"- `{UPSTREAM['retail_tools_source']}`",
        "",
        "核验结论：",
        "",
        "1. Retail Policy 要求 exchange 使用不同 product option。",
        "2. Agent 执行了 old_item_id == new_item_id。",
        "3. exchange Tool 未强制阻止该非法动作。",
        "4. NL Evaluator 只检查是否发生 exchange，没有检查 Policy compliance。",
        "",
        "### v2",
        "",
        "```text",
        "VALID_AGENT_FAILURE",
        "POLICY_GROUNDING_FAILURE",
        "POLICY_TOOL_ENFORCEMENT_GAP",
        "ELIGIBLE_AFTER_CORRECTION",
        "```",
        "",
        "## 未发生变化",
        "",
        "- Task 59：仍为 Benchmark Alignment Failure。",
        "- Task 98：仍为 Mixed Badcase。",
        "- Task 95：仍为 Valid Agent Failure。",
        "- Raw Baseline：仍为 16/20 = 80%。",
        "",
        "## 版本原则",
        "",
        "不删除历史结论。",
        "",
        "v1 用于保留第一次人工审计过程；",
        "v2 用于记录源码核验后的最新裁决。",
        "",
    ]

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
            "v2",

        "artifact_name":
            "Failure Taxonomy and Training Data Eligibility Matrix",

        "baseline":
            BASELINE,

        "upstream":
            UPSTREAM,

        "version_change": {
            "changed_task_ids": [
                "107"
            ],

            "reason":
                (
                    "Task 107 Policy and Tool semantics were verified "
                    "against the frozen upstream source code."
                ),
        },

        "audit_policy": {
            "raw_reward_is_not_training_label":
                True,

            "raw_baseline_must_not_be_retroactively_modified":
                True,

            "benchmark_alignment_cases_must_be_excluded_from_raw_negative_pool":
                True,

            "mixed_badcases_require_segment_level_relabeling":
                True,

            "policy_grounding_failures_are_valid_training_candidates":
                True,
        },

        "failures":
            FAILURES,
    }

    write_json(
        JSON_OUTPUT,
        artifact,
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

    DIFF_OUTPUT.write_text(
        build_diff_markdown(),
        encoding="utf-8",
    )

    # ASCII-only terminal output for Windows compatibility.
    print(
        "FAILURE_TAXONOMY_V2_CREATED"
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
        "MARKDOWN =",
        MD_OUTPUT,
    )

    print(
        "DIFF =",
        DIFF_OUTPUT,
    )

    print(
        "RAW_BASELINE = 16/20 = 80%"
    )

    print(
        "TASK_59 = BENCHMARK_ALIGNMENT_FAILURE"
    )

    print(
        "TASK_98 = MIXED_BADCASE"
    )

    print(
        "TASK_95 = VALID_AGENT_FAILURE"
    )

    print(
        "TASK_107 = VALID_AGENT_FAILURE"
    )

    print(
        "VALID_AGENT_FAILURE_COUNT = 2"
    )

    print(
        "FAILURE_TAXONOMY_V2_OK"
    )


if __name__ == "__main__":
    main()
