"""
Build Failure Taxonomy v1 and Training Data Eligibility Matrix v1.

Source
------
Human audit of the four Reward=0 tasks from:

Retail Prompt Base Trial-1
Run:
20260722_110504_retail_baseline20_trial1_deepseek

Important
---------
This taxonomy is a human-audited project artifact.

It does NOT modify the frozen raw baseline score.

Raw baseline remains:

16 / 20 = 80%

The purpose of this artifact is to distinguish:

- real Agent failures
- benchmark / evaluator alignment problems
- mixed badcases
- unresolved policy / tool semantics

before using trajectories for:

- SFT
- Preference Optimization
- Verifier training
- RL / post-training

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
    / "failure_taxonomy_v1.json"
)

CSV_OUTPUT = (
    RUN_DIR
    / "training_data_eligibility_matrix_v1.csv"
)

MD_OUTPUT = (
    DOCS_DIR
    / "20260722_failure_taxonomy_v1.md"
)


# =============================================================================
# Frozen baseline metadata
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

    "important_note":
        (
            "Raw baseline score remains 16/20 = 80%. "
            "Human audit labels do not retroactively modify "
            "the frozen benchmark result."
        ),
}


# =============================================================================
# Failure Taxonomy v1
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

        "human_audit_summary":
            (
                "User Simulator explicitly instructed the Agent to cancel "
                "#W2702727 and leave #W8268610 unchanged, while Static Golden "
                "expected cancellation of #W8268610 and address modification "
                "of #W2702727. The Agent followed the final explicit user "
                "authorization after clarification and confirmation."
            ),

        "first_divergence_point":
            (
                "The divergence begins in the generated user trajectory: "
                "the User Simulator identifies #W2702727 as the older order "
                "to cancel, conflicting with the Static Golden branch."
            ),

        "real_agent_failure":
            False,

        "benchmark_suspect":
            True,

        "evaluator_suspect":
            True,

        "policy_issue":
            False,

        "tool_execution_failure":
            False,

        "claim_action_consistency":
            "PASS",

        "whole_trajectory_negative_sft":
            "NO",

        "corrected_positive_sft_candidate":
            "POSSIBLE_AFTER_ADJUDICATION",

        "preference_training":
            "NO_AS_RAW_REWARD_PAIR",

        "verifier_training":
            "YES_FOR_BENCHMARK_ALIGNMENT_AND_FINAL_INTENT",

        "rl_training":
            "NO",

        "training_priority":
            "AUDIT_ONLY",

        "training_status":
            "EXCLUDE_FROM_RAW_NEGATIVE_POOL",

        "recommended_use": [
            "Benchmark / evaluator auditing",
            "User Simulator consistency checking",
            "Final authorized intent tracking",
            "Reward-label-noise detection",
        ],

        "do_not_do": [
            "Do not treat Reward=0 as direct Agent negative label",
            "Do not train the Agent to ignore the user's final explicit intent",
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
            "DYNAMIC_INTENT_STATIC_GOLD_MISMATCH",

        "secondary_labels": [
            "CLAIM_ACTION_INCONSISTENCY",
            "ACTION_SCOPE_CONFIRMATION_FAILURE",
            "EVALUATOR_COVERAGE_GAP",
        ],

        "human_audit_summary":
            (
                "All three write tools actually executed successfully. "
                "The two exchange actions mismatched Static Golden only because "
                "the Agent used the Visa card explicitly confirmed by the user, "
                "while Static Golden expected a different card. "
                "Separately, the Agent made a real production-risk error: "
                "cancel_pending_order cancelled the entire order and refunded "
                "$1058.79, but the Agent described it as cancelling only the "
                "skateboard with a $202.13 refund."
            ),

        "first_divergence_point":
            (
                "Benchmark divergence occurs when the user explicitly confirms "
                "credit_card_3951670 while Static Golden expects "
                "credit_card_8105988. "
                "The first real Agent error occurs after cancellation when "
                "the Agent misrepresents whole-order cancellation as "
                "single-item cancellation."
            ),

        "real_agent_failure":
            True,

        "benchmark_suspect":
            True,

        "evaluator_suspect":
            True,

        "policy_issue":
            True,

        "tool_execution_failure":
            False,

        "claim_action_consistency":
            "FAIL_ON_CANCELLATION",

        "whole_trajectory_negative_sft":
            "NO",

        "corrected_positive_sft_candidate":
            "SEGMENT_LEVEL_ONLY",

        "preference_training":
            "YES_AFTER_SEGMENTATION_AND_RELABELING",

        "verifier_training":
            "YES_HIGH_VALUE",

        "rl_training":
            "NOT_YET",

        "training_priority":
            "HIGH",

        "training_status":
            "SEGMENT_AND_RELABEL",

        "recommended_use": [
            "Action-scope verifier",
            "Post-tool state grounding",
            "Claim-action consistency verifier",
            "Final authorized payment-method tracking",
            "Mixed reward-label-noise analysis",
        ],

        "do_not_do": [
            "Do not treat the entire Reward=0 trajectory as a single negative",
            "Do not penalize the user-confirmed payment method solely because Static Golden differs",
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

        "human_audit_summary":
            (
                "The Agent correctly found the target laptop variant "
                "9844888101, but incorrectly interpreted one matching Variant "
                "as one physical inventory unit. It therefore concluded that "
                "only one of the two laptops could be exchanged and "
                "prematurely escalated to a human. Static Golden shows that "
                "both orders should use the same target variant."
            ),

        "first_divergence_point":
            (
                "After finding item_id=9844888101 with available=true, "
                "the Agent incorrectly infers that only one physical unit "
                "exists and that a second exchange to the same Variant is "
                "impossible."
            ),

        "real_agent_failure":
            True,

        "benchmark_suspect":
            False,

        "evaluator_suspect":
            False,

        "policy_issue":
            False,

        "tool_execution_failure":
            False,

        "claim_action_consistency":
            "NO_FAKE_SUCCESS_BUT_FALSE_CAPABILITY_CLAIM",

        "whole_trajectory_negative_sft":
            "YES_WITH_CORRECTED_TARGET",

        "corrected_positive_sft_candidate":
            "YES",

        "preference_training":
            "YES",

        "verifier_training":
            "YES_HIGH_VALUE",

        "rl_training":
            "NOT_YET",

        "training_priority":
            "VERY_HIGH",

        "training_status":
            "ELIGIBLE_AFTER_CORRECTION",

        "recommended_use": [
            "SFT badcase correction",
            "Environment-schema grounding",
            "Tool-state semantic reasoning",
            "Premature-escalation verifier",
            "Multi-goal completeness verifier",
            "Preference pair construction",
        ],

        "do_not_do": [
            "Do not confuse Variant identity with physical inventory count",
            "Do not escalate without evidence of a real capability boundary",
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
            "UNRESOLVED_POLICY_TOOL_SEMANTICS",

        "primary_label":
            "CONDITIONAL_BRANCH_STATIC_GOLD_MISMATCH",

        "secondary_labels": [
            "TOOL_SEMANTICS_AMBIGUITY",
            "POLICY_SEMANTICS_NEEDS_VERIFICATION",
            "EVALUATOR_BRANCH_ALIGNMENT_ISSUE",
        ],

        "human_audit_summary":
            (
                "The user first requested a fresh pair of hiking boots with "
                "the same specs. Only if that exchange was not allowed should "
                "the fallback be size 9, leather, waterproof. "
                "The Agent performed a same-variant exchange, which the Tool "
                "accepted successfully. Static Golden instead fixes the "
                "fallback variant 8106223139. "
                "Policy semantics must be verified before declaring the Agent wrong."
            ),

        "first_divergence_point":
            (
                "The Agent chooses the primary same-spec replacement branch "
                "because the original variant is available and the Tool accepts "
                "same-item exchange, while Static Golden assumes the fallback branch."
            ),

        "real_agent_failure":
            None,

        "benchmark_suspect":
            True,

        "evaluator_suspect":
            True,

        "policy_issue":
            "UNRESOLVED",

        "tool_execution_failure":
            False,

        "claim_action_consistency":
            "PASS",

        "whole_trajectory_negative_sft":
            "NO",

        "corrected_positive_sft_candidate":
            "HOLD",

        "preference_training":
            "HOLD",

        "verifier_training":
            "YES_FOR_BRANCH_AND_POLICY_CHECKING",

        "rl_training":
            "NO",

        "training_priority":
            "VERIFY_FIRST",

        "training_status":
            "HOLD_UNTIL_POLICY_VERIFIED",

        "recommended_use": [
            "Conditional-intent tracking",
            "Branch-aware evaluator research",
            "Policy/tool semantic consistency audit",
            "Same-variant replacement rule verification",
        ],

        "do_not_do": [
            "Do not use as a negative Agent sample before Policy verification",
            "Do not assume DB=0 proves the Tool execution was incorrect",
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


def bool_text(
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


def build_csv() -> None:

    fields = [
        "task_id",
        "raw_reward",
        "human_audit_category",
        "primary_label",
        "real_agent_failure",
        "benchmark_suspect",
        "evaluator_suspect",
        "policy_issue",
        "tool_execution_failure",
        "claim_action_consistency",
        "whole_trajectory_negative_sft",
        "corrected_positive_sft_candidate",
        "preference_training",
        "verifier_training",
        "rl_training",
        "training_priority",
        "training_status",
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

                value = item.get(
                    field
                )

                if isinstance(
                    value,
                    bool,
                ) or value is None:

                    value = bool_text(
                        value
                    )

                row[field] = value

            writer.writerow(
                row
            )


def build_markdown() -> str:

    lines: list[str] = []

    lines.extend([
        "# Failure Taxonomy v1 + Training Data Eligibility Matrix v1",
        "",
        "## 1. 实验背景",
        "",
        "来源实验：Retail Prompt Base / Trial-1",
        "",
        f"- 总任务数：{BASELINE['task_count']}",
        f"- Raw 成功：{BASELINE['business_success_count']}",
        f"- Raw 失败：{BASELINE['business_failure_count']}",
        f"- Raw Success Rate：{BASELINE['raw_success_rate']:.0%}",
        "",
        "> 重要：人工审计不会回改冻结的 Raw Baseline。",
        "> Raw Baseline 仍然保持 16/20 = 80%。",
        "",
        "人工审计的目标不是“提高分数”，而是判断 Reward=0 到底代表：",
        "",
        "- 真正的 Agent Failure；",
        "- Benchmark / Evaluator 对齐问题；",
        "- 混合型 Badcase；",
        "- Policy / Tool 语义尚未解决。",
        "",
        "---",
        "",
        "## 2. Failure Taxonomy v1",
        "",
        "| Task | Raw Reward | 人工审计分类 | 主标签 | 是否真实 Agent Failure | 是否 Benchmark 可疑 | 训练状态 |",
        "|---|---:|---|---|---|---|---|",
    ])

    for item in FAILURES:

        lines.append(
            "| "
            + str(
                item["task_id"]
            )
            + " | "
            + str(
                item["raw_reward"]
            )
            + " | "
            + str(
                item["human_audit_category"]
            )
            + " | "
            + str(
                item["primary_label"]
            )
            + " | "
            + bool_text(
                item["real_agent_failure"]
            )
            + " | "
            + bool_text(
                item["benchmark_suspect"]
            )
            + " | "
            + str(
                item["training_status"]
            )
            + " |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Training Data Eligibility Matrix v1",
        "",
        "| Task | 整条作为负 SFT | 修正后正向 SFT | Preference | Verifier | RL | 优先级 |",
        "|---|---|---|---|---|---|---|",
    ])

    for item in FAILURES:

        lines.append(
            "| "
            + str(
                item["task_id"]
            )
            + " | "
            + str(
                item["whole_trajectory_negative_sft"]
            )
            + " | "
            + str(
                item["corrected_positive_sft_candidate"]
            )
            + " | "
            + str(
                item["preference_training"]
            )
            + " | "
            + str(
                item["verifier_training"]
            )
            + " | "
            + str(
                item["rl_training"]
            )
            + " | "
            + str(
                item["training_priority"]
            )
            + " |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. 四条失败的人工审计结论",
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
            str(
                item["human_audit_summary"]
            ),
            "",
            "**第一次偏离点：**",
            "",
            str(
                item["first_divergence_point"]
            ),
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
            "**禁止直接做的事情：**",
            "",
        ])

        for warning in item[
            "do_not_do"
        ]:

            lines.append(
                f"- {warning}"
            )

        lines.extend([
            "",
            "---",
            "",
        ])

    lines.extend([
        "## 5. 当前最重要的项目结论",
        "",
        "### 5.1 Reward=0 不等于 Agent Failure",
        "",
        "四条 Raw Failure 经人工审计后：",
        "",
        "- Task 59：主要是 User Simulator / Static Golden 对齐问题；",
        "- Task 98：Benchmark mismatch 与真实 Agent 错误并存；",
        "- Task 95：当前最干净、最明确的真实 Agent Failure；",
        "- Task 107：需要先验证 Policy / Tool 对 same-variant exchange 的真实语义。",
        "",
        "因此不能直接执行：",
        "",
        "```text",
        "Reward=1 -> 正样本",
        "Reward=0 -> 负样本",
        "```",
        "",
        "这种粗粒度数据构造会把 Evaluator Label Noise 注入后训练数据。",
        "",
        "### 5.2 当前最有价值的 SFT Badcase",
        "",
        "Task 95。",
        "",
        "核心能力缺口：",
        "",
        "```text",
        "Environment Schema Understanding",
        "-> Variant / item_id 语义",
        "-> Capability Boundary Judgment",
        "-> Multi-goal Completion",
        "-> Avoid Premature Escalation",
        "```",
        "",
        "### 5.3 当前最有价值的 Verifier Badcase",
        "",
        "Task 98。",
        "",
        "核心需要验证：",
        "",
        "```text",
        "User Authorized Scope",
        "-> Tool Actual Scope",
        "-> Tool Result / Final DB State",
        "-> Agent Final Claim",
        "```",
        "",
        "尤其需要检测：",
        "",
        "```text",
        "Tool refund = $1058.79",
        "Agent claim = $202.13",
        "```",
        "",
        "这种 Claim-State Inconsistency 是真实业务 Agent 的高风险问题。",
        "",
        "### 5.4 当前暂时不能用于训练的样本",
        "",
        "Task 59 和 Task 107。",
        "",
        "原因不是它们没有研究价值，恰恰相反：",
        "",
        "它们主要用于：",
        "",
        "- Benchmark Audit；",
        "- Evaluator Alignment；",
        "- Policy / Tool Semantics Verification；",
        "- Reward Label Noise Detection。",
        "",
        "在人工裁决完成前，不应直接作为 Agent 负样本。",
        "",
        "---",
        "",
        "## 6. 对后训练路线的影响",
        "",
        "当前证据只支持：",
        "",
        "```text",
        "Prompt Base",
        "-> Failure Audit",
        "-> Data Cleaning",
        "-> SFT / Verifier Dataset Construction",
        "```",
        "",
        "目前还不能证明必须直接进入 RL。",
        "",
        "RL 是否必要，需要后续通过：",
        "",
        "- 更大规模 Base failure distribution；",
        "- SFT 后残余失败；",
        "- Verifier 能否解决核心错误；",
        "- Preference / policy-compliance error 是否仍持续存在；",
        "",
        "再决定。",
        "",
        "因此当前结论是：",
        "",
        "> 先做好数据裁决、SFT 和 Verifier，再决定 RL。",
        "",
        "---",
        "",
        "## 7. 当前版本边界",
        "",
        "Failure Taxonomy v1 只基于当前冻结 20-task Trial-1 中的 4 个 Reward=0 样本。",
        "",
        "它不能直接代表整个 tau2-bench 的全局失败分布。",
        "",
        "后续随着：",
        "",
        "- 更多任务；",
        "- 多 Trial 稳定性实验；",
        "- 16 条 Reward=1 成功轨迹质量审计；",
        "- Policy 源码核验；",
        "",
        "Taxonomy 需要继续升级为 v2、v3。",
        "",
    ])

    return "\n".join(
        lines
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

    # Task 95 is currently the cleanest valid Agent failure.
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

    assert (
        task95[
            "real_agent_failure"
        ]
        is True
    )

    # Task 107 remains unresolved until Policy semantics are verified.
    task107 = next(
        item
        for item in FAILURES
        if item[
            "task_id"
        ] == "107"
    )

    assert (
        task107[
            "training_status"
        ]
        == "HOLD_UNTIL_POLICY_VERIFIED"
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
            "Failure Taxonomy and Training Data Eligibility Matrix",

        "baseline":
            BASELINE,

        "human_audit_policy": {
            "raw_reward_is_not_training_label":
                True,

            "raw_baseline_must_not_be_retroactively_modified":
                True,

            "benchmark_suspect_cases_require_separate_annotation":
                True,

            "mixed_badcases_require_segment_level_relabeling":
                True,

            "unresolved_policy_cases_must_not_enter_negative_training_pool":
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

    markdown = (
        build_markdown()
    )

    MD_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    MD_OUTPUT.write_text(
        markdown,
        encoding="utf-8",
    )

    # ASCII-only console output:
    # safe on Windows GBK terminals.
    print(
        "FAILURE_TAXONOMY_V1_CREATED"
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
        "TASK_107 = UNRESOLVED_POLICY_TOOL_SEMANTICS"
    )

    print(
        "FAILURE_TAXONOMY_V1_OK"
    )


if __name__ == "__main__":
    main()