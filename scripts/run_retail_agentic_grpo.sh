#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-environment-preflight}"
PROJECT_ROOT="${POLICYAGENT_PROJECT_ROOT:-/root/autodl-tmp/PolicyAgent-PostTrain}"
TAU2_ROOT="${POLICYAGENT_TAU2_ROOT:-/root/autodl-tmp/tau2-bench}"
RUN_DIR="${POLICYAGENT_AGENTIC_RL_RUN_DIR:-/root/autodl-tmp/policyagent-runs/20260811-retail-agentic-grpo-v1}"

cd "${PROJECT_ROOT}"
export POLICYAGENT_TAU2_ROOT="${TAU2_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${TAU2_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export POLICYAGENT_USER_MODEL="${POLICYAGENT_USER_MODEL:-deepseek/deepseek-v4-flash}"
if [[ -z "${POLICYAGENT_USER_LLM_ARGS_JSON:-}" ]]; then
  export POLICYAGENT_USER_LLM_ARGS_JSON='{"temperature":0.0}'
fi

case "${MODE}" in
  environment-preflight)
    python -m src.training.run_retail_agentic_grpo --environment-only-preflight
    ;;
  opening-smoke)
    : "${DEEPSEEK_API_KEY:?Set DEEPSEEK_API_KEY before generating a customer opening}"
    python -m src.rl.prepare_user_openings --limit 1
    ;;
  prepare-openings)
    : "${DEEPSEEK_API_KEY:?Set DEEPSEEK_API_KEY before generating customer openings}"
    python -m src.rl.prepare_user_openings
    ;;
  prepare-qwen3-diagnostic-openings)
    : "${DEEPSEEK_API_KEY:?Set DEEPSEEK_API_KEY before generating customer openings}"
    python -m src.rl.prepare_user_openings \
      --limit 8 \
      --output data/retail_agentic_rl_v1/qwen3_diagnostic8_initial_user_messages.jsonl \
      --manifest data/retail_agentic_rl_v1/qwen3_diagnostic8_initial_user_messages_manifest.json
    ;;
  gpu-preflight)
    python -m src.training.run_retail_agentic_grpo --preflight-only
    ;;
  gpu-preflight-sanity)
    python -m src.training.run_retail_agentic_grpo \
      --config configs/retail_agentic_grpo_sanity_v1.json \
      --preflight-only
    ;;
  train-sanity)
    : "${DEEPSEEK_API_KEY:?Set DEEPSEEK_API_KEY for the dynamic customer simulator}"
    python -m src.training.run_retail_agentic_grpo \
      --config configs/retail_agentic_grpo_sanity_v1.json \
      --output-dir "${RUN_DIR}-sanity"
    ;;
  train)
    : "${DEEPSEEK_API_KEY:?Set DEEPSEEK_API_KEY for the dynamic customer simulator}"
    python -m src.training.run_retail_agentic_grpo --output-dir "${RUN_DIR}"
    ;;
  qwen3-rollout-diagnostic)
    : "${DEEPSEEK_API_KEY:?Set DEEPSEEK_API_KEY for the dynamic customer simulator}"
    DIAGNOSTIC_DIR="${POLICYAGENT_QWEN3_DIAGNOSTIC_DIR:-/root/autodl-tmp/policyagent-runs/20260811-qwen3-4b-rollout-diagnostic-v1}"
    python -m src.training.run_retail_agentic_grpo \
      --config configs/retail_agentic_qwen3_4b_rollout_diagnostic_v1.json \
      --output-dir "${DIAGNOSTIC_DIR}"
    python -m src.analysis.analyze_agentic_rollout_diagnostic \
      --rollouts "${DIAGNOSTIC_DIR}/raw_rollouts.jsonl" \
      --expected-rollouts 32 \
      --expected-tasks 8 \
      --output "${DIAGNOSTIC_DIR}/diagnostic_report.json"
    ;;
  qwen3-tool-sft-rollout-diagnostic)
    : "${DEEPSEEK_API_KEY:?Set DEEPSEEK_API_KEY for the dynamic customer simulator}"
    TOOL_SFT_DIAGNOSTIC_DIR="${POLICYAGENT_QWEN3_TOOL_SFT_DIAGNOSTIC_DIR:-/root/autodl-tmp/policyagent-runs/20260811-qwen3-4b-tool-sft-rollout-diagnostic-v1}"
    python -m src.training.run_retail_agentic_grpo \
      --config configs/retail_agentic_qwen3_4b_tool_sft_rollout_diagnostic_v1.json \
      --output-dir "${TOOL_SFT_DIAGNOSTIC_DIR}"
    python -m src.analysis.analyze_agentic_rollout_diagnostic \
      --rollouts "${TOOL_SFT_DIAGNOSTIC_DIR}/raw_rollouts.jsonl" \
      --expected-rollouts 32 \
      --expected-tasks 8 \
      --output "${TOOL_SFT_DIAGNOSTIC_DIR}/diagnostic_report.json"
    ;;
  *)
    echo "Usage: $0 {environment-preflight|opening-smoke|prepare-openings|prepare-qwen3-diagnostic-openings|gpu-preflight-sanity|gpu-preflight|train-sanity|train|qwen3-rollout-diagnostic|qwen3-tool-sft-rollout-diagnostic}" >&2
    exit 2
    ;;
esac
