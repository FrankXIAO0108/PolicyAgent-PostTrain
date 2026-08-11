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
  gpu-preflight)
    python -m src.training.run_retail_agentic_grpo --preflight-only
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
  *)
    echo "Usage: $0 {environment-preflight|opening-smoke|prepare-openings|gpu-preflight|train-sanity|train}" >&2
    exit 2
    ;;
esac
