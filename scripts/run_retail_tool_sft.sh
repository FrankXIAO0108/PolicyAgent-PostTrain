#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${POLICYAGENT_PROJECT_ROOT:-/root/autodl-tmp/PolicyAgent-PostTrain}"
TAU2_ROOT="${POLICYAGENT_TAU2_ROOT:-/root/autodl-tmp/tau2-bench}"
PYTHON_BIN="${POLICYAGENT_PYTHON:-/root/autodl-tmp/venvs/policyagent/bin/python}"
RUN_DIR="${POLICYAGENT_TOOL_SFT_RUN_DIR:-/root/autodl-tmp/policyagent-runs/20260811-qwen3-4b-tool-sft-v1}"
MODE="${1:-preflight}"

cd "${PROJECT_ROOT}"
export POLICYAGENT_TAU2_ROOT="${TAU2_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${TAU2_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false

case "${MODE}" in
  preflight)
    "${PYTHON_BIN}" -m src.training.run_retail_tool_sft --preflight-only
    ;;
  train)
    "${PYTHON_BIN}" -m src.training.run_retail_tool_sft --output-dir "${RUN_DIR}"
    ;;
  *)
    echo "用法: bash scripts/run_retail_tool_sft.sh {preflight|train}" >&2
    exit 2
    ;;
esac
