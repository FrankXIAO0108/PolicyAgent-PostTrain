#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "用法: bash scripts/run_posttrain_stage.sh <config.json> <绝对运行目录> <base|sft|dpo|grpo>" >&2
  exit 2
fi

CONFIG="$1"
RUN_DIR="$2"
STAGE="$3"
if [[ "${RUN_DIR}" != /* ]]; then
  echo "运行目录必须是绝对路径。" >&2
  exit 2
fi
case "${STAGE}" in
  base|sft|dpo|grpo) ;;
  *) echo "未知阶段: ${STAGE}" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${POLICYAGENT_PYTHON:-/root/autodl-tmp/venvs/policyagent/bin/python}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/huggingface}"
export TOKENIZERS_PARALLELISM=false
cd "${REPO_ROOT}"

LOG_PATH="${RUN_DIR}.${STAGE}.console.log"
"${PYTHON_BIN}" -m src.training.run_posttrain_stage \
  --config "${CONFIG}" \
  --output-dir "${RUN_DIR}" \
  --stage "${STAGE}" 2>&1 | tee "${LOG_PATH}"

mkdir -p "${RUN_DIR}/logs"
mv "${LOG_PATH}" "${RUN_DIR}/logs/${STAGE}_console.log"

if [[ "${STAGE}" == "grpo" ]]; then
  "${PYTHON_BIN}" -m src.training.verify_posttrain_engineering_smoke \
    --run-dir "${RUN_DIR}" \
    --output "${RUN_DIR}/verification_report.json"
  "${PYTHON_BIN}" -m pip freeze > "${RUN_DIR}/pip_freeze.txt"
fi

echo "阶段完成: ${STAGE}"
