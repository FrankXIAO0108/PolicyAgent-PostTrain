#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "用法: bash scripts/run_posttrain_engineering_smoke.sh /绝对路径/本次运行目录" >&2
  exit 2
fi

RUN_DIR="$1"
if [[ "${RUN_DIR}" != /* ]]; then
  echo "运行目录必须是绝对路径，且建议位于仓库外部的数据盘。" >&2
  exit 2
fi
if [[ -e "${RUN_DIR}" ]]; then
  echo "拒绝覆盖已存在的运行目录: ${RUN_DIR}" >&2
  exit 2
fi
SANITY_DIR="${RUN_DIR}-api-sanity"
if [[ -e "${SANITY_DIR}" ]]; then
  echo "拒绝覆盖已存在的 API sanity 目录: ${SANITY_DIR}" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

python -m pip install -r requirements-posttrain-smoke.txt
python -m src.training.run_posttrain_engineering_smoke \
  --config configs/posttrain_engineering_smoke_v1.json \
  --output-dir "${RUN_DIR}" \
  --preflight-only

nvidia-smi
python -m src.training.run_posttrain_engineering_smoke \
  --config configs/posttrain_engineering_api_sanity_v1.json \
  --output-dir "${SANITY_DIR}" 2>&1 | tee "${SANITY_DIR}.console.log"
mv "${SANITY_DIR}.console.log" "${SANITY_DIR}/console.log"
python -m src.training.verify_posttrain_engineering_smoke \
  --run-dir "${SANITY_DIR}" \
  --output "${SANITY_DIR}/verification_report.json"

python -m src.training.run_posttrain_engineering_smoke \
  --config configs/posttrain_engineering_smoke_v1.json \
  --output-dir "${RUN_DIR}" 2>&1 | tee "${RUN_DIR}.console.log"
mv "${RUN_DIR}.console.log" "${RUN_DIR}/console.log"

python -m src.training.verify_posttrain_engineering_smoke \
  --run-dir "${RUN_DIR}" \
  --output "${RUN_DIR}/verification_report.json"

python -m pip freeze > "${RUN_DIR}/pip_freeze.txt"
tar -C "$(dirname "${RUN_DIR}")" -czf "${RUN_DIR}.evidence.tar.gz" \
  "$(basename "${RUN_DIR}")/run_manifest.json" \
  "$(basename "${RUN_DIR}")/verification_report.json" \
  "$(basename "${RUN_DIR}")/environment.json" \
  "$(basename "${RUN_DIR}")/evaluation_base.json" \
  "$(basename "${RUN_DIR}")/evaluation_sft.json" \
  "$(basename "${RUN_DIR}")/evaluation_dpo.json" \
  "$(basename "${RUN_DIR}")/evaluation_grpo.json" \
  "$(basename "${RUN_DIR}")/pip_freeze.txt"

echo "训练与验收完成。"
echo "接口冒烟目录: ${SANITY_DIR}"
echo "完整 checkpoint: ${RUN_DIR}"
echo "轻量证据包: ${RUN_DIR}.evidence.tar.gz"
