#!/usr/bin/env bash
set -euo pipefail

# BL0a formal training launcher.
# Default: full frozen RNA-FM + CCLMoff-style MLP training with sgRNA_type-safe split.
#
# Usage:
#   bash run/run_bl0a_formal_frozen.sh
#
# Optional overrides:
#   CUDA_VISIBLE_DEVICES=1 bash run/run_bl0a_formal_frozen.sh
#   CONFIG_PATH=configs/bl0a_formal_dryrun.yaml bash run/run_bl0a_formal_frozen.sh

ROOT_DIR="/data/zwf/code1/reborn_seed"
PYTHON_BIN="/data/zwf/conda/envs/reborn_seed/bin/python"
CONFIG_PATH="${CONFIG_PATH:-configs/bl0a_formal_frozen.yaml}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1

cd "${ROOT_DIR}"

mkdir -p runs/job_logs
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="runs/job_logs/bl0a_formal_frozen_${STAMP}.log"

echo "[BL0a] root=${ROOT_DIR}"
echo "[BL0a] python=${PYTHON_BIN}"
echo "[BL0a] config=${CONFIG_PATH}"
echo "[BL0a] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[BL0a] log=${LOG_PATH}"

"${PYTHON_BIN}" - <<'PY'
import sys
import torch
print("[BL0a] cuda_available=", torch.cuda.is_available())
print("[BL0a] device_count=", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"[BL0a] device_{i}=", torch.cuda.get_device_name(i))
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    print("[BL0a] ERROR: CUDA is not available. Refusing to run formal RNA-FM training on CPU.", file=sys.stderr)
    sys.exit(1)
PY

"${PYTHON_BIN}" scripts/train_bl0a_formal.py --config "${CONFIG_PATH}" 2>&1 | tee "${LOG_PATH}"

echo "[BL0a] done"
echo "[BL0a] log=${LOG_PATH}"
