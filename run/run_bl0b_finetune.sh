#!/usr/bin/env bash
set -euo pipefail

# BL0b full fine-tune launcher.
# Default: 2-GPU DDP, full CCLMoff CSV, RNA-FM unfrozen.
#
# Usage:
#   bash run/run_bl0b_finetune.sh
#
# Optional overrides:
#   CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 bash run/run_bl0b_finetune.sh
#   CONFIG_PATH=configs/bl0b_finetune_dryrun.yaml NPROC_PER_NODE=1 CUDA_VISIBLE_DEVICES=0 bash run/run_bl0b_finetune.sh
#   CONFIG_PATH=configs/bl0b_finetune.yaml NPROC_PER_NODE=1 CUDA_VISIBLE_DEVICES=0 bash run/run_bl0b_finetune.sh

ROOT_DIR="/data/zwf/code1/reborn_seed"
PYTHON_BIN="/data/zwf/conda/envs/reborn_seed/bin/python"
CONFIG_PATH="${CONFIG_PATH:-configs/bl0b_finetune.yaml}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

cd "${ROOT_DIR}"

mkdir -p runs/job_logs
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="runs/job_logs/bl0b_finetune_${STAMP}.log"

echo "[BL0b] root=${ROOT_DIR}"
echo "[BL0b] python=${PYTHON_BIN}"
echo "[BL0b] config=${CONFIG_PATH}"
echo "[BL0b] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[BL0b] NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "[BL0b] log=${LOG_PATH}"

"${PYTHON_BIN}" - "${NPROC_PER_NODE}" <<'PY'
import sys
import torch

required = int(sys.argv[1])
print("[BL0b] cuda_available=", torch.cuda.is_available())
print("[BL0b] device_count=", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"[BL0b] device_{i}=", torch.cuda.get_device_name(i))
if not torch.cuda.is_available() or torch.cuda.device_count() < required:
    print(
        f"[BL0b] ERROR: need at least {required} visible CUDA device(s).",
        file=sys.stderr,
    )
    sys.exit(1)
PY

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  "${PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node="${NPROC_PER_NODE}" \
    scripts/train_bl0a_formal.py \
    --config "${CONFIG_PATH}" 2>&1 | tee "${LOG_PATH}"
else
  "${PYTHON_BIN}" scripts/train_bl0a_formal.py \
    --config "${CONFIG_PATH}" 2>&1 | tee "${LOG_PATH}"
fi

echo "[BL0b] done"
echo "[BL0b] log=${LOG_PATH}"
