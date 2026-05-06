#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/data/zwf/conda/envs/reborn_seed/bin/python}"
SEEDS="${SEEDS:-42 43 44}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-8}"
BATCH_SIZE="${BATCH_SIZE:-512}"
CPU_THREADS="${CPU_THREADS:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
AMP="${AMP:-auto}"
WARMSTART_FREEZE_EPOCHS="${WARMSTART_FREEZE_EPOCHS:-3}"

DATASETS=("$@")
if [[ ${#DATASETS[@]} -eq 0 ]]; then
  DATASETS=(K562 SITE Tasi "CHANGE-seq" "GUIDE-seq")
fi

for dataset in "${DATASETS[@]}"; do
  echo "========================================"
  echo "Dataset: $dataset"
  echo "========================================"
  SEEDS="$SEEDS" \
    DEVICE="$DEVICE" \
    EPOCHS="$EPOCHS" \
    PATIENCE="$PATIENCE" \
    BATCH_SIZE="$BATCH_SIZE" \
    CPU_THREADS="$CPU_THREADS" \
    NUM_WORKERS="$NUM_WORKERS" \
    AMP="$AMP" \
    WARMSTART_FREEZE_EPOCHS="$WARMSTART_FREEZE_EPOCHS" \
    bash scripts/run_deepfocus_multiseed.sh "$dataset"
done

echo "All DeepFocus experiments finished."
