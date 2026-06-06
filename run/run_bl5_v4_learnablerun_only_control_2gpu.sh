#!/usr/bin/env bash
set -euo pipefail

cd /data/zwf/code1/reborn_seed
source /data/zwf/Conda/miniconda3/etc/profile.d/conda.sh
conda activate reborn_seed

export CUDA_VISIBLE_DEVICES=0,1

echo "[BL5-v4-LearnableRun-only-control] Starting formal 2-GPU training..."
echo "Config: configs/bl5_v4_learnablerun_only_control.yaml"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date -Iseconds)"

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=2 \
  scripts/train_bl5.py \
  --config configs/bl5_v4_learnablerun_only_control.yaml

echo "End time: $(date -Iseconds)"
