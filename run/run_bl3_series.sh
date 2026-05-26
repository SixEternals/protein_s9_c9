#!/bin/bash
set -e
ROOT=/data/zwf/code1/reborn_seed
PYTHON=/data/zwf/conda/envs/reborn_seed/bin/python

echo "=== BL3 Series Batch Run ==="
echo "Start: $(date)"

# 1. BL3-hard-A: Hard gradient
echo "[1/4] BL3-hard-A (Hard gradient 1x/2x)..."
CUDA_VISIBLE_DEVICES=0 $PYTHON $ROOT/scripts/train_bl3.py --config $ROOT/configs/bl3_hard_a.yaml
echo "BL3-hard-A done: $(date)"

# 2. BL3-hard-C: Learnable gradient
echo "[2/4] BL3-hard-C (Learnable gradient)..."
CUDA_VISIBLE_DEVICES=0 $PYTHON $ROOT/scripts/train_bl3.py --config $ROOT/configs/bl3_hard_c.yaml
echo "BL3-hard-C done: $(date)"

# 3. Ablation: Region only
echo "[3/4] Ablation: Region only..."
CUDA_VISIBLE_DEVICES=0 $PYTHON $ROOT/scripts/train_bl3.py --config $ROOT/configs/bl3_ablation_region.yaml
echo "Region-only done: $(date)"

# 4. Ablation: Run only
echo "[4/4] Ablation: Run only..."
CUDA_VISIBLE_DEVICES=0 $PYTHON $ROOT/scripts/train_bl3.py --config $ROOT/configs/bl3_ablation_run.yaml
echo "Run-only done: $(date)"

echo "=== All 4 experiments complete ==="
echo "End: $(date)"
