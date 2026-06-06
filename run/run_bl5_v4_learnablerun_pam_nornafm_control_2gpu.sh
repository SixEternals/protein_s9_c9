#!/usr/bin/env bash
# BL5-v4-LearnableRun-PAM-noRNAFM-control formal run
# LearnableRunEncoder + PAM Encoder, no RNA-FM
# 2-GPU DDP

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

CONFIG="$PROJECT_ROOT/configs/bl5_v4_learnablerun_pam_nornafm_control.yaml"
GPU_COUNT=2

echo "========================================"
echo "BL5-v4-LearnableRun-PAM-noRNAFM-control"
echo "Config: $CONFIG"
echo "GPUs:   $GPU_COUNT"
echo "Start:  $(date -Iseconds)"
echo "========================================"

cd "$PROJECT_ROOT"

/data/zwf/conda/envs/reborn_seed/bin/python3 -m torch.distributed.run \
    --nproc_per_node="$GPU_COUNT" \
    --master_port=29521 \
    scripts/train_bl5.py \
    --config "$CONFIG"

echo "========================================"
echo "Finished: $(date -Iseconds)"
echo "========================================"
