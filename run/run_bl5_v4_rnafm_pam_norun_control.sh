#!/usr/bin/env bash
# BL5-v4-RNAFM-PAM-noRun-control formal run
# RNA-FM CLS + PAM Encoder, no Run/LearnableRun
# 2-GPU DDP, Route A batch size

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

CONFIG="$PROJECT_ROOT/configs/bl5_v4_rnafm_pam_norun_control.yaml"
GPU_COUNT=2

echo "========================================"
echo "BL5-v4-RNAFM-PAM-noRun-control"
echo "Config: $CONFIG"
echo "GPUs:   $GPU_COUNT"
echo "Start:  $(date -Iseconds)"
echo "========================================"

cd "$PROJECT_ROOT"

/data/zwf/conda/envs/reborn_seed/bin/python3 -m torch.distributed.run \
    --nproc_per_node="$GPU_COUNT" \
    --master_port=29520 \
    scripts/train_bl5.py \
    --config "$CONFIG"

echo "========================================"
echo "Finished: $(date -Iseconds)"
echo "========================================"
