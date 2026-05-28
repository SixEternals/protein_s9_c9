#!/usr/bin/env bash
# Run BL5-v4-NoPAM-control on the formal BL5 split.
# Identical hyperparameters to BL5-v4-PAM, only difference is use_pam_encoder=false.
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="configs/bl5_v4_nopam_control.yaml"
GPUS=2

echo "[run_bl5_v4_nopam_control] Starting BL5-v4-NoPAM-control"
echo "[run_bl5_v4_nopam_control] Config: $CONFIG"
echo "[run_bl5_v4_nopam_control] GPUs: $GPUS"

conda run -n reborn_seed torchrun \
  --nproc_per_node="$GPUS" \
  scripts/train_bl5.py \
  --config "$CONFIG" \
  "$@"

echo "[run_bl5_v4_nopam_control] Finished"
