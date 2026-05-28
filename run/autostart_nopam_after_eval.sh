#!/usr/bin/env bash
# Auto-start NoPAM after eval_bl0b_best.py finishes.
set -euo pipefail

cd "$(dirname "$0")/.."

BL0B_DIR="results/bl0b_on_bl5split"
NOPAM_CONFIG="configs/bl5_v4_nopam_control.yaml"
GPUS=2
MASTER_PORT=29500

poll_interval=60

echo "[autostart-v2] Monitoring BL0b eval completion at ${BL0B_DIR}"

while true; do
    if [[ -f "${BL0B_DIR}/summary.json" && -f "${BL0B_DIR}/test_predictions.csv" ]]; then
        status=$(python3 -c "import json,sys; print(json.load(open('${BL0B_DIR}/summary.json'))['status'])" 2>/dev/null || echo "unknown")
        if [[ "$status" == "completed_eval_only" ]]; then
            echo "[autostart-v2] BL0b eval completed. Waiting 30s for GPU memory cleanup..."
            sleep 30
            break
        fi
    fi
    echo "[autostart-v2] BL0b eval not yet complete. Next check in ${poll_interval}s..."
    sleep ${poll_interval}
done

while ss -tlnp 2>/dev/null | grep -q ":${MASTER_PORT} "; do
    echo "[autostart-v2] Port ${MASTER_PORT} still in use. Waiting 30s..."
    sleep 30
done

echo "[autostart-v2] Starting BL5-v4-NoPAM-control"
conda run -n reborn_seed torchrun \
    --nproc_per_node="${GPUS}" \
    --master_port="${MASTER_PORT}" \
    scripts/train_bl5.py \
    --config "${NOPAM_CONFIG}"

echo "[autostart-v2] NoPAM training finished."
