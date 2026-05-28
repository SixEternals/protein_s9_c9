#!/usr/bin/env bash
# Auto-start BL5-v4-NoPAM-control after BL0b-on-BL5split completes.
# Run this as a background watcher before going to sleep.
set -euo pipefail

cd "$(dirname "$0")/.."

BL0B_DIR="results/bl0b_on_bl5split"
NOPAM_CONFIG="configs/bl5_v4_nopam_control.yaml"
GPUS=2
MASTER_PORT=29500

poll_interval=60

echo "[autostart] Monitoring BL0b-on-BL5split at ${BL0B_DIR}"
echo "[autostart] Will auto-start NoPAM when BL0b completes"

# Wait for BL0b to finish by watching for summary.json + best.pt + completed status
while true; do
    if [[ -f "${BL0B_DIR}/summary.json" && -f "${BL0B_DIR}/checkpoints/best.pt" ]]; then
        status=$(python3 -c "import json,sys; print(json.load(open('${BL0B_DIR}/summary.json'))['status'])" 2>/dev/null || echo "unknown")
        if [[ "$status" == "completed" ]]; then
            echo "[autostart] BL0b completed successfully. Waiting 60s for DDP port cleanup..."
            sleep 60
            break
        else
            echo "[autostart] BL0b summary exists but status='$status'. Continuing to monitor..."
        fi
    fi
    echo "[autostart] BL0b not yet complete. Next check in ${poll_interval}s..."
    sleep ${poll_interval}
done

# Extra safety: ensure the port is free before launching NoPAM
while ss -tlnp 2>/dev/null | grep -q ":${MASTER_PORT} "; do
    echo "[autostart] Port ${MASTER_PORT} still in use. Waiting 30s..."
    sleep 30
done

echo "[autostart] Starting BL5-v4-NoPAM-control"
conda run -n reborn_seed torchrun \
    --nproc_per_node="${GPUS}" \
    --master_port="${MASTER_PORT}" \
    scripts/train_bl5.py \
    --config "${NOPAM_CONFIG}"

echo "[autostart] NoPAM training finished."
