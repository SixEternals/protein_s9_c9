#!/usr/bin/env bash
# Auto-start NoPAM GAG holdout training once PAM best.pt is available.
set -euo pipefail

PROJECT_ROOT="/data/zwf/code1/reborn_seed"
cd "$PROJECT_ROOT"

BEST_PT="results/bl5_v4_pam_holdout_gag/checkpoints/best.pt"
TMUX_PAM="bl5_pam_holdout_gag"
TMUX_NOPAM="bl5_nopam_holdout_gag"

# Check if PAM best.pt exists
if [ ! -f "$BEST_PT" ]; then
    echo "[INFO] PAM best.pt not yet ready: $BEST_PT"
    exit 0
fi

# Check if NoPAM already running or done
if tmux has-session -t "$TMUX_NOPAM" 2>/dev/null; then
    echo "[INFO] NoPAM tmux already running: $TMUX_NOPAM"
    exit 0
fi

if [ -f "results/bl5_v4_nopam_holdout_gag/summary.json" ]; then
    echo "[INFO] NoPAM already completed"
    exit 0
fi

echo "[INFO] PAM best.pt found. Starting NoPAM GAG holdout..."
bash run/run_bl5_v4_holdout_gag.sh nopam
echo "[INFO] NoPAM started successfully."
