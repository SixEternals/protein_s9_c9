#!/usr/bin/env bash
# Auto-start NoPAM holdout training when PAM holdout training completes.
# Designed to be called by cron every 30 minutes.

set -euo pipefail

PROJECT_ROOT="/data/zwf/code1/reborn_seed"
cd "$PROJECT_ROOT"

PAM_OUTDIR="results/bl5_v4_pam_holdout_agg"
NLOG="/tmp/auto_nopam_holdout.log"

# Check if PAM training has produced best.pt (completed)
if [ ! -f "$PAM_OUTDIR/checkpoints/best.pt" ]; then
    echo "$(date): PAM holdout not yet complete (no best.pt)" >> "$NLOG"
    exit 0
fi

# Check if NoPAM training is already running
if tmux has-session -t bl5_nopam_holdout_agg 2>/dev/null; then
    echo "$(date): NoPAM holdout already running" >> "$NLOG"
    exit 0
fi

# Check if NoPAM training already completed
if [ -f "results/bl5_v4_nopam_holdout_agg/checkpoints/best.pt" ]; then
    echo "$(date): NoPAM holdout already completed" >> "$NLOG"
    exit 0
fi

echo "$(date): PAM holdout complete. Starting NoPAM holdout..." >> "$NLOG"
bash run/run_bl5_v4_holdout_agg.sh nopam >> "$NLOG" 2>&1
