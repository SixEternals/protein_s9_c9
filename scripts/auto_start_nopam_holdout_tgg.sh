#!/usr/bin/env bash
# Auto-start NoPAM holdout TGG training when PAM holdout TGG completes.

set -euo pipefail

PROJECT_ROOT="/data/zwf/code1/reborn_seed"
cd "$PROJECT_ROOT"

PAM_OUTDIR="results/bl5_v4_pam_holdout_tgg"
NLOG="/tmp/auto_nopam_holdout_tgg.log"

# Check if PAM training has produced best.pt (completed)
if [ ! -f "$PAM_OUTDIR/checkpoints/best.pt" ]; then
    echo "$(date): PAM holdout TGG not yet complete (no best.pt)" >> "$NLOG"
    exit 0
fi

# Check if NoPAM training is already running
if tmux has-session -t bl5_nopam_holdout_tgg 2>/dev/null; then
    echo "$(date): NoPAM holdout TGG already running" >> "$NLOG"
    exit 0
fi

# Check if NoPAM training already completed
if [ -f "results/bl5_v4_nopam_holdout_tgg/checkpoints/best.pt" ]; then
    echo "$(date): NoPAM holdout TGG already completed" >> "$NLOG"
    exit 0
fi

echo "$(date): PAM holdout TGG complete. Starting NoPAM holdout TGG..." >> "$NLOG"
bash run/run_bl5_v4_holdout_tgg.sh nopam >> "$NLOG" 2>&1
