#!/usr/bin/env bash
# Auto-run all post-NoPAM analyses after training completes.
# Designed to run overnight; will wait for NoPAM summary.json then execute everything.
set -euo pipefail

cd "$(dirname "$0")/.."

NOPAM_DIR="results/bl5_v4_nopam_control"
POLL_INTERVAL=120

echo "[post-nopam] Waiting for NoPAM training to complete..."
echo "[post-nopam] Monitoring ${NOPAM_DIR}/summary.json"

while true; do
    if [[ -f "${NOPAM_DIR}/summary.json" && -f "${NOPAM_DIR}/test_predictions.csv" ]]; then
        status=$(python3 -c "import json,sys; d=json.load(open('${NOPAM_DIR}/summary.json')); print(d.get('status','unknown'))" 2>/dev/null || echo "unknown")
        if [[ "$status" == "completed" || "$status" == "completed_eval_only" ]]; then
            echo "[post-nopam] NoPAM completed. Waiting 60s for file flush..."
            sleep 60
            break
        fi
    fi
    echo "[post-nopam] Not yet complete. Next check in ${POLL_INTERVAL}s..."
    sleep ${POLL_INTERVAL}
done

echo "[post-nopam] Starting analysis pipeline..."

# 1. Stratified evaluation by PAM
conda run -n reborn_seed python scripts/eval_stratified_by_pam.py \
    --bl0b results/bl0b_on_bl5split/test_predictions.csv \
    --nopam results/bl5_v4_nopam_control/test_predictions.csv \
    --pam results/bl5_v4_pam/test_predictions.csv || true

# 2. Paired comparison
conda run -n reborn_seed python scripts/paired_comparison.py \
    --bl0b results/bl0b_on_bl5split/test_predictions.csv \
    --nopam results/bl5_v4_nopam_control/test_predictions.csv \
    --pam results/bl5_v4_pam/test_predictions.csv || true

# 3. Per-sgRNA / per-PAM analysis
conda run -n reborn_seed python scripts/per_sgrna_and_pam_analysis.py \
    --bl0b results/bl0b_on_bl5split/test_predictions.csv \
    --nopam results/bl5_v4_nopam_control/test_predictions.csv \
    --pam results/bl5_v4_pam/test_predictions.csv || true

# 4. kNN baseline
conda run -n reborn_seed python scripts/knn_baseline.py || true

# 5. Contribution decomposition report
conda run -n reborn_seed python -c "
import json, pandas as pd, sys
from pathlib import Path

bl0b = json.loads(Path('results/bl0b_on_bl5split/summary.json').read_text())
nopam = json.loads(Path('results/bl5_v4_nopam_control/summary.json').read_text())
pam  = json.loads(Path('results/bl5_v4_pam/summary.json').read_text())

rows = [
    {'model': 'BL0b-on-BL5split', 'AUROC': bl0b['test_metrics']['auroc'], 'AUPRC': bl0b['test_metrics']['auprc']},
    {'model': 'BL5-v4-NoPAM',    'AUROC': nopam['test_metrics']['auroc'], 'AUPRC': nopam['test_metrics']['auprc']},
    {'model': 'BL5-v4-PAM',      'AUROC': pam['test_metrics']['auroc'],  'AUPRC': pam['test_metrics']['auprc']},
]

df = pd.DataFrame(rows)
df['delta_from_BL0b'] = df['AUPRC'] - df.loc[0, 'AUPRC']
df['delta_from_NoPAM'] = df['AUPRC'] - df.loc[1, 'AUPRC']

print(df.to_string(index=False))

out = {
    'models': rows,
    'decomposition': {
        'NoPAM_minus_BL0b_AUPRC': rows[1]['AUPRC'] - rows[0]['AUPRC'],
        'PAM_minus_NoPAM_AUPRC':  rows[2]['AUPRC'] - rows[1]['AUPRC'],
        'PAM_minus_BL0b_AUPRC':   rows[2]['AUPRC'] - rows[0]['AUPRC'],
    }
}
Path('results/bl5_v4_contribution_decomposition.json').write_text(json.dumps(out, indent=2))
print('Wrote results/bl5_v4_contribution_decomposition.json')
" || true

echo "[post-nopam] All analyses finished."
echo "[post-nopam] Results available in results/"
