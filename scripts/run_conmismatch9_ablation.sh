#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

DATASET="${1:-GUIDE-seq}"
PYTHON_BIN="${PYTHON_BIN:-/data/zwf/conda/envs/reborn_seed/bin/python}"
DATA_ROOT="${DATA_ROOT:-/data/zwf/project/zhb/data}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-8}"
BATCH_SIZE="${BATCH_SIZE:-512}"
CPU_THREADS="${CPU_THREADS:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
AMP="${AMP:-auto}"
RUN_BASE="${RUN_BASE:-runs}"
ARTIFACT_BASE="${ARTIFACT_BASE:-artifacts}"
MODES="${MODES:-full no_mi no_run_attn no_fusion only_cnn no_mi_no_run_attn}"
DRY_RUN="${DRY_RUN:-0}"

dataset_slug="${DATASET//[^A-Za-z0-9_]/_}"
dataset_slug_lower="${dataset_slug,,}"
dataset_file="$DATA_ROOT/$DATASET/${DATASET}_9bit.npz"

if [[ ! -f "$dataset_file" ]]; then
  echo "Missing dataset file: $dataset_file" >&2
  exit 1
fi

echo "ConMismatch9 ablation plan"
echo "  dataset: $DATASET"
echo "  data:    $dataset_file"
echo "  modes:   $MODES"
echo

for mode in $MODES; do
  run_root="$RUN_BASE/ablation_${dataset_slug_lower}_conmismatch9/$mode"
  artifact_root="$ARTIFACT_BASE/ablation_${dataset_slug_lower}_conmismatch9/$mode"
  feature_cache_dir="$RUN_BASE/ablation_${dataset_slug_lower}_conmismatch9/feature_cache"

  echo "==> $DATASET / $mode"
  echo "    run_root:      $run_root"
  echo "    artifact_root: $artifact_root"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "    dry-run: RUN_ROOT=$run_root ARTIFACT_ROOT=$artifact_root FEATURE_CACHE_DIR=$feature_cache_dir ABLATION_MODE=$mode bash scripts/train_all_datasets.sh --full --device $DEVICE $dataset_file"
    echo
    continue
  fi

  RUN_ROOT="$run_root" \
  ARTIFACT_ROOT="$artifact_root" \
  FEATURE_CACHE_DIR="$feature_cache_dir" \
  PYTHON_BIN="$PYTHON_BIN" \
  MODEL=conmismatch9 \
  ENCODER=c9 \
  EPOCHS="$EPOCHS" \
  PATIENCE="$PATIENCE" \
  BATCH_SIZE="$BATCH_SIZE" \
  CPU_THREADS="$CPU_THREADS" \
  NUM_WORKERS="$NUM_WORKERS" \
  AMP="$AMP" \
  ABLATION_MODE="$mode" \
  bash scripts/train_all_datasets.sh --full --device "$DEVICE" "$dataset_file"

  echo
done

echo "Ablation jobs finished."
