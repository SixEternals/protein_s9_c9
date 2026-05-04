#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/data/zwf/conda/envs/reborn_seed/bin/python}"
DATA_ROOT="${DATA_ROOT:-/data/zwf/project/zhb/data}"
MODEL="${MODEL:-conmismatch9}"
ENCODER="${ENCODER:-c9}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-30}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
PATIENCE="${PATIENCE:-8}"
BATCH_SIZE="${BATCH_SIZE:-512}"
CPU_THREADS="${CPU_THREADS:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
AMP="${AMP:-auto}"
HIDDEN_DIM="${HIDDEN_DIM:-96}"
DROPOUT="${DROPOUT:-0.20}"
ATTN_HEADS="${ATTN_HEADS:-4}"
ATTN_LAYERS="${ATTN_LAYERS:-2}"
SEED="${SEED:-42}"
RUN_BASE="${RUN_BASE:-runs}"
ARTIFACT_BASE="${ARTIFACT_BASE:-artifacts}"
DRY_RUN="${DRY_RUN:-0}"

if [[ $# -gt 0 ]]; then
  DATASETS=("$@")
else
  DATASETS=("Tasi" "CHANGE-seq" "GUIDE-seq")
fi

run_dataset() {
  local dataset_name="$1"
  local file_slug="${dataset_name//[^A-Za-z0-9_]/_}"
  local dir_slug="${file_slug,,}"
  local run_name="${MODEL}_${ENCODER}_${file_slug}"
  local dataset_file="$DATA_ROOT/$dataset_name/${dataset_name}_9bit.npz"
  local run_root="$RUN_BASE/full_${dir_slug}_conmismatch9"
  local artifact_root="$ARTIFACT_BASE/full_${dir_slug}_conmismatch9"
  local feature_cache_dir="$run_root/feature_cache"
  local config_path="$run_root/generated_configs/${run_name}.json"
  local summary_path="$run_root/train_summaries/${run_name}.json"
  local log_path="$run_root/train_logs/${run_name}.log"
  local weights_path="$artifact_root/${run_name}.pt"

  if [[ ! -f "$dataset_file" ]]; then
    echo "Missing dataset file: $dataset_file" >&2
    exit 1
  fi

  mkdir -p "$run_root" "$artifact_root" "$feature_cache_dir"

  echo
  echo "==> $dataset_name"
  echo "    data:    $dataset_file"
  echo "    config:  $config_path"
  echo "    weights: $weights_path"
  echo "    summary: $summary_path"
  echo "    log:     $log_path"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "    dry-run: RUN_ROOT=$run_root ARTIFACT_ROOT=$artifact_root FEATURE_CACHE_DIR=$feature_cache_dir PYTHON_BIN=$PYTHON_BIN bash scripts/train_all_datasets.sh --full --device $DEVICE $dataset_file"
    return
  fi

  RUN_ROOT="$run_root" \
  ARTIFACT_ROOT="$artifact_root" \
  FEATURE_CACHE_DIR="$feature_cache_dir" \
  PYTHON_BIN="$PYTHON_BIN" \
  CPU_THREADS="$CPU_THREADS" \
  NUM_WORKERS="$NUM_WORKERS" \
  AMP="$AMP" \
  bash scripts/train_all_datasets.sh --full --device "$DEVICE" "$dataset_file"
}

for dataset_name in "${DATASETS[@]}"; do
  run_dataset "$dataset_name"
done

echo
echo "All requested full baselines finished."
