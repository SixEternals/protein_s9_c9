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
SEEDS="${SEEDS:-42 43 44}"
MODES="${MODES:-full no_mi no_run_attn no_fusion only_cnn}"
RUN_BASE="${RUN_BASE:-runs}"
ARTIFACT_BASE="${ARTIFACT_BASE:-artifacts}"
DRY_RUN="${DRY_RUN:-0}"

dataset_safe_name="${DATASET//[^A-Za-z0-9_]/_}"
dataset_slug_lower="${dataset_safe_name,,}"
dataset_file="$DATA_ROOT/$DATASET/${DATASET}_9bit.npz"
run_name="conmismatch9_c9_${dataset_safe_name}"

if [[ ! -f "$dataset_file" ]]; then
  echo "Missing dataset file: $dataset_file" >&2
  exit 1
fi

read -r -a seed_list <<< "$SEEDS"
read -r -a mode_list <<< "$MODES"

if [[ ${#seed_list[@]} -eq 0 ]]; then
  echo "No seeds specified." >&2
  exit 1
fi

if [[ ${#mode_list[@]} -eq 0 ]]; then
  echo "No modes specified." >&2
  exit 1
fi

ablation_root="$RUN_BASE/ablation_${dataset_slug_lower}_conmismatch9"
artifact_root_base="$ARTIFACT_BASE/ablation_${dataset_slug_lower}_conmismatch9"

echo "ConMismatch9 multiseed ablation plan"
echo "  dataset: $DATASET"
echo "  data:    $dataset_file"
echo "  seeds:   ${seed_list[*]}"
echo "  modes:   ${mode_list[*]}"
echo

for seed in "${seed_list[@]}"; do
  seed_root="$ablation_root/seed_${seed}"
  artifact_seed_root="$artifact_root_base/seed_${seed}"
  feature_cache_dir="$seed_root/feature_cache"

  echo "==> seed $seed"
  for mode in "${mode_list[@]}"; do
    run_root="$seed_root/$mode"
    artifact_root="$artifact_seed_root/$mode"
    config_path="$run_root/generated_configs/${run_name}.json"
    summary_path="$run_root/train_summaries/${run_name}.json"
    log_path="$run_root/train_logs/${run_name}.log"
    weights_path="$artifact_root/${run_name}.pt"

    echo "    mode:    $mode"
    echo "    run_root:      $run_root"
    echo "    artifact_root:  $artifact_root"
    echo "    summary:        $summary_path"
    echo "    log:            $log_path"

    if [[ "$DRY_RUN" == "1" ]]; then
      echo "    dry-run: RUN_ROOT=$run_root ARTIFACT_ROOT=$artifact_root FEATURE_CACHE_DIR=$feature_cache_dir SEED=$seed ABLATION_MODE=$mode bash scripts/train_all_datasets.sh --full --device $DEVICE $dataset_file"
      echo
      continue
    fi

    mkdir -p "$run_root" "$artifact_root" "$feature_cache_dir"

    RUN_ROOT="$run_root" \
    ARTIFACT_ROOT="$artifact_root" \
    FEATURE_CACHE_DIR="$feature_cache_dir" \
    PYTHON_BIN="$PYTHON_BIN" \
    MODEL=conmismatch9 \
    ENCODER=c9 \
    SEED="$seed" \
    EPOCHS="$EPOCHS" \
    PATIENCE="$PATIENCE" \
    BATCH_SIZE="$BATCH_SIZE" \
    CPU_THREADS="$CPU_THREADS" \
    NUM_WORKERS="$NUM_WORKERS" \
    AMP="$AMP" \
    ABLATION_MODE="$mode" \
    bash scripts/train_all_datasets.sh --full --device "$DEVICE" "$dataset_file"

    echo "    weights:        $weights_path"
    echo
  done
done

echo "Multiseed ablation jobs finished."
