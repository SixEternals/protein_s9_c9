#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

DATASET="${1:-K562}"
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
RUN_BASE="${RUN_BASE:-runs}"
ARTIFACT_BASE="${ARTIFACT_BASE:-artifacts}"
REUSE_INCEPTION_ONLY="${REUSE_INCEPTION_ONLY:-1}"
RUN_BASELINE_FULL="${RUN_BASELINE_FULL:-0}"
WARMSTART_FREEZE_EPOCHS="${WARMSTART_FREEZE_EPOCHS:-3}"
DRY_RUN="${DRY_RUN:-0}"

dataset_safe_name="${DATASET//[^A-Za-z0-9_]/_}"
dataset_slug_lower="${dataset_safe_name,,}"
dataset_file="$DATA_ROOT/$DATASET/${DATASET}_9bit.npz"
run_name="deepfocus_r9_${dataset_safe_name}"

if [[ ! -f "$dataset_file" ]]; then
  echo "Missing dataset file: $dataset_file" >&2
  exit 1
fi

read -r -a seed_list <<< "$SEEDS"
if [[ ${#seed_list[@]} -eq 0 ]]; then
  echo "No seeds specified." >&2
  exit 1
fi

upgrade_root="$RUN_BASE/full_upgrade_${dataset_slug_lower}_deepfocus"
artifact_root_base="$ARTIFACT_BASE/full_upgrade_${dataset_slug_lower}_deepfocus"

echo "DeepFocus full-upgrade multiseed plan"
echo "  dataset:         $DATASET"
echo "  data:            $dataset_file"
echo "  seeds:           ${seed_list[*]}"
echo "  reuse_inception: $REUSE_INCEPTION_ONLY"
echo "  freeze_epochs:   $WARMSTART_FREEZE_EPOCHS"
echo

run_mode() {
  local seed="$1"
  local mode="$2"
  local warmstart_path="$3"

  local seed_root="$upgrade_root/seed_${seed}"
  local artifact_seed_root="$artifact_root_base/seed_${seed}"
  local run_root="$seed_root/$mode"
  local artifact_root="$artifact_seed_root/$mode"
  local feature_cache_dir="$seed_root/feature_cache"
  local summary_path="$run_root/train_summaries/${run_name}.json"
  local log_path="$run_root/train_logs/${run_name}.log"
  local weights_path="$artifact_root/${run_name}.pt"

  echo "    mode:           $mode"
  echo "    run_root:       $run_root"
  echo "    artifact_root:  $artifact_root"
  echo "    summary:        $summary_path"
  echo "    log:            $log_path"
  if [[ -n "$warmstart_path" ]]; then
    echo "    warmstart:      $warmstart_path"
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "    dry-run: RUN_ROOT=$run_root ARTIFACT_ROOT=$artifact_root FEATURE_CACHE_DIR=$feature_cache_dir SEED=$seed ABLATION_MODE=$mode WARMSTART_WEIGHTS_PATH=$warmstart_path bash scripts/train_all_datasets.sh --full --device $DEVICE $dataset_file"
    echo
    return
  fi

  mkdir -p "$run_root" "$artifact_root" "$feature_cache_dir"

  RUN_ROOT="$run_root" \
  ARTIFACT_ROOT="$artifact_root" \
  FEATURE_CACHE_DIR="$feature_cache_dir" \
  PYTHON_BIN="$PYTHON_BIN" \
  MODEL=deepfocus \
  ENCODER=r9 \
  SEED="$seed" \
  EPOCHS="$EPOCHS" \
  PATIENCE="$PATIENCE" \
  BATCH_SIZE="$BATCH_SIZE" \
  CPU_THREADS="$CPU_THREADS" \
  NUM_WORKERS="$NUM_WORKERS" \
  AMP="$AMP" \
  ABLATION_MODE="$mode" \
  WARMSTART_WEIGHTS_PATH="$warmstart_path" \
  WARMSTART_FREEZE_EPOCHS="$WARMSTART_FREEZE_EPOCHS" \
  bash scripts/train_all_datasets.sh --full --device "$DEVICE" "$dataset_file"

  echo "    weights:        $weights_path"
  echo
}

for seed in "${seed_list[@]}"; do
  echo "==> seed $seed"
  current_inception_weights="$artifact_root_base/seed_${seed}/inception_only/${run_name}.pt"

  if [[ "$REUSE_INCEPTION_ONLY" == "1" && -f "$current_inception_weights" ]]; then
    inception_weights="$current_inception_weights"
    echo "    mode:           inception_only"
    echo "    reuse weights:  $inception_weights"
    echo
  else
    run_mode "$seed" "inception_only" ""
    inception_weights="$current_inception_weights"
  fi

  if [[ "$DRY_RUN" != "1" && ! -f "$inception_weights" ]]; then
    echo "Missing inception_only checkpoint for warm-start: $inception_weights" >&2
    exit 1
  fi

  run_mode "$seed" "full" "$inception_weights"

  if [[ "$RUN_BASELINE_FULL" == "1" ]]; then
    run_mode "$seed" "full" ""
  fi
done

echo "DeepFocus full-upgrade multiseed jobs finished."
