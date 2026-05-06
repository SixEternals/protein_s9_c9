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
RUN_BASE="${RUN_BASE:-runs}"
ARTIFACT_BASE="${ARTIFACT_BASE:-artifacts}"
ONLY_CNN_ARTIFACT_BASE="${ONLY_CNN_ARTIFACT_BASE:-}"
REUSE_ONLY_CNN="${REUSE_ONLY_CNN:-1}"
RUN_LEGACY_FULL="${RUN_LEGACY_FULL:-0}"
ABLATION_MODE="${ABLATION_MODE:-full}"
FULL_WARMSTART_FREEZE_EPOCHS="${FULL_WARMSTART_FREEZE_EPOCHS:-3}"
FULL_DISTILL_ALPHA="${FULL_DISTILL_ALPHA:-0.10}"
FULL_DISTILL_TEMPERATURE="${FULL_DISTILL_TEMPERATURE:-2.0}"
FULL_AUX_INIT_SCALE="${FULL_AUX_INIT_SCALE:-0.0}"
FULL_AUX_MAX_SCALE="${FULL_AUX_MAX_SCALE:-0.50}"
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
if [[ ${#seed_list[@]} -eq 0 ]]; then
  echo "No seeds specified." >&2
  exit 1
fi

upgrade_root="$RUN_BASE/full_upgrade_${dataset_slug_lower}_conmismatch9"
artifact_root_base="$ARTIFACT_BASE/full_upgrade_${dataset_slug_lower}_conmismatch9"
only_cnn_artifact_root_base="$artifact_root_base"
if [[ -n "$ONLY_CNN_ARTIFACT_BASE" ]]; then
  only_cnn_artifact_root_base="$ONLY_CNN_ARTIFACT_BASE/full_upgrade_${dataset_slug_lower}_conmismatch9"
fi

echo "ConMismatch9 full-upgrade multiseed plan"
echo "  dataset:         $DATASET"
echo "  data:            $dataset_file"
echo "  seeds:           ${seed_list[*]}"
echo "  reuse_only_cnn:  $REUSE_ONLY_CNN"
echo "  only_cnn_base:   $only_cnn_artifact_root_base"
echo "  run_legacy_full: $RUN_LEGACY_FULL"
echo "  full freeze:     $FULL_WARMSTART_FREEZE_EPOCHS epoch(s)"
echo "  full distill:    alpha=$FULL_DISTILL_ALPHA temperature=$FULL_DISTILL_TEMPERATURE"
echo

run_mode() {
  local seed="$1"
  local mode="$2"
  local warmstart_path="$3"
  local teacher_path="$4"
  local freeze_epochs="$5"
  local distill_alpha="$6"
  local distill_temperature="$7"

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
  if [[ -n "$teacher_path" ]]; then
    echo "    teacher:        $teacher_path"
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "    dry-run: RUN_ROOT=$run_root ARTIFACT_ROOT=$artifact_root FEATURE_CACHE_DIR=$feature_cache_dir SEED=$seed ABLATION_MODE=$mode WARMSTART_WEIGHTS_PATH=$warmstart_path TEACHER_WEIGHTS_PATH=$teacher_path DISTILL_ALPHA=$distill_alpha bash scripts/train_all_datasets.sh --full --device $DEVICE $dataset_file"
    echo
    return
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
  WARMSTART_WEIGHTS_PATH="$warmstart_path" \
  TEACHER_WEIGHTS_PATH="$teacher_path" \
  WARMSTART_FREEZE_EPOCHS="$freeze_epochs" \
  DISTILL_ALPHA="$distill_alpha" \
  DISTILL_TEMPERATURE="$distill_temperature" \
  AUX_INIT_SCALE="$FULL_AUX_INIT_SCALE" \
  AUX_MAX_SCALE="$FULL_AUX_MAX_SCALE" \
  bash scripts/train_all_datasets.sh --full --device "$DEVICE" "$dataset_file"

  echo "    weights:        $weights_path"
  echo
}

for seed in "${seed_list[@]}"; do
  echo "==> seed $seed"
  current_only_cnn_weights="$artifact_root_base/seed_${seed}/only_cnn/${run_name}.pt"
  reference_only_cnn_weights="$only_cnn_artifact_root_base/seed_${seed}/only_cnn/${run_name}.pt"
  only_cnn_weights="$current_only_cnn_weights"

  if [[ "$REUSE_ONLY_CNN" == "1" && -f "$reference_only_cnn_weights" ]]; then
    only_cnn_weights="$reference_only_cnn_weights"
    echo "    mode:           only_cnn"
    echo "    reuse weights:  $only_cnn_weights"
    echo
  elif [[ "$REUSE_ONLY_CNN" == "1" && -f "$current_only_cnn_weights" ]]; then
    only_cnn_weights="$current_only_cnn_weights"
    echo "    mode:           only_cnn"
    echo "    reuse weights:  $only_cnn_weights"
    echo
  else
    run_mode "$seed" "only_cnn" "" "" "0" "0.0" "$FULL_DISTILL_TEMPERATURE"
    only_cnn_weights="$current_only_cnn_weights"
  fi

  if [[ "$DRY_RUN" != "1" && ! -f "$only_cnn_weights" ]]; then
    echo "Missing only_cnn checkpoint for warm-start: $only_cnn_weights" >&2
    exit 1
  fi

  run_mode \
    "$seed" \
    "$ABLATION_MODE" \
    "$only_cnn_weights" \
    "$only_cnn_weights" \
    "$FULL_WARMSTART_FREEZE_EPOCHS" \
    "$FULL_DISTILL_ALPHA" \
    "$FULL_DISTILL_TEMPERATURE"

  if [[ "$RUN_LEGACY_FULL" == "1" ]]; then
    run_mode "$seed" "legacy_full" "" "" "0" "0.0" "$FULL_DISTILL_TEMPERATURE"
  fi
done

echo "Full-upgrade multiseed jobs finished."
