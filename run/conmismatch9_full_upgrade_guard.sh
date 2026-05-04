#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/data/zwf/conda/envs/reborn_seed/bin/python}"
DATA_ROOT="${DATA_ROOT:-/data/zwf/project/zhb/data}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-8}"
BATCH_SIZE="${BATCH_SIZE:-512}"
CPU_THREADS="${CPU_THREADS:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
AMP="${AMP:-auto}"
RECIPES="${RECIPES:-guard_b guard_c}"
STOP_AFTER_EXTERNAL="${STOP_AFTER_EXTERNAL:-0}"
DRY_RUN="${DRY_RUN:-0}"
FULL_DISTILL_TEMPERATURE="${FULL_DISTILL_TEMPERATURE:-2.0}"
FULL_AUX_INIT_SCALE="${FULL_AUX_INIT_SCALE:-0.0}"
LOG_DIR="$PROJECT_ROOT/runs/job_logs"

mkdir -p "$LOG_DIR"
script_log="$LOG_DIR/conmismatch9_full_upgrade_guard_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$script_log") 2>&1

echo "ConMismatch9 full-upgrade guard plan"
echo "  log:             $script_log"
echo "  recipes:         $RECIPES"
echo "  stop_after_ext:   $STOP_AFTER_EXTERNAL"
echo "  dry_run:         $DRY_RUN"
echo "  device:          $DEVICE"
echo

run_dataset() {
  local recipe="$1"
  local dataset="$2"
  local seeds="$3"
  local freeze_epochs="$4"
  local distill_alpha="$5"
  local aux_max_scale="$6"

  local run_base="$PROJECT_ROOT/runs/full_upgrade_${recipe}"
  local artifact_base="$PROJECT_ROOT/artifacts/full_upgrade_${recipe}"

  echo ">>> recipe=$recipe dataset=$dataset seeds=[$seeds]"

  RUN_BASE="$run_base" \
  ARTIFACT_BASE="$artifact_base" \
  PYTHON_BIN="$PYTHON_BIN" \
  DATA_ROOT="$DATA_ROOT" \
  DEVICE="$DEVICE" \
  EPOCHS="$EPOCHS" \
  PATIENCE="$PATIENCE" \
  BATCH_SIZE="$BATCH_SIZE" \
  CPU_THREADS="$CPU_THREADS" \
  NUM_WORKERS="$NUM_WORKERS" \
  AMP="$AMP" \
  SEEDS="$seeds" \
  FULL_WARMSTART_FREEZE_EPOCHS="$freeze_epochs" \
  FULL_DISTILL_ALPHA="$distill_alpha" \
  FULL_DISTILL_TEMPERATURE="$FULL_DISTILL_TEMPERATURE" \
  FULL_AUX_INIT_SCALE="$FULL_AUX_INIT_SCALE" \
  FULL_AUX_MAX_SCALE="$aux_max_scale" \
  REUSE_ONLY_CNN=1 \
  RUN_LEGACY_FULL=0 \
  DRY_RUN="$DRY_RUN" \
  bash scripts/run_conmismatch9_full_upgrade_multiseed.sh "$dataset"

  echo
}

run_recipe() {
  local recipe="$1"
  local freeze_epochs
  local distill_alpha
  local aux_max_scale

  case "$recipe" in
    guard_b)
      freeze_epochs=30
      distill_alpha=0.30
      aux_max_scale=0.15
      ;;
    guard_c)
      freeze_epochs=30
      distill_alpha=0.50
      aux_max_scale=0.10
      ;;
    *)
      echo "Unknown recipe: $recipe" >&2
      exit 2
      ;;
  esac

  echo "=== recipe $recipe ==="
  echo "  freeze_epochs=$freeze_epochs"
  echo "  distill_alpha=$distill_alpha"
  echo "  aux_max_scale=$aux_max_scale"
  echo

  run_dataset "$recipe" "Tasi" "42 43 44" "$freeze_epochs" "$distill_alpha" "$aux_max_scale"
  run_dataset "$recipe" "K562" "42 43 44 45 46" "$freeze_epochs" "$distill_alpha" "$aux_max_scale"

  if [[ "$STOP_AFTER_EXTERNAL" == "1" ]]; then
    echo "STOP_AFTER_EXTERNAL=1, skipping core datasets for recipe=$recipe."
    return
  fi

  run_dataset "$recipe" "CHANGE-seq" "42 43 44" "$freeze_epochs" "$distill_alpha" "$aux_max_scale"
  run_dataset "$recipe" "SITE" "42 43 44 45 46" "$freeze_epochs" "$distill_alpha" "$aux_max_scale"
}

for recipe in $RECIPES; do
  run_recipe "$recipe"
done

echo "ConMismatch9 full-upgrade guard jobs finished."
