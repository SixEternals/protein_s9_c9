#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

usage() {
  cat <<'EOF'
Usage:
  bash run/conmismatch9_full_upgrade_guard_d.sh [smoke|external|core|all|full]

Phases:
  smoke     Run the smallest real validation: Tasi seed 42 and SITE seed 45.
  external  Run external datasets only: Tasi seeds 42/43/44 and K562 seeds 42/43/44/45/46.
  core      Run core datasets only: CHANGE-seq seeds 42/43/44 and SITE seeds 42/43/44/45/46.
  all/full  Run external + core datasets.

Important env vars:
  DRY_RUN=1                Print commands without training.
  SKIP_EXISTING=0          Re-run even if guard_d full summary and checkpoint already exist.
  CUDA_VISIBLE_DEVICES=N   Bind GPU on the host.
  DEVICE=cuda              Training device, default cuda.

This wrapper runs guard_d only:
  freeze_epochs=30, distill_alpha=0.45, aux_max_scale=0.10.

It defaults to reusing guard_c only_cnn checkpoints as warmstart / teacher so
guard_d remains a clean distill-alpha micro-tune against guard_c.
EOF
}

PHASE="${1:-${PHASE:-smoke}}"
case "$PHASE" in
  -h|--help|help)
    usage
    exit 0
    ;;
esac

PYTHON_BIN="${PYTHON_BIN:-/data/zwf/conda/envs/reborn_seed/bin/python}"
DATA_ROOT="${DATA_ROOT:-/data/zwf/project/zhb/data}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-8}"
BATCH_SIZE="${BATCH_SIZE:-512}"
CPU_THREADS="${CPU_THREADS:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
AMP="${AMP:-auto}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

GUARD_D_RUN_BASE="${GUARD_D_RUN_BASE:-$PROJECT_ROOT/runs/full_upgrade_guard_d}"
GUARD_D_ARTIFACT_BASE="${GUARD_D_ARTIFACT_BASE:-$PROJECT_ROOT/artifacts/full_upgrade_guard_d}"
ONLY_CNN_ARTIFACT_BASE="${ONLY_CNN_ARTIFACT_BASE:-$PROJECT_ROOT/artifacts/full_upgrade_guard_c}"

FULL_WARMSTART_FREEZE_EPOCHS="${FULL_WARMSTART_FREEZE_EPOCHS:-30}"
FULL_DISTILL_ALPHA="${FULL_DISTILL_ALPHA:-0.45}"
FULL_DISTILL_TEMPERATURE="${FULL_DISTILL_TEMPERATURE:-2.0}"
FULL_AUX_INIT_SCALE="${FULL_AUX_INIT_SCALE:-0.0}"
FULL_AUX_MAX_SCALE="${FULL_AUX_MAX_SCALE:-0.10}"

LOG_DIR="$PROJECT_ROOT/runs/job_logs"
mkdir -p "$LOG_DIR"
script_log="$LOG_DIR/conmismatch9_full_upgrade_guard_d_${PHASE}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$script_log") 2>&1

safe_dataset_name() {
  local value="$1"
  value="${value//[^A-Za-z0-9_]/_}"
  printf '%s' "$value"
}

run_seed() {
  local dataset="$1"
  local seed="$2"

  local safe_name
  local dataset_slug_lower
  safe_name="$(safe_dataset_name "$dataset")"
  dataset_slug_lower="${safe_name,,}"

  local run_name="conmismatch9_c9_${safe_name}"
  local dataset_file="$DATA_ROOT/$dataset/${dataset}_9bit.npz"
  local full_summary="$GUARD_D_RUN_BASE/full_upgrade_${dataset_slug_lower}_conmismatch9/seed_${seed}/full/train_summaries/${run_name}.json"
  local full_weights="$GUARD_D_ARTIFACT_BASE/full_upgrade_${dataset_slug_lower}_conmismatch9/seed_${seed}/full/${run_name}.pt"
  local reference_only_cnn="$ONLY_CNN_ARTIFACT_BASE/full_upgrade_${dataset_slug_lower}_conmismatch9/seed_${seed}/only_cnn/${run_name}.pt"

  if [[ ! -f "$dataset_file" ]]; then
    echo "Missing dataset file: $dataset_file" >&2
    exit 1
  fi

  if [[ "$DRY_RUN" != "1" && ! -f "$reference_only_cnn" ]]; then
    echo "Missing guard_c only_cnn reference checkpoint: $reference_only_cnn" >&2
    echo "Run guard_c first, or override ONLY_CNN_ARTIFACT_BASE to a compatible only_cnn artifact root." >&2
    exit 1
  fi

  if [[ "$SKIP_EXISTING" == "1" && -f "$full_summary" && -f "$full_weights" ]]; then
    echo "==> $dataset seed=$seed"
    echo "    skip existing: $full_summary"
    echo
    return
  fi

  echo "==> $dataset seed=$seed"
  echo "    data:          $dataset_file"
  echo "    only_cnn_ref:  $reference_only_cnn"
  echo "    summary:       $full_summary"
  echo "    weights:       $full_weights"

  RUN_BASE="$GUARD_D_RUN_BASE" \
  ARTIFACT_BASE="$GUARD_D_ARTIFACT_BASE" \
  ONLY_CNN_ARTIFACT_BASE="$ONLY_CNN_ARTIFACT_BASE" \
  PYTHON_BIN="$PYTHON_BIN" \
  DATA_ROOT="$DATA_ROOT" \
  DEVICE="$DEVICE" \
  EPOCHS="$EPOCHS" \
  PATIENCE="$PATIENCE" \
  BATCH_SIZE="$BATCH_SIZE" \
  CPU_THREADS="$CPU_THREADS" \
  NUM_WORKERS="$NUM_WORKERS" \
  AMP="$AMP" \
  SEEDS="$seed" \
  FULL_WARMSTART_FREEZE_EPOCHS="$FULL_WARMSTART_FREEZE_EPOCHS" \
  FULL_DISTILL_ALPHA="$FULL_DISTILL_ALPHA" \
  FULL_DISTILL_TEMPERATURE="$FULL_DISTILL_TEMPERATURE" \
  FULL_AUX_INIT_SCALE="$FULL_AUX_INIT_SCALE" \
  FULL_AUX_MAX_SCALE="$FULL_AUX_MAX_SCALE" \
  REUSE_ONLY_CNN=1 \
  RUN_LEGACY_FULL=0 \
  DRY_RUN="$DRY_RUN" \
  bash scripts/run_conmismatch9_full_upgrade_multiseed.sh "$dataset"
}

run_dataset() {
  local dataset="$1"
  local seeds="$2"
  local seed_list
  read -r -a seed_list <<< "$seeds"

  echo ">>> guard_d dataset=$dataset seeds=[$seeds]"
  for seed in "${seed_list[@]}"; do
    run_seed "$dataset" "$seed"
  done
}

run_external() {
  run_dataset "Tasi" "42 43 44"
  run_dataset "K562" "42 43 44 45 46"
}

run_core() {
  run_dataset "CHANGE-seq" "42 43 44"
  run_dataset "SITE" "42 43 44 45 46"
}

echo "ConMismatch9 guard_d staged full++ plan"
echo "  log:             $script_log"
echo "  phase:           $PHASE"
echo "  dry_run:         $DRY_RUN"
echo "  skip_existing:   $SKIP_EXISTING"
echo "  device:          $DEVICE"
echo "  run_base:        $GUARD_D_RUN_BASE"
echo "  artifact_base:   $GUARD_D_ARTIFACT_BASE"
echo "  only_cnn_base:   $ONLY_CNN_ARTIFACT_BASE"
echo "  freeze_epochs:   $FULL_WARMSTART_FREEZE_EPOCHS"
echo "  distill_alpha:   $FULL_DISTILL_ALPHA"
echo "  aux_max_scale:   $FULL_AUX_MAX_SCALE"
echo

case "$PHASE" in
  smoke)
    run_dataset "Tasi" "42"
    run_dataset "SITE" "45"
    ;;
  external)
    run_external
    ;;
  core)
    run_core
    ;;
  all|full)
    run_external
    run_core
    ;;
  *)
    echo "Unknown phase: $PHASE" >&2
    usage >&2
    exit 2
    ;;
esac

echo "ConMismatch9 guard_d staged jobs finished."
