#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

usage() {
  cat <<'EOF'
Usage:
  bash run/export_five_dataset_predictions_full.sh

Environment overrides:
  PYTHON_BIN=/path/to/python       Python interpreter to use.
  DEVICE=cuda|cuda:0|cuda:1|cpu    Torch inference device. Default: cuda.
  CUDA_VISIBLE_DEVICES=0           Optional host GPU binding.
  BATCH_SIZE=65536                 Inference batch size. Default: 65536.
  CPU_THREADS=32                   CPU thread env vars. Default: nproc.
  OUTPUT_ROOT=/path/to/output      Output directory. Default: <project>/output.
  PACKAGE_NAME=name                Output package folder/zip name.
  CONFIG=/path/to/config.yaml      Dataset config. Default: configs/c9_conmismatch9.yaml.
  DRY_RUN=1                        Print the resolved command without running inference.

Outputs:
  output/<package>/predictions.csv
  output/<package>/predictions.json
  output/<package>/summary.json
  output/<package>.zip
  output/logs/<run>.log
EOF
}

case "${1:-}" in
  -h|--help|help)
    usage
    exit 0
    ;;
esac

PYTHON_BIN="${PYTHON_BIN:-/data/zwf/conda/envs/reborn_seed/bin/python}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-65536}"
CPU_THREADS="${CPU_THREADS:-$(nproc)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/output}"
CONFIG="${CONFIG:-$PROJECT_ROOT/configs/c9_conmismatch9.yaml}"
PACKAGE_NAME="${PACKAGE_NAME:-crispr_dualpred_five_dataset_full_$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$OUTPUT_ROOT/logs"
LOG_FILE="$OUTPUT_ROOT/logs/export_five_dataset_predictions_full_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter is not executable: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -f "$PROJECT_ROOT/scripts/export_five_dataset_predictions.py" ]]; then
  echo "Missing export script: $PROJECT_ROOT/scripts/export_five_dataset_predictions.py" >&2
  exit 1
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "Missing config: $CONFIG" >&2
  exit 1
fi

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CPU_THREADS"
export NUMEXPR_NUM_THREADS="$CPU_THREADS"

echo "CRISPR-DualPred five-dataset full export"
echo "  project_root:          $PROJECT_ROOT"
echo "  python:                $PYTHON_BIN"
echo "  config:                $CONFIG"
echo "  device:                $DEVICE"
echo "  cuda_visible_devices:  ${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "  batch_size:            $BATCH_SIZE"
echo "  cpu_threads:           $CPU_THREADS"
echo "  output_root:           $OUTPUT_ROOT"
echo "  package_name:          $PACKAGE_NAME"
echo "  log:                   $LOG_FILE"
echo

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
  echo
fi

if [[ "$DEVICE" == cuda* && "$DRY_RUN" != "1" ]]; then
  "$PYTHON_BIN" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' || {
    echo "Torch CUDA is not available in this environment. Set DEVICE=cpu only if you intentionally want CPU inference." >&2
    exit 1
  }
fi

cmd=(
  "$PYTHON_BIN"
  "$PROJECT_ROOT/scripts/export_five_dataset_predictions.py"
  --config "$CONFIG"
  --device "$DEVICE"
  --batch-size "$BATCH_SIZE"
  --output-root "$OUTPUT_ROOT"
  --package-name "$PACKAGE_NAME"
)

echo "Command:"
printf '  %q' "${cmd[@]}"
echo
echo

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1, not running inference."
  exit 0
fi

"${cmd[@]}"

echo
echo "Export finished."
echo "Package directory: $OUTPUT_ROOT/$PACKAGE_NAME"
echo "Package zip:       $OUTPUT_ROOT/$PACKAGE_NAME.zip"
