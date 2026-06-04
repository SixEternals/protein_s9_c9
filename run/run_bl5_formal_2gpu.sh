#!/usr/bin/env bash
# Launch a BL5 formal run on two GPUs.
#
# Usage:
#   bash run/run_bl5_formal_2gpu.sh
#   bash run/run_bl5_formal_2gpu.sh configs/bl5_v4_pam_shuffle_control.yaml
#   bash run/run_bl5_formal_2gpu.sh configs/bl5_v4_pam.yaml results/bl5_v4_pam_rerun
#
# Environment overrides:
#   PYTHON_BIN=/path/to/python
#   CUDA_VISIBLE_DEVICES=0,1
#   MASTER_PORT=29531
#   TMUX_NAME=bl5_formal
#   FOREGROUND=1

set -euo pipefail

PROJECT_ROOT="/data/zwf/code1/reborn_seed"
cd "$PROJECT_ROOT"

CONFIG="${1:-${CONFIG:-configs/bl5_v4_pam.yaml}}"
OUTPUT_DIR="${2:-${OUTPUT_DIR:-}}"

PYTHON_BIN="${PYTHON_BIN:-/data/zwf/conda/envs/reborn_seed/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
MASTER_PORT="${MASTER_PORT:-29531}"
TMUX_NAME="${TMUX_NAME:-bl5_formal}"
FOREGROUND="${FOREGROUND:-0}"
LOG_DIR="${LOG_DIR:-/tmp}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${TMUX_NAME}_${TIMESTAMP}.log"

if [ ! -f "$CONFIG" ]; then
  echo "[ERROR] Config not found: $CONFIG" >&2
  exit 1
fi

if [ ! -x "$PYTHON_BIN" ]; then
  echo "[ERROR] Python not executable: $PYTHON_BIN" >&2
  exit 1
fi

echo "========================================"
echo " BL5 formal 2-GPU launcher"
echo "========================================"
echo "[INFO] Project: $PROJECT_ROOT"
echo "[INFO] Config:  $CONFIG"
echo "[INFO] Python:  $PYTHON_BIN"
echo "[INFO] GPUs:    $CUDA_VISIBLE_DEVICES"
echo "[INFO] Port:    $MASTER_PORT"
echo "[INFO] Log:     $LOG_FILE"
if [ -n "$OUTPUT_DIR" ]; then
  echo "[INFO] Output:  $OUTPUT_DIR"
fi

echo "[INFO] Python package check..."
"$PYTHON_BIN" - <<'PY'
import sys
import torch
print(f"python={sys.executable}")
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_device_count={torch.cuda.device_count()}")
PY

echo "[INFO] GPU status before launch..."
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader || true

TRAIN_CMD=(
  env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
  "$PYTHON_BIN" -m torch.distributed.run
  --nproc_per_node=2
  --master_port="$MASTER_PORT"
  scripts/train_bl5.py
  --config "$CONFIG"
)

if [ -n "$OUTPUT_DIR" ]; then
  TRAIN_CMD+=(--output-dir "$OUTPUT_DIR")
fi

if [ "$FOREGROUND" = "1" ]; then
  echo "[INFO] Running in foreground."
  "${TRAIN_CMD[@]}" 2>&1 | tee "$LOG_FILE"
  exit "${PIPESTATUS[0]}"
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "[WARN] tmux not found; running in foreground instead."
  "${TRAIN_CMD[@]}" 2>&1 | tee "$LOG_FILE"
  exit "${PIPESTATUS[0]}"
fi

if tmux has-session -t "$TMUX_NAME" 2>/dev/null; then
  echo "[ERROR] tmux session already exists: $TMUX_NAME" >&2
  echo "        Attach: tmux attach -t $TMUX_NAME" >&2
  echo "        Or choose another name: TMUX_NAME=bl5_formal_2 bash $0 $CONFIG ${OUTPUT_DIR:-}" >&2
  exit 1
fi

printf -v TRAIN_CMD_STR "%q " "${TRAIN_CMD[@]}"
tmux new-session -d -s "$TMUX_NAME" \
  "cd '$PROJECT_ROOT' && \
   echo '[INFO] Started at '\"\$(date)\" | tee '$LOG_FILE'; \
   $TRAIN_CMD_STR 2>&1 | tee -a '$LOG_FILE'; \
   status=\${PIPESTATUS[0]}; \
   echo '[INFO] Finished at '\"\$(date)\"' status='\"\$status\" | tee -a '$LOG_FILE'; \
   exit \"\$status\""

echo "[OK] Started in tmux session: $TMUX_NAME"
echo "[OK] Attach: tmux attach -t $TMUX_NAME"
echo "[OK] Log:    tail -f $LOG_FILE"
echo "[OK] GPU:    watch -n 2 nvidia-smi"
