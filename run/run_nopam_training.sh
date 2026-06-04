#!/usr/bin/env bash
# NoPAM Training Launcher
# 用法:
#   bash run/run_nopam_training.sh        # 默认单卡（避开 DDP NCCL 超时问题）
#   bash run/run_nopam_training.sh --ddp  # 双卡 DDP（torchrun）

set -euo pipefail

# 解析参数
USE_DDP=false
if [ "${1:-}" = "--ddp" ]; then
    USE_DDP=true
fi

echo "==================================="
echo " NoPAM Training Launcher"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "==================================="

# 1. 确认当前目录
PROJECT_ROOT="/data/zwf/code1/reborn_seed"
cd "$PROJECT_ROOT" || { echo "ERROR: Cannot cd to $PROJECT_ROOT"; exit 1; }
echo "[OK] Working dir: $(pwd)"

# 2. 切换 git branch
TARGET_BRANCH="fair-bl0-bl5split"
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$CURRENT_BRANCH" != "$TARGET_BRANCH" ]; then
    echo "[WARN] Current branch is '$CURRENT_BRANCH', switching to '$TARGET_BRANCH'..."
    git checkout "$TARGET_BRANCH"
    git status --short
else
    echo "[OK] Already on branch '$TARGET_BRANCH'"
fi

# 3. 切换 conda 环境
CONDA_ENV="reborn_seed"
CONDA_SH="/data/zwf/Conda/miniconda3/etc/profile.d/conda.sh"
if [ -z "${CONDA_PREFIX:-}" ] || [[ "$CONDA_PREFIX" != *"$CONDA_ENV"* ]]; then
    echo "[INFO] Activating conda env: $CONDA_ENV"
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
fi
echo "[OK] Conda env: $CONDA_PREFIX"
echo "[OK] Python: $(which python) ($(python --version))"

# 4. GPU 检查
echo "--- GPU Status ---"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader

# 5. 清理残留训练进程
echo "[INFO] Checking for stale train_bl5.py processes..."
STALE_PIDS=$(pgrep -f "train_bl5.py.*nopam" || true)
if [ -n "$STALE_PIDS" ]; then
    echo "[WARN] Found stale PIDs: $STALE_PIDS, killing..."
    kill -9 $STALE_PIDS 2>/dev/null || true
    sleep 2
fi

# 6. 检查已有 tmux session
TMUX_NAME="nopam"
if tmux has-session -t "$TMUX_NAME" 2>/dev/null; then
    echo "[WARN] tmux session '$TMUX_NAME' already exists."
    read -rp "Kill existing session and restart? [y/N]: " answer
    if [[ "$answer" =~ ^[Yy]$ ]]; then
        tmux kill-session -t "$TMUX_NAME"
        echo "[OK] Killed old tmux session."
    else
        echo "[INFO] Keeping existing session. Attach with: tmux attach -t $TMUX_NAME"
        exit 0
    fi
fi

# 7. 准备日志
TIMESTAMP=$(date +%m%d_%H%M)
LOG_FILE="/tmp/nopam_train_${TIMESTAMP}.log"
CONFIG="configs/bl5_v4_nopam_control.yaml"
OUTPUT_DIR="results/bl5_v4_nopam_control"

echo "[INFO] Config: $CONFIG"
echo "[INFO] Output: $OUTPUT_DIR"
echo "[INFO] Log:    $LOG_FILE"

# 8. 清理旧的空 checkpoints 目录（如果存在且为空）
if [ -d "$OUTPUT_DIR/checkpoints" ] && [ ! "$(ls -A "$OUTPUT_DIR/checkpoints" 2>/dev/null)" ]; then
    echo "[INFO] Removing empty checkpoints dir from previous crash..."
    rm -rf "$OUTPUT_DIR/checkpoints"
fi

# 9. 启动训练
if [ "$USE_DDP" = true ]; then
    echo "[INFO] Mode: DDP (2 GPUs)"
    TRAIN_CMD="torchrun --nproc_per_node=2 --master_port=29501 scripts/train_bl5.py --config '$CONFIG'"
else
    echo "[INFO] Mode: Single GPU (cuda:0)"
    TRAIN_CMD="CUDA_VISIBLE_DEVICES=0 python scripts/train_bl5.py --config '$CONFIG'"
fi

echo "[INFO] Starting training in tmux session '$TMUX_NAME'..."
tmux new-session -d -s "$TMUX_NAME" \
    "cd '$PROJECT_ROOT' && \
     source '$CONDA_SH' && \
     conda activate '$CONDA_ENV' && \
     $TRAIN_CMD 2>&1 | tee '$LOG_FILE'; \
     echo \"EXIT_CODE=\$?\" >> '$LOG_FILE'; \
     echo \"Training finished at \$(date)\" >> '$LOG_FILE'"

sleep 1
if tmux has-session -t "$TMUX_NAME" 2>/dev/null; then
    echo ""
    echo "==================================="
    echo " Training started successfully!"
    echo "==================================="
    echo "Attach:     tmux attach -t $TMUX_NAME"
    echo "Detach:     Ctrl+b 然后按 d"
    echo "Log file:   tail -f $LOG_FILE"
    echo "Check GPU:  watch -n 2 nvidia-smi"
    echo "Check ckpt: ls -lh $OUTPUT_DIR/checkpoints/"
    echo ""
else
    echo "[ERROR] Failed to create tmux session!"
    exit 1
fi
