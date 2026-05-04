#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${DATA_ROOT:-/data/zwf/project/zhb/data}"
MODEL="${MODEL:-conmismatch9}"
ENCODER="${ENCODER:-c9}"
DATA_VARIANT_DIGIT="${DATA_VARIANT_DIGIT:-9}"
EPOCHS="${EPOCHS:-30}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
PATIENCE="${PATIENCE:-8}"
BATCH_SIZE="${BATCH_SIZE:-512}"
DEVICE="${DEVICE:-auto}"
POS_CAP="${POS_CAP:-5000}"
NEG_CAP="${NEG_CAP:-15000}"
SEED="${SEED:-42}"
RUN_ROOT="${RUN_ROOT:-runs}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts}"
FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR:-runs/feature_cache}"
CPU_THREADS="${CPU_THREADS:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
AMP="${AMP:-auto}"
HIDDEN_DIM="${HIDDEN_DIM:-96}"
DROPOUT="${DROPOUT:-0.20}"
ATTN_HEADS="${ATTN_HEADS:-4}"
ATTN_LAYERS="${ATTN_LAYERS:-2}"
ABLATION_MODE="${ABLATION_MODE:-full}"
WARMSTART_WEIGHTS_PATH="${WARMSTART_WEIGHTS_PATH:-}"
TEACHER_WEIGHTS_PATH="${TEACHER_WEIGHTS_PATH:-}"
WARMSTART_FREEZE_EPOCHS="${WARMSTART_FREEZE_EPOCHS:-0}"
DISTILL_ALPHA="${DISTILL_ALPHA:-0.0}"
DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-2.0}"
AUX_INIT_SCALE="${AUX_INIT_SCALE:-0.0}"
AUX_MAX_SCALE="${AUX_MAX_SCALE:-0.50}"
DRY_RUN=0
AUTO_ALL=0
FIRST_ONLY=1
FULL_TRAIN=0
SELECT_DATASET=""
DATASET_PATHS=()

usage() {
  cat <<'EOF'
Usage:
  bash scripts/train_all_datasets.sh [options] [DATASET_NPZ_OR_DIR ...]

Default:
  With no argument, auto-discover local data and train only the smallest dataset.
  A directory argument is treated as a data root; if it has a data/ child, only that child is scanned.

Examples:
  bash scripts/train_all_datasets.sh
  bash scripts/train_all_datasets.sh /path/to/K562_encoded.npz
  bash scripts/train_all_datasets.sh /path/a.npz /path/b.npz
  bash scripts/train_all_datasets.sh /data/zwf/project/zhb/
  bash scripts/train_all_datasets.sh --all
  bash scripts/train_all_datasets.sh --full --device cuda

Options:
  --dry-run             Print dataset order and commands without training.
  --all                 Auto-discover and train every local dataset from small to large.
  --first-only          Train only the smallest auto-discovered dataset. This is the default.
  --full                Do not cap positives or negatives.
  --dataset NAME        Auto-discover and train one dataset by directory name, e.g. K562.
  --model NAME          Model name, default: conmismatch9.
  --encoder NAME        Encoder name, default: c9.
  --epochs N            Training epochs, default: 30.
  --pos-cap N           Positive sample cap per dataset, default: 5000.
  --neg-cap N           Negative sample cap per dataset, default: 15000.
  --seed N              Random seed, default: 42.
  --data-root PATH      Data root, default: /data/zwf/project/zhb/data.
  --device DEVICE       Training device, default: auto.
  --batch-size N        Batch size, default: 512.
  --cpu-threads N       Torch compute threads, default: 8.
  --num-workers N       DataLoader workers, default: 4.
  --amp MODE            AMP mode: auto/true/false, default: auto.
  --hidden-dim N        Hidden width for the torch backbone, default: 96.
  --dropout VALUE       Dropout rate, default: 0.20.
  --attn-heads N        Attention heads, default: 4.
  --attn-layers N       Attention layers, default: 2.
  --ablation-mode NAME  ConMismatch9 ablation mode, default: full. Use legacy_full for the old gated full.
  Warm-start and distillation settings are read from env vars:
    WARMSTART_WEIGHTS_PATH, TEACHER_WEIGHTS_PATH, WARMSTART_FREEZE_EPOCHS,
    DISTILL_ALPHA, DISTILL_TEMPERATURE, AUX_INIT_SCALE, AUX_MAX_SCALE
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --all)
      AUTO_ALL=1
      FIRST_ONLY=0
      shift
      ;;
    --first-only)
      FIRST_ONLY=1
      AUTO_ALL=0
      shift
      ;;
    --full)
      FULL_TRAIN=1
      POS_CAP="all"
      NEG_CAP="all"
      shift
      ;;
    --dataset)
      SELECT_DATASET="${2:?--dataset requires a value}"
      FIRST_ONLY=0
      shift 2
      ;;
    --model)
      MODEL="${2:?--model requires a value}"
      shift 2
      ;;
    --encoder)
      ENCODER="${2:?--encoder requires a value}"
      shift 2
      ;;
    --epochs)
      EPOCHS="${2:?--epochs requires a value}"
      shift 2
      ;;
    --pos-cap)
      POS_CAP="${2:?--pos-cap requires a value}"
      shift 2
      ;;
    --neg-cap)
      NEG_CAP="${2:?--neg-cap requires a value}"
      shift 2
      ;;
    --seed)
      SEED="${2:?--seed requires a value}"
      shift 2
      ;;
    --device)
      DEVICE="${2:?--device requires a value}"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="${2:?--batch-size requires a value}"
      shift 2
      ;;
    --cpu-threads)
      CPU_THREADS="${2:?--cpu-threads requires a value}"
      shift 2
      ;;
    --num-workers)
      NUM_WORKERS="${2:?--num-workers requires a value}"
      shift 2
      ;;
    --amp)
      AMP="${2:?--amp requires a value}"
      shift 2
      ;;
    --hidden-dim)
      HIDDEN_DIM="${2:?--hidden-dim requires a value}"
      shift 2
      ;;
    --dropout)
      DROPOUT="${2:?--dropout requires a value}"
      shift 2
      ;;
    --attn-heads)
      ATTN_HEADS="${2:?--attn-heads requires a value}"
      shift 2
      ;;
    --attn-layers)
      ATTN_LAYERS="${2:?--attn-layers requires a value}"
      shift 2
      ;;
    --ablation-mode)
      ABLATION_MODE="${2:?--ablation-mode requires a value}"
      shift 2
      ;;
    --data-root)
      DATA_ROOT="${2:?--data-root requires a value}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        DATASET_PATHS+=("$1")
        shift
      done
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      DATASET_PATHS+=("$1")
      shift
      ;;
  esac
done

mkdir -p "$RUN_ROOT/generated_configs" "$RUN_ROOT/train_summaries" "$RUN_ROOT/train_logs" "$ARTIFACT_ROOT"

if [[ ${#DATASET_PATHS[@]} -gt 0 ]]; then
  mapfile -t DATASET_ROWS < <("$PYTHON_BIN" - "$DATA_VARIANT_DIGIT" "$SELECT_DATASET" "${DATASET_PATHS[@]}" <<'PY'
from pathlib import Path
import sys

from utils.io import load_npz_archive

variant_digit = sys.argv[1]
selected = sys.argv[2].strip()
suffix = variant_digit + "bit"
rows = []

def add_npz(path: Path):
    archive = load_npz_archive(path, names={"y"})
    labels = archive["y"]
    stem = path.stem
    if "_" in stem:
        name = stem.rsplit("_", 1)[0]
    else:
        name = path.parent.name
    if selected and name != selected:
        return
    rows.append((labels.length, name, str(path)))

def add_data_root(root: Path):
    if (root / "data").is_dir():
        root = root / "data"

    direct_candidate = root / f"{root.name}_{suffix}.npz"
    if direct_candidate.exists():
        add_npz(direct_candidate)
        return

    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        name = directory.name
        if selected and name != selected:
            continue
        npz_path = directory / f"{name}_{suffix}.npz"
        if npz_path.exists():
            add_npz(npz_path)

for raw_path in sys.argv[3:]:
    path = Path(raw_path).expanduser()
    if path.is_dir():
        add_data_root(path)
    elif path.is_file():
        add_npz(path)
    else:
        raise FileNotFoundError(str(path))

for sample_count, name, path in sorted(rows):
    print(f"{sample_count}\t{name}\t{path}")
PY
)
else
  mapfile -t DATASET_ROWS < <("$PYTHON_BIN" - "$DATA_ROOT" "$DATA_VARIANT_DIGIT" "$SELECT_DATASET" <<'PY'
from pathlib import Path
import sys

from utils.io import load_npz_archive

root = Path(sys.argv[1])
variant_digit = sys.argv[2]
selected = sys.argv[3].strip()
suffix = variant_digit + "bit"

rows = []
for directory in sorted(path for path in root.iterdir() if path.is_dir()):
    name = directory.name
    if selected and name != selected:
        continue
    npz_path = directory / f"{name}_{suffix}.npz"
    if not npz_path.exists():
        continue
    archive = load_npz_archive(npz_path, names={"y"})
    labels = archive["y"]
    rows.append((labels.length, name, str(npz_path)))

for sample_count, name, path in sorted(rows):
    print(f"{sample_count}\t{name}\t{path}")
PY
)
fi

if [[ ${#DATASET_ROWS[@]} -eq 0 ]]; then
  echo "No datasets found." >&2
  exit 1
fi

if [[ "$AUTO_ALL" != "1" && "$FIRST_ONLY" == "1" ]]; then
  DATASET_ROWS=("${DATASET_ROWS[0]}")
fi

echo "Training plan:"
printf '  %-12s %12s %s\n' "dataset" "samples" "file"
for row in "${DATASET_ROWS[@]}"; do
  IFS=$'\t' read -r sample_count dataset_name dataset_file <<< "$row"
  printf '  %-12s %12s %s\n' "$dataset_name" "$sample_count" "$dataset_file"
done

trained_count=0
for row in "${DATASET_ROWS[@]}"; do
  IFS=$'\t' read -r sample_count dataset_name dataset_file <<< "$row"
  safe_name="${dataset_name//[^A-Za-z0-9_]/_}"
  run_name="${MODEL}_${ENCODER}_${safe_name}"
  config_path="$RUN_ROOT/generated_configs/${run_name}.json"
  summary_path="$RUN_ROOT/train_summaries/${run_name}.json"
  log_path="$RUN_ROOT/train_logs/${run_name}.log"
  weights_path="$ARTIFACT_ROOT/${run_name}.pt"

  "$PYTHON_BIN" - "$config_path" "$MODEL" "$ENCODER" "$dataset_name" "$dataset_file" "$weights_path" "$EPOCHS" "$LEARNING_RATE" "$WEIGHT_DECAY" "$PATIENCE" "$POS_CAP" "$NEG_CAP" "$SEED" "$DEVICE" "$BATCH_SIZE" "$FEATURE_CACHE_DIR" "$CPU_THREADS" "$NUM_WORKERS" "$AMP" "$HIDDEN_DIM" "$DROPOUT" "$ATTN_HEADS" "$ATTN_LAYERS" "$ABLATION_MODE" "$WARMSTART_WEIGHTS_PATH" "$TEACHER_WEIGHTS_PATH" "$WARMSTART_FREEZE_EPOCHS" "$DISTILL_ALPHA" "$DISTILL_TEMPERATURE" "$AUX_INIT_SCALE" "$AUX_MAX_SCALE" <<'PY'
import json
from pathlib import Path
import sys

(
    config_path,
    model,
    encoder,
    dataset_name,
    dataset_file,
    weights_path,
    epochs,
    learning_rate,
    weight_decay,
    patience,
    pos_cap,
    neg_cap,
    seed,
    device,
    batch_size,
    feature_cache_dir,
    cpu_threads,
    num_workers,
    amp,
    hidden_dim,
    dropout,
    attn_heads,
    attn_layers,
    ablation_mode,
    warmstart_weights_path,
    teacher_weights_path,
    warmstart_freeze_epochs,
    distill_alpha,
    distill_temperature,
    aux_init_scale,
    aux_max_scale,
) = sys.argv[1:]

def _cap(value: str):
    if value.lower() in {"all", "none", "null", ""}:
        return None
    return int(value)

payload = {
    "model": model,
    "encoder": encoder,
    "device": device,
    "weights_path": weights_path,
    "dataset_files": [dataset_file],
    "dataset_name": dataset_name,
    "warmstart_weights_path": warmstart_weights_path or None,
    "teacher_weights_path": teacher_weights_path or None,
    "feature_cache_dir": feature_cache_dir,
    "cache_features": True,
    "sampling": {
        "positive_cap": _cap(pos_cap),
        "negative_cap": _cap(neg_cap),
    },
    "training": {
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "patience": int(patience),
        "batch_size": int(batch_size),
        "cpu_threads": int(cpu_threads),
        "num_workers": int(num_workers),
        "amp": amp,
        "hidden_dim": int(hidden_dim),
        "dropout": float(dropout),
        "attn_heads": int(attn_heads),
        "attn_layers": int(attn_layers),
        "ablation_mode": ablation_mode,
        "warmstart_freeze_epochs": int(warmstart_freeze_epochs),
        "distill_alpha": float(distill_alpha),
        "distill_temperature": float(distill_temperature),
        "aux_init_scale": float(aux_init_scale),
        "aux_max_scale": float(aux_max_scale),
    },
    "seed": int(seed),
}

path = Path(config_path)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
PY

  echo
  echo "==> ${dataset_name} (${sample_count} samples)"
  echo "    config:  $config_path"
  echo "    weights: $weights_path"
  echo "    summary: $summary_path"
  echo "    log:     $log_path"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "    dry-run: $PYTHON_BIN train.py --config $config_path --output $summary_path"
  else
    "$PYTHON_BIN" train.py --config "$config_path" --output "$summary_path" 2>&1 | tee "$log_path"
  fi

  trained_count=$((trained_count + 1))
done

echo
if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run complete. Planned datasets: $trained_count"
else
  echo "Training complete. Finished datasets: $trained_count"
fi
