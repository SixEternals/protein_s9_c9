from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from encoders.c9_encoder import C9Encoder
from encoders.r9_encoder import R9Encoder
from train import _checkpoint_payload, _instantiate_model_from_payload
from utils.config import load_config


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def optional_cap(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "all", "none", "null"}:
        return None
    return int(value)


def configure_runtime(config: dict[str, Any]) -> dict[str, Any]:
    evaluation = config.get("evaluation", {})
    cpu_threads = int(evaluation.get("cpu_threads", 8))
    os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(max(1, min(4, cpu_threads // 2)))
    except RuntimeError:
        pass
    return {"cpu_threads": cpu_threads}


def choose_device(config: dict[str, Any]) -> torch.device:
    requested = str(config.get("device", "auto")).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def select_rows(labels: np.ndarray, observed_positive_cap: int | None, unobserved_candidate_cap: int | None, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    observed = np.flatnonzero(labels.astype(np.int64) == 1)
    candidates = np.flatnonzero(labels.astype(np.int64) == 0)
    if observed_positive_cap is not None and len(observed) > observed_positive_cap:
        observed = rng.choice(observed, size=observed_positive_cap, replace=False)
    if unobserved_candidate_cap is not None and len(candidates) > unobserved_candidate_cap:
        candidates = rng.choice(candidates, size=unobserved_candidate_cap, replace=False)
    selected = np.concatenate([observed, candidates])
    rng.shuffle(selected)
    return selected


def stratified_split(labels: np.ndarray, seed: int, train_ratio: float, val_ratio: float) -> SplitIndices:
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for value in (0, 1):
        cls = np.flatnonzero(labels.astype(np.int64) == value)
        rng.shuffle(cls)
        n = len(cls)
        train_end = int(round(n * train_ratio))
        val_end = train_end + int(round(n * val_ratio))
        train_parts.append(cls[:train_end])
        val_parts.append(cls[train_end:val_end])
        test_parts.append(cls[val_end:])
    train = np.concatenate(train_parts)
    val = np.concatenate(val_parts)
    test = np.concatenate(test_parts)
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return SplitIndices(train=train, val=val, test=test)


def encode_reference(encoder_name: str, on_seq: np.ndarray, off_seq: np.ndarray) -> np.ndarray:
    encoder = R9Encoder() if encoder_name == "r9" else C9Encoder()
    encoded = np.empty((len(on_seq), 23, 9), dtype=np.uint8)
    for idx, (on, off) in enumerate(zip(on_seq, off_seq)):
        encoded[idx] = np.asarray(encoder.encode_pair(str(on), str(off)), dtype=np.uint8)
    return encoded


def load_features(
    cache_path: Path,
    encoder_name: str,
    selected_rows: np.ndarray,
    data: np.lib.npyio.NpzFile,
    verify_first_n: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)
    features = np.load(cache_path, allow_pickle=False)
    summary: dict[str, Any] = {
        "path": str(cache_path),
        "shape": list(features.shape),
        "dtype": str(features.dtype),
        "verified_first_n": 0,
        "cache_order_ok": None,
    }
    if features.shape[0] != len(selected_rows) or features.shape[1:] != (23, 9):
        raise ValueError(f"feature cache shape mismatch for {cache_path}: {features.shape}; selected_rows={len(selected_rows)}")
    if verify_first_n > 0:
        n = min(verify_first_n, len(selected_rows))
        on_seq = np.asarray(data["on_seq"][selected_rows[:n]], dtype=str)
        off_seq = np.asarray(data["off_seq"][selected_rows[:n]], dtype=str)
        expected = encode_reference(encoder_name, on_seq, off_seq)
        actual = np.asarray(features[:n], dtype=np.uint8)
        cache_order_ok = bool(np.array_equal(expected, actual))
        summary["verified_first_n"] = int(n)
        summary["cache_order_ok"] = cache_order_ok
        if not cache_order_ok:
            raise ValueError(f"feature cache order/content check failed for {cache_path}")
    return features, summary


def load_model(checkpoint_path: Path) -> tuple[str, str, torch.nn.Module, dict[str, Any]]:
    payload = _checkpoint_payload(checkpoint_path)
    model_name, encoder_name, model = _instantiate_model_from_payload(payload)
    model.eval()
    metadata = {
        "path": str(checkpoint_path),
        "model_name": model_name,
        "encoder_name": encoder_name,
        "best_epoch": payload.get("best_epoch") if isinstance(payload, dict) else None,
        "best_val_aupr": payload.get("best_val_aupr") if isinstance(payload, dict) else None,
        "test_metrics_from_checkpoint": payload.get("test_metrics") if isinstance(payload, dict) else None,
        "model_config": payload.get("model_config") if isinstance(payload, dict) else None,
    }
    return model_name, encoder_name, model, metadata


@torch.no_grad()
def predict_probabilities(
    model: torch.nn.Module,
    features: np.ndarray,
    row_indices: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    subset = np.ascontiguousarray(features[row_indices], dtype=np.float32)
    dataset = TensorDataset(torch.from_numpy(subset))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = model.to(device)
    model.eval()
    outputs: list[np.ndarray] = []
    for (x,) in loader:
        x = x.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
        logits = model(x)
        outputs.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(outputs).astype(np.float64, copy=False)


def metrics_for(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, float]:
    labels_int = labels.astype(np.int64)
    preds = (probs >= threshold).astype(np.int64)
    if len(np.unique(labels_int)) < 2:
        auroc = 0.5
        auprc = 0.0
    else:
        auroc = float(roc_auc_score(labels_int, probs))
        auprc = float(average_precision_score(labels_int, probs))
    return {
        "auroc": auroc,
        "auprc": auprc,
        "accuracy": float(accuracy_score(labels_int, preds)),
        "precision": float(precision_score(labels_int, preds, zero_division=0)),
        "recall": float(recall_score(labels_int, preds, zero_division=0)),
        "f1": float(f1_score(labels_int, preds, zero_division=0)),
    }


def build_weight_grid(start: float, stop: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError("weight_grid_step must be positive")
    count = int(round((stop - start) / step))
    grid = start + np.arange(count + 1, dtype=np.float64) * step
    grid = np.clip(grid, min(start, stop), max(start, stop))
    return np.unique(np.round(grid, 10))


def tune_weight(
    labels: np.ndarray,
    region_probs: np.ndarray,
    run_probs: np.ndarray,
    grid: np.ndarray,
    threshold: float,
) -> tuple[float, list[dict[str, float]]]:
    rows: list[dict[str, float]] = []
    best_row: dict[str, float] | None = None
    for weight_region in grid:
        fused = weight_region * region_probs + (1.0 - weight_region) * run_probs
        scores = metrics_for(labels, fused, threshold)
        row = {"weight_region_r9": float(weight_region), **scores}
        rows.append(row)
        if best_row is None:
            best_row = row
            continue
        if (row["auprc"], row["auroc"]) > (best_row["auprc"], best_row["auroc"]):
            best_row = row
    if best_row is None:
        raise RuntimeError("empty weight grid")
    return float(best_row["weight_region_r9"]), rows


def git_info() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip() != ""
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def append_experiment(csv_path: Path, summary: dict[str, Any], config_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "version",
        "date",
        "commit_hash",
        "status",
        "auroc",
        "auprc",
        "train_time",
        "gpu_mem",
        "epochs",
        "best_epoch",
        "config_path",
        "notes",
    ]
    exists = csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists or csv_path.stat().st_size == 0:
            writer.writeheader()
        test = summary["test_metrics"]["weighted_average"]
        writer.writerow(
            {
                "version": "BL0-P0-weighted-average",
                "date": summary["generated_at"],
                "commit_hash": summary["git"]["commit"],
                "status": "completed",
                "auroc": f"{test['auroc']:.6f}",
                "auprc": f"{test['auprc']:.6f}",
                "train_time": "no_training",
                "gpu_mem": summary["device"],
                "epochs": 0,
                "best_epoch": 0,
                "config_path": str(config_path),
                "notes": (
                    f"dataset={summary['dataset']['name']}; split=test; "
                    f"weight_region_r9={summary['best_weight_region_r9']:.2f}; "
                    f"run_c9_weight={1.0 - summary['best_weight_region_r9']:.2f}"
                ),
            }
        )


def write_outputs(summary: dict[str, Any], output_dir: Path, config_path: Path, append_csv: bool, experiments_csv: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    grid_path = output_dir / "val_weight_grid.csv"
    with grid_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["weight_region_r9", "auroc", "auprc", "accuracy", "precision", "recall", "f1"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary["val_weight_grid"]:
            writer.writerow({key: row[key] for key in fieldnames})

    report_lines = [
        "# P0/BL0 加权平均实验报告",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Dataset: `{summary['dataset']['name']}`",
        f"- Dataset file: `{summary['dataset']['file']}`",
        f"- Device: `{summary['device']}`",
        f"- Selected rows: `{summary['dataset']['selected_rows']}`",
        f"- observed_positive: `{summary['dataset']['observed_positive']}`",
        f"- unobserved_candidate: `{summary['dataset']['unobserved_candidate']}`",
        f"- Split sizes: train `{summary['split_sizes']['train']}`, val `{summary['split_sizes']['val']}`, test `{summary['split_sizes']['test']}`",
        "",
        "## 模型来源",
        "",
        f"- Region/R9 checkpoint: `{summary['checkpoints']['region_r9']['path']}`",
        f"- Run/C9 checkpoint: `{summary['checkpoints']['run_c9']['path']}`",
        f"- R9 cache verified: `{summary['feature_cache']['region_r9']['cache_order_ok']}`",
        f"- C9 cache verified: `{summary['feature_cache']['run_c9']['cache_order_ok']}`",
        "",
        "## 验证集调权",
        "",
        f"- Best `weight_region_r9`: `{summary['best_weight_region_r9']:.2f}`",
        f"- Best `weight_run_c9`: `{1.0 - summary['best_weight_region_r9']:.2f}`",
        f"- Val weighted metrics: `{summary['val_metrics']['weighted_average']}`",
        "",
        "## 测试集结果",
        "",
        f"- Region/R9 only: `{summary['test_metrics']['region_r9']}`",
        f"- Run/C9 only: `{summary['test_metrics']['run_c9']}`",
        f"- Weighted average: `{summary['test_metrics']['weighted_average']}`",
        "",
        "## 解释",
        "",
        "- 这个 P0/BL0 不训练新神经网络，只做两个已训练旧模块概率的验证集调权和测试集评估。",
        "- 代码和报告中继续把 `label=0` 解释为 `unobserved_candidate`，不是实验证明安全。",
    ]
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    if append_csv:
        append_experiment(experiments_csv, summary, config_path)


def run(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    runtime = configure_runtime(config)
    device = choose_device(config)
    seed = int(config.get("seed", 43))
    dataset_cfg = config["dataset"]
    sampling_cfg = config.get("sampling", {})
    split_cfg = config.get("split", {})
    eval_cfg = config.get("evaluation", {})
    cache_cfg = config.get("feature_cache", {})
    checkpoint_cfg = config["checkpoints"]

    dataset_file = Path(dataset_cfg["file"])
    with np.load(dataset_file, allow_pickle=False) as data:
        labels_all = np.asarray(data["y"], dtype=np.float32)
        selected = select_rows(
            labels_all,
            optional_cap(sampling_cfg.get("observed_positive_cap")),
            optional_cap(sampling_cfg.get("unobserved_candidate_cap")),
            seed,
        )
        labels = labels_all[selected].astype(np.float32, copy=False)
        splits = stratified_split(
            labels,
            seed,
            float(split_cfg.get("train", 0.70)),
            float(split_cfg.get("val", 0.15)),
        )
        verify_first_n = int(cache_cfg.get("verify_first_n", 64))
        region_features, region_cache_summary = load_features(Path(cache_cfg["region_r9"]), "r9", selected, data, verify_first_n)
        run_features, run_cache_summary = load_features(Path(cache_cfg["run_c9"]), "c9", selected, data, verify_first_n)

    region_model_name, region_encoder_name, region_model, region_meta = load_model(Path(checkpoint_cfg["region_r9"]))
    run_model_name, run_encoder_name, run_model, run_meta = load_model(Path(checkpoint_cfg["run_c9"]))
    if (region_model_name, region_encoder_name) != ("deepfocus", "r9"):
        raise ValueError(f"region checkpoint mismatch: {(region_model_name, region_encoder_name)}")
    if (run_model_name, run_encoder_name) != ("conmismatch9", "c9"):
        raise ValueError(f"run checkpoint mismatch: {(run_model_name, run_encoder_name)}")

    batch_size = int(eval_cfg.get("batch_size", 1024))
    threshold = float(eval_cfg.get("threshold", 0.5))
    val_labels = labels[splits.val]
    test_labels = labels[splits.test]

    region_val = predict_probabilities(region_model, region_features, splits.val, batch_size, device)
    region_test = predict_probabilities(region_model, region_features, splits.test, batch_size, device)
    run_val = predict_probabilities(run_model, run_features, splits.val, batch_size, device)
    run_test = predict_probabilities(run_model, run_features, splits.test, batch_size, device)

    grid = build_weight_grid(
        float(eval_cfg.get("weight_grid_start", 0.0)),
        float(eval_cfg.get("weight_grid_stop", 1.0)),
        float(eval_cfg.get("weight_grid_step", 0.01)),
    )
    best_weight, val_grid = tune_weight(val_labels, region_val, run_val, grid, threshold)
    weighted_val = best_weight * region_val + (1.0 - best_weight) * run_val
    weighted_test = best_weight * region_test + (1.0 - best_weight) * run_test

    summary: dict[str, Any] = {
        "generated_at": utc_now(),
        "config_path": str(config_path),
        "runtime": runtime,
        "device": str(device),
        "git": git_info(),
        "dataset": {
            "name": dataset_cfg.get("name", dataset_file.stem),
            "file": str(dataset_file),
            "total_rows": int(labels_all.shape[0]),
            "selected_rows": int(labels.shape[0]),
            "observed_positive": int((labels == 1).sum()),
            "unobserved_candidate": int((labels == 0).sum()),
            "label_semantics": dataset_cfg.get("label_semantics", {}),
        },
        "split_sizes": {
            "train": int(len(splits.train)),
            "val": int(len(splits.val)),
            "test": int(len(splits.test)),
        },
        "checkpoints": {
            "region_r9": region_meta,
            "run_c9": run_meta,
        },
        "feature_cache": {
            "region_r9": region_cache_summary,
            "run_c9": run_cache_summary,
        },
        "best_weight_region_r9": best_weight,
        "val_weight_grid": val_grid,
        "val_metrics": {
            "region_r9": metrics_for(val_labels, region_val, threshold),
            "run_c9": metrics_for(val_labels, run_val, threshold),
            "weighted_average": metrics_for(val_labels, weighted_val, threshold),
        },
        "test_metrics": {
            "region_r9": metrics_for(test_labels, region_test, threshold),
            "run_c9": metrics_for(test_labels, run_test, threshold),
            "weighted_average": metrics_for(test_labels, weighted_test, threshold),
        },
        "env": {
            "python": sys.executable,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
        },
    }

    outputs_cfg = config.get("outputs", {})
    output_dir = Path(outputs_cfg.get("dir", "results/bl0_weighted_average"))
    write_outputs(
        summary,
        output_dir,
        config_path,
        bool(outputs_cfg.get("append_experiments_csv", False)),
        Path(outputs_cfg.get("experiments_csv", "results/experiments.csv")),
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate weighted average of DeepFocus/R9 and ConMismatch9/C9 checkpoints.")
    parser.add_argument("--config", type=Path, default=Path("configs/bl0_weighted_average.yaml"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run(args.config)
    print(json.dumps(
        {
            "output_dir": load_config(args.config).get("outputs", {}).get("dir"),
            "best_weight_region_r9": summary["best_weight_region_r9"],
            "val_metrics": summary["val_metrics"]["weighted_average"],
            "test_metrics": summary["test_metrics"]["weighted_average"],
        },
        indent=2,
        ensure_ascii=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
