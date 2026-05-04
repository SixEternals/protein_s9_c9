from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from encoders.c9_encoder import C9Encoder
from encoders.r9_encoder import R9Encoder
from models.conmismatch9_torch import ConMismatch9TorchConfig, ConMismatch9TorchModel
from models.deepfocus_torch import DeepFocusTorchConfig, DeepFocusTorchModel
from utils.config import load_config, resolve_dataset_files


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(config: dict[str, Any]) -> torch.device:
    requested = str(config.get("device", "auto")).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("Requested CUDA but torch.cuda.is_available() is false; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"", "none", "null", "all"}:
            return None
    return int(value)


def _resolve_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    cleaned = str(value).strip().lower()
    if cleaned in {"1", "true", "yes", "on"}:
        return True
    if cleaned in {"0", "false", "no", "off"}:
        return False
    if cleaned == "auto":
        return default
    return default


def _safe_stem(path: str | os.PathLike[str]) -> str:
    stem = Path(path).stem
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in stem)


def encode_r9_batch(on_seq: Sequence[str], off_seq: Sequence[str]) -> np.ndarray:
    encoder = R9Encoder()
    encoded = np.empty((len(on_seq), 23, 9), dtype=np.uint8)
    for idx, (on, off) in enumerate(zip(on_seq, off_seq)):
        encoded[idx] = np.asarray(encoder.encode_pair(str(on), str(off)), dtype=np.uint8)
    return encoded


def encode_c9_batch(on_seq: Sequence[str], off_seq: Sequence[str]) -> np.ndarray:
    encoder = C9Encoder()
    encoded = np.empty((len(on_seq), 23, 9), dtype=np.uint8)
    for idx, (on, off) in enumerate(zip(on_seq, off_seq)):
        encoded[idx] = np.asarray(encoder.encode_pair(str(on), str(off)), dtype=np.uint8)
    return encoded


def sample_indices(labels: np.ndarray, positive_cap: int | None, negative_cap: int | None, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    labels = labels.astype(np.int64, copy=False)
    pos = np.flatnonzero(labels == 1)
    neg = np.flatnonzero(labels == 0)
    if positive_cap is not None and len(pos) > positive_cap:
        pos = rng.choice(pos, size=positive_cap, replace=False)
    if negative_cap is not None and len(neg) > negative_cap:
        neg = rng.choice(neg, size=negative_cap, replace=False)
    indices = np.concatenate([pos, neg])
    rng.shuffle(indices)
    return indices


def split_indices(labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = labels.astype(np.int64, copy=False)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for value in (0, 1):
        cls = np.flatnonzero(labels == value)
        rng.shuffle(cls)
        n = len(cls)
        train_end = int(round(n * 0.70))
        val_end = train_end + int(round(n * 0.15))
        train_parts.append(cls[:train_end])
        val_parts.append(cls[train_end:val_end])
        test_parts.append(cls[val_end:])
    train = np.concatenate(train_parts)
    val = np.concatenate(val_parts)
    test = np.concatenate(test_parts)
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def _cache_key(dataset_file: str, encoder_name: str, indices: np.ndarray) -> str:
    stat = Path(dataset_file).stat()
    payload = "|".join(
        [
            str(Path(dataset_file).resolve()),
            encoder_name,
            str(stat.st_size),
            str(stat.st_mtime_ns),
            str(len(indices)),
            hashlib.sha1(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest(),
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _load_or_encode_c9(
    dataset_file: str,
    on_seq: np.ndarray,
    off_seq: np.ndarray,
    indices: np.ndarray,
    cache_dir: str | None,
    cache_features: bool,
) -> tuple[np.ndarray, str]:
    if cache_features and cache_dir:
        cache_root = Path(cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_path = cache_root / f"{_safe_stem(dataset_file)}_c9_{_cache_key(dataset_file, 'c9', indices)}.npy"
        if cache_path.exists():
            return np.load(cache_path, allow_pickle=False), "cached_c9"

    encoded = encode_c9_batch(on_seq, off_seq)
    if cache_features and cache_dir:
        cache_root = Path(cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_path = cache_root / f"{_safe_stem(dataset_file)}_c9_{_cache_key(dataset_file, 'c9', indices)}.npy"
        tmp_path = cache_path.with_suffix(".tmp")
        with tmp_path.open("wb") as handle:
            np.save(handle, encoded)
        tmp_path.replace(cache_path)
    return encoded, "encoded_c9"


def _load_or_encode_r9(
    dataset_file: str,
    data: np.lib.npyio.NpzFile,
    on_seq: np.ndarray,
    off_seq: np.ndarray,
    indices: np.ndarray,
    cache_dir: str | None,
    cache_features: bool,
    use_stored_r9: bool = False,
) -> tuple[np.ndarray, str]:
    if use_stored_r9 and "X" in data:
        return np.asarray(data["X"][indices], dtype=np.uint8), "stored_x_r9"

    if cache_features and cache_dir:
        cache_root = Path(cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_path = cache_root / f"{_safe_stem(dataset_file)}_r9_{_cache_key(dataset_file, 'r9', indices)}.npy"
        if cache_path.exists():
            return np.load(cache_path, allow_pickle=False), "cached_r9"

    encoded = encode_r9_batch(on_seq, off_seq)
    if cache_features and cache_dir:
        cache_root = Path(cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_path = cache_root / f"{_safe_stem(dataset_file)}_r9_{_cache_key(dataset_file, 'r9', indices)}.npy"
        tmp_path = cache_path.with_suffix(".tmp")
        with tmp_path.open("wb") as handle:
            np.save(handle, encoded)
        tmp_path.replace(cache_path)
    return encoded, "encoded_r9"


def load_dataset(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[str], list[dict[str, Any]]]:
    dataset_files = resolve_dataset_files(config)
    if not dataset_files:
        raise ValueError("config must define dataset_files")

    model_name = str(config.get("model", "conmismatch9")).lower()
    encoder_name = str(config.get("encoder", "c9")).lower()
    sampling = config.get("sampling", {})
    positive_cap = _optional_int(sampling.get("positive_cap", 5000))
    negative_cap = _optional_int(sampling.get("negative_cap", 15000))
    cache_dir = str(config.get("feature_cache_dir", "runs/feature_cache"))
    cache_features = _resolve_bool(config.get("cache_features", True), True)
    use_stored_r9 = _resolve_bool(config.get("use_stored_r9", False), False)
    seed = int(config.get("seed", 42))

    feature_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    summaries: list[dict[str, Any]] = []

    for dataset_file in dataset_files:
        with np.load(dataset_file, allow_pickle=False) as data:
            labels_all = np.asarray(data["y"], dtype=np.float32)
            indices = sample_indices(labels_all, positive_cap, negative_cap, seed)
            labels = labels_all[indices].astype(np.float32, copy=False)

            if encoder_name == "c9":
                on_seq = np.asarray(data["on_seq"][indices], dtype=str)
                off_seq = np.asarray(data["off_seq"][indices], dtype=str)
                features, feature_source = _load_or_encode_c9(
                    dataset_file=dataset_file,
                    on_seq=on_seq,
                    off_seq=off_seq,
                    indices=indices,
                    cache_dir=cache_dir,
                    cache_features=cache_features,
                )
            elif encoder_name == "r9":
                on_seq = np.asarray(data["on_seq"][indices], dtype=str)
                off_seq = np.asarray(data["off_seq"][indices], dtype=str)
                features, feature_source = _load_or_encode_r9(
                    dataset_file=dataset_file,
                    data=data,
                    on_seq=on_seq,
                    off_seq=off_seq,
                    indices=indices,
                    cache_dir=cache_dir,
                    cache_features=cache_features,
                    use_stored_r9=use_stored_r9,
                )
            else:
                raise ValueError(f"unsupported encoder: {encoder_name}")

        feature_parts.append(np.ascontiguousarray(features, dtype=np.uint8))
        label_parts.append(np.ascontiguousarray(labels, dtype=np.float32))
        summaries.append(
            {
                "dataset_file": dataset_file,
                "total_samples": int(labels_all.shape[0]),
                "selected_samples": int(len(indices)),
                "positives": int((labels == 1).sum()),
                "negatives": int((labels == 0).sum()),
                "feature_source": feature_source,
            }
        )

    features = np.concatenate(feature_parts, axis=0)
    labels = np.concatenate(label_parts, axis=0)
    if model_name == "conmismatch9" and encoder_name != "c9":
        raise ValueError("ConMismatch9 formal training expects encoder c9")
    if model_name == "deepfocus" and encoder_name != "r9":
        raise ValueError("DeepFocus formal training expects encoder r9")
    return features, labels, dataset_files, summaries


def make_loader(
    features: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    x = torch.from_numpy(np.ascontiguousarray(features[indices]))
    y = torch.from_numpy(np.ascontiguousarray(labels[indices])).to(dtype=torch.float32)
    dataset = TensorDataset(x, y)
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **loader_kwargs)


def _autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        try:
            return torch.amp.autocast(device_type="cuda")
        except AttributeError:  # pragma: no cover - older torch fallback
            return torch.cuda.amp.autocast()
    return nullcontext()


def _make_grad_scaler(enabled: bool):
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda")
    except Exception:  # pragma: no cover - older torch fallback
        return torch.cuda.amp.GradScaler()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    all_labels: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    for x, y in loader:
        x = x.to(device=device, dtype=torch.float32, non_blocking=True)
        y = y.to(device=device, dtype=torch.float32, non_blocking=True)
        with _autocast_context(device, use_amp):
            logits = model(x)
            loss = criterion(logits, y)
        total_loss += float(loss.item()) * y.numel()
        total_count += y.numel()
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y.detach().cpu().numpy())

    labels = np.concatenate(all_labels) if all_labels else np.asarray([], dtype=np.float32)
    probs = np.concatenate(all_probs) if all_probs else np.asarray([], dtype=np.float32)
    preds = (probs >= 0.5).astype(np.int64)
    if len(np.unique(labels)) < 2:
        auroc = 0.5
        aupr = 0.0
    else:
        auroc = float(roc_auc_score(labels, probs))
        aupr = float(average_precision_score(labels, probs))
    return {
        "loss": total_loss / max(1, total_count),
        "auroc": auroc,
        "aupr": aupr,
        "acc": float(accuracy_score(labels.astype(np.int64), preds)) if labels.size else 0.0,
    }


def configure_runtime(config: dict[str, Any]) -> dict[str, Any]:
    training = config.get("training", {})
    total_cpu = os.cpu_count() or 1
    cpu_threads = _optional_int(training.get("cpu_threads"))
    if cpu_threads is None:
        cpu_threads = max(4, min(16, max(1, total_cpu // 2)))
    interop_threads = _optional_int(training.get("interop_threads"))
    if interop_threads is None:
        interop_threads = min(4, max(1, cpu_threads // 2))

    os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(interop_threads)
    except RuntimeError:
        pass

    if hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True

    return {
        "cpu_threads": cpu_threads,
        "interop_threads": interop_threads,
    }


def build_model(config: dict[str, Any]) -> tuple[str, str, nn.Module, dict[str, Any]]:
    model_name = str(config.get("model", "conmismatch9")).lower()
    training = config.get("training", {})
    hidden_dim = int(training.get("hidden_dim", 96))
    dropout = float(training.get("dropout", 0.20))
    attn_heads = int(training.get("attn_heads", 4))
    attn_layers = int(training.get("attn_layers", 2))
    ablation_mode = str(training.get("ablation_mode", config.get("ablation_mode", "full")))

    if model_name == "conmismatch9":
        model_config = ConMismatch9TorchConfig(
            hidden_dim=hidden_dim,
            dropout=dropout,
            attn_heads=attn_heads,
            attn_layers=attn_layers,
            run_base_width=int(training.get("run_base_width", 2)),
            run_state_width=int(training.get("run_state_width", 2)),
            ablation_mode=ablation_mode,
        )
        return model_name, "c9", ConMismatch9TorchModel(model_config), asdict(model_config)

    if model_name == "deepfocus":
        model_config = DeepFocusTorchConfig(
            hidden_dim=hidden_dim,
            dropout=dropout,
            attn_heads=attn_heads,
            attn_layers=attn_layers,
        )
        return model_name, "r9", DeepFocusTorchModel(model_config), asdict(model_config)

    raise ValueError(f"unsupported model: {model_name}")


def train(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config.get("seed", 42))
    set_seed(seed)
    runtime = configure_runtime(config)
    device = choose_device(config)
    model_name, expected_encoder, model, model_config = build_model(config)
    encoder_name = str(config.get("encoder", expected_encoder)).lower()
    if encoder_name != expected_encoder:
        raise ValueError(f"{model_name} expects encoder {expected_encoder}, got {encoder_name}")

    features, labels, dataset_files, dataset_summaries = load_dataset(config)
    train_idx, val_idx, test_idx = split_indices(labels, seed)

    training = config.get("training", {})
    batch_size = int(training.get("batch_size", 512))
    epochs = int(training.get("epochs", 30))
    learning_rate = float(training.get("learning_rate", 1e-3))
    weight_decay = float(training.get("weight_decay", 1e-4))
    patience = int(training.get("patience", 8))
    num_workers = int(training.get("num_workers", 4))
    use_amp = _resolve_bool(training.get("amp", "auto"), device.type == "cuda") and device.type == "cuda"

    train_loader = make_loader(features, labels, train_idx, batch_size, shuffle=True, num_workers=num_workers, pin_memory=device.type == "cuda")
    val_loader = make_loader(features, labels, val_idx, batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")
    test_loader = make_loader(features, labels, test_idx, batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")

    train_labels = labels[train_idx]
    positives = float((train_labels == 1).sum())
    negatives = float((train_labels == 0).sum())
    pos_weight_value = float(training.get("pos_weight") or (negatives / max(1.0, positives)))

    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, device=device, dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    scaler = _make_grad_scaler(use_amp)

    best_state: dict[str, torch.Tensor] | None = None
    best_val_aupr = -1.0
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float | int]] = []

    cuda_name = torch.cuda.get_device_name(device) if device.type == "cuda" and torch.cuda.is_available() else None
    print(
        f"device={device} cuda_name={cuda_name} samples={len(labels)} "
        f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} "
        f"pos_weight={pos_weight_value:.4f} amp={use_amp} cpu_threads={runtime['cpu_threads']} num_workers={num_workers}"
    )

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_count = 0
        for x, y in train_loader:
            x = x.to(device=device, dtype=torch.float32, non_blocking=True)
            y = y.to(device=device, dtype=torch.float32, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            train_loss_total += float(loss.item()) * y.numel()
            train_count += y.numel()

        val_metrics = evaluate(model, val_loader, criterion, device, use_amp)
        scheduler.step(val_metrics["aupr"])
        entry = {
            "epoch": epoch,
            "train_loss": train_loss_total / max(1, train_count),
            "val_loss": val_metrics["loss"],
            "val_auroc": val_metrics["auroc"],
            "val_aupr": val_metrics["aupr"],
            "val_acc": val_metrics["acc"],
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(entry)
        print(
            "epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            "val_auroc={val_auroc:.6f} val_aupr={val_aupr:.6f} val_acc={val_acc:.6f} lr={lr:.6g}".format(**entry),
            flush=True,
        )

        if val_metrics["aupr"] > best_val_aupr:
            best_val_aupr = val_metrics["aupr"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"early_stop epoch={epoch} best_epoch={best_epoch} best_val_aupr={best_val_aupr:.6f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, criterion, device, use_amp)
    weights_path = config.get("weights_path")
    if weights_path:
        weights = Path(weights_path)
        weights.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_name": model_name,
                "encoder_name": encoder_name,
                "model_config": model_config,
                "config": config,
                "best_epoch": best_epoch,
                "best_val_aupr": best_val_aupr,
                "test_metrics": test_metrics,
            },
            weights,
        )

    return {
        "model": model_name,
        "encoder": encoder_name,
        "mode": "torch",
        "device": str(device),
        "cuda_name": cuda_name,
        "amp": use_amp,
        "cpu_threads": runtime["cpu_threads"],
        "interop_threads": runtime["interop_threads"],
        "dataset_files": dataset_files,
        "dataset_summaries": dataset_summaries,
        "feature_shape": [int(features.shape[1]), int(features.shape[2])],
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "test_size": int(len(test_idx)),
        "best_epoch": int(best_epoch),
        "best_val_aupr": float(best_val_aupr),
        "test_metrics": test_metrics,
        "model_config": model_config,
        "history": history,
        "weights_path": str(weights_path) if weights_path else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the formal PyTorch DeepFocus or ConMismatch9 model")
    parser.add_argument("--config", required=True, help="Path to a JSON-compatible YAML config")
    parser.add_argument("--output", default=None, help="Optional path to write the training summary")
    args = parser.parse_args()

    config = load_config(args.config)
    summary = train(config)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
