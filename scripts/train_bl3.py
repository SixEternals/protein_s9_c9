"""Train BL3-hard: Region + Run prior encoding with CNN + MLP.

Uses local .npz files (e.g. GUIDE-seq_9bit.npz) which contain:
  X, y, reads, on_seq, off_seq
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from encoders.run_encoder import RegionEncoder, RunEncoder
from models.bl3_hard_prior import BL3HardPrior
from models.bl3b_seed_regression import BL3bSeedRegression
from models.bl3_5_fusion import BL35FullFusion
from models.bl4_rnafm_fusion import BL3RNAFMFusion
from utils.rnafm import load_rnafm, normalize_pair_sequence, tokenize_rnafm_sequences
from utils.config import load_config


def setup_distributed() -> dict[str, Any]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requested but CUDA unavailable")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    return {"distributed": distributed, "rank": rank, "local_rank": local_rank, "world_size": world_size}


def cleanup_distributed(distributed: bool) -> None:
    if distributed and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(dist_info: dict[str, Any]) -> bool:
    return int(dist_info.get("rank", 0)) == 0


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def metric_payload(labels: np.ndarray, probabilities: np.ndarray, loss_values: list[float]) -> dict[str, float]:
    if labels.size == 0:
        return {"loss": float("nan"), "auroc": float("nan"), "auprc": float("nan")}
    payload = {"loss": float(np.mean(loss_values)) if loss_values else float("nan")}
    labels_int = labels.astype(np.int64)
    predicted = (probabilities >= 0.5).astype(np.int64)
    if len(np.unique(labels_int)) == 2:
        payload["auroc"] = float(roc_auc_score(labels_int, probabilities))
        payload["auprc"] = float(average_precision_score(labels_int, probabilities))
    else:
        payload["auroc"] = float("nan")
        payload["auprc"] = float("nan")
    payload["accuracy"] = float(accuracy_score(labels_int, predicted))
    payload["precision"] = float(precision_score(labels_int, predicted, zero_division=0))
    payload["recall"] = float(recall_score(labels_int, predicted, zero_division=0))
    payload["f1"] = float(f1_score(labels_int, predicted, zero_division=0))
    return payload


class LocalNpzDataset(Dataset):
    """Dataset that loads pre-encoded Region + Run features from NPZ.
    Supports precomputed RNA-FM embeddings for fast training."""

    def __init__(
        self,
        npz_path: str | Path,
        split_indices: np.ndarray | None = None,
        seed: int = 42,
        weight_mode: str = "soft",
        rnafm_emb_path: str | Path | None = None,
    ):
        data = np.load(npz_path, allow_pickle=False)
        self.labels = data["y"].astype(np.float32)
        self.on_seqs = data["on_seq"]
        self.off_seqs = data["off_seq"]

        if split_indices is not None:
            self.labels = self.labels[split_indices]
            self.on_seqs = self.on_seqs[split_indices]
            self.off_seqs = self.off_seqs[split_indices]

        # Use pre-encoded features if available
        if "region_features" in data and "run_features" in data and "seed_weights" in data:
            self.region_features = data["region_features"]
            self.run_features = data["run_features"]
            self.seed_weights = data["seed_weights"]
            if split_indices is not None:
                self.region_features = self.region_features[split_indices]
                self.run_features = self.run_features[split_indices]
            print(f"[LocalNpzDataset] Loaded pre-encoded features: {self.run_features.shape}")
        else:
            # Fallback: encode on-the-fly
            region_enc = RegionEncoder(length=20)
            run_enc = RunEncoder(length=20, tau=4.0, weight_mode=weight_mode)
            self.seed_weights = run_enc.seed_weights()
            pairs = [(str(on), str(off)) for on, off in zip(self.on_seqs, self.off_seqs)]
            self.region_features = region_enc.encode_batch(pairs)
            self.run_features = run_enc.encode_batch(pairs)
            print(f"[LocalNpzDataset] Encoded on-the-fly: {self.run_features.shape}")

        # Load precomputed RNA-FM embeddings
        self.rnafm_embeddings = None
        if rnafm_emb_path is not None and Path(rnafm_emb_path).exists():
            emb_data = np.load(rnafm_emb_path, allow_pickle=False)
            self.rnafm_embeddings = emb_data["rnafm_embeddings"].astype(np.float32)
            if split_indices is not None:
                self.rnafm_embeddings = self.rnafm_embeddings[split_indices]
            print(f"[LocalNpzDataset] Loaded precomputed RNA-FM embeddings: {self.rnafm_embeddings.shape}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        region = torch.from_numpy(self.region_features[idx]).float()
        run = torch.from_numpy(self.run_features[idx]).float()
        sw = torch.from_numpy(self.seed_weights).float()
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        if self.rnafm_embeddings is not None:
            rnafm_emb = torch.from_numpy(self.rnafm_embeddings[idx]).float()
            return rnafm_emb, region, run, sw, label
        return region, run, sw, label, str(self.on_seqs[idx]), str(self.off_seqs[idx])


def _to_device(batch, device: torch.device):
    """Move a batch to device. Handles both old and new batch formats."""
    if len(batch) == 6:
        # Legacy format: (region, run, sw, label, on_seq, off_seq) from default collate
        regions, runs, sws, labels, _on, _off = batch
        sw = sws[0] if sws.ndim == 2 else sws
        return None, regions.to(device), runs.to(device), sw.to(device), labels.to(device)
    elif len(batch) == 5:
        a, b, c, d, e = batch
        # Check if first element is RNA-FM embedding (float, shape ends with 640)
        if a.dtype == torch.float32 and a.dim() == 2 and a.shape[1] == 640:
            # Precomputed: (rnafm_emb, region, run, sw, label)
            sw = d[0] if d.ndim == 2 else d
            return a.to(device), b.to(device), c.to(device), sw.to(device), e.to(device)
        # Real-time: (tokens, region, run, sw, label) from custom collate
        sw = d[0] if d.ndim == 2 else d
        return a.to(device), b.to(device), c.to(device), sw.to(device), e.to(device)
    else:
        region, run, sw, labels = batch
        return None, region.to(device), run.to(device), sw.to(device), labels.to(device)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    labels_all: list[np.ndarray] = []
    probs_all: list[np.ndarray] = []
    losses: list[float] = []
    for batch in loader:
        tokens, region, run, sw, labels = _to_device(batch, device)
        if tokens is not None:
            logits = model(tokens, run, sw)
        else:
            logits = model(region, run, sw)
        loss = F.binary_cross_entropy_with_logits(logits.squeeze(-1), labels)
        losses.append(float(loss.item()))
        labels_all.append(labels.detach().cpu().numpy())
        probs_all.append(torch.sigmoid(logits.squeeze(-1)).detach().cpu().numpy())
    labels_np = np.concatenate(labels_all)
    probs_np = np.concatenate(probs_all)
    return metric_payload(labels_np, probs_np, losses)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float | None,
    pos_weight: torch.Tensor | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0
    for batch in loader:
        tokens, region, run, sw, labels = _to_device(batch, device)
        optimizer.zero_grad()
        if tokens is not None:
            logits = model(tokens, run, sw).squeeze(-1)
        else:
            logits = model(region, run, sw).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += float(loss.item())
        num_batches += 1
    return total_loss / max(num_batches, 1)


def make_split(
    labels: np.ndarray,
    seed: int,
    group_labels: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    indices = np.arange(len(labels), dtype=np.int64)
    if group_labels is not None:
        # Group-safe split: same group never appears in two splits
        unique_groups = np.unique(group_labels)
        rng = np.random.default_rng(seed)
        rng.shuffle(unique_groups)
        group_counts = {g: int(np.sum(group_labels == g)) for g in unique_groups}
        total = len(labels)
        train_end = int(total * 0.70)
        val_end = int(total * 0.85)
        train_groups, val_groups, test_groups = [], [], []
        cumsum = 0
        for g in unique_groups:
            count = group_counts[g]
            if cumsum < train_end:
                train_groups.append(g)
            elif cumsum < val_end:
                val_groups.append(g)
            else:
                test_groups.append(g)
            cumsum += count
        train_idx = indices[np.isin(group_labels, train_groups)]
        val_idx = indices[np.isin(group_labels, val_groups)]
        test_idx = indices[np.isin(group_labels, test_groups)]
        return {"train": train_idx, "val": val_idx, "test": test_idx}
    # Fallback: stratified random split
    train_idx, holdout_idx = train_test_split(indices, test_size=0.3, random_state=seed, stratify=labels)
    val_idx, test_idx = train_test_split(holdout_idx, test_size=0.5, random_state=seed + 1, stratify=labels[holdout_idx])
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    dist_info = setup_distributed()
    device = torch.device(f"cuda:{dist_info['local_rank']}" if torch.cuda.is_available() else "cpu")
    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset_cfg = config["dataset"]
    npz_path = dataset_cfg["file"]
    training_cfg = config.get("training", {})
    model_cfg = config.get("model_config", {})
    output_dir = Path(args.output_dir or config.get("outputs", {}).get("dir", "results/bl3_hard"))
    if is_main_process(dist_info):
        output_dir.mkdir(parents=True, exist_ok=True)

    # Load NPZ and split
    with np.load(npz_path, allow_pickle=False) as data:
        labels_all = data["y"].astype(np.int64)
    group_csv = dataset_cfg.get("group_csv")
    group_col = dataset_cfg.get("group_column", "sgRNA_type")
    if group_csv:
        import pandas as pd
        df = pd.read_csv(group_csv, usecols=[group_col])
        group_labels = df[group_col].values
        split_indices = make_split(labels_all, seed, group_labels)
    else:
        split_indices = make_split(labels_all, seed)

    weight_mode = config.get("run_encoder", {}).get("weight_mode", "soft")
    use_region = config.get("model_config", {}).get("use_region", True)
    use_run = config.get("model_config", {}).get("use_run", True)

    model_name = config.get("model", {}).get("name", "BL3HardPrior")

    # Check for precomputed RNA-FM embeddings
    rnafm_emb_path = config.get("rnafm", {}).get("embeddings_path")
    use_precomputed_rnafm = rnafm_emb_path is not None and Path(rnafm_emb_path).exists()

    train_dataset = LocalNpzDataset(npz_path, split_indices["train"], seed, weight_mode=weight_mode, rnafm_emb_path=rnafm_emb_path)
    val_dataset = LocalNpzDataset(npz_path, split_indices["val"], seed, weight_mode=weight_mode, rnafm_emb_path=rnafm_emb_path)
    test_dataset = LocalNpzDataset(npz_path, split_indices["test"], seed, weight_mode=weight_mode, rnafm_emb_path=rnafm_emb_path)

    batch_size = int(training_cfg.get("batch_size", 128))
    num_workers = int(training_cfg.get("num_workers", 0))

    # Setup collate_fn for RNA-FM models (real-time mode only)
    collate_fn = None
    rnafm_model_raw = None
    alphabet = None
    if model_name == "BL3RNAFMFusion" and not use_precomputed_rnafm:
        rnafm_cfg = dict(config.get("rnafm", {}))
        rnafm_checkpoint = rnafm_cfg.get("checkpoint_path")
        rnafm_model_raw, alphabet = load_rnafm(rnafm_checkpoint, trust_local_checkpoint=True)
        rnafm_model_raw = rnafm_model_raw.to(device)

        def _collate(batch):
            regions, runs, sws, labels, on_seqs, off_seqs = zip(*batch)
            sequences = [normalize_pair_sequence(on, off) for on, off in zip(on_seqs, off_seqs)]
            tokens = tokenize_rnafm_sequences(alphabet, sequences)
            return (
                tokens,
                torch.stack(regions),
                torch.stack(runs),
                sws[0],
                torch.stack(labels),
            )
        collate_fn = _collate

    if dist_info["distributed"]:
        train_sampler = DistributedSampler(train_dataset, num_replicas=dist_info["world_size"], rank=dist_info["rank"], shuffle=True, seed=seed)
        val_sampler = DistributedSampler(val_dataset, num_replicas=dist_info["world_size"], rank=dist_info["rank"], shuffle=False)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, sampler=val_sampler, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)

    test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)

    if model_name == "BL3bSeedRegression":
        model = BL3bSeedRegression(model_cfg).to(device)
    elif model_name == "BL35FullFusion":
        model = BL35FullFusion(model_cfg).to(device)
    elif model_name == "BL3RNAFMFusion":
        model = BL3RNAFMFusion(
            rnafm_model=rnafm_model_raw,
            padding_idx=alphabet.padding_idx if alphabet else 0,
            config=model_cfg,
        ).to(device)
    else:
        model = BL3HardPrior(model_cfg).to(device)
    if dist_info["distributed"]:
        model = DistributedDataParallel(model, device_ids=[dist_info["local_rank"]], output_device=dist_info["local_rank"], find_unused_parameters=True)

    # Class weighting for imbalanced data
    train_labels = train_dataset.labels
    pos_count = float(train_labels.sum())
    neg_count = float(len(train_labels) - pos_count)
    pos_weight_val = neg_count / max(pos_count, 1.0)
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
    print(f"[Training] pos_weight={pos_weight_val:.2f} (pos={pos_count:.0f}, neg={neg_count:.0f})")

    lr = float(training_cfg.get("learning_rate", 1e-3))
    weight_decay = float(training_cfg.get("weight_decay", 1e-5))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ReduceLROnPlateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    epochs = int(training_cfg.get("epochs", 50))
    grad_clip = training_cfg.get("gradient_clip", None)
    if grad_clip is not None:
        grad_clip = float(grad_clip)

    best_metric = -float("inf")
    best_epoch = 0
    monitor = str(training_cfg.get("monitor", "auprc"))
    monitor_mode = str(training_cfg.get("monitor_mode", "max"))

    epoch_metrics = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        if dist_info["distributed"]:
            train_sampler.set_epoch(epoch)

        train_loss = train_one_epoch(model, train_loader, optimizer, device, grad_clip, pos_weight)
        val_metrics = evaluate(model, val_loader, device)

        if is_main_process(dist_info):
            # LR scheduler step
            val_monitor = val_metrics.get(monitor, float("nan"))
            if not math.isnan(val_monitor):
                scheduler.step(val_monitor)

            row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
            epoch_metrics.append(row)
            print(f"epoch={epoch} train_loss={train_loss:.6f} val_auroc={val_metrics.get('auroc', float('nan')):.6f} val_auprc={val_metrics.get('auprc', float('nan')):.6f} lr={optimizer.param_groups[0]['lr']:.2e}")

            current = val_metrics.get(monitor, float("nan"))
            is_best = (monitor_mode == "max" and current > best_metric) or (monitor_mode == "min" and current < best_metric)
            if is_best and not math.isnan(current):
                best_metric = current
                best_epoch = epoch
                (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
                ckpt = {
                    "model_state_dict": model.module.state_dict() if dist_info["distributed"] else model.state_dict(),
                    "epoch": epoch,
                    "metrics": val_metrics,
                    "config": config,
                }
                torch.save(ckpt, output_dir / "checkpoints" / "best.pt")

    # Final test: AGENTS.md constraint #6 — must load best.pt, not last.pt
    best_ckpt_path = output_dir / "checkpoints" / "best.pt"
    if best_ckpt_path.exists():
        best_ckpt = torch.load(best_ckpt_path, map_location=device)
        if dist_info["distributed"]:
            model.module.load_state_dict(best_ckpt["model_state_dict"])
        else:
            model.load_state_dict(best_ckpt["model_state_dict"])
        if is_main_process(dist_info):
            print(f"[Test] Loaded best checkpoint from epoch {best_ckpt['epoch']}")

    test_metrics = evaluate(model, test_loader, device)
    if is_main_process(dist_info):
        print(f"\nBest epoch={best_epoch} val_{monitor}={best_metric:.6f}")
        print(f"Test metrics: {test_metrics}")

        # Save outputs
        (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        last_ckpt = {
            "model_state_dict": model.module.state_dict() if dist_info["distributed"] else model.state_dict(),
            "epoch": epochs,
            "metrics": test_metrics,
            "config": config,
        }
        torch.save(last_ckpt, output_dir / "checkpoints" / "last.pt")

        with open(output_dir / "epoch_metrics.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=epoch_metrics[0].keys())
            writer.writeheader()
            writer.writerows(epoch_metrics)

        summary = {
            "version": config.get("version", "BL3-hard-B"),
            "status": "completed",
            "generated_at": utc_now(),
            "commit_hash": git_hash(),
            "device": str(device),
            "distributed": dist_info["distributed"],
            "train_seconds": time.time() - start_time,
            "best_epoch": best_epoch,
            "best_metric_name": monitor,
            "best_metric_value": best_metric,
            "test_metrics": test_metrics,
        }
        with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    cleanup_distributed(dist_info["distributed"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
