#!/usr/bin/env python3
"""Train BL5-3 fine-tuned RNA-FM + Run dynamic fusion baseline.

AGENTS.md compliance: [use_rnafm=True, freeze_rnafm=False,
                       split_mode=sgrna_safe, pos_weight=None,
                       focal_loss=True]
确认本文件遵守 AGENTS.md 约束：BL5-3 使用 Cross-Attn + Gated Fusion，
Run 编码只覆盖 positions 1-20，test 评估只加载 best.pt。
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
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from encoders.learnable_run_encoder import encode_base_pair_indices
from encoders.pam_encoder import encode_pam_onehot
from models.bl5_dynamic_fusion import BL5RunOnlyDynamicFusion
from utils.config import load_config
from utils.guardrails import check_eval_procedure, check_model_config, report_metrics
from utils.rnafm import load_rnafm, normalize_pair_sequence, tokenize_rnafm_sequences


def setup_distributed() -> dict[str, Any]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    cuda = torch.cuda.is_available()
    if distributed:
        backend = "nccl" if cuda else "gloo"
        if cuda:
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend)
    return {
        "distributed": distributed,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "cuda": cuda,
    }


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


def make_split(labels: np.ndarray, seed: int, group_labels: np.ndarray | None) -> dict[str, np.ndarray]:
    indices = np.arange(len(labels), dtype=np.int64)
    if group_labels is None:
        train_idx, holdout_idx = train_test_split(indices, test_size=0.3, random_state=seed, stratify=labels)
        val_idx, test_idx = train_test_split(
            holdout_idx, test_size=0.5, random_state=seed + 1, stratify=labels[holdout_idx]
        )
        return {"train": train_idx, "val": val_idx, "test": test_idx}

    unique_groups = np.unique(group_labels)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)
    group_counts = {g: int(np.sum(group_labels == g)) for g in unique_groups}
    train_end = int(len(labels) * 0.70)
    val_end = int(len(labels) * 0.85)
    train_groups: list[Any] = []
    val_groups: list[Any] = []
    test_groups: list[Any] = []
    cumsum = 0
    for group in unique_groups:
        count = group_counts[group]
        if cumsum < train_end:
            train_groups.append(group)
        elif cumsum < val_end:
            val_groups.append(group)
        else:
            test_groups.append(group)
        cumsum += count
    return {
        "train": indices[np.isin(group_labels, train_groups)],
        "val": indices[np.isin(group_labels, val_groups)],
        "test": indices[np.isin(group_labels, test_groups)],
    }


def formal_group_json_split(
    labels: np.ndarray,
    group_labels: np.ndarray,
    split_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Load a pre-computed formal split and assign indices by group membership."""
    import pandas as pd

    split_path = Path(split_cfg["formal_split_json"])
    if not split_path.exists():
        raise FileNotFoundError(f"formal split JSON not found: {split_path}")
    payload = json.loads(split_path.read_text(encoding="utf-8"))

    groups_by_split = {
        name: set(str(item) for item in payload["splits"][name]["sgRNA_types"])
        for name in ("train", "val", "test")
    }

    overlaps = {
        "train_val": sorted(groups_by_split["train"] & groups_by_split["val"]),
        "train_test": sorted(groups_by_split["train"] & groups_by_split["test"]),
        "val_test": sorted(groups_by_split["val"] & groups_by_split["test"]),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"formal split group leakage detected: {overlaps}")

    indices = np.arange(len(labels), dtype=np.int64)
    group_str = group_labels.astype(str)
    split_indices = {
        name: indices[np.isin(group_str, sorted(groups_by_split[name]))]
        for name in ("train", "val", "test")
    }

    assigned = sum(len(v) for v in split_indices.values())
    if assigned != len(labels):
        raise ValueError(f"formal split assigned {assigned} rows but data has {len(labels)} rows")

    for name, idx in split_indices.items():
        expected = payload["splits"][name]
        pos = int((labels[idx] == 1).sum())
        neg = int((labels[idx] == 0).sum())
        if int(expected["samples"]) != len(idx):
            raise ValueError(f"{name} row count mismatch against formal split JSON")
        if int(expected["observed_positive"]) != pos:
            raise ValueError(f"{name} observed_positive mismatch against formal split JSON")
        if int(expected["unobserved_candidate"]) != neg:
            raise ValueError(f"{name} unobserved_candidate mismatch against formal split JSON")

    return {
        "train": split_indices["train"],
        "val": split_indices["val"],
        "test": split_indices["test"],
        "metadata": {
            "split_source": "formal_group_json",
            "formal_group_json_path": str(split_path),
            "source_version": payload.get("version"),
            "seed": payload.get("seed"),
            "split_mode": payload.get("split_mode"),
            "group_column": split_cfg.get("group_column", "sgRNA_type"),
            "leakage_safe_by_sgrna_type": True,
            "group_counts": {
                name: int(payload["splits"][name]["sgRNA_type_count"])
                for name in ("train", "val", "test")
            },
            "groups": {name: sorted(groups_by_split[name]) for name in ("train", "val", "test")},
            "label_counts": {
                name: {
                    "samples": int(payload["splits"][name]["samples"]),
                    "observed_positive": int(payload["splits"][name]["observed_positive"]),
                    "unobserved_candidate": int(payload["splits"][name]["unobserved_candidate"]),
                    "positive_ratio": float(payload["splits"][name]["positive_ratio"]),
                }
                for name in ("train", "val", "test")
            },
            "leakage_check": {k: v for k, v in overlaps.items()},
        },
    }


def apply_pam_holdout_split(
    split_indices: dict[str, np.ndarray],
    holdout_split_dir: str | Path,
) -> dict[str, np.ndarray]:
    """Apply PAM holdout split masks to formal split indices.

    Reads split_indices.npz produced by scripts/build_pam_strict_holdout_split.py
    and intersects each formal split with the corresponding holdout mask.
    """
    holdout_dir = Path(holdout_split_dir)
    npz_path = holdout_dir / "split_indices.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Holdout split indices not found: {npz_path}")
    npz = np.load(npz_path, allow_pickle=False)

    name_map = {
        "train": "train_H",
        "val": "val_H",
        "test": "test_H",
    }

    result: dict[str, np.ndarray] = {}
    for split_name, npz_key in name_map.items():
        if npz_key not in npz:
            raise KeyError(f"Expected key {npz_key} in {npz_path}")
        holdout_mask = npz[npz_key].astype(bool)
        if len(holdout_mask) == 0:
            raise ValueError(f"Holdout mask {npz_key} is empty")
        holdout_indices = np.nonzero(holdout_mask)[0].astype(np.int64)
        result[split_name] = np.intersect1d(
            split_indices[split_name], holdout_indices, assume_unique=True
        )
        if len(result[split_name]) == 0:
            raise ValueError(
                f"Holdout split {split_name} is empty after intersecting with formal split"
            )

    return result


def enforce_run_states_20nt(run_features: np.ndarray) -> np.ndarray:
    """Recompute C9 run-state bits inside positions 1-20 only.

    Existing caches may have been produced by a 23nt C9 helper and then sliced.
    This fixes the last two state bits from the 20nt mismatch mask without
    touching base/event bits.
    """
    if run_features.ndim != 3 or run_features.shape[1:] != (20, 9):
        raise ValueError(f"run_features must have shape (N, 20, 9), got {run_features.shape}")

    mismatch = (run_features[:, :, 5] == 1) & (run_features[:, :, 6] == 0)
    left = np.zeros(mismatch.shape, dtype=np.int8)
    right = np.zeros(mismatch.shape, dtype=np.int8)
    for pos in range(20):
        prev = left[:, pos - 1] if pos else 0
        left[:, pos] = np.where(mismatch[:, pos], prev + 1, 0)
    for pos in range(19, -1, -1):
        nxt = right[:, pos + 1] if pos < 19 else 0
        right[:, pos] = np.where(mismatch[:, pos], nxt + 1, 0)

    run_len = left + right - 1
    state = np.zeros(mismatch.shape, dtype=np.int8)
    state[(mismatch) & (run_len == 1)] = 1
    state[(mismatch) & (run_len == 2)] = 2
    state[(mismatch) & (run_len >= 3)] = 3

    run_features[:, :, 7] = ((state == 2) | (state == 3)).astype(np.float32)
    run_features[:, :, 8] = ((state == 1) | (state == 3)).astype(np.float32)
    return run_features


class BL5Arrays:
    def __init__(self, config: dict[str, Any]):
        model_cfg = config.get("model", {})
        self.use_learnable_run = bool(model_cfg.get("use_learnable_run", False))
        self.use_run = bool(model_cfg.get("use_run", True))
        data_cfg = config.get("data", {})
        npz_value = data_cfg.get("npz_path")
        if not npz_value:
            raise ValueError("BL5 config requires data.npz_path")
        npz_path = Path(npz_value)
        if not npz_path.exists():
            raise FileNotFoundError(f"BL5 NPZ not found: {npz_path}")
        npz = np.load(npz_path, allow_pickle=False)
        self.labels = npz["y"].astype(np.float32, copy=False)
        self.on_seqs = npz["on_seq"]
        self.off_seqs = npz["off_seq"]
        if self.use_learnable_run or not self.use_run:
            self.run_features = None
            self.seed_weights = np.empty((0,), dtype=np.float32)
        else:
            self.run_features = npz["run_features"].astype(np.float32, copy=True)
            enforce_run_states_20nt(self.run_features)
            self.seed_weights = npz["seed_weights"].astype(np.float32, copy=False)


class BL5Dataset(Dataset):
    def __init__(
        self,
        arrays: BL5Arrays,
        indices: np.ndarray,
        pam_shuffle_indices: np.ndarray | None = None,
    ):
        self.arrays = arrays
        self.indices = indices.astype(np.int64)
        self.pam_shuffle_indices = pam_shuffle_indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        row = int(self.indices[idx])
        if self.arrays.use_learnable_run or not getattr(self.arrays, "use_run", True):
            run = torch.empty(0, dtype=torch.long)
            sw = torch.empty(0, dtype=torch.float32)
        else:
            run = torch.from_numpy(self.arrays.run_features[row]).float()
            sw = torch.from_numpy(self.arrays.seed_weights).float()
        label = torch.tensor(self.arrays.labels[row], dtype=torch.float32)
        on_seq = str(self.arrays.on_seqs[row])
        if self.pam_shuffle_indices is not None:
            off_row = int(self.pam_shuffle_indices[idx])
            off_seq = str(self.arrays.off_seqs[off_row])
        else:
            off_seq = str(self.arrays.off_seqs[row])
        return run, sw, label, on_seq, off_seq


class SequentialDistributedSampler(Sampler[int]):
    def __init__(self, dataset: Dataset, rank: int, world_size: int):
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self) -> int:
        return max((len(self.dataset) + self.world_size - 1 - self.rank) // self.world_size, 0)


def make_live_collate(
    alphabet,
    *,
    use_rnafm: bool = True,
    use_run: bool = True,
    use_learnable_run: bool = False,
    use_pam_encoder: bool = False,
    shuffle_pam: bool = False,
    shuffle_pam_mode: str = "batch",
):
    def _collate(batch):
        runs, sws, labels, on_seqs, off_seqs = zip(*batch)
        if use_rnafm:
            sequences = [normalize_pair_sequence(on, off) for on, off in zip(on_seqs, off_seqs)]
            tokens = tokenize_rnafm_sequences(alphabet, sequences)
        else:
            tokens = torch.empty(0, dtype=torch.long)
        if not use_run:
            run_input = torch.empty(0, dtype=torch.float32)
            seed_weights = torch.empty(0, dtype=torch.float32)
        elif use_learnable_run:
            run_input = encode_base_pair_indices(on_seqs, off_seqs)
            seed_weights = torch.empty(0, dtype=torch.float32)
        else:
            run_input = torch.stack(runs)
            seed_weights = sws[0]
        if use_pam_encoder:
            pam_input = encode_pam_onehot(off_seqs)
            if shuffle_pam and shuffle_pam_mode == "batch":
                perm = torch.randperm(pam_input.size(0))
                pam_input = pam_input[perm]
        else:
            pam_input = torch.empty(0, dtype=torch.float32)
        return tokens, run_input, seed_weights, pam_input, torch.stack(labels)

    return _collate


def to_device(batch, device: torch.device):
    if len(batch) == 4:
        tokens_or_emb, run_input, sw, labels = batch
        pam_input = torch.empty(0, dtype=torch.float32)
    elif len(batch) == 5:
        tokens_or_emb, run_input, sw, pam_input, labels = batch
    else:
        raise ValueError(f"Unexpected BL5 batch width: {len(batch)}")
    return (
        tokens_or_emb.to(device),
        run_input.to(device),
        sw.to(device),
        pam_input.to(device),
        labels.to(device),
    )


def focal_loss_with_logits(logits: torch.Tensor, labels: torch.Tensor, gamma: float) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    probs = torch.sigmoid(logits)
    p_t = probs * labels + (1.0 - probs) * (1.0 - labels)
    return (((1.0 - p_t).clamp_min(1e-6) ** gamma) * bce).mean()


def loss_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    focal_loss: bool,
    focal_gamma: float,
    pos_weight: torch.Tensor | None,
) -> torch.Tensor:
    logits = logits.squeeze(-1)
    if focal_loss:
        return focal_loss_with_logits(logits, labels, focal_gamma)
    return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)


def metric_payload(labels: np.ndarray, probabilities: np.ndarray, losses: list[float]) -> dict[str, float]:
    finite_mask = np.isfinite(probabilities)
    nonfinite_count = int((~finite_mask).sum())
    total_count = int(probabilities.shape[0])
    if nonfinite_count:
        labels = labels[finite_mask]
        probabilities = probabilities[finite_mask]

    labels_int = labels.astype(np.int64)
    finite_losses = [loss for loss in losses if math.isfinite(loss)]
    payload = {
        "loss": float(np.mean(finite_losses)) if finite_losses else float("nan"),
        "nonfinite_prob_count": float(nonfinite_count),
        "nonfinite_prob_fraction": float(nonfinite_count / max(total_count, 1)),
    }
    if len(labels_int) == 0:
        payload.update(
            {
                "auroc": float("nan"),
                "auprc": float("nan"),
                "accuracy": float("nan"),
                "precision": float("nan"),
                "recall": float("nan"),
                "f1": float("nan"),
            }
        )
        return payload

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


def gather_numpy(array: np.ndarray, dist_info: dict[str, Any]) -> np.ndarray:
    if not dist_info["distributed"]:
        return array
    gathered: list[np.ndarray | None] = [None for _ in range(dist_info["world_size"])]
    dist.all_gather_object(gathered, array)
    return np.concatenate([item for item in gathered if item is not None], axis=0)


def restore_sequential_distributed_order(
    gathered: np.ndarray,
    dataset_len: int,
    world_size: int,
) -> np.ndarray:
    """Restore output order from SequentialDistributedSampler rank-concat order."""
    if world_size <= 1 or len(gathered) != dataset_len:
        return gathered
    positions = np.concatenate(
        [np.arange(rank, dataset_len, world_size, dtype=np.int64) for rank in range(world_size)]
    )
    restored = np.empty_like(gathered)
    restored[positions] = gathered
    return restored


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    dist_info: dict[str, Any],
    *,
    focal_loss: bool,
    focal_gamma: float,
    pos_weight: torch.Tensor | None,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    labels_all: list[np.ndarray] = []
    probs_all: list[np.ndarray] = []
    losses: list[float] = []
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        tokens_or_emb, runs, sw, pam_input, labels = to_device(batch, device)
        logits = model(tokens_or_emb, runs, sw, pam_input)
        loss = loss_from_logits(
            logits, labels, focal_loss=focal_loss, focal_gamma=focal_gamma, pos_weight=pos_weight
        )
        losses.append(float(loss.item()))
        labels_all.append(labels.detach().cpu().numpy())
        probs_all.append(torch.sigmoid(logits.squeeze(-1)).detach().cpu().numpy())

    labels_np = gather_numpy(np.concatenate(labels_all), dist_info)
    probs_np = gather_numpy(np.concatenate(probs_all), dist_info)
    losses_np = gather_numpy(np.asarray(losses, dtype=np.float32), dist_info)
    return metric_payload(labels_np, probs_np, [float(x) for x in losses_np])


@torch.no_grad()
def predict_probabilities(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    dist_info: dict[str, Any],
    max_batches: int | None = None,
) -> np.ndarray:
    model.eval()
    probs_all: list[np.ndarray] = []
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        tokens_or_emb, runs, sw, pam_input, _labels = to_device(batch, device)
        logits = model(tokens_or_emb, runs, sw, pam_input)
        probs_all.append(torch.sigmoid(logits.squeeze(-1)).detach().cpu().numpy())
    probs_np = gather_numpy(np.concatenate(probs_all), dist_info)
    return probs_np


def write_test_predictions(
    csv_path: str,
    test_indices: np.ndarray,
    probabilities: np.ndarray,
    output_path: Path,
    *,
    pam_shuffle_indices: np.ndarray | None = None,
    split_name: str = "test",
) -> None:
    import pandas as pd

    cols = ["sgRNA_type", "sgRNA_seq", "off_seq", "label"]
    df = pd.read_csv(csv_path, usecols=lambda c: c in cols or c == "Direction")
    test_df = df.iloc[test_indices].reset_index(drop=True)
    if len(test_df) != len(probabilities):
        raise ValueError(f"prediction count mismatch: rows={len(test_df)} probs={len(probabilities)}")
    if pam_shuffle_indices is not None and len(pam_shuffle_indices) != len(test_indices):
        raise ValueError(
            "pam shuffle index count mismatch: "
            f"rows={len(test_indices)} shuffle={len(pam_shuffle_indices)}"
        )
    shuffle_df = (
        df.iloc[pam_shuffle_indices].reset_index(drop=True)
        if pam_shuffle_indices is not None
        else test_df
    )
    pam_original = [str(seq)[20:23] for seq in test_df["off_seq"].astype(str)]
    pam_shuffled = [str(seq)[20:23] for seq in shuffle_df["off_seq"].astype(str)]
    out = pd.DataFrame(
        {
            "sample_index": test_indices.astype(np.int64),
            "sgRNA_type": test_df["sgRNA_type"].astype(str),
            "on_seq": test_df["sgRNA_seq"].astype(str),
            "off_seq": test_df["off_seq"].astype(str),
            "PAM_original": pam_original,
            "PAM_shuffled": pam_shuffled,
            "PAM": pam_original,
            "label": test_df["label"].astype(int),
            "probability": probabilities.astype(np.float32),
            "Direction": test_df["Direction"].astype(str) if "Direction" in test_df.columns else "",
            "split": split_name,
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    *,
    use_amp: bool,
    grad_clip: float | None,
    focal_loss: bool,
    focal_gamma: float,
    pos_weight: torch.Tensor | None,
    max_batches: int | None = None,
    gate_l2_lambda: float = 0.0,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        tokens_or_emb, runs, sw, pam_input, labels = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            if gate_l2_lambda > 0.0:
                logits, aux = model(tokens_or_emb, runs, sw, pam_input, return_aux=True)
            else:
                logits = model(tokens_or_emb, runs, sw, pam_input)
                aux = {}
            loss = loss_from_logits(
                logits, labels, focal_loss=focal_loss, focal_gamma=focal_gamma, pos_weight=pos_weight
            )
            gate_weights = aux.get("gate_weights") if isinstance(aux, dict) else None
            if gate_l2_lambda > 0.0 and gate_weights is not None:
                loss = loss + gate_l2_lambda * gate_weights.square().sum(dim=-1).mean()
        scaler.scale(loss).backward()
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.item())
        num_batches += 1
    return total_loss / max(num_batches, 1)


def build_optimizer(model: BL5RunOnlyDynamicFusion, config: dict[str, Any]) -> torch.optim.Optimizer:
    training_cfg = config.get("training", {})
    weight_decay = float(training_cfg.get("weight_decay", 1e-5))
    if training_cfg.get("use_param_groups", False):
        rnafm_model = getattr(model, "rnafm_model", None)
        rnafm_params = (
            [p for p in rnafm_model.parameters() if p.requires_grad]
            if rnafm_model is not None
            else []
        )
        rnafm_ids = {id(p) for p in rnafm_params}
        run_encoder = getattr(model, "run_encoder", None)
        run_params = (
            [p for p in run_encoder.parameters() if p.requires_grad]
            if run_encoder is not None
            else []
        )
        run_ids = {id(p) for p in run_params}
        pam_encoder = getattr(model, "pam_encoder", None)
        pam_params = (
            [p for p in pam_encoder.parameters() if p.requires_grad]
            if pam_encoder is not None
            else []
        )
        pam_ids = {id(p) for p in pam_params}
        fusion_backend = getattr(model, "fusion_backend", None)
        attn_params = (
            [p for p in fusion_backend.parameters() if p.requires_grad]
            if fusion_backend is not None
            else []
        )
        attn_ids = {id(p) for p in attn_params}
        head_params = [
            p for p in model.parameters()
            if (
                p.requires_grad
                and id(p) not in rnafm_ids
                and id(p) not in run_ids
                and id(p) not in pam_ids
                and id(p) not in attn_ids
            )
        ]
        groups = []
        if rnafm_params:
            groups.append({"params": rnafm_params, "lr": float(training_cfg.get("lr_transformer", 5e-4))})
        if run_params:
            groups.append(
                {
                    "params": run_params,
                    "lr": float(training_cfg.get("lr_run_encoder", training_cfg.get("lr_mlp", 1e-3))),
                }
            )
        if pam_params:
            groups.append(
                {
                    "params": pam_params,
                    "lr": float(training_cfg.get("lr_pam_encoder", training_cfg.get("lr_mlp", 1e-3))),
                }
            )
        if attn_params:
            groups.append({"params": attn_params, "lr": float(training_cfg.get("lr_attn", 1e-3))})
        if head_params:
            groups.append({"params": head_params, "lr": float(training_cfg.get("lr_mlp", training_cfg.get("lr", 1e-3)))})
        return torch.optim.AdamW(groups, weight_decay=weight_decay)
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=float(training_cfg.get("lr", 1e-3)), weight_decay=weight_decay)


def write_report(output_dir: Path, summary: dict[str, Any], config_path: str) -> None:
    metrics = summary["test_metrics"]
    lines = [
        f"# {summary['version']} Report",
        "",
        f"- Status: {summary['status']}",
        f"- Config: `{config_path}`",
        f"- Split mode: `{summary['split_mode']}`",
        f"- Checkpoint: `best.pt` (epoch {summary['best_epoch']})",
        f"- AUROC: {metrics.get('auroc', float('nan')):.6f}",
        f"- AUPRC: {metrics.get('auprc', float('nan')):.6f}",
        f"- Train seconds: {summary['train_seconds']:.1f}",
        f"- Device: `{summary['device']}`",
        "",
        summary["notes"],
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_experiment(output_csv: Path, summary: dict[str, Any], config_path: str) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
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
    exists = output_csv.exists()
    metrics = summary["test_metrics"]
    row = {
        "version": summary["version"],
        "date": summary["generated_at"],
        "commit_hash": summary["commit_hash"],
        "status": summary["status"],
        "auroc": f"{metrics.get('auroc', float('nan')):.6f}",
        "auprc": f"{metrics.get('auprc', float('nan')):.6f}",
        "train_time": f"{summary['train_seconds'] / 60:.1f}m",
        "gpu_mem": summary["gpu_mem"],
        "epochs": summary["epochs"],
        "best_epoch": summary["best_epoch"],
        "config_path": config_path,
        "notes": summary["notes"],
    }
    with output_csv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train BL5-3 fine-tuned RNA-FM + Run dynamic fusion.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Load best.pt and run test evaluation without training.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path for --eval-only; defaults to output_dir/checkpoints/best.pt.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    check_model_config(config)
    dist_info = setup_distributed()
    device = torch.device(
        f"cuda:{dist_info['local_rank']}" if torch.cuda.is_available() else "cpu"
    )

    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    model_cfg = config.get("model", {})
    rnafm_cfg = config.get("rnafm", {})
    training_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    output_dir = Path(args.output_dir or config.get("output_dir", f"results/{config.get('version', 'bl5')}"))
    if is_main_process(dist_info):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    if dist_info["distributed"]:
        dist.barrier()

    use_rnafm = bool(model_cfg.get("use_rnafm", True))
    freeze_rnafm = bool(model_cfg.get("freeze_rnafm", rnafm_cfg.get("freeze_rnafm", False)))
    fusion_type = str(model_cfg.get("fusion_type", "cross_attn_gate")).lower()
    rna_pooling = str(model_cfg.get("rna_pooling", "mean")).lower()
    use_pam_encoder = bool(model_cfg.get("use_pam_encoder", False))
    if use_rnafm and freeze_rnafm:
        raise ValueError("BL5-3 formal run requires model.freeze_rnafm=false")

    arrays = BL5Arrays(config)
    group_labels = None
    if config.get("split_mode") == "sgrna_safe":
        import pandas as pd

        csv_path = data_cfg.get("cclmoff_csv")
        group_col = data_cfg.get("group_column", "sgRNA_type")
        if not csv_path:
            raise ValueError("sgrna_safe split requires data.cclmoff_csv")
        group_labels = pd.read_csv(csv_path, usecols=[group_col])[group_col].values
        if len(group_labels) != len(arrays.labels):
            raise ValueError(f"group label count mismatch: {len(group_labels)} vs {len(arrays.labels)}")

    split_cfg = config.get("split", {})
    if split_cfg.get("strategy") == "formal_group_json" and split_cfg.get("formal_split_json"):
        split_result = formal_group_json_split(
            arrays.labels.astype(np.int64),
            group_labels,
            split_cfg,
        )
        split_indices = {
            "train": split_result["train"],
            "val": split_result["val"],
            "test": split_result["test"],
        }
        split_metadata = split_result["metadata"]
    else:
        split_indices = make_split(arrays.labels.astype(np.int64), seed, group_labels)
        split_metadata = {
            "split_source": "make_split",
            "seed": seed,
            "split_mode": config.get("split_mode"),
            "group_column": data_cfg.get("group_column", "sgRNA_type") if group_labels is not None else None,
        }

    # Apply PAM holdout split if configured
    holdout_split_dir = split_cfg.get("holdout_split_dir")
    if holdout_split_dir:
        split_indices = apply_pam_holdout_split(split_indices, holdout_split_dir)
        split_metadata["holdout_split_dir"] = str(holdout_split_dir)
        # Update counts in metadata for audit
        for name, idx in split_indices.items():
            split_metadata.setdefault("holdout_counts", {})[name] = {
                "samples": int(len(idx)),
                "observed_positive": int((arrays.labels[idx] == 1).sum()),
                "unobserved_candidate": int((arrays.labels[idx] == 0).sum()),
            }

    # PAM shuffle (within-split) setup
    shuffle_pam = bool(training_cfg.get("shuffle_pam", False))
    shuffle_pam_mode = str(training_cfg.get("shuffle_pam_mode", "batch"))
    shuffle_pam_seed = int(training_cfg.get("shuffle_pam_seed", 42))

    def _make_pam_shuffle(indices: np.ndarray, seed: int) -> np.ndarray | None:
        if len(indices) <= 1:
            return None
        rng = np.random.default_rng(seed)
        perm = np.arange(len(indices))
        rng.shuffle(perm)
        return indices[perm]

    train_shuffle = None
    val_shuffle = None
    test_shuffle = None
    if shuffle_pam and shuffle_pam_mode == "within_split":
        train_shuffle = _make_pam_shuffle(split_indices["train"], shuffle_pam_seed)
        val_shuffle = _make_pam_shuffle(split_indices["val"], shuffle_pam_seed + 1)
        test_shuffle = _make_pam_shuffle(split_indices["test"], shuffle_pam_seed + 2)

    train_dataset = BL5Dataset(arrays, split_indices["train"], train_shuffle)
    val_dataset = BL5Dataset(arrays, split_indices["val"], val_shuffle)
    test_dataset = BL5Dataset(arrays, split_indices["test"], test_shuffle)

    # PAM shuffle audit
    if shuffle_pam and is_main_process(dist_info):
        from collections import Counter

        def _pam_dist(off_seqs, idx_arr):
            pams = [str(off_seqs[i])[-3:] for i in idx_arr]
            return dict(Counter(pams))

        def _same_ratio(orig, shuf):
            if shuf is None:
                return 1.0
            same = int(np.sum(orig == shuf))
            return round(same / len(orig), 6)

        audit = {
            "shuffle_pam": True,
            "shuffle_pam_mode": shuffle_pam_mode,
            "shuffle_pam_seed": shuffle_pam_seed,
            "splits": {},
        }
        for split_name, orig_idx, shuf_idx in [
            ("train", split_indices["train"], train_shuffle),
            ("val", split_indices["val"], val_shuffle),
            ("test", split_indices["test"], test_shuffle),
        ]:
            changed = int(np.sum(orig_idx != shuf_idx)) if shuf_idx is not None else 0
            unchanged = int(np.sum(orig_idx == shuf_idx)) if shuf_idx is not None else len(orig_idx)
            audit["splits"][split_name] = {
                "n_samples": len(orig_idx),
                "original_dist": _pam_dist(arrays.off_seqs, orig_idx),
                "shuffled_dist": _pam_dist(arrays.off_seqs, shuf_idx) if shuf_idx is not None else _pam_dist(arrays.off_seqs, orig_idx),
                "same_position_ratio": _same_ratio(orig_idx, shuf_idx),
                "changed": changed,
                "unchanged": unchanged,
            }
        audit_dir = output_dir
        audit_dir.mkdir(parents=True, exist_ok=True)
        (audit_dir / "pam_shuffle_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
        md_lines = ["# PAM Shuffle Audit\n", f"- mode: {shuffle_pam_mode}\n", f"- seed: {shuffle_pam_seed}\n\n"]
        for split_name, info in audit["splits"].items():
            md_lines.append(f"## {split_name}\n")
            md_lines.append(f"- samples: {info['n_samples']}\n")
            md_lines.append(f"- same_position_ratio: {info['same_position_ratio']}\n")
            md_lines.append(f"- changed: {info['changed']}\n")
            md_lines.append(f"- unchanged: {info['unchanged']}\n")
            md_lines.append(f"- original_dist: {info['original_dist']}\n")
            md_lines.append(f"- shuffled_dist: {info['shuffled_dist']}\n\n")
        (audit_dir / "pam_shuffle_audit.md").write_text("".join(md_lines), encoding="utf-8")

    if use_rnafm:
        rnafm_model, alphabet = load_rnafm(rnafm_cfg.get("checkpoint_path"), trust_local_checkpoint=True)
        rnafm_model = rnafm_model.to(device)
        # Disable gradients for RNA-FM heads that are not used when return_contacts=False,
        # so DDP can run with find_unused_parameters=False and avoid NCCL timeouts.
        for name, param in rnafm_model.named_parameters():
            if "contact_head" in name or "lm_head" in name:
                param.requires_grad = False
    else:
        rnafm_model = None
        alphabet = None

    use_learnable_run = bool(model_cfg.get("use_learnable_run", False))
    use_run = bool(model_cfg.get("use_run", True))
    collate_fn = make_live_collate(
        alphabet,
        use_rnafm=use_rnafm,
        use_run=use_run,
        use_learnable_run=use_learnable_run,
        use_pam_encoder=use_pam_encoder,
        shuffle_pam=shuffle_pam,
        shuffle_pam_mode=shuffle_pam_mode,
    )
    batch_size = int(training_cfg.get("batch_size", 128))
    num_workers = int(training_cfg.get("num_workers", 0))
    dataloader_kwargs: dict[str, Any] = {}
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = bool(training_cfg.get("persistent_workers", False))
        dataloader_kwargs["prefetch_factor"] = int(training_cfg.get("prefetch_factor", 2))

    if dist_info["distributed"]:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=dist_info["world_size"],
            rank=dist_info["rank"],
            shuffle=True,
            seed=seed,
            drop_last=False,
        )
        val_sampler = SequentialDistributedSampler(val_dataset, dist_info["rank"], dist_info["world_size"])
        test_sampler = SequentialDistributedSampler(test_dataset, dist_info["rank"], dist_info["world_size"])
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers, pin_memory=torch.cuda.is_available(), collate_fn=collate_fn,
            **dataloader_kwargs,
        )
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=torch.cuda.is_available(), collate_fn=collate_fn,
            **dataloader_kwargs,
        )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size * 2, sampler=val_sampler, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(), collate_fn=collate_fn,
        **dataloader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size * 2, sampler=test_sampler, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(), collate_fn=collate_fn,
        **dataloader_kwargs,
    )

    model = BL5RunOnlyDynamicFusion(
        rnafm_model=rnafm_model,
        padding_idx=alphabet.padding_idx if alphabet else 0,
        config=config,
    ).to(device)
    optimizer = build_optimizer(model, config)
    find_unused_parameters = bool(training_cfg.get("find_unused_parameters", False))
    if dist_info["distributed"]:
        if torch.cuda.is_available():
            model_for_train: nn.Module = DistributedDataParallel(
                model,
                device_ids=[dist_info["local_rank"]],
                output_device=dist_info["local_rank"],
                find_unused_parameters=find_unused_parameters,
            )
        else:
            model_for_train = DistributedDataParallel(
                model,
                find_unused_parameters=find_unused_parameters,
            )
    else:
        model_for_train = model

    focal = bool(training_cfg.get("focal_loss", False))
    focal_gamma = float(training_cfg.get("focal_gamma", 2.0))
    pos_weight_value = training_cfg.get("pos_weight")
    pos_weight = None
    if pos_weight_value is not None:
        pos_weight = torch.tensor([float(pos_weight_value)], dtype=torch.float32, device=device)

    use_amp = bool(config.get("hardware", {}).get("use_amp", False) and torch.cuda.is_available())
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    epochs = int(training_cfg.get("epochs", 10))
    monitor = str(training_cfg.get("monitor", "auprc"))
    monitor_mode = str(training_cfg.get("monitor_mode", "max"))
    grad_clip = training_cfg.get("gradient_clip")
    grad_clip = None if grad_clip is None else float(grad_clip)
    gate_l2_lambda = float(training_cfg.get("gate_l2_lambda", 0.0))
    early_stopping_patience = training_cfg.get("early_stopping_patience")
    early_stopping_patience = (
        None if early_stopping_patience is None else int(early_stopping_patience)
    )
    early_stopping_min_delta = float(training_cfg.get("early_stopping_min_delta", 0.0))
    max_train_batches = training_cfg.get("max_train_batches")
    max_eval_batches = training_cfg.get("max_eval_batches")
    max_train_batches = None if max_train_batches is None else int(max_train_batches)
    max_eval_batches = None if max_eval_batches is None else int(max_eval_batches)
    best_metric = -float("inf") if monitor_mode == "max" else float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    completed_epochs = 0
    epoch_metrics: list[dict[str, Any]] = []
    start_time = time.time()

    if args.eval_only:
        ckpt_path = Path(args.checkpoint) if args.checkpoint else output_dir / "checkpoints" / "best.pt"
        check_eval_procedure(ckpt_path, checkpoint_type="best", require_exists=True)
        best_ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"])
        if dist_info["distributed"]:
            model_for_train.module.load_state_dict(best_ckpt["model_state_dict"])
        test_metrics = evaluate(
            model_for_train,
            test_loader,
            device,
            dist_info,
            focal_loss=focal,
            focal_gamma=focal_gamma,
            pos_weight=pos_weight,
            max_batches=max_eval_batches,
        )

        # Export test predictions in eval-only mode too
        test_predictions_path = None
        probabilities = None
        if bool(config.get("outputs", {}).get("export_test_predictions", False)):
            probabilities = predict_probabilities(
                model_for_train, test_loader, device, dist_info, max_batches=max_eval_batches
            )
            if is_main_process(dist_info):
                probabilities = restore_sequential_distributed_order(
                    probabilities,
                    len(test_dataset),
                    int(dist_info.get("world_size", 1)),
                )
                test_predictions_path = output_dir / "test_predictions.csv"
                csv_path = data_cfg.get("cclmoff_csv")
                if csv_path:
                    write_test_predictions(
                        csv_path,
                        split_indices["test"],
                        probabilities,
                        test_predictions_path,
                        pam_shuffle_indices=test_shuffle,
                        split_name="test",
                    )
                else:
                    test_predictions_path = None

        if is_main_process(dist_info):
            report_metrics(test_metrics.get("auroc"), test_metrics.get("auprc"), config.get("split_mode"))
            eval_seconds = time.time() - start_time
            gpu_mem = "cpu"
            if torch.cuda.is_available():
                gpu_mem = f"{torch.cuda.max_memory_allocated(device) / (1024 ** 3):.2f}GB"
            best_metrics = best_ckpt.get("metrics", {})
            summary = {
                "version": config.get("version", "BL5-3"),
                "status": "completed_eval_only",
                "generated_at": utc_now(),
                "commit_hash": git_hash(),
                "device": str(device),
                "distributed": dist_info["distributed"],
                "split_mode": config.get("split_mode"),
                "use_rnafm": use_rnafm,
                "use_run": use_run,
                "freeze_rnafm": freeze_rnafm,
                "use_learnable_run": use_learnable_run,
                "use_pam_encoder": use_pam_encoder,
                "fusion_type": fusion_type,
                "rna_pooling": rna_pooling,
                "train_seconds": eval_seconds,
                "gpu_mem": gpu_mem,
                "epochs": 0,
                "planned_epochs": epochs,
                "best_epoch": int(best_ckpt.get("epoch", 0)),
                "best_metric_name": monitor,
                "best_metric_value": float(best_metrics.get(monitor, float("nan"))),
                "test_metrics": test_metrics,
                "notes": (
                    f"Eval-only recovery from best.pt after interrupted run; "
                    f"{'Fine-tuned RNA-FM + ' if use_rnafm else ''}"
                    f"{'PAM Encoder' if fusion_type == 'pam_only' else 'LearnableRunEncoder' if use_learnable_run else 'Run token CNN'} "
                    f"+ {'raw RNA-FM/Run simple concat MLP' if fusion_type == 'simple_concat' else 'Cross-Attn + Softmax Gate' if fusion_type == 'cross_attn_gate' else 'PAM-Gated Fusion' if fusion_type == 'pam_gated_fusion' else 'PAM-only' if fusion_type == 'pam_only' else 'Run-only'}; "
                    f"rna_pooling={rna_pooling}; "
                    f"use_pam_encoder={use_pam_encoder}; "
                    f"focal_loss gamma={focal_gamma}; "
                    f"gate_l2_lambda={gate_l2_lambda:g}; "
                    f"early_stopping_patience={early_stopping_patience}; "
                    "best.pt test evaluation"
                ),
            }
            (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
            write_report(output_dir, summary, args.config)
            if config.get("logging", {}).get("append_experiment", True):
                append_experiment(Path("results/experiments.csv"), summary, args.config)
        cleanup_distributed(dist_info["distributed"])
        return 0

    if is_main_process(dist_info):
        print(
            f"[BL5] version={config.get('version')} use_rnafm={use_rnafm} use_run={use_run} freeze_rnafm={freeze_rnafm} "
            f"fusion_type={fusion_type} "
            f"use_pam_encoder={use_pam_encoder} "
            f"device={device} "
            f"distributed={dist_info['distributed']}"
        )
        print(f"[BL5] split sizes: train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)}")

    for epoch in range(1, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_loss = train_one_epoch(
            model_for_train,
            train_loader,
            optimizer,
            device,
            scaler,
            use_amp=use_amp,
            grad_clip=grad_clip,
            focal_loss=focal,
            focal_gamma=focal_gamma,
            pos_weight=pos_weight,
            max_batches=max_train_batches,
            gate_l2_lambda=gate_l2_lambda,
        )
        val_metrics = evaluate(
            model_for_train,
            val_loader,
            device,
            dist_info,
            focal_loss=focal,
            focal_gamma=focal_gamma,
            pos_weight=pos_weight,
            max_batches=max_eval_batches,
        )
        current = val_metrics.get(monitor, float("nan"))
        current_is_finite = math.isfinite(current)
        if current_is_finite:
            scheduler.step(current)
        if monitor_mode == "max" and current_is_finite:
            is_best = current > best_metric + early_stopping_min_delta
        elif current_is_finite:
            is_best = current < best_metric - early_stopping_min_delta
        else:
            is_best = False
        if is_best:
            best_metric = current
            best_epoch = epoch
            epochs_without_improvement = 0
            if is_main_process(dist_info):
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch,
                        "metrics": val_metrics,
                        "config": config,
                    },
                    output_dir / "checkpoints" / "best.pt",
                )
        else:
            epochs_without_improvement += 1

        if is_main_process(dist_info):
            row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
            epoch_metrics.append(row)
            print(
                f"epoch={epoch} train_loss={train_loss:.6f} "
                f"val_auroc={val_metrics.get('auroc', float('nan')):.6f} "
                f"val_auprc={val_metrics.get('auprc', float('nan')):.6f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )
            if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
                print(
                    f"[EarlyStopping] no {monitor} improvement for "
                    f"{epochs_without_improvement} epoch(s); best_epoch={best_epoch}"
                )
        if dist_info["distributed"]:
            dist.barrier()
        completed_epochs = epoch
        if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
            break

    best_ckpt_path = output_dir / "checkpoints" / "best.pt"
    check_eval_procedure(best_ckpt_path, checkpoint_type="best", require_exists=True)
    best_ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    if dist_info["distributed"]:
        model_for_train.module.load_state_dict(best_ckpt["model_state_dict"])
    test_metrics = evaluate(
        model_for_train,
        test_loader,
        device,
        dist_info,
        focal_loss=focal,
        focal_gamma=focal_gamma,
        pos_weight=pos_weight,
        max_batches=max_eval_batches,
    )

    test_predictions_path = None
    probabilities = None
    if bool(config.get("outputs", {}).get("export_test_predictions", False)):
        probabilities = predict_probabilities(
            model_for_train, test_loader, device, dist_info, max_batches=max_eval_batches
        )
        if is_main_process(dist_info):
            probabilities = restore_sequential_distributed_order(
                probabilities,
                len(test_dataset),
                int(dist_info.get("world_size", 1)),
            )
            test_predictions_path = output_dir / "test_predictions.csv"
            csv_path = data_cfg.get("cclmoff_csv")
            if csv_path:
                write_test_predictions(
                    csv_path,
                    split_indices["test"],
                    probabilities,
                    test_predictions_path,
                    pam_shuffle_indices=test_shuffle,
                    split_name="test",
                )
            else:
                test_predictions_path = None

    if is_main_process(dist_info):
        report_metrics(test_metrics.get("auroc"), test_metrics.get("auprc"), config.get("split_mode"))
        train_seconds = time.time() - start_time
        gpu_mem = "cpu"
        if torch.cuda.is_available():
            gpu_mem = f"{torch.cuda.max_memory_allocated(device) / (1024 ** 3):.2f}GB"
        summary = {
            "version": config.get("version", "BL5-3"),
            "status": "completed",
            "generated_at": utc_now(),
            "commit_hash": git_hash(),
            "device": str(device),
            "distributed": dist_info["distributed"],
            "split_mode": config.get("split_mode"),
            "use_rnafm": use_rnafm,
            "use_run": use_run,
            "freeze_rnafm": freeze_rnafm,
            "use_learnable_run": use_learnable_run,
            "use_pam_encoder": use_pam_encoder,
            "fusion_type": fusion_type,
            "rna_pooling": rna_pooling,
            "train_seconds": train_seconds,
            "gpu_mem": gpu_mem,
            "epochs": completed_epochs,
            "planned_epochs": epochs,
            "best_epoch": int(best_ckpt["epoch"]),
            "best_metric_name": monitor,
            "best_metric_value": float(best_metric),
            "test_metrics": test_metrics,
            "split": split_metadata,
            "artifacts": {
                "best_checkpoint": str(best_ckpt_path),
                "epoch_metrics": str(output_dir / "epoch_metrics.csv"),
                "summary": str(output_dir / "summary.json"),
                "test_predictions": str(test_predictions_path) if test_predictions_path is not None else None,
            },
            "notes": (
                f"{'Fine-tuned RNA-FM + ' if use_rnafm else ''}"
                f"{'PAM Encoder' if fusion_type == 'pam_only' else 'LearnableRunEncoder' if use_learnable_run else 'Run token CNN'} "
                f"+ {'raw RNA-FM/Run simple concat MLP' if fusion_type == 'simple_concat' else 'Cross-Attn + Softmax Gate' if fusion_type == 'cross_attn_gate' else 'PAM-Gated Fusion' if fusion_type == 'pam_gated_fusion' else 'PAM-only' if fusion_type == 'pam_only' else 'Run-only'}; "
                f"rna_pooling={rna_pooling}; "
                f"use_pam_encoder={use_pam_encoder}; "
                f"focal_loss gamma={focal_gamma}; "
                f"gate_l2_lambda={gate_l2_lambda:g}; "
                f"early_stopping_patience={early_stopping_patience}; "
                "best.pt test evaluation"
            ),
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        if epoch_metrics:
            with (output_dir / "epoch_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(epoch_metrics[0].keys()))
                writer.writeheader()
                writer.writerows(epoch_metrics)
        write_report(output_dir, summary, args.config)
        if config.get("logging", {}).get("append_experiment", True):
            append_experiment(Path("results/experiments.csv"), summary, args.config)

    cleanup_distributed(dist_info["distributed"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
