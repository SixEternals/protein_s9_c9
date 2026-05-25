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
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Sampler, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.cclmoff_dataset import CCLMoffFrameDataset, CCLMoffSample, load_cclmoff_dataframe
from models.bl0_cclmoff import BL0CCLMoffConfig, build_bl0_with_rnafm
from utils.config import load_config
from utils.rnafm import count_parameters, rnafm_model_specs, tokenize_rnafm_sequences


def setup_distributed() -> dict[str, Any]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP was requested through torchrun, but CUDA is not available")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    return {
        "distributed": distributed,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
    }


def cleanup_distributed(distributed: bool) -> None:
    if distributed and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(dist_info: dict[str, Any]) -> bool:
    return int(dist_info.get("rank", 0)) == 0


def barrier(dist_info: dict[str, Any]) -> None:
    if bool(dist_info.get("distributed", False)):
        dist.barrier()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def choose_device(config: dict[str, Any], dist_info: dict[str, Any] | None = None) -> torch.device:
    if dist_info and bool(dist_info.get("distributed", False)):
        return torch.device(f"cuda:{int(dist_info['local_rank'])}")
    requested = str(config.get("device", "auto")).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(requested)


def make_model_config(config: dict[str, Any]) -> BL0CCLMoffConfig:
    model_cfg = dict(config.get("model_config", {}))
    rnafm_cfg = dict(config.get("rnafm", {}))
    model_cfg["freeze_rnafm"] = bool(rnafm_cfg.get("freeze", True))
    model_cfg["repr_layer"] = int(rnafm_cfg.get("repr_layer", model_cfg.get("repr_layer", 12)))
    return BL0CCLMoffConfig(**model_cfg)


def make_collate_fn(alphabet: Any):
    def collate(samples: list[CCLMoffSample]) -> tuple[torch.Tensor, torch.Tensor]:
        sequences = [sample.sequence for sample in samples]
        labels = torch.tensor([sample.label for sample in samples], dtype=torch.float32)
        return tokenize_rnafm_sequences(alphabet, sequences), labels

    return collate


def label_counts(labels: np.ndarray) -> dict[str, int]:
    return {
        "observed_positive": int((labels == 1).sum()),
        "unobserved_candidate": int((labels == 0).sum()),
    }


def has_both_labels(labels: np.ndarray) -> bool:
    return len(np.unique(labels)) == 2


def row_stratified_split(df, split_cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    indices = np.arange(len(df), dtype=np.int64)
    train_fraction = float(split_cfg.get("train_fraction", 0.8))
    val_fraction = float(split_cfg.get("val_fraction", 0.1))
    test_fraction = float(split_cfg.get("test_fraction", 0.1))
    if not math.isclose(train_fraction + val_fraction + test_fraction, 1.0, abs_tol=1e-6):
        raise ValueError("split fractions must sum to 1.0")
    holdout_fraction = val_fraction + test_fraction
    train_idx, holdout_idx = train_test_split(
        indices,
        test_size=holdout_fraction,
        random_state=seed,
        stratify=df["label"].to_numpy(),
    )
    relative_test_fraction = test_fraction / holdout_fraction
    holdout_labels = df.iloc[holdout_idx]["label"].to_numpy()
    val_idx, test_idx = train_test_split(
        holdout_idx,
        test_size=relative_test_fraction,
        random_state=seed + 1,
        stratify=holdout_labels,
    )
    return {
        "strategy": "row_stratified",
        "train": np.asarray(train_idx, dtype=np.int64),
        "val": np.asarray(val_idx, dtype=np.int64),
        "test": np.asarray(test_idx, dtype=np.int64),
        "metadata": {"leakage_safe_by_sgrna_type": False},
    }


def group_safe_split(df, split_cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    group_column = str(split_cfg.get("group_column", "sgRNA_type"))
    train_fraction = float(split_cfg.get("train_fraction", 0.8))
    val_fraction = float(split_cfg.get("val_fraction", 0.1))
    test_fraction = float(split_cfg.get("test_fraction", 0.1))
    max_attempts = int(split_cfg.get("max_attempts", 200))
    allow_single_class = bool(split_cfg.get("allow_single_class_split", False))
    if not math.isclose(train_fraction + val_fraction + test_fraction, 1.0, abs_tol=1e-6):
        raise ValueError("split fractions must sum to 1.0")
    if group_column not in df.columns:
        raise ValueError(f"group split column not found: {group_column}")

    groups = df[group_column].astype(str).to_numpy()
    labels = df["label"].to_numpy()
    holdout_fraction = val_fraction + test_fraction
    relative_test_fraction = test_fraction / holdout_fraction
    last_error = ""
    for attempt in range(max_attempts):
        attempt_seed = seed + attempt
        splitter = GroupShuffleSplit(n_splits=1, test_size=holdout_fraction, random_state=attempt_seed)
        train_idx, holdout_idx = next(splitter.split(df, labels, groups))
        holdout_df = df.iloc[holdout_idx]
        holdout_groups = holdout_df[group_column].astype(str).to_numpy()
        holdout_labels = holdout_df["label"].to_numpy()
        splitter_2 = GroupShuffleSplit(n_splits=1, test_size=relative_test_fraction, random_state=attempt_seed + 10_000)
        val_rel, test_rel = next(splitter_2.split(holdout_df, holdout_labels, holdout_groups))
        val_idx = holdout_idx[val_rel]
        test_idx = holdout_idx[test_rel]

        split_labels = {
            "train": labels[train_idx],
            "val": labels[val_idx],
            "test": labels[test_idx],
        }
        if allow_single_class or all(has_both_labels(values) for values in split_labels.values()):
            train_groups = set(df.iloc[train_idx][group_column].astype(str))
            val_groups = set(df.iloc[val_idx][group_column].astype(str))
            test_groups = set(df.iloc[test_idx][group_column].astype(str))
            overlaps = {
                "train_val": sorted(train_groups & val_groups),
                "train_test": sorted(train_groups & test_groups),
                "val_test": sorted(val_groups & test_groups),
            }
            if any(overlaps.values()):
                raise RuntimeError(f"group leakage detected: {overlaps}")
            return {
                "strategy": "sgRNA_type_group",
                "train": np.asarray(train_idx, dtype=np.int64),
                "val": np.asarray(val_idx, dtype=np.int64),
                "test": np.asarray(test_idx, dtype=np.int64),
                "metadata": {
                    "group_column": group_column,
                    "attempt_seed": attempt_seed,
                    "leakage_safe_by_sgrna_type": True,
                    "group_counts": {
                        "train": len(train_groups),
                        "val": len(val_groups),
                        "test": len(test_groups),
                    },
                    "groups": {
                        "train": sorted(train_groups),
                        "val": sorted(val_groups),
                        "test": sorted(test_groups),
                    },
                },
            }
        last_error = json.dumps({name: label_counts(values) for name, values in split_labels.items()}, ensure_ascii=False)
    raise RuntimeError(f"could not create a group-safe split with both labels in every split; last counts={last_error}")


def make_split(df, split_cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    strategy = str(split_cfg.get("strategy", "sgRNA_type_group"))
    if strategy == "sgRNA_type_group":
        return group_safe_split(df, split_cfg, seed)
    if strategy == "row_stratified":
        return row_stratified_split(df, split_cfg, seed)
    raise ValueError(f"unsupported split strategy: {strategy}")


def make_balanced_sampler(
    labels: np.ndarray,
    *,
    num_samples: int | None = None,
    seed: int | None = None,
) -> WeightedRandomSampler:
    counts = label_counts(labels)
    if counts["observed_positive"] == 0 or counts["unobserved_candidate"] == 0:
        raise ValueError(f"balanced sampler needs both label classes, got {counts}")
    weights_by_label = {
        1: 1.0 / counts["observed_positive"],
        0: 1.0 / counts["unobserved_candidate"],
    }
    weights = torch.tensor([weights_by_label[int(label)] for label in labels], dtype=torch.double)
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    return WeightedRandomSampler(
        weights=weights,
        num_samples=int(num_samples or len(weights)),
        replacement=True,
        generator=generator,
    )


def make_loader(
    dataset: CCLMoffFrameDataset,
    alphabet: Any,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    balanced: bool,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 42,
) -> DataLoader:
    sampler: Sampler | None = None
    if balanced:
        num_samples = math.ceil(len(dataset) / world_size) if distributed else len(dataset)
        sampler = make_balanced_sampler(dataset.labels, num_samples=num_samples, seed=seed + rank)
    elif distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=False,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False if sampler is not None else shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=make_collate_fn(alphabet),
    )


def metric_payload(labels: np.ndarray, probabilities: np.ndarray, loss_values: list[float]) -> dict[str, float]:
    if labels.size == 0:
        return {"loss": float("nan"), "auroc": float("nan"), "auprc": float("nan")}
    payload = {"loss": float(np.mean(loss_values)) if loss_values else float("nan")}
    if has_both_labels(labels):
        payload["auroc"] = float(roc_auc_score(labels, probabilities))
        payload["auprc"] = float(average_precision_score(labels, probabilities))
    else:
        payload["auroc"] = float("nan")
        payload["auprc"] = float("nan")
    predicted = (probabilities >= 0.5).astype(np.int64)
    payload["accuracy"] = float(accuracy_score(labels, predicted))
    payload["precision"] = float(precision_score(labels, predicted, zero_division=0))
    payload["recall"] = float(recall_score(labels, predicted, zero_division=0))
    payload["f1"] = float(f1_score(labels, predicted, zero_division=0))
    return payload


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    precision: str,
    *,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    labels_all: list[np.ndarray] = []
    probs_all: list[np.ndarray] = []
    losses: list[float] = []
    autocast_enabled = device.type == "cuda" and precision in {"fp16", "bf16"}
    autocast_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    with torch.no_grad():
        for batch_idx, (tokens, labels) in enumerate(loader, start=1):
            if max_batches is not None and batch_idx > max_batches:
                break
            tokens = tokens.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_enabled):
                logits = model(tokens)
                loss = F.binary_cross_entropy_with_logits(logits.float(), labels.float())
            losses.append(float(loss.item()))
            labels_all.append(labels.detach().cpu().numpy())
            probs_all.append(torch.sigmoid(logits.float()).detach().cpu().numpy())
    labels_np = np.concatenate(labels_all) if labels_all else np.array([], dtype=np.float32)
    probs_np = np.concatenate(probs_all) if probs_all else np.array([], dtype=np.float32)
    return metric_payload(labels_np, probs_np, losses)


def build_optimizer(model: torch.nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    training_cfg = dict(config.get("training", {}))
    head_lr = float(training_cfg.get("head_learning_rate", training_cfg.get("lr_mlp", training_cfg.get("learning_rate", 1e-3))))
    rnafm_lr = float(
        training_cfg.get("rnafm_learning_rate", training_cfg.get("lr_transformer", training_cfg.get("learning_rate", 1e-5)))
    )
    weight_decay = float(training_cfg.get("weight_decay", 1e-5))
    head_params = [param for param in model.head.parameters() if param.requires_grad]
    rnafm_params = [param for param in model.rnafm_model.parameters() if param.requires_grad]
    groups: list[dict[str, Any]] = []
    if rnafm_params:
        groups.append({"params": rnafm_params, "lr": rnafm_lr, "name": "rnafm"})
    if head_params:
        groups.append({"params": head_params, "lr": head_lr, "name": "head"})
    if not groups:
        raise RuntimeError("no trainable parameters found")
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_epochs: float,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LambdaLR | None:
    warmup_steps = int(float(warmup_epochs) * int(steps_per_epoch))
    if warmup_steps <= 0:
        return None

    def lr_lambda(step: int) -> float:
        return min(1.0, float(step + 1) / float(warmup_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def current_lr_by_group(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    return {str(group.get("name", f"group_{index}")): float(group["lr"]) for index, group in enumerate(optimizer.param_groups)}


def reduced_train_loss(loss_sum: float, loss_count: int, device: torch.device, dist_info: dict[str, Any]) -> float:
    if loss_count <= 0:
        return float("nan")
    if not bool(dist_info.get("distributed", False)):
        return float(loss_sum / loss_count)
    payload = torch.tensor([float(loss_sum), float(loss_count)], device=device, dtype=torch.float64)
    dist.all_reduce(payload, op=dist.ReduceOp.SUM)
    total_count = float(payload[1].item())
    return float(payload[0].item() / total_count) if total_count > 0 else float("nan")


def append_experiment(path: Path, row: dict[str, Any]) -> None:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def save_checkpoint(path: Path, model, optimizer, config: dict[str, Any], epoch: int, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_to_save = unwrap_model(model)
    payload = {
        "model": "bl0a_rnafm_mlp",
        "epoch": epoch,
        "metrics": metrics,
        "head_state_dict": model_to_save.head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "label_semantics": {"1": "observed_positive", "0": "unobserved_candidate"},
    }
    if not bool(config.get("rnafm", {}).get("freeze", True)):
        payload["model_state_dict"] = model_to_save.state_dict()
    torch.save(payload, path)


def load_best_checkpoint(path: Path, model) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model_to_load = unwrap_model(model)
    if "model_state_dict" in checkpoint:
        model_to_load.load_state_dict(checkpoint["model_state_dict"])
    else:
        model_to_load.head.load_state_dict(checkpoint["head_state_dict"])


def write_epoch_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_split_report(output_dir: Path, split_info: dict[str, Any], df, split_indices: dict[str, np.ndarray]) -> None:
    payload = {
        "strategy": split_info["strategy"],
        "metadata": split_info["metadata"],
        "splits": {},
    }
    for name in ("train", "val", "test"):
        idx = split_indices[name]
        labels = df.iloc[idx]["label"].to_numpy()
        payload["splits"][name] = {
            "rows": int(len(idx)),
            "label_counts": label_counts(labels),
            "sgRNA_type_unique": int(df.iloc[idx]["sgRNA_type"].nunique()),
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "split_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_report(output_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# BL0 RNA-FM Training Report",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Version: `{summary['version']}`",
        f"- Status: `{summary['status']}`",
        f"- Config: `{summary['config_path']}`",
        f"- Device: `{summary['device']}`",
        f"- Split strategy: `{summary['split']['strategy']}`",
        f"- Train rows: `{summary['split']['train']['rows']}`",
        f"- Val rows: `{summary['split']['val']['rows']}`",
        f"- Test rows: `{summary['split']['test']['rows']}`",
        f"- RNA-FM specs: `{summary['model']['rnafm_specs']}`",
        f"- Total params: `{summary['model']['total_params']}`",
        f"- Trainable params: `{summary['model']['trainable_params']}`",
        f"- Best epoch: `{summary['best_epoch']}`",
        f"- Best val metric: `{summary['best_metric_name']}={summary['best_metric_value']}`",
        f"- Test AUROC: `{summary['test_metrics']['auroc']}`",
        f"- Test AUPRC: `{summary['test_metrics']['auprc']}`",
        f"- Best checkpoint: `{summary['artifacts']['best_checkpoint']}`",
        "",
        "## Label Semantics",
        "",
        "- `label=1`: `observed_positive`",
        "- `label=0`: `unobserved_candidate`, not verified safe",
        "",
        "## Notes",
        "",
        "- This is a BL0 RNA-FM + CCLMoff-style MLP head baseline.",
        "- No Region encoder or Run encoder is used in this baseline.",
        "- If `split.strategy=sgRNA_type_group`, sgRNA_type groups do not overlap across train/val/test.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def train(config_path: Path) -> dict[str, Any]:
    start_time = time.time()
    dist_info = setup_distributed()
    main_process = is_main_process(dist_info)
    config = load_config(config_path)
    seed = int(config.get("seed", 42))
    torch.manual_seed(seed + int(dist_info["rank"]))
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    output_dir = Path(config.get("outputs", {}).get("output_dir", "results/bl0a_formal_frozen"))
    if main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    barrier(dist_info)
    config_copy_path = output_dir / "config_used.json"
    if main_process:
        config_copy_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    df = load_cclmoff_dataframe(
        config["dataset"]["csv_path"],
        max_rows=config.get("dataset", {}).get("max_rows"),
    )
    split_info = make_split(df, dict(config.get("split", {})), seed)
    split_indices = {name: split_info[name] for name in ("train", "val", "test")}
    if main_process:
        write_split_report(output_dir, split_info, df, split_indices)
    barrier(dist_info)

    replace_t_with_u = bool(config.get("dataset", {}).get("replace_t_with_u", True))
    train_dataset = CCLMoffFrameDataset(df.iloc[split_indices["train"]], replace_t_with_u=replace_t_with_u)
    val_dataset = CCLMoffFrameDataset(df.iloc[split_indices["val"]], replace_t_with_u=replace_t_with_u)
    test_dataset = CCLMoffFrameDataset(df.iloc[split_indices["test"]], replace_t_with_u=replace_t_with_u)

    device = choose_device(config, dist_info)
    rnafm_cfg = dict(config.get("rnafm", {}))
    model, alphabet = build_bl0_with_rnafm(
        checkpoint_path=rnafm_cfg.get("checkpoint_path"),
        allow_download=bool(rnafm_cfg.get("allow_download", False)),
        config=make_model_config(config),
    )
    model.to(device)
    optimizer = build_optimizer(model, config)
    if bool(dist_info["distributed"]):
        model = DistributedDataParallel(model, device_ids=[int(dist_info["local_rank"])], output_device=int(dist_info["local_rank"]), find_unused_parameters=True)

    training_cfg = dict(config.get("training", {}))
    batch_size = int(training_cfg.get("batch_size", 128))
    eval_batch_size = int(training_cfg.get("eval_batch_size", batch_size))
    num_workers = int(training_cfg.get("num_workers", 0))
    precision = str(training_cfg.get("precision", "none")).lower()
    if precision not in {"none", "fp16", "bf16"}:
        raise ValueError("training.precision must be one of: none, fp16, bf16")
    max_train_batches = training_cfg.get("max_train_batches")
    max_eval_batches = training_cfg.get("max_eval_batches")
    max_train_batches = int(max_train_batches) if max_train_batches is not None else None
    max_eval_batches = int(max_eval_batches) if max_eval_batches is not None else None
    log_every = int(training_cfg.get("log_every", 50))
    gradient_clip = training_cfg.get("gradient_clip")
    gradient_clip = float(gradient_clip) if gradient_clip is not None else None

    train_loader = make_loader(
        train_dataset,
        alphabet,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=not bool(training_cfg.get("use_balanced_sampler", True)),
        balanced=bool(training_cfg.get("use_balanced_sampler", True)),
        distributed=bool(dist_info["distributed"]),
        rank=int(dist_info["rank"]),
        world_size=int(dist_info["world_size"]),
        seed=seed,
    )
    val_loader = make_loader(
        val_dataset,
        alphabet,
        batch_size=eval_batch_size,
        num_workers=num_workers,
        shuffle=False,
        balanced=False,
        distributed=False,
    )
    test_loader = make_loader(
        test_dataset,
        alphabet,
        batch_size=eval_batch_size,
        num_workers=num_workers,
        shuffle=False,
        balanced=False,
        distributed=False,
    )

    epochs = int(training_cfg.get("epochs", 10))
    monitor = str(training_cfg.get("monitor", "auprc"))
    monitor_mode = str(training_cfg.get("monitor_mode", "max"))
    if monitor_mode not in {"max", "min"}:
        raise ValueError("training.monitor_mode must be 'max' or 'min'")
    best_value = -float("inf") if monitor_mode == "max" else float("inf")
    best_epoch = 0
    best_metrics: dict[str, float] = {}
    epoch_rows: list[dict[str, Any]] = []
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and precision == "fp16")
    autocast_enabled = device.type == "cuda" and precision in {"fp16", "bf16"}
    autocast_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    scheduler = build_scheduler(
        optimizer,
        warmup_epochs=float(training_cfg.get("warmup_epochs", 0)),
        steps_per_epoch=max_train_batches or len(train_loader),
    )
    checkpoints_dir = output_dir / "checkpoints"
    best_checkpoint = checkpoints_dir / "best.pt"
    last_checkpoint = checkpoints_dir / "last.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        epoch_start = time.time()
        for batch_idx, (tokens, labels) in enumerate(train_loader, start=1):
            if max_train_batches is not None and batch_idx > max_train_batches:
                break
            tokens = tokens.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_enabled):
                logits = model(tokens)
                loss = F.binary_cross_entropy_with_logits(logits.float(), labels.float())
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if gradient_clip is not None and gradient_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if gradient_clip is not None and gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), gradient_clip)
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            train_losses.append(float(loss.item()))
            if main_process and log_every > 0 and batch_idx % log_every == 0:
                print(f"epoch={epoch} step={batch_idx} train_loss={np.mean(train_losses[-log_every:]):.6f}", flush=True)

        train_loss = reduced_train_loss(float(np.sum(train_losses)), len(train_losses), device, dist_info)
        if main_process:
            val_metrics = evaluate(unwrap_model(model), val_loader, device, precision, max_batches=max_eval_batches)
            current = float(val_metrics.get(monitor, float("nan")))
            improved = current > best_value if monitor_mode == "max" else current < best_value
            if improved:
                best_value = current
                best_epoch = epoch
                best_metrics = dict(val_metrics)
                save_checkpoint(best_checkpoint, model, optimizer, config, epoch, val_metrics)
            save_checkpoint(last_checkpoint, model, optimizer, config, epoch, val_metrics)
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_auroc": val_metrics["auroc"],
                "val_auprc": val_metrics["auprc"],
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
                "seconds": round(time.time() - epoch_start, 3),
                "is_best": int(improved),
                **{f"lr_{name}": value for name, value in current_lr_by_group(optimizer).items()},
            }
            epoch_rows.append(row)
            write_epoch_metrics(output_dir / "epoch_metrics.csv", epoch_rows)
            print(
                f"epoch={epoch} train_loss={train_loss:.6f} "
                f"val_auroc={val_metrics['auroc']:.6f} val_auprc={val_metrics['auprc']:.6f} "
                f"best_epoch={best_epoch}",
                flush=True,
            )
        barrier(dist_info)

    if not main_process:
        return {"status": "completed_worker_rank", "rank": int(dist_info["rank"])}

    if best_epoch == 0:
        raise RuntimeError("training finished without a valid best checkpoint")
    load_best_checkpoint(best_checkpoint, model)
    model.to(device)
    test_metrics = evaluate(unwrap_model(model), test_loader, device, precision, max_batches=max_eval_batches)
    train_seconds = time.time() - start_time
    gpu_mem = "cpu"
    if device.type == "cuda":
        gpu_mem = f"{torch.cuda.max_memory_allocated() / 1024**3:.2f}GB"
    split_summary = {}
    for name in ("train", "val", "test"):
        idx = split_indices[name]
        split_summary[name] = {
            "rows": int(len(idx)),
            "label_counts": label_counts(df.iloc[idx]["label"].to_numpy()),
            "sgRNA_type_unique": int(df.iloc[idx]["sgRNA_type"].nunique()),
        }
    summary = {
        "generated_at": utc_now(),
        "version": str(config.get("version", "BL0a-formal-frozen")),
        "status": "completed",
        "config_path": str(config_path),
        "config_used": str(config_copy_path),
        "commit_hash": git_hash(),
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "distributed": {
            "enabled": bool(dist_info["distributed"]),
            "world_size": int(dist_info["world_size"]),
        },
        "train_seconds": train_seconds,
        "gpu_mem": gpu_mem,
        "split": {
            "strategy": split_info["strategy"],
            "metadata": split_info["metadata"],
            **split_summary,
        },
        "model": {
            "rnafm_specs": rnafm_model_specs(unwrap_model(model).rnafm_model),
            "total_params": count_parameters(unwrap_model(model)),
            "trainable_params": count_parameters(unwrap_model(model), trainable_only=True),
            "freeze_rnafm": bool(rnafm_cfg.get("freeze", True)),
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "eval_batch_size": eval_batch_size,
            "precision": precision,
            "monitor": monitor,
            "monitor_mode": monitor_mode,
            "gradient_clip": gradient_clip,
            "warmup_epochs": float(training_cfg.get("warmup_epochs", 0)),
            "learning_rates": current_lr_by_group(optimizer),
        },
        "best_epoch": best_epoch,
        "best_metric_name": monitor,
        "best_metric_value": best_value,
        "best_val_metrics": best_metrics,
        "test_metrics": test_metrics,
        "artifacts": {
            "best_checkpoint": str(best_checkpoint),
            "last_checkpoint": str(last_checkpoint),
            "epoch_metrics": str(output_dir / "epoch_metrics.csv"),
            "split_summary": str(output_dir / "split_summary.json"),
            "summary": str(output_dir / "summary.json"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(output_dir, summary)

    experiment_csv = Path(config.get("outputs", {}).get("experiment_csv", "results/experiments.csv"))
    append_experiment(
        experiment_csv,
        {
            "version": summary["version"],
            "date": summary["generated_at"],
            "commit_hash": summary["commit_hash"],
            "status": summary["status"],
            "auroc": f"{test_metrics['auroc']:.6f}",
            "auprc": f"{test_metrics['auprc']:.6f}",
            "train_time": f"{train_seconds / 3600:.3f}h",
            "gpu_mem": gpu_mem,
            "epochs": epochs,
            "best_epoch": best_epoch,
            "config_path": str(config_path),
            "notes": str(config.get("outputs", {}).get("notes", "BL0 formal training")),
        },
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Formal BL0 RNA-FM + MLP training with sgRNA_type-safe split.")
    parser.add_argument("--config", type=Path, default=Path("configs/bl0a_formal_frozen.yaml"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = train(args.config)
    except Exception as exc:  # noqa: BLE001
        print(f"BL0 formal training failed: {exc.__class__.__name__}: {exc}", file=sys.stderr)
        cleanup_distributed(dist.is_available() and dist.is_initialized())
        return 1
    cleanup_distributed(dist.is_available() and dist.is_initialized())
    if summary.get("status") != "completed_worker_rank":
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
