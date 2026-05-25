from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.cclmoff_dataset import CCLMoffCSVDataset, CCLMoffSample
from models.bl0_cclmoff import BL0CCLMoffConfig, build_bl0_with_rnafm
from utils.config import load_config
from utils.rnafm import count_parameters, rnafm_model_specs, tokenize_rnafm_sequences


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def choose_device(config: dict[str, Any]) -> torch.device:
    requested = str(config.get("device", "auto")).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("Requested CUDA but torch.cuda.is_available() is false; falling back to CPU.")
        return torch.device("cpu")
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


def build_dataset(config: dict[str, Any]) -> CCLMoffCSVDataset:
    ds_cfg = dict(config.get("dataset", {}))
    if "csv_path" not in ds_cfg:
        raise ValueError("config.dataset.csv_path is required")
    return CCLMoffCSVDataset(
        ds_cfg["csv_path"],
        max_rows=ds_cfg.get("max_rows"),
        observed_positive_cap=ds_cfg.get("observed_positive_cap"),
        unobserved_candidate_cap=ds_cfg.get("unobserved_candidate_cap"),
        seed=int(config.get("seed", 42)),
        replace_t_with_u=bool(ds_cfg.get("replace_t_with_u", True)),
    )


def evaluate_on_loader(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    labels_all: list[np.ndarray] = []
    probs_all: list[np.ndarray] = []
    with torch.no_grad():
        for tokens, labels in loader:
            tokens = tokens.to(device)
            labels = labels.to(device)
            logits = model(tokens)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            losses.append(float(loss.item()))
            labels_all.append(labels.detach().cpu().numpy())
            probs_all.append(torch.sigmoid(logits).detach().cpu().numpy())

    labels_np = np.concatenate(labels_all) if labels_all else np.array([], dtype=np.float32)
    probs_np = np.concatenate(probs_all) if probs_all else np.array([], dtype=np.float32)
    if labels_np.size and len(np.unique(labels_np)) == 2:
        auroc = float(roc_auc_score(labels_np, probs_np))
        auprc = float(average_precision_score(labels_np, probs_np))
    else:
        auroc = float("nan")
        auprc = float("nan")
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "auroc": auroc,
        "auprc": auprc,
    }


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


def write_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# BL0a Smoke Training Report",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Config: `{summary['config_path']}`",
        f"- Device: `{summary['device']}`",
        f"- Status: `{summary['status']}`",
        f"- Dataset rows used: `{summary['dataset']['rows']}`",
        f"- Label counts: `{summary['dataset']['label_counts']}`",
        f"- RNA-FM specs: `{summary['model']['rnafm_specs']}`",
        f"- Total params: `{summary['model']['total_params']}`",
        f"- Trainable params: `{summary['model']['trainable_params']}`",
        f"- Epochs: `{summary['training']['epochs']}`",
        f"- Batch size: `{summary['training']['batch_size']}`",
        f"- Final loss: `{summary['metrics']['loss']}`",
        f"- AUROC on same tiny smoke set: `{summary['metrics']['auroc']}`",
        f"- AUPRC on same tiny smoke set: `{summary['metrics']['auprc']}`",
        f"- Head-only checkpoint: `{summary['artifacts']['head_checkpoint']}`",
        "",
        "## Notes",
        "",
        "- This is a training-chain smoke test, not a reproducible BL0a benchmark.",
        "- `label=0` is treated as `unobserved_candidate`, not verified safe.",
        "- The checkpoint intentionally stores only the MLP head because RNA-FM is loaded from the audited shared checkpoint.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def train(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    torch.manual_seed(int(config.get("seed", 42)))
    np.random.seed(int(config.get("seed", 42)))

    device = choose_device(config)
    output_dir = Path(config.get("outputs", {}).get("output_dir", "results/bl0a_smoke_train"))
    rnafm_cfg = dict(config.get("rnafm", {}))
    model, alphabet = build_bl0_with_rnafm(
        checkpoint_path=rnafm_cfg.get("checkpoint_path"),
        allow_download=bool(rnafm_cfg.get("allow_download", False)),
        config=make_model_config(config),
    )
    model.to(device)

    dataset = build_dataset(config)
    training_cfg = dict(config.get("training", {}))
    batch_size = int(training_cfg.get("batch_size", 2))
    use_balanced_sampler = bool(training_cfg.get("use_balanced_sampler", True))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=dataset.get_balanced_sampler() if use_balanced_sampler else None,
        shuffle=False if use_balanced_sampler else True,
        num_workers=int(training_cfg.get("num_workers", 0)),
        collate_fn=make_collate_fn(alphabet),
    )

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(training_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-5)),
    )

    epochs = int(training_cfg.get("epochs", 1))
    losses: list[float] = []
    model.train()
    for _epoch in range(1, epochs + 1):
        for tokens, labels in loader:
            tokens = tokens.to(device)
            labels = labels.to(device)
            logits = model(tokens)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

    metrics_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(training_cfg.get("num_workers", 0)),
        collate_fn=make_collate_fn(alphabet),
    )
    metrics = evaluate_on_loader(model, metrics_loader, device)
    if losses:
        metrics["last_train_batch_loss"] = float(losses[-1])

    output_dir.mkdir(parents=True, exist_ok=True)
    head_checkpoint = output_dir / "bl0a_head_smoke.pt"
    torch.save(
        {
            "model": "bl0a_rnafm_mlp",
            "head_state_dict": model.head.state_dict(),
            "config": config,
            "rnafm_checkpoint_path": rnafm_cfg.get("checkpoint_path"),
            "label_semantics": {"1": "observed_positive", "0": "unobserved_candidate"},
        },
        head_checkpoint,
    )

    summary = {
        "generated_at": utc_now(),
        "config_path": str(config_path),
        "commit_hash": git_hash(),
        "version": str(config.get("version", "BL0a-smoke-train")),
        "status": str(config.get("status", "train_chain_passed")),
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "dataset": {
            "csv_path": str(Path(config["dataset"]["csv_path"])),
            "rows": len(dataset),
            "label_counts": dataset.label_counts(),
        },
        "model": {
            "rnafm_specs": rnafm_model_specs(model.rnafm_model),
            "total_params": count_parameters(model),
            "trainable_params": count_parameters(model, trainable_only=True),
            "freeze_rnafm": bool(rnafm_cfg.get("freeze", True)),
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": float(training_cfg.get("learning_rate", 1e-4)),
            "weight_decay": float(training_cfg.get("weight_decay", 1e-5)),
            "use_balanced_sampler": use_balanced_sampler,
        },
        "metrics": metrics,
        "artifacts": {"head_checkpoint": str(head_checkpoint)},
    }
    write_summary(output_dir, summary)

    experiment_csv = Path(config.get("outputs", {}).get("experiment_csv", "results/experiments.csv"))
    append_experiment(
        experiment_csv,
        {
            "version": summary["version"],
            "date": summary["generated_at"],
            "commit_hash": summary["commit_hash"],
            "status": summary["status"],
            "auroc": "" if np.isnan(metrics["auroc"]) else f"{metrics['auroc']:.6f}",
            "auprc": "" if np.isnan(metrics["auprc"]) else f"{metrics['auprc']:.6f}",
            "gpu_mem": "cpu" if device.type == "cpu" else f"{torch.cuda.max_memory_allocated() / 1024**3:.2f}GB",
            "epochs": epochs,
            "best_epoch": epochs,
            "config_path": str(config_path),
            "notes": str(config.get("outputs", {}).get("notes", "BL0a training-chain check; not a formal benchmark")),
        },
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train-chain smoke test for BL0a RNA-FM + MLP.")
    parser.add_argument("--config", type=Path, default=Path("configs/bl0a_smoke.yaml"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = train(args.config)
    except Exception as exc:  # noqa: BLE001 - keep command-line failure explicit.
        print(f"BL0a smoke training failed: {exc.__class__.__name__}: {exc}")
        return 1
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
