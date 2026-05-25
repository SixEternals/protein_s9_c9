from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.bl0_cclmoff import BL0CCLMoffConfig, BL0CCLMoffHead, build_bl0_with_rnafm
from utils.config import load_config
from utils.rnafm import normalize_pair_sequence, tokenize_rnafm_sequences


class PairSequenceDataset(Dataset):
    def __init__(
        self,
        on_seq: np.ndarray,
        off_seq: np.ndarray,
        labels: np.ndarray,
        pair_joiner: str,
        replace_t_with_u: bool,
    ):
        self.on_seq = on_seq.astype(str)
        self.off_seq = off_seq.astype(str)
        self.labels = labels.astype(np.float32)
        self.pair_joiner = pair_joiner
        self.replace_t_with_u = replace_t_with_u

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[str, float]:
        if self.pair_joiner != "<sep>":
            raise ValueError("RNA-FM BL0 currently supports only '<sep>' as pair_joiner")
        seq = normalize_pair_sequence(
            str(self.on_seq[index]),
            str(self.off_seq[index]),
            replace_t_with_u=self.replace_t_with_u,
        )
        return seq, float(self.labels[index])


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


def make_config(payload: dict[str, Any]) -> BL0CCLMoffConfig:
    model_cfg = dict(payload.get("model_config", {}))
    rnafm_cfg = dict(payload.get("rnafm", {}))
    model_cfg["freeze_rnafm"] = bool(rnafm_cfg.get("freeze", True))
    model_cfg["repr_layer"] = int(rnafm_cfg.get("repr_layer", model_cfg.get("repr_layer", 12)))
    return BL0CCLMoffConfig(**model_cfg)


def sample_indices(labels: np.ndarray, observed_cap: int | None, unobserved_cap: int | None, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    observed = np.flatnonzero(labels == 1)
    unobserved = np.flatnonzero(labels == 0)
    if observed_cap is not None and observed.shape[0] > observed_cap:
        observed = rng.choice(observed, size=observed_cap, replace=False)
    if unobserved_cap is not None and unobserved.shape[0] > unobserved_cap:
        unobserved = rng.choice(unobserved, size=unobserved_cap, replace=False)
    indices = np.concatenate([observed, unobserved])
    rng.shuffle(indices)
    return indices


def split_indices(labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
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


def load_pair_dataset(config: dict[str, Any]) -> tuple[PairSequenceDataset, np.ndarray]:
    dataset_files = [Path(item) for item in config.get("dataset_files", [])]
    if not dataset_files:
        raise ValueError("configs/bl0.yaml must define dataset_files")
    if len(dataset_files) != 1:
        raise ValueError("P0 BL0 trainer currently expects one dataset file; merge later only after BL0 is stable.")

    sampling = config.get("sampling", {})
    observed_cap = sampling.get("observed_positive_cap")
    unobserved_cap = sampling.get("unobserved_candidate_cap")
    seed = int(config.get("seed", 42))
    rnafm_cfg = config.get("rnafm", {})
    pair_joiner = str(rnafm_cfg.get("pair_joiner", "<sep>"))
    replace_t_with_u = bool(rnafm_cfg.get("replace_t_with_u", True))

    with np.load(dataset_files[0], allow_pickle=False) as data:
        labels_all = np.asarray(data["y"], dtype=np.float32)
        indices = sample_indices(labels_all, observed_cap, unobserved_cap, seed)
        dataset = PairSequenceDataset(
            on_seq=np.asarray(data["on_seq"][indices], dtype=str),
            off_seq=np.asarray(data["off_seq"][indices], dtype=str),
            labels=labels_all[indices],
            pair_joiner=pair_joiner,
            replace_t_with_u=replace_t_with_u,
        )
    return dataset, dataset.labels


def make_collate_fn(alphabet: Any):
    def collate(batch: list[tuple[str, float]]) -> tuple[torch.Tensor, torch.Tensor]:
        seqs, labels = zip(*batch)
        tokens = tokenize_rnafm_sequences(alphabet, list(seqs))
        return tokens, torch.tensor(labels, dtype=torch.float32)

    return collate


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    labels_all: list[np.ndarray] = []
    probs_all: list[np.ndarray] = []
    losses: list[float] = []
    for tokens, labels in loader:
        tokens = tokens.to(device)
        labels = labels.to(device)
        logits = model(tokens)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        losses.append(float(loss.item()))
        labels_all.append(labels.detach().cpu().numpy())
        probs_all.append(torch.sigmoid(logits).detach().cpu().numpy())
    labels_np = np.concatenate(labels_all)
    probs_np = np.concatenate(probs_all)
    if len(np.unique(labels_np)) < 2:
        auroc = 0.5
        auprc = 0.0
    else:
        auroc = float(roc_auc_score(labels_np, probs_np))
        auprc = float(average_precision_score(labels_np, probs_np))
    return {"loss": float(np.mean(losses)) if losses else 0.0, "auroc": auroc, "auprc": auprc}


def append_experiment(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def run_smoke(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    torch.manual_seed(int(config.get("seed", 42)))
    bl0_config = make_config(config)
    head = BL0CCLMoffHead(bl0_config)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4)
    embeddings = torch.randn(8, 47, bl0_config.input_dim)
    labels = torch.randint(0, 2, (8,), dtype=torch.float32)
    logits = head(embeddings)
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()
    optimizer.step()
    audit_dir = Path(config.get("outputs", {}).get("audit_dir", "results/audits"))
    audit_dir.mkdir(parents=True, exist_ok=True)
    report = audit_dir / "bl0_smoke_report.md"
    report.write_text(
        "\n".join(
            [
                "# BL0 Smoke Report",
                "",
                f"- Generated at: `{utc_now()}`",
                f"- Config: `{config_path}`",
                f"- Input embeddings: `{list(embeddings.shape)}`",
                f"- Output logits: `{list(logits.shape)}`",
                f"- Smoke loss: `{float(loss.item()):.6f}`",
                "- This is not a BL0 training metric and must not be used for `BL0-v1.0`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {"status": "smoke_passed", "loss": float(loss.item()), "report": str(report)}


def run_train(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    device = choose_device(config)
    rnafm_cfg = config.get("rnafm", {})
    model, alphabet = build_bl0_with_rnafm(
        checkpoint_path=rnafm_cfg.get("checkpoint_path"),
        allow_download=bool(rnafm_cfg.get("allow_download", False)),
        config=make_config(config),
    )
    model.to(device)
    dataset, labels = load_pair_dataset(config)
    train_idx, val_idx, test_idx = split_indices(labels, int(config.get("seed", 42)))
    collate_fn = make_collate_fn(alphabet)
    training = config.get("training", {})
    batch_size = int(training.get("batch_size", 8))
    num_workers = int(training.get("num_workers", 0))
    loaders = {
        "train": DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers),
        "val": DataLoader(torch.utils.data.Subset(dataset, val_idx), batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers),
        "test": DataLoader(torch.utils.data.Subset(dataset, test_idx), batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers),
    }
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(training.get("learning_rate", 1e-4)),
        weight_decay=float(training.get("weight_decay", 1e-5)),
    )
    epochs = int(training.get("epochs", 1))
    best_val = {"auprc": -1.0, "epoch": 0, "state": None}
    for epoch in range(1, epochs + 1):
        model.train()
        for tokens, labels_batch in loaders["train"]:
            tokens = tokens.to(device)
            labels_batch = labels_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(tokens)
            loss = F.binary_cross_entropy_with_logits(logits, labels_batch)
            loss.backward()
            optimizer.step()
        val = evaluate(model, loaders["val"], device)
        if val["auprc"] > best_val["auprc"]:
            best_val = {"auprc": val["auprc"], "epoch": epoch, "state": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
    if best_val["state"] is not None:
        model.load_state_dict(best_val["state"])
    test = evaluate(model, loaders["test"], device)
    experiment_csv = Path(config.get("outputs", {}).get("experiment_csv", "results/experiments.csv"))
    gpu_mem = ""
    if device.type == "cuda":
        gpu_mem = f"{torch.cuda.max_memory_allocated() / (1024 ** 3):.2f}GB"
    append_experiment(
        experiment_csv,
        {
            "version": "BL0",
            "date": utc_now(),
            "commit_hash": git_hash(),
            "status": "trained",
            "auroc": f"{test['auroc']:.6f}",
            "auprc": f"{test['auprc']:.6f}",
            "gpu_mem": gpu_mem or "cpu",
            "epochs": epochs,
            "best_epoch": best_val["epoch"],
            "config_path": str(config_path),
            "notes": "BL0 official CCLMoff MLP head; no region/run encoders.",
        },
    )
    return {"status": "trained", "test": test, "best_epoch": best_val["epoch"]}


def p0_status_from_config(config: dict[str, Any], exc: Exception) -> tuple[str, str]:
    data_status = str(config.get("data_source_status", "")).strip()
    status = "blocked"
    if data_status:
        status = data_status
    notes = f"{exc.__class__.__name__}: {exc}"
    if data_status:
        notes = f"{data_status}; {notes}"
    return status, notes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or smoke-test BL0 CCLMoff-style RNA-FM baseline.")
    parser.add_argument("--config", type=Path, default=Path("configs/bl0.yaml"))
    parser.add_argument("--mode", choices=["auto", "smoke", "train"], default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    mode = args.mode if args.mode != "auto" else str(config.get("mode", "smoke"))
    try:
        if mode == "smoke":
            result = run_smoke(config, args.config)
        elif mode == "train":
            result = run_train(config, args.config)
        else:
            raise ValueError(f"unsupported mode: {mode}")
        print(result)
        return 0
    except Exception as exc:  # noqa: BLE001 - P0 needs a durable blocker record.
        experiment_csv = Path(config.get("outputs", {}).get("experiment_csv", "results/experiments.csv"))
        status, notes = p0_status_from_config(config, exc)
        append_experiment(
            experiment_csv,
            {
                "version": "BL0-P0",
                "date": utc_now(),
                "commit_hash": git_hash(),
                "status": status,
                "config_path": str(args.config),
                "notes": notes,
            },
        )
        print(f"blocked: {exc.__class__.__name__}: {exc}")
        print(f"Wrote blocker row to {experiment_csv}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
