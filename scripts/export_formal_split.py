#!/usr/bin/env python3
"""Export the formal BL5 seed-42 sgRNA-safe split.

AGENTS.md compliance: [use_rnafm=False, freeze_rnafm=N/A,
                       split_mode=sgrna_safe, pos_weight=None]
确认本文件遵守 AGENTS.md 约束：只导出分组 split，不训练模型，不改变数据。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def make_bl5_group_split(labels: np.ndarray, seed: int, group_labels: np.ndarray) -> dict[str, np.ndarray]:
    """Replicate scripts/train_bl5.py::make_split for sgrna_safe mode."""
    indices = np.arange(len(labels), dtype=np.int64)
    unique_groups = np.unique(group_labels)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)
    group_counts = {group: int(np.sum(group_labels == group)) for group in unique_groups}
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


def label_counts(labels: np.ndarray) -> dict[str, int | float]:
    observed = int((labels == 1).sum())
    unobserved = int((labels == 0).sum())
    total = int(labels.shape[0])
    return {
        "samples": total,
        "observed_positive": observed,
        "unobserved_candidate": unobserved,
        "positive_ratio": float(observed / total) if total else 0.0,
    }


def split_payload(
    *,
    csv_path: Path,
    npz_path: Path,
    group_column: str,
    seed: int,
) -> dict[str, Any]:
    npz = np.load(npz_path, allow_pickle=False)
    labels = npz["y"].astype(np.int64, copy=False)
    frame = pd.read_csv(csv_path, usecols=[group_column, "label"])
    if len(frame) != len(labels):
        raise ValueError(f"CSV/NPZ row mismatch: csv={len(frame)} npz={len(labels)}")
    csv_labels = frame["label"].to_numpy(dtype=np.int64)
    label_match = bool(np.array_equal(csv_labels, labels))
    if not label_match:
        raise ValueError("CSV label column does not match NPZ y array; refusing to export split")

    groups = frame[group_column].astype(str).to_numpy()
    split_indices = make_bl5_group_split(labels, seed, groups)

    payload: dict[str, Any] = {
        "version": "formal_split_bl5_seed42",
        "seed": int(seed),
        "split_mode": "sgrna_safe",
        "split_logic": "BL5 make_split: np.unique(group_labels), rng.shuffle(seed), cumulative group counts at 70%/85%",
        "fractions": {"train": 0.70, "val": 0.15, "test": 0.15},
        "data": {
            "cclmoff_csv": str(csv_path),
            "npz_path": str(npz_path),
            "group_column": group_column,
            "rows": int(len(labels)),
            "unique_groups": int(len(np.unique(groups))),
            "label_source": "npz['y']; verified equal to CSV label",
        },
        "splits": {},
        "leakage_check": {},
    }

    split_group_sets: dict[str, set[str]] = {}
    for name in ("train", "val", "test"):
        idx = split_indices[name]
        split_labels = labels[idx]
        split_groups = sorted(set(groups[idx].tolist()))
        split_group_sets[name] = set(split_groups)
        payload["splits"][name] = {
            **label_counts(split_labels),
            "sgRNA_type_count": int(len(split_groups)),
            "sgRNA_types": split_groups,
        }

    payload["leakage_check"] = {
        "train_val_overlap": sorted(split_group_sets["train"] & split_group_sets["val"]),
        "train_test_overlap": sorted(split_group_sets["train"] & split_group_sets["test"]),
        "val_test_overlap": sorted(split_group_sets["val"] & split_group_sets["test"]),
    }
    if any(payload["leakage_check"].values()):
        raise RuntimeError(f"sgRNA_type leakage detected: {payload['leakage_check']}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export formal BL5 seed-42 split groups.")
    parser.add_argument("--csv", type=Path, default=Path("data/cclmoff/09212024_CCLMoff_dataset.csv"))
    parser.add_argument("--npz", type=Path, default=Path("data/cclmoff/cclmoff_9bit.npz"))
    parser.add_argument("--group-column", default="sgRNA_type")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("formal_split_bl5_seed42.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = split_payload(
        csv_path=args.csv,
        npz_path=args.npz,
        group_column=args.group_column,
        seed=args.seed,
    )
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for name in ("train", "val", "test"):
        split = payload["splits"][name]
        print(
            f"{name}: samples={split['samples']} positives={split['observed_positive']} "
            f"unobserved={split['unobserved_candidate']} ratio={split['positive_ratio']:.6f} "
            f"sgRNA_type_count={split['sgRNA_type_count']}"
        )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
