#!/usr/bin/env python3
"""Nearest-neighbor similarity analysis between train and test.

Computes minimum Hamming distance from each test sample to the train set.
Uses sgRNA_type stratification for efficiency.

Usage:
    python scripts/train_test_nearest_neighbor.py \
        --csv data/cclmoff/09212024_CCLMoff_dataset.csv \
        --split formal_split_bl5_seed42.json \
        --output results/train_test_nearest_neighbor_audit.json

Outputs:
    results/train_test_nearest_neighbor_audit.json
    results/train_test_nearest_neighbor_audit.md
    results/nearest_distance_histogram_positive.png
    results/nearest_distance_histogram_negative.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def hamming(a: str, b: str) -> int:
    return sum(c1 != c2 for c1, c2 in zip(a, b))


def nearest_distances(test_df: pd.DataFrame, train_df: pd.DataFrame, sample_size: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return array of minimum Hamming distances for test rows."""
    if sample_size is not None and len(test_df) > sample_size:
        test_df = test_df.sample(sample_size, random_state=42)

    dists = []
    for _, trow in test_df.iterrows():
        t_seq = str(trow["sgRNA_seq"]) + str(trow["off_seq"])
        # Same sgRNA_type only
        candidates = train_df[train_df["sgRNA_type"] == trow["sgRNA_type"]]
        if len(candidates) == 0:
            candidates = train_df
        c_seqs = (candidates["sgRNA_seq"].astype(str) + candidates["off_seq"].astype(str)).tolist()
        min_dist = min(hamming(t_seq, c_seq) for c_seq in c_seqs)
        dists.append(min_dist)
    return np.array(dists, dtype=np.int64), test_df["label"].to_numpy(dtype=np.int64)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/cclmoff/09212024_CCLMoff_dataset.csv")
    parser.add_argument("--split", default="formal_split_bl5_seed42.json")
    parser.add_argument("--sample-size", type=int, default=5000, help="Max test samples per class")
    parser.add_argument("--output", default="results/train_test_nearest_neighbor_audit.json")
    args = parser.parse_args()

    print("[nn] Loading data...")
    df = pd.read_csv(args.csv, usecols=["sgRNA_seq", "off_seq", "label", "sgRNA_type"])
    df["sgRNA_type"] = df["sgRNA_type"].astype(str)

    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    train_groups = set(split["splits"]["train"]["sgRNA_types"])
    test_groups = set(split["splits"]["test"]["sgRNA_types"])

    train_df = df[df["sgRNA_type"].isin(train_groups)].copy()
    test_df = df[df["sgRNA_type"].isin(test_groups)].copy()

    print(f"[nn] Train: {len(train_df):,}, Test: {len(test_df):,}")

    test_pos = test_df[test_df["label"] == 1]
    test_neg = test_df[test_df["label"] == 0]

    print(f"[nn] Sampling up to {args.sample_size} positives and {args.sample_size} negatives...")
    pos_dists, pos_labels = nearest_distances(test_pos, train_df, args.sample_size)
    neg_dists, neg_labels = nearest_distances(test_neg, train_df, args.sample_size)

    def summarize(dists: np.ndarray) -> dict[str, float]:
        return {
            "n": int(len(dists)),
            "mean": float(np.mean(dists)),
            "median": float(np.median(dists)),
            "max": int(np.max(dists)),
            "min": int(np.min(dists)),
            "prop_dist_0": float(np.mean(dists == 0)),
            "prop_dist_le1": float(np.mean(dists <= 1)),
            "prop_dist_le2": float(np.mean(dists <= 2)),
            "prop_dist_le3": float(np.mean(dists <= 3)),
        }

    audit = {
        "test_positive_nearest": summarize(pos_dists),
        "test_negative_nearest": summarize(neg_dists),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    # Try to generate histograms if matplotlib is available
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(pos_dists, bins=range(0, int(max(pos_dists.max(), neg_dists.max())) + 2), alpha=0.7, color="red")
        axes[0].set_title("Test Positive: Nearest Train Distance")
        axes[0].set_xlabel("Hamming Distance")
        axes[0].set_ylabel("Count")

        axes[1].hist(neg_dists, bins=range(0, int(max(pos_dists.max(), neg_dists.max())) + 2), alpha=0.7, color="blue")
        axes[1].set_title("Test Negative: Nearest Train Distance")
        axes[1].set_xlabel("Hamming Distance")
        axes[1].set_ylabel("Count")

        fig.tight_layout()
        hist_path = out_path.parent / "nearest_distance_histogram.png"
        fig.savefig(hist_path, dpi=150)
        print(f"[nn] Saved histogram to {hist_path}")
    except Exception as exc:
        print(f"[nn] Histogram generation skipped: {exc}")

    md = [
        "# Train-Test Nearest Neighbor Audit",
        "",
        "## Test Positive Nearest Distance",
        f"- n: {audit['test_positive_nearest']['n']:,}",
        f"- mean: {audit['test_positive_nearest']['mean']:.2f}",
        f"- median: {audit['test_positive_nearest']['median']:.2f}",
        f"- prop_dist_0: {audit['test_positive_nearest']['prop_dist_0']:.4f}",
        f"- prop_dist_le1: {audit['test_positive_nearest']['prop_dist_le1']:.4f}",
        f"- prop_dist_le2: {audit['test_positive_nearest']['prop_dist_le2']:.4f}",
        "",
        "## Test Negative Nearest Distance",
        f"- n: {audit['test_negative_nearest']['n']:,}",
        f"- mean: {audit['test_negative_nearest']['mean']:.2f}",
        f"- median: {audit['test_negative_nearest']['median']:.2f}",
        f"- prop_dist_0: {audit['test_negative_nearest']['prop_dist_0']:.4f}",
        f"- prop_dist_le1: {audit['test_negative_nearest']['prop_dist_le1']:.4f}",
        f"- prop_dist_le2: {audit['test_negative_nearest']['prop_dist_le2']:.4f}",
    ]
    md_path = out_path.with_suffix(".md")
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"[nn] Wrote {out_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
