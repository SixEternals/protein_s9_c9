#!/usr/bin/env python3
"""k-NN baseline for off-target prediction.

Uses Hamming distance on sgRNA_seq + off_seq (or off_seq alone).
For efficiency, we restrict search to the same sgRNA_type and sample negatives.

Usage:
    python scripts/knn_baseline.py \
        --csv data/cclmoff/09212024_CCLMoff_dataset.csv \
        --split formal_split_bl5_seed42.json \
        --k 1 5 \
        --output results/knn_baseline_summary.json

Outputs:
    results/knn_baseline_summary.json
    results/knn_baseline_report.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def hamming_distance(a: str, b: str) -> int:
    return sum(c1 != c2 for c1, c2 in zip(a, b))


def knn_predict(train_df: pd.DataFrame, test_df: pd.DataFrame, k: int) -> np.ndarray:
    """Return mean label of k nearest train neighbors for each test row."""
    probs = []
    for _, trow in test_df.iterrows():
        # Restrict to same sgRNA_type for efficiency
        candidates = train_df[train_df["sgRNA_type"] == trow["sgRNA_type"]]
        if len(candidates) == 0:
            # Fallback to all train
            candidates = train_df
        # Compute Hamming distance on combined sequence
        dists = candidates["sgRNA_seq"].astype(str).str.cat(candidates["off_seq"].astype(str)).apply(
            lambda s: hamming_distance(s, str(trow["sgRNA_seq"]) + str(trow["off_seq"]))
        )
        # Get k nearest
        nearest = candidates.loc[dists.nsmallest(k).index]
        probs.append(float(nearest["label"].mean()))
    return np.array(probs, dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/cclmoff/09212024_CCLMoff_dataset.csv")
    parser.add_argument("--split", default="formal_split_bl5_seed42.json")
    parser.add_argument("--k", nargs="+", type=int, default=[1, 5])
    parser.add_argument("--test-sample-size", type=int, default=10000, help="Max test samples per class to evaluate")
    parser.add_argument("--output", default="results/knn_baseline_summary.json")
    args = parser.parse_args()

    print("[knn] Loading data...")
    df = pd.read_csv(args.csv, usecols=["sgRNA_seq", "off_seq", "label", "sgRNA_type"])
    df["sgRNA_type"] = df["sgRNA_type"].astype(str)

    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    train_groups = set(split["splits"]["train"]["sgRNA_types"])
    test_groups = set(split["splits"]["test"]["sgRNA_types"])

    train_df = df[df["sgRNA_type"].isin(train_groups)].copy()
    test_df = df[df["sgRNA_type"].isin(test_groups)].copy()

    # Subsample test for speed
    test_pos = test_df[test_df["label"] == 1]
    test_neg = test_df[test_df["label"] == 0]
    test_sample = pd.concat([
        test_pos.sample(min(len(test_pos), args.test_sample_size), random_state=42),
        test_neg.sample(min(len(test_neg), args.test_sample_size), random_state=42),
    ]).reset_index(drop=True)

    results = []
    for k in args.k:
        print(f"[knn] Computing {k}-NN predictions on {len(test_sample)} test samples...")
        probs = knn_predict(train_df, test_sample, k)
        labels = test_sample["label"].to_numpy(dtype=np.int64)
        preds = (probs >= 0.5).astype(np.int64)

        result = {
            "k": k,
            "test_samples": int(len(test_sample)),
            "AUROC": float(roc_auc_score(labels, probs)),
            "AUPRC": float(average_precision_score(labels, probs)),
            "Accuracy": float(accuracy_score(labels, preds)),
            "Precision": float(precision_score(labels, preds, zero_division=0)),
            "Recall": float(recall_score(labels, preds, zero_division=0)),
            "F1": float(f1_score(labels, preds, zero_division=0)),
        }
        results.append(result)
        print(f"[knn] {k}-NN AUPRC = {result['AUPRC']:.4f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"knn_results": results}, indent=2, ensure_ascii=False), encoding="utf-8")

    md = ["# k-NN Baseline Report", ""]
    for r in results:
        md.extend([
            f"## {r['k']}-NN",
            f"- test_samples: {r['test_samples']:,}",
            f"- AUROC: {r['AUROC']:.4f}",
            f"- AUPRC: {r['AUPRC']:.4f}",
            f"- Accuracy: {r['Accuracy']:.4f}",
            f"- Precision: {r['Precision']:.4f}",
            f"- Recall: {r['Recall']:.4f}",
            f"- F1: {r['F1']:.4f}",
            "",
        ])
    md_path = out_path.with_suffix(".md")
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"[knn] Wrote {out_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
