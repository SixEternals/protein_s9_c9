#!/usr/bin/env python3
"""Stratified evaluation: All / NGG-only / non-NGG-only for BL0b / NoPAM / PAM.

Usage:
    python scripts/eval_stratified_by_pam.py \
        --bl0b results/bl0b_on_bl5split/test_predictions.csv \
        --nopam results/bl5_v4_nopam_control/test_predictions.csv \
        --pam results/bl5_v4_pam/test_predictions.csv \
        --output results/stratified_metrics_all_ngg_nongg.csv

Outputs:
    results/stratified_metrics_all_ngg_nongg.csv
    results/stratified_metrics_all_ngg_nongg.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def compute_metrics(df: pd.DataFrame) -> dict[str, float]:
    labels = df["label"].to_numpy(dtype=np.int64)
    probs = df["probability"].to_numpy(dtype=np.float64)
    preds = (probs >= 0.5).astype(np.int64)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())

    result: dict[str, float | str] = {
        "samples": int(len(labels)),
        "positive": n_pos,
        "negative": n_neg,
        "positive_ratio": float(n_pos / len(labels)) if len(labels) else 0.0,
    }

    if n_pos > 0 and n_neg > 0:
        result["AUROC"] = float(roc_auc_score(labels, probs))
        result["AUPRC"] = float(average_precision_score(labels, probs))
    else:
        result["AUROC"] = "undefined (single class)"
        result["AUPRC"] = "undefined (single class)"

    result["Accuracy"] = float(accuracy_score(labels, preds))
    result["Precision"] = float(precision_score(labels, preds, zero_division=0))
    result["Recall"] = float(recall_score(labels, preds, zero_division=0))
    result["F1"] = float(f1_score(labels, preds, zero_division=0))

    if n_pos > 0:
        pos_probs = probs[labels == 1]
        result["mean_prob_positive"] = float(np.mean(pos_probs))
        result["median_prob_positive"] = float(np.median(pos_probs))
    else:
        result["mean_prob_positive"] = "N/A"
        result["median_prob_positive"] = "N/A"

    if n_neg > 0:
        neg_probs = probs[labels == 0]
        result["mean_prob_negative"] = float(np.mean(neg_probs))
        result["median_prob_negative"] = float(np.median(neg_probs))
    else:
        result["mean_prob_negative"] = "N/A"
        result["median_prob_negative"] = "N/A"

    return result  # type: ignore[return-value]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bl0b", required=True)
    parser.add_argument("--nopam", required=True)
    parser.add_argument("--pam", required=True)
    parser.add_argument("--output", default="results/stratified_metrics_all_ngg_nongg.csv")
    args = parser.parse_args()

    paths = {"BL0b": args.bl0b, "NoPAM": args.nopam, "PAM": args.pam}
    for name, p in paths.items():
        if not Path(p).exists():
            print(f"Missing {name}: {p}", file=sys.stderr)
            return 1

    rows = []
    for name, p in paths.items():
        df = pd.read_csv(p)
        df["PAM"] = df["off_seq"].astype(str).str[-3:]
        df["is_NGG"] = df["PAM"].str[1:3] == "GG"

        for subset_name, mask in (
            ("All", pd.Series(True, index=df.index)),
            ("NGG-only", df["is_NGG"]),
            ("non-NGG-only", ~df["is_NGG"]),
        ):
            sub = df[mask]
            metrics = compute_metrics(sub)
            rows.append({"model": name, "subset": subset_name, **metrics})

    out_df = pd.DataFrame(rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    # Markdown
    md = ["# Stratified Metrics by PAM Type", ""]
    for subset in ("All", "NGG-only", "non-NGG-only"):
        md.append(f"## {subset}")
        sub_df = out_df[out_df["subset"] == subset]
        md.append(sub_df.to_markdown(index=False))
        md.append("")

    md_path = out_path.with_suffix(".md")
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {out_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
