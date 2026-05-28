#!/usr/bin/env python3
"""Per-sgRNA and per-PAM analysis for BL0b / NoPAM / PAM.

Usage:
    python scripts/per_sgrna_and_pam_analysis.py \
        --bl0b results/bl0b_on_bl5split/test_predictions.csv \
        --nopam results/bl5_v4_nopam_control/test_predictions.csv \
        --pam results/bl5_v4_pam/test_predictions.csv \
        --output-prefix results/per

Outputs:
    results/per_sgrna_metrics.csv / .md
    results/per_pam_metrics.csv / .md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def safe_auprc(labels: np.ndarray, probs: np.ndarray) -> float | str:
    if len(np.unique(labels)) < 2:
        return "NA (single class)"
    return float(average_precision_score(labels, probs))


def safe_auroc(labels: np.ndarray, probs: np.ndarray) -> float | str:
    if len(np.unique(labels)) < 2:
        return "NA (single class)"
    return float(roc_auc_score(labels, probs))


def per_group_analysis(
    df: pd.DataFrame, group_col: str, models: dict[str, str]
) -> pd.DataFrame:
    rows = []
    for group, sub in df.groupby(group_col):
        labels = sub["label"].to_numpy(dtype=np.int64)
        n_pos = int((labels == 1).sum())
        n_neg = int((labels == 0).sum())
        row: dict[str, object] = {
            group_col: group,
            "samples": len(sub),
            "positive": n_pos,
            "negative": n_neg,
            "positive_ratio": float(n_pos / len(sub)) if len(sub) else 0.0,
        }
        for model_name, prob_col in models.items():
            probs = sub[prob_col].to_numpy(dtype=np.float64)
            row[f"{model_name}_mean_prob"] = float(np.mean(probs))
            row[f"{model_name}_AUPRC"] = safe_auprc(labels, probs)
            row[f"{model_name}_AUROC"] = safe_auroc(labels, probs)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bl0b", required=True)
    parser.add_argument("--nopam", required=True)
    parser.add_argument("--pam", required=True)
    parser.add_argument("--output-prefix", default="results/per")
    args = parser.parse_args()

    for name, p in (("BL0b", args.bl0b), ("NoPAM", args.nopam), ("PAM", args.pam)):
        if not Path(p).exists():
            print(f"Missing {name}: {p}", file=sys.stderr)
            return 1

    bl0b = pd.read_csv(args.bl0b)
    nopam = pd.read_csv(args.nopam)
    pam = pd.read_csv(args.pam)

    df = bl0b[["sgRNA_type", "PAM", "label"]].copy()
    df["prob_bl0b"] = bl0b["probability"].values
    df["prob_nopam"] = nopam["probability"].values
    df["prob_pam"] = pam["probability"].values

    models = {"BL0b": "prob_bl0b", "NoPAM": "prob_nopam", "PAM": "prob_pam"}

    out_dir = Path(args.output_prefix).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-sgRNA
    sgrna_df = per_group_analysis(df, "sgRNA_type", models)
    sgrna_df["NoPAM_minus_BL0b"] = sgrna_df["NoPAM_AUPRC"] - sgrna_df["BL0b_AUPRC"]
    sgrna_df["PAM_minus_NoPAM"] = sgrna_df["PAM_AUPRC"] - sgrna_df["NoPAM_AUPRC"]
    sgrna_df["PAM_minus_BL0b"] = sgrna_df["PAM_AUPRC"] - sgrna_df["BL0b_AUPRC"]
    sgrna_path = out_dir / "per_sgrna_metrics.csv"
    sgrna_df.to_csv(sgrna_path, index=False)

    # Per-PAM
    pam_df = per_group_analysis(df, "PAM", models)
    pam_df["NoPAM_minus_BL0b"] = pam_df["NoPAM_AUPRC"] - pam_df["BL0b_AUPRC"]
    pam_df["PAM_minus_NoPAM"] = pam_df["PAM_AUPRC"] - pam_df["NoPAM_AUPRC"]
    pam_df["PAM_minus_BL0b"] = pam_df["PAM_AUPRC"] - pam_df["BL0b_AUPRC"]
    pam_path = out_dir / "per_pam_metrics.csv"
    pam_df.to_csv(pam_path, index=False)

    # Markdown summaries
    for path, title, group_col in (
        (sgrna_path, "Per-sgRNA Metrics", "sgRNA_type"),
        (pam_path, "Per-PAM Metrics", "PAM"),
    ):
        sub_df = pd.read_csv(path)
        md = [f"# {title}", ""]
        md.append(sub_df.to_markdown(index=False))
        md.append("")
        md_path = path.with_suffix(".md")
        md_path.write_text("\n".join(md), encoding="utf-8")
        print(f"Wrote {path} and {md_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
