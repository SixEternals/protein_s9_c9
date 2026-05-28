#!/usr/bin/env python3
"""Paired probability comparison across BL0b / NoPAM / PAM on the same test set.

Usage:
    python scripts/paired_comparison.py \
        --bl0b results/bl0b_on_bl5split/test_predictions.csv \
        --nopam results/bl5_v4_nopam_control/test_predictions.csv \
        --pam results/bl5_v4_pam/test_predictions.csv \
        --output results/paired_comparison_test_predictions.csv

Outputs:
    results/paired_comparison_test_predictions.csv
    results/paired_comparison_summary.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def summarize(df: pd.DataFrame, subset_mask: pd.Series, subset_name: str) -> dict[str, float]:
    sub = df[subset_mask]
    if len(sub) == 0:
        return {"subset": subset_name, "n": 0}

    rows: dict[str, float | int | str] = {"subset": subset_name, "n": int(len(sub))}
    for delta_col in ("delta_nopam_minus_bl0b", "delta_pam_minus_nopam", "delta_pam_minus_bl0b"):
        arr = sub[delta_col].to_numpy(dtype=np.float64)
        rows[f"{delta_col}_mean"] = float(np.mean(arr))
        rows[f"{delta_col}_median"] = float(np.median(arr))
        rows[f"{delta_col}_q25"] = float(np.quantile(arr, 0.25))
        rows[f"{delta_col}_q75"] = float(np.quantile(arr, 0.75))
        rows[f"{delta_col}_gt0"] = float(np.mean(arr > 0))
        rows[f"{delta_col}_lt0"] = float(np.mean(arr < 0))
    return rows  # type: ignore[return-value]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bl0b", required=True)
    parser.add_argument("--nopam", required=True)
    parser.add_argument("--pam", required=True)
    parser.add_argument("--output", default="results/paired_comparison_test_predictions.csv")
    args = parser.parse_args()

    for name, p in (("BL0b", args.bl0b), ("NoPAM", args.nopam), ("PAM", args.pam)):
        if not Path(p).exists():
            print(f"Missing {name}: {p}", file=sys.stderr)
            return 1

    bl0b = pd.read_csv(args.bl0b)
    nopam = pd.read_csv(args.nopam)
    pam = pd.read_csv(args.pam)

    # Align by index order (all three use the same formal split)
    if not (len(bl0b) == len(nopam) == len(pam)):
        raise ValueError("Row count mismatch across models")

    df = bl0b[["sgRNA_type", "on_seq", "off_seq", "PAM", "label", "Direction"]].copy()
    df["prob_bl0b"] = bl0b["probability"].values
    df["prob_nopam"] = nopam["probability"].values
    df["prob_pam"] = pam["probability"].values
    df["delta_nopam_minus_bl0b"] = df["prob_nopam"] - df["prob_bl0b"]
    df["delta_pam_minus_nopam"] = df["prob_pam"] - df["prob_nopam"]
    df["delta_pam_minus_bl0b"] = df["prob_pam"] - df["prob_bl0b"]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    # Summaries
    df["is_NGG"] = df["PAM"].str[1:3] == "GG"
    summaries = []
    for mask, name in (
        (pd.Series(True, index=df.index), "All"),
        (df["label"] == 1, "Positive"),
        (df["label"] == 0, "Negative"),
        (df["is_NGG"], "NGG-only"),
        (~df["is_NGG"], "non-NGG-only"),
    ):
        summaries.append(summarize(df, mask, name))

    summary_df = pd.DataFrame(summaries)
    md_path = out_path.with_suffix(".md")
    md = ["# Paired Probability Comparison Summary", ""]
    md.append(summary_df.to_markdown(index=False))
    md.append("")
    md.append("## Key Questions")
    md.append("- Does PAM mainly increase positive probability without raising many negatives?")
    md.append(f"  - Positive mean delta (PAM - BL0b): {summary_df[summary_df['subset']=='Positive']['delta_pam_minus_bl0b_mean'].values[0]:.6f}")
    md.append(f"  - Negative mean delta (PAM - BL0b): {summary_df[summary_df['subset']=='Negative']['delta_pam_minus_bl0b_mean'].values[0]:.6f}")
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {out_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
