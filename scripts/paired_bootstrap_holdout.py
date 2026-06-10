#!/usr/bin/env python3
"""
AGENTS.md compliance: [use_rnafm=False, freeze_rnafm=False,
                       split_mode=irrelevant, pos_weight=None]
确认本文件遵守 AGENTS.md 约束

说明：本脚本不训练模型，只读取已有预测文件做统计比较。

Paired bootstrap comparison for BL5 holdout models.
Reads two test_predictions.csv files and computes paired ΔAUROC / ΔAUPRC with 95% CI.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Compute AUROC and AUPRC; return NaN if calculation fails."""
    try:
        auroc = roc_auc_score(y_true, y_score)
    except Exception:
        auroc = float("nan")
    try:
        auprc = average_precision_score(y_true, y_score)
    except Exception:
        auprc = float("nan")
    return auroc, auprc


def paired_bootstrap(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Paired bootstrap resampling for two models on the same test set."""
    rng = np.random.default_rng(seed)
    n = len(y_true)

    # Point estimates
    auroc_a, auprc_a = compute_metrics(y_true, score_a)
    auroc_b, auprc_b = compute_metrics(y_true, score_b)
    delta_auroc_point = auroc_b - auroc_a
    delta_auprc_point = auprc_b - auprc_a

    boot_auroc_a = []
    boot_auprc_a = []
    boot_auroc_b = []
    boot_auprc_b = []
    boot_delta_auroc = []
    boot_delta_auprc = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_boot = y_true[idx]
        s_a_boot = score_a[idx]
        s_b_boot = score_b[idx]

        # Skip if no observed_positive or no unobserved_candidate in bootstrap sample
        if y_boot.sum() == 0 or y_boot.sum() == n:
            continue

        auroc_a_b, auprc_a_b = compute_metrics(y_boot, s_a_boot)
        auroc_b_b, auprc_b_b = compute_metrics(y_boot, s_b_boot)

        if np.isfinite(auroc_a_b) and np.isfinite(auroc_b_b):
            boot_auroc_a.append(auroc_a_b)
            boot_auroc_b.append(auroc_b_b)
            boot_delta_auroc.append(auroc_b_b - auroc_a_b)

        if np.isfinite(auprc_a_b) and np.isfinite(auprc_b_b):
            boot_auprc_a.append(auprc_a_b)
            boot_auprc_b.append(auprc_b_b)
            boot_delta_auprc.append(auprc_b_b - auprc_a_b)

    def ci(arr: list[float]) -> tuple[float, float]:
        a = np.array(arr)
        return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

    delta_auroc_ci = ci(boot_delta_auroc)
    delta_auprc_ci = ci(boot_delta_auprc)

    return {
        "PAM": {
            "auroc": auroc_a,
            "auprc": auprc_a,
            "auroc_ci": ci(boot_auroc_a),
            "auprc_ci": ci(boot_auprc_a),
        },
        "NoPAM": {
            "auroc": auroc_b,
            "auprc": auprc_b,
            "auroc_ci": ci(boot_auroc_b),
            "auprc_ci": ci(boot_auprc_b),
        },
        "Delta": {
            "auroc": delta_auroc_point,
            "auprc": delta_auprc_point,
            "auroc_ci": delta_auroc_ci,
            "auprc_ci": delta_auprc_ci,
        },
        "n_bootstrap_valid": {
            "auroc": len(boot_delta_auroc),
            "auprc": len(boot_delta_auprc),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pam", default="results/bl5_v4_pam_holdout_agg/test_predictions.csv")
    parser.add_argument("--nopam", default="results/bl5_v4_nopam_holdout_agg/test_predictions.csv")
    parser.add_argument("--holdout-pam", default="AGG", help="Holdout PAM motif for metadata (e.g. AGG, TGG, GAG, CGG)")
    parser.add_argument("--n_bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="results/bl5_generalization/pam_strict_holdout/AGG/paired_bootstrap_results.json")
    args = parser.parse_args()

    df_pam = pd.read_csv(args.pam)
    df_nopam = pd.read_csv(args.nopam)

    if len(df_pam) != len(df_nopam):
        print(f"ERROR: Row count mismatch: PAM={len(df_pam)} NoPAM={len(df_nopam)}")
        return 1

    # Row alignment checks
    key_cols = ["sample_index", "sgRNA_type", "on_seq", "off_seq", "PAM_original", "label"]
    missing_cols_pam = [c for c in key_cols if c not in df_pam.columns]
    missing_cols_nopam = [c for c in key_cols if c not in df_nopam.columns]
    if missing_cols_pam or missing_cols_nopam:
        print(f"ERROR: Missing columns in PAM: {missing_cols_pam}, NoPAM: {missing_cols_nopam}")
        return 1

    for c in key_cols:
        if not df_pam[c].equals(df_nopam[c]):
            print(f"ERROR: Row alignment failed on column '{c}'")
            return 1

    print("✅ Row alignment check passed")

    # PAM_original sanity checks
    if not (df_pam["PAM_original"] == df_pam["off_seq"].str.slice(20, 23)).all():
        print("ERROR: PAM_original != off_seq[20:23] in PAM predictions")
        return 1
    if not (df_nopam["PAM_original"] == df_nopam["off_seq"].str.slice(20, 23)).all():
        print("ERROR: PAM_original != off_seq[20:23] in NoPAM predictions")
        return 1
    print("✅ PAM_original == off_seq[20:23] check passed")

    # Probability range checks
    if not df_pam["probability"].between(0, 1).all():
        print("ERROR: PAM probabilities outside [0, 1]")
        return 1
    if not df_nopam["probability"].between(0, 1).all():
        print("ERROR: NoPAM probabilities outside [0, 1]")
        return 1
    print("✅ Probability range [0, 1] check passed")

    # Align by row index (same test set, same order)
    y_true = df_pam["label"].values.astype(np.int64)
    score_pam = df_pam["probability"].values.astype(np.float64)
    score_nopam = df_nopam["probability"].values.astype(np.float64)

    n_pos = int(y_true.sum())
    n_neg = int((y_true == 0).sum())

    print(f"Samples: {len(y_true):,}, observed_positive: {n_pos:,}, unobserved_candidate: {n_neg:,}")
    results = paired_bootstrap(y_true, score_pam, score_nopam, n_bootstrap=args.n_bootstrap, seed=args.seed)

    # Add metadata
    delta_auroc_ci = results["Delta"]["auroc_ci"]
    delta_auprc_ci = results["Delta"]["auprc_ci"]

    full_results = {
        "holdout_pam": args.holdout_pam,
        "delta_definition": "NoPAM - PAM",
        "n_samples": len(y_true),
        "observed_positive": n_pos,
        "unobserved_candidate": n_neg,
        **results,
        "delta_AUROC_ci_crosses_zero": (delta_auroc_ci[0] <= 0) and (delta_auroc_ci[1] >= 0),
        "delta_AUPRC_ci_crosses_zero": (delta_auprc_ci[0] <= 0) and (delta_auprc_ci[1] >= 0),
    }

    print(f"\n=== Paired Bootstrap Results ({args.n_bootstrap:,} resamples) ===")
    print(f"\nPAM holdout {args.holdout_pam}:")
    print(f"  AUROC = {results['PAM']['auroc']:.6f}  95% CI: [{results['PAM']['auroc_ci'][0]:.6f}, {results['PAM']['auroc_ci'][1]:.6f}]")
    print(f"  AUPRC = {results['PAM']['auprc']:.6f}  95% CI: [{results['PAM']['auprc_ci'][0]:.6f}, {results['PAM']['auprc_ci'][1]:.6f}]")
    print(f"\nNoPAM holdout {args.holdout_pam}:")
    print(f"  AUROC = {results['NoPAM']['auroc']:.6f}  95% CI: [{results['NoPAM']['auroc_ci'][0]:.6f}, {results['NoPAM']['auroc_ci'][1]:.6f}]")
    print(f"  AUPRC = {results['NoPAM']['auprc']:.6f}  95% CI: [{results['NoPAM']['auprc_ci'][0]:.6f}, {results['NoPAM']['auprc_ci'][1]:.6f}]")
    print(f"\nDelta (NoPAM − PAM):")
    print(f"  ΔAUROC = {results['Delta']['auroc']:+.6f}  95% CI: [{results['Delta']['auroc_ci'][0]:+.6f}, {results['Delta']['auroc_ci'][1]:+.6f}]")
    print(f"  ΔAUPRC = {results['Delta']['auprc']:+.6f}  95% CI: [{results['Delta']['auprc_ci'][0]:+.6f}, {results['Delta']['auprc_ci'][1]:+.6f}]")
    print(f"\nValid bootstrap samples: AUROC={results['n_bootstrap_valid']['auroc']:,}, AUPRC={results['n_bootstrap_valid']['auprc']:,}")
    print(f"ΔAUROC CI crosses zero: {full_results['delta_AUROC_ci_crosses_zero']}")
    print(f"ΔAUPRC CI crosses zero: {full_results['delta_AUPRC_ci_crosses_zero']}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(full_results, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
