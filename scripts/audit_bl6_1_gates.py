#!/usr/bin/env python3
"""BL6-1 gate audit: descriptive analysis of per-sample gate weights.

AGENTS.md compliance: [use_rnafm=False, freeze_rnafm=N/A,
                       split_mode=sgrna_safe, pos_weight=N/A,
                       analysis_only=True]
确认本文件遵守 AGENTS.md 约束：本脚本只读取已导出的 BL6-1 gate_predictions.csv 做描述性 audit；
不训练、不加载 checkpoint、不做模型 forward；PAM 坐标使用 off_seq[20:23]。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "sample_index", "sgRNA_type", "on_seq", "off_seq", "PAM_original",
    "label", "probability", "gate_rnafm", "gate_run", "gate_pam",
    "gate_sum", "gate_entropy", "gate_max", "gate_argmax", "pam_family", "split",
]

EXPECTED_ROWS = 954326
EXPECTED_OBSERVED_POSITIVE = 3057
EXPECTED_UNOBSERVED_CANDIDATE = 951269

GATE_COLS = ["gate_rnafm", "gate_run", "gate_pam"]


def validate_input(df: pd.DataFrame) -> dict[str, Any]:
    """Run input validation; raise ValueError on hard failures."""
    result: dict[str, Any] = {"passed": True, "checks": []}

    # Required columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    check_ok = len(missing) == 0
    result["checks"].append({"check": "required_columns", "ok": check_ok, "missing": missing})
    if not check_ok:
        result["passed"] = False
        raise ValueError(f"Missing columns: {missing}")

    # Row count
    rows = int(len(df))
    rows_ok = rows == EXPECTED_ROWS
    result["checks"].append({"check": "rows", "ok": rows_ok, "actual": rows, "expected": EXPECTED_ROWS})
    if not rows_ok:
        result["passed"] = False
        raise ValueError(f"Row count mismatch: {rows} vs {EXPECTED_ROWS}")

    # Label counts
    label_counts = df["label"].value_counts().to_dict()
    obs_pos = int(label_counts.get(1, 0))
    unobs = int(label_counts.get(0, 0))
    labels_ok = obs_pos == EXPECTED_OBSERVED_POSITIVE and unobs == EXPECTED_UNOBSERVED_CANDIDATE
    result["checks"].append({"check": "label_counts", "ok": labels_ok,
                             "observed_positive": obs_pos, "unobserved_candidate": unobs})

    # Gate ranges
    gate_min = float(df[GATE_COLS].min().min())
    gate_max = float(df[GATE_COLS].max().max())
    gates_ok = 0.0 <= gate_min and gate_max <= 1.0
    result["checks"].append({"check": "gate_range_01", "ok": gates_ok, "min": gate_min, "max": gate_max})

    # Gate sum
    gate_sum_err = float((df["gate_sum"] - 1.0).abs().max())
    gate_sum_ok = gate_sum_err <= 1e-5
    result["checks"].append({"check": "gate_sum_approx_1", "ok": gate_sum_ok, "max_error": gate_sum_err})

    # PAM coordinate
    pam_from_off = df["off_seq"].astype(str).str.slice(20, 23)
    pam_ok = bool((df["PAM_original"].astype(str) == pam_from_off).all())
    result["checks"].append({"check": "pam_coordinate", "ok": pam_ok})

    # pam_family valid
    valid_families = {"NGG", "non-NGG"}
    families = set(df["pam_family"].unique())
    family_ok = families.issubset(valid_families)
    result["checks"].append({"check": "pam_family_values", "ok": family_ok, "found": sorted(families)})

    # split all "test"
    split_ok = bool((df["split"] == "test").all())
    result["checks"].append({"check": "split_all_test", "ok": split_ok})

    # probability in [0,1]
    p_min = float(df["probability"].min())
    p_max = float(df["probability"].max())
    prob_ok = 0.0 <= p_min and p_max <= 1.0
    result["checks"].append({"check": "probability_range_01", "ok": prob_ok, "min": p_min, "max": p_max})

    result["passed"] = all(c["ok"] for c in result["checks"])
    if not result["passed"]:
        raise ValueError(f"Validation failed: {json.dumps(result['checks'], indent=2)}")
    return result


def quantile_stats(series: pd.Series) -> dict[str, float]:
    """Compute mean, std, and common quantiles."""
    qs = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    vals = series.quantile(qs).to_dict()
    return {
        "mean": float(series.mean()),
        "std": float(series.std()),
        "min": float(series.min()),
        "p01": float(vals[0.01]),
        "p05": float(vals[0.05]),
        "p25": float(vals[0.25]),
        "p50": float(vals[0.50]),
        "p75": float(vals[0.75]),
        "p95": float(vals[0.95]),
        "p99": float(vals[0.99]),
        "max": float(series.max()),
    }


def gate_stats_df(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Return a DataFrame with quantile stats per gate column."""
    rows = []
    for c in cols:
        s = quantile_stats(df[c])
        s["column"] = c
        rows.append(s)
    result = pd.DataFrame(rows)
    cols_order = ["column", "mean", "std", "min", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "max"]
    return result[cols_order]


def gate_concentration(df: pd.DataFrame, threshold: float = 0.99) -> dict[str, float]:
    """Fraction of rows where each gate >= threshold."""
    fracs = {}
    for c in GATE_COLS:
        for t in [0.90, 0.95, 0.99]:
            fracs[f"fraction_{c}_ge_{str(t).replace('.', '_')}"] = float((df[c] >= t).mean())
    for t in [0.90, 0.95, 0.99]:
        fracs[f"fraction_gate_max_ge_{str(t).replace('.', '_')}"] = float((df["gate_max"] >= t).mean())
    return fracs


def stratified_stats(df: pd.DataFrame, group_col: str, extra_cols: list[str] | None = None) -> pd.DataFrame:
    """Compute per-group gate statistics."""
    groups = []
    for name, grp in df.groupby(group_col, sort=False):
        n = len(grp)
        obs_pos = int((grp["label"] == 1).sum())
        unobs = int((grp["label"] == 0).sum())
        row: dict[str, Any] = {
            group_col: name,
            "n": n,
            "observed_positive": obs_pos,
            "unobserved_candidate": unobs,
            "observed_positive_rate": float(obs_pos / max(n, 1)),
            "probability_mean": float(grp["probability"].mean()),
            "probability_median": float(grp["probability"].median()),
            "probability_p95": float(grp["probability"].quantile(0.95)),
            "probability_p99": float(grp["probability"].quantile(0.99)),
        }
        for g in GATE_COLS:
            row[f"{g}_mean"] = float(grp[g].mean())
            row[f"{g}_median"] = float(grp[g].median())
            row[f"{g}_p95"] = float(grp[g].quantile(0.95))
            row[f"{g}_p99"] = float(grp[g].quantile(0.99))
        row["gate_entropy_mean"] = float(grp["gate_entropy"].mean())
        row["gate_entropy_median"] = float(grp["gate_entropy"].median())
        row["gate_entropy_p95"] = float(grp["gate_entropy"].quantile(0.95))
        row["gate_entropy_p99"] = float(grp["gate_entropy"].quantile(0.99))
        for view in ["rnafm", "run", "pam"]:
            count = int((grp["gate_argmax"] == view).sum())
            row[f"gate_argmax_{view}_count"] = count
            row[f"gate_argmax_{view}_fraction"] = float(count / max(n, 1))
        for t in [0.99]:
            for g in GATE_COLS:
                row[f"fraction_{g}_ge_{str(t).replace('.', '_')}"] = float((grp[g] >= t).mean())
        if extra_cols:
            for ec in extra_cols:
                if ec in grp.columns:
                    row[ec] = grp[ec].iloc[0] if len(grp) > 0 else None
        groups.append(row)
    return pd.DataFrame(groups)


def topk_analysis(df: pd.DataFrame, ks: list[int]) -> pd.DataFrame:
    """Compute gate statistics per Top-K slice."""
    sorted_df = df.sort_values("probability", ascending=False)
    rows = []
    for k in ks:
        topk = sorted_df.head(k)
        n = len(topk)
        obs_pos = int((topk["label"] == 1).sum())
        row: dict[str, Any] = {
            "k": k,
            "n": n,
            "observed_positive_hits": obs_pos,
            "observed_positive_hits_rate": float(obs_pos / max(n, 1)),
        }
        for g in GATE_COLS:
            row[f"{g}_mean"] = float(topk[g].mean())
            row[f"{g}_median"] = float(topk[g].median())
            row[f"{g}_p95"] = float(topk[g].quantile(0.95))
            row[f"{g}_p99"] = float(topk[g].quantile(0.99))
        row["gate_entropy_mean"] = float(topk["gate_entropy"].mean())
        row["gate_entropy_median"] = float(topk["gate_entropy"].median())
        row["gate_entropy_p95"] = float(topk["gate_entropy"].quantile(0.95))
        row["gate_entropy_p99"] = float(topk["gate_entropy"].quantile(0.99))
        for view in ["rnafm", "run", "pam"]:
            count = int((topk["gate_argmax"] == view).sum())
            row[f"gate_argmax_{view}_count"] = count
            row[f"gate_argmax_{view}_fraction"] = float(count / max(n, 1))
        row["NGG_count"] = int((topk["pam_family"] == "NGG").sum())
        row["NGG_fraction"] = float(row["NGG_count"] / max(n, 1))
        row["non_NGG_count"] = int((topk["pam_family"] == "non-NGG").sum())
        row["non_NGG_fraction"] = float(row["non_NGG_count"] / max(n, 1))
        rows.append(row)
    return pd.DataFrame(rows)


def probability_bins_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Compute gate statistics per probability bin."""
    bins = [0, 0.001, 0.01, 0.05, 0.1, 0.5, 0.9, 1.0]
    labels = ["[0, 0.001)", "[0.001, 0.01)", "[0.01, 0.05)", "[0.05, 0.1)",
              "[0.1, 0.5)", "[0.5, 0.9)", "[0.9, 1.0]"]
    df_bin = df.copy()
    df_bin["prob_bin"] = pd.cut(df["probability"], bins=bins, labels=labels, right=False, include_lowest=True)
    df_bin.loc[df["probability"] >= 1.0, "prob_bin"] = "[0.9, 1.0]"

    rows = []
    for bin_name in labels:
        grp = df_bin[df_bin["prob_bin"] == bin_name]
        n = len(grp)
        if n == 0:
            continue
        obs_pos = int((grp["label"] == 1).sum())
        row: dict[str, Any] = {
            "bin": bin_name,
            "n": n,
            "observed_positive": obs_pos,
            "observed_positive_rate": float(obs_pos / max(n, 1)),
        }
        for g in GATE_COLS:
            row[f"{g}_mean"] = float(grp[g].mean())
            row[f"{g}_median"] = float(grp[g].median())
            row[f"{g}_p95"] = float(grp[g].quantile(0.95))
            row[f"{g}_p99"] = float(grp[g].quantile(0.99))
        row["gate_entropy_mean"] = float(grp["gate_entropy"].mean())
        row["gate_entropy_median"] = float(grp["gate_entropy"].median())
        row["gate_entropy_p95"] = float(grp["gate_entropy"].quantile(0.95))
        row["gate_entropy_p99"] = float(grp["gate_entropy"].quantile(0.99))
        for view in ["rnafm", "run", "pam"]:
            count = int((grp["gate_argmax"] == view).sum())
            row[f"gate_argmax_{view}_count"] = count
            row[f"gate_argmax_{view}_fraction"] = float(count / max(n, 1))
        rows.append(row)
    return pd.DataFrame(rows)


def extreme_rows(df: pd.DataFrame, top_n: int = 100) -> pd.DataFrame:
    """Export rows with extreme gate/probability values."""
    slices: list[pd.DataFrame] = []
    for reason, col, ascending in [
        ("top_gate_pam", "gate_pam", False),
        ("top_gate_rnafm", "gate_rnafm", False),
        ("top_gate_entropy", "gate_entropy", False),
        ("top_probability", "probability", False),
    ]:
        sub = df.nlargest(top_n, col).copy()
        sub["selection_reason"] = reason
        slices.append(sub)

    combined = pd.concat(slices, ignore_index=True)
    out_cols = [
        "sample_index", "sgRNA_type", "off_seq", "PAM_original", "pam_family",
        "label", "probability", "gate_rnafm", "gate_run", "gate_pam",
        "gate_sum", "gate_entropy", "gate_max", "gate_argmax", "selection_reason",
    ]
    return combined[out_cols].reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="BL6-1 gate audit analysis")
    parser.add_argument("--input", default="results/bl6_1_pam_gated_fusion/gate_predictions.csv")
    parser.add_argument("--output-dir", default="results/bl6_1_pam_gated_fusion/gate_audit")
    parser.add_argument("--topk", default="100,500,1000,2000,5000,10000")
    parser.add_argument("--top-n-extreme", type=int, default=100)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ks = [int(x.strip()) for x in args.topk.split(",")]

    print(f"[gate_audit] loading {args.input} ...")
    df = pd.read_csv(args.input)
    print(f"[gate_audit] rows={len(df)}")

    # ---- Step 1: Validation ----
    print("[gate_audit] validating input ...")
    validation = validate_input(df)
    print("[gate_audit] validation passed")
    (output_dir / "gate_audit_input_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- Step 2: Overall stats ----
    print("[gate_audit] computing overall gate distribution ...")
    pam_family_counts = df["pam_family"].value_counts().to_dict()
    gate_argmax_counts = df["gate_argmax"].value_counts().to_dict()
    gate_argmax_fraction = {k: v / len(df) for k, v in gate_argmax_counts.items()}

    overall_stats: dict[str, Any] = {
        "n_samples": int(len(df)),
        "observed_positive": int((df["label"] == 1).sum()),
        "unobserved_candidate": int((df["label"] == 0).sum()),
        "pam_family_counts": {str(k): int(v) for k, v in pam_family_counts.items()},
        "gate_argmax_counts": {str(k): int(v) for k, v in gate_argmax_counts.items()},
        "gate_argmax_fraction": {str(k): float(v) for k, v in gate_argmax_fraction.items()},
    }
    for g in GATE_COLS:
        overall_stats[g] = quantile_stats(df[g])
    overall_stats["gate_entropy"] = quantile_stats(df["gate_entropy"])
    overall_stats["gate_max"] = quantile_stats(df["gate_max"])
    overall_stats["gate_sum_max_error"] = float((df["gate_sum"] - 1.0).abs().max())
    overall_stats.update(gate_concentration(df))

    (output_dir / "gate_audit_overall.json").write_text(
        json.dumps(overall_stats, indent=2, ensure_ascii=False), encoding="utf-8")

    # overall summary CSV
    overall_summary = gate_stats_df(df, GATE_COLS + ["gate_entropy", "gate_max"])
    overall_summary["gate_argmax_rnafm"] = gate_argmax_counts.get("rnafm", 0)
    overall_summary["gate_argmax_run"] = gate_argmax_counts.get("run", 0)
    overall_summary["gate_argmax_pam"] = gate_argmax_counts.get("pam", 0)
    overall_summary.to_csv(output_dir / "gate_audit_overall_summary.csv", index=False)

    # ---- Step 3: By label ----
    print("[gate_audit] label stratification ...")
    df["label_name"] = df["label"].map({1: "observed_positive", 0: "unobserved_candidate"})
    by_label = stratified_stats(df, "label_name")
    by_label.to_csv(output_dir / "gate_audit_by_label.csv", index=False)
    df.drop(columns=["label_name"], inplace=True)

    # ---- Step 4: By PAM family ----
    print("[gate_audit] PAM family stratification ...")
    by_pam_family = stratified_stats(df, "pam_family")
    by_pam_family.to_csv(output_dir / "gate_audit_by_pam_family.csv", index=False)

    # ---- Step 5: By PAM motif ----
    print("[gate_audit] PAM motif stratification ...")
    by_motif = stratified_stats(df, "PAM_original", extra_cols=["pam_family"])
    # Reorder: n descending, then PAM_original ascending
    by_motif = by_motif.sort_values(["n", "PAM_original"], ascending=[False, True]).reset_index(drop=True)
    by_motif.to_csv(output_dir / "gate_audit_by_pam_motif.csv", index=False)

    # ---- Step 6: By sgRNA_type ----
    print("[gate_audit] per-sgRNA stratification ...")
    by_sgrna = stratified_stats(df, "sgRNA_type")
    by_sgrna = by_sgrna.sort_values("n", ascending=False).reset_index(drop=True)
    by_sgrna.to_csv(output_dir / "gate_audit_by_sgrna_type.csv", index=False)

    # ---- Step 7: Top-K ----
    print(f"[gate_audit] Top-K analysis (k={ks}) ...")
    by_topk = topk_analysis(df, ks)
    by_topk.to_csv(output_dir / "gate_audit_topk.csv", index=False)

    # ---- Step 8: Probability bins ----
    print("[gate_audit] probability bin analysis ...")
    by_bins = probability_bins_analysis(df)
    by_bins.to_csv(output_dir / "gate_audit_probability_bins.csv", index=False)

    # ---- Step 9: Extreme rows ----
    print(f"[gate_audit] extreme rows (top {args.top_n_extreme}) ...")
    extreme = extreme_rows(df, top_n=args.top_n_extreme)
    extreme.to_csv(output_dir / "gate_audit_extreme_gate_rows.csv", index=False)

    # ---- Summary print ----
    print("\n=== Gate Audit Summary ===")
    print(f"  n_samples: {overall_stats['n_samples']}")
    print(f"  gate_argmax: {overall_stats['gate_argmax_counts']}")
    print(f"  gate_run mean: {overall_stats['gate_run']['mean']:.6f}")
    print(f"  gate_pam mean: {overall_stats['gate_pam']['mean']:.6f}")
    print(f"  gate_rnafm mean: {overall_stats['gate_rnafm']['mean']:.6e}")
    print(f"  fraction gate_run >= 0.99: {overall_stats['fraction_gate_run_ge_0_99']:.6f}")
    print(f"  fraction gate_pam >= 0.99: {overall_stats['fraction_gate_pam_ge_0_99']:.6f}")
    print(f"  fraction gate_rnafm >= 0.99: {overall_stats['fraction_gate_rnafm_ge_0_99']:.6f}")
    print(f"  fraction gate_max >= 0.99: {overall_stats['fraction_gate_max_ge_0_99']:.6f}")
    print(f"  gate_entropy mean: {overall_stats['gate_entropy']['mean']:.6f}")
    print(f"\n[gate_audit] artifacts written to {output_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
