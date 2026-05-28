#!/usr/bin/env python3
"""Audit exact duplicates, sgRNA_type leakage, and PAM distribution.

Outputs:
    results/leakage_exact_duplicate_audit.json / .md
    results/leakage_sgrna_type_audit.json / .md
    results/pam_distribution_by_split.csv / .md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    csv_path = Path("data/cclmoff/09212024_CCLMoff_dataset.csv")
    split_path = Path("formal_split_bl5_seed42.json")
    if not csv_path.exists():
        print(f"Missing {csv_path}", file=sys.stderr)
        return 1
    if not split_path.exists():
        print(f"Missing {split_path}", file=sys.stderr)
        return 1

    print("[audit] Loading formal split...")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    test_groups = set(split["splits"]["test"]["sgRNA_types"])
    train_groups = set(split["splits"]["train"]["sgRNA_types"])
    val_groups = set(split["splits"]["val"]["sgRNA_types"])

    # 3.2 sgRNA_type leakage check
    leakage = {
        "train_val_overlap": sorted(train_groups & val_groups),
        "train_test_overlap": sorted(train_groups & test_groups),
        "val_test_overlap": sorted(val_groups & test_groups),
    }
    leakage_passed = not any(leakage.values())

    leakage_audit = {
        "leakage_check": leakage,
        "leakage_passed": leakage_passed,
        "train_sgRNA_type_count": len(train_groups),
        "val_sgRNA_type_count": len(val_groups),
        "test_sgRNA_type_count": len(test_groups),
    }
    Path("results/leakage_sgrna_type_audit.json").write_text(
        json.dumps(leakage_audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    md = [
        "# sgRNA_type Leakage Audit",
        "",
        f"- train_sgRNA_type_count: {leakage_audit['train_sgRNA_type_count']}",
        f"- val_sgRNA_type_count: {leakage_audit['val_sgRNA_type_count']}",
        f"- test_sgRNA_type_count: {leakage_audit['test_sgRNA_type_count']}",
        "",
        "## Overlap Check",
    ]
    for k, v in leakage.items():
        status = "✅ PASS (empty)" if not v else f"❌ FAIL ({len(v)} overlap)"
        md.append(f"- **{k}**: {status}")
        if v:
            md.append(f"  - examples: {v[:10]}")
    md.append(f"- **Overall passed**: {'✅ YES' if leakage_passed else '❌ NO'}")
    Path("results/leakage_sgrna_type_audit.md").write_text("\n".join(md), encoding="utf-8")
    print("[audit] sgRNA_type leakage check done.")

    # Load minimal columns
    print("[audit] Loading CCLMoff CSV (required columns only)...")
    usecols = ["sgRNA_seq", "off_seq", "label", "sgRNA_type", "chr", "Location", "Direction"]
    df = pd.read_csv(csv_path, usecols=lambda c: c in usecols, dtype={"Location": str})
    df["sgRNA_type"] = df["sgRNA_type"].astype(str)

    # Assign split by sgRNA_type
    df["split"] = df["sgRNA_type"].apply(
        lambda x: "train" if x in train_groups else ("val" if x in val_groups else ("test" if x in test_groups else "unknown"))
    )
    if (df["split"] == "unknown").any():
        raise RuntimeError("Some rows not assigned to any split")

    # 3.1 Exact duplicate check
    print("[audit] Checking exact duplicates...")
    df["key1"] = df["sgRNA_seq"] + "_" + df["off_seq"]
    df["key2"] = df["sgRNA_seq"] + "_" + df["off_seq"] + "_" + df["label"].astype(str)
    df["key3"] = df["sgRNA_seq"] + "_" + df["off_seq"] + "_" + df["chr"].astype(str) + "_" + df["Location"] + "_" + df["Direction"].astype(str)

    dup_results = {}
    for key_name in ("key1", "key2", "key3"):
        train_keys = set(df[df["split"] == "train"][key_name])
        val_keys = set(df[df["split"] == "val"][key_name])
        test_keys = set(df[df["split"] == "test"][key_name])
        dup_results[key_name] = {
            "train_val": len(train_keys & val_keys),
            "train_test": len(train_keys & test_keys),
            "val_test": len(val_keys & test_keys),
        }

    dup_audit = {
        "exact_duplicate_check": dup_results,
        "train_test_duplicate_passed": all(v["train_test"] == 0 for v in dup_results.values()),
    }
    Path("results/leakage_exact_duplicate_audit.json").write_text(
        json.dumps(dup_audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    md = ["# Exact Duplicate Leakage Audit", ""]
    for key_name, d in dup_results.items():
        md.append(f"## {key_name}")
        for k, v in d.items():
            status = "✅ PASS (0)" if v == 0 else f"⚠️ {v} duplicates"
            md.append(f"- **{k}**: {status}")
        md.append("")
    md.append(f"- **train_test_duplicate_passed**: {'✅ YES' if dup_audit['train_test_duplicate_passed'] else '❌ NO'}")
    Path("results/leakage_exact_duplicate_audit.md").write_text("\n".join(md), encoding="utf-8")
    print("[audit] Exact duplicate check done.")

    # 4.1 PAM distribution
    print("[audit] Analyzing PAM distribution...")
    df["PAM"] = df["off_seq"].astype(str).str[-3:]
    df["is_NGG"] = df["PAM"].str[1:3] == "GG"

    pam_stats = []
    for split_name in ("train", "val", "test"):
        subset = df[df["split"] == split_name]
        total = len(subset)
        for pam, group in subset.groupby("PAM"):
            pos = int((group["label"] == 1).sum())
            neg = int((group["label"] == 0).sum())
            pam_stats.append({
                "split": split_name,
                "PAM": pam,
                "count": len(group),
                "ratio": len(group) / total,
                "positive_count": pos,
                "negative_count": neg,
                "positive_ratio": pos / len(group) if len(group) else 0.0,
            })

    pam_df = pd.DataFrame(pam_stats)
    pam_csv = Path("results/pam_distribution_by_split.csv")
    pam_df.to_csv(pam_csv, index=False)

    # Summarize NGG vs non-NGG
    ngg_summary = []
    for split_name in ("train", "val", "test"):
        subset = df[df["split"] == split_name]
        ngg = subset[subset["is_NGG"]]
        non_ngg = subset[~subset["is_NGG"]]
        ngg_summary.append({
            "split": split_name,
            "total": len(subset),
            "NGG_count": len(ngg),
            "non_NGG_count": len(non_ngg),
            "non_NGG_positive": int((non_ngg["label"] == 1).sum()),
            "non_NGG_negative": int((non_ngg["label"] == 0).sum()),
            "non_NGG_positive_ratio": float((non_ngg["label"] == 1).sum() / len(non_ngg)) if len(non_ngg) else 0.0,
        })

    ngg_df = pd.DataFrame(ngg_summary)
    print("[audit] PAM distribution done.")

    # Write PAM markdown
    md = ["# PAM Distribution by Split", ""]
    for _, row in ngg_df.iterrows():
        md.extend([
            f"## {row['split'].upper()}",
            f"- Total: {row['total']:,}",
            f"- NGG: {row['NGG_count']:,}",
            f"- non-NGG: {row['non_NGG_count']:,}",
            f"- non-NGG positive: {row['non_NGG_positive']:,}",
            f"- non-NGG negative: {row['non_NGG_negative']:,}",
            f"- non-NGG positive_ratio: {row['non_NGG_positive_ratio']:.4f}",
            "",
        ])

    # Check if non-NGG are all positive in test
    test_non_ngg = ngg_df[ngg_df["split"] == "test"].iloc[0]
    if test_non_ngg["non_NGG_positive_ratio"] == 1.0:
        md.append("⚠️ **ALERT**: Test set non-NGG samples are 100% positive. This is a potential shortcut signal.")
    else:
        md.append("✅ Test set non-NGG positive ratio is not 100%.")

    Path("results/pam_distribution_by_split.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[audit] Wrote {pam_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
