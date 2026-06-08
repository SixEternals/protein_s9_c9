#!/usr/bin/env python3
"""
AGENTS.md compliance: [use_rnafm=irrelevant, freeze_rnafm=irrelevant,
                       split_mode=sgrna_safe, pos_weight=irrelevant]
确认本文件遵守 AGENTS.md 约束

PAM Strict Holdout Split Constructor
====================================
从 formal_split_bl5_seed42.json（sgRNA-safe）出发，构造 PAM motif strict holdout split。

Rule:
  train_H  = formal_train  AND  PAM_original != holdout_pam
  val_H    = formal_val    AND  PAM_original != holdout_pam
  test_H   = formal_test   AND  PAM_original == holdout_pam
  test_seenPAM = formal_test AND  PAM_original != holdout_pam

PAM coordinate: off_seq[20:23] (positions 21-23, 0-indexed)
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Build PAM strict holdout split")
    parser.add_argument("--holdout_pam", type=str, required=True,
                        help="PAM motif to hold out, e.g. AGG")
    parser.add_argument("--cclmoff_csv", type=str,
                        default="data/cclmoff/09212024_CCLMoff_dataset.csv")
    parser.add_argument("--formal_split_json", type=str,
                        default="formal_split_bl5_seed42.json")
    parser.add_argument("--out_dir", type=str,
                        default="results/bl5_generalization/pam_strict_holdout")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-size-validation", action="store_true",
                        help="Skip minimum sample count validation (for small exploratory holdouts like GAG)")
    return parser.parse_args()


def extract_pam(off_seq: pd.Series) -> pd.Series:
    """Extract PAM from off_seq[20:23] (positions 21-23)."""
    return off_seq.str[20:23]


def build_split(args):
    holdout_pam = args.holdout_pam.upper().strip()
    print(f"[{datetime.now()}] Building PAM strict holdout split for: {holdout_pam}")

    # ------------------------------------------------------------------
    # 1. Read formal split
    # ------------------------------------------------------------------
    print(f"[{datetime.now()}] Loading formal split: {args.formal_split_json}")
    with open(args.formal_split_json, "r") as f:
        formal = json.load(f)

    # Support both nested and flat structures
    if "splits" in formal:
        formal_train_types = set(formal["splits"]["train"]["sgRNA_types"])
        formal_val_types   = set(formal["splits"]["val"]["sgRNA_types"])
        formal_test_types  = set(formal["splits"]["test"]["sgRNA_types"])
    else:
        formal_train_types = set(formal["train"]["sgRNA_types"])
        formal_val_types   = set(formal["val"]["sgRNA_types"])
        formal_test_types  = set(formal["test"]["sgRNA_types"])

    print(f"  formal_train sgRNA types: {len(formal_train_types)}")
    print(f"  formal_val   sgRNA types: {len(formal_val_types)}")
    print(f"  formal_test  sgRNA types: {len(formal_test_types)}")

    # ------------------------------------------------------------------
    # 2. Read CCLMoff CSV (minimal columns)
    # ------------------------------------------------------------------
    print(f"[{datetime.now()}] Loading CCLMoff CSV: {args.cclmoff_csv}")
    usecols = ["sgRNA_seq", "off_seq", "sgRNA_type", "label"]
    df = pd.read_csv(args.cclmoff_csv, usecols=usecols, dtype={"sgRNA_type": str})
    n_total = len(df)
    print(f"  Total rows: {n_total:,}")

    # ------------------------------------------------------------------
    # 3. Map each row to formal split by sgRNA_type
    # ------------------------------------------------------------------
    print(f"[{datetime.now()}] Mapping rows to formal split...")
    # Build a fast mapper
    type_to_split = {}
    for t in formal_train_types:
        type_to_split[t] = "train"
    for t in formal_val_types:
        type_to_split[t] = "val"
    for t in formal_test_types:
        type_to_split[t] = "test"

    df["formal_split"] = df["sgRNA_type"].map(type_to_split)
    unmatched = df["formal_split"].isna().sum()
    if unmatched > 0:
        print(f"  WARNING: {unmatched:,} rows have sgRNA_type not in formal split")
    else:
        print(f"  All rows matched to formal split")

    # ------------------------------------------------------------------
    # 4. Extract PAM
    # ------------------------------------------------------------------
    print(f"[{datetime.now()}] Extracting PAM (off_seq[20:23])...")
    df["pam"] = extract_pam(df["off_seq"])

    # Validate PAM length
    pam_len = df["pam"].str.len()
    bad_pam = (pam_len != 3).sum()
    if bad_pam > 0:
        print(f"  WARNING: {bad_pam:,} rows have PAM length != 3")
    else:
        print(f"  All PAMs are length 3")

    # ------------------------------------------------------------------
    # 5. Construct holdout splits
    # ------------------------------------------------------------------
    print(f"[{datetime.now()}] Constructing holdout splits...")
    is_train = df["formal_split"] == "train"
    is_val   = df["formal_split"] == "val"
    is_test  = df["formal_split"] == "test"
    pam_is_holdout = df["pam"] == holdout_pam

    train_H_mask = is_train & (~pam_is_holdout)
    val_H_mask   = is_val   & (~pam_is_holdout)
    test_H_mask  = is_test  & pam_is_holdout
    test_seenPAM_mask = is_test & (~pam_is_holdout)

    masks = {
        "train_H": train_H_mask,
        "val_H": val_H_mask,
        "test_H": test_H_mask,
        "test_seenPAM": test_seenPAM_mask,
    }

    # ------------------------------------------------------------------
    # 6. Validate strictness conditions
    # ------------------------------------------------------------------
    print(f"[{datetime.now()}] Validating strictness conditions...")
    errors = []

    # 6a. test_H sgRNA types must NOT appear in train_H or val_H
    test_H_types = set(df.loc[test_H_mask, "sgRNA_type"].unique())
    train_H_types = set(df.loc[train_H_mask, "sgRNA_type"].unique())
    val_H_types = set(df.loc[val_H_mask, "sgRNA_type"].unique())
    overlap_test_train = test_H_types & train_H_types
    overlap_test_val = test_H_types & val_H_types
    if overlap_test_train:
        errors.append(f"FAIL: {len(overlap_test_train)} test_H sgRNA types overlap with train_H")
    else:
        print("  [PASS] test_H sgRNA types: 0% overlap with train_H")
    if overlap_test_val:
        errors.append(f"FAIL: {len(overlap_test_val)} test_H sgRNA types overlap with val_H")
    else:
        print("  [PASS] test_H sgRNA types: 0% overlap with val_H")

    # 6b. test_seenPAM PAMs should all be seen in train_H
    train_H_pams = set(df.loc[train_H_mask, "pam"].unique())
    test_seenPAM_pams = set(df.loc[test_seenPAM_mask, "pam"].unique())
    unseen_in_train = test_seenPAM_pams - train_H_pams
    if unseen_in_train:
        print(f"  [INFO] test_seenPAM PAMs not in train_H: {unseen_in_train} (count={len(unseen_in_train)})")
    else:
        print("  [PASS] test_seenPAM PAMs: 100% seen in train_H")

    # 6c. train_H/val_H must have 0% holdout PAM
    train_H_holdout_count = (df.loc[train_H_mask, "pam"] == holdout_pam).sum()
    val_H_holdout_count = (df.loc[val_H_mask, "pam"] == holdout_pam).sum()
    if train_H_holdout_count > 0:
        errors.append(f"FAIL: train_H contains {train_H_holdout_count} rows with PAM={holdout_pam}")
    else:
        print(f"  [PASS] train_H: 0% PAM={holdout_pam}")
    if val_H_holdout_count > 0:
        errors.append(f"FAIL: val_H contains {val_H_holdout_count} rows with PAM={holdout_pam}")
    else:
        print(f"  [PASS] val_H: 0% PAM={holdout_pam}")

    # 6d. test_H must have 100% holdout PAM
    test_H_non_holdout = (df.loc[test_H_mask, "pam"] != holdout_pam).sum()
    if test_H_non_holdout > 0:
        errors.append(f"FAIL: test_H contains {test_H_non_holdout} rows with PAM!={holdout_pam}")
    else:
        print(f"  [PASS] test_H: 100% PAM={holdout_pam}")

    # 6e. Minimum sample counts
    counts = {name: mask.sum() for name, mask in masks.items()}
    print(f"  Counts: { {k: int(v) for k, v in counts.items()} }")

    if not args.skip_size_validation:
        if counts["train_H"] < 200_000:
            errors.append(f"FAIL: train_H too small ({counts['train_H']:,} < 200k)")
        if counts["val_H"] < 20_000:
            errors.append(f"FAIL: val_H too small ({counts['val_H']:,} < 20k)")
        if counts["test_H"] < 100_000:
            errors.append(f"FAIL: test_H too small ({counts['test_H']:,} < 100k)")
        test_H_pos = df.loc[test_H_mask, "label"].sum()
        if test_H_pos < 500:
            errors.append(f"FAIL: test_H observed_positive too small ({int(test_H_pos)} < 500)")
        else:
            print(f"  [PASS] test_H observed_positive: {int(test_H_pos)} >= 500")
    else:
        print(f"  [SKIP] size validation disabled (--skip-size-validation)")
        test_H_pos = df.loc[test_H_mask, "label"].sum()

    # 6f. sgRNA type overlap: test_H vs formal_train (guaranteed 0 by sgRNA-safe design)
    test_H_formal_train_overlap = test_H_types & formal_train_types
    if test_H_formal_train_overlap:
        errors.append(f"FAIL: test_H overlaps with formal_train sgRNA types")
    else:
        print("  [PASS] test_H vs formal_train sgRNA_type overlap: 0 (sgRNA-safe guarantee)")

    # 6g. Exact pair overlap: test_H vs formal_train (sgRNA_seq, off_seq)
    formal_train_mask = df["formal_split"] == "train"
    formal_train_pairs = set(
        zip(df.loc[formal_train_mask, "sgRNA_seq"], df.loc[formal_train_mask, "off_seq"])
    )
    test_H_pairs = set(
        zip(df.loc[test_H_mask, "sgRNA_seq"], df.loc[test_H_mask, "off_seq"])
    )
    exact_pair_overlap = len(formal_train_pairs & test_H_pairs)
    if exact_pair_overlap > 0:
        errors.append(f"FAIL: exact pair overlap test_H vs formal_train = {exact_pair_overlap}")
    else:
        print(f"  [PASS] exact pair overlap test_H vs formal_train: {exact_pair_overlap}")

    if errors:
        print(f"[{datetime.now()}] VALIDATION FAILED with {len(errors)} errors:")
        for e in errors:
            print(f"  ERROR: {e}")
        sys.exit(1)
    else:
        print(f"[{datetime.now()}] ALL VALIDATIONS PASSED")

    # ------------------------------------------------------------------
    # 7. Save outputs
    # ------------------------------------------------------------------
    out_dir = Path(args.out_dir) / holdout_pam
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{datetime.now()}] Saving outputs to: {out_dir}")

    # 7a. Boolean masks as npz (most compact)
    npz_path = out_dir / "split_indices.npz"
    np.savez_compressed(
        npz_path,
        train_H=train_H_mask.to_numpy(),
        val_H=val_H_mask.to_numpy(),
        test_H=test_H_mask.to_numpy(),
        test_seenPAM=test_seenPAM_mask.to_numpy(),
    )
    print(f"  Saved: {npz_path}")

    # 7b. Manifest JSON
    manifest = {
        "holdout_pam": holdout_pam,
        "pam_coordinate": "off_seq[20:23] (positions 21-23)",
        "base_split": str(args.formal_split_json),
        "split_mode": "sgrna_safe",
        "n_total_cclmoff_rows": int(n_total),
        "construction_rules": {
            "train_H": "formal_train AND PAM_original != holdout_pam",
            "val_H":   "formal_val   AND PAM_original != holdout_pam",
            "test_H":  "formal_test  AND PAM_original == holdout_pam",
            "test_seenPAM": "formal_test AND PAM_original != holdout_pam",
        },
        "validations": {
            "test_H_train_overlap_sgRNA": 0,
            "test_H_val_overlap_sgRNA": 0,
            "train_H_holdout_pam_count": int(train_H_holdout_count),
            "val_H_holdout_pam_count": int(val_H_holdout_count),
            "test_H_non_holdout_pam_count": int(test_H_non_holdout),
            "test_seenPAM_unseen_in_train": list(unseen_in_train) if unseen_in_train else [],
            "exact_pair_overlap_formal_train": int(exact_pair_overlap),
        },
        "counts": {k: int(v) for k, v in counts.items()},
        "test_H_observed_positive": int(test_H_pos),
        "test_H_unobserved_candidate": int(counts["test_H"] - test_H_pos),
        "test_H_sgRNA_types": int(len(test_H_types)),
        "seed": args.seed,
        "created_at": datetime.now().isoformat(),
    }
    manifest_path = out_dir / "split_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Saved: {manifest_path}")

    # 7c. Counts CSV
    counts_data = []
    for split_name, mask in masks.items():
        sub = df.loc[mask]
        counts_data.append({
            "split": split_name,
            "n_samples": int(len(sub)),
            "observed_positive": int(sub["label"].sum()),
            "unobserved_candidate": int(len(sub) - sub["label"].sum()),
            "n_sgRNA_types": int(sub["sgRNA_type"].nunique()),
            "positive_rate": float(sub["label"].mean()),
        })
    counts_df = pd.DataFrame(counts_data)
    counts_csv_path = out_dir / "split_counts.csv"
    counts_df.to_csv(counts_csv_path, index=False)
    print(f"  Saved: {counts_csv_path}")

    # 7d. PAM distribution per split
    pam_dist = {}
    for split_name, mask in masks.items():
        pam_counts = df.loc[mask, "pam"].value_counts().to_dict()
        pam_dist[split_name] = {k: int(v) for k, v in pam_counts.items()}
    pam_dist_path = out_dir / "pam_distribution.json"
    with open(pam_dist_path, "w") as f:
        json.dump(pam_dist, f, indent=2)
    print(f"  Saved: {pam_dist_path}")

    print(f"[{datetime.now()}] Done.")
    return manifest


if __name__ == "__main__":
    args = parse_args()
    build_split(args)
