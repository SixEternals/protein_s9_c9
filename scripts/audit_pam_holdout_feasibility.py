#!/usr/bin/env python3
"""
AGENTS.md compliance: [use_rnafm=False, freeze_rnafm=False,
                       split_mode=sgrna_safe, pos_weight=None]
确认本文件遵守 AGENTS.md 约束

PAM-holdout feasibility audit.
判断 CCLMoff formal BL5 split 是否适合做 strict Cross-PAM holdout generalization。
不训练模型，只读取数据并统计。
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="PAM-holdout feasibility audit")
    p.add_argument(
        "--formal_split_json",
        default="formal_split_bl5_seed42.json",
        help="Path to formal split JSON",
    )
    p.add_argument(
        "--cclmoff_csv",
        default="data/cclmoff/09212024_CCLMoff_dataset.csv",
        help="Path to CCLMoff CSV",
    )
    p.add_argument(
        "--output_dir",
        default="results/bl5_generalization/pam_holdout_feasibility",
        help="Output directory",
    )
    p.add_argument("--pam_start", type=int, default=20, help="0-based PAM start index")
    p.add_argument("--pam_end", type=int, default=23, help="0-based PAM end index (exclusive)")
    p.add_argument("--min_test_positive", type=int, default=100)
    p.add_argument("--min_test_unobserved", type=int, default=1000)
    p.add_argument("--min_test_sgrna_types", type=int, default=10)
    p.add_argument("--min_train_positive_after_exclusion", type=int, default=1000)
    p.add_argument("--min_val_positive_after_exclusion", type=int, default=100)
    return p.parse_args()


def load_formal_split(split_path: str) -> dict:
    path = Path(split_path)
    if not path.exists():
        # Search common locations
        candidates = [
            Path(split_path),
            Path("formal_split_bl5_seed42.json"),
            Path("data/cclmoff/formal_split_bl5_seed42.json"),
        ]
        for c in candidates:
            if c.exists():
                path = c
                break
        else:
            raise FileNotFoundError(f"formal split JSON not found: {split_path}")
    return json.loads(path.read_text(encoding="utf-8"))


def assign_split(df: pd.DataFrame, split_json: dict) -> pd.DataFrame:
    """Assign split label to each row based on sgRNA_type membership."""
    train_types = set(split_json["splits"]["train"]["sgRNA_types"])
    val_types = set(split_json["splits"]["val"]["sgRNA_types"])
    test_types = set(split_json["splits"]["test"]["sgRNA_types"])

    def _split_of(t):
        if t in train_types:
            return "train"
        if t in val_types:
            return "val"
        if t in test_types:
            return "test"
        return "unknown"

    df["split"] = df["sgRNA_type"].apply(_split_of)
    return df


def extract_pam(off_seq: pd.Series, start: int, end: int) -> pd.Series:
    """Extract PAM motif using absolute coordinate contract: off_seq[20:23]."""
    return off_seq.str[start:end]


CANONICAL_PAM_RE = re.compile(r'^[ACGT]{3}$')


def compute_feasibility(
    row: pd.Series,
    min_test_pos: int,
    min_test_unobs: int,
    min_test_sgrna: int,
    min_train_pos: int,
    min_val_pos: int,
    overall_test_positive_ratio: float,
) -> tuple[str, list[str], str]:
    flags = []

    has_both_classes = row["test_H_observed_positive"] > 0 and row["test_H_unobserved_candidate"] > 0

    # ── Hard infeasible conditions (MUST be checked first) ──
    is_infeasible = False
    if not has_both_classes:
        flags.append("single_class_test")
        is_infeasible = True
    if row["test_H_observed_positive"] < 20:
        flags.append("too_few_observed_positive")
        is_infeasible = True
    if row["test_H_unobserved_candidate"] < 200:
        flags.append("too_few_unobserved_candidate")
        is_infeasible = True
    if row["test_H_sgRNA_type_count"] < 3:
        flags.append("too_few_sgRNA_type")
        is_infeasible = True
    if row["train_remaining_observed_positive"] < min_train_pos:
        flags.append("train_too_small_after_exclusion")
        is_infeasible = True
    if row["val_remaining_observed_positive"] < min_val_pos:
        flags.append("val_too_small_after_exclusion")
        is_infeasible = True

    if is_infeasible:
        # Add motif-only flags but don't change status
        if row["test_H_observed_positive"] > 0 and row["test_H_unobserved_candidate"] == 0:
            flags.append("motif_has_only_observed_positive")
        if row["test_H_unobserved_candidate"] > 0 and row["test_H_observed_positive"] == 0:
            flags.append("motif_has_only_unobserved_candidate")
        return "infeasible", flags, "Not recommended. Insufficient sample size, class balance, or sgRNA coverage for reliable heldout-PAM generalization."

    # ── Feasible check ──
    if (
        row["test_H_observed_positive"] >= min_test_pos
        and row["test_H_unobserved_candidate"] >= min_test_unobs
        and row["test_H_sgRNA_type_count"] >= min_test_sgrna
        and row["train_remaining_observed_positive"] >= min_train_pos
        and row["val_remaining_observed_positive"] >= min_val_pos
    ):
        status = "feasible"
        recommendation = "Recommended for strict PAM-holdout training (BL5-v4-PAM-holdout-H + BL5-v4-NoPAM-holdout-H)."
    # ── Marginal check (only after infeasible hard conditions are excluded) ──
    elif (
        (20 <= row["test_H_observed_positive"] < min_test_pos)
        or (200 <= row["test_H_unobserved_candidate"] < min_test_unobs)
        or (3 <= row["test_H_sgRNA_type_count"] < min_test_sgrna)
    ):
        status = "marginal"
        recommendation = "Exploratory only. Sample size or sgRNA coverage is limited. Use bootstrap CI and report as supplementary."
        # Add specific marginal flags
        if 20 <= row["test_H_observed_positive"] < min_test_pos:
            flags.append("too_few_observed_positive")
        if 200 <= row["test_H_unobserved_candidate"] < min_test_unobs:
            flags.append("too_few_unobserved_candidate")
        if 3 <= row["test_H_sgRNA_type_count"] < min_test_sgrna:
            flags.append("too_few_sgRNA_type")
    else:
        status = "infeasible"
        recommendation = "Not recommended. Insufficient sample size, class balance, or sgRNA coverage for reliable heldout-PAM generalization."

    # ── Motif-only flags ──
    if row["test_H_observed_positive"] > 0 and row["test_H_unobserved_candidate"] == 0:
        flags.append("motif_has_only_observed_positive")
    if row["test_H_unobserved_candidate"] > 0 and row["test_H_observed_positive"] == 0:
        flags.append("motif_has_only_unobserved_candidate")

    # ── Extreme positive ratio shift (relative to overall test positive_ratio) ──
    test_h_ratio = row["test_H_positive_ratio"]
    if (
        pd.notna(test_h_ratio)
        and test_h_ratio > 0
        and overall_test_positive_ratio is not None
        and overall_test_positive_ratio > 0
        and not np.isnan(overall_test_positive_ratio)
    ):
        ratio_to_overall = test_h_ratio / overall_test_positive_ratio
        if ratio_to_overall > 10 or ratio_to_overall < 0.1:
            flags.append("extreme_positive_ratio_shift")
    elif (
        pd.notna(test_h_ratio)
        and (test_h_ratio > 0.5 or test_h_ratio == 0)
        and overall_test_positive_ratio is not None
        and overall_test_positive_ratio > 0
    ):
        # Edge case: extreme absolute ratio even if overall is very small
        if test_h_ratio > 0.1 and overall_test_positive_ratio < 0.01:
            flags.append("extreme_positive_ratio_shift")

    return status, flags, recommendation


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[audit] Loading formal split...")
    split_json = load_formal_split(args.formal_split_json)

    print("[audit] Loading CCLMoff CSV...")
    df = pd.read_csv(args.cclmoff_csv, usecols=["sgRNA_type", "off_seq", "label"])
    df["label"] = df["label"].astype(int)

    print("[audit] Assigning split and extracting PAM...")
    df = assign_split(df, split_json)
    df["PAM_original"] = extract_pam(df["off_seq"], args.pam_start, args.pam_end)

    # Filter to formal split rows only
    df_split = df[df["split"].isin(["train", "val", "test"])].copy()

    # ── PAM QC: length distribution and noncanonical motifs ──
    print("[audit] Running PAM QC...")
    pam_lengths = df_split["PAM_original"].str.len()
    pam_length_dist = pam_lengths.value_counts().sort_index()
    noncanonical_mask = ~df_split["PAM_original"].str.match(CANONICAL_PAM_RE, na=False)
    noncanonical_pams = sorted(df_split.loc[noncanonical_mask, "PAM_original"].unique())
    noncanonical_counts = df_split.loc[noncanonical_mask, "PAM_original"].value_counts()

    # ── 1. PAM motif by split counts ──
    print("[audit] Computing PAM motif by split counts...")
    pam_split_stats = []
    for split_name in ["train", "val", "test"]:
        sub = df_split[df_split["split"] == split_name]
        for pam, g in sub.groupby("PAM_original"):
            pam_split_stats.append({
                "split": split_name,
                "PAM_original": pam,
                "samples": len(g),
                "observed_positive": int((g["label"] == 1).sum()),
                "unobserved_candidate": int((g["label"] == 0).sum()),
                "positive_ratio": float((g["label"] == 1).mean()),
                "sgRNA_type_count": g["sgRNA_type"].nunique(),
            })

    pam_split_df = pd.DataFrame(pam_split_stats)
    pam_split_df = pam_split_df.sort_values(["split", "samples"], ascending=[True, False])
    pam_split_path = out_dir / "pam_motif_by_split_counts.csv"
    pam_split_df.to_csv(pam_split_path, index=False)
    print(f"[audit] Wrote {pam_split_path}")

    # ── 2. Holdout candidate table ──
    print("[audit] Computing holdout candidate table...")
    all_pams = sorted(df_split["PAM_original"].unique())

    # Pre-compute per-split stats
    train_df = df_split[df_split["split"] == "train"]
    val_df = df_split[df_split["split"] == "val"]
    test_df = df_split[df_split["split"] == "test"]

    # Overall test positive_ratio for extreme_positive_ratio_shift flag
    overall_test_positive_ratio = float((test_df["label"] == 1).mean())

    candidate_rows = []
    for pam_h in all_pams:
        train_ex = train_df[train_df["PAM_original"] != pam_h]
        val_ex = val_df[val_df["PAM_original"] != pam_h]
        test_h = test_df[test_df["PAM_original"] == pam_h]

        row = {
            "holdout_pam": pam_h,
            "train_remaining_samples": len(train_ex),
            "train_remaining_observed_positive": int((train_ex["label"] == 1).sum()),
            "train_remaining_unobserved_candidate": int((train_ex["label"] == 0).sum()),
            "train_remaining_sgRNA_type_count": train_ex["sgRNA_type"].nunique(),
            "val_remaining_samples": len(val_ex),
            "val_remaining_observed_positive": int((val_ex["label"] == 1).sum()),
            "val_remaining_unobserved_candidate": int((val_ex["label"] == 0).sum()),
            "val_remaining_sgRNA_type_count": val_ex["sgRNA_type"].nunique(),
            "test_H_samples": len(test_h),
            "test_H_observed_positive": int((test_h["label"] == 1).sum()),
            "test_H_unobserved_candidate": int((test_h["label"] == 0).sum()),
            "test_H_positive_ratio": float((test_h["label"] == 1).mean()) if len(test_h) > 0 else np.nan,
            "test_H_sgRNA_type_count": test_h["sgRNA_type"].nunique(),
            "test_H_ngg_status": "NGG" if pam_h in {"AGG", "TGG", "GGG", "CGG"} else "non-NGG",
        }

        status, flags, recommendation = compute_feasibility(
            pd.Series(row),
            args.min_test_positive,
            args.min_test_unobserved,
            args.min_test_sgrna_types,
            args.min_train_positive_after_exclusion,
            args.min_val_positive_after_exclusion,
            overall_test_positive_ratio,
        )
        row["feasibility_status"] = status
        row["risk_flags"] = ";".join(flags) if flags else "none"
        row["recommendation"] = recommendation
        candidate_rows.append(row)

    candidate_df = pd.DataFrame(candidate_rows)
    # Sort: feasible first, then marginal, then infeasible; within each group, descending test samples
    status_order = {"feasible": 0, "marginal": 1, "infeasible": 2}
    candidate_df["_status_sort"] = candidate_df["feasibility_status"].map(status_order)
    candidate_df = candidate_df.sort_values(["_status_sort", "test_H_samples"], ascending=[True, False])
    candidate_df = candidate_df.drop(columns=["_status_sort"])
    candidate_path = out_dir / "pam_holdout_candidate_table.csv"
    candidate_df.to_csv(candidate_path, index=False)
    print(f"[audit] Wrote {candidate_path}")

    # ── 3. Recommended holdout motifs JSON ──
    feasible = candidate_df[candidate_df["feasibility_status"] == "feasible"]["holdout_pam"].tolist()
    marginal = candidate_df[candidate_df["feasibility_status"] == "marginal"]["holdout_pam"].tolist()
    infeasible = candidate_df[candidate_df["feasibility_status"] == "infeasible"]["holdout_pam"].tolist()

    recommendation = {
        "coordinate_contract": f"PAM_original = off_seq[{args.pam_start}:{args.pam_end}] (positions 21-23)",
        "formal_split_json": str(args.formal_split_json),
        "overall_test_positive_ratio": overall_test_positive_ratio,
        "overall_test_samples": len(test_df),
        "overall_test_observed_positive": int((test_df["label"] == 1).sum()),
        "overall_test_unobserved_candidate": int((test_df["label"] == 0).sum()),
        "total_unique_pam_motifs": len(all_pams),
        "canonical_3nt_motif_count": sum(1 for p in all_pams if CANONICAL_PAM_RE.match(p)),
        "noncanonical_motif_count": len(noncanonical_pams),
        "noncanonical_motifs": noncanonical_pams,
        "feasible_count": len(feasible),
        "marginal_count": len(marginal),
        "infeasible_count": len(infeasible),
        "recommended_for_training": feasible,
        "marginal_candidates": marginal,
        "not_recommended": infeasible,
        "decision": (
            "run_strict_holdout_if_at_least_one_feasible_candidate_exists"
            if feasible
            else "no_feasible_candidate_strict_holdout_not_recommended"
        ),
    }

    json_path = out_dir / "recommended_holdout_motifs.json"
    json_path.write_text(json.dumps(recommendation, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[audit] Wrote {json_path}")

    # ── 4. Markdown report ──
    print("[audit] Writing Markdown report...")

    # Overall stats
    overall_train = df_split[df_split["split"] == "train"]
    overall_val = df_split[df_split["split"] == "val"]
    overall_test = df_split[df_split["split"] == "test"]

    # Identify key stats for report
    agg_test = candidate_df[candidate_df["holdout_pam"] == "AGG"].iloc[0] if "AGG" in all_pams else None
    tgg_test = candidate_df[candidate_df["holdout_pam"] == "TGG"].iloc[0] if "TGG" in all_pams else None
    gag_test = candidate_df[candidate_df["holdout_pam"] == "GAG"].iloc[0] if "GAG" in all_pams else None

    md_lines = [
        "# PAM Holdout Feasibility Audit",
        "",
        "> **Purpose**: Determine whether the CCLMoff formal BL5 split supports strict Cross-PAM holdout generalization experiments. This is a feasibility audit, not a training result.",
        ">",
        "> **Coordinate Contract**: `PAM_original = off_seq[20:23]` (canonical positions 21-23). Use of `off_seq[-3:]` is prohibited.",
        "",
        "---",
        "",
        "## 1. Formal Split Summary",
        "",
        "| Split | Samples | observed_positive | unobserved_candidate | positive_ratio | sgRNA_type_count |",
        "|:---|:---:|:---:|:---:|:---:|:---:|",
        f"| train | {len(overall_train)} | {(overall_train['label']==1).sum()} | {(overall_train['label']==0).sum()} | {(overall_train['label']==1).mean():.6f} | {overall_train['sgRNA_type'].nunique()} |",
        f"| val | {len(overall_val)} | {(overall_val['label']==1).sum()} | {(overall_val['label']==0).sum()} | {(overall_val['label']==1).mean():.6f} | {overall_val['sgRNA_type'].nunique()} |",
        f"| test | {len(overall_test)} | {(overall_test['label']==1).sum()} | {(overall_test['label']==0).sum()} | {overall_test_positive_ratio:.6f} | {overall_test['sgRNA_type'].nunique()} |",
        "",
        "---",
        "",
        "## 2. PAM Motif Distribution by Split",
        "",
        "Top 15 PAM motifs per split (by sample count):",
        "",
    ]

    for split_name in ["train", "val", "test"]:
        md_lines.append(f"### {split_name}")
        md_lines.append("")
        md_lines.append("| PAM | Samples | observed_positive | unobserved_candidate | positive_ratio | sgRNA_type_count |")
        md_lines.append("|:---|:---:|:---:|:---:|:---:|:---:|")
        sub = pam_split_df[pam_split_df["split"] == split_name].head(15)
        for _, r in sub.iterrows():
            md_lines.append(
                f"| {r['PAM_original']} | {r['samples']} | {r['observed_positive']} | {r['unobserved_candidate']} | {r['positive_ratio']:.6f} | {r['sgRNA_type_count']} |"
            )
        md_lines.append("")

    md_lines.extend([
        "---",
        "",
        "## 3. Data Quality: PAM QC",
        "",
        "### 3.1 PAM Length Distribution",
        "",
        "| Length | Count |",
        "|:---:|:---:|",
    ])
    for length, count in pam_length_dist.items():
        md_lines.append(f"| {length} | {count} |")
    md_lines.append("")

    if noncanonical_pams:
        md_lines.extend([
            "### 3.2 Noncanonical PAM Motifs",
            "",
            f"Found **{len(noncanonical_pams)}** motif(s) not matching `^[ACGT]{{3}}$` (non-3nt, lowercase, or containing gaps):",
            "",
            "| Motif | Total Count |",
            "|:---|:---:|",
        ])
        for motif in noncanonical_pams:
            md_lines.append(f"| `{motif}` | {int(noncanonical_counts.get(motif, 0))} |")
        md_lines.append("")
        md_lines.append(
            "These motifs come from sequences where `off_seq[20:23]` returns malformed strings "
            "(e.g., short sequences or sequences with gaps at those positions). "
            "They are excluded from feasible recommendation and their feasibility status is kept as `infeasible` "
            "if they fail sample-size thresholds."
        )
        md_lines.append("")
    else:
        md_lines.extend([
            "### 3.2 Noncanonical PAM Motifs",
            "",
            "No noncanonical PAM motifs detected. All PAM motifs match `^[ACGT]{3}$`.",
            "",
        ])

    md_lines.extend([
        "---",
        "",
        "## 4. Holdout Candidate Summary",
        "",
        f"- **Total unique PAM motifs**: {len(all_pams)} (canonical 3nt: {sum(1 for p in all_pams if CANONICAL_PAM_RE.match(p))}, noncanonical: {len(noncanonical_pams)})",
        f"- **Feasible**: {len(feasible)}",
        f"- **Marginal**: {len(marginal)}",
        f"- **Infeasible**: {len(infeasible)}",
        f"- **Overall test positive_ratio**: {overall_test_positive_ratio:.6f}",
        "",
    ])

    if feasible:
        md_lines.append(f"- **Recommended for training**: {', '.join(feasible)}")
    else:
        md_lines.append("- **No feasible candidate found**. Strict PAM-holdout training is **not recommended** on this formal split.")

    if marginal:
        md_lines.append(f"- **Marginal candidates** (exploratory only): {', '.join(marginal)}")

    md_lines.extend([
        "",
        "---",
        "",
        "## 5. Feasible / Marginal Candidates",
        "",
    ])

    if not feasible and not marginal:
        md_lines.append("No feasible or marginal candidates. See Section 6 for infeasible details.")
        md_lines.append("")
    else:
        md_lines.append("| holdout_pam | test_H_samples | test_H_pos | test_H_neg | test_H_sgRNA | train_rem_pos | val_rem_pos | status | risk_flags |")
        md_lines.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|")
        for _, r in candidate_df.iterrows():
            if r["feasibility_status"] in ("feasible", "marginal"):
                md_lines.append(
                    f"| {r['holdout_pam']} | {r['test_H_samples']} | {r['test_H_observed_positive']} | {r['test_H_unobserved_candidate']} | {r['test_H_sgRNA_type_count']} | {r['train_remaining_observed_positive']} | {r['val_remaining_observed_positive']} | {r['feasibility_status']} | {r['risk_flags']} |"
                )
        md_lines.append("")

    # ── Key observations ──
    md_lines.extend([
        "### 5.1 Key Observations",
        "",
    ])
    if tgg_test is not None:
        md_lines.append(
            f"- **TGG** has the largest test_H sample count (**{int(tgg_test['test_H_samples']):,}**), "
            f"but AGG has the most observed_positive ({int(agg_test['test_H_observed_positive'])}) — "
            f"better statistical power for AUPRC evaluation."
        )
    if agg_test is not None:
        md_lines.append(
            f"- **AGG** is the strongest feasible candidate overall: "
            f"largest test observed_positive ({int(agg_test['test_H_observed_positive'])}), "
            f"broad sgRNA coverage ({int(agg_test['test_H_sgRNA_type_count'])} sgRNA types), "
            f"and sufficient unobserved_candidate ({int(agg_test['test_H_unobserved_candidate']):,})."
        )
    if gag_test is not None and gag_test["feasibility_status"] == "feasible":
        md_lines.append(
            f"- **GAG** is the only feasible non-NGG PAM motif "
            f"(test observed_positive={int(gag_test['test_H_observed_positive'])}, "
            f"test unobserved_candidate={int(gag_test['test_H_unobserved_candidate']):,}, "
            f"sgRNA types={int(gag_test['test_H_sgRNA_type_count'])}). "
            f"It is a valuable **non-NGG exploratory/extension candidate**, but its positive_ratio "
            f"({gag_test['test_H_positive_ratio']:.4f}) is higher than the overall test positive_ratio "
            f"({overall_test_positive_ratio:.4f}), so results must be interpreted cautiously — "
            f"higher positive density in the held-out non-NGG subset may inflate AUPRC relative to NGG holdouts."
        )
    md_lines.append("")

    md_lines.extend([
        "---",
        "",
        "## 6. Infeasible Candidates (top 20 by test samples)",
        "",
        "| holdout_pam | test_H_samples | test_H_pos | test_H_neg | test_H_sgRNA | status | risk_flags |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---|",
    ])
    infeas_sub = candidate_df[candidate_df["feasibility_status"] == "infeasible"].sort_values("test_H_samples", ascending=False).head(20)
    for _, r in infeas_sub.iterrows():
        md_lines.append(
            f"| {r['holdout_pam']} | {r['test_H_samples']} | {r['test_H_observed_positive']} | {r['test_H_unobserved_candidate']} | {r['test_H_sgRNA_type_count']} | {r['feasibility_status']} | {r['risk_flags']} |"
        )
    md_lines.append("")

    md_lines.extend([
        "---",
        "",
        "## 7. Recommended Next Step",
        "",
    ])

    if feasible:
        md_lines.append(
            f"**Train strict PAM-holdout models** for feasible candidate(s): {', '.join(feasible)}. "
            "At minimum, run paired experiments: BL5-v4-PAM-holdout-H and BL5-v4-NoPAM-holdout-H."
        )
        md_lines.append("")
        md_lines.append(
            "> ⚠️ **Important**: This is a **same-dataset strict PAM-motif holdout**, not cross-dataset, cross-cell-line, or cross-species generalization. "
            "Results from this experiment demonstrate whether the PAM Encoder generalizes to held-out PAM motifs within the CCLMoff dataset, "
            "but do not guarantee generalization to entirely different experimental conditions."
        )
    elif marginal:
        md_lines.append(
            "**Exploratory holdout only.** Only marginal candidates exist. "
            "If run, must be paired with bootstrap CI and reported as supplementary/sanity check, not as strong main conclusion."
        )
        md_lines.append("")
        md_lines.append(
            "Cross-PAM strict holdout on this formal split has **limited statistical power** and should be interpreted with appropriate caveats."
        )
    else:
        md_lines.append(
            "**Do not train strict PAM-holdout models.** No PAM motif has sufficient class balance, sample size, and sgRNA coverage to support a reliable heldout-PAM generalization experiment."
        )
        md_lines.append("")
        md_lines.append(
            "Cross-PAM strict holdout is reported as **infeasible** on the current CCLMoff formal split and left as future work pending a suitable dataset."
        )

    md_lines.extend([
        "",
        "---",
        "",
        "## 8. Limitations",
        "",
        "- **NGG/non-NGG stratified evaluation is NOT the same as strict PAM-holdout.** Stratified evaluation trains on all PAM motifs and tests on a subset; strict holdout excludes a PAM motif from train/val entirely.",
        "- If heldout PAM candidates are too few in test, AUPRC / AUROC variance will be large and interpretation unreliable.",
        "- If heldout PAM covers too few sgRNA types, the result may reflect sgRNA-specific signal rather than true PAM generalization.",
        "- All PAM motifs in this audit use the coordinate contract `off_seq[20:23]` (positions 21-23). Using `off_seq[-3:]` would produce different counts for sequences with gaps or non-standard lengths.",
        "- **This is a feasibility audit, not a Cross-PAM generalization training result.** It only determines whether the data supports such an experiment.",
        "",
    ])

    md_path = out_dir / "pam_holdout_feasibility_report.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[audit] Wrote {md_path}")

    print("\n[audit] Done.")
    print(f"  Feasible:   {len(feasible)}  → {feasible}")
    print(f"  Marginal:   {len(marginal)}  → {marginal}")
    print(f"  Infeasible: {len(infeasible)}")
    print(f"  Noncanonical PAMs: {noncanonical_pams}")
    if feasible:
        print(f"  Decision: run_strict_holdout for {feasible}")
    else:
        print("  No feasible candidate. Strict PAM-holdout NOT recommended.")


if __name__ == "__main__":
    main()
