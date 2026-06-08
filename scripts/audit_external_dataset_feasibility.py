#!/usr/bin/env python3
"""
AGENTS.md compliance: [use_rnafm=False, freeze_rnafm=False,
                       split_mode=sgrna_safe, pos_weight=None]
确认本文件遵守 AGENTS.md 约束

External Dataset Feasibility Audit v2 for BL5-v4-PAM cross-dataset generalization.
Does NOT train models, does NOT call GPU, does NOT modify existing code/checkpoints.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="External dataset feasibility audit v2")
    p.add_argument("--cclmoff_csv", default="data/cclmoff/09212024_CCLMoff_dataset.csv")
    p.add_argument("--formal_split_json", default="formal_split_bl5_seed42.json")
    p.add_argument("--output_dir", default="results/bl5_generalization/external_dataset_feasibility")
    return p.parse_args()


SGRNA_COLS = ["sgrna_seq", "sgrna", "guide_seq", "guide", "grna", "on_seq", "on", "protospacer"]
OFF_COLS = ["off_seq", "dna", "target_seq", "off_target_seq", "offtarget_seq", "off", "target", "offsite_seq"]
LABEL_COLS = ["label", "y", "class", "is_positive", "observed", "cleavage", "active"]
READ_COLS = ["read", "reads", "read_count", "count", "counts", "cleavage_count"]
DATASET_COLS = ["dataset", "dataset_name", "method", "Method"]


def detect_columns(cols):
    col_lower = {c.lower(): c for c in cols}
    sgrna = next((col_lower[k] for k in SGRNA_COLS if k in col_lower), None)
    off = next((col_lower[k] for k in OFF_COLS if k in col_lower), None)
    label = next((col_lower[k] for k in LABEL_COLS if k in col_lower), None)
    read = next((col_lower[k] for k in READ_COLS if k in col_lower), None)
    dataset = next((col_lower[k] for k in DATASET_COLS if k in col_lower), None)
    return sgrna, off, label, read, dataset


def classify_inventory_file(fpath, root):
    """Classify a file for inventory purposes without reading contents."""
    suffix = fpath.suffix.lower()
    name_lower = fpath.name.lower()
    size = fpath.stat().st_size
    rel = str(fpath.relative_to(root))

    # Self-exclusion
    if "external_dataset_feasibility" in rel:
        return "non_data", False, False, "audit_self_output"

    # Checkpoint / model weights
    if suffix in {".pt", ".pth", ".ckpt", ".safetensors"}:
        return "model_weight_or_checkpoint", False, False, "model_weight"

    # Non-data
    if suffix in {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".html", ".htm", ".log", ".md", ".txt"}:
        if "log" in name_lower or "readme" in name_lower or "requirements" in name_lower or ".gitignore" in name_lower:
            return "non_data", False, False, "non_data_artifact"

    # Training artifacts by name
    if name_lower in {"summary.json", "config_used.json", "split_summary.json"}:
        return "training_artifact", False, False, "training_summary_or_config"
    if "epoch_metrics" in name_lower:
        return "training_artifact", False, False, "training_metric"
    if "test_predictions" in name_lower and suffix in {".csv", ".npz"}:
        return "derived_result_artifact", True, True, "model_prediction_artifact_with_raw_possible"
    if "config" in name_lower and suffix == ".json":
        return "config_or_manifest", False, False, "config_artifact"

    # CCLMoff baseline reference
    if "09212024_CCLMoff_dataset.csv" in name_lower:
        return "baseline_reference_cclmoff", False, False, "cclmoff_source_csv_not_external"
    if "cclmoff_" in name_lower and (name_lower.endswith("bit.npz") or "rnafm_embeddings.npz" in name_lower):
        return "baseline_reference_cclmoff", False, False, "cclmoff_derived_npz_not_external"

    # Large file handling
    if size > 100_000_000:
        if suffix == ".npz":
            return "external_npz_candidate", True, False, "too_large_needs_targeted_audit"
        return "too_large_needs_targeted_audit", True, False, "file_too_large"

    # Fasta
    if suffix in {".fasta", ".fa"}:
        return "sequence_fasta_unpaired", False, False, "fasta_unpaired_sequences"

    # JSON metadata / report
    if suffix == ".json":
        if "audit" in name_lower or "report" in name_lower:
            return "report_artifact", False, False, "audit_report_json"
        if "manifest" in name_lower:
            return "config_or_manifest", False, False, "package_manifest"
        return "metadata_only", False, False, "json_metadata"

    # Tabular candidate
    if suffix in {".csv", ".tsv", ".parquet", ".jsonl"}:
        return "raw_table_candidate", True, True, "tabular_data_candidate"

    # NPZ
    if suffix == ".npz":
        return "external_npz_candidate", True, True, "npz_candidate"

    return "non_data", False, False, "unclassified"


def audit_subset(df, sgrna_col, off_col, label_col, read_col, dataset_col,
                 cclmoff_train_pairs, cclmoff_val_pairs, cclmoff_test_pairs, cclmoff_any_pairs,
                 cclmoff_train_sgrnas, cclmoff_val_sgrnas, cclmoff_test_sgrnas, cclmoff_any_sgrnas,
                 subset_name, source_kind, file_path, file_note):
    """Audit a DataFrame subset and return a dict row."""
    n_rows = len(df)
    n_cols = len(df.columns)
    cols = list(df.columns)

    has_sgrna = sgrna_col is not None and sgrna_col in df.columns
    has_off = off_col is not None and off_col in df.columns
    has_label = label_col is not None and label_col in df.columns

    # Schema
    if has_sgrna and has_off and has_label:
        schema_status = "compatible"
    elif has_sgrna and has_off:
        schema_status = "partial (missing label)"
    else:
        schema_status = "incompatible"

    # Label audit
    pos_count = np.nan
    neg_count = np.nan
    unknown_label_count = np.nan
    positive_ratio = np.nan
    can_auroc = False
    label_semantics = "unknown"

    if has_label and label_col in df.columns:
        lbl = pd.to_numeric(df[label_col], errors="coerce")
        pos_count = int((lbl == 1).sum())
        neg_count = int((lbl == 0).sum())
        unknown_label_count = int(lbl.isna().sum())
        can_auroc = pos_count > 0 and neg_count > 0
        positive_ratio = float(pos_count / n_rows) if n_rows > 0 else 0.0
        if can_auroc:
            label_semantics = "clear_binary"
        elif pos_count > 0 and neg_count == 0:
            label_semantics = "positive_only"
        elif pos_count == 0 and neg_count > 0:
            label_semantics = "candidate_only"
        else:
            label_semantics = "unclear"

    # Sequence audit
    seq_audit = {}
    if has_sgrna and has_off:
        sub = df[[sgrna_col, off_col]].dropna()
        n_checked = len(sub)
        sgrna_lens = sub[sgrna_col].astype(str).str.len()
        off_lens = sub[off_col].astype(str).str.len()
        off_seqs = sub[off_col].astype(str).str.upper()
        can_extract = off_lens >= 23
        pam_seqs = off_seqs[can_extract].str[20:23] if can_extract.any() else pd.Series([], dtype=str)

        seq_audit = {
            "sgRNA_length_distribution": str(sgrna_lens.value_counts().sort_index().to_dict()),
            "off_seq_length_distribution": str(off_lens.value_counts().sort_index().to_dict()),
            "canonical_23nt_pair_count": int(((sgrna_lens == 23) & (off_lens == 23)).sum()),
            "can_extract_pam_offseq_20_23": bool(can_extract.any()),
            "pam_motif_count": pam_seqs.nunique(),
            "ngg_count": int(pam_seqs.isin(["AGG", "TGG", "GGG", "CGG"]).sum()),
            "non_ngg_count": int(len(pam_seqs) - pam_seqs.isin(["AGG", "TGG", "GGG", "CGG"]).sum()),
            "malformed_pam_count": int((off_lens < 23).sum()),
        }
    else:
        seq_audit = {
            "sgRNA_length_distribution": "",
            "off_seq_length_distribution": "",
            "canonical_23nt_pair_count": np.nan,
            "can_extract_pam_offseq_20_23": False,
            "pam_motif_count": np.nan,
            "ngg_count": np.nan,
            "non_ngg_count": np.nan,
            "malformed_pam_count": np.nan,
        }

    # Overlap
    overlap = {}
    unique_sgrna_count = np.nan
    unique_pair_count = np.nan
    if has_sgrna and has_off:
        sub = df[[sgrna_col, off_col]].dropna()
        sgrnas = set(sub[sgrna_col].astype(str).str.upper())
        pairs = set(zip(sub[sgrna_col].astype(str).str.upper(), sub[off_col].astype(str).str.upper()))
        unique_sgrna_count = len(sgrnas)
        unique_pair_count = len(pairs)

        o_train = len(pairs & cclmoff_train_pairs)
        o_val = len(pairs & cclmoff_val_pairs)
        o_test = len(pairs & cclmoff_test_pairs)
        o_any = len(pairs & cclmoff_any_pairs)
        frac = round(o_any / len(pairs), 6) if len(pairs) > 0 else 0.0

        s_train = len(sgrnas & cclmoff_train_sgrnas)
        s_val = len(sgrnas & cclmoff_val_sgrnas)
        s_test = len(sgrnas & cclmoff_test_sgrnas)
        s_any = len(sgrnas & cclmoff_any_sgrnas)

        if frac == 0.0:
            ostatus = "independent_by_exact_pair"
        elif frac < 0.1:
            ostatus = "low_pair_overlap"
        elif frac < 0.5:
            ostatus = "moderate_pair_overlap"
        else:
            ostatus = "heavy_pair_overlap"

        overlap = {
            "pair_overlap_cclmoff_train": o_train,
            "pair_overlap_cclmoff_val": o_val,
            "pair_overlap_cclmoff_test": o_test,
            "pair_overlap_cclmoff_any": o_any,
            "pair_overlap_fraction_any": frac,
            "sgRNA_overlap_cclmoff_train": s_train,
            "sgRNA_overlap_cclmoff_val": s_val,
            "sgRNA_overlap_cclmoff_test": s_test,
            "sgRNA_overlap_cclmoff_any": s_any,
            "overlap_status": ostatus,
        }
    else:
        overlap = {
            "pair_overlap_cclmoff_train": np.nan,
            "pair_overlap_cclmoff_val": np.nan,
            "pair_overlap_cclmoff_test": np.nan,
            "pair_overlap_cclmoff_any": np.nan,
            "pair_overlap_fraction_any": np.nan,
            "sgRNA_overlap_cclmoff_train": np.nan,
            "sgRNA_overlap_cclmoff_val": np.nan,
            "sgRNA_overlap_cclmoff_test": np.nan,
            "sgRNA_overlap_cclmoff_any": np.nan,
            "overlap_status": "no_sequence_columns",
        }

    # Feasibility
    flags = []

    if source_kind == "derived_result_artifact":
        flags.append("result_artifact")
    if not can_auroc:
        flags.append("cannot_compute_auroc_auprc")
    if overlap.get("overlap_status") in ("heavy_pair_overlap", "moderate_pair_overlap"):
        flags.append(overlap.get("overlap_status"))
    if overlap.get("pair_overlap_cclmoff_train", 0) > 0:
        flags.append("train_pair_overlap_nonzero")
    if seq_audit.get("can_extract_pam_offseq_20_23", False) is False:
        flags.append("cannot_extract_pam")
    if seq_audit.get("canonical_23nt_pair_count", 0) < n_rows * 0.9 if n_rows > 0 else False:
        if has_sgrna and has_off:
            flags.append("non_canonical_sequence_lengths")
    if (pos_count if not np.isnan(pos_count) else 0) < 100:
        if has_label:
            flags.append("too_few_observed_positive")
    if (neg_count if not np.isnan(neg_count) else 0) < 1000:
        if has_label:
            flags.append("too_few_unobserved_candidate")
    if (unique_sgrna_count if not np.isnan(unique_sgrna_count) else 0) < 10:
        if has_sgrna:
            flags.append("unique_sgRNA_below_10")

    # Determine feasibility status
    if schema_status == "incompatible":
        fstatus = "infeasible"
        frecommendation = "Incompatible schema — missing required columns."
    elif not has_sgrna or not has_off:
        fstatus = "metadata_only"
        frecommendation = "Missing paired sequence columns."
    elif not can_auroc:
        fstatus = "positive_only_not_auc_feasible" if (pos_count > 0 if not np.isnan(pos_count) else False) else "infeasible"
        frecommendation = "Cannot compute AUROC/AUPRC due to missing binary labels."
    elif overlap.get("overlap_status") == "heavy_pair_overlap":
        fstatus = "overlap_not_independent"
        frecommendation = "Heavy pair overlap with CCLMoff; cannot serve as strict external benchmark."
    elif overlap.get("overlap_status") == "moderate_pair_overlap":
        fstatus = "overlap_not_independent"
        frecommendation = "Moderate pair overlap with CCLMoff; not independent enough."
    elif source_kind == "derived_result_artifact":
        if (pos_count >= 100 if not np.isnan(pos_count) else False) and (neg_count >= 1000 if not np.isnan(neg_count) else False) and (unique_sgrna_count >= 5 if not np.isnan(unique_sgrna_count) else False):
            if overlap.get("pair_overlap_fraction_any", 1.0) < 0.1:
                fstatus = "provenance_required_limited_candidate"
                frecommendation = "Derived from result artifact; requires provenance confirmation of raw source, label semantics, and candidate generation before external eval."
            else:
                fstatus = "overlap_not_independent"
                frecommendation = "Result artifact with significant overlap."
        else:
            fstatus = "result_artifact_not_dataset"
            frecommendation = "Result artifact with insufficient samples for external eval."
    elif (pos_count >= 100 if not np.isnan(pos_count) else False) and (neg_count >= 1000 if not np.isnan(neg_count) else False) and (unique_sgrna_count >= 10 if not np.isnan(unique_sgrna_count) else False) and overlap.get("pair_overlap_fraction_any", 1.0) < 0.01 and overlap.get("pair_overlap_cclmoff_train", 0) == 0:
        fstatus = "ready_for_strict_external_eval"
        frecommendation = "Recommended for full cross-dataset evaluation."
    elif (pos_count >= 20 if not np.isnan(pos_count) else False) and (neg_count >= 200 if not np.isnan(neg_count) else False) and (unique_sgrna_count >= 3 if not np.isnan(unique_sgrna_count) else False) and overlap.get("pair_overlap_fraction_any", 1.0) < 0.1:
        fstatus = "limited_external_eval_candidate"
        frecommendation = "Limited evaluation possible with explicit caveats about sample size or sgRNA coverage."
    elif n_rows < 100 and can_auroc:
        fstatus = "smoke_test_only"
        frecommendation = "Sample size too small for stable AUROC/AUPRC; smoke test only."
    else:
        fstatus = "infeasible"
        frecommendation = "Does not meet criteria for external evaluation."

    candidate_id = f"{file_path}::{subset_name}" if subset_name != "ALL" else file_path

    return {
        "candidate_id": candidate_id,
        "file_path": file_path,
        "subset_name": subset_name,
        "source_kind": source_kind,
        "file_note": file_note,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "columns": ", ".join(cols),
        "sgRNA_col": sgrna_col if has_sgrna else "",
        "off_seq_col": off_col if has_off else "",
        "label_col": label_col if has_label else "",
        "read_col": read_col if read_col is not None else "",
        "dataset_col": dataset_col if dataset_col is not None else "",
        "schema_status": schema_status,
        "label_semantics_status": label_semantics,
        "observed_positive_count": pos_count,
        "unobserved_candidate_count": neg_count,
        "unknown_label_count": unknown_label_count,
        "positive_ratio": positive_ratio,
        "can_compute_auroc_auprc": can_auroc,
        "unique_sgRNA_count": unique_sgrna_count,
        "unique_pair_count": unique_pair_count,
        **{k: v for k, v in seq_audit.items()},
        **{k: v for k, v in overlap.items()},
        "risk_flags": ";".join(flags) if flags else "none",
        "feasibility_status": fstatus,
        "recommended_next_step": frecommendation,
    }


def main():
    args = parse_args()
    root = Path(".")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[audit] Loading CCLMoff for overlap baseline...")
    cclmoff = pd.read_csv(args.cclmoff_csv, usecols=["sgRNA_seq", "off_seq", "sgRNA_type"])
    cclmoff["sgRNA_seq_u"] = cclmoff["sgRNA_seq"].astype(str).str.upper()
    cclmoff["off_seq_u"] = cclmoff["off_seq"].astype(str).str.upper()

    print("[audit] Loading formal split...")
    with open(args.formal_split_json) as f:
        split_data = json.load(f)

    train_sgrnas = set(split_data["splits"]["train"]["sgRNA_types"])
    val_sgrnas = set(split_data["splits"]["val"]["sgRNA_types"])
    test_sgrnas = set(split_data["splits"]["test"]["sgRNA_types"])

    cclmoff_train = cclmoff[cclmoff["sgRNA_type"].isin(train_sgrnas)]
    cclmoff_val = cclmoff[cclmoff["sgRNA_type"].isin(val_sgrnas)]
    cclmoff_test = cclmoff[cclmoff["sgRNA_type"].isin(test_sgrnas)]

    cclmoff_train_pairs = set(zip(cclmoff_train["sgRNA_seq_u"], cclmoff_train["off_seq_u"]))
    cclmoff_val_pairs = set(zip(cclmoff_val["sgRNA_seq_u"], cclmoff_val["off_seq_u"]))
    cclmoff_test_pairs = set(zip(cclmoff_test["sgRNA_seq_u"], cclmoff_test["off_seq_u"]))
    cclmoff_any_pairs = cclmoff_train_pairs | cclmoff_val_pairs | cclmoff_test_pairs

    cclmoff_train_sgrnas = set(cclmoff_train["sgRNA_seq_u"])
    cclmoff_val_sgrnas = set(cclmoff_val["sgRNA_seq_u"])
    cclmoff_test_sgrnas = set(cclmoff_test["sgRNA_seq_u"])
    cclmoff_any_sgrnas = cclmoff_train_sgrnas | cclmoff_val_sgrnas | cclmoff_test_sgrnas

    print(f"[audit] CCLMoff train pairs: {len(cclmoff_train_pairs):,}, val: {len(cclmoff_val_pairs):,}, test: {len(cclmoff_test_pairs):,}")

    # ------------------------------------------------------------------
    # 1. Inventory scan
    # ------------------------------------------------------------------
    print("[audit] Building inventory...")
    scan_dirs = ["data", "reference", "results", "output", "doc", "reborn_doc", "artifacts", "offtarget_fusion_project"]
    all_files = []

    # Scan subdirectories recursively
    for dname in scan_dirs:
        dpath = root / dname
        if not dpath.exists():
            continue
        for fpath in dpath.rglob("*"):
            if not fpath.is_file():
                continue
            # Skip self-exclusion, git, pycache
            rel = str(fpath.relative_to(root))
            if "external_dataset_feasibility" in rel:
                continue
            if ".git/" in rel or "__pycache__/" in rel:
                continue

            inv_class, is_candidate, needs_audit, reason = classify_inventory_file(fpath, root)

            # Try to get row/col info for small tabular files
            n_rows = np.nan
            n_cols = np.nan
            columns = ""
            read_status = "not_attempted"
            suffix = fpath.suffix.lower()
            size = fpath.stat().st_size

            if needs_audit and size < 50_000_000 and suffix in {".csv", ".tsv", ".json", ".jsonl"}:
                try:
                    if suffix == ".csv":
                        df_tmp = pd.read_csv(fpath, nrows=5)
                    elif suffix == ".tsv":
                        df_tmp = pd.read_csv(fpath, sep="\t", nrows=5)
                    elif suffix == ".jsonl":
                        df_tmp = pd.read_json(fpath, lines=True, nrows=5)
                    elif suffix == ".json":
                        with open(fpath) as f:
                            data = json.load(f)
                        if isinstance(data, list) and data:
                            df_tmp = pd.DataFrame(data[:5])
                        else:
                            df_tmp = pd.DataFrame([data])
                    n_cols = len(df_tmp.columns)
                    columns = ", ".join(list(df_tmp.columns))
                    read_status = "peek_success"
                except Exception as e:
                    read_status = f"peek_failed:{e}"

            all_files.append({
                "file_path": rel,
                "file_type": suffix.lstrip(".") if suffix else "",
                "file_size_bytes": size,
                "source_group": dname,
                "inventory_class": inv_class,
                "is_external_dataset_candidate": is_candidate,
                "needs_detailed_audit": needs_audit,
                "reason_if_excluded": reason,
                "n_rows_if_readable": n_rows,
                "n_cols_if_readable": n_cols,
                "columns_if_readable": columns,
                "read_status": read_status,
            })

    # Also scan root-level files
    for fpath in root.iterdir():
        if not fpath.is_file():
            continue
        rel = str(fpath.relative_to(root))
        if "external_dataset_feasibility" in rel:
            continue
        inv_class, is_candidate, needs_audit, reason = classify_inventory_file(fpath, root)
        suffix = fpath.suffix.lower()
        size = fpath.stat().st_size
        all_files.append({
            "file_path": rel,
            "file_type": suffix.lstrip(".") if suffix else "",
            "file_size_bytes": size,
            "source_group": "root",
            "inventory_class": inv_class,
            "is_external_dataset_candidate": is_candidate,
            "needs_detailed_audit": needs_audit,
            "reason_if_excluded": reason,
            "n_rows_if_readable": np.nan,
            "n_cols_if_readable": np.nan,
            "columns_if_readable": "",
            "read_status": "not_attempted",
        })

    inventory_df = pd.DataFrame(all_files)
    inventory_path = out_dir / "external_dataset_inventory.csv"
    inventory_df.to_csv(inventory_path, index=False)
    print(f"[audit] Wrote {inventory_path} ({len(all_files)} files)")

    # ------------------------------------------------------------------
    # 2. Detailed audit for candidates
    # ------------------------------------------------------------------
    print("[audit] Auditing detailed candidates...")
    audit_rows = []

    # Files that need detailed audit
    audit_targets = [f for f in all_files if f["needs_detailed_audit"]]

    # Forced audit targets (explicitly requested regardless of size/classification)
    forced_paths = [
        "output/crispr_dualpred_five_dataset_full_20260507_204525/predictions.csv",
    ]
    existing_paths = {f["file_path"] for f in audit_targets}
    for fp in forced_paths:
        if fp not in existing_paths:
            fpath = root / fp
            if fpath.exists():
                audit_targets.append({
                    "file_path": fp,
                    "file_type": fpath.suffix.lstrip("."),
                    "file_size_bytes": fpath.stat().st_size,
                    "source_group": "output",
                    "inventory_class": "raw_table_candidate",
                    "is_external_dataset_candidate": True,
                    "needs_detailed_audit": True,
                    "reason_if_excluded": "forced_audit_target",
                    "n_rows_if_readable": np.nan,
                    "n_cols_if_readable": np.nan,
                    "columns_if_readable": "",
                    "read_status": "not_attempted",
                })

    for inv in audit_targets:
        fpath = root / inv["file_path"]
        suffix = fpath.suffix.lower()
        size = fpath.stat().st_size
        inv_class = inv["inventory_class"]

        print(f"  -> {inv['file_path']} ({inv_class})")

        if inv_class == "too_large_needs_targeted_audit":
            audit_rows.append({
                "candidate_id": inv["file_path"],
                "file_path": inv["file_path"],
                "subset_name": "ALL",
                "source_kind": inv_class,
                "file_note": "File too large for automatic audit; requires targeted/chunked inspection.",
                "n_rows": np.nan,
                "n_cols": np.nan,
                "columns": "",
                "sgRNA_col": "",
                "off_seq_col": "",
                "label_col": "",
                "read_col": "",
                "dataset_col": "",
                "schema_status": "unknown (too large)",
                "label_semantics_status": "unknown",
                "observed_positive_count": np.nan,
                "unobserved_candidate_count": np.nan,
                "unknown_label_count": np.nan,
                "positive_ratio": np.nan,
                "can_compute_auroc_auprc": False,
                "unique_sgRNA_count": np.nan,
                "unique_pair_count": np.nan,
                "sgRNA_length_distribution": "",
                "off_seq_length_distribution": "",
                "canonical_23nt_pair_count": np.nan,
                "can_extract_pam_offseq_20_23": False,
                "pam_motif_count": np.nan,
                "ngg_count": np.nan,
                "non_ngg_count": np.nan,
                "malformed_pam_count": np.nan,
                "pair_overlap_cclmoff_train": np.nan,
                "pair_overlap_cclmoff_val": np.nan,
                "pair_overlap_cclmoff_test": np.nan,
                "pair_overlap_cclmoff_any": np.nan,
                "pair_overlap_fraction_any": np.nan,
                "sgRNA_overlap_cclmoff_train": np.nan,
                "sgRNA_overlap_cclmoff_val": np.nan,
                "sgRNA_overlap_cclmoff_test": np.nan,
                "sgRNA_overlap_cclmoff_any": np.nan,
                "overlap_status": "unknown (too large)",
                "risk_flags": "too_large_needs_targeted_audit",
                "feasibility_status": "future_work_needed",
                "recommended_next_step": "Requires chunked or sampled audit due to file size.",
            })
            continue

        if inv_class == "external_npz_candidate":
            try:
                npz = np.load(fpath)
                keys = list(npz.keys())
                has_seq = any(k.lower() in {"on_seq", "sgRNA_seq", "sgrna", "off_seq", "dna", "target_seq"} for k in keys)
                has_label = any(k.lower() in {"y", "label"} for k in keys)
                if has_seq and has_label:
                    note = "NPZ with sequence and label keys; candidate but needs full extraction audit."
                    fstatus = "provenance_required_limited_candidate" if size < 100_000_000 else "future_work_needed"
                elif has_label and not has_seq:
                    note = "NPZ with label but missing sequence keys; cannot do RNA-FM/PAM external eval."
                    fstatus = "infeasible"
                else:
                    note = "NPZ without clear sequence/label keys."
                    fstatus = "infeasible"
                audit_rows.append({
                    "candidate_id": inv["file_path"],
                    "file_path": inv["file_path"],
                    "subset_name": "ALL",
                    "source_kind": inv_class,
                    "file_note": note,
                    "n_rows": np.nan,
                    "n_cols": len(keys),
                    "columns": ", ".join(keys),
                    "sgRNA_col": "",
                    "off_seq_col": "",
                    "label_col": "",
                    "read_col": "",
                    "dataset_col": "",
                    "schema_status": "npz_keys_only",
                    "label_semantics_status": "unknown",
                    "observed_positive_count": np.nan,
                    "unobserved_candidate_count": np.nan,
                    "unknown_label_count": np.nan,
                    "positive_ratio": np.nan,
                    "can_compute_auroc_auprc": False,
                    "unique_sgRNA_count": np.nan,
                    "unique_pair_count": np.nan,
                    "sgRNA_length_distribution": "",
                    "off_seq_length_distribution": "",
                    "canonical_23nt_pair_count": np.nan,
                    "can_extract_pam_offseq_20_23": False,
                    "pam_motif_count": np.nan,
                    "ngg_count": np.nan,
                    "non_ngg_count": np.nan,
                    "malformed_pam_count": np.nan,
                    "pair_overlap_cclmoff_train": np.nan,
                    "pair_overlap_cclmoff_val": np.nan,
                    "pair_overlap_cclmoff_test": np.nan,
                    "pair_overlap_cclmoff_any": np.nan,
                    "pair_overlap_fraction_any": np.nan,
                    "sgRNA_overlap_cclmoff_train": np.nan,
                    "sgRNA_overlap_cclmoff_val": np.nan,
                    "sgRNA_overlap_cclmoff_test": np.nan,
                    "sgRNA_overlap_cclmoff_any": np.nan,
                    "overlap_status": "unknown",
                    "risk_flags": "npz_not_fully_audited",
                    "feasibility_status": fstatus,
                    "recommended_next_step": "Extract raw arrays to tabular form for full audit." if has_seq and has_label else "Infeasible for BL5 external eval without sequence arrays.",
                })
            except Exception as e:
                print(f"     NPZ error: {e}")
            continue

        # Tabular files
        if suffix not in {".csv", ".tsv", ".parquet", ".json", ".jsonl"}:
            continue

        try:
            if suffix == ".csv":
                df = pd.read_csv(fpath)
            elif suffix == ".tsv":
                df = pd.read_csv(fpath, sep="\t")
            elif suffix == ".parquet":
                df = pd.read_parquet(fpath)
            elif suffix == ".jsonl":
                df = pd.read_json(fpath, lines=True)
            elif suffix == ".json":
                with open(fpath) as f:
                    data = json.load(f)
                if isinstance(data, list) and data:
                    df = pd.DataFrame(data)
                else:
                    df = pd.DataFrame([data])
        except Exception as e:
            print(f"     read error: {e}")
            audit_rows.append({
                "candidate_id": inv["file_path"],
                "file_path": inv["file_path"],
                "subset_name": "ALL",
                "source_kind": inv_class,
                "file_note": f"Read error: {e}",
                "n_rows": np.nan, "n_cols": np.nan, "columns": "",
                "sgRNA_col": "", "off_seq_col": "", "label_col": "", "read_col": "", "dataset_col": "",
                "schema_status": "unreadable", "label_semantics_status": "unknown",
                "observed_positive_count": np.nan, "unobserved_candidate_count": np.nan,
                "unknown_label_count": np.nan, "positive_ratio": np.nan,
                "can_compute_auroc_auprc": False,
                "unique_sgRNA_count": np.nan, "unique_pair_count": np.nan,
                "sgRNA_length_distribution": "", "off_seq_length_distribution": "",
                "canonical_23nt_pair_count": np.nan, "can_extract_pam_offseq_20_23": False,
                "pam_motif_count": np.nan, "ngg_count": np.nan, "non_ngg_count": np.nan,
                "malformed_pam_count": np.nan,
                "pair_overlap_cclmoff_train": np.nan, "pair_overlap_cclmoff_val": np.nan,
                "pair_overlap_cclmoff_test": np.nan, "pair_overlap_cclmoff_any": np.nan,
                "pair_overlap_fraction_any": np.nan,
                "sgRNA_overlap_cclmoff_train": np.nan, "sgRNA_overlap_cclmoff_val": np.nan,
                "sgRNA_overlap_cclmoff_test": np.nan, "sgRNA_overlap_cclmoff_any": np.nan,
                "overlap_status": "unreadable",
                "risk_flags": "unreadable_file",
                "feasibility_status": "infeasible",
                "recommended_next_step": "Fix file encoding or format.",
            })
            continue

        cols = list(df.columns)
        sgrna_col, off_col, label_col, read_col, dataset_col = detect_columns(cols)

        # Determine source kind more precisely
        if "predictions" in inv["file_path"].lower() and "off_target_prob" in cols:
            source_kind = "derived_result_artifact"
        else:
            source_kind = inv_class

        # Audit ALL subset
        row_all = audit_subset(
            df, sgrna_col, off_col, label_col, read_col, dataset_col,
            cclmoff_train_pairs, cclmoff_val_pairs, cclmoff_test_pairs, cclmoff_any_pairs,
            cclmoff_train_sgrnas, cclmoff_val_sgrnas, cclmoff_test_sgrnas, cclmoff_any_sgrnas,
            "ALL", source_kind, inv["file_path"], ""
        )
        audit_rows.append(row_all)

        # Per-dataset subsets only for compatible schema or explicit predictions.csv
        is_predictions_csv = "predictions.csv" in inv["file_path"] and "crispr_dualpred" in inv["file_path"]
        has_compatible_schema = sgrna_col is not None and off_col is not None and label_col is not None
        if dataset_col is not None and dataset_col in df.columns and (has_compatible_schema or is_predictions_csv):
            for ds_name, ds_df in df.groupby(dataset_col, sort=False):
                if len(ds_df) == 0:
                    continue
                row_ds = audit_subset(
                    ds_df, sgrna_col, off_col, label_col, read_col, dataset_col,
                    cclmoff_train_pairs, cclmoff_val_pairs, cclmoff_test_pairs, cclmoff_any_pairs,
                    cclmoff_train_sgrnas, cclmoff_val_sgrnas, cclmoff_test_sgrnas, cclmoff_any_sgrnas,
                    str(ds_name), source_kind, inv["file_path"], f"subset of {inv['file_path']}"
                )
                audit_rows.append(row_ds)

    audit_df = pd.DataFrame(audit_rows)

    # ------------------------------------------------------------------
    # 3. Write outputs
    # ------------------------------------------------------------------
    # Schema
    schema_cols = ["candidate_id", "file_path", "subset_name", "source_kind", "n_rows", "n_cols", "columns",
                   "sgRNA_col", "off_seq_col", "label_col", "read_col", "dataset_col", "schema_status",
                   "feasibility_status", "risk_flags", "recommended_next_step"]
    audit_df[[c for c in schema_cols if c in audit_df.columns]].to_csv(out_dir / "external_dataset_schema_audit.csv", index=False)
    print("[audit] Wrote external_dataset_schema_audit.csv")

    # Label
    label_cols = ["candidate_id", "file_path", "subset_name", "label_semantics_status",
                  "observed_positive_count", "unobserved_candidate_count", "unknown_label_count",
                  "positive_ratio", "can_compute_auroc_auprc",
                  "feasibility_status", "risk_flags", "recommended_next_step"]
    audit_df[[c for c in label_cols if c in audit_df.columns]].to_csv(out_dir / "external_dataset_label_audit.csv", index=False)
    print("[audit] Wrote external_dataset_label_audit.csv")

    # Sequence
    seq_cols = ["candidate_id", "file_path", "subset_name",
                "sgRNA_length_distribution", "off_seq_length_distribution",
                "canonical_23nt_pair_count", "can_extract_pam_offseq_20_23",
                "pam_motif_count", "ngg_count", "non_ngg_count", "malformed_pam_count",
                "unique_sgRNA_count", "unique_pair_count",
                "feasibility_status", "risk_flags", "recommended_next_step"]
    audit_df[[c for c in seq_cols if c in audit_df.columns]].to_csv(out_dir / "external_dataset_sequence_pam_audit.csv", index=False)
    print("[audit] Wrote external_dataset_sequence_pam_audit.csv")

    # Overlap
    overlap_cols = ["candidate_id", "file_path", "subset_name",
                    "pair_overlap_cclmoff_train", "pair_overlap_cclmoff_val", "pair_overlap_cclmoff_test",
                    "pair_overlap_cclmoff_any", "pair_overlap_fraction_any",
                    "sgRNA_overlap_cclmoff_train", "sgRNA_overlap_cclmoff_val", "sgRNA_overlap_cclmoff_test",
                    "sgRNA_overlap_cclmoff_any", "overlap_status",
                    "feasibility_status", "risk_flags", "recommended_next_step"]
    audit_df[[c for c in overlap_cols if c in audit_df.columns]].to_csv(out_dir / "external_dataset_overlap_with_cclmoff.csv", index=False)
    print("[audit] Wrote external_dataset_overlap_with_cclmoff.csv")

    # Feasibility table
    feas_cols = ["candidate_id", "file_path", "subset_name", "n_rows", "schema_status",
                 "label_semantics_status", "can_compute_auroc_auprc",
                 "canonical_23nt_pair_count", "can_extract_pam_offseq_20_23",
                 "overlap_status", "pair_overlap_fraction_any",
                 "unique_sgRNA_count", "observed_positive_count", "unobserved_candidate_count",
                 "feasibility_status", "risk_flags", "recommended_next_step"]
    audit_df[[c for c in feas_cols if c in audit_df.columns]].to_csv(out_dir / "external_dataset_feasibility_table.csv", index=False)
    print("[audit] Wrote external_dataset_feasibility_table.csv")

    # JSON recommendation
    full = audit_df[audit_df["feasibility_status"] == "ready_for_strict_external_eval"]["candidate_id"].tolist()
    prov = audit_df[audit_df["feasibility_status"] == "provenance_required_limited_candidate"]["candidate_id"].tolist()
    limited = audit_df[audit_df["feasibility_status"] == "limited_external_eval_candidate"]["candidate_id"].tolist()
    smoke = audit_df[audit_df["feasibility_status"] == "smoke_test_only"]["candidate_id"].tolist()
    not_rec = audit_df[audit_df["feasibility_status"].isin([
        "overlap_not_independent", "result_artifact_not_dataset", "infeasible",
        "metadata_only", "positive_only_not_auc_feasible"
    ])]["candidate_id"].tolist()

    if full:
        decision = "ready_for_strict_external_eval_exists"
    elif prov:
        decision = "no_ready_strict_external_dataset_found_but_provenance_candidates_exist"
    elif limited:
        decision = "limited_candidates_only"
    else:
        decision = "no_feasible_external_dataset_in_repository"

    recommendation = {
        "purpose": "external dataset feasibility audit for BL5-v4-PAM cross-dataset generalization",
        "coordinate_contract": "PAM_original = off_seq[20:23]",
        "do_not_compare_historical_metrics_directly": True,
        "ready_for_strict_external_eval": full,
        "provenance_required_limited_candidates": prov,
        "limited_external_eval_candidates": limited,
        "smoke_test_only": smoke,
        "not_recommended": not_rec,
        "decision": decision,
        "recommended_next_step": (
            "Do provenance audit for limited candidates before any external evaluation."
            if prov else (
                "Limited candidates available with caveats." if limited else
                "No feasible external dataset found in repository."
            )
        ),
        "recommended_model_set_if_eval_after_provenance": [
            "BL0b-on-BL5split",
            "BL5-v4-NoPAM-control",
            "BL5-v4-PAM",
            "BL6-1-PAM-Gated-Fusion optional"
        ],
        "required_metrics_if_eval_after_provenance": [
            "AUROC", "AUPRC", "positive_ratio", "test_samples",
            "observed_positive", "unobserved_candidate", "unique_sgRNA_count",
            "NGG/non-NGG stratified metrics", "bootstrap CI"
        ]
    }
    json_path = out_dir / "recommended_external_eval.json"
    json_path.write_text(json.dumps(recommendation, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[audit] Wrote recommended_external_eval.json")

    # Markdown report
    build_markdown_report(out_dir, inventory_df, audit_df, full, prov, limited, smoke, not_rec, decision)
    print("[audit] Wrote external_dataset_feasibility_report.md")

    print("\n[audit] Done.")
    print(f"  Files scanned:          {len(all_files)}")
    print(f"  Detailed audit rows:    {len(audit_rows)}")
    print(f"  Ready strict:           {len(full)}")
    print(f"  Provenance required:    {len(prov)}")
    print(f"  Limited eval:           {len(limited)}")
    print(f"  Smoke test only:        {len(smoke)}")
    print(f"  Not recommended:        {len(not_rec)}")


def build_markdown_report(out_dir, inventory_df, audit_df, full, prov, limited, smoke, not_rec, decision):
    md_lines = [
        "# External Dataset Feasibility Audit v2",
        "",
        "> **Purpose**: Determine whether any files in the repository can serve as strict cross-dataset external benchmarks for BL5-v4-PAM generalization.",
        ">",
        "> **Coordinate Contract**: `PAM_original = off_seq[20:23]` (positions 21-23). Use of `off_seq[-3:]` is prohibited.",
        "",
        "---",
        "",
        "## 1. Summary",
        "",
        f"- **Total files scanned**: {len(inventory_df)}",
        f"- **External dataset candidates**: {int(inventory_df['is_external_dataset_candidate'].sum())}",
        f"- **Ready for strict external eval**: {len(full)}",
        f"- **Provenance-required limited candidates**: {len(prov)}",
        f"- **Limited external eval candidates**: {len(limited)}",
        f"- **Smoke test only**: {len(smoke)}",
        f"- **Not recommended / infeasible**: {len(not_rec)}",
        "",
        "---",
        "",
        "## 2. Inventory Classification",
        "",
        "| Inventory Class | Count |",
        "|:---|---:|",
    ]
    for cls, count in inventory_df["inventory_class"].value_counts().items():
        md_lines.append(f"| {cls} | {count} |")

    md_lines.extend([
        "",
        "---",
        "",
        "## 3. Detailed Candidate Audit Table",
        "",
        "| Candidate | Rows | Schema | Observed positive | Unobserved candidate | PAM 23nt? | Pair Overlap (train/val/test/any) | sgRNA | Status |",
        "|:---|---:|:---|---:|---:|:---:|:---|---:|:---|",
    ])

    for _, r in audit_df.iterrows():
        if r["feasibility_status"] in ["future_work_needed", "infeasible"] and r["subset_name"] != "ALL":
            continue
        po = r.get("pair_overlap_cclmoff_train", 0)
        pv = r.get("pair_overlap_cclmoff_val", 0)
        pt = r.get("pair_overlap_cclmoff_test", 0)
        pa = r.get("pair_overlap_cclmoff_any", 0)
        pf = r.get("pair_overlap_fraction_any", 0)
        overlap_str = f"{po}/{pv}/{pt}/{pa} ({pf:.2%})" if not np.isnan(pa) else "N/A"
        md_lines.append(
            f"| `{r['candidate_id']}` | {r.get('n_rows', '')} | {r.get('schema_status', '')} | "
            f"{r.get('observed_positive_count', '')} | {r.get('unobserved_candidate_count', '')} | "
            f"{'✅' if r.get('can_extract_pam_offseq_20_23') else '❌'} | {overlap_str} | "
            f"{r.get('unique_sgRNA_count', '')} | {r['feasibility_status']} |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        "## 4. Per-Dataset Audit for predictions.csv",
        "",
    ])

    pred_rows = audit_df[audit_df["file_path"].str.contains("predictions.csv", na=False)]
    for _, r in pred_rows.iterrows():
        md_lines.append(f"### `{r['candidate_id']}`")
        md_lines.append(f"- Rows: {r.get('n_rows', '')}, Schema: {r.get('schema_status', '')}")
        md_lines.append(f"- Label: {r.get('label_semantics_status', '')}, observed_positive={r.get('observed_positive_count', '')}, unobserved_candidate={r.get('unobserved_candidate_count', '')}")
        md_lines.append(f"- Sequence: canonical_23nt={r.get('canonical_23nt_pair_count', '')}, PAM_extractable={'✅' if r.get('can_extract_pam_offseq_20_23') else '❌'}")
        md_lines.append(f"- Overlap: {r.get('overlap_status', '')} ({r.get('pair_overlap_fraction_any', 0):.4%}) — train={r.get('pair_overlap_cclmoff_train', '')}, val={r.get('pair_overlap_cclmoff_val', '')}, test={r.get('pair_overlap_cclmoff_test', '')}, any={r.get('pair_overlap_cclmoff_any', '')}")
        md_lines.append(f"- Feasibility: **{r['feasibility_status']}**")
        md_lines.append(f"- Risk flags: {r['risk_flags']}")
        md_lines.append(f"- Recommendation: {r['recommended_next_step']}")
        md_lines.append("")

    md_lines.extend([
        "---",
        "",
        "## 5. Overlap by CCLMoff Train/Val/Test",
        "",
        "All overlap counts are **exact pair overlaps** (`sgRNA`, `off_seq`). sgRNA-only overlap is reported separately but does not disqualify a candidate on its own.",
        "",
    ])

    for _, r in audit_df.iterrows():
        if not np.isnan(r.get("pair_overlap_cclmoff_any", np.nan)):
            md_lines.append(
                f"- `{r['candidate_id']}`: train={r.get('pair_overlap_cclmoff_train', '')}, "
                f"val={r.get('pair_overlap_cclmoff_val', '')}, test={r.get('pair_overlap_cclmoff_test', '')}, "
                f"any={r.get('pair_overlap_cclmoff_any', '')} ({r.get('pair_overlap_fraction_any', 0):.4%})"
            )

    md_lines.extend([
        "",
        "---",
        "",
        "## 6. Feasibility Decision",
        "",
    ])

    if full:
        md_lines.append(f"**Strict external evaluation is ready** for: {', '.join(full)}")
    elif prov:
        md_lines.append("当前仓库内没有可直接支持 strict raw cross-dataset AUROC/AUPRC evaluation 的外部数据集。旧 five-dataset predictions.csv 整体因 heavy exact pair overlap 且为 old-model prediction artifact，不能作为整体 external benchmark。GUIDE-seq / CHANGE-seq / Tasi 子集也因高 pair overlap 不独立。")
        md_lines.append("")
        md_lines.append(f"**但以下子集可作为 provenance_required_limited_candidate**：{', '.join(prov)}")
        md_lines.append("")
        md_lines.append("正式 external evaluation 前，必须确认这些 rows 的原始数据来源、label semantics、candidate generation 口径，并从 prediction artifact 中剥离 raw sgRNA/dna/label 表。")
    elif limited:
        md_lines.append(f"**Limited external evaluation candidates**: {', '.join(limited)}")
    else:
        md_lines.append("**No feasible external dataset found in the repository.**")

    md_lines.extend([
        "",
        "---",
        "",
        "## 7. Recommended Next Step",
        "",
    ])

    if decision == "no_feasible_external_dataset_in_repository":
        md_lines.append("1. **Current repository cannot support strict cross-dataset evaluation.**")
        md_lines.append("2. External datasets with reliable paired sgRNA/off_seq, explicit label semantics, and sufficient candidate background are required.")
        md_lines.append("3. This is left as **future work**.")
    elif decision == "no_ready_strict_external_dataset_found_but_provenance_candidates_exist":
        md_lines.append("1. **Audit provenance** for SITE and K562 subsets: confirm original data source, label semantics, and candidate generation protocol.")
        md_lines.append("2. **Extract raw tables** from predictions.csv (sgRNA, dna, label only) if provenance is confirmed.")
        md_lines.append("3. **Re-run overlap audit** after provenance confirmation to verify independence.")
        md_lines.append("4. Only then run external eval with model set: BL0b-on-BL5split, BL5-v4-NoPAM-control, BL5-v4-PAM, BL6-1-PAM-Gated-Fusion optional.")
    else:
        md_lines.append(f"Decision: {decision}")

    md_lines.extend([
        "",
        "---",
        "",
        "## 8. Limitations",
        "",
        "- **GUIDE-seq P0/BL3 historical metrics cannot be directly compared** to CCLMoff formal BL5 split results. Different datasets, different candidate universes, different detection methods.",
        "- **Predictions artifacts are not raw datasets**: Files like `predictions.csv` contain model outputs, not original experimental data.",
        "- **Label semantics may differ across datasets**: CCLMoff `label=0` means `unobserved_candidate` (not detected), not `verified_safe`.",
        "- **CCLMoff `Method` / `Length` metadata is incomplete**: ~3.28M rows have empty `Method`. Cross-cell-line or cross-detection-method generalization cannot be reliably performed.",
        "- **PAM coordinate consistency**: All PAM extraction in this audit uses `off_seq[20:23]`.",
        "- **Large NPZ files** were not fully audited due to memory constraints; they require targeted chunked inspection if suspected to contain external data.",
        "",
    ])

    md_path = out_dir / "external_dataset_feasibility_report.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
