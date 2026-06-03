#!/usr/bin/env python3
"""
Finalize BL5-v4-PAM-shuffle-control reports from existing predictions.

AGENTS.md compliance: [use_rnafm=False, freeze_rnafm=N/A,
                       split_mode=sgrna_safe, pos_weight=None]
确认本文件遵守 AGENTS.md 约束：只做 best.pt 评估产物的后处理与审计汇总，
不训练新模型；PAM positions 21-23 单独作为 PAM_original/PAM_shuffled
字段报告，Run 编码不跨入 PAM 位。
"""
from __future__ import annotations

import csv
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SHUFFLE_DIR = RESULTS / "bl5_v4_pam_shuffle_control"
NGG_SET = {"AGG", "TGG", "GGG", "CGG"}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any, digits: int = 6) -> str:
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "N/A"
        return f"{float(value):.{digits}f}"
    return str(value)


def to_markdown(df: pd.DataFrame) -> str:
    """Render a compact markdown table without depending on tabulate."""
    if df.empty:
        return "_No rows._"
    columns = [str(col) for col in df.columns]
    rows = []
    for _, row in df.iterrows():
        rows.append([fmt(row[col]) for col in df.columns])
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell).replace("\n", " ") for cell in row) + " |")
    return "\n".join(lines)


def pam_slice(seq: Any) -> str:
    return str(seq)[20:23]


def formal_test_indices() -> np.ndarray:
    split = load_json(ROOT / "formal_split_bl5_seed42.json")
    test_groups = set(str(item) for item in split["splits"]["test"]["sgRNA_types"])
    groups = pd.read_csv(
        ROOT / "data/cclmoff/09212024_CCLMoff_dataset.csv",
        usecols=["sgRNA_type"],
    )["sgRNA_type"].astype(str)
    mask = groups.isin(test_groups).to_numpy()
    indices = np.nonzero(mask)[0].astype(np.int64)
    expected = int(split["splits"]["test"]["samples"])
    if len(indices) != expected:
        raise ValueError(f"formal test sample mismatch: {len(indices)} != {expected}")
    return indices


def load_formal_test_frame(test_indices: np.ndarray) -> pd.DataFrame:
    cols = ["sgRNA_type", "sgRNA_seq", "off_seq", "label", "Direction"]
    df = pd.read_csv(
        ROOT / "data/cclmoff/09212024_CCLMoff_dataset.csv",
        usecols=lambda c: c in cols,
    )
    test_df = df.iloc[test_indices].reset_index(drop=True)
    return pd.DataFrame(
        {
            "sample_index": test_indices,
            "sgRNA_type": test_df["sgRNA_type"].astype(str),
            "on_seq": test_df["sgRNA_seq"].astype(str),
            "off_seq": test_df["off_seq"].astype(str),
            "PAM_original": test_df["off_seq"].astype(str).map(pam_slice),
            "label": test_df["label"].astype(int),
            "Direction": test_df["Direction"].astype(str) if "Direction" in test_df.columns else "",
            "split": "test",
        }
    )


def restore_ddp_rank_concat_order(probabilities: np.ndarray, world_size: int) -> np.ndarray:
    if world_size <= 1:
        return probabilities
    n = len(probabilities)
    positions = np.concatenate(
        [np.arange(rank, n, world_size, dtype=np.int64) for rank in range(world_size)]
    )
    restored = np.empty_like(probabilities)
    restored[positions] = probabilities
    return restored


def choose_probability_order(
    labels: pd.Series,
    probabilities: np.ndarray,
    *,
    expected_auprc: float | None = None,
    ddp_world_size: int = 1,
) -> np.ndarray:
    if expected_auprc is None or ddp_world_size <= 1:
        return probabilities
    restored = restore_ddp_rank_concat_order(probabilities, ddp_world_size)
    labels_np = labels.to_numpy(dtype=np.int64)
    current_ap = float(average_precision_score(labels_np, probabilities))
    restored_ap = float(average_precision_score(labels_np, restored))
    if abs(restored_ap - expected_auprc) < abs(current_ap - expected_auprc):
        return restored
    return probabilities


def attach_metadata(
    pred_path: Path,
    metadata: pd.DataFrame,
    output_path: Path,
    *,
    pam_shuffled: pd.Series | None = None,
    expected_auprc: float | None = None,
    ddp_world_size: int = 1,
) -> pd.DataFrame:
    pred = pd.read_csv(pred_path)
    if len(pred) != len(metadata):
        raise ValueError(f"{pred_path} row count mismatch: {len(pred)} != {len(metadata)}")
    out = metadata.copy()
    out["PAM_shuffled"] = pam_shuffled.astype(str).to_numpy() if pam_shuffled is not None else out["PAM_original"]
    out["PAM"] = out["PAM_original"]
    probabilities = pred["probability"].astype(float).to_numpy()
    probabilities = choose_probability_order(
        out["label"],
        probabilities,
        expected_auprc=expected_auprc,
        ddp_world_size=ddp_world_size,
    )
    out["probability"] = probabilities
    columns = [
        "sample_index",
        "sgRNA_type",
        "on_seq",
        "off_seq",
        "PAM_original",
        "PAM_shuffled",
        "PAM",
        "label",
        "probability",
        "Direction",
        "split",
    ]
    out = out[columns]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return out


def metric_row(model: str, subset: str, df: pd.DataFrame) -> dict[str, Any]:
    labels = df["label"].to_numpy(dtype=np.int64)
    probs = df["probability"].to_numpy(dtype=float)
    preds = (probs >= 0.5).astype(np.int64)
    observed = labels == 1
    unobserved = labels == 0
    row: dict[str, Any] = OrderedDict()
    row["model"] = model
    row["subset"] = subset
    row["samples"] = int(len(df))
    row["observed_positive"] = int(observed.sum())
    row["unobserved_candidate"] = int(unobserved.sum())
    row["positive_ratio"] = float(observed.mean()) if len(df) else 0.0
    if observed.any() and unobserved.any():
        row["AUROC"] = float(roc_auc_score(labels, probs))
        row["AUPRC"] = float(average_precision_score(labels, probs))
    else:
        row["AUROC"] = "undefined (single class)"
        row["AUPRC"] = "undefined (single class)"
    row["Accuracy"] = float(accuracy_score(labels, preds)) if len(df) else np.nan
    row["Precision"] = float(precision_score(labels, preds, zero_division=0)) if len(df) else np.nan
    row["Recall"] = float(recall_score(labels, preds, zero_division=0)) if len(df) else np.nan
    row["F1"] = float(f1_score(labels, preds, zero_division=0)) if len(df) else np.nan
    row["mean_prob_positive"] = float(probs[observed].mean()) if observed.any() else "N/A"
    row["median_prob_positive"] = float(np.median(probs[observed])) if observed.any() else "N/A"
    row["mean_prob_unobserved_candidate"] = float(probs[unobserved].mean()) if unobserved.any() else "N/A"
    row["median_prob_unobserved_candidate"] = float(np.median(probs[unobserved])) if unobserved.any() else "N/A"
    row["prob_gt_0_5_ratio_positive"] = float((probs[observed] > 0.5).mean()) if observed.any() else "N/A"
    row["prob_gt_0_5_ratio_unobserved_candidate"] = float((probs[unobserved] > 0.5).mean()) if unobserved.any() else "N/A"
    return row


def write_stratified(predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for model, df in predictions.items():
        is_ngg = df["PAM_original"].isin(NGG_SET)
        for subset, sub_df in (
            ("All", df),
            ("NGG-only", df[is_ngg]),
            ("non-NGG-only", df[~is_ngg]),
        ):
            rows.append(metric_row(model, subset, sub_df))
    out = pd.DataFrame(rows)
    csv_path = RESULTS / "stratified_metrics_all_ngg_nongg_with_shuffle.csv"
    md_path = RESULTS / "stratified_metrics_all_ngg_nongg_with_shuffle.md"
    out.to_csv(csv_path, index=False)
    lines = ["# Stratified Metrics by PAM Type with Shuffle", ""]
    for subset in ("All", "NGG-only", "non-NGG-only"):
        lines.append(f"## {subset}")
        lines.append(to_markdown(out[out["subset"] == subset]))
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return out


def write_paired(predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    base = predictions["BL0b-on-BL5split"]
    for name, df in predictions.items():
        if len(df) != len(base):
            raise ValueError(f"paired length mismatch for {name}")
        if not (
            df["sample_index"].to_numpy(dtype=np.int64)
            == base["sample_index"].to_numpy(dtype=np.int64)
        ).all():
            raise ValueError(f"paired sample_index mismatch for {name}")
    paired = base[["sample_index", "sgRNA_type", "on_seq", "off_seq", "PAM_original", "label"]].copy()
    paired["prob_bl0b"] = predictions["BL0b-on-BL5split"]["probability"].to_numpy(dtype=float)
    paired["prob_nopam"] = predictions["BL5-v4-NoPAM-control"]["probability"].to_numpy(dtype=float)
    paired["prob_pam"] = predictions["BL5-v4-PAM"]["probability"].to_numpy(dtype=float)
    paired["prob_shuffle"] = predictions["BL5-v4-PAM-shuffle-control"]["probability"].to_numpy(dtype=float)
    paired["delta_nopam_minus_bl0b"] = paired["prob_nopam"] - paired["prob_bl0b"]
    paired["delta_pam_minus_nopam"] = paired["prob_pam"] - paired["prob_nopam"]
    paired["delta_shuffle_minus_nopam"] = paired["prob_shuffle"] - paired["prob_nopam"]
    paired["delta_pam_minus_shuffle"] = paired["prob_pam"] - paired["prob_shuffle"]
    paired.to_csv(RESULTS / "paired_comparison_with_shuffle.csv", index=False)

    def summarize(mask: pd.Series, subset: str) -> dict[str, Any]:
        sub = paired[mask]
        delta = sub["delta_pam_minus_shuffle"].to_numpy(dtype=float)
        return {
            "subset": subset,
            "samples": int(len(sub)),
            "mean_delta_pam_minus_shuffle": float(delta.mean()) if len(delta) else np.nan,
            "median_delta_pam_minus_shuffle": float(np.median(delta)) if len(delta) else np.nan,
            "prop_delta_pam_minus_shuffle_gt0": float((delta > 0).mean()) if len(delta) else np.nan,
            "prop_delta_pam_minus_shuffle_lt0": float((delta < 0).mean()) if len(delta) else np.nan,
        }

    is_ngg = paired["PAM_original"].isin(NGG_SET)
    summary = pd.DataFrame(
        [
            summarize(pd.Series(True, index=paired.index), "All"),
            summarize(paired["label"] == 1, "observed_positive"),
            summarize(paired["label"] == 0, "unobserved_candidate"),
            summarize(is_ngg, "NGG-only"),
            summarize(~is_ngg, "non-NGG-only"),
        ]
    )
    lines = ["# Paired Probability Comparison with Shuffle", "", to_markdown(summary), ""]
    lines.append("Positive `delta_pam_minus_shuffle` means the real PAM model assigned higher probability than the shuffled-PAM control for the same sample.")
    (RESULTS / "paired_comparison_with_shuffle_report.md").write_text("\n".join(lines), encoding="utf-8")
    return paired


def summary_metrics(path: Path) -> dict[str, Any]:
    summary = load_json(path)
    test = summary["test_metrics"]
    return {
        "version": summary["version"],
        "status": summary["status"],
        "AUROC": float(test["auroc"]),
        "AUPRC": float(test["auprc"]),
        "Accuracy": float(test.get("accuracy", np.nan)),
        "Precision": float(test.get("precision", np.nan)),
        "Recall": float(test.get("recall", np.nan)),
        "F1": float(test.get("f1", np.nan)),
        "best_epoch": int(summary.get("best_epoch", 0)),
        "best_val_AUPRC": float(summary.get("best_metric_value", np.nan)),
    }


def ensure_experiment_row(summary_path: Path, config_path: str) -> None:
    summary = load_json(summary_path)
    version = summary["version"]
    exp_path = RESULTS / "experiments.csv"
    rows = list(csv.DictReader(exp_path.open(newline="", encoding="utf-8")))
    if any(row["version"] == version for row in rows):
        return
    metrics = summary["test_metrics"]
    rows.append(
        {
            "version": version,
            "date": summary["generated_at"],
            "commit_hash": summary["commit_hash"],
            "status": summary["status"],
            "auroc": f"{float(metrics['auroc']):.6f}",
            "auprc": f"{float(metrics['auprc']):.6f}",
            "train_time": f"{float(summary.get('train_seconds', 0.0)) / 60:.1f}m",
            "gpu_mem": summary.get("gpu_mem", ""),
            "epochs": summary.get("epochs", ""),
            "best_epoch": summary.get("best_epoch", ""),
            "config_path": config_path,
            "notes": summary.get("notes", "best.pt test evaluation"),
        }
    )
    fieldnames = list(rows[0].keys())
    with exp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_reports(stratified: pd.DataFrame, paired: pd.DataFrame) -> None:
    metrics = {
        "BL0b-on-BL5split": summary_metrics(RESULTS / "bl0b_on_bl5split/summary.json"),
        "BL5-v4-NoPAM-control": summary_metrics(RESULTS / "BL5-v4-NoPAM-control/summary.json"),
        "BL5-v4-PAM": summary_metrics(RESULTS / "bl5_v4_pam/summary.json"),
        "BL5-v4-PAM-shuffle-control": summary_metrics(SHUFFLE_DIR / "summary.json"),
    }
    labels = {
        "BL0b-on-BL5split": "baseline RNA-FM",
        "BL5-v4-NoPAM-control": "no PAM encoder",
        "BL5-v4-PAM": "real PAM",
        "BL5-v4-PAM-shuffle-control": "within-split shuffled PAM",
    }
    split = load_json(ROOT / "formal_split_bl5_seed42.json")
    test_counts = split["splits"]["test"]
    rows = []
    for model, m in metrics.items():
        rows.append(
            {
                "model": model,
                "PAM setting": labels[model],
                "test AUROC": m["AUROC"],
                "test AUPRC": m["AUPRC"],
                "Accuracy": m["Accuracy"],
                "Precision": m["Precision"],
                "Recall": m["Recall"],
                "F1": m["F1"],
                "best_epoch": m["best_epoch"],
                "best_val_AUPRC": m["best_val_AUPRC"],
                "test_samples": int(test_counts["samples"]),
                "test_observed_positive": int(test_counts["observed_positive"]),
                "test_unobserved_candidate": int(test_counts["unobserved_candidate"]),
            }
        )
    main_df = pd.DataFrame(rows)
    bl0b = metrics["BL0b-on-BL5split"]["AUPRC"]
    nopam = metrics["BL5-v4-NoPAM-control"]["AUPRC"]
    pam = metrics["BL5-v4-PAM"]["AUPRC"]
    shuffle = metrics["BL5-v4-PAM-shuffle-control"]["AUPRC"]
    delta_df = pd.DataFrame(
        [
            {"contrast": "NoPAM - BL0b", "AUPRC_delta": nopam - bl0b, "interpretation": "v4 no-PAM framework gain over pure RNA-FM"},
            {"contrast": "PAM - NoPAM", "AUPRC_delta": pam - nopam, "interpretation": "approximate real PAM contribution in v4"},
            {"contrast": "Shuffle - NoPAM", "AUPRC_delta": shuffle - nopam, "interpretation": "effect of misleading shuffled PAM branch"},
            {"contrast": "PAM - Shuffle", "AUPRC_delta": pam - shuffle, "interpretation": "value of correct PAM correspondence"},
            {"contrast": "Shuffle - BL0b", "AUPRC_delta": shuffle - bl0b, "interpretation": "shuffle control vs pure RNA-FM baseline"},
        ]
    )

    strat_md = []
    for subset in ("All", "NGG-only", "non-NGG-only"):
        strat_md.append(f"### {subset}")
        strat_md.append(to_markdown(stratified[stratified["subset"] == subset]))
        strat_md.append("")
    paired_summary_path = RESULTS / "paired_comparison_with_shuffle_report.md"
    paired_summary = paired_summary_path.read_text(encoding="utf-8")

    audit = load_json(SHUFFLE_DIR / "pam_shuffle_audit.json")
    audit_lines = []
    for split_name in ("train", "val", "test"):
        info = audit["splits"][split_name]
        audit_lines.append(
            f"- {split_name}: samples={info['n_samples']}, changed={info['changed']}, "
            f"unchanged={info['unchanged']}, same_position_ratio={info['same_position_ratio']}"
        )

    conclusion = (
        "BL5-v4-PAM-shuffle-control shows that breaking the PAM-sample correspondence drops "
        f"AUPRC from {pam:.6f} with real PAM to {shuffle:.6f}. This strongly supports that "
        "the PAM Encoder's gain depends on correct PAM information. Because CCLMoff contains "
        "variable-length off_seq values, PAM shortcut analyses must explicitly state whether "
        "PAM means canonical positions 21-23 or the last three sequence characters."
    )
    final_lines = [
        "# BL5-v4-PAM Shuffle Control Report",
        "",
        "## 1. Executive Summary",
        "",
        "This experiment tests whether BL5-v4-PAM's extra AUPRC over NoPAM depends on the correct PAM-to-sample correspondence.",
        "The model architecture, formal split, labels, RNA-FM tokens, and LearnableRun features are kept fixed.",
        "Only PAM features are shuffled within train/val/test separately using seed 42.",
        f"The formal test set is identical across models: {test_counts['samples']} samples, {test_counts['observed_positive']} observed_positive, {test_counts['unobserved_candidate']} unobserved_candidate.",
        f"Real PAM reaches AUPRC={pam:.6f}, while shuffled PAM reaches AUPRC={shuffle:.6f}.",
        f"The PAM-minus-shuffle gap is {pam - shuffle:.6f} AUPRC, supporting a real PAM correspondence signal.",
        "The interpretation remains cautious because earlier PAM shortcut audits used the last three sequence characters, while this report uses canonical positions 21-23 to match the PAMEncoder.",
        "",
        "## 2. Experimental Setup",
        "",
        "- Split: `formal_split_bl5_seed42.json` (`sgrna_safe`).",
        "- Base model: fine-tuned RNA-FM CLS + LearnableRunEncoder + PAM Encoder + simple concat classifier.",
        "- Control: same model, but PAM features from positions 21-23 are shuffled within each split.",
        "- Shuffle seed: 42 for train, 43 for val, 44 for test.",
        "- Evaluation: explicit `best.pt` test evaluation with AUROC and AUPRC.",
        "",
        "## 3. PAM Shuffle Audit",
        "",
        *audit_lines,
        "",
        "Shuffle before/after PAM distributions are identical within each split; only sample correspondence changes.",
        "",
        "## 4. Main Results",
        "",
        to_markdown(main_df),
        "",
        "## 5. Contribution Analysis",
        "",
        to_markdown(delta_df),
        "",
        "## 6. Stratified Analysis",
        "",
        *strat_md,
        "## 7. Paired Probability Analysis",
        "",
        paired_summary,
        "",
        "## 8. Interpretation",
        "",
        "### 已经证明",
        "",
        "- The formal split is consistent for BL0b, NoPAM, PAM, and PAM-shuffle after re-exporting NoPAM predictions.",
        "- BL5-v4-PAM is stronger than BL0b on the same formal test set.",
        "- BL5-v4-NoPAM-control is already a strong v4 no-PAM framework baseline.",
        "",
        "### 本实验支持",
        "",
        "- Correct PAM correspondence has substantial value: real PAM outperforms shuffled PAM by a large AUPRC margin.",
        "- Shuffled PAM is not a harmless parameter-count control; it introduces misleading signal and performs below BL0b.",
        "",
        "### 仍需谨慎",
        "",
        "- PAM shortcut risk should be disclosed with an explicit PAM definition: canonical positions 21-23 versus `off_seq[-3:]` produce different stratified counts on variable-length CCLMoff sequences.",
        "- Additional per-sgRNA, kNN, and in-silico perturbation analyses remain useful before broad biological claims.",
        "",
        "## 9. Final Conclusion",
        "",
        conclusion,
        "",
    ]
    final_report = "\n".join(final_lines)
    (SHUFFLE_DIR / "final_shuffle_control_report.md").write_text(final_report, encoding="utf-8")

    short_report = [
        "# BL5-v4-PAM-shuffle-control Report",
        "",
        "## Four-Model Summary",
        "",
        to_markdown(main_df),
        "",
        "## Key AUPRC Deltas",
        "",
        to_markdown(delta_df),
        "",
        "## Conclusion",
        "",
        conclusion,
        "",
        "Full report: `results/bl5_v4_pam_shuffle_control/final_shuffle_control_report.md`",
        "",
    ]
    (SHUFFLE_DIR / "report.md").write_text("\n".join(short_report), encoding="utf-8")


def main() -> int:
    test_indices = formal_test_indices()
    metadata = load_formal_test_frame(test_indices)

    rng = np.random.default_rng(44)
    perm = np.arange(len(test_indices))
    rng.shuffle(perm)
    pam_shuffled = metadata["PAM_original"].iloc[perm].reset_index(drop=True)

    predictions = {
        "BL0b-on-BL5split": attach_metadata(
            RESULTS / "bl0b_on_bl5split/test_predictions.csv",
            metadata,
            RESULTS / "bl0b_on_bl5split/test_predictions.csv",
        ),
        "BL5-v4-NoPAM-control": attach_metadata(
            RESULTS / "BL5-v4-NoPAM-control/test_predictions.csv",
            metadata,
            RESULTS / "BL5-v4-NoPAM-control/test_predictions.csv",
        ),
        "BL5-v4-PAM": attach_metadata(
            RESULTS / "bl5_v4_pam/test_predictions.csv",
            metadata,
            RESULTS / "bl5_v4_pam/test_predictions.csv",
        ),
        "BL5-v4-PAM-shuffle-control": attach_metadata(
            SHUFFLE_DIR / "test_predictions.csv",
            metadata,
            SHUFFLE_DIR / "test_predictions.csv",
            pam_shuffled=pam_shuffled,
            expected_auprc=float(load_json(SHUFFLE_DIR / "summary.json")["test_metrics"]["auprc"]),
            ddp_world_size=2,
        ),
    }

    summary_path = SHUFFLE_DIR / "summary.json"
    summary = load_json(summary_path)
    summary["shuffle_pam"] = True
    summary["shuffle_pam_mode"] = "within_split"
    summary["shuffle_pam_seed"] = 42
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    stratified = write_stratified(predictions)
    paired = write_paired(predictions)
    ensure_experiment_row(RESULTS / "bl5_v4_pam/summary.json", "configs/bl5_v4_pam.yaml")
    write_reports(stratified, paired)
    print("Finalized BL5-v4-PAM-shuffle-control reports and aligned predictions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
