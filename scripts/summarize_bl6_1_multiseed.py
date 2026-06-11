#!/usr/bin/env python3
"""BL6-1 multi-seed (42/43/44) summary — reads summary.json only, no training.

AGENTS.md compliance: [use_rnafm=False, freeze_rnafm=N/A,
                       split_mode=sgrna_safe, pos_weight=N/A,
                       analysis_only=True]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

BL5_V4_PAM_AUPRC = 0.531281

SEED_DIRS = {
    42: "results/bl6_1_pam_gated_fusion",
    43: "results/bl6_1_pam_gated_fusion_seed43",
    44: "results/bl6_1_pam_gated_fusion_seed44",
}

OUTPUT_CSV = "results/bl6_1_validation/multiseed_summary.csv"
OUTPUT_JSON = "results/bl6_1_validation/multiseed_summary.json"


def load_seed(seed: int, dirname: str) -> dict:
    sp = Path(dirname) / "summary.json"
    if not sp.exists():
        raise FileNotFoundError(f"summary.json not found for seed {seed}: {sp}")
    s = json.loads(sp.read_text())
    return {
        "seed": seed,
        "version": s.get("version", ""),
        "output_dir": dirname,
        "status": s.get("status", ""),
        "auroc": s["test_metrics"]["auroc"],
        "auprc": s["test_metrics"]["auprc"],
        "delta_auprc_vs_bl5_v4_pam": s["test_metrics"]["auprc"] - BL5_V4_PAM_AUPRC,
        "above_bl5_v4_pam": s["test_metrics"]["auprc"] > BL5_V4_PAM_AUPRC,
        "best_epoch": s.get("best_epoch"),
        "epochs": s.get("epochs"),
        "planned_epochs": s.get("planned_epochs"),
        "train_seconds": s.get("train_seconds"),
        "train_time_min": s.get("train_seconds", 0) / 60.0,
        "gpu_mem": s.get("gpu_mem", ""),
        "best_checkpoint": s.get("artifacts", {}).get("best_checkpoint", ""),
        "prediction_csv": s.get("artifacts", {}).get("test_predictions", ""),
    }


def main() -> int:
    rows = []
    for seed in [42, 43, 44]:
        row = load_seed(seed, SEED_DIRS[seed])
        rows.append(row)
        print(f"seed={seed} AUROC={row['auroc']:.6f} AUPRC={row['auprc']:.6f} "
              f"delta={row['delta_auprc_vs_bl5_v4_pam']:+.6f} above={row['above_bl5_v4_pam']}")

    auprcs = np.array([r["auprc"] for r in rows])
    aurocs = np.array([r["auroc"] for r in rows])
    deltas = np.array([r["delta_auprc_vs_bl5_v4_pam"] for r in rows])

    n_above = int(np.sum([r["above_bl5_v4_pam"] for r in rows]))
    all_above = bool(n_above == 3)

    interpretation = (
        "BL6-1 multi-seed repeat shows mixed stability. "
        "Seeds 42 and 43 are above the BL5-v4-PAM baseline, but seed44 is below it. "
        f"The three-seed mean AUPRC ({auprcs.mean():.4f}) is "
        f"{'above' if auprcs.mean() > BL5_V4_PAM_AUPRC else 'below'} "
        f"the BL5 baseline ({BL5_V4_PAM_AUPRC}), and variance is large "
        f"(sample std={auprcs.std(ddof=1):.4f}). "
        "Current evidence does not support promoting BL6-1 to the main model "
        "or claiming stable advantage. "
        "Gate-collapse caveat remains based on seed42 gate audit; "
        "seed43/44 gate behavior has not yet been audited."
    )

    summary = {
        "baseline_bl5_v4_pam_auprc": BL5_V4_PAM_AUPRC,
        "seeds_included": [42, 43, 44],
        "n_completed": 3,
        "auprc_mean": float(auprcs.mean()),
        "auprc_std_sample": float(auprcs.std(ddof=1)),
        "auprc_min": float(auprcs.min()),
        "auprc_max": float(auprcs.max()),
        "auroc_mean": float(aurocs.mean()),
        "auroc_std_sample": float(aurocs.std(ddof=1)),
        "delta_auprc_vs_bl5_mean": float(deltas.mean()),
        "delta_auprc_vs_bl5_std_sample": float(deltas.std(ddof=1)),
        "delta_auprc_vs_bl5_min": float(deltas.min()),
        "delta_auprc_vs_bl5_max": float(deltas.max()),
        "n_above_bl5": n_above,
        "all_above_bl5": all_above,
        "interpretation": interpretation,
    }

    # Write CSV
    import pandas as pd
    out_csv = Path(OUTPUT_CSV)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    field_order = [
        "seed", "version", "output_dir", "status", "auroc", "auprc",
        "delta_auprc_vs_bl5_v4_pam", "above_bl5_v4_pam",
        "best_epoch", "epochs", "planned_epochs",
        "train_seconds", "train_time_min", "gpu_mem",
        "best_checkpoint", "prediction_csv",
    ]
    df[field_order].to_csv(out_csv, index=False)
    print(f"\nCSV written: {out_csv}")

    # Write JSON
    out_json = Path(OUTPUT_JSON)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"JSON written: {out_json}")

    # Print summary
    print(f"\n=== Multi-seed Summary ===")
    print(f"AUPRC: mean={auprcs.mean():.6f} std={auprcs.std(ddof=1):.6f} min={auprcs.min():.6f} max={auprcs.max():.6f}")
    print(f"Δ vs BL5: mean={deltas.mean():+.6f} std={deltas.std(ddof=1):.6f}")
    print(f"n_above_bl5: {n_above}/3, all_above: {all_above}")
    print(f"\n{interpretation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
