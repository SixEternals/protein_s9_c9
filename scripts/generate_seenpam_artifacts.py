"""
AGENTS.md compliance: [use_rnafm=False, freeze_rnafm=N/A,
                       split_mode=N/A, pos_weight=N/A]
本脚本仅做数据聚合与指标计算，不训练模型，不涉及 RNA-FM 推理。
"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, accuracy_score
from pathlib import Path
import json
from joblib import Parallel, delayed
import sys

out_dir = Path("results/bl5_generalization/pam_strict_holdout_seenpam_sanity")
out_dir.mkdir(parents=True, exist_ok=True)

models = [
    ("AGG", "PAM", "results/bl5_v4_pam_holdout_agg/test_seenPAM_predictions.csv"),
    ("AGG", "NoPAM", "results/bl5_v4_nopam_holdout_agg/test_seenPAM_predictions.csv"),
    ("TGG", "PAM", "results/bl5_v4_pam_holdout_tgg/test_seenPAM_predictions.csv"),
    ("TGG", "NoPAM", "results/bl5_v4_nopam_holdout_tgg/test_seenPAM_predictions.csv"),
    ("GAG", "PAM", "results/bl5_v4_pam_holdout_gag/test_seenPAM_predictions.csv"),
    ("GAG", "NoPAM", "results/bl5_v4_nopam_holdout_gag/test_seenPAM_predictions.csv"),
]

# 1. Pooled metrics
pooled_rows = []
for pam, model, path in models:
    df = pd.read_csv(path)
    y_true = df["label"].values
    y_prob = df["probability"].values
    y_pred = (y_prob >= 0.5).astype(int)
    n = len(df)
    pos = int(y_true.sum())
    neg = int(n - pos)
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    pooled_rows.append({
        "holdout_pam": pam,
        "model": model,
        "n_samples": n,
        "observed_positive": pos,
        "unobserved_candidate": neg,
        "AUROC": f"{auroc:.6f}",
        "AUPRC": f"{auprc:.6f}",
        "threshold_0.5_accuracy": f"{acc:.6f}",
        "threshold_0.5_precision": f"{prec:.6f}",
        "threshold_0.5_recall": f"{rec:.6f}",
        "threshold_0.5_f1": f"{f1:.6f}",
    })

pd.DataFrame(pooled_rows).to_csv(out_dir / "seenpam_pooled_metrics.csv", index=False)
print("Wrote seenpam_pooled_metrics.csv")

# 2. Pair deltas
pair_rows = []
for pam in ["AGG", "TGG", "GAG"]:
    pam_path = [path for p, m, path in models if p == pam and m == "PAM"][0]
    nopam_path = [path for p, m, path in models if p == pam and m == "NoPAM"][0]
    pam_df = pd.read_csv(pam_path)
    nopam_df = pd.read_csv(nopam_path)
    y_true = pam_df["label"].values
    pam_prob = pam_df["probability"].values
    nopam_prob = nopam_df["probability"].values
    pam_auroc = roc_auc_score(y_true, pam_prob)
    pam_auprc = average_precision_score(y_true, pam_prob)
    nopam_auroc = roc_auc_score(y_true, nopam_prob)
    nopam_auprc = average_precision_score(y_true, nopam_prob)
    pair_rows.append({
        "holdout_pam": pam,
        "delta_AUROC": f"{nopam_auroc - pam_auroc:.6f}",
        "delta_AUPRC": f"{nopam_auprc - pam_auprc:.6f}",
        "PAM_AUROC": f"{pam_auroc:.6f}",
        "PAM_AUPRC": f"{pam_auprc:.6f}",
        "NoPAM_AUROC": f"{nopam_auroc:.6f}",
        "NoPAM_AUPRC": f"{nopam_auprc:.6f}",
    })

pd.DataFrame(pair_rows).to_csv(out_dir / "seenpam_pair_deltas.csv", index=False)
print("Wrote seenpam_pair_deltas.csv")

# 3. Stratified metrics (TGG NGG vs non-NGG)
strat_rows = []
for pam, model, path in models:
    if pam != "TGG":
        continue
    df = pd.read_csv(path)
    for subset_name, mask in [
        ("NGG", df["PAM"].str.endswith("GG")),
        ("non_NGG", ~df["PAM"].str.endswith("GG")),
    ]:
        g = df[mask]
        if len(g) < 100:
            continue
        y_true = g["label"].values
        y_prob = g["probability"].values
        if y_true.sum() < 2 or (y_true == 0).sum() < 2:
            continue
        auroc = roc_auc_score(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
        strat_rows.append({
            "holdout_pam": pam,
            "model": model,
            "subset": subset_name,
            "n_samples": len(g),
            "observed_positive": int(y_true.sum()),
            "unobserved_candidate": int((y_true == 0).sum()),
            "AUROC": f"{auroc:.6f}",
            "AUPRC": f"{auprc:.6f}",
        })

pd.DataFrame(strat_rows).to_csv(out_dir / "seenpam_stratified_metrics.csv", index=False)
print("Wrote seenpam_stratified_metrics.csv")

# 4. Paired bootstrap
np.random.seed(42)
n_bootstrap = 10000
bootstrap_results = {}

def _bootstrap_iter(i, y_true, pam_prob, nopam_prob, n):
    idx = np.random.randint(0, n, size=n)
    yb = y_true[idx]
    if yb.sum() < 2 or (yb == 0).sum() < 2:
        return None
    pam_auroc = roc_auc_score(yb, pam_prob[idx])
    pam_auprc = average_precision_score(yb, pam_prob[idx])
    nopam_auroc = roc_auc_score(yb, nopam_prob[idx])
    nopam_auprc = average_precision_score(yb, nopam_prob[idx])
    return (nopam_auroc - pam_auroc, nopam_auprc - pam_auprc)

for pam in ["AGG", "TGG", "GAG"]:
    pam_path = [path for p, m, path in models if p == pam and m == "PAM"][0]
    nopam_path = [path for p, m, path in models if p == pam and m == "NoPAM"][0]
    pam_df = pd.read_csv(pam_path)
    nopam_df = pd.read_csv(nopam_path)
    y_true = pam_df["label"].values.astype(int)
    pam_prob = pam_df["probability"].values.astype(float)
    nopam_prob = nopam_df["probability"].values.astype(float)
    n = len(y_true)

    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(_bootstrap_iter)(i, y_true, pam_prob, nopam_prob, n)
        for i in range(n_bootstrap)
    )
    results = [r for r in results if r is not None]
    valid = len(results)
    deltas_auroc = np.array([r[0] for r in results])
    deltas_auprc = np.array([r[1] for r in results])
    auroc_ci = (float(np.percentile(deltas_auroc, 2.5)), float(np.percentile(deltas_auroc, 97.5)))
    auprc_ci = (float(np.percentile(deltas_auprc, 2.5)), float(np.percentile(deltas_auprc, 97.5)))

    bootstrap_results[pam] = {
        "n_bootstrap": n_bootstrap,
        "valid_samples": valid,
        "delta_AUROC_mean": float(np.mean(deltas_auroc)),
        "delta_AUROC_std": float(np.std(deltas_auroc)),
        "delta_AUROC_ci_95": auroc_ci,
        "delta_AUROC_ci_crosses_zero": (auroc_ci[0] <= 0 <= auroc_ci[1]),
        "delta_AUPRC_mean": float(np.mean(deltas_auprc)),
        "delta_AUPRC_std": float(np.std(deltas_auprc)),
        "delta_AUPRC_ci_95": auprc_ci,
        "delta_AUPRC_ci_crosses_zero": (auprc_ci[0] <= 0 <= auprc_ci[1]),
    }
    print(f"Bootstrap {pam}: valid={valid}, delta_AUROC_ci={auroc_ci}, delta_AUPRC_ci={auprc_ci}")

with open(out_dir / "paired_bootstrap_seenpam.json", "w") as f:
    json.dump(bootstrap_results, f, indent=2)
print("Wrote paired_bootstrap_seenpam.json")
print("All artifacts generated.")
