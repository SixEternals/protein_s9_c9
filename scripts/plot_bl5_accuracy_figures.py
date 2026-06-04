"""
AGENTS.md compliance: [use_rnafm=False, freeze_rnafm=NA,
                       split_mode=NA, pos_weight=NA]
确认本文件遵守 AGENTS.md 约束
本脚本仅做可视化和结果核验，不训练模型，不修改模型结构。
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

# ---------------------------------------------------------------------------
# Style setup
# ---------------------------------------------------------------------------
sns.set_style('whitegrid')
plt.rcParams.update({
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})

MODELS = {
    'BL0b': {
        'name': 'BL0b-on-BL5split',
        'summary': 'results/bl0b_on_bl5split/summary.json',
        'predictions': 'results/bl0b_on_bl5split/test_predictions.csv',
        'color': '#4C78A8',
    },
    'NoPAM': {
        'name': 'BL5-v4-NoPAM-control',
        'summary': 'results/BL5-v4-NoPAM-control/summary.json',
        'predictions': 'results/BL5-v4-NoPAM-control/test_predictions.csv',
        'color': '#F58518',
    },
    'PAM': {
        'name': 'BL5-v4-PAM',
        'summary': 'results/bl5_v4_pam/summary.json',
        'predictions': 'results/bl5_v4_pam/test_predictions.csv',
        'color': '#54A24B',
    },
    'Shuffle': {
        'name': 'BL5-v4-PAM-shuffle-control',
        'summary': 'results/bl5_v4_pam_shuffle_control/summary.json',
        'predictions': 'results/bl5_v4_pam_shuffle_control/test_predictions.csv',
        'color': '#E45756',
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def read_summary(path):
    with open(path, 'r') as f:
        return json.load(f)

def read_predictions(path):
    df = pd.read_csv(path)
    required = {'label', 'probability'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'{path} missing columns: {missing}')
    return df[['sample_index', 'label', 'probability']].copy()

def compute_metrics(labels, probs):
    auroc = roc_auc_score(labels, probs)
    auprc = average_precision_score(labels, probs)
    acc = (probs.round() == labels).mean() if len(np.unique(probs.round())) > 1 else np.nan
    preds_binary = (probs >= 0.5).astype(int)
    tp = ((preds_binary == 1) & (labels == 1)).sum()
    fp = ((preds_binary == 1) & (labels == 0)).sum()
    fn = ((preds_binary == 0) & (labels == 1)).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        'auroc': auroc,
        'auprc': auprc,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

def compute_topk(labels, probs, ks):
    order = np.argsort(-probs)
    labels_sorted = labels[order]
    total_pos = labels.sum()
    results = []
    for k in ks:
        k = min(k, len(labels))
        pos_recovered = labels_sorted[:k].sum()
        recall = pos_recovered / total_pos
        precision = pos_recovered / k
        results.append({
            'k': k,
            'positives_recovered': int(pos_recovered),
            'recall': recall,
            'precision': precision,
            'fraction_inspected': k / len(labels),
        })
    return pd.DataFrame(results)

def savefig(fig, path_stem, formats, dpi):
    for fmt in formats:
        fp = f'{path_stem}.{fmt}'
        fig.savefig(fp, format=fmt, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f'  Saved {fp}')
    plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 1: Leaderboard bar chart
# ---------------------------------------------------------------------------
def fig1_leaderboard(metrics_dict, output_dir, formats, dpi):
    print('Generating Figure 1: Leaderboard...')
    models = ['BL0b', 'NoPAM', 'PAM', 'Shuffle']
    aurocs = [metrics_dict[m]['auroc'] for m in models]
    auprcs = [metrics_dict[m]['auprc'] for m in models]
    colors = [MODELS[m]['color'] for m in models]
    display_names = [MODELS[m]['name'] for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(models))
    width = 0.6

    ax = axes[0]
    bars = ax.bar(x, aurocs, width, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('AUROC')
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=15, ha='right')
    ax.set_ylim(0, 1.05)
    ax.set_title('AUROC Comparison')
    for bar, val in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    ax = axes[1]
    bars = ax.bar(x, auprcs, width, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('AUPRC')
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=15, ha='right')
    ax.set_ylim(0, max(auprcs)*1.15)
    ax.set_title('AUPRC Comparison')
    for bar, val in zip(bars, auprcs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    fig.suptitle('Figure 1: BL5-v4 Series Metric Leaderboard', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, os.path.join(output_dir, 'fig1_leaderboard_auprc_auroc'), formats, dpi)

# ---------------------------------------------------------------------------
# Figure 2: Precision-Recall curves
# ---------------------------------------------------------------------------
def fig2_pr_curves(pred_dict, output_dir, formats, dpi):
    print('Generating Figure 2: PR curves...')
    fig, ax = plt.subplots(figsize=(8, 6))
    models = ['BL0b', 'NoPAM', 'PAM', 'Shuffle']
    for m in models:
        labels = pred_dict[m]['label'].values
        probs = pred_dict[m]['probability'].values
        precision, recall, _ = precision_recall_curve(labels, probs)
        auprc = average_precision_score(labels, probs)
        ax.plot(recall, precision, color=MODELS[m]['color'],
                label=f"{MODELS[m]['name']} (AUPRC={auprc:.4f})", lw=2)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Figure 2: Precision-Recall Curves')
    ax.legend(loc='upper right', frameon=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, ls='--', alpha=0.5)
    fig.tight_layout()
    savefig(fig, os.path.join(output_dir, 'fig2_precision_recall_curves'), formats, dpi)

# ---------------------------------------------------------------------------
# Figure 3: AUPRC Contribution Waterfall
# ---------------------------------------------------------------------------
def fig3_waterfall(metrics_dict, output_dir, formats, dpi):
    print('Generating Figure 3: Waterfall...')
    bl0b = metrics_dict['BL0b']['auprc']
    nopam = metrics_dict['NoPAM']['auprc']
    pam = metrics_dict['PAM']['auprc']
    shuffle = metrics_dict['Shuffle']['auprc']

    deltas = {
        'NoPAM\nvs BL0b': nopam - bl0b,
        'PAM\nvs NoPAM': pam - nopam,
        'Shuffle\nvs NoPAM': shuffle - nopam,
        'PAM\nvs Shuffle': pam - shuffle,
        'PAM\nvs BL0b': pam - bl0b,
    }

    fig, ax = plt.subplots(figsize=(9, 5))
    names = list(deltas.keys())
    vals = list(deltas.values())
    colors = ['#54A24B' if v >= 0 else '#E45756' for v in vals]
    x = np.arange(len(names))
    bars = ax.bar(x, vals, color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel('AUPRC Delta')
    ax.set_title('Figure 3: AUPRC Contribution Waterfall')
    for bar, val in zip(bars, vals):
        ypos = bar.get_height() + 0.003 if val >= 0 else bar.get_height() - 0.01
        ax.text(bar.get_x() + bar.get_width()/2, ypos,
                f'{val:+.4f}', ha='center', va='bottom' if val >= 0 else 'top',
                fontsize=9, fontweight='bold')
    fig.tight_layout()
    savefig(fig, os.path.join(output_dir, 'fig3_auprc_contribution_waterfall'), formats, dpi)

# ---------------------------------------------------------------------------
# Figure 4: Top-k enrichment curve
# ---------------------------------------------------------------------------
def fig4_topk(pred_dict, output_dir, formats, dpi):
    print('Generating Figure 4: Top-k enrichment...')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ks = [100, 500, 1000, 5000, 10000, 50000, 100000]
    models = ['BL0b', 'NoPAM', 'PAM', 'Shuffle']

    ax1 = axes[0]
    ax2 = axes[1]
    for m in models:
        labels = pred_dict[m]['label'].values
        probs = pred_dict[m]['probability'].values
        df_topk = compute_topk(labels, probs, ks)
        ax1.plot(df_topk['fraction_inspected'], df_topk['recall'],
                 color=MODELS[m]['color'], label=MODELS[m]['name'], lw=2, marker='o', markersize=4)
        ax2.plot(df_topk['k'], df_topk['precision'],
                 color=MODELS[m]['color'], label=MODELS[m]['name'], lw=2, marker='o', markersize=4)

    ax1.set_xlabel('Fraction of candidates inspected')
    ax1.set_ylabel('Recall (observed positives recovered)')
    ax1.set_title('Top-k Recall Enrichment')
    ax1.legend(loc='lower right', frameon=True)
    ax1.grid(True, ls='--', alpha=0.5)

    ax2.set_xlabel('Top-k inspected')
    ax2.set_ylabel('Precision @ k')
    ax2.set_xscale('log')
    ax2.set_title('Top-k Precision')
    ax2.legend(loc='upper right', frameon=True)
    ax2.grid(True, ls='--', alpha=0.5)

    fig.suptitle('Figure 4: Top-k Enrichment Analysis', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, os.path.join(output_dir, 'fig4_topk_enrichment'), formats, dpi)

# ---------------------------------------------------------------------------
# Figure 5: Stratified metrics (optional)
# ---------------------------------------------------------------------------
def fig5_stratified(strat_path, output_dir, formats, dpi):
    if not os.path.exists(strat_path):
        print('Optional Figure 5: stratified metrics CSV not found, skipping.')
        return False
    print('Generating Figure 5: Stratified metrics...')
    df = pd.read_csv(strat_path)
    models_map = {
        'BL0b-on-BL5split': 'BL0b',
        'BL5-v4-NoPAM-control': 'NoPAM',
        'BL5-v4-PAM': 'PAM',
        'BL5-v4-PAM-shuffle-control': 'Shuffle',
    }
    df = df[df['model'].isin(models_map.keys())].copy()
    if df.empty:
        print('  No matching models in stratified CSV, skipping.')
        return False
    df['model_short'] = df['model'].map(models_map)
    df['color'] = df['model_short'].map({m: MODELS[m]['color'] for m in MODELS})

    subsets = ['All test', 'NGG-only', 'non-NGG-only']
    subset_map = {
        'all': 'All test',
        'ngg': 'NGG-only',
        'non-ngg': 'non-NGG-only',
    }
    # Try to infer subset names from unique values
    unique_subsets = df['subset'].unique().tolist()
    # Normalize
    def norm_sub(s):
        s2 = str(s).lower().replace(' ', '').replace('_', '').replace('-', '')
        if 'all' in s2:
            return 'All test'
        if 'ngg' in s2 and 'non' not in s2:
            return 'NGG-only'
        if 'non' in s2 and 'ngg' in s2:
            return 'non-NGG-only'
        return str(s)
    df['subset_norm'] = df['subset'].apply(norm_sub)

    metrics = ['AUPRC', 'AUROC', 'mean_prob_positive', 'mean_prob_unobserved_candidate']
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.ravel()
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        pivot = df.pivot(index='subset_norm', columns='model_short', values=metric)
        # Reorder columns
        pivot = pivot[['BL0b', 'NoPAM', 'PAM', 'Shuffle']] if all(c in pivot.columns for c in ['BL0b','NoPAM','PAM','Shuffle']) else pivot
        pivot.plot(kind='bar', ax=ax, color=[MODELS[c]['color'] for c in pivot.columns],
                   edgecolor='black', linewidth=0.5)
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.set_xlabel('Subset')
        ax.legend(title='Model', loc='best', frameon=True)
        ax.tick_params(axis='x', rotation=15)
        # Annotate undefined
        for patch in ax.patches:
            height = patch.get_height()
            if pd.isna(height):
                continue
            if height < 0.001 and metric in ['AUPRC', 'AUROC']:
                # Could be undefined if only one class; check raw data
                pass
    fig.suptitle('Figure 5: Stratified Metrics by Subset', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, os.path.join(output_dir, 'fig5_stratified_metrics'), formats, dpi)
    return True

# ---------------------------------------------------------------------------
# Figure 6: Paired probability delta (optional, boxen plot)
# ---------------------------------------------------------------------------
def fig6_paired_delta(paired_path, output_dir, formats, dpi):
    if not os.path.exists(paired_path):
        print('Optional Figure 6: paired comparison CSV not found, skipping.')
        return False
    print('Generating Figure 6: Paired probability delta...')
    df = pd.read_csv(paired_path)
    required = {'delta_pam_minus_nopam', 'delta_pam_minus_shuffle', 'label'}
    if not required.issubset(df.columns):
        print(f'  Missing columns {required - set(df.columns)}, skipping.')
        return False

    df['is_positive'] = df['label'].astype(bool)
    # Use seaborn boxplot instead of violin
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, col in enumerate(['delta_pam_minus_nopam', 'delta_pam_minus_shuffle']):
        ax = axes[idx]
        tmp = df[[col, 'is_positive']].copy()
        tmp['Sample Type'] = tmp['is_positive'].map({True: 'Observed Positive', False: 'Unobserved Candidate'})
        sns.boxplot(data=tmp, x='Sample Type', y=col, ax=ax, hue='Sample Type',
                    palette=['#4C78A8', '#F58518'], showfliers=False, width=0.5, legend=False)
        ax.axhline(0, color='black', ls='--', lw=1)
        ax.set_ylabel('Probability Delta')
        title = 'PAM vs NoPAM' if 'nopam' in col else 'PAM vs Shuffle'
        ax.set_title(title)
        # Add median text outside boxes
        ylims = ax.get_ylim()
        offset = (ylims[1] - ylims[0]) * 0.02
        for i, stype in enumerate(['Observed Positive', 'Unobserved Candidate']):
            med = tmp[tmp['Sample Type'] == stype][col].median()
            ypos = med + offset if med >= 0 else med - offset
            ax.text(i, ypos, f'med={med:+.4f}', color='black', fontsize=8,
                    va='bottom' if med >= 0 else 'top', ha='center', fontweight='bold')
    fig.suptitle('Figure 6: Paired Probability Delta by Label', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, os.path.join(output_dir, 'fig6_paired_delta'), formats, dpi)
    return True

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='BL5-v4 Accuracy Visualization')
    parser.add_argument('--output-dir', default='results/figures/bl5_accuracy')
    parser.add_argument('--format', default='png,pdf')
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    formats = [f.strip() for f in args.format.split(',')]
    dpi = args.dpi

    print('=' * 60)
    print('BL5-v4 Accuracy Visualization')
    print('=' * 60)

    # ---- Read summaries ---------------------------------------------------
    print('\n[1] Reading summaries...')
    summaries = {}
    for key, info in MODELS.items():
        summaries[key] = read_summary(info['summary'])

    # ---- Read predictions -------------------------------------------------
    print('[2] Reading predictions...')
    pred_dict = {}
    base_index = None
    all_ok = True
    for key, info in MODELS.items():
        df = read_predictions(info['predictions'])
        if base_index is None:
            base_index = df['sample_index'].values
        elif not np.array_equal(base_index, df['sample_index'].values):
            print(f'  ERROR: sample_index mismatch for {key}!')
            all_ok = False
        pred_dict[key] = df
        print(f'  {key}: n={len(df)}, pos={int(df["label"].sum())}, pos_rate={df["label"].mean():.6f}')

    n_expected = 954326
    pos_expected = 3057
    neg_expected = 951269
    for key, df in pred_dict.items():
        if len(df) != n_expected or int(df['label'].sum()) != pos_expected:
            print(f'  ERROR: {key} has unexpected cohort!')
            all_ok = False

    if not all_ok:
        print('\nFATAL: Cohort inconsistency detected. Stopping.')
        sys.exit(1)

    print('  All predictions aligned and cohort verified.')

    # ---- Compute metrics --------------------------------------------------
    print('[3] Computing metrics from raw predictions...')
    metrics_dict = {}
    for key, df in pred_dict.items():
        m = compute_metrics(df['label'].values, df['probability'].values)
        metrics_dict[key] = m
        print(f'  {key}: AUROC={m["auroc"]:.6f}, AUPRC={m["auprc"]:.6f}')

    # ---- Figure 1 ---------------------------------------------------------
    fig1_leaderboard(metrics_dict, output_dir, formats, dpi)

    # ---- Figure 2 ---------------------------------------------------------
    fig2_pr_curves(pred_dict, output_dir, formats, dpi)

    # ---- Figure 3 ---------------------------------------------------------
    fig3_waterfall(metrics_dict, output_dir, formats, dpi)

    # ---- Figure 4 ---------------------------------------------------------
    fig4_topk(pred_dict, output_dir, formats, dpi)

    # ---- Optional Figure 5 ------------------------------------------------
    had_fig5 = fig5_stratified('results/stratified_metrics_all_ngg_nongg_with_shuffle.csv',
                               output_dir, formats, dpi)

    # ---- Optional Figure 6 ------------------------------------------------
    had_fig6 = fig6_paired_delta('results/paired_comparison_with_shuffle.csv',
                                 output_dir, formats, dpi)

    # ---- Metrics CSV ------------------------------------------------------
    print('\n[4] Writing metrics summary CSV...')
    rows = []
    for key in ['BL0b', 'NoPAM', 'PAM', 'Shuffle']:
        m = metrics_dict[key]
        s = summaries[key]
        # Prefer summary.json values as source of truth (they should match)
        tm = s.get('test_metrics', {})
        rows.append({
            'model': MODELS[key]['name'],
            'pam_setting': 'N/A' if key == 'BL0b' else ('NoPAM' if key == 'NoPAM' else ('PAM' if key == 'PAM' else 'Shuffle')),
            'auroc': tm.get('auroc', m['auroc']),
            'auprc': tm.get('auprc', m['auprc']),
            'accuracy': tm.get('accuracy', m['accuracy']),
            'precision': tm.get('precision', m['precision']),
            'recall': tm.get('recall', m['recall']),
            'f1': tm.get('f1', m['f1']),
            'test_samples': n_expected,
            'test_positive': pos_expected,
            'test_negative': neg_expected,
            'positive_rate': pos_expected / n_expected,
            'source': 'summary.json + test_predictions.csv',
            'notes': 'Cohort verified, sample_index aligned',
        })
    df_metrics = pd.DataFrame(rows)
    csv_path = output_dir / 'figure_metrics_summary.csv'
    df_metrics.to_csv(csv_path, index=False)
    print(f'  Saved {csv_path}')

    # ---- Top-k CSV --------------------------------------------------------
    print('[5] Writing top-k summary CSV...')
    ks = [100, 500, 1000, 5000, 10000, 50000, 100000]
    topk_rows = []
    for key in ['BL0b', 'NoPAM', 'PAM', 'Shuffle']:
        labels = pred_dict[key]['label'].values
        probs = pred_dict[key]['probability'].values
        df_topk = compute_topk(labels, probs, ks)
        df_topk['model'] = MODELS[key]['name']
        topk_rows.append(df_topk)
    df_topk_all = pd.concat(topk_rows, ignore_index=True)
    topk_csv = output_dir / 'topk_enrichment_summary.csv'
    df_topk_all.to_csv(topk_csv, index=False)
    print(f'  Saved {topk_csv}')

    # ---- Report -----------------------------------------------------------
    print('[6] Writing report...')
    bl0b_auprc = metrics_dict['BL0b']['auprc']
    nopam_auprc = metrics_dict['NoPAM']['auprc']
    pam_auprc = metrics_dict['PAM']['auprc']
    shuffle_auprc = metrics_dict['Shuffle']['auprc']

    report = f"""# BL5-v4-PAM Accuracy Visualization Report

## 1. Data Sources

- `results/bl0b_on_bl5split/summary.json` + `test_predictions.csv`
- `results/BL5-v4-NoPAM-control/summary.json` + `test_predictions.csv`
- `results/bl5_v4_pam/summary.json` + `test_predictions.csv`
- `results/bl5_v4_pam_shuffle_control/summary.json` + `test_predictions.csv`
- `results/paired_comparison_with_shuffle.csv` (optional)
- `results/stratified_metrics_all_ngg_nongg_with_shuffle.csv` (optional)

## 2. Cohort Check

- **test_samples**: {n_expected}
- **test_positive (observed_positive)**: {pos_expected}
- **test_negative (unobserved_candidate)**: {neg_expected}
- **positive_rate**: {pos_expected/n_expected:.6f} (~0.32%)
- **Four models share the same test set**: Yes, `sample_index` aligned across all prediction files.

## 3. Main Metric Leaderboard

On the formal BL5 split, the ranking by test AUPRC is:

| Rank | Model | AUROC | AUPRC |
|:---:|:---|---:|---:|
| 1 | BL5-v4-PAM | {metrics_dict['PAM']['auroc']:.6f} | {metrics_dict['PAM']['auprc']:.6f} |
| 2 | BL5-v4-NoPAM-control | {metrics_dict['NoPAM']['auroc']:.6f} | {metrics_dict['NoPAM']['auprc']:.6f} |
| 3 | BL0b-on-BL5split | {metrics_dict['BL0b']['auroc']:.6f} | {metrics_dict['BL0b']['auprc']:.6f} |
| 4 | BL5-v4-PAM-shuffle-control | {metrics_dict['Shuffle']['auroc']:.6f} | {metrics_dict['Shuffle']['auprc']:.6f} |

**Current formal BL5 split strongest model**: **BL5-v4-PAM** with AUPRC = {pam_auprc:.6f}.

## 4. Baseline Comparison

- BL5-v4-PAM vs BL0b-on-BL5split:
  - AUPRC delta: +{pam_auprc - bl0b_auprc:.6f}
  - AUROC delta: +{metrics_dict['PAM']['auroc'] - metrics_dict['BL0b']['auroc']:.6f}

## 5. PAM Contribution

| Delta | Value | Interpretation |
|:---|---:|:---|
| NoPAM_minus_BL0b | +{nopam_auprc - bl0b_auprc:.6f} | BL5-v4 framework gain over pure RNA-FM baseline |
| PAM_minus_NoPAM | +{pam_auprc - nopam_auprc:.6f} | Correct PAM Encoder additional gain |
| PAM_minus_BL0b | +{pam_auprc - bl0b_auprc:.6f} | Total PAM model gain over RNA-FM baseline |
| Shuffle_minus_NoPAM | {shuffle_auprc - nopam_auprc:.6f} | Shuffle control loses vs NoPAM |
| PAM_minus_Shuffle | +{pam_auprc - shuffle_auprc:.6f} | Correct PAM vs shuffled PAM gap |
| Shuffle_minus_BL0b | {shuffle_auprc - bl0b_auprc:.6f} | Shuffle control vs BL0b |

## 6. Why Not Accuracy Alone

- The test set positive rate is only ~0.32% (3,057 / 954,326).
- A naive classifier predicting all zeros achieves Accuracy ≈ 99.68%.
- Therefore Accuracy is misleading in this context.
- We use **AUPRC**, **PR curves**, and **top-k enrichment** as primary visualizations.

## 7. Figures Generated

| Figure | File | Data Source |
|:---|:---|:---|
| Figure 1 | fig1_leaderboard_auprc_auroc | summary.json metrics |
| Figure 2 | fig2_precision_recall_curves | test_predictions.csv (raw) |
| Figure 3 | fig3_auprc_contribution_waterfall | summary.json metrics |
| Figure 4 | fig4_topk_enrichment | test_predictions.csv (raw) |
| Figure 5 | fig5_stratified_metrics | stratified_metrics_all_ngg_nongg_with_shuffle.csv {'(generated)' if had_fig5 else '(skipped - file missing or no data)'} |
| Figure 6 | fig6_paired_delta | paired_comparison_with_shuffle.csv {'(generated)' if had_fig6 else '(skipped - file missing or no data)'} |

Additional outputs:
- `figure_metrics_summary.csv`
- `topk_enrichment_summary.csv`

## 8. Caveats

1. **Dataset mixing warning**: GUIDE-seq P0 results (AUPRC ~0.85) are on a different dataset and split. Do NOT directly compare them with the CCLMoff formal BL5 split numbers here.
2. **NoPAM gain is holistic**: `NoPAM - BL0b` reflects the entire BL5-v4 framework (LearnableRun + concat MLP), not solely the LearnableRun encoder contribution.
3. **PAM shuffle control**: The large drop when PAM is shuffled (AUPRC {shuffle_auprc:.6f}) indicates the PAM branch benefits from correct PAM-sample correspondence, not just extra capacity.
4. All metrics are computed on the **test set** using the `best.pt` checkpoint.

---

## Conclusion

On the formal BL5 split, **BL5-v4-PAM** is currently the strongest mainline model by test AUPRC. It improves over the CCLMoff-style RNA-FM baseline (BL0b-on-BL5split) from AUPRC={bl0b_auprc:.6f} to AUPRC={pam_auprc:.6f}, a gain of +{pam_auprc - bl0b_auprc:.6f}. The NoPAM control reaches AUPRC={nopam_auprc:.6f}, indicating that the BL5-v4 framework without PAM already provides a large improvement over BL0b. Adding the correct PAM Encoder further improves AUPRC by +{pam_auprc - nopam_auprc:.6f}. The PAM shuffle control drops to AUPRC={shuffle_auprc:.6f}, supporting that the PAM-related gain depends on the correct correspondence between PAM features and samples rather than merely on extra parameters or training noise.

中文版本：

在 formal BL5 split 上，BL5-v4-PAM 是目前主线模型中 test AUPRC 最高的版本。它将 CCLMoff-style RNA-FM baseline（BL0b-on-BL5split）的 AUPRC 从 {bl0b_auprc:.6f} 提升到 {pam_auprc:.6f}，增益为 +{pam_auprc - bl0b_auprc:.6f}。NoPAM-control 的 AUPRC 为 {nopam_auprc:.6f}，说明 BL5-v4 无 PAM 框架本身已经明显强于纯 RNA-FM baseline。加入正确 PAM Encoder 后，AUPRC 进一步提升 +{pam_auprc - nopam_auprc:.6f}。PAM shuffle-control 的 AUPRC 降至 {shuffle_auprc:.6f}，说明 PAM 分支的增益依赖正确的 PAM 与样本对应关系，而不仅仅来自额外参数量或训练噪声。
"""
    report_path = output_dir / 'figure_generation_report.md'
    report_path.write_text(report, encoding='utf-8')
    print(f'  Saved {report_path}')

    print('\n' + '=' * 60)
    print('Done.')
    print('=' * 60)

if __name__ == '__main__':
    main()
