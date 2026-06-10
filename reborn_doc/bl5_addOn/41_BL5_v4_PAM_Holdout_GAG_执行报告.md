# BL5-v4 PAM Holdout GAG 执行报告

> **日期**: 2026-06-08  
> **执行 AI**: Kimi  
> **Commit**: `532529e`

---

## 1. 任务概述

BL5-v4 PAM 泛化实验的第三个 strict PAM holdout。AGG/TGG 已完成（均为高支持 NGG motif，方向相反）。GAG 是当前 feasibility audit 中唯一 feasible 的 **non-NGG** candidate，用于检验 PAM encoder 在 non-NGG unseen PAM 上的跨 PAM 泛化行为。

---

## 2. 执行步骤

### Phase 1 — Split 构造

- 脚本: `scripts/build_pam_strict_holdout_split.py`
- 参数: `--holdout_pam GAG --skip-size-validation`
- 输出: `results/bl5_generalization/pam_strict_holdout/GAG/`
- 验证:
  - train_H: 4,391,061 (0% GAG) ✅
  - val_H: 702,303 (0% GAG) ✅
  - test_H: 9,061 (100% GAG, 111 pos, 23 sgRNA types) ✅
  - sgRNA 无泄漏 ✅
  - PAM_original = off_seq[20:23] ✅
  - exact pair overlap test_H vs formal_train = 0 ✅

### Phase 2 — Config + Launcher

- `configs/bl5_v4_pam_holdout_gag.yaml` (use_pam_encoder=true)
- `configs/bl5_v4_nopam_holdout_gag.yaml` (use_pam_encoder=false)
- `run/run_bl5_v4_holdout_gag.sh`
- `scripts/auto_start_nopam_holdout_gag.sh`

### Phase 3 — 训练

| 模型 | 状态 | Best Epoch | Val AUPRC | 正式训练耗时 (experiments.csv) |
|:---|:---:|:---:|:---|:---|
| PAM | ✅ 完成 | 8 | 0.642083 | **163.5m** |
| NoPAM | ✅ 完成 | 8 | 0.633884 | **163.0m** |

### Phase 4 — Test_H Eval + Predictions Export

- PAM eval-only: test AUROC=0.999850, AUPRC=0.990692
- NoPAM eval-only: test AUROC=0.999801, AUPRC=0.989688
- Predictions CSV 行数一致 (9,061) ✅
- Label 一致 ✅
- PAM_original 全部为 GAG ✅

### Phase 5 — Paired Bootstrap

- n_bootstrap=**10,000**
- ΔAUROC = −0.000048, CI [−0.000160, +0.000037] → CI 跨 0
- ΔAUPRC = −0.001004, CI [−0.006062, +0.003711] → CI 跨 0
- **结论: GAG 上 PAM vs NoPAM 差异不显著**

### Phase 6 — Per-sgRNA Label Composition Sanity Audit

- GAG test_H 23 个 sgRNA_type
- 21 个 positive_only（105 pos, 0 neg）
- 0 个 unobserved_only
- 2 个 mixed（6 pos, 8,950 neg）
- 全部 unobserved_candidate 集中在 2 个 mixed sgRNA_type

### Phase 7 — Mixed-only Sanity Metrics

- mixed sgRNA types = 2, n = 8,956, observed_positive = 6
- PAM mixed-only: AUROC=0.997756, AUPRC=0.493841
- NoPAM mixed-only: AUROC=0.996397, AUPRC=0.439468
- mixed-only AUPRC 远低于 full pooled (~0.99)，说明 composition confounding 显著

### Phase 8 — 报告

- 技术报告: `results/bl5_generalization/pam_strict_holdout/GAG/report.md`
- 总监简报: `results/bl5_generalization/pam_strict_holdout/GAG/executive_report.md`
- 执行报告: 本文档

---

## 3. 核心结果汇总

### 三 Motif 对比表

| Holdout PAM | PAM family | test_H samples | observed_positive | PAM AUPRC | NoPAM AUPRC | ΔAUPRC (NoPAM−PAM) | CI 跨 0 | Bootstrap n |
|:---|:---|---:|---:|---:|---:|---:|:---:|:---:|
| **AGG** | NGG | 277,247 | 744 | 0.027617 | 0.203836 | +0.176219 | ❌ NO | 500 |
| **TGG** | NGG | 292,861 | 703 | 0.103985 | 0.051045 | −0.052940 | ❌ NO | 500 |
| **GAG** | **non-NGG** | 9,061 | 111 | 0.990692 | 0.989688 | −0.001004 | ✅ YES | **10,000** |

### Per-sgRNA Composition

| category | sgRNA_type 数 | n | observed_positive | unobserved_candidate |
|:---|---:|---:|---:|---:|
| positive_only | 21 | 105 | 105 | 0 |
| unobserved_only | 0 | 0 | 0 | 0 |
| mixed | 2 | 8,956 | 6 | 8,950 |

### Full Test_H 阈值指标

| 模型 | AUROC | AUPRC | Accuracy | Precision | Recall | F1 |
|:---|---:|---:|---:|---:|---:|---:|
| PAM | 0.999850 | 0.990692 | 0.999007 | 0.972222 | 0.945946 | 0.958904 |
| NoPAM | 0.999801 | 0.989688 | 0.999338 | 1.000000 | 0.945946 | 0.972222 |

### Mixed-only 阈值指标

| 模型 | AUROC | AUPRC | Accuracy | Precision | Recall | F1 |
|:---|---:|---:|---:|---:|---:|---:|
| PAM | 0.997756 | 0.493841 | 0.999330 | 0.500000 | 0.500000 | 0.500000 |
| NoPAM | 0.996397 | 0.439468 | 0.999442 | 1.000000 | 0.166667 | 0.285714 |

---

## 4. 科学结论（降级口径）

AGG/TGG 给出两个显著但方向相反的 NGG strict holdout 结果；GAG 作为唯一 feasible non-NGG exploratory subset，PAM 与 NoPAM 的 ΔAUPRC CI 跨 0，未检测到显著差异。

但 GAG test_H 存在明显的 **per-sgRNA label-composition confounding**：
- 21 个 sgRNA_type 是 positive_only（105 pos, 0 neg）
- 全部 8,950 个 unobserved_candidate 集中在 2 个 mixed sgRNA_type 中
- mixed-only 子集 AUPRC 降至 ~0.44-0.49，与 full pooled ~0.99 差距巨大

因此：
- **GAG 只能作为 exploratory evidence**，不能作为确认"第三模式"或 non-NGG 泛化结论
- **GAG 结果不能推广到所有 non-NGG**
- **GAG 结果也不能否定 AGG/TGG 的 motif-specific 发现**
- **GAG 的主用途是提醒**：non-NGG holdout 子集需要 per-sgRNA composition sanity audit

---

## 5. 产出文件清单

```
configs/bl5_v4_pam_holdout_gag.yaml
configs/bl5_v4_nopam_holdout_gag.yaml
run/run_bl5_v4_holdout_gag.sh
scripts/auto_start_nopam_holdout_gag.sh
scripts/build_pam_strict_holdout_split.py  (patched: --skip-size-validation, exact_pair_overlap)
results/bl5_generalization/pam_strict_holdout/GAG/
  ├── split_indices.npz
  ├── split_manifest.json
  ├── split_counts.csv
  ├── pam_distribution.json
  ├── per_sgrna_label_composition.csv
  ├── paired_bootstrap.json
  ├── report.md
  └── executive_report.md
results/bl5_v4_pam_holdout_gag/
  ├── checkpoints/best.pt
  ├── epoch_metrics.csv
  ├── summary.json
  └── test_predictions.csv
results/bl5_v4_nopam_holdout_gag/
  ├── checkpoints/best.pt
  ├── epoch_metrics.csv
  ├── summary.json
  └── test_predictions.csv
reborn_doc/41_BL5_v4_PAM_Holdout_GAG_执行报告.md
```

---

## 6. 合规声明

- ✅ 未 commit / push
- ✅ 未删除/覆盖 data/ / reference/ 原始数据
- ✅ 未改动 AGG/TGG 已完成模型产物
- ✅ 未训练（本次为返修，仅文档/台账/审计修正）
- ✅ 未调用 GPU
- ✅ use_rnafm=true, freeze_rnafm=false 显式声明
- ✅ split_mode=sgrna_safe 显式声明
- ✅ Test 评估使用 best checkpoint
- ✅ AUROC + AUPRC 同时报告
- ✅ Δ 定义为 NoPAM − PAM
- ✅ PAM 坐标使用 off_seq[20:23]
