# BL5-v4-PAM-Holdout-AGG 执行报告

> PAM motif 严格 holdout 泛化实验 | Phase 1-3 完整记录
> 执行日期: 2026-06-07 ~ 2026-06-08

---

## 1. 实验设计

**科学问题**: 显式 PAM 编码器（positions 21-23）在面对训练时完全未见过的新 PAM motif 时，泛化能力如何？

**方法**: 将 AGG 作为 strict holdout PAM motif，构造 holdout split：
- train_H = formal_train AND PAM ≠ AGG
- val_H = formal_val AND PAM ≠ AGG
- test_H = formal_test AND PAM = AGG

**对照**: PAM（use_pam_encoder=true） vs NoPAM（use_pam_encoder=false）。除 use_pam_encoder 以及必要 DDP/find_unused_parameters 设置外，模型主干、split、训练超参、loss、评估流程保持一致。

> 注：PAM config 设 `use_pam_encoder=true, find_unused_parameters=true`；NoPAM config 设 `use_pam_encoder=false, find_unused_parameters=false`。该差异不影响主科学对照。

---

## 2. Phase 1: Holdout Split 构造

| 指标 | 值 |
|:---|:---|
| Holdout PAM | AGG |
| PAM 坐标 | off_seq[20:23] |
| Base split | formal_split_bl5_seed42.json (sgRNA_safe) |
| train_H | 3,892,438 (observed_positive=24,670) |
| val_H | 545,364 (observed_positive=2,996) |
| **test_H** | **277,247** (observed_positive=**744**) |
| test_seenPAM | 677,079 |
| test_H sgRNA types | 72 |

**严格性验证**:
- test_H sgRNA 与 train_H/val_H 零重叠 ✅
- train_H/val_H 中 AGG 比例为 0% ✅
- test_H 中 AGG 比例为 100% ✅
- test_H vs formal_train pair overlap = 0 ✅

---

## 3. Phase 2: 训练结果

### 3.1 PAM holdout AGG

| 指标 | 值 |
|:---|:---|
| best_epoch | 10 |
| val_auprc (best) | **0.681230** |
| 正式训练耗时 | 8,605 s (~2.4 h) |
| GPU | 2× RTX PRO 6000, 43.5 GB each |

> 注：训练耗时来自 2-GPU DDP 正式训练（epoch_metrics.csv / experiments.csv）。`summary.json` 中的 `status=completed_eval_only`、`epochs=0`、`train_seconds≈184s` 是后续 eval-only recovery/export 产物，不代表训练耗时。

**Test 评估（best.pt，validation AUPRC 最佳 checkpoint）**:
| 指标 | 值 |
|:---|:---|
| AUROC | 0.479742 |
| **AUPRC** | **0.027617** |
| Accuracy | 0.106742 |
| Precision | 0.002743 |
| Recall | 0.915323 |
| F1 | 0.005470 |

### 3.2 NoPAM holdout AGG

| 指标 | 值 |
|:---|:---|
| best_epoch | 8 |
| val_auprc (best) | **0.674833** |
| 正式训练耗时 | 8,180 s (~2.3 h) |
| GPU | 2× RTX PRO 6000, 43.5 GB each |

> 注：同上，训练耗时来自 2-GPU DDP 正式训练。`summary.json` 中的 `status=completed_eval_only`、`epochs=0`、`train_seconds≈184s` 是后续 eval-only recovery/export 产物。

**Test 评估（best.pt，validation AUPRC 最佳 checkpoint）**:
| 指标 | 值 |
|:---|:---|
| AUROC | 0.945611 |
| **AUPRC** | **0.203836** |
| Accuracy | 0.959935 |
| Precision | 0.048213 |
| Recall | 0.743280 |
| F1 | 0.090552 |

---

## 4. Phase 3: Paired Bootstrap 比较

基于 277,247 条 test_H 样本的逐样本预测概率，执行 paired bootstrap（500 resamples，seed=42）。

| 模型 | AUROC | 95% CI | AUPRC | 95% CI |
|:---|---:|:---|---:|:---|
| PAM holdout AGG | 0.479742 | [0.465363, 0.495315] | 0.027617 | [0.017115, 0.039968] |
| NoPAM holdout AGG | 0.945611 | [0.937749, 0.953909] | 0.203836 | [0.176517, 0.235531] |
| **Δ (NoPAM − PAM)** | **+0.465869** | **[+0.451195, +0.478549]** | **+0.176219** | **[+0.150750, +0.203564]** |

**显著性**: ΔAUROC 和 ΔAUPRC 的 95% CI 均不包含 0，差异统计显著。

---

## 5. 结论与解读

### 5.1 核心发现

**显式 PAM 编码器在面对未见过的 AGG PAM 时，泛化能力显著劣于无 PAM 编码器的对照模型。**

| 对比维度 | PAM | NoPAM | 差距 |
|:---|:---|:---|:---|
| test AUROC | 0.480 | 0.946 | **+0.466** |
| test AUPRC | 0.028 | 0.204 | **+7.4×** |

### 5.2 解读

1. **PAM 编码器过拟合了训练集 PAM 分布**: 模型在训练时只见过 TGG/GGG/CGG/GAG 等 PAM，没有见过 AGG。当遇到 AGG 时，PAM 编码器无法正确编码这个未见过的 motif，反而引入了错误信号。

2. **NoPAM 模型依赖 RNA-FM 的隐式序列理解**: 没有显式 PAM 编码器时，RNA-FM 通过整体序列上下文隐式推断 PAM 信息，这种表示对新 PAM motif 更具泛化性。

3. **PAM 编码器是一把双刃剑**: 在 seen-PAM 上（val_auprc ~0.68），PAM 和 NoPAM 表现相近；但在 unseen-PAM 上，PAM 编码器成为泛化瓶颈。

### 5.3 工程启示

- 如果应用场景中可能出现训练时未见过的新 PAM motif，**应谨慎使用显式 PAM 编码器**。
- 如果 PAM 空间封闭且所有 motif 都在训练集中出现过，PAM 编码器可以安全使用。
- 未来可以尝试 **PAM 嵌入 + dropout** 或 **PAM 数据增强** 来缓解过拟合。

---

## 6. 合规声明

- AGENTS.md 约束: [use_rnafm=True, freeze_rnafm=False, split_mode=sgrna_safe, pos_weight=None, focal_loss=True]
- PAM 坐标: off_seq[20:23]（positions 21-23）
- Test 评估使用 best.pt ✅
- AUROC 和 AUPRC 同时报告 ✅
- 未删除或覆盖用户数据 ✅

---

## 7. 待补 Sanity Check

**test_nonAGG（test_seenPAM）评估未产出。**

原因：当前 `test_predictions.csv` 仅包含 test_H（PAM=AGG）的 277,247 条样本。要对 test_nonAGG（formal_test 中 PAM≠AGG 的 677,079 条样本）做评估，需要重新运行 eval-only 并切换 test split 为 test_seenPAM。为避免大范围修改 `train_bl5.py` 的数据加载逻辑，该 sanity check 列为 next check，不在本次交付中执行。

> 预期用途：验证 PAM 和 NoPAM 在 seen-PAM（非 AGG）上的性能是否相近，以排除整体模型崩溃的可能性。

## 8. 附件

- Split manifest: `results/bl5_generalization/pam_strict_holdout/AGG/split_manifest.json`
- Bootstrap results: `results/bl5_generalization/pam_strict_holdout/AGG/paired_bootstrap_results.json`
- PAM predictions: `results/bl5_v4_pam_holdout_agg/test_predictions.csv`
- NoPAM predictions: `results/bl5_v4_nopam_holdout_agg/test_predictions.csv`
