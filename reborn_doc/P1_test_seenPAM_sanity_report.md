# P1: test_seenPAM Sanity Evaluation Report

> **Task**: 验证 AGG/TGG/GAG PAM holdout 模型在 seen-PAM formal-test subset 上是否正常，排除"整体模型崩溃"对 test_H 结果的干扰。
> 
> **Date**: 2026-06-09
> 
> **Branch**: `feat/bl5-generalization`
> 
> **Eval-only**: 无重新训练，直接加载各模型 `best.pt` 在 `test_seenPAM` 上推理。

---

## 1. 实验设计

### 1.1 背景

在 PAM strict holdout 实验中，我们观察到：

| PAM | test_H AUPRC (PAM) | test_H AUPRC (NoPAM) | Δ |
|:---:|:---:|:---:|:---:|
| AGG | 0.028 | 0.204 | +0.176 (NoPAM >> PAM) |
| TGG | 0.104 | 0.051 | −0.053 (PAM > NoPAM) |
| GAG | 0.991 | 0.990 | −0.001 (不显著) |

一个关键问题是：**test_H 上的差异是否因为模型整体崩溃？** 即 PAM holdout 训练是否导致模型在所有 PAM 上性能都下降？

### 1.2 Sanity 策略

对每一对 (PAM, NoPAM) 模型，在 **seen-PAM test subset** (`test_seenPAM`) 上执行 eval-only：

- `test_seenPAM` = `test` ∩ `seenPAM_mask`，即 formal test 中 PAM 在训练集中出现过的样本
- 如果 PAM 模型在 seen-PAM 上也明显劣于 NoPAM 模型 → **整体崩溃**，test_H 结果不可信
- 如果 PAM 模型在 seen-PAM 上与 NoPAM 相当或更优 → **无整体崩溃**，test_H 差异是 PAM-specific 的

### 1.3 模型列表

| # | Holdout | Model | Checkpoint |
|:---:|:---:|:---|:---|
| 1 | AGG | PAM | `results/bl5_v4_pam_holdout_agg/checkpoints/best.pt` |
| 2 | AGG | NoPAM | `results/bl5_v4_nopam_holdout_agg/checkpoints/best.pt` |
| 3 | TGG | PAM | `results/bl5_v4_pam_holdout_tgg/checkpoints/best.pt` |
| 4 | TGG | NoPAM | `results/bl5_v4_nopam_holdout_tgg/checkpoints/best.pt` |
| 5 | GAG | PAM | `results/bl5_v4_pam_holdout_gag/checkpoints/best.pt` |
| 6 | GAG | NoPAM | `results/bl5_v4_nopam_holdout_gag/checkpoints/best.pt` |

### 1.4 test_seenPAM 规模

| Holdout | test_seenPAM 样本数 | observed_positive | unobserved_candidate | Pos Rate |
|:---:|:---:|:---:|:---:|:---:|
| AGG | 677,079 | 2,313 | 674,766 | 0.34% |
| TGG | 661,465 | 2,354 | 659,111 | 0.36% |
| GAG | 945,265 | 2,946 | 942,319 | 0.31% |

---

## 2. Pooled Results (test_seenPAM)

| Holdout | Model | AUROC | AUPRC | Pair ΔAUPRC (NoPAM−PAM) |
|:---:|:---:|:---:|:---:|:---:|
| **AGG** | PAM | 0.9836 | 0.5900 | — |
| **AGG** | NoPAM | 0.9869 | 0.5700 | −0.0201 |
| **TGG** | PAM | 0.9877 | 0.5028 | — |
| **TGG** | NoPAM | 0.9849 | 0.5744 | +0.0716 |
| **GAG** | PAM | 0.9826 | 0.4665 | — |
| **GAG** | NoPAM | 0.9847 | 0.4498 | −0.0167 |

> Pair Δ 的定义在全报告中统一为 **NoPAM−PAM**，并显示在每个 holdout 的 NoPAM 行。

原始数据见：`results/bl5_generalization/pam_strict_holdout_seenpam_sanity/seenpam_pooled_metrics.csv`

配对差异见：`results/bl5_generalization/pam_strict_holdout_seenpam_sanity/seenpam_pair_deltas.csv`

Bootstrap CI 见：`results/bl5_generalization/pam_strict_holdout_seenpam_sanity/paired_bootstrap_seenpam.json`

### 2.1 解读（含 paired bootstrap CI, n=10,000, seed=42）

**AGG**: 按 Δ=NoPAM−PAM，seen-PAM AUPRC Δ=−0.020 (95% CI [−0.029, −0.012]，不跨 0)，表示 PAM 模型 AUPRC 高于 NoPAM；AUROC Δ=+0.003 (95% CI [0.002, 0.005]，不跨 0)，表示 NoPAM AUROC 略高于 PAM。
→ 未观察到 catastrophic collapse。test_H 上 NoPAM >> PAM 的差异是 **AGG-specific** 的，即模型确实失去了对 unseen AGG PAM 的泛化能力。

**TGG**: 按 Δ=NoPAM−PAM，seen-PAM AUPRC Δ=+0.072 (95% CI [0.063, 0.080]，不跨 0)，表示 NoPAM AUPRC 高于 PAM；AUROC Δ=−0.003 (95% CI [−0.004, −0.002]，不跨 0)，表示 PAM AUROC 略高于 NoPAM。
→ 未观察到 catastrophic collapse（PAM 模型 AUROC≈0.988），但存在显著的 seen-PAM AUPRC drop。test_H 上 PAM>NoPAM 不能简单解释为稳定 cross-PAM benefit，需要作为 subset-dependent 现象谨慎解读。

**GAG**: 按 Δ=NoPAM−PAM，seen-PAM AUPRC Δ=−0.017 (95% CI [−0.022, −0.011]，不跨 0)，表示 PAM 模型 AUPRC 高于 NoPAM；AUROC Δ=+0.002 (95% CI [0.001, 0.003]，不跨 0)，表示 NoPAM AUROC 略高于 PAM。
→ 未观察到 catastrophic collapse。与 test_H 结论一致（GAG PAM≈NoPAM）。

---

## 3. Stratified by PAM Motif (粗分层)

### 3.1 TGG PAM vs NoPAM: NGG vs non-NGG

| Model | Subset | n | observed_positive | unobserved_candidate | pos_rate | AUROC | AUPRC |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| TGG PAM | NGG | 527,128 | 1,651 | 525,477 | 0.31% | 0.9818 | 0.3173 |
| TGG NoPAM | NGG | 527,128 | 1,651 | 525,477 | 0.31% | 0.9758 | 0.3615 |
| TGG PAM | non-NGG | 134,337 | 703 | 133,634 | 0.52% | 0.9984 | 0.8895 |
| TGG NoPAM | non-NGG | 134,337 | 703 | 133,634 | 0.52% | 0.9994 | 0.9492 |

原始数据见：`results/bl5_generalization/pam_strict_holdout_seenpam_sanity/seenpam_stratified_metrics.csv`

**发现**：
- **non-NGG**: NoPAM observed AUPRC higher than PAM (0.949 vs 0.890)
- **NGG**: NoPAM observed AUPRC higher than PAM (0.362 vs 0.317)
- TGG PAM holdout 在 NGG 和 non-NGG 两个粗分层上均未观察到 seen-PAM AUPRC 优势

### 3.2 AGG PAM Model on seenPAM (按 PAM motif 细分，参考性)

| PAM | n | observed_positive | unobserved_candidate | pos_rate | AUROC | AUPRC |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| GAG | 9,061 | 111 | 8,950 | 1.23% | 1.0000 | 0.9975 |
| GCG | 1,757 | 17 | 1,740 | 0.97% | 1.0000 | 0.9967 |
| GTG | 6,606 | 46 | 6,560 | 0.70% | 0.9999 | 0.9902 |
| CAG | 9,863 | 88 | 9,775 | 0.89% | 0.9999 | 0.9897 |
| CCG | 1,833 | 9 | 1,824 | 0.49% | 0.9993 | 0.9389 |
| CTG | 8,908 | 19 | 8,889 | 0.21% | 0.9991 | 0.9224 |
| ATG | 18,612 | 56 | 18,556 | 0.30% | 0.9993 | 0.8963 |
| AAG | 23,114 | 74 | 23,040 | 0.32% | 0.9988 | 0.8605 |
| ACG | 3,777 | 12 | 3,765 | 0.32% | 0.9987 | 0.8438 |
| TAG | 20,155 | 55 | 20,100 | 0.27% | 0.9979 | 0.7855 |
| TTG | 25,379 | 43 | 25,336 | 0.17% | 0.9987 | 0.7776 |
| **GGG** | 203,284 | 716 | 202,568 | 0.35% | 0.9789 | 0.4070 |
| TCG | 5,106 | 7 | 5,099 | 0.14% | 0.9948 | 0.3923 |
| **TGG** | 292,861 | 703 | 292,158 | 0.24% | 0.9713 | 0.3530 |
| **CGG** | 46,592 | 186 | 46,406 | 0.40% | 0.9542 | 0.2992 |

> ⚠️ 上述 per-motif 表格为参考性，未做多重检验校正。报告中对 motif 层面的比较仅限于 NGG vs non-NGG 两个粗分层。

---

## 4. 与 test_H 结果对比

| Holdout | test_H 结论 | seenPAM 结论 | 一致性 |
|:---:|:---|:---|:---:|
| **AGG** | NoPAM >> PAM (+0.176) | PAM observed AUPRC higher (Δ=−0.020, NoPAM−PAM) | ✅ 一致：holdout 仅损害 unseen-PAM |
| **TGG** | PAM > NoPAM (−0.053) | NoPAM observed AUPRC higher (Δ=+0.072, NoPAM−PAM) | ⚠️ 方向相反：test_H 优势不能简单解释为稳定 cross-PAM benefit |
| **GAG** | PAM ≈ NoPAM (−0.001) | PAM observed AUPRC higher (Δ=−0.017, NoPAM−PAM) | ✅ 一致 |

---

## 5. 结论

1. **未观察到 catastrophic collapse**：6 个 seenPAM eval 的 AUROC 均约 0.98+，AUPRC 保持在 0.45–0.59。但 TGG PAM 相对 NoPAM 存在 observed seenPAM AUPRC drop。

2. **AGG holdout 是干净的**：seen-PAM 上 PAM≈NoPAM，test_H 上 NoPAM>>PAM。说明 AGG 确实是一个有意义的 unseen-PAM 泛化失败案例。

3. **TGG holdout 方向相反**：TGG seenPAM 上 PAM 模型 observed AUPRC lower than NoPAM，而 test_H 上方向相反；这说明 TGG test_H 的 PAM>NoPAM 不能简单解释为稳定 cross-PAM benefit，需要作为 subset-dependent 现象谨慎解读。

4. **GAG holdout 是中性**：seen-PAM 和 test_H 都显示 PAM≈NoPAM，与 GAG 本身频率低、test_H 样本小的观察一致。

5. **建议**：
   - AGG 结果可信：PAM holdout 确实损害了 unseen-PAM 泛化
   - TGG 结果需要谨慎解读：test_H 上 PAM>NoPAM 不等于"TGG 更容易泛化"
   - GAG 结果作为探索性参考即可

---

## 6. 产物清单

| 文件 | 说明 |
|:---|:---|
| `results/bl5_generalization/pam_strict_holdout_seenpam_sanity/seenpam_pooled_metrics.csv` | 6 模型 pooled seenPAM 指标 |
| `results/bl5_generalization/pam_strict_holdout_seenpam_sanity/seenpam_pair_deltas.csv` | AGG/TGG/GAG 三对 ΔAUROC/ΔAUPRC (NoPAM−PAM) |
| `results/bl5_generalization/pam_strict_holdout_seenpam_sanity/seenpam_stratified_metrics.csv` | TGG NGG / non-NGG 粗分层指标 |
| `results/bl5_generalization/pam_strict_holdout_seenpam_sanity/paired_bootstrap_seenpam.json` | 配对 bootstrap CI (n_bootstrap=10,000, seed=42) |
| `{model_dir}/test_seenPAM_predictions.csv` | 6 模型 seenPAM predictions（保留） |

---

## 7. 技术细节

- **Eval 命令**: `python scripts/train_bl5.py --config ... --eval-only --eval-split-key test_seenPAM`
- **Checkpoint**: 均加载 `best.pt`（validation AUPRC 最佳）
- **Guardrails**: 全部通过 `check_model_config` + `check_eval_procedure`
- **Predictions**: 已保存为 `{output_dir}/test_seenPAM_predictions.csv`
- **运行时间**: AGG ~7min, TGG ~7min, GAG ~11min（GPU 100% 利用）
