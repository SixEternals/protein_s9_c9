# 21. 公平基线重置与 PAM 贡献拆解报告

> 生成时间：2026-05-29
> 作者：Kimi Code CLI
> 关联分支：`fair-bl0-bl5split`
> 关联 commit：`cd70c5b`

---

## 1. 背景：为什么要做公平基线重置

此前 BL5-v4-PAM（AUROC=0.984, AUPRC=0.531）与历史 BL0b（AUPRC=0.522）的对比存在 **split 不一致** 的问题：

- BL5-v4-PAM 使用 `seed=42, sgrna_safe` 分组逻辑
- 历史 BL0b 使用另一套 `sgRNA_type_group` 划分
- 两者的 test set 包含的 sgRNA_type 数量、样本数、positive ratio 均不同

**在不同 split 上比较模型性能，结论不可信。** 因此必须进行公平基线重置：
1. 固定同一套 formal split
2. 重跑 BL0b
3. 跑 NoPAM control
4. 在完全相同的 test set 上拆解 PAM Encoder 的净贡献

---

## 2. 实验设计：三个模型的对比关系

```
BL0b-on-BL5split (纯 RNA-FM)
        │
        ▼  + LearnableRunEncoder + simple_concat
BL5-v4-NoPAM-control (RNA-FM + Run)
        │
        ▼  + PAM Encoder (positions 21-23)
    BL5-v4-PAM (完整版)
```

| 模型 | RNA-FM | LearnableRun | PAM Encoder | 目的 |
|:---|:---:|:---:|:---:|:---|
| BL0b-on-BL5split | ✅ | ❌ | ❌ | 纯 RNA-FM 基线 |
| BL5-v4-NoPAM-control | ✅ | ✅ | ❌ | 拆解 Run 先验的贡献 |
| BL5-v4-PAM | ✅ | ✅ | ✅ | 验证 PAM 的额外价值 |

**关键控制变量**：
- 同一数据集：CCLMoff 6,393,373 条
- 同一 split：`formal_split_bl5_seed42.json`
- 同一 seed：42
- 同一 batch size：1024 per GPU
- 同一 loss：focal_loss gamma=2.0
- 同一 classifier 深度：NoPAM 768→256→64→1，PAM 784→256→64→1

---

## 3. Formal Split 验证

基于 `formal_split_bl5_seed42.json`，确认三个模型使用完全相同的 test set：

| 指标 | 数值 |
|:---|---:|
| test_samples | 954,326 |
| test_positive | 3,057 |
| test_negative | 951,269 |
| test_sgRNA_type_count | 72 |

BL5-v4-PAM 的历史 run 使用 `seed=42, split_mode=sgrna_safe`，其 `make_split` 逻辑与 `export_formal_split.py` 完全一致。因此可确认 BL5-v4-PAM 的 test set 与上述数字完全重合。

---

## 4. 当前结果

### 4.1 已完成

**BL0b-on-BL5split**
- 模型：RNA-FM CLS + official MLP 640→64→1
- freeze_rnafm：false
- best_epoch：8 / 10
- Test AUROC：**0.8578**
- Test AUPRC：**0.2957**
- Precision：78.3% | Recall：25.2% | F1：38.1%

> 注：AUPRC（0.296）显著低于历史 BL0b（0.522），原因是 formal split 的 test set 包含 72 个 sgRNA_type（vs 历史 29 个），positive ratio 更低（0.32%）。这是公平对比的必要代价。

**BL5-v4-PAM（历史结果）**
- Test AUROC：**0.9842**
- Test AUPRC：**0.5313**
- best_epoch：9 / 10

### 4.2 进行中

**BL5-v4-NoPAM-control**
- 状态：tmux 中训练，已跑至 epoch 3
- epoch 3 val AUPRC：0.632（接近 BL5-v4-PAM 的 best_val 0.638）
- 预计完成：凌晨 03:00-03:30

---

## 5. 初步结论

基于 formal_split_bl5_seed42.json，我们重新评估了纯 RNA-FM 基线 BL0b，并确认其 test set 与 BL5-v4-PAM 完全一致，均包含 954,326 条样本、3,057 条 positive、951,269 条 negative 和 72 个 test sgRNA_type。在该严格一致的 formal split 下，BL0b-on-BL5split 获得 AUROC=0.8578、AUPRC=0.2957，而 BL5-v4-PAM 获得 AUROC=0.984194、AUPRC=0.531281。结果表明，BL5-v4-PAM 整体框架在相同 test set 上显著优于纯 RNA-FM baseline。后续将通过 BL5-v4-NoPAM-control 进一步拆解 PAM Encoder 的净贡献。

**关键发现（待 NoPAM 完成后最终确认）**：
- BL0b → NoPAM 的 AUPRC 提升：反映 **LearnableRunEncoder + 更大 classifier** 的价值
- NoPAM → PAM 的 AUPRC 提升：反映 **PAM Encoder (positions 21-23)** 的净贡献
- 从 epoch 3 的 val AUPRC（0.632）看，NoPAM 已接近 PAM（0.638），**PAM 的额外贡献可能很小**

---

## 6. 关键文件清单

| 文件 | 用途 |
|:---|:---|
| `formal_split_bl5_seed42.json` | 统一 split，三个模型共用 |
| `configs/bl0b_on_bl5split.yaml` | BL0b 配置 |
| `configs/bl5_v4_pam.yaml` | PAM 配置 |
| `configs/bl5_v4_nopam_control.yaml` | NoPAM 配置 |
| `results/bl0b_on_bl5split/summary.json` | BL0b 结果 |
| `results/bl0b_on_bl5split/test_predictions.csv` | BL0b test 预测 |
| `results/bl5_v4_pam/summary.json` | PAM 结果 |
| `results/bl5_v4_nopam_control/summary.json` | NoPAM 结果（待生成） |

---

## 5. 已发现的风险（必须披露）

### 5.1 PAM Shortcut 信号

对 formal split 的 train/val/test 分别统计 PAM 分布：

| Split | NGG 样本数 | non-NGG 样本数 | non-NGG positive | non-NGG negative | non-NGG positive_ratio |
|:---|---:|---:|---:|---:|---:|
| Train | 4,690,641 | 6,854 | 6,854 | **0** | **100%** |
| Val | 740,806 | 746 | 746 | **0** | **100%** |
| Test | 953,810 | 516 | 516 | **0** | **100%** |

**结论**：数据集中所有 non-NGG PAM 样本均为 positive。模型可能学到 shortcut：
> "如果 PAM 不是 NGG，就更可能是 positive。"

这**必须**在论文中明确披露，也是 PAM shuffle control 实验（任务 4.3）的核心动机。

### 5.2 Leakage 检查通过

- **Exact duplicate**：train-test 零重复（key1/key2/key3 均通过）
- **sgRNA_type overlap**：train/val/test 无 sgRNA_type 交集
- **Split 一致性**：BL0b、BL5-v4-PAM 使用完全相同的 test set（954,326 / 3,057 / 951,269 / 72）

---

## 6. 已准备好的脚本和配置（全部就绪，等 NoPAM 完成后一键执行）

| 任务 | 脚本/配置 | 目的 | 状态 |
|:---|:---|:---|:---:|
| **4.2 分层评估** | `scripts/eval_stratified_by_pam.py` | All / NGG-only / non-NGG-only 三模型对比 | ⏳ 待跑 |
| **4.3 PAM shuffle control** | `configs/bl5_v4_pam_shuffle_control.yaml` + `train_bl5.py --shuffle-pam` | 验证 PAM 信息是否被真实利用 | ⏳ 待跑 |
| **5.1 per-sgRNA AUPRC** | `scripts/per_sgrna_and_pam_analysis.py` | 按 sgRNA_type 拆解 AUPRC 差异 | ⏳ 待跑 |
| **5.2 per-PAM metrics** | `scripts/per_sgrna_and_pam_analysis.py` | 按 PAM 类型拆解概率和 AUPRC | ⏳ 待跑 |
| **5.3 Paired comparison** | `scripts/paired_comparison.py` | 同一样本在三模型下的 probability 差异 | ⏳ 待跑 |
| **6 kNN baseline** | `scripts/knn_baseline.py` | 验证模型是否只是"查相似题" | ⏳ 待跑 |
| **3.3 最近邻相似性** | `scripts/train_test_nearest_neighbor.py` | test 到 train 的最小 Hamming 距离分布 | 🔄 后台运行 |
| **9 in-silico perturbation** | `scripts/in_silico_perturbation.py` | PAM / seed mismatch / consecutive mismatch 扰动 | ⏳ 待跑 |
| **最终报告** | `results/anti_memorization_validation_report.md` | 汇总所有证据链 | ⏳ 待生成 |

---

## 7. 执行顺序（NoPAM 完成后）

```text
1. 等待 NoPAM 训练完成（tmux nopam，预计 03:30）
2. 自动执行 post-NoPAM 分析流水线（nohup PID 330256）：
   ├── 分层评估（All / NGG-only / non-NGG-only）
   ├── Paired comparison
   ├── per-sgRNA / per-PAM 分析
   ├── kNN baseline
   └── 贡献拆解报告
3. 手动执行 PAM shuffle control 训练
4. 手动执行 in-silico perturbation
5. 生成 anti_memorization_validation_report.md 总报告
```

---

## 8. 关键文件清单

### 已完成
| 文件 | 用途 |
|:---|:---|
| `formal_split_bl5_seed42.json` | 统一 split，三个模型共用 |
| `results/formal_split_bl5_seed42_audit.json` / `.md` | Split 核验报告 |
| `results/fair_split_comparison_audit.json` / `.md` | 三模型 split 一致性确认 |
| `results/leakage_exact_duplicate_audit.json` / `.md` | Exact duplicate 检查 |
| `results/leakage_sgrna_type_audit.json` / `.md` | sgRNA_type 泄漏检查 |
| `results/pam_distribution_by_split.csv` / `.md` | PAM 分布统计（含 non-NGG 100% positive 警报） |
| `configs/bl0b_on_bl5split.yaml` | BL0b 配置 |
| `configs/bl5_v4_pam.yaml` | PAM 配置 |
| `results/bl0b_on_bl5split/summary.json` | BL0b 结果 |
| `results/bl0b_on_bl5split/test_predictions.csv` | BL0b test 预测 |
| `results/bl5_v4_pam/summary.json` | PAM 结果 |

### 待生成
| 文件 | 用途 |
|:---|:---|
| `results/bl5_v4_nopam_control/summary.json` | NoPAM 结果 |
| `results/bl5_v4_nopam_control/test_predictions.csv` | NoPAM test 预测 |
| `results/stratified_metrics_all_ngg_nongg.csv` / `.md` | 分层评估 |
| `results/paired_comparison_test_predictions.csv` / `.md` | 配对对比 |
| `results/per_sgrna_metrics.csv` / `.md` | per-sgRNA 分析 |
| `results/per_pam_metrics.csv` / `.md` | per-PAM 分析 |
| `results/knn_baseline_summary.json` / `.md` | kNN baseline |
| `results/bl5_v4_contribution_decomposition.json` / `.md` | 贡献拆解 |
| `results/in_silico_perturbation_examples.csv` / `.md` | 扰动实验 |
| `results/anti_memorization_validation_report.md` | 总报告 |
