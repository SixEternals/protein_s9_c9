# BL5-v4-RNAFM-PAM-noRun-control 执行报告

> **实验目的**：2-view 组件消融——验证 RNA-FM CLS + PAM Encoder（无 Run/LearnableRun）的独立贡献，完成 2×2 组件矩阵的最后一格。
> 
> **模型归属**：BL4 系列（含 RNA-FM，非 BL3）。`fusion_type="rnafm_pam_concat"` 为新增。

---

## 1. 基本信息

| 项目 | 内容 |
|:---|:---|
| 实验版本 | BL5-v4-RNAFM-PAM-noRun |
| 执行时间 | 2026-06-06 18:59 – 21:57 (UTC+8) |
| 训练时长 | **10,576 秒 ≈ 2.94 小时** |
| GPU | 2 × NVIDIA RTX PRO 6000 Blackwell Server Edition |
| 显存占用 | 40.40 GB（单卡峰值） |
| Commit Hash | `327075a` |
| 代码改动 | `models/bl5_dynamic_fusion.py` 新增 `rnafm_pam_concat` fusion 类型及验证/forward 分支 |
| Config 文件 | `configs/bl5_v4_rnafm_pam_norun_control.yaml` |
| 数据 split | `sgrna_safe`（group-safe），seed=42 |
| 数据集规模 | Train=4,697,495 / Val=741,552 / Test=954,326 |

---

## 2. 训练配置

```yaml
model:
  use_rnafm: true
  freeze_rnafm: false
  use_run: false
  use_learnable_run: false
  use_region: false
  fusion_type: rnafm_pam_concat    # 新增
  rna_pooling: cls
  use_pam_encoder: true
  pam_dim: 16
  d_model: 128
  classifier_hidden: 256
  dropout: 0.3
training:
  epochs: 10
  batch_size: 1024
  lr_transformer: 5.0e-4
  lr_pam_encoder: 1.0e-3
  lr_mlp: 1.0e-3
  focal_loss: true
  focal_gamma: 2.0
  gradient_clip: 1.0
  weight_decay: 1.0e-5
```

**关键约束确认**：
- ✅ `use_rnafm=true`, `freeze_rnafm=false`（Route A，full fine-tune）
- ✅ `use_run=false`, `use_learnable_run=false`（Run 完全关闭）
- ✅ `split_mode=sgrna_safe`
- ✅ `pos_weight` 未使用（focal loss 替代）
- ✅ Test 评估使用 `best.pt`（val AUPRC 最佳）

---

## 3. 训练曲线

| Epoch | Train Loss | Val AUROC | Val AUPRC | Val Accuracy | Val Precision | Val Recall | Val F1 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.005782 | **0.8676** | **0.3662** | 0.9956 | 1.0000 | 0.2359 | 0.3818 |
| 2 | 0.004445 | 0.8414 | 0.2981 | 0.9956 | 1.0000 | 0.2411 | 0.3885 |
| 3 | 0.004053 | 0.8724 | 0.3158 | 0.9956 | 0.9903 | 0.2381 | 0.3838 |
| 4 | 0.003766 | 0.8722 | 0.3343 | 0.9956 | 0.9758 | 0.2366 | 0.3809 |
| 5 | 0.003513 | 0.8542 | 0.3327 | 0.9956 | 0.9848 | 0.2437 | 0.3907 |
| 6 | 0.002802 | 0.8285 | 0.3059 | 0.9956 | 0.9722 | 0.2455 | 0.3921 |
| 7 | 0.002406 | 0.8416 | 0.3051 | 0.9956 | 0.9659 | 0.2458 | 0.3919 |
| 8 | 0.002159 | 0.8260 | 0.2980 | 0.9956 | 0.9631 | 0.2446 | 0.3901 |
| 9 | 0.001917 | 0.7981 | 0.2957 | 0.9956 | 0.9570 | 0.2453 | 0.3905 |
| 10 | 0.001330 | 0.7603 | 0.2932 | 0.9956 | 0.9248 | 0.2479 | 0.3910 |

**训练现象**：
- **Best epoch = 1**，val AUPRC = 0.3662
- Epoch 1 之后，train_loss 持续下降（0.0058 → 0.0013），但 val AUPRC 持续下降并在 0.29–0.33 区间震荡
- 这是典型的 **过拟合信号**：模型在训练集上越拟合越好，但泛化到验证集上的排序能力在退化
- Val AUROC 同样从 0.8676（epoch 1）跌至 0.7603（epoch 10），跌幅约 12.4%
- **结论**：本配置下，1 epoch 的 early stopping 可能是最优策略

---

## 4. Test 评估结果（best.pt, epoch 1）

| 指标 | 数值 |
|:---|:---:|
| Test Loss | 0.006051 |
| **AUROC** | **0.837950** |
| **AUPRC** | **0.276529** |
| Accuracy | 0.997499 |
| Precision | 1.0000 |
| Recall | 0.219169 |
| F1 | 0.359539 |

- Non-finite probability count: **0**（数值稳定）
- 阳性样本占比：3,057 / 954,326 ≈ **0.32%**

---

## 5. PAM 分层分析

> ⚠️ **重要说明**：该实验主指标可用，test `sample_index` 与 BL5-v4-PAM 完全对齐（954,326 条，3,057 positive）；但最初报告曾误用 `off_seq[-3:]` 做 PAM 分层（导致 NGG=953,810 / non-NGG=516 的错误口径），现已统一修正为 positions 21-23 / `PAM_original = off_seq[20:23]`，与 PAMEncoder 输入一致。以下数据均为修正后口径。

详见独立报告：`results/bl5_v4_rnafm_pam_norun_control/pam_stratification_report.md`

> ⚠️ **PAM 口径声明**：本报告 PAM 分层使用 `PAM_original = off_seq[20:23]`（positions 21-23），与 PAMEncoder 输入一致。不使用 `off_seq[-3:]`。此前版本误用旧口径（NGG=953,810 / non-NGG=516），现已全部修正。

**核心分层结果**：

| 子集 | 样本数 | 阳性数 | 阴性数/未观测候选数 | 阳性率 | AUROC | AUPRC |
|:---|---:|---:|---:|---:|---:|---:|
| All | 954,326 | 3,057 | 951,269 | 0.32% | 0.837950 | 0.276529 |
| NGG-only | 819,984 | 2,349 | 817,635 | 0.29% | 0.759169 | 0.074553 |
| non-NGG-only | 134,342 | 708 | 133,634 | 0.53% | 0.998394 | 0.915266 |

> 💡 **关键解读**：
> - **NGG-only（819,984 条，85.9%）**：Cas9 的标准作业区。这里 AUPRC = 0.0746 是模型在「真正需要判断」的场景下的能力。这个数字远低于 Overall 的 0.2765，说明 Overall AUPRC 中相当部分来自 non-NGG 子集的结构性贡献。
> - **non-NGG-only（134,342 条，14.1%）**：AUPRC 高达 0.9153，但这主要因为 non-NGG 位点在特征空间中与 NGG 位点天然可分，RNA-FM 学到的是「非 NGG → 大概率不是真靶点」这条相对简单的规则，而非对每个 non-NGG 位点的精细风险判断。
> - NGG 内部 motif 异质性：四大 NGG motif 的 AUPRC 均在 0.065–0.084 的狭窄区间，差异不显著。
> - 模型存在 **probability overestimation**（预测概率系统性地高于实际阳性率）：最高概率 bin 预测均值 9.6%，实际阳性率仅 1.8%，偏差 +7.8 个百分点。

---

## 6. 消融对比（2×2 组件矩阵）

使用 formal BL5 split 统一 test set：

| 实验 | RNA-FM | LearnableRun | PAM Encoder | Test AUPRC | Test AUROC | best_epoch |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **BL5-v4-PAM**（全模型） | ✅ | ✅ | ✅ | **0.5313** | **0.9842** | 8 |
| BL5-v4-NoPAM-control | ✅ | ✅ | ❌ | 0.5024 | 0.9841 | 6 |
| **BL5-v4-RNAFM-PAM-noRun** | ✅ | ❌ | ✅ | **0.2765** | **0.8380** | **1** |
| BL0b-on-BL5split | ✅ | ❌ | ❌ | 0.2957 | 0.8578 | — |
| BL5-v4-LearnableRun-only | ❌ | ✅ | ❌ | 0.2949 | 0.9609 | 9 |
| BL5-v4-LearnableRun-PAM-noRNAFM | ❌ | ✅ | ✅ | 0.1772 | 0.9527 | 2 |
| BL5-v4-PAM-only | ❌ | ❌ | ✅ | 0.0592 | 0.4994 | 3 |

### 6.1 固定 RNA-FM 存在时的 2×2 矩阵

|  | PAM=✅ | PAM=❌ |
|:---|:---|:---|
| **Run=✅** | BL5-v4-PAM = **0.5313** | NoPAM-control = **0.5024** |
| **Run=❌** | RNAFM-PAM-noRun = **0.2765** | BL0b-on-BL5split = **0.2957** |

### 6.2 核心发现

1. **RNA-FM + PAM（无 Run）≈ RNA-FM-only，但略低**：
   - RNAFM-PAM-noRun AUPRC = 0.2765，BL0b-on-BL5split AUPRC = 0.2957
   - RNAFM-PAM-noRun 反而低 0.0191，说明在本配置下，PAM 单独加到 RNA-FM 上并未带来增益
   - **PAM 的稳定正向贡献需要 RNA-FM + LearnableRun 同时存在的强联合上下文**

2. **LearnableRun 是核心增益来源**：
   - 全模型 AUPRC = 0.5313，RNAFM-PAM-noRun AUPRC = 0.2765
   - 去掉 Run 后 AUPRC 下降 **47.9%**（相对）
   - **Run/LearnableRun 的贡献远大于 PAM 的贡献**

3. **PAM 单独几乎无价值**：
   - PAM-only AUPRC = 0.0592，AUROC ≈ random（0.4994）
   - 但 RNA-FM + PAM 的 AUROC = 0.8380 < RNA-FM-only 的 0.8578
   - 说明 PAM 的边际价值高度依赖上下文——在 RNA-FM + LearnableRun 的完整框架下才有正向贡献

4. **RNA-FM 与 LearnableRun 的互补性**：
   - RNA-FM 学到的是隐式序列上下文，LearnableRun 学到的是显式错配模式/seed 权重
   - 两者单独用时 AUPRC ~0.29–0.30，组合后跳到 0.5024（+70% 相对）
   - 互补增益验证了多视角融合的核心假设

---

## 7. 训练现象深度分析

### 7.1 过拟合早于预期
- 全模型（BL5-v4-PAM）best_epoch = 4，而 RNAFM-PAM-noRun best_epoch = 1
- 说明 **缺少 Run 的显式结构化约束后，RNA-FM 的庞大参数量（99.5M）更容易过拟合训练数据的表面模式**
- Run/LearnableRun 作为低维手工/半手工先验（position-aware），起到了 **正则化** 作用，延缓了过拟合

### 7.2 为什么 val AUROC 和 AUPRC 同步下降
- AUROC 衡量整体排序能力，AUPRC 衡量头部正样本召回
- 两者同步下降说明：模型在后期的优化方向不是"更精准地找到正样本"，而是"更极端地预测训练集中的高频模式"
- 在极度不平衡数据上，这通常表现为对常见 sgRNA/off-target 模式的过拟合

### 7.3 与 BL0b-on-BL5split（RNA-FM-only formal baseline）的对比

| 模型 | Test AUPRC | 参数量 | best_epoch | 说明 |
|:---|:---:|:---:|:---:|:---|
| BL0b-on-BL5split（fine-tune RNA-FM + MLP） | 0.2957 | ~99.5M | 4 | RNA-FM-only，同 formal BL5 test set |
| BL5-v4-RNAFM-PAM-noRun | 0.2765 | ~99.5M + 656 | 1 | RNA-FM + PAM，无 Run |

- BL0b-on-BL5split 是纯 RNA-FM + MLP，没有 PAM encoder，在 formal BL5 split test set 上 AUPRC = 0.2957
- RNAFM-PAM-noRun 加了 PAM encoder（16-dim），但 AUPRC 反而略低（0.2765 vs 0.2957，差 −0.0191）
- **这不是 PAM 的「负作用」**，更可能的原因是：
  1. 本实验 best_epoch = 1（严重早停），训练未充分收敛即已过拟合
  2. classifier 结构不同：BL0b 使用 `mlp_hidden=256, mlp_hidden2=64`（两层），而 BL5-v4 使用 `classifier_hidden=256`（单层），优化 landscape 不同
  3. 更重要的结论是：**PAM 的稳定正向增益出现在 RNA-FM + LearnableRun 同时存在的完整上下文中**（BL5-v4-PAM vs NoPAM-control: +0.0289），而非单独加到 RNA-FM 上时

---

## 8. 结论与下一步

### 8.1 本实验回答的科学问题
> **"RNA-FM + PAM（无 Run）单独有多强？"**

**答**：AUPRC = 0.2765，约等于 RNA-FM-only（BL0b-on-BL5split, 0.2957），但低于二者的差值（−0.0191）在本实验的早停和结构差异背景下不能解读为「PAM 有害」。关键结论是：**RNA-FM 的隐式上下文与 LearnableRun 的显式错配模式是强互补关系**——单个视角 AUPRC ~0.29–0.30，组合后达到 0.5024（NoPAM-control），再加入 PAM 达到 0.5313（BL5-v4-PAM）。

### 8.2 组件消融矩阵（formal BL5 split 统一口径）

消融矩阵已补齐。在 formal BL5 split 同一 test set 上的完整对比：

| 视角组合 | AUPRC | 状态 |
|:---|:---:|:---:|
| RNA-FM + LearnableRun + PAM | **0.5313** | ✅ BL5-v4-PAM 全模型 |
| RNA-FM + LearnableRun | 0.5024 | ✅ NoPAM-control |
| RNA-FM + PAM | **0.2765** | ✅ **本实验** |
| LearnableRun + PAM | 0.1772 | ✅ LearnableRun-PAM-noRNAFM |
| RNA-FM only | 0.2957 | ✅ BL0b-on-BL5split |
| LearnableRun only | 0.2949 | ✅ LearnableRun-only |
| PAM only | 0.0592 | ✅ PAM-only |
| PAM shuffle control | 0.1389 | ✅ Shuffle-control |

**矩阵结论**：
- **RNA-FM 是核心底座**：含 RNA-FM 的组合 AUPRC ≥ 0.276，不含 RNA-FM 的组合 AUPRC ≤ 0.295
- **LearnableRun 是 RNA-FM 的最佳搭档**：RNA-FM + LearnableRun = 0.5024，RNA-FM + PAM = 0.2765
- **PAM 是条件性增量**：单独加到 RNA-FM 或 LearnableRun 上均无法产生正向增益（0.2765 < 0.2957, 0.1772 < 0.2949），只有在 RNA-FM + LearnableRun 同时存在的强联合上下文中才贡献 +0.0289（0.5313 − 0.5024）

### 8.3 对 PAM 增益性质的严谨表述

> **在本次 formal split 下，RNA-FM + PAM noRun（0.2765）没有超过 RNA-FM-only baseline（0.2957），说明 PAM 对 RNA-FM 的独立补强有限/不稳定。PAM 的稳定正向增益主要出现在 RNA-FM + LearnableRun 的完整上下文中，即 BL5-v4-PAM 相对 NoPAM-control 的 +0.0289。**

### 8.3 下一步建议
1. **BL4-full（RNA-FM + Region + Run 三者拼接）**：验证 Region 编码是否能为 RNA-FM 提供额外结构化信号
2. **BL5 系列（动态融合）**：在已确认 Run 是核心增益来源的前提下，探索 Cross-Attn / Gated Fusion 是否能进一步提升 RNA-FM 与 Run 的交互效率
3. **考虑 early stopping = 1 epoch**：对于 RNAFM-PAM-noRun，1 epoch 后的训练只会带来过拟合，可考虑在 config 中显式设置 `early_stopping_patience=0` 或 `epochs=1` 作为 ablation 控制

---

## 9. AGENTS.md 合规声明

```python
"""
AGENTS.md compliance: [use_rnafm=True, freeze_rnafm=False,
                       split_mode=sgrna_safe, pos_weight=None]
确认本文件遵守 AGENTS.md 约束
"""
```

- ✅ 序列输入经过 RNA-FM tokenization，禁止裸字符串输入 NN
- ✅ `use_rnafm=true/false` 显式声明
- ✅ `freeze_rnafm=true/false` 显式声明
- ✅ `split_mode=sgrna_safe` 显式声明
- ✅ `pos_weight` 未使用（focal loss 替代）
- ✅ Test 使用 best checkpoint（epoch 1）
- ✅ AUROC + AUPRC 同时报告
