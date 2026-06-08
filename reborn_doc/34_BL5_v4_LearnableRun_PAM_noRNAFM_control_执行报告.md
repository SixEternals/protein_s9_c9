# BL5-v4-LearnableRun-PAM-noRNAFM-control 执行报告

> **实验目的**：2-view 组件消融——验证 LearnableRunEncoder + PAM Encoder（无 RNA-FM）的独立贡献，补全 no-RNAFM 条件下的 2×2 组件矩阵。
> 
> **模型归属**：BL3 系列（无 RNA-FM，先验特征主导）。`fusion_type="run_pam_concat"` 为新增。

---

## 1. 基本信息

| 项目 | 内容 |
|:---|:---|
| 实验版本 | BL5-v4-LearnableRun-PAM-noRNAFM-control |
| 执行时间 | 2026-06-06 22:56 – 23:03 (UTC+8) |
| 训练时长 | **390.9 秒 ≈ 6.5 分钟** |
| GPU | 2 × NVIDIA RTX PRO 6000 Blackwell Server Edition |
| 显存占用 | **0.12 GB**（单卡峰值，无 RNA-FM） |
| Commit Hash | `327075a` |
| 代码改动 | `models/bl5_dynamic_fusion.py` 新增 `run_pam_concat` fusion 类型及验证/forward 分支 |
| Config 文件 | `configs/bl5_v4_learnablerun_pam_nornafm_control.yaml` |
| 数据 split | `sgrna_safe`（formal_group_json），seed=42 |
| 数据集规模 | Train=4,697,495 / Val=741,552 / Test=954,326 |

---

## 2. 训练配置

```yaml
model:
  use_rnafm: false
  freeze_rnafm: false
  use_run: true
  use_learnable_run: true
  use_region: false
  use_pam_encoder: true
  fusion_type: run_pam_concat    # 新增
  pam_dim: 16
  d_model: 128
  mlp_hidden: 256
  mlp_hidden2: 64
  dropout: 0.3
  dropout2: 0.2

training:
  epochs: 10
  batch_size: 1024
  lr_run_encoder: 1.0e-3
  lr_pam_encoder: 1.0e-3
  lr_mlp: 1.0e-3
  focal_loss: true
  focal_gamma: 2.0
  gradient_clip: 1.0
  weight_decay: 1.0e-5
```

**关键约束确认**：
- ✅ `use_rnafm=false`（明确关闭 RNA-FM，不加载 checkpoint，不 tokenize）
- ✅ `use_run=true`, `use_learnable_run=true`（LearnableRunEncoder 启用）
- ✅ `use_pam_encoder=true`（PAM Encoder 启用）
- ✅ `split_mode=sgrna_safe`, `split.strategy=formal_group_json`
- ✅ `pos_weight` 未使用（focal loss 替代）
- ✅ Test 使用 best checkpoint（epoch 2）
- ✅ AUROC + AUPRC 同时报告

---

## 3. 训练曲线

| Epoch | Train Loss | Val AUROC | Val AUPRC | Val Accuracy | Val Precision | Val Recall | Val F1 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.006151 | 0.947445 | 0.465471 | 0.995292 | 0.656590 | 0.381678 | 0.482738 |
| 2 | 0.004654 | 0.950538 | **0.482704** | 0.995472 | 0.699561 | 0.373711 | 0.487172 |
| 3 | 0.004320 | 0.946510 | 0.445638 | 0.994901 | 0.587874 | 0.381678 | 0.462850 |
| 4 | 0.004139 | 0.946133 | 0.360180 | 0.992971 | 0.398929 | 0.436504 | 0.416872 |
| 5 | 0.003992 | 0.917362 | 0.193461 | 0.981837 | 0.134910 | 0.398313 | 0.201553 |
| 6 | 0.003891 | 0.903975 | 0.149905 | 0.952442 | 0.052568 | 0.426664 | 0.093603 |
| 7 | 0.003641 | 0.907844 | 0.167645 | 0.970053 | 0.080414 | 0.402765 | 0.134061 |
| 8 | 0.003627 | 0.906679 | 0.163115 | 0.968993 | 0.071883 | 0.368322 | 0.120289 |
| 9 | 0.003556 | 0.922761 | 0.196034 | 0.981798 | 0.121349 | 0.346532 | 0.179752 |
| 10 | 0.003501 | 0.878148 | 0.138243 | 0.929612 | 0.032428 | 0.389410 | 0.059870 |

**训练现象**：
- **Best epoch = 2**，val AUPRC = 0.4827
- Epoch 2 之后，val AUPRC 持续下降并在 0.14–0.20 区间震荡
- 与 RNAFM-PAM-noRun（best_epoch=1）类似，本实验也是 **early overfitting**
- 但与 RNAFM-PAM-noRun 不同，本实验的 val AUROC 在 epoch 1–4 保持在 0.94–0.95 的高位，说明 **LearnableRun 提供了非常稳定的排序信号**

---

## 4. Test 评估结果（best.pt, epoch 2）

| 指标 | 数值 |
|:---|:---:|
| Test Loss | 0.005805 |
| **AUROC** | **0.952749** |
| **AUPRC** | **0.177171** |
| Accuracy | 0.993558 |
| Precision | 0.142163 |
| Recall | 0.200851 |
| F1 | 0.166486 |

- Non-finite probability count: **0**（数值稳定）
- 阳性样本占比：3,057 / 954,326 ≈ **0.32%**

---

## 5. PAM 分层分析

> 📌 **PAM 口径**：按 `PAM_original = off_seq[20:23]`（positions 21-23），与 PAMEncoder 输入一致。NGG-only 为 819,984 条（85.9%），non-NGG-only 为 134,342 条（14.1%）。

详见独立报告：`results/bl5_v4_learnablerun_pam_nornafm_control/stratified_report.md`

| 子集 | 样本数 | 阳性数 | 阳性率 | AUROC | AUPRC |
|:---|:---:|:---:|:---:|:---:|:---:|
| Overall | 954,326 | 3,057 | 0.32% | 0.952749 | 0.177171 |
| NGG | 819,984 | 2,349 | 0.29% | 0.966826 | 0.230739 |
| non-NGG | 134,342 | 708 | 0.53% | 0.887270 | 0.360340 |

- NGG 子集占 test 的 85.9%，是实际评估核心
- non-NGG 子集 134,342 条，708 条阳性，阳性率 0.53%
- NGG 内部 motif 异质性：GGG 最优（AUPRC=0.2802），TGG 最弱（AUPRC=0.1972）

---

## 6. 消融对比全景表

| 模型 | 视角 | Test AUROC | Test AUPRC | 相对 BL5-v4-PAM |
|:---|:---|:---:|:---:|:---:|
| **BL5-v4-PAM** | RNA-FM + LearnableRun + PAM | 0.984194 | **0.531281** | 基准 |
| BL5-v4-NoPAM-control | RNA-FM + LearnableRun | 0.984098 | 0.502389 | −0.028892 |
| BL5-v4-RNAFM-PAM-noRun | RNA-FM + PAM | 0.837950 | 0.276529 | −0.254752 |
| BL0b-on-BL5split | RNA-FM only | 0.857756 | 0.295678 | −0.235603 |
| LearnableRun-only | LearnableRun only | 0.960909 | 0.294920 | −0.236361 |
| **LearnableRun-PAM-noRNAFM** | **LearnableRun + PAM** | **0.952749** | **0.177171** | **−0.354110** |
| PAM-only | PAM only | 0.499426 | 0.059223 | −0.472058 |

### 6.1 核心发现

**发现 1：LearnableRun + PAM 的 AUROC 极高（0.95），但 AUPRC 显著低于 LearnableRun-only（0.177 vs 0.295）**

这是本实验最意外的结果。按理来说，加入 PAM 应该提升或至少保持 AUPRC，但实际却下降了约 **40%**（相对）。

可能的原因：
- **PAM encoder 的 16-dim 信号干扰了 LearnableRun 的优化 landscape**：在缺少 RNA-FM 的强上下文约束下，PAM 的局部 motif 信息可能引入了噪声
- **Focal loss + PAM 的组合导致模型过度关注 PAM 维度**：由于 PAM 在 NGG 子集中高度同质化（>99% 为 NGG），PAM encoder 的输出变化极小，可能导致梯度更新方向分散
- **Concat 融合方式过于简单**：`run_pam_concat` 只是简单拼接，没有门控或注意力机制来平衡两个视角的贡献

**发现 2：去掉 RNA-FM 的损失远大于去掉 PAM 或 Run 单独的损失**

| 消融路径 | AUPRC 变化 | 相对损失 |
|:---|:---:|:---:|
| BL5-v4-PAM → NoPAM | 0.5313 → 0.5024 | −5.4% |
| BL5-v4-PAM → RNAFM-PAM-noRun | 0.5313 → 0.2765 | −47.9% |
| BL5-v4-PAM → **LearnableRun-PAM-noRNAFM** | 0.5313 → **0.1772** | **−66.7%** |

**去掉 RNA-FM 的损失（−66.7%）> 去掉 Run 的损失（−47.9%）> 去掉 PAM 的损失（−5.4%）**

这说明：**RNA-FM 是 BL5-v4-PAM 的绝对核心**。没有 RNA-FM 时，即使保留了 LearnableRun + PAM 两个结构化视角，AUPRC 也只有 0.177，远低于任何含 RNA-FM 的配置。

**发现 3：LearnableRun-only（0.2949）> LearnableRun-PAM-noRNAFM（0.1772）**

这说明：**在没有 RNA-FM 的情况下，PAM 不仅不能补强 LearnableRun，反而可能损害其性能**。

这与 "PAM 的稳定增益主要依赖 RNA-FM+LearnableRun 强上下文" 的假设一致。PAM 在全模型中的 +0.0289 增益，确实需要 RNA-FM 的上下文支撑才能发挥。

### 6.2 no-RNAFM 条件下的 2×2 矩阵

|  | PAM=✅ | PAM=❌ |
|:---|:---|:---|
| **Run=✅** | **LearnableRun-PAM-noRNAFM = 0.1772** | LearnableRun-only = 0.2949 |
| **Run=❌** | PAM-only = 0.0592 | random ≈ 0.0032 |

- 左上角（Run+PAM noRNAFM）已补全
- 矩阵清晰显示：**Run 是 no-RNAFM 条件下的绝对主导**
- PAM 在 Run=✅ 时反而拉低性能（0.1772 < 0.2949）

---

## 7. 回答计划中的关键问题

### Q1：PAM 是否能直接增强 LearnableRun？

**答**：**不能**。在没有 RNA-FM 的情况下，PAM 不仅不能增强 LearnableRun，反而将 AUPRC 从 0.2949 拉低到 0.1772。PAM 的增益是 **条件性的**——只有在 RNA-FM 与 LearnableRun **同时存在**的强联合上下文中，PAM 才能作为增量信号发挥作用。

### Q2：LearnableRun + PAM 是否能接近 RNA-FM + LearnableRun？

**答**：**不能**。LearnableRun-PAM-noRNAFM 的 AUPRC = 0.1772，而 NoPAM-control（RNA-FM + LearnableRun）的 AUPRC = 0.5024，差距达 **+183%**（相对）。RNA-FM 是把 LearnableRun 推到 0.50+ 级别的关键催化剂。

### Q3：完整 BL5-v4-PAM 的高分是否必须依赖 RNA-FM？

**答**：**是**。去掉 RNA-FM 后，即使保留 LearnableRun + PAM，AUPRC 也只有 0.1772（相比全模型 0.5313，下降 66.7%）。RNA-FM 是 BL5-v4-PAM 的绝对核心，LearnableRun 是重要支撑，PAM 是 RNA-FM+Run 上下文下的增量调节。

### Q4：PAM 的作用更依赖 Run 上下文，还是 RNA-FM 上下文？

**答**：**PAM 的正向增益依赖 RNA-FM 与 LearnableRun 同时存在的强联合上下文**，而非单独依赖其中某一个。证据：
- RNA-FM only = 0.2957，RNA-FM + PAM = 0.2765（低于 RNA-FM only，PAM 无增益）
- LearnableRun only = 0.2949，LearnableRun + PAM = 0.1772（低于 LearnableRun only，PAM 有害）
- RNA-FM + LearnableRun = 0.5024，RNA-FM + LearnableRun + PAM = 0.5313（高于 NoPAM，PAM 有增益）

**结论**：PAM 的正向增益不是在任意两视角组合中都成立的。RNA-FM+PAM noRun（0.2765）低于 RNA-FM-only baseline（0.2957），LearnableRun+PAM noRNAFM（0.1772）低于 LearnableRun-only（0.2949）。只有 RNA-FM+LearnableRun+PAM（0.5313）高于 RNA-FM+LearnableRun（0.5024），说明 PAM 的稳定正向贡献依赖 RNA-FM 与 LearnableRun 同时存在的强联合上下文。

---

## 8. 训练现象深度分析

### 8.1 AUROC 高但 AUPRC 低的矛盾

本实验 AUROC = 0.9527（极高），但 AUPRC = 0.1772（较低）。这说明：
- 模型**整体排序能力优秀**：能把大部分阳性排在阴性前面
- 但**头部精度差**：在概率最高的 top-k 区域，阳性浓度不够高
- 原因可能是 focal loss（γ=2.0）在极度不平衡数据上的副作用：模型倾向于压低所有概率以避免 false positive，导致头部正样本的概率不够突出

### 8.2 过拟合模式

- best_epoch = 2，与 LearnableRun-only（best_epoch=9）形成对比
- 加入 PAM 后，模型更快过拟合（2 epoch vs 9 epoch）
- 这再次说明 PAM 在 no-RNAFM 条件下引入了不稳定的梯度信号

### 8.3 与 BL5-v4-PAM 的显存对比

| 模型 | 显存峰值 | 训练时间 |
|:---|:---:|:---:|
| BL5-v4-PAM | 40.40 GB | ~3.5 h |
| LearnableRun-PAM-noRNAFM | **0.12 GB** | **~6.5 min** |

无 RNA-FM 时，模型参数量极小（LearnableRunEncoder + PAM Encoder + MLP，约 10K–100K 级别），显存和训练时间大幅下降。

---

## 9. 结论与下一步

### 9.1 本实验回答的科学问题

> **"在没有 RNA-FM 的情况下，PAM 能否补强 LearnableRun？"**

**答**：**不能**。LearnableRun-PAM-noRNAFM 的 AUPRC = 0.1772，低于 LearnableRun-only 的 0.2949。这说明 PAM 并不能在无 RNA-FM 条件下直接增强 LearnableRun，简单 concat 甚至会损害头部排序能力。结合 RNAFM-PAM-noRun（0.2765）也低于 RNA-FM-only（0.2957），而完整 BL5-v4-PAM（0.5313）高于 NoPAM-control（0.5024），可以说明 **PAM 的正向增益不是单独来自 PAM motif，也不是任意两视角组合都成立，而是依赖 RNA-FM + LearnableRun 的强联合上下文**。

### 9.2 三视角消融矩阵已完成

| 视角组合 | AUPRC | 状态 |
|:---|:---:|:---:|
| RNA-FM + LearnableRun + PAM | 0.5313 | ✅ 全模型 |
| RNA-FM + LearnableRun | 0.5024 | ✅ NoPAM |
| RNA-FM + PAM | 0.2765 | ✅ RNAFM-PAM-noRun |
| LearnableRun + PAM | **0.1772** | ✅ **本实验** |
| RNA-FM only | 0.2957 | ✅ BL0b |
| LearnableRun only | 0.2949 | ✅ LearnableRun-only |
| PAM only | 0.0592 | ✅ PAM-only |

**矩阵结论**：
- **RNA-FM 是核心**：任何含 RNA-FM 的组合 AUPRC ≥ 0.276，不含 RNA-FM 的组合 AUPRC ≤ 0.295
- **LearnableRun 是 RNA-FM 的最佳搭档**：RNA-FM + LearnableRun = 0.5024，RNA-FM + PAM = 0.2765
- **PAM 是增量调节**：只在 RNA-FM + LearnableRun **同时存在**的强联合上下文下贡献 +0.0289；单独加到 RNA-FM 或 LearnableRun 上均无法产生正向增益

### 9.3 对 BL5→BL6 的启示

1. **RNA-FM 不可移除**：任何后续模型必须保留 RNA-FM（fine-tuned）
2. **LearnableRun 是次核心**：NoPAM（RNA-FM + LearnableRun）已达 0.5024，接近全模型 0.5313
3. **PAM 的融合方式需要优化**：当前 `run_pam_concat` / `simple_concat` 是粗暴拼接，BL5/BL6 的动态融合（Cross-Attn / Gate）可能更好地发挥 PAM 的增量作用
4. **Early stopping 对轻量模型更敏感**：无 RNA-FM 时 best_epoch 提前到 2，训练脚本应考虑加入 early stopping patience

---

## 10. AGENTS.md 合规声明

```python
"""
AGENTS.md compliance: [use_rnafm=False, freeze_rnafm=False,
                       split_mode=sgrna_safe, pos_weight=None]
确认本文件遵守 AGENTS.md 约束
"""
```

- ✅ 序列输入经过 LearnableRunEncoder（base pair indices）和 PAM Encoder（one-hot），禁止裸字符串输入 NN
- ✅ `use_rnafm=true/false` 显式声明
- ✅ `freeze_rnafm=true/false` 显式声明（虽不影响运行）
- ✅ `split_mode=sgrna_safe` 显式声明
- ✅ `pos_weight` 未使用（focal loss 替代）
- ✅ Test 使用 best checkpoint（epoch 2）
- ✅ AUROC + AUPRC 同时报告
- ✅ formal split 已使用（`split_source=formal_group_json`）
