# BL5-v4-PAM-only-control 执行报告

> **版本**: BL5-v4-PAM-only-control  
> **执行时间**: 2026-06-06  
> **训练时长**: 264.7s (~4.4 min)  
> **GPU**: 2× (CUDA 0,1), DDP  
> **Commit**: 327075a  

---

## 1. 任务目标

执行 PAM-only 组件级消融实验，补齐 BL5-v4-PAM 的消融矩阵：

> **只保留 PAM Encoder（positions 21-23 one-hot），关闭 RNA-FM，关闭 Run，关闭 LearnableRun**，测量 PAM 单视图在 Formal Split 上的性能。

**核心科学问题**：
1. PAM 单独是否具有预测能力？
2. BL5-v4-PAM 的增益是否可能来自 PAM 分布 shortcut？
3. PAM Encoder 的价值是否依赖 RNA-FM / Run 上下文？

---

## 2. 模型配置

| 配置项 | 值 | 说明 |
|:---|:---|:---|
| `use_rnafm` | `false` | **关闭 RNA-FM** |
| `use_run` | `false` | **关闭 Run** |
| `use_learnable_run` | `false` | **关闭 LearnableRun** |
| `use_pam_encoder` | `true` | 使用 PAM Encoder |
| `fusion_type` | `pam_only` | 仅使用 PAM 特征 |
| `pam_dim` | 16 | PAM Encoder 输出维度 |
| `mlp_hidden` | 256 | 分类器第一层 |
| `mlp_hidden2` | 64 | 分类器第二层 |
| `dropout` | 0.3 | Dropout |
| `focal_gamma` | 2.0 | Focal loss |
| `batch_size` | 1024 | 2 GPU 各 512 |
| `epochs` | 10 | 固定 epoch |
| `lr_pam_encoder` | 1e-3 | PAM 编码器学习率 |
| `lr_mlp` | 1e-3 | MLP 学习率 |

**参数量**: ~3K（PAMEncoder + 小型 MLP），无 RNA-FM，无 Run Encoder。

---

## 3. 数据集与划分

- **数据来源**: CCLMoff 完整数据集（6,393,373 条）
- **划分方式**: `sgrna_safe` group split via `formal_split_bl5_seed42.json`
- **Test 规模**: 954,326 条，3,057 positive，951,269 unobserved_candidate，72 sgRNA 类型
- **Test 对齐验证**: ✅ sample_index 和 label 与 BL5-v4-PAM 完全匹配

---

## 4. 训练过程

| Epoch | Train Loss | Val AUROC | Val AUPRC | Val Precision | Val Recall | Val F1 |
|:---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.013478 | 0.5370 | 0.0625 | 1.0000 | 0.0553 | 0.1048 |
| 2 | 0.012042 | 0.5716 | 0.0624 | 1.0000 | 0.0558 | 0.1056 |
| **3** | 0.011927 | 0.5765 | **0.0630** | 1.0000 | 0.0560 | 0.1061 |
| 4 | 0.011863 | 0.5746 | 0.0628 | 1.0000 | 0.0560 | 0.1061 |
| 5 | 0.011760 | 0.5782 | 0.0629 | 1.0000 | 0.0560 | 0.1061 |
| 6 | 0.011806 | 0.5744 | 0.0628 | 1.0000 | 0.0560 | 0.1061 |
| 7 | 0.011809 | 0.5743 | 0.0628 | 1.0000 | 0.0560 | 0.1061 |
| 8 | 0.011707 | 0.5646 | 0.0626 | 1.0000 | 0.0560 | 0.1061 |
| 9 | 0.011746 | 0.5745 | 0.0628 | 1.0000 | 0.0560 | 0.1061 |
| 10 | 0.011727 | 0.5667 | 0.0626 | 1.0000 | 0.0560 | 0.1061 |

- **Best epoch**: 3（Val AUPRC=0.06296）
- 所有 epoch 的 Val Precision=1.0，Recall≈0.056，说明模型在阈值 0.5 下几乎只预测负类
- 学习率变化: epoch 1-7 保持 1e-3，epoch 8-10 ReduceLROnPlateau 降至 5e-4

---

## 5. Test 评估结果（best.pt，epoch 3）

| 指标 | 值 | 说明 |
|:---|---:|:---|
| **AUROC** | **0.4994** | 接近随机（0.5），pairwise ranking 能力极弱 |
| **AUPRC** | **0.0592** | 高于 random positive-rate baseline（~0.0032），但远低于所有单/多视角 baseline |
| Accuracy | 0.9970 | 阈值 0.5，大量预测为负类 |
| Precision | 1.0000 | 阈值 0.5 |
| Recall | 0.0559 | 阈值 0.5 |
| F1 | 0.1059 | 阈值 0.5 |
| Test Loss | 0.0073 | BCE + Focal |

---

## 6. PAM 分层分析

### 6.1 主表（Formal Split 口径）

| Model | Test AUROC | Test AUPRC |
|:---|---:|---:|
| BL0b-on-BL5split | 0.857756 | 0.295678 |
| LearnableRun-only | 0.960909 | 0.294920 |
| **PAM-only** | **0.499426** | **0.059223** |
| BL5-v4-NoPAM-control | 0.984098 | 0.502389 |
| BL5-v4-PAM-shuffle-control | 0.669701 | 0.138883 |
| BL5-v4-PAM | 0.984194 | 0.531281 |
| BL6-1-PAM-Gated-Fusion | 0.984993 | 0.539917 |

### 6.2 PAM Motif 分层

| Layer | Samples | Positive | Positive Ratio | AUROC | AUPRC |
|:---|---:|---:|---:|---:|---:|
| **All test** | 954,326 | 3,057 | 0.003203 | 0.499 | 0.059 |
| **NGG-only** | 819,984 | 2,349 | 0.002865 | 0.488 | 0.003 |
| **non-NGG-only** | 134,342 | 708 | 0.005270 | 0.670 | 0.248 |
| PAM=AGG | 277,247 | 744 | 0.002684 | 0.500 | 0.003 |
| PAM=TGG | 292,861 | 703 | 0.002400 | 0.500 | 0.002 |
| PAM=GGG | 203,284 | 716 | 0.003522 | 0.500 | 0.004 |
| PAM=CGG | 46,592 | 186 | 0.003992 | 0.500 | 0.004 |
| PAM=others | 134,342 | 708 | 0.005270 | 0.670 | 0.248 |

### 6.3 分层细节（Mean/Median Probability）

| Layer | mean_prob_pos | mean_prob_neg | median_prob_pos | median_prob_neg |
|:---|---:|---:|---:|---:|
| All test | 0.1830 | 0.1361 | 0.1425 | 0.1425 |
| NGG-only | 0.1394 | 0.1400 | 0.1425 | 0.1425 |
| non-NGG-only | 0.3274 | 0.1121 | 0.1190 | 0.1062 |
| PAM=AGG | 0.1436 | 0.1436 | 0.1436 | 0.1436 |
| PAM=TGG | 0.1425 | 0.1425 | 0.1425 | 0.1425 |
| PAM=GGG | 0.1280 | 0.1280 | 0.1280 | 0.1280 |
| PAM=CGG | 0.1551 | 0.1551 | 0.1551 | 0.1551 |
| PAM=others | 0.3274 | 0.1121 | 0.1190 | 0.1062 |

**关键观察**：
- 每个 canonical NGG motif（AGG/TGG/GGG/CGG）内部，模型对所有样本输出**完全相同的概率**（mean=median，pos=neg）
- 这说明 PAM-only 模型学到了一个简单的 **motif-level lookup table**：AGG→0.144, TGG→0.142, GGG→0.128, CGG→0.155
- **模型无法区分同一 PAM motif 内的 observed_positive 与 unobserved_candidate**，它做的不是"这个 off-target 是否危险"，而是"这个 PAM motif 历史上 positive rate 高不高"
- non-NGG-only 的 AUROC=0.670 / AUPRC=0.248 主要反映**不同 non-NGG PAM motif 之间存在较强的 label-rate heterogeneity**，而非 PAM motif 本身的样本级生物学判别信号

---

## 7. 科学结论

### 7.1 PAM 单独不足以预测

- **PAM-only AUROC ≈ 0.5（随机）**，说明其整体 pairwise ranking 能力很弱
- **AUPRC = 0.059**，虽高于 random positive-rate baseline（~0.0032），但远低于 LearnableRun-only（0.295）和 BL0b-on-BL5split（0.296）
- 甚至低于 PAM-shuffle-control（0.139）——说明即使 PAM 被 shuffle，只要附着在 RNA-FM+Run 上下文中，就比 PAM 单独更有价值
- PAM-only 的 AUPRC 信号更像是 **motif-level 分布偏置**，而不是样本级脱靶判别能力

### 7.2 NGG vs non-NGG 分布偏置

- **NGG-only**（86% 的 test 数据）AUROC = 0.488，几乎随机；每个 canonical NGG motif 内部模型输出恒定概率，说明模型无法从 NGG PAM 内部区分正负样本
- **non-NGG-only**（14% 的 test 数据）AUROC = 0.670 / AUPRC = 0.248，这个提升主要来自**不同 non-NGG PAM motif 之间的 label-rate heterogeneity**，而非 PAM motif 本身的样本级生物学判别信号
- 由于 PAM-only 模型对同一 PAM motif 输出恒定概率，它只能学习 motif-level lookup table，不能区分同一 motif 内的 observed_positive 与 unobserved_candidate

### 7.3 PAM Encoder 的价值定位

BL5-v4-PAM-only-control 只使用 off-target positions 21-23 的 PAM one-hot 特征，不使用 RNA-FM、LearnableRun 或 protospacer 序列上下文。结果显示：

- **PAM-only 的 test AUROC = 0.499426，接近随机**，说明其整体 pairwise ranking 能力很弱
- **test AUPRC = 0.059223，虽高于 random positive-rate baseline**，但远低于 BL0b-on-BL5split（0.296）、LearnableRun-only（0.295）、NoPAM-control（0.502）和 BL5-v4-PAM（0.531）
- 分层结果显示，canonical NGG motif 内部模型对同一 PAM 输出恒定概率，无法区分 observed_positive 与 unobserved_candidate
- non-NGG 子集中的较高 AUPRC 主要反映不同 PAM motif 的 label-rate heterogeneity

> **结论：PAM motif 单独不足以解释 BL5-v4-PAM 的性能提升，PAM Encoder 的有效增益更可能来自与 RNA-FM 和 LearnableRun 上下文的组合。**

### 7.4 与消融矩阵的对照

| 实验 | 包含 RNA-FM | 包含 Run | 包含 PAM | Test AUPRC |
|:---|:---:|:---:|:---:|:---:|:---:|
| BL0b-on-BL5split | ✅ | ❌ | ❌ | 0.296 |
| LearnableRun-only | ❌ | ✅ | ❌ | 0.295 |
| PAM-only | ❌ | ❌ | ✅ | **0.059** |
| NoPAM-control | ✅ | ✅ | ❌ | 0.502 |
| PAM-shuffle-control | ✅ | ✅ | ✅(shuffled) | 0.139 |
| BL5-v4-PAM | ✅ | ✅ | ✅ | 0.531 |
| BL6-1 | ✅ | ✅ | ✅(Gated) | 0.540 |

**模式清晰**：
- 任意单一视角（RNA-FM / Run / PAM）单独使用时，AUPRC ≈ 0.06-0.30
- RNA-FM + Run 组合跃升至 0.50
- 加上 PAM 后进一步提升至 0.53-0.54
- **三种视角呈递进互补关系**，没有任何一个可以单独解释最终性能

---

## 8. 后续建议

**当前已完成的消融矩阵**：

| 实验 | 组成 | Test AUPRC |
|:---|:---|---:|
| BL0b-on-BL5split | RNA-FM only | 0.296 |
| LearnableRun-only | Run only | 0.295 |
| **PAM-only** | **PAM only** | **0.059** |
| NoPAM-control | RNA-FM + LearnableRun | 0.502 |
| PAM-shuffle-control | RNA-FM + LearnableRun + shuffled PAM | 0.139 |
| BL5-v4-PAM | RNA-FM + LearnableRun + PAM | 0.531 |
| BL6-1 | RNA-FM + LearnableRun + PAM (Gated) | 0.540 |

**仍未完成**（如需完整 2-view / 3-view 矩阵闭环）：
- RNA-FM + PAM，无 Run
- LearnableRun + PAM，无 RNA-FM

当前矩阵已足够回答核心科学问题（PAM 单独不强、shuffle 不强、只有正确融合才强）。如需更严格的组件贡献拆解，可补上述两个实验。

**下一步**：基于当前消融矩阵，可以开始设计更复杂的融合结构（如 BL6-2/BL6-3）或尝试其他 PAM 编码方式（如可学习 PAM embedding 替代 one-hot）。

---

## 9. 文件与产物

| 文件 | 路径 |
|:---|:---|
| Config | `configs/bl5_v4_pam_only_control.yaml` |
| Smoke Config | `configs/bl5_v4_pam_only_control_smoke.yaml` |
| Run Script | `run/run_bl5_v4_pam_only_control_2gpu.sh` |
| Best Checkpoint | `results/bl5_v4_pam_only_control/checkpoints/best.pt` |
| Epoch Metrics | `results/bl5_v4_pam_only_control/epoch_metrics.csv` |
| Test Predictions | `results/bl5_v4_pam_only_control/test_predictions.csv` |
| Summary JSON | `results/bl5_v4_pam_only_control/summary.json` |
| PAM 分层报告 | `results/bl5_v4_pam_only_control/pam_only_ablation_report.md` |
| 模型代码 | `models/bl5_dynamic_fusion.py`（新增 `pam_only` fusion_type） |
| 训练脚本 | `scripts/train_bl5.py`（新增 `use_run=false` 支持） |
