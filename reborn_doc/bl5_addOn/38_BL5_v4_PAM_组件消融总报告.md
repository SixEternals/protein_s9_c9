# 38. BL5-v4-PAM 组件消融总报告

> 📅 2026-06-08 ｜ 🎯 将 BL5-v4-PAM 的完整组件消融矩阵整理为独立、可汇报文档
>
> **性质**：跨实验消融总结，基于已有 results 和单实验执行报告。未训练模型、未运行推理、未调用 GPU。

---

## 1. 实验矩阵总览

所有实验均在 **同一 formal BL5 split**（`formal_split_bl5_seed42.json`, `sgrna_safe`）的 **同一 test cohort**（954,326 样本，3,057 observed_positive，951,269 unobserved_candidate，72 个 unseen sgRNA_type）上评估。

| # | 实验 | RNA-FM | LearnableRun | PAM | Gate | AUROC | AUPRC | 类型 | 目的 |
|:---:|:---|:---:|:---:|:---:|:---:|---:|---:|:---|:---|
| 1 | BL0b-on-BL5split | ✅ | ❌ | ❌ | ❌ | 0.8578 | 0.2957 | 单组件 baseline | RNA-FM-only 基线 |
| 2 | LearnableRun-only | ❌ | ✅ | ❌ | ❌ | 0.9609 | 0.2949 | 单组件 baseline | LearnableRun 单独贡献 |
| 3 | PAM-only | ❌ | ❌ | ✅ | ❌ | 0.4994 | 0.0592 | 单组件 baseline | PAM 单独贡献，AUROC≈random |
| 4 | NoPAM-control | ✅ | ✅ | ❌ | ❌ | 0.9841 | 0.5024 | 严格消融 | 去掉 PAM 后的性能 |
| 5 | RNAFM-PAM-noRun | ✅ | ❌ | ✅ | ❌ | 0.8380 | 0.2765 | 严格消融 | 去掉 LearnableRun 后 RNA-FM+PAM |
| 6 | LearnableRun-PAM-noRNAFM | ❌ | ✅ | ✅ | ❌ | 0.9527 | 0.1772 | 严格消融 | 去掉 RNA-FM 后 Run+PAM |
| 7 | **BL5-v4-PAM** | ✅ | ✅ | ✅ | ❌ | 0.9842 | **0.5313** | **主模型** | BL5 anchor |
| 8 | PAM-shuffle-control | ✅ | ✅ | ⚠ shuffled | ❌ | 0.6697 | 0.1389 | Negative control | PAM 对应关系被破坏后的效果 |
| 9 | BL6-1-PAM-Gated-Fusion | ✅ | ✅ | ✅ | ✅ | 0.9850 | 0.5399 | Architecture variant | Gate 加法实验（非严格 BL5 消融） |

---

## 2. 消融类型分类

不是所有对比都是「消融」。下表明确各类对比的命名和解释边界：

| 对比 | 是否严格消融 | 正确名称 | AUPRC 差 | 解释 |
|:---|:---:|:---|---:|:---|
| `PAM − NoPAM` | ✅ | PAM Encoder ablation | **+0.0289** | PAM Encoder 在 v4 框架下的边际贡献 |
| `PAM − Shuffle` | ✅ | PAM correspondence control | **+0.3924** | 正确 PAM 与样本对应关系被破坏后性能崩溃，说明 PAM 信息确实被模型使用 |
| `Shuffle − (PAM − NoPAM)` | — | — | — | Shuffle（0.1389）远低于 NoPAM（0.5024），说明 shuffle 破坏的不只是 PAM 增益，还干扰了整体融合 |
| `NoPAM − BL0b` | ❌ | Framework baseline comparison | **+0.2067** | **不是** LearnableRun 的纯贡献，而是 BL5-v4 no-PAM framework（含不同 classifier/head/training recipe）的综合增益 |
| `PAM − RNAFM-PAM-noRun` | ✅ | LearnableRun ablation | **+0.2548** | 去掉 LearnableRun 后 AUPRC 下降 47.9%，Run 是 BL5-v4-PAM 的核心增益来源之一 |
| `PAM − LearnableRun-PAM-noRNAFM` | ✅ | RNA-FM ablation | **+0.3541** | 去掉 RNA-FM 后 AUPRC 下降 66.7%，RNA-FM 是 BL5-v4-PAM 的绝对核心 |
| `BL6-1 − PAM` | ⚠️ | Gate addition | **+0.0086** | Single-run gate improvement；需 bootstrap CI + multi-seed 后定性 |
| `RNAFM-PAM-noRun − BL0b` | — | PAM addition to RNA-FM | **−0.0191** | PAM 单独加到 RNA-FM 上未带来增益，略低于 RNA-FM-only |
| `LearnableRun-PAM-noRNAFM − LearnableRun-only` | — | PAM addition to Run | **−0.1177** | PAM 单独加到 Run 上反而降低性能，可能干扰优化 |

---

## 3. 核心发现

### 3.1 组件贡献层次

```
RNA-FM 是绝对核心     → 去掉后 AUPRC 从 0.5313 → 0.1772（−66.7%）
LearnableRun 是核心搭档 → 去掉后 AUPRC 从 0.5313 → 0.2765（−47.9%）
PAM 是条件性增量       → 去掉后 AUPRC 从 0.5313 → 0.5024（−5.4%）
Gate 是微调增益        → 加上后 AUPRC 从 0.5313 → 0.5399（+1.6%）
```

### 3.2 三组件协同关系

| 发现 | 证据 | 解释 |
|:---|:---|:---|
| **RNA-FM + LearnableRun 是主性能来源** | RNA-FM only=0.2957, Run only=0.2949, 组合=0.5024 | 两个视角单独时信息量相当（AUPRC ≈ 0.29），融合后跃升 ~70%。RNA-FM 提供隐式序列上下文，LearnableRun 提供显式错配模式/seed 权重，互补性极强 |
| **PAM 单独几乎无价值** | PAM-only AUPRC=0.0592, AUROC≈0.5 | PAM motif 本身在 NGG 子集内无区分能力（所有 NGG 位点的 PAM 都是 NGG）。PAM 的区分信号来自「NGG vs 非 NGG」，不是「这个 NGG 切不切」 |
| **PAM 的稳定贡献需要强联合上下文** | PAM+RNAFM=0.2765 < RNAFM-only=0.2957; PAM+Run=0.1772 < Run-only=0.2949; 但在 RNA-FM+Run 上下文中 PAM 贡献 +0.0289 | PAM 单独加到 RNA-FM 或 Run 上都不能产生正向增益。只有 RNA-FM 和 LearnableRun **同时存在**时，PAM 才能作为增量信号稳定贡献 |
| **PAM 增益依赖正确对应关系** | Shuffle AUPRC=0.1389 << NoPAM=0.5024 | PAM shuffle 破坏了 PAM 与样本的对应关系，模型性能崩溃（不仅失去 PAM 增益，还因错误 PAM 信号被误导）。这证明 PAM 增益不是来自额外参数量或偶然 |

> 💡 **比喻理解**：RNA-FM 像一位语言学教授（理解序列上下文），LearnableRun 像一位校对员（统计连续错配数量），PAM 像一个只认识三个字母（NGG）的路牌。教授和校对员合作已经很强（0.5024），路牌本身告诉不了你文章写得好不好（0.0592）。但当他俩都在工作时，看一眼路牌能帮他们快速定位（→0.5313）。如果把路牌随机换成假的（shuffle），反而会误导他们走错方向（→0.1389）。

### 3.3 性能分布图谱

```
AUPRC
0.55 ┤                                   BL6-1(0.5399) ●
     │                                   BL5-v4-PAM(0.5313) ●
0.50 ┤               NoPAM(0.5024) ●
     │
0.30 ┤  BL0b(0.2957) ●  Run-only(0.2949) ●
     │  RNAFM-PAM-noRun(0.2765) ●
0.25 ┤
     │
0.18 ┤  LearnableRun-PAM-noRNAFM(0.1772) ●
     │
0.14 ┤  Shuffle(0.1389) ●
     │
0.06 ┤  PAM-only(0.0592) ●
     └──┴────┴────┴────┴────┴────┴────┴──
```

清晰分为三层：
- **上层（AUPRC ≥ 0.50）**：RNA-FM + LearnableRun 的强联合。加 PAM → 0.53，加 Gate → 0.54。
- **中层（AUPRC 0.27–0.30）**：单个核心组件（RNA-FM 或 LearnableRun 各自单独）。PAM 在此层无增益。
- **下层（AUPRC < 0.18）**：PAM 单独、无 RNA-FM 的双组件组合、或 PAM 对应关系被 shuffle 破坏。模型基本不具备实际可用性。

---

## 4. 数据与评估一致性

| 检查项 | 状态 |
|:---|:---:|
| 九实验在同一 formal BL5 split test set 上评估 | ✅ |
| test sample_index 跨实验完全对齐 | ✅ |
| test label 跨实验完全一致 | ✅ |
| test 使用 best.pt（val AUPRC 最佳 checkpoint） | ✅ |
| 同时报告 AUROC 和 AUPRC | ✅ |
| PAM 坐标使用 `off_seq[20:23]`（positions 21-23） | ✅ |
| label=0 解释为 unobserved_candidate | ✅ |

---

## 5. 解释边界

以下表述**不应**出现在汇报或论文中：

| ❌ 不能说 | ✅ 应该说 |
|:---|:---|
| "LearnableRun 贡献 +0.21 AUPRC" | "NoPAM − BL0b = +0.2067 是 BL5-v4 no-PAM framework 的综合增益，不能归因于单一组件" |
| "PAM 增强了 RNA-FM" | "RNA-FM + PAM noRun（0.2765）低于 RNA-FM-only（0.2957），PAM 单独加到 RNA-FM 上未产生增益" |
| "BL6-1 已全面超越 BL5-v4-PAM" | "BL6-1 single-run AUPRC=0.5399 高于 BL5 historical best 0.5313，但需 bootstrap CI、gate audit、multi-seed repeat 确认稳定性" |
| "PAM 是核心组件" | "PAM 是条件性增量：单独无用，在 RNA-FM+LearnableRun 强联合上下文中贡献 +0.0289" |
| "消融矩阵证明 PAM 有效" | "消融矩阵证明 PAM 在完整上下文中的边际增益（+0.0289）依赖正确 PAM 与样本的对应关系（shuffle → −0.3924）" |

---

## 6. 未覆盖的消融维度

| 维度 | 状态 | 说明 |
|:---|:---:|:---|
| Region encoder 消融 | 未做 | BL5-v4-PAM 未使用 Region 编码（use_region=false） |
| RNA-FM frozen vs fine-tune 消融 | 未做 | BL5-v4 所有实验均为 fine-tune（freeze_rnafm=false） |
| Classifier depth 消融 | 未做 | 不同实验间 classifier 层数不同（1 层 vs 2 层），未做同结构控制 |
| Training seed repeat | 部分完成 | BL5-v4-PAM 有两个 seed（0.5313 / 0.5161），差异 ~0.015；其他实验仅 single-run |
| Focal loss gamma 消融 | 未做 | 所有实验使用 gamma=2.0 |

---

## 7. 源码与结果索引

| 实验 | config | results 目录 | 执行报告 |
|:---|:---|:---|:---|
| BL0b-on-BL5split | — | `results/bl0b_on_bl5split/` | 间接（#32 #34） |
| LearnableRun-only | `bl5_v4_learnablerun_only_control.yaml` | `results/bl5_v4_learnablerun_only_control/` | #30 |
| PAM-only | `bl5_v4_pam_only_control.yaml` | `results/bl5_v4_pam_only_control/` | #31 |
| NoPAM-control | `bl5_v4_nopam_control.yaml` | `results/BL5-v4-NoPAM-control/` | 间接 |
| RNAFM-PAM-noRun | `bl5_v4_rnafm_pam_norun_control.yaml` | `results/bl5_v4_rnafm_pam_norun_control/` | #32, #33 |
| LearnableRun-PAM-noRNAFM | `bl5_v4_learnablerun_pam_nornafm_control.yaml` | `results/bl5_v4_learnablerun_pam_nornafm_control/` | #34 |
| BL5-v4-PAM | `bl5_v4_pam.yaml` | `results/bl5_v4_pam/` | 多份引用 |
| PAM-shuffle-control | `bl5_v4_pam_shuffle_control.yaml` | `results/bl5_v4_pam_shuffle_control/` | 间接 |
| BL6-1-PAM-Gated-Fusion | `bl6_1_pam_gated_fusion.yaml` | `results/bl6_1_pam_gated_fusion/` | #26, #27, #28 |

---

## 8. 下一步建议

| 优先级 | 行动 | 理由 |
|:---:|:---|:---|
| 🔴 P0 | 统一消融总图（单张 figure） | 当前只有表格，缺可视化。建议做 stacked bar + delta 箭头图 |
| 🟡 P1 | BL5-v4-PAM multi-seed：补第 3 个 seed | 两个 seed 差异 ~0.015，需要更多 seed 估计训练方差 |
| 🟡 P1 | BL6-1 multi-seed repeat | 确定 gate 增益是否跨 seed 稳定 |
| 🟢 P2 | BL4-full formal split | 补 Region 编码的消融证据 |
| 🟢 P2 | kNN baseline | 作为非深度学习的 reference point |

---

## 9. 合规说明

- 本文档仅整理已有实验结果，未训练模型，未运行推理，未调用 GPU。
- 所有 AUROC/AUPRC 数据来自已有 `results/experiments.csv`、`summary.json` 和单实验执行报告。
- 未删除或覆盖 data/ / reference/ 下任何文件。
- 未 commit / push。
- PAM 坐标统一为 `off_seq[20:23]`（positions 21-23）。
- label=0 解释为 unobserved_candidate，不是 safe site。
