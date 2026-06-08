# BL5-v4-RNAFM-PAM-noRun-control 实验评语与消融解读

> **定位**：本文件是对 `reborn_doc/32_BL5_v4_RNAFM_PAM_noRun_control_执行报告.md` 的**高层解读与答辩口径**，用于向导师汇报时直接引用。

---

## 一句话结论

**BL5-v4-RNAFM-PAM-noRun-control 显示，在 BL5-v4-PAM 中移除 LearnableRun 后，test AUPRC 从 0.5313 大幅下降到 0.2765（下降 0.2548），远大于移除 PAM 的下降幅度 0.0289（约 8.8 倍）。这说明 BL5-v4-PAM 的主体性能依赖 RNA-FM 与 LearnableRun 的互补，PAM Encoder 提供的是在强上下文基础上的额外增量信号，而不是单独驱动模型性能的 shortcut。**

---

## 1. 实验定位

本实验**不是**为了追求高分的新模型，而是一个**remove-Run ablation（移除 Run/LearnableRun 的消融实验）**。它的核心价值不在于"跑出了多高的 AUPRC"，而在于**"跑出的低分本身就是证据"**——它证明了 BL5-v4-PAM 的强性能不是 RNA-FM+PAM 自己撑起来的，LearnableRun/Run 视角才是核心支柱。

| 维度 | 评价 |
|:---|:---|
| 消融价值 | **高** |
| 科学解释价值 | **高** |
| 作为主模型的独立价值 | **低**（预期内） |
| 对 BL5-v4-PAM 结构合理性的支持 | **强支持** |
| 对 Run/LearnableRun 必要性的支持 | **强支持** |
| 对 PAM 是主力信号的支持 | **不支持** |

---

## 2. 消融对比全景表

| 模型 | 视角 | Test AUROC | Test AUPRC | 相对 BL5-v4-PAM |
|:---|:---|:---:|:---:|:---:|
| **BL5-v4-PAM** | RNA-FM + LearnableRun + PAM | 0.984194 | **0.531281** | 基准 |
| BL5-v4-NoPAM-control | RNA-FM + LearnableRun | 0.984098 | 0.502389 | −0.028892 |
| **BL5-v4-RNAFM-PAM-noRun** | RNA-FM + PAM | 0.837950 | **0.276529** | **−0.254752** |
| BL0b-on-BL5split | RNA-FM only | 0.857756 | 0.295678 | −0.235603 |
| LearnableRun-only | LearnableRun only | 0.960909 | 0.294920 | −0.236361 |
| PAM-only | PAM only | 0.499426 | 0.059223 | −0.472058 |

**关键观察**：
- RNAFM-PAM-noRun（0.2765）与 BL0b（0.2957）、LearnableRun-only（0.2949）处于同一水平
- 这说明 **RNA-FM 的隐式上下文 ≈ LearnableRun 的显式错配模式**，两者单独能力相当，但组合后产生质变
- PAM-only（0.0592）远低于所有含 RNA-FM 或 Run 的模型，说明 PAM 单独不具备排序能力

---

## 3. 核心量化发现

### 3.1 去掉 PAM 的损失
```
BL5-v4-PAM      0.531281
− NoPAM         0.502389
─────────────────────────
  绝对下降      0.028892
  相对下降      −5.4%
```

### 3.2 去掉 Run 的损失
```
BL5-v4-PAM            0.531281
− RNAFM-PAM-noRun     0.276529
───────────────────────────────
  绝对下降            0.254752
  相对下降            −47.9%
```

### 3.3 两者的倍数关系
```
去掉 Run 的损失 / 去掉 PAM 的损失 = 0.254752 / 0.028892 ≈ 8.82
```

**结论**：去掉 Run 的性能损失大约是去掉 PAM 的 **8.8 倍**。这个数量级差异非常清晰地表明：
- **LearnableRun 是 BL5-v4-PAM 的主干增益来源**
- **PAM 是在 RNA-FM + LearnableRun 已经形成强上下文之后提供的额外增量**
- PAM 不是主力单视角，也不能单独支撑 BL5-v4-PAM 的性能

---

## 4. 导师可能问的问题 & 标准答法

### Q1：BL5-v4-PAM 为什么强？

**答**：不是单靠 PAM，也不是单靠 RNA-FM。我们做了 remove-PAM 和 remove-Run 两个关键消融。去掉 PAM 后 AUPRC 从 0.5313 降到 0.5024，说明 PAM 有真实增量；但**去掉 LearnableRun 后 AUPRC 直接降到 0.2765**，说明 LearnableRun/Run 视角是完整 BL5-v4-PAM 的核心支撑。PAM 的作用更像是在强 RNA-FM+Run 框架上做局部调节，而不是单独决策。

### Q2：PAM 是不是 shortcut？

**答**：不是简单 PAM shortcut。PAM-only 只有 AUPRC=0.0592，PAM-shuffle 只有 AUPRC=0.1389，而正确 PAM + RNA-FM + LearnableRun 可以达到 0.5313。这说明：
- PAM 单独不强
- 错误对应的 PAM（shuffle）也不强
- 只有**正确 PAM 放在 RNA-FM+Run 上下文里**才有稳定增益

### Q3：这个 RNAFM-PAM-noRun 消融好不好？

**答**：作为新模型它不好，因为 AUPRC 只有 0.2765；但**作为消融实验它很好**，因为它证明了去掉 Run 后完整模型性能大幅崩塌。这个结果是**预期内的性能崩塌，崩塌本身就是消融证据**。它支持我们把 LearnableRun 视为 BL5-v4-PAM 的核心组件。

---

## 5. 汇报口径建议

### ❌ 不要这样汇报
> "RNAFM-PAM-noRun 表现很差，AUPRC 只有 0.27，比全模型低很多。"

### ✅ 要这样汇报
> "RNAFM-PAM-noRun 是 BL5-v4-PAM 的 remove-Run 消融。它的低分是**预期内的**，因为去掉 Run 后 AUPRC 从 0.53 跌到 0.28，跌幅（0.25）是去掉 PAM 跌幅（0.03）的 **8.8 倍**。这个数量级差异直接说明：BL5-v4-PAM 的强性能主要依赖 RNA-FM 与 LearnableRun 的互补，PAM 是次要增量。"

### 进一步可补充的论点
1. **"Run 是主干，PAM 是调味"**：如果导师追问架构设计，可以说 PAM encoder（16-dim）在全模型中只占极小参数比例，它的作用不是独立决策，而是给 RNA-FM+Run 的联合表示补充局部 PAM 上下文。
2. **"消融矩阵已完成"**：2×2 组件矩阵（Run×PAM）的四格已有三格，对角关系清晰，支持多视角融合的合理性。
3. **"与 BL0b 的对照"**：RNAFM-PAM-noRun（0.2765）与 BL0b（0.2957）接近，说明在缺少 Run 的情况下，加 PAM 并不能显著提升纯 RNA-FM 的表现，再次验证 Run 的必要性。

---

## 6. 与之前实验的关系

| 实验 | 角色 | 本实验如何引用它 |
|:---|:---|:---|
| BL5-v4-PAM | **基准** | "全模型 AUPRC=0.5313，是我们的消融基准" |
| BL5-v4-NoPAM-control | remove-PAM 消融 | "去掉 PAM 掉 0.029，去掉 Run 掉 0.255，两者差距 8.8 倍" |
| BL5-v4-LearnableRun-only | LearnableRun 单视角 | "LearnableRun-only=0.2949，RNAFM-PAM-noRun=0.2765，两者接近，说明单独能力相当" |
| BL5-v4-PAM-only | PAM 单视角 | "PAM-only=0.0592，证明 PAM 不能单独决策" |
| BL0b-on-BL5split | RNA-FM 单视角 | "BL0b=0.2957，RNAFM-PAM-noRun=0.2765，接近，说明缺 Run 时加 PAM 帮助不大" |

---

## 7. 对下一步的启示

本消融结果支持以下结论，可作为 BL5→BL6 递进的设计依据：

1. **Run/LearnableRun 必须保留**：任何后续模型（BL5-1~3、BL6）都不应移除 Run 视角
2. **PAM 是可选增量，非核心**：如果计算资源受限，可以先做 RNA-FM+Run，再逐步加入 PAM
3. **RNA-FM 与 Run 的交互方式是关键**：既然两者单独能力相当、组合后质变，那么 BL5/BL6 的融合设计（Cross-Attn / Gated Fusion）应该优先优化 **RNA-FM ↔ Run** 的交互效率，而非 RNA-FM ↔ PAM
4. **Early stopping 对 RNA-FM-heavy 配置更敏感**：本实验 best_epoch=1，而全模型 best_epoch=4，说明 Run 视角同时起到了**正则化作用**，延缓了 RNA-FM 的过拟合

---

## 8. AGENTS.md 合规声明

本解读基于以下已验证的实验数据：
- `results/bl5_v4_rnafm_pam_norun_control/summary.json`（best.pt test 评估）
- `results/bl5_v4_rnafm_pam_norun_control/epoch_metrics.csv`（10 epoch 完整曲线）
- `results/bl5_v4_rnafm_pam_norun_control/test_predictions.csv`（PAM 分层分析）

所有对比数据均来自同一 `sgrna_safe` split（seed=42），确保消融可比性。
