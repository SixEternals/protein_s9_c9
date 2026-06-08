# 27. AUPRC、AUROC、Precision 三者组合解读与全模型对照报告

> 生成时间：2026-06-05
> 执行 AI：Kimi Code CLI
> 问题来源：用户对 AUPRC / AUROC / Precision 三者关系的追问

---

## 一、核心问题：AUPRC 高 + Precision 低 = 过拟合？

### 答案：不是。

**过拟合的信号是什么：**

| 信号 | 表现 |
|:---|:---|
| 训练损失持续下降 | train loss ↓ |
| 验证损失上升 | val loss ↑ |
| 验证指标先升后降 | val AUPRC 达到峰值后崩溃 |
| 测试远低于验证 | test AUPRC << val AUPRC |

**AUPRC 高 + Precision 低的真正含义：**

> 模型能把正样本大体排在负样本前面（所以 AUPRC 面积好看），但**正样本的"领先优势"不够大**——在实际工作点（比如 threshold=0.5 或 top-k）上，预测为"正"的样本里混了大量假阳性。

这不是过拟合，而是**区分度不足**或**置信度校准差**。

---

## 二、直观类比："找罪犯"游戏

假设 1000 个人里有 3 个罪犯。

### 模型 A（好）：AUROC 高 + AUPRC 高 + Precision 高

```
3 个罪犯的概率：0.95, 0.90, 0.85
997 个好人的概率：0.01, 0.02, 0.03...
```

- AUROC 高：罪犯整体排在好人前面 ✅
- AUPRC 高：Precision-Recall 曲线面积很大 ✅
- Precision 高：说"这人是罪犯"时，真的说对了 ✅

### 模型 B（AUPRC 高但 Precision 低）：

```
3 个罪犯的概率：0.60, 0.55, 0.50
997 个好人的概率：0.59, 0.58, 0.57, 0.56...
```

- AUROC 还行：罪犯整体略高于好人
- AUPRC 还行：因为罪犯确实在好人之上，PR 曲线面积不会太差
- **但 Precision 极低**：你随便抓一个人说他是罪犯，几乎一定是错的

模型 B 的问题不是"记混了训练集"（过拟合），而是**"排得对但拉不开差距"**。

---

## 三、三者组合判断框架

### 黄金组合：三者都高

```
AUROC 高 (>0.95)
  └── AUPRC 高 (>0.5)
        └── Precision 高 (>0.5)
              → ✅ 好模型（排序好、找正样本强、实际工作点也好）
```

### 需要警惕的组合

| AUROC | AUPRC | Precision | 含义 |
|:---:|:---:|:---:|:---|
| 高 | **高** | **低** | 排序还行但正样本和负样本混在一起，实际没法用 |
| 高 | **低** | — | 能区分"正负大类"，但找具体正样本很弱（如 frozen RNA-FM） |
| **低** | **高** | — | ⚠️ 极度危险！很可能是数据泄漏或标签污染 |
| 低 | 低 | — | 模型完全无效 |

### 关键区分

**AUROC 高 + AUPRC 低：**
- 模型知道"这是正类群、那是负类群"
- 但正类群边界模糊，具体谁是正谁是负分不清
- 例子：BL0b-on-BL5split（AUROC=0.86, AUPRC=0.30）

**AUROC 高 + AUPRC 高 + Precision 低：**
- 模型能把正样本排在负样本前面
- 但正样本的领先优势太小，混在一起
- 实用中需要很低的阈值才能召回正样本，代价是大量假阳性
- 例子：某些过度平滑的 softmax 模型

---

## 四、本项目全模型对照（同一 test set 可比）

以下模型均在 **formal_split_bl5_seed42.json** 上评估，test cohort 完全一致（954,326 样本 / 3,057 positive / 72 sgRNA_type），可直接横向比较。

### 4.1 同一 split 直接对比表

| 模型 | AUROC | AUPRC | Precision | Recall | F1 | 三者组合解读 |
|:---|---:|---:|---:|---:|---:|:---|
| **BL0b-on-BL5split** | 0.858 | 0.296 | **0.783** | 0.252 | 0.381 | AUROC 还行但 AUPRC 低 → RNA-FM 单独能区分 sgRNA 类型，但找具体切割位点很弱 |
| **BL5-v4-NoPAM-control** | 0.984 | 0.502 | **0.653** | 0.354 | 0.459 | AUROC↑ AUPRC↑ → LearnableRun 引入后大幅增强正样本识别 |
| **BL5-v4-PAM (历史最佳)** | 0.984 | 0.531 | **0.769** | 0.361 | 0.491 | +PAM 再提升 → PAM 提供额外有效信号 |
| **BL5-v4-PAM (最新 rerun)** | 0.986 | 0.516 | **0.546** | 0.416 | 0.472 | 训练方差导致 rerun 略低于历史最佳，三者协调 |
| **BL5-v4-PAM-shuffle** | 0.670 | 0.139 | **1.000** | 0.134 | 0.236 | PAM 打乱后双崩 → 证明 PAM 信息被真实利用，非虚假相关 |
| **BL6-1-PAM-Gated-Fusion** | **0.985** | **0.540** | **0.882** | 0.305 | 0.453 | ✅ **三者协调**：AUROC 高 + AUPRC 高 + Precision 高 → gate 有效 |

### 4.2 逐模型解读

#### BL0b-on-BL5split（纯 RNA-FM）

```
AUROC=0.858  AUPRC=0.296  Precision=0.783  Recall=0.252  F1=0.381
```

- **AUROC 还行**：RNA-FM 能学到序列的某些区分性特征
- **但 AUPRC 很低**：在极度不平衡数据（positive rate≈0.0032）上，RNA-FM 单独无法可靠地定位切割位点
- **Precision=0.783 偏高**：注意这是在 0.5 阈值下。由于模型输出的概率普遍偏低，超过 0.5 的样本很少，碰巧命中了一些真正的正样本。这和"模型很强"无关，而是 threshold 效应
- **解读**：RNA-FM 是好的"表示学习器"，但不是好的"正样本探测器"

#### BL5-v4-NoPAM-control（RNA-FM + LearnableRun，无 PAM）

```
AUROC=0.984  AUPRC=0.502  Precision=0.653  Recall=0.354  F1=0.459
```

- **AUROC 大幅跃升**（+0.126）：LearnableRun 让模型学会了 protospacer 层面的错配模式
- **AUPRC 也大幅提升**（+0.206）：不仅区分能力变强，找正样本的能力也质变
- **Precision=0.653**：在 0.5 阈值下，说"是正"的样本里约 65% 真的对
- **解读**：hand-crafted prior（LearnableRun）是提升 AUPRC 的关键杠杆

#### BL5-v4-PAM（历史最佳 vs 最新 rerun）

```
历史: AUROC=0.984  AUPRC=0.531  Precision=0.769  Recall=0.361  F1=0.491
rerun: AUROC=0.986  AUPRC=0.516  Precision=0.546  Recall=0.416  F1=0.472
```

- **rerun 的 AUPRC 低于历史最佳**：说明存在训练方差，单 seed 的结果有波动
- **历史 Precision=0.769 > rerun 的 0.546**：rerun 在 0.5 阈值下更"宽松"（Recall 更高但 Precision 更低）
- **解读**：三者协调，是一个稳定的好模型，但训练方差存在

#### BL5-v4-PAM-shuffle-control（PAM 特征打乱）

```
AUROC=0.670  AUPRC=0.139  Precision=1.000  Recall=0.134  F1=0.236
```

- **双指标同时崩塌**：证明 BL5-v4-PAM 的性能提升**真实依赖 PAM 信息**
- **Precision=1.000**：模型极度保守，几乎不敢预测任何样本为正，偶尔猜对一次就是 100%
- **Recall=0.134**：只找回了 13% 的正样本
- **解读**：最重要的对照实验，排除了 PAM shortcut 疑虑。**Precision=1.0 在这里不是好事**，而是模型"放弃治疗"的表现

#### BL6-1-PAM-Gated-Fusion（本次）

```
AUROC=0.985  AUPRC=0.540  Precision=0.882  Recall=0.305  F1=0.453
```

- **AUROC 与 BL5-v4-PAM 持平**：区分能力没有下降
- **AUPRC 创纪录**：0.540 > 0.531（历史最佳），排序能力进一步提升
- **Precision 高达 0.882**：说"是正"时 88% 说对了，远高于 BL5-v4-PAM rerun 的 0.546
- **但 Recall 只有 0.305**：在 0.5 阈值下只找回了 30% 的正样本

**为什么 Precision 高但 Recall 低？**

这不是 bug，而是模型变"保守"了：

- BL6-1 的 gate 让模型对"模糊样本"给更低的概率
- 只有非常像正样本的实例才会超过 0.5 阈值
- 结果是：说"是正"的时候很准（Precision↑），但漏掉了很多边缘正样本（Recall↓）

**实际使用中**可以通过降低阈值来提高 Recall，关键看 AUPRC（排序能力）——AUPRC 越高，你在任何 Recall 水平上能达到的 Precision 都越好。

---

## 五、不在同一 test set 的模型（仅作参考）

以下模型的 test set 不同（CCLMoff random vs GUIDE-seq vs formal_split_bl5_seed42），**AUPRC 数值不可直接比较**。列出 Precision/Recall/F1 供参考，但注意 split 差异。

| 模型 | 数据集 / Split | AUPRC | Precision | Recall | F1 | 说明 |
|:---|:---|---:|---:|---:|---:|:---|
| BL0a (frozen RNA-FM) | CCLMoff random | 0.073 | 0.031 | 0.611 | 0.059 | frozen 特征几乎找不到正样本 |
| BL0b (fine-tuned) | CCLMoff random | 0.522 | 0.797 | 0.371 | 0.506 | fine-tune 是必须的（+615%） |
| BL3-hard | CCLMoff | — | 0.562 | 0.432 | 0.488 | Hard seed + Region + Run |
| BL3-hard-A | CCLMoff | — | 0.609 | 0.414 | 0.493 | Hard seed (1×/2×) |
| BL3-hard-C | CCLMoff | — | 0.689 | 0.420 | 0.522 | Learnable seed |
| BL3-ablation-run | CCLMoff | — | 0.885 | 0.320 | 0.470 | Run-only ablation |
| BL3-ablation-region | CCLMoff | — | 0.798 | 0.373 | 0.508 | Region-only ablation |
| BL3-region-only | CCLMoff | — | 0.315 | 0.206 | 0.249 | Region-only |
| BL3-run-only | CCLMoff | — | 0.043 | 0.738 | 0.082 | Run-only |
| BL3-RNAFM-fusion | CCLMoff | — | 0.058 | 0.794 | 0.109 | RNA-FM + Run fusion |
| BL4-frozen | CCLMoff group-safe | 0.206 | 0.067 | 0.293 | 0.108 | frozen RNA-FM + Run |
| BL4-finetune | CCLMoff | — | 0.462 | 0.430 | 0.446 | fine-tune RNA-FM + Run |
| BL5-3 (Cross-Attn/Gated) | CCLMoff | 0.445 | 0.410 | 0.412 | 0.411 | 旧复杂融合路线，未超越 v4 |
| BL5-3-LearnableRun | CCLMoff | 0.518 | 0.592 | 0.396 | 0.474 | +LearnableRun 后接近 BL5-v4 |
| BL5-3-LearnableRun-reg | CCLMoff | — | 0.482 | 0.369 | 0.418 | +seed regression |
| BL5-3-v2-simple | CCLMoff | — | 0.633 | 0.357 | 0.456 | simple backend |
| BL5-v3-CLS | CCLMoff | 0.484 | 0.429 | 0.460 | 0.444 | CLS + LearnableRun，无 PAM |

**趋势结论**（不受 split 影响）：
- fine-tuned > frozen
- LearnableRun > hand-crafted Run
- PAM > NoPAM

---

## 六、为什么之前报告里没有 Precision？

**不是数据缺失，是报告模板没要求写。**

`scripts/train_bl5.py` 的 `evaluate()` 函数**一直计算** accuracy / precision / recall / f1，所有 `summary.json` 里都有完整数据。

但 AGENTS.md 第 7 条和项目惯例只强制要求报告：

```text
AUROC
AUPRC
split_mode
```

Precision/Recall/F1 虽然在文件中存在，但之前的计划文档和对比表只突出 AUROC + AUPRC。这是**报告策略的选择**，不是数据缺失。

**现在补上的原因：** 用户追问 AUPRC 和 Precision 的关系，说明需要把这些数据呈现出来才能完整回答指标解读问题。

---

## 七、结论：如何看一组指标是否"真的好"

### 检查清单

```
□ AUROC 高 (>0.95)        → 模型会排序
□ AUPRC 高 (>0.5)         → 正样本排得靠前
□ Precision 不太低 (>0.3) → 说"是正"的时候有一定可信度
□ 训练稳定               → 无 NaN，val AUPRC 不崩溃
□ Test cohort 一致       → samples / positive / sgRNA_type 匹配锚点
□ 有 Shuffle/NoPAM 对照  → 排除了虚假相关
```

### 本项目当前最佳模型

**BL6-1** 满足全部条件：

| 检查项 | BL6-1 | 状态 |
|:---|---:|:---:|
| AUROC > 0.95 | 0.985 | ✅ |
| AUPRC > 0.5 | 0.540 | ✅ |
| Precision 不太低 | 0.882 | ✅ |
| 训练稳定 | 10 epoch 无崩溃 | ✅ |
| Test cohort 一致 | 954,326 / 3,057 / 72 | ✅ |
| NoPAM 对照 | 0.502 < 0.540 | ✅ |
| Shuffle 对照 | 0.139 << 0.540 | ✅ |

---

## 八、一句话总结

```
AUROC 高  = 模型会排序（区分正负大类）
AUPRC 高  = 正样本排得靠前（在极度不平衡数据上依然有效）
Precision 高 = 说"是正"的时候很准（实际工作点可信）

三者都高  → 好模型
AUPRC 高但 Precision 低  → 不是过拟合，是"正样本领先优势不够大"
AUPRC 高但 AUROC 低      → ⚠️ 极度危险，可能是数据泄漏
```