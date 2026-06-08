# 28. AUPRC / Threshold / Precision 解疑实录

> 记录时间：2026-06-05
> 参与者：用户（提问者）+ Kimi Code CLI（解答者）
> 背景：BL6-1 训练完成后，用户对 AUPRC 指标产生系列困惑

---

## 问题起点

BL6-1 跑完后，用户看到报告里的指标：

```
AUPRC = 0.5399
Precision = 0.8817
Recall = 0.3049
```

产生核心困惑：

> "AUPRC 为什么不是越高越好？"
> "如果 AUPRC 高但 Precision 低，是不是过拟合？"
> "AUPRC 到底怎么算的？"
> "threshold 是什么？"

---

## 第一轮：AUPRC 高 + Precision 低 ≠ 过拟合

**用户的理解：**
> AUPRC 高但 Precision 低 → 模型有问题 / 过拟合

**纠正：**
- 过拟合的信号是 train loss ↓ 但 val loss ↑
- AUPRC 高但 Precision 低的真正含义是：模型能把正样本大体排在前面，但"领先优势不够大"
- 正样本和负样本混在一起，区分度不足，不是过拟合

**关键比喻：**
> 1000 个人里有 3 个罪犯。
> 模型 A：罪犯概率 0.95/0.90/0.85，好人概率 0.01/0.02 → 三者都好
> 模型 B：罪犯概率 0.60/0.55/0.50，好人概率 0.59/0.58 → AUPRC 还行但 Precision 极低

---

## 第二轮：Baseline 到底是谁？

**用户的理解：**
> AUPRC 到不了 1，是不是模型很差？

**纠正：**
- AUPRC 理论上可以到 1.0（完美分类器）
- 到不了 1 的原因是数据本身有噪声、特征有天花板、标签有模糊性
- Baseline 有两个层面：
  1. 统计 baseline（random classifier）≈ positive_rate = 0.0032
  2. 项目 baseline（上一个最佳模型）= BL5-v4-PAM 0.5313
- BL6-1 的 0.54 相对于 random 提升了 169 倍，不是"差 46 分"

---

## 第三轮：AUPRC 的公式到底是什么？

**用户的困惑：**
> 如果数据里本来就有 label=0 和 label=1，AUPRC 还是靠近不了 1 吗？

**解答过程：**
- 用真实数据演示：最高负样本概率 0.717，最低正样本概率 0.001
- 这意味着有负样本排在了正样本前面
- AUPRC = 1.0 的要求是：所有正样本概率 > 所有负样本概率
- 只要有 1 个负样本超越了 1 个正样本，AUPRC 就不是 1.0

**关键认识：**
> AUPRC 到不了 1 ≠ 模型不好
> = 数据里有些位点特征太相似，谁来了都分不开

---

## 第四轮：Threshold 是什么？

**用户的困惑：**
> threshold 是从训练集里划分出来的吗？
> 是根据什么定的？

**纠正：**
- threshold 不是从训练集里"学"出来的
- threshold 是**使用模型的人自己定的及格线**
- 模型只输出概率（0~1），threshold 是你决定"多高的概率才算正"

**项目场景举例：**
```
模型给 10,000 个候选位点打分：
  off-target #1: 0.95
  off-target #2: 0.72
  off-target #3: 0.23
  ...

threshold = 0.5（保守）：只验证 #1、#2 → 省钱但可能漏掉
threshold = 0.2（激进）：验证更多 → 更全面但更贵
```

**关键认识：**
- threshold 是应用层面的选择，不是训练出来的
- 不同的 threshold 会导致不同的 Precision
- AUPRC 之所以重要，是因为它不依赖某一个 threshold

---

## 最终理解

用户的认识演进：

```
AUPRC 越高越好？
  → 要在正确的 split、baseline、稳定训练前提下才有意义

AUPRC 高 + Precision 低 = 过拟合？
  → 不是，是"正样本领先优势不够大"

AUPRC 为什么不能到 1？
  → 能到 1，但数据里有不可分的样本对

Threshold 是什么？
  → 你自己定的"多少分算及格"的及格线

Precision 和 AUPRC 的区别？
  → Precision 是某一行的结果，AUPRC 是所有行的综合
```

---

## 核心金句

> "AUPRC 是录像，Precision 是照片。"
>
> "Threshold 是你自己定的及格线，不是模型学出来的。"
>
> "0.54 离 1 有 0.46 的差距，但离 random（0.0032）已经走了 99.4% 的路。"
>
> "到不了 1 不是模型差 46 分，是数据里有些罪犯和好人长得实在太像了。"

---

## Image Prompt（学术报告风格画图提示词 — 带解释对话框版）

A five-panel educational comic-style scientific diagram in Nature/Cell journal figure aesthetic, illustrating the step-by-step pedagogical dialogue between a questioning researcher and an explaining mentor about machine learning metrics (AUPRC, Precision, Threshold). Each panel contains speech bubbles with Q&A text, explanatory annotations, and visual metaphors. Flat vector illustration, clean sans-serif typography, professional scientific color palette (navy #1a365d, amber #d69e2e, slate #64748b, white background).

Panel 1 - "The Question": Left side shows a researcher with a puzzled expression pointing at a screen displaying "AUPRC=0.54, Precision=0.88, Recall=0.30". A speech bubble from the researcher reads: "Why isn't AUPRC closer to 1.0? Is high AUPRC with low precision overfitting?" Right side shows a mentor figure. Background: scattered floating numbers in light mist. Caption below: "Initial confusion about metric interpretation."

Panel 2 - "The Analogy": The mentor holds up a visual analogy board showing two scenarios. Scenario A (top): "Criminals: 0.95, 0.90, 0.85 | Innocent: 0.01, 0.02" with green checkmarks. Scenario B (bottom): "Criminals: 0.60, 0.55 | Innocent: 0.59, 0.58" with a yellow warning sign. A speech bubble from mentor reads: "Not overfitting — insufficient separation. The positives rank above negatives but without clear margin." Arrows connect the probability bars to precision values. Caption: "Key insight: AUPRC measures ranking, not just single-threshold accuracy."

Panel 3 - "The Baseline": A dual-axis explanation diagram. Left: a tiny bar labeled "Random baseline = 0.0032 (positive rate)". Right: a tall bar labeled "Project anchor BL5-v4-PAM = 0.5313" with an even slightly taller bar "BL6-1 = 0.5399" on top. An arrow shows "169× improvement from random". A speech bubble reads: "0.54 is not '46 points away from perfect' — it's 169× better than random guessing." Small text box explains: "Gap to 1.0 = biological noise, not model failure." Caption: "Contextualizing AUPRC against appropriate baselines."

Panel 4 - "The Mechanics": A horizontal sorted probability plot. Gold dots (3,057) represent positives, gray dots (951,269) represent negatives, arranged along a probability axis 0→1. Several gray dots appear above the lowest gold dot, visually demonstrating overlap. Annotated callout: "Max negative prob = 0.717 > Min positive prob = 0.001". A formula box shows: "AUPRC = Σ(ΔRecall × Precision)". Speech bubble: "AUPRC=1.0 requires ALL positives above ALL negatives. One overlap breaks perfection." Caption: "Mathematical essence of AUPRC calculation."

Panel 5 - "The Threshold & Synthesis": Top half shows a horizontal slider labeled "Threshold" with three positions marked: "Strict (0.9) → Precision=1.0, Recall=0.05", "Medium (0.5) → Precision=0.5, Recall=0.5", "Lenient (0.1) → Precision=0.1, Recall=1.0". Arrows show the trade-off. Bottom half shows a final framework box: "AUROC↑ → AUPRC↑ → Precision↑ = Valid Model" with BL6-1 checked at each step. A concluding speech bubble reads: "Precision is a snapshot at one threshold. AUPRC is the entire video across all thresholds." Caption: "Threshold as a user-chosen cutoff, not a learned parameter."

Overall composition: Panels arranged in a 2-3 grid or vertical flow with connecting curved arrows between panels. Speech bubbles use rounded rectangles with thin borders. All text is legible and integrated into the illustration. No decorative anime elements, no fantasy, strictly educational scientific communication design. White background, subtle grid lines, clear visual hierarchy.

---

## 关联文档

- `25_BL6_based_on_BL5_v4_PAM_plan.md` — BL6 计划
- `26_BL6_1_PAM_Gated_Fusion_执行报告.md` — BL6-1 结果
- `27_AUPRC_AUROC_Precision_指标解读与全模型对照报告.md` — 指标详解