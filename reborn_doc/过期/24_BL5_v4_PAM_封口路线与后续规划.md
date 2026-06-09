# BL5-v4-PAM 主结果封口路线与后续规划

> **文档性质**：项目路线决策文档  
> **核心决策**：当前不进入 BL6，不继续堆叠 BL5 新架构，优先将 BL5-v4-PAM 作为阶段主结果封口。  
> **依据来源**：BL0b / BL5-v4-NoPAM / BL5-v4-PAM / BL5-v4-PAM-shuffle-control 的完整实验证据链，以及 per-sgRNA、per-PAM、分层指标和 paired probability delta 分析。  
> **合规声明**：本文档遵守 AGENTS.md 约束，所有指标同时报告 AUROC 与 AUPRC，test 评估均基于 `best.pt` checkpoint。

---

## 0. 决策总览（一句话版）

**当前不要开 BL6。当前不要继续堆 BL5 新架构。优先封口 BL5-v4-PAM：**

第一优先级（不需要训练，直接增强面对老师提问的能力）：
1. per-sgRNA 分析
2. per-PAM motif 分析
3. bootstrap confidence interval
4. threshold / top-k operating point 表

第二优先级（小型 baseline，排除质疑）：
5. kNN / nearest-neighbor baseline
6. PAM-only baseline

第三优先级（解释性扰动）：
7. in-silico perturbation

第四优先级（训练稳定性）：
8. BL5-v4-PAM 多 seed 重复

**然后**视老师要求补 BL4-full 作为路线完整性 baseline。  
**最后**再考虑 BL6。

---

## 1. 当前科学现状（基于已完成的实验）

### 1.0 为什么 Accuracy 不能做主指标？为什么 AUROC 不够？

在讲实验结果之前，必须先理解一个关键背景：**我们的数据极度不平衡**。在 954,326 条 test 样本中，只有 3,057 个是 observed_positive（约 0.32%），其余 951,269 个都是 unobserved_candidate。

这意味着什么？意味着一个"傻子模型"——把全部样本都预测成 0（unobserved_candidate）——它的 **Accuracy 已经高达 99.68%**。所以如果你只报告 Accuracy，老师会觉得"这模型不是跟瞎猜差不多吗？"

那 AUROC 呢？AUROC 衡量的是"随机抽一个 positive 和一个 negative，模型把 positive 排在 negative 前面的概率"。在极度不平衡的数据上，AUROC 很容易虚高，因为 negative 样本实在太多了，模型只要能把大部分 negative 排在后面，AUROC 就会很好看。

所以我们需要 **AUPRC**（Area Under Precision-Recall Curve）。它问的是另一个更实际的问题：**在模型打分最高的那些候选位点里，有多少是真正的 observed_positive？** 这和实验室的真实场景完全对应——你不能测 95 万个位点，你只能测前 1000 个，你需要这前 1000 个里 true positive 越多越好。

![Concept Fig A: AUROC vs AUPRC 的核心区别](pic/concept_a_auroc_vs_auprc.png)

**这张图怎么读（小白版）**：

这张图用红蓝点告诉我们 AUROC 和 AUPRC 到底在问什么问题。

- **左图 AUROC：全局分离**。想象你把所有样本铺开，红点（observed_positive）应该整体偏上，蓝点（unobserved_candidate）应该整体偏下。AUROC 问的是："随机抓一个红点和蓝点，红点分数比蓝点高的概率是多少？" 这很好，但它是个"全局"指标，不care你具体 inspect 了哪些。
- **右图 AUPRC：top-ranked 实用性**。想象你把所有样本按模型打分从高到低排成一列。AUPRC 问的是："只看最前面的这一截（top predictions inspected），里面红点占多大比例？" 这才是实验室关心的——预算有限，只能测前 k 个，前 k 个里 true positive 的比例就是 Precision@k。

底部那句话最关键："Because observed_positive is rare, AUPRC is the primary metric for practical screening." 因为 positive 很稀有，所以 **AUPRC 才是实际筛查场景下的主指标**。这也是为什么我们整篇文档都把 AUPRC 放在比 AUROC 更重要的位置。

---

### 1.1 四模型核心指标（formal BL5 split, test set）

| 模型 | AUROC | AUPRC | 相对 BL0b AUPRC Δ |
|:---|---:|---:|---:|
| BL0b-on-BL5split | 0.8578 | 0.2957 | — |
| BL5-v4-NoPAM-control | 0.9841 | 0.5024 | **+0.2067** |
| **BL5-v4-PAM** | **0.9842** | **0.5313** | **+0.2356** |
| BL5-v4-PAM-shuffle-control | 0.6697 | 0.1389 | -0.1568 |

**关键数字**：
- `NoPAM − BL0b = +0.2067`：BL5-v4 框架（LearnableRun + concat MLP）本身的巨大增益。
- `PAM − NoPAM = +0.0289`：正确 PAM Encoder 的额外增益。
- `PAM − Shuffle = +0.3924`：正确 PAM vs 随机 PAM 的鸿沟。
- `Shuffle − BL0b = -0.1568`：Shuffle 控制甚至比纯 RNA-FM baseline 还差。

![Fig 1: 四模型 AUROC/AUPRC Leaderboard](../results/figures/bl5_accuracy/fig1_leaderboard_auprc_auroc.png)

![Fig 2: Precision-Recall 曲线](../results/figures/bl5_accuracy/fig2_precision_recall_curves.png)

> 标准学术对比图。绿色（PAM, AUPRC=0.5313）全面包络橙色（NoPAM, 0.5024）；红色（Shuffle, 0.1389）在 Recall≈0.12 处垂直掉落至 0，然后贴着 X 轴走，是典型的模型失效曲线。

![Concept Fig D: 四模型输入信号与 PAM 消融设计](pic/concept_d_pam_ablation_design.png)

**这张图怎么读（小白版）**：

这张图用"输入信号 → 输出指标"的方式，把四个模型之间的差异讲得一清二楚。你可以把它理解成四个不同配置的"黑箱实验"，唯一变量就是模型能"看到"哪些信息。

- **第一行 BL0b（蓝色）**：模型只看 **RNA-FM CLS**（RNA-FM 预训练模型输出的 640 维向量），没有 Run block，没有 PAM block。这是最简单的 RNA-FM baseline，AUPRC = 0.2957。你可以把它理解为"只有序列上下文，没有任何工程先验"。
- **第二行 NoPAM（橙色）**：在 BL0b 基础上加了 **LearnableRun**（positions 1-20 的连续错配状态编码），但仍然没有 PAM Encoder。AUPRC 跳到 0.5024，增益 +0.2067。这说明：**Run 特征（连续错配状态）本身就是极强的信号**，它让模型从"序列上下文"升级到了"知道哪里连续错配"。
- **第三行 PAM（绿色）**：在 NoPAM 基础上再加 **Correct PAM Encoder**（positions 21-23）。AUPRC 再跳到 0.5313，增益 +0.0289。这说明：**正确的 PAM 信息提供了额外的、虽然不大但是真实的增益**。注意这里写的不是"PAM 本身"，而是"Correct PAM"——因为下一个控制证明，如果 PAM 错了，结果会崩。
- **第四行 Shuffle（红色）**：和 PAM 完全一样的架构，但 PAM 输入被**随机打乱**（Sample A 拿到了 Sample B 的 PAM）。AUPRC 暴跌到 0.1389，比 BL0b 还差。这说明：**PAM Encoder 不是来凑参数的，它极度依赖"正确的 PAM 与样本对应关系"**。如果对应关系错了，模型会被严重误导。

底部三个图标是这份报告最重要的"质量控制声明"：
- 🔒 same formal split（四个模型用完全一样的 train/val/test 划分）
- 🏷️ same labels（标签完全一致）
- 📄 best.pt evaluation（全部用验证集 AUPRC 最佳时的 checkpoint 评估，不用 last checkpoint）

### 1.2 Test Cohort 核验

- test_samples = 954,326
- test_positive (observed_positive) = 3,057
- test_negative (unobserved_candidate) = 951,269
- positive_rate = 0.003203 (~0.32%)
- 四模型共用同一 test set，`sample_index` 完全对齐。

![Concept Fig C: 模型理想行为——按 label 区分概率](pic/concept_c_mean_prob_by_label.png)

**这张图怎么读（小白版）**：

这张图告诉我们，一个"好模型"应该长什么样。

- **左图（红点：observed_positive）**：这些是在实验里真的被检测到有切割的位点。理想情况下，模型应该给它们较高的预测概率。图中红点的平均概率大约在 0.7~0.9 之间，分布相对集中，说明模型对 true positive 有自信。
- **右图（蓝点：unobserved_candidate）**：这些是实验里没有检测到切割的位点（注意：label=0 **不代表安全**，只是"没被检测到"）。理想情况下，模型应该给它们较低的预测概率。图中蓝点大量挤在 0 附近，说明模型对 negative 是"不确定/不看好"的。
- **中间的 Separation gap（分离间隙）**：左图红点的均值（~0.8）和右图蓝点的均值（~0.05）之间存在巨大鸿沟。这个鸿沟越大，模型区分能力越强。

底部还有一句重要的话："label=0 means unobserved_candidate, not verified safe." 这对应 AGENTS.md 的术语规范——我们不用 "negative" 或 "safe"，只用 "unobserved_candidate"。因为没被检测到切割，可能只是实验灵敏度不够，不代表这个位点绝对安全。

### 1.3 已有的证据链

当前实验已经形成了**从 baseline → 框架增益 → PAM 特异性 → 机制解释**的完整闭环：

```
BL0b (0.296)
    ↓  +0.207
NoPAM (0.502)  ← BL5-v4 框架增益（LearnableRun + MLP）
    ↓  +0.029
PAM (0.531)    ← 正确 PAM 额外增益
    ↑
Shuffle (0.139) ← 打乱 PAM 后崩溃，证明增益依赖正确对应关系
```

**分层证据**：
- **NGG-only**（86% 样本，positive rate 0.29%）：Shuffle AUPRC ≈ 0.004（几乎失效），PAM AUPRC = 0.356（主战场）。
- **non-NGG-only**（14% 样本，positive rate 0.53%）：所有模型都不错，Shuffle AUPRC = 0.605。

![Fig 5: NGG / non-NGG 分层指标](../results/figures/bl5_accuracy/fig5_stratified_metrics.png)

![Concept Fig E: 按 PAM 子集的分层对比阅读指南](pic/concept_e_stratified_pam_subset.png)

**这张图怎么读（小白版）**：

这张图回答了老师最可能问的一个问题："PAM 的提升是不是主要来自 non-NGG 的 shortcut？"

我们先把 test set 切成三块，然后分别看四模型的 AUPRC：

- **左图 All test（全部样本）**：整体情况和 Fig 1 一致——PAM（0.531）> NoPAM（0.502）> BL0b（0.296）> Shuffle（0.139）。
- **中图 NGG-only（只保留 PAM 为 NGG 的样本，占 86%）**：这是数据的大头，也是 hardest case（positive rate 只有 0.29%）。关键发现：
  - BL0b 在 NGG-only 上几乎失效（AUPRC = 0.114），说明纯 RNA-FM 在 NGG 场景下非常吃力。
  - NoPAM 提升到 0.324，PAM 进一步提升到 0.356——**PAM 在 NGG-only 上仍然有明确的增益**，这不是 non-NGG shortcut。
  - Shuffle 崩到 0.004，几乎为 0。
- **右图 non-NGG-only（只保留 non-NGG 样本，占 14%）**：这是相对容易的子集（positive rate 0.53%）。所有模型表现都不错，Shuffle 也有 0.605。这说明 non-NGG 位点本身可能更容易判别，但**它们不是 PAM 模型优势的主要来源**（因为所有模型在这里都不错）。

底部的问题直接点出了这张图的核心叙事："does PAM still help inside NGG-only, or is the gain dominated by non-NGG shortcut?" 答案是：**PAM 在 NGG-only 上仍然显著有帮助（0.324 → 0.356），提升不是被 non-NGG shortcut 主导的。**

右侧小字提醒："Subsets are evaluation slices, not different model types." 意思是这三组图是**把同一个模型在不同子集上重新评估**，而不是训练了三个不同的模型。这是分层分析的关键——保证公平对比。

- **Paired delta**：PAM vs Shuffle 时，Observed Positive median Δ = +0.1523，Unobserved Candidate median Δ = -0.1003——正确 PAM **选择性抬高 positive、压低 negative**，不是全员灌水。

![Fig 6: PAM vs Shuffle 的 Paired Probability Delta](../results/figures/bl5_accuracy/fig6_paired_delta.png)

> 左图：PAM vs NoPAM 的 delta 很小（median ≈ -0.01），说明正确 PAM 是在强 baseline 上的微调。  
> 右图：PAM vs Shuffle 的 delta 巨大，Observed Positive median = +0.1523，Unobserved Candidate median = -0.1003，证明正确 PAM 是选择性增强而非全员灌水。

---

## 2. 为什么现在不进 BL6 / 不继续堆 BL5 新架构

### 2.1 为什么不进 BL6

BL6 的定义是"Cross-View Attention + Gated Fusion + 多层交互 + 不确定性感知"。在当前节点推进 BL6 有以下风险：

| 风险 | 说明 |
|:---|:---|
| **证据不足** | 当前尚未验证"简单拼接（BL4-full）是否优于单一视角"。BL5-3 才是 Cross-Attn + Gated，BL6 在此基础上再加多层交互，属于"在尚未夯实的地基上盖楼"。 |
| **解释链断裂** | 老师追问"BL6 比 BL5-3 好多少？"时，如果 BL5-3 本身还没有在 CCLMoff formal split 上跑完，无法回答。 |
| **资源错配** | BL6 参数量更大、训练更慢、调参空间更广。在 BL5-v4-PAM 的封口工作尚未完成时，进 BL6 会导致"最强模型的故事讲不清楚，新模型的故事又讲不扎实"。 |
| **路线违规** | AGENTS.md 第 18 章明确禁止跳步："跳过 BL4 直接做 BL5"和"跳过 BL3-gradient 直接做 BL4-full"都是禁止行为。BL6 是 BL5-3 之上的增强，必须先有稳定的 BL5-3/BL5-v4 主结果。 |

### 2.2 为什么不继续堆 BL5 新架构

BL5 系列的核心科学问题是"动态融合是否优于简单拼接"。当前 BL5-v4-PAM 已经用 **simple concat** 达到了 AUPRC=0.5313。如果继续在同一代码基上尝试 BL5-1 (Cross-Attn)、BL5-2 (Gated)、BL5-3 (Full)，可能出现：

- **收益递减**：从 0.5313 提升到 0.55+ 的边际收益，远不如先把 0.5313 **讲稳、讲透**。
- **叙事混乱**：组会上老师问"你现在的主结果到底是哪个？"，如果同时有 v4-PAM、v5-CrossAttn、v6-...，主线不清晰。
- **实验债务**：每增加一个子版本，就需要配套的 NoPAM/shuffle/per-sgRNA 分析，工作量指数增长。

**结论**：BL5-v4-PAM 的 simple concat 已经很强。当前最急迫的不是继续追新模型，而是把现在这个结果从"看起来赢了"推进到"经得住追问"。

![Fig 3: AUPRC 贡献瀑布图](../results/figures/bl5_accuracy/fig3_auprc_contribution_waterfall.png)

> 最推荐组会主图：绿柱 vs 红柱的对比极具冲击力。NoPAM 比 BL0b 高 +0.2067，PAM 比 NoPAM 再高 +0.0289，而 Shuffle 比 NoPAM 低 -0.3635。说明 PAM 的价值极度依赖"正确的 PAM-样本对应"，而不是单纯多了几层参数。

![Concept Fig F: Correct PAM vs No PAM vs Shuffled PAM 架构对比](pic/concept_f_correct_vs_shuffled_pam.png)

**这张图怎么读（小白版）**：

这张图用三列流程图，把"PAM 到底在干什么"讲得连非技术背景的人都能看懂。

- **左列 NoPAM（橙色）**：模型收到两个输入——RNA-FM sequence context（序列上下文）和 Run features（连续错配状态，positions 1-20）。PAM 这一块用虚线框标着"PAM not provided"，意思是模型**不知道** positions 21-23 是什么。它只能凭前 20 个位置的序列和错配模式来猜。结果 AUPRC = 0.502。
- **中列 Correct PAM（绿色）**：和 NoPAM 完全一样的前两个输入，但多了一个实线框的 **Correct PAM**（positions 21-23）。这个 PAM 是**真实的、和当前样本一一对应的**。模型现在知道了"这个 sgRNA 的 PAM 是 NGG"，结果 AUPRC 提升到 0.531。
- **右列 Shuffled PAM（红色）**：架构和中列完全一样，也收到了 PAM 输入，但**这个 PAM 是从另一个随机样本偷来的**。例如 Sample A 的序列配上了 Sample B 的 PAM。PAM 分布的统计特征没变（还是那些 AGG/TGG/GGG），但**对应关系被打破了**。结果 AUPRC 暴跌到 0.139。

底部结论只有一句话，但分量极重："**Correct correspondence matters.**" 正确的对应关系才是最重要的。PAM Encoder 不是来堆参数的，它是来提供"这个 PAM 和这个 sgRNA 是一对"的生物学信号的。信号对了，模型受益；信号错了，模型被严重误导。

![Concept Fig G: BL5-v4 PAM 贡献拆解与解读](pic/concept_g_pam_contribution_decomposition.png)

**这张图怎么读（小白版）**：

这张图是整篇文档的"核心叙事图"，适合放在组会 PPT 的最后一页或结论页。它把从 BL0b 到 PAM 的完整递进关系画成了流程。

- **起点 BL0b**：只用一个蓝色方块表示"RNA-FM only"，AUPRC = 0.2957。这是我们的起跑线。
- **第一步 +LearnableRun + v4 classifier → NoPAM**：加了橙色模块（Run 特征 + 分类器），AUPRC 跳到 0.5024，增益 +0.2067。右侧解读第一条："NoPAM already gives a strong v4 framework." 这说明 BL5-v4 框架本身（LearnableRun + MLP）就是巨大的进步来源。
- **第二步 +Correct PAM Encoder → PAM**：加了绿色模块（Correct PAM），AUPRC 再跳到 0.5313，增益 +0.0289。右侧解读第二条："Correct PAM adds a modest but real gain." 这个增益不大，但是真实存在的。
- **红色下箭头 Shuffle PAM correspondence → Shuffle control**：从 PAM 往下拐了一个红箭头，标注 "-0.3924"，指向 Shuffle control（AUPRC = 0.1389）。右侧解读第三条："Shuffled PAM severely misleads the model." 打乱的 PAM 不仅没帮助，反而严重误导了模型。

底部黄色警告框也很重要："Disclose PAM shortcut risk and specify PAM definition: positions 21-23." 这提醒我们：在汇报时必须明确告诉老师，PAM 是指 positions 21-23，而且我们要主动披露"PAM shortcut 风险"——即模型可能过度依赖 PAM 而忽视序列上下文。这种主动披露反而会增强可信度。

---

## 3. Phase 1：BL5-v4-PAM 封口（按优先级分四层）

### 第一优先级：不需要训练的验证分析

> 这一步最划算，直接增强你面对老师提问的能力。所有分析直接基于已有的 test_predictions.csv 和 summary.json，不需要 GPU 训练。

---

#### 3.1 per-sgRNA 分析

**回答老师可能会问的**：
- 是不是只在少数几个 sgRNA 上赢？
- 有没有某些 sgRNA 特别拖后腿？
- BL5-v4-PAM 相比 NoPAM 的提升是不是普遍存在？

**做什么**：
- 在 72 个 test sgRNA 上分别计算 AUPRC、AUROC、Precision、Recall。
- 计算 **PAM − NoPAM** 的 per-sgRNA AUPRC delta。
- 计算 **PAM − Shuffle** 的 per-sgRNA AUPRC delta。
- 记录每个 sgRNA_type 的 samples 数、observed_positive 数、positive_rate。
- 画出分布：per-sgRNA AUPRC histogram、PAM−NoPAM delta scatter（按样本数加权）、worst-5 / best-5 sgRNA 表格。

**为什么重要**：
- 整体 AUPRC 高不够，最好能说明提升不是集中在一两个 sgRNA 上。
- 如果 72 个 sgRNA 中有 60 个 PAM > NoPAM，叙事是"普遍提升"；如果只有 15 个，叙事需要调整。

**预期产出**：
- `results/per_sgrna_metrics_with_pam.csv`
- `results/per_sgrna_metrics_with_pam.md`
- `results/figures/bl5_accuracy/per_sgrna_delta_plot.png/pdf`

**关键检查项**：
- PAM > NoPAM 的 sgRNA 占比
- worst-5 sgRNA 的 AUPRC、样本数、positive count
- per-sgRNA AUPRC 与样本数 / positive rate 的相关性

![Fig 7: per-sgRNA AUPRC 与 delta 分布](../results/figures/bl5_accuracy/per_sgrna_delta_plot.png)

> 【待插入】左图：72 个 test sgRNA 的 AUPRC 分布直方图；右图：PAM − NoPAM per-sgRNA AUPRC delta 的 scatter plot（点大小按 sgRNA 样本数加权）。需要能看出提升是否普遍分布在多数 sgRNA 上，而非集中在少数几个。

---

#### 3.2 per-PAM motif 分析

**回答老师可能会问的**：
- PAM Encoder 是不是只靠 NGG / non-NGG shortcut？
- 哪些 PAM motif 贡献最大？
- AGG / TGG / GGG / CGG 内部表现是否一致？

**做什么**：
- 把 test set 按 PAM motif 分组（AGG, TGG, GGG, CGG, non-NGG, rare PAM）。
- 每组计算：samples、observed_positive、positive_ratio、mean probability、AUPRC、AUROC。
- 对比四模型（BL0b / NoPAM / PAM / Shuffle）在各 PAM 组上的 AUPRC。
- 画出 PAM-group 柱状图或点图。

**为什么重要**：
- 当前已有 NGG-only / non-NGG-only 分析，但 per-PAM 更细，更能应对"是不是 shortcut"的追问。
- 如果 PAM 模型在 AGG/TGG/GGG/CGG 上表现一致，说明学到了"NGG 家族"的共性；如果只有 NGG 高、其他 NGG-like 都低，说明可能是 shortcut。

**预期产出**：
- `results/per_pam_metrics_with_shuffle.csv`
- `results/per_pam_metrics_with_shuffle.md`
- `results/figures/bl5_accuracy/per_pam_metric_plot.png/pdf`

**关键检查项**：
- NGG 家族（AGG/TGG/GGG/CGG）内部 AUPRC 是否一致
- Shuffle 控制在各 PAM 组上的崩溃程度是否均匀
- PAM-only 分析（见 3.6）可以与 per-PAM 结果交叉验证

![Fig 8: per-PAM motif AUPRC 对比](../results/figures/bl5_accuracy/per_pam_metric_plot.png)

> 【待插入】X 轴：PAM 分组（AGG, TGG, GGG, CGG, non-NGG, rare）；Y 轴：AUPRC；四组柱状图分别对应 BL0b / NoPAM / PAM / Shuffle。需要能直观看出 PAM 模型在各 NGG 子类型上是否表现一致，以及 Shuffle 控制是否均匀崩溃。

---

#### 3.3 bootstrap confidence interval

**回答老师可能会问的**：
- 0.5313 比 0.5024 高 0.0289，这个稳不稳？
- 你的结果是撞出来的还是可重复的？

**做什么**：
- 对 test set 做 **stratified bootstrap resampling**（保持 positive/negative 比例），重复 1,000 次。
- 每次计算四模型的 AUROC 和 AUPRC。
- 计算 95% CI：
  - BL5-v4-PAM AUPRC: [L, U]
  - BL5-v4-NoPAM AUPRC: [L, U]
- 计算 **paired bootstrap**：
  - PAM − NoPAM AUPRC delta 的 95% CI
  - PAM − Shuffle AUPRC delta 的 95% CI
- 如果 CI 不跨 0，说明差异统计显著。

**为什么重要**：
- 单次 test AUPRC 是点估计。老师问"稳不稳"时，必须有 CI。
- +0.0289 看起来小，但如果 95% CI 是 [0.015, 0.042] 且不跨 0，就可以说"虽然绝对增量不大，但是统计显著的、可重复的"。

**预期产出**：
- `results/bootstrap_bl5_pam_vs_controls.csv`
- `results/bootstrap_bl5_pam_vs_controls.md`
- `results/figures/bl5_accuracy/bootstrap_auprc_ci.png/pdf`

**关键检查项**：
- PAM AUPRC 95% CI 宽度（越窄越稳）
- PAM − NoPAM delta 95% CI 是否包含 0
- PAM − Shuffle delta 95% CI 是否包含 0（预期不包含）

![Fig 9: Bootstrap 95% CI 分布](../results/figures/bl5_accuracy/bootstrap_auprc_ci.png)

> 【待插入】左图：1000 次 bootstrap 得到的 PAM / NoPAM / Shuffle AUPRC 分布直方图，标注 95% CI 边界；右图：PAM−NoPAM 和 PAM−Shuffle paired delta 的 bootstrap 分布，标注 95% CI 和是否跨 0。最关键的是右图——如果 PAM−NoPAM delta 的 95% CI 完全在 0 右侧，+0.0289 就是统计显著的。

---

#### 3.4 threshold / top-k operating point 表

**回答老师可能会问的**：
- 实际筛查时好不好用？
- 如果实验室预算只够测前 1000 个位点，能找回多少 true positive？

![Concept Fig B: Top-k 评估流程与意义](pic/concept_b_topk_evaluation.png)

**这张图怎么读（小白版）**：

这张图用四步流程解释了"Top-k 评估"到底是什么意思，以及为什么它特别贴合实验室的真实场景。

想象你是一个做 wet lab 的研究生，老板给了你一笔预算，说"你只能测 1000 个候选位点，你给我挑最值得测的"。你怎么挑？Top-k 评估回答的就是这个问题。

- **Step 1: Model scores all candidates（模型给所有候选位点打分）**
  模型对 test set 里的 95 万个位点逐一输出一个概率（例如 0.98, 0.91, 0.72...）。这个概率代表"模型认为这个位点是 true off-target 的信心"。
- **Step 2: Sort by predicted risk（按预测风险排序）**
  把所有位点按概率从高到低排成一列。注意红点（observed_positive）应该尽量排在前面，蓝点（unobserved_candidate）尽量排在后面。模型的好坏就看它能不能做到这一点。
- **Step 3: Inspect top-k（只看前 k 个）**
  由于预算有限，你只能 inspect 前 k 个。图中的 "top-k" 括号就是把最前面的 k 个位点圈出来。
- **Step 4: Count recovered observed_positive（统计找回了多少 true positive）**
  在前 k 个里面，你数有多少个是红点（真的 observed_positive）。这引出两个关键指标：
  - **Precision@k** = 前 k 个里的红点个数 / k。代表"你测了 k 个，命中率是多少"。
  - **Recall@k** = 前 k 个里的红点个数 / 全部红点总数。代表"你把所有 true positive 找回来百分之多少"。

底部那句话是核心："**Top-k directly matches limited validation budget: if we can test only k sites, which model gives the best shortlist?**" Top-k 直接对应有限的验证预算：如果我们只能测 k 个位点，哪个模型能给出最好的候选清单？

这就是为什么 AUPRC 虽然是个综合指标，但 Top-k 更能让老师感同身受——因为老师也关心"我的钱花在刀刃上了吗？"

**做什么**：
- 基于已有的 test_predictions.csv，为每个模型生成 top-k operating point 表。
- k 取值：100, 500, 1000, 5000, 10000, 50000, 100000。
- 每个 (model, k) 报告：
  - positives_recovered（找回的 observed positive 数）
  - recall@k（占全部 3057 的比例）
  - precision@k（positives_recovered / k）
  - enrichment_over_random = precision@k / positive_rate（相比随机筛选富集了多少倍）

**为什么重要**：
- 这种数字比单独 AUPRC 更容易打动老师。
- 例如：PAM 在 top 1000 找回 924 / 3057，recall ≈ 30.23%，precision ≈ 92.4%，enrichment ≈ 288×。这意味着实验室只需要测 0.1% 的候选位点，就能抓回 30% 的 true positives。

**预期产出**：
- 直接写入 `results/figures/bl5_accuracy/topk_enrichment_summary.csv`（已生成，需补充 enrichment_over_random 列）
- `results/figures/bl5_accuracy/topk_operating_table.md`（PPT-ready 表格）

![Fig 4: Top-k Enrichment 分析](../results/figures/bl5_accuracy/fig4_topk_enrichment.png)

> 左图：前 1% 位点（k≈9,500）PAM/NoPAM 即可召回 ~70% observed positives，而 BL0b 仅 ~35%，Shuffle 仅 ~14%。  
> 右图：前 100 个位点 Precision≈100%，前 1000 个位点 Precision≈92%。这意味着实验室只需测 0.1% 的候选位点，就能以 >90% 的精度抓回 30% 的 true positives。

**关键检查项**：
- PAM vs NoPAM 在 top-1000 的 recall 差距
- Shuffle 在 top-10000 是否已经被 PAM 的 top-1000 超越
- enrichment_over_random 是否随 k 增加而单调下降（验证排序质量）

---

#### 3.4.1 Precision-Recall Trade-off 与实验室场景对照（基于真实 test 数据）

> 本节直接回答老师最关心的问题：**"如果实验室预算只够测 k 个位点，能找回多少 true positive？命中率是多少？漏掉多少？和随机抽相比强多少倍？"** 所有数字均来自 `results/figures/bl5_accuracy/topk_enrichment_summary.csv`，test set n=954,326，total observed_positive=3,057。

**随机基线**：由于 positive_rate = 3,057 / 954,326 ≈ 0.3203%，随机抽取 k 个位点，预期只命中 k × 0.3203% 个 true positive。例如随机抽 1,000 个，预期只中 **3.2 个**。

| 场景 | k | PAM pos_recovered | PAM Recall | PAM Precision | BL0b Recall | BL0b Precision | 富集倍数 (PAM) |
|:---|---:|---:|---:|---:|---:|---:|---:|
| 高精度筛查 | 1,000 | 924 / 3,057 | 30.23% | **92.40%** | 25.19% | 77.00% | **288.5×** |
| 中等规模 | 5,000 | 1,787 / 3,057 | 58.46% | **35.74%** | 31.17% | 19.06% | **111.6×** |
| 较大规模 | 10,000 | 2,128 / 3,057 | 69.61% | **21.28%** | 34.94% | 10.68% | **66.4×** |
| 高召回筛查 | 50,000 | 2,858 / 3,057 | 93.49% | **5.72%** | 51.16% | 3.13% | **17.8×** |
| 接近全检 | 100,000 | 2,963 / 3,057 | 96.93% | **2.96%** | 61.43% | 1.88% | **9.2×** |
| 随机基线 | 1,000 | ~3 / 3,057 | 0.32% | 0.32% | 0.32% | 0.32% | 1.0× |

**逐项解读（把读者当小白）**：

- **高精度筛查（k=1,000）**：模型从 95 万个位点中挑出最可能的前 1,000 个。这 1,000 个里面，有 **924 个是真的 observed_positive**，命中率（Precision）高达 **92.4%**。但代价是只找回了全部 3,057 个 positive 中的 **30.2%**，还有约 2,133 个 true positive 被漏掉了。
  - 对比 BL0b：同样测 1,000 个，BL0b 只能找回 770 个，命中率 77%，**PAM 比 BL0b 多找回 154 个 true positive，命中率高出 15.4 个百分点**。
  - 对比随机：随机抽 1,000 个只能中 3.2 个，**PAM 的富集倍数是随机的 288.5 倍**。

- **中等规模（k=5,000）**：把筛查范围扩大到前 5,000 个，找回的 true positive 增加到 1,787 个（Recall 58.5%），但命中率下降到 35.7%。这意味着每测 3 个位点，大约 1 个是对的、2 个是白测的。对于预算较宽裕的实验室，这个 operating point 可能是"性价比"最优的——找回了过半的 true positive，而假阳性成本尚可接受。

- **高召回筛查（k=50,000）**：如果实验室的目标是把几乎所有 true positive 都找回来（例如临床前安全性评估），需要 inspect 前 50,000 个位点。此时 Recall 达到 **93.5%**，但 Precision 暴跌到 **5.7%**——每测 20 个位点，只有 1 个是对的，其余 19 个都是 unobserved_candidate。这就是老师说的"预测很多但只有少数对"的极端情况。
  - 注意：即便如此，**PAM 的 5.7% 仍然是随机基线（0.32%）的 17.8 倍**，模型远非无用，只是 operating point 选得太宽了。

- **接近全检（k=100,000）**：inspect 前 10 万个位点，找回 2,963 个（Recall 96.9%），Precision 只有 2.96%。此时已经接近"把大半 test set 筛一遍"，失去了排序筛选的意义。

**核心结论**：

> **不存在一个 k 值能同时满足 Precision > 90% 且 Recall > 90%**。在 top-1,000 处，模型非常"精"（Precision=92.4%）但不够"全"（Recall=30.2%）；在 top-50,000 处，模型非常"全"（Recall=93.5%）但不够"精"（Precision=5.7%）。实验室必须根据自身的验证预算和假阳性容忍度，主动选择 operating point。

**汇报话术（可直接用）**：

> "老师，我们的模型在 top-1,000  operating point 下，Precision 达到 92.4%，也就是说如果实验室只测前 1,000 个候选位点，命中率超过九成。但代价是只找回了 30% 的全部 true positive，会漏掉大约七成。如果实验室的预算更宽松，可以 inspect 前 5,000 个，此时命中率约 36%，但找回率提升到 58%。不存在一个阈值能同时满足高命中率和高找回率，这是所有排序模型的内在 trade-off。"

---

#### 3.4.2 False Discovery Rate (FDR) 分析

> 本节回答："如果我测了 k 个位点，预期有多少是白测的（假阳性）？"

FDR（False Discovery Rate）= 1 − Precision。它和 Precision 是一枚硬币的两面。

| k | PAM Precision | PAM FDR | 含义 |
|---:|---:|---:|:---|
| 1,000 | 92.40% | **7.6%** | 测 1,000 个，约 76 个是假阳性 |
| 5,000 | 35.74% | **64.3%** | 测 5,000 个，约 3,213 个是假阳性 |
| 10,000 | 21.28% | **78.7%** | 测 10,000 个，约 7,872 个是假阳性 |
| 50,000 | 5.72% | **94.3%** | 测 50,000 个，约 47,142 个是假阳性 |

**关键洞察**：
- 在 top-1,000 处，FDR 只有 7.6%，这意味着实验室每花 100 份测序钱，只有约 7~8 份是"冤枉钱"。
- 但如果把 k 放大到 50,000，FDR 飙升到 94.3%，几乎每测 20 个位点就有 19 个是白测的。
- **这不是模型变烂了，而是 operating point 选得太宽了**。模型的排序能力没变，只是你 inspect 的范围越大，尾部 noise 越多。

**对比 BL0b**：
- BL0b 在 top-1,000 处的 FDR = 23.0%（Precision = 77.0%），是 PAM 的 **3 倍**。
- 也就是说，用 BL0b 测 1,000 个位点，预期浪费 230 份测序钱；用 PAM 只浪费 76 份。**PAM 把 wet lab 的无效支出降低了约 67%**。

---

#### 3.4.3 模型互补性：PAM ∪ BL0b 并集分析

> 本节回答："PAM 漏掉的那 70% true positive，能不能用另一个模型（比如 BL0b）找回来？两个模型取并集会怎么样？"

**做法**：对 test set 中的每个样本，取 PAM 和 BL0b 预测概率的较大值作为联合分数，然后重新排序取 top-k。计算并集找回的 observed_positive 数。

| k | PAM alone Recall | BL0b alone Recall | PAM ∪ BL0b Recall | 互补增益 |
|---:|---:|---:|---:|---:|
| 1,000 | 30.23% | 25.19% | **32.68%** | +2.45% |
| 5,000 | 58.46% | 31.24% | **62.09%** | +3.63% |
| 10,000 | 69.61% | 34.94% | **73.05%** | +3.43% |
| 50,000 | 93.49% | 51.13% | **95.03%** | +1.54% |
| 100,000 | 96.93% | 61.14% | **97.81%** | +0.88% |

**关键洞察**：
- 互补增益只有 **2~3 个百分点**，说明 BL0b 能找到的 true positive 绝大多数已经被 PAM 覆盖了。
- PAM 并不是"漏掉了 BL0b 能发现的独特信号"，而是**PAM 本身就包含了 BL0b 的判别信息并做了增强**。
- 这也印证了文档的核心叙事：BL5-v4-PAM 是当前最强单一模型，late fusion（并集）带来的边际收益很小，不值得为了这 2~3% 的增益维护两个模型管线。

---

#### 3.4.4 per-sgRNA 极端案例分析（Worst-case & Best-case）

> 本节回答："模型在最差的 sgRNA 上表现如何？有没有某个 sgRNA 上一个 true positive 都没找到？"

**做法**：把 test set 按 sgRNA 分组（共 72 个 test sgRNA），在每个 sgRNA 内部按 PAM 模型概率排序取 top-1,000，计算该 sgRNA 的 recall@1000。

**PAM 模型 per-sgRNA recall@1000**：
- 均值：**97.66%**
- 中位数：**100.00%**
- 最低值：**75.00%**
- **没有任何一个 sgRNA 的 recall 为 0%**

**Worst-5 sgRNA（按 recall@1000 排序）**：

| sgRNA_type | n_samples | n_positive | top-1,000 找回 | recall |
|:---|---:|---:|---:|---:|
| ATAGGAGAAGATGATGTATANGG | 18,331 | 12 | 9 | **75.0%** |
| GGGGGTTCCAGGGCCTGTCTNGG | 14,111 | 537 | 426 | **79.3%** |
| GCCTCTCCAGCCAGGGGCTGNGG | 22,805 | 395 | 315 | **79.7%** |
| GGCTGAGGAAGCTGAGGAGGNGG | 45,848 | 26 | 22 | **84.6%** |
| GCTGTGTTTGCGTCTCTCCCNGG | 10,971 | 66 | 56 | **84.8%** |

**关键洞察**：
- 即使是最差的 sgRNA（ATAGGAGAAGATGATGTATANGG），recall@1000 也达到 **75%**（12 个 true positive 找回了 9 个）。
- 超过半数的 sgRNA（中位数 = 100%）在 top-1,000 内就能找回 **全部** true positive。
- 这说明 PAM 模型的优势**不是集中在少数几个 sgRNA 上**，而是普遍分布在 72 个 test sgRNA 中。

**对比 BL0b 的 per-sgRNA recall@1000**：
- 均值：**41.59%**
- 中位数：**37.50%**
- 最低值：**0.00%**（存在完全找不回的 sgRNA）
- 最高值：**100.00%**

**对比结论**：BL0b 在某些 sgRNA 上完全失效（recall=0%），而 PAM 模型没有任何一个 sgRNA 是"死穴"。这进一步支持了 BL5-v4-PAM 作为阶段主结果的可靠性。

**汇报话术（可直接用）**：

> "老师，您担心模型可能只在少数 sgRNA 上表现好，这是个非常关键的问题。我们计算了 72 个 test sgRNA 各自的 recall@1000，PAM 模型的中位数是 100%——意味着超过一半的 sgRNA 在前 1,000 个预测里就找回全部 true positive。最差的 sgRNA recall 也有 75%，没有任何一个 sgRNA 是完全失效的。相比之下，纯 RNA-FM baseline（BL0b）在某些 sgRNA 上 recall 直接为 0。这说明 PAM 模型的优势是普遍的，不是集中在少数几个 sgRNA 上的。"

---

### 第二优先级：做小型 baseline，不急着大训练

> 这一步用来排除"模型只是记住了训练集"或"模型只看 PAM"的质疑。训练成本极低（kNN 无训练，PAM-only 是 tiny MLP）。

---

#### 3.5 kNN / nearest-neighbor baseline

**回答老师可能会问的**：
- 模型是不是只是记住了和训练集相似的样本？
- 换一个简单的非参数方法，能不能达到差不多的效果？

**做什么**：
- 从训练集提取特征：RNA-FM CLS embedding（640-d）+ LearnableRun（或简化为 off-target 序列的 one-hot / mismatch profile）。
- 对 test 样本，在 train 集中找 k 个最近邻（k=5, 10, 50；距离度量 cosine / euclidean）。
- 预测概率 = 近邻中 positive 的比例，或近邻平均 read / label。
- 在 test set 上评估 AUROC / AUPRC。

**为什么重要**：
- 如果 kNN 明显弱于 BL5-v4-PAM（例如 kNN AUPRC ≈ 0.25，而 PAM = 0.53），你就可以说："模型不是简单查相似题，而是学到了超越局部相似性的判别模式。"
- 如果 kNN 也达到 0.45+，说明特征空间本身已经高度结构化，此时应强调"特征工程（RNA-FM + Run + PAM）的价值大于 MLP 架构"。

**预期产出**：
- `results/knn_baseline_predictions.csv`
- `results/knn_baseline_summary.json`
- `results/knn_baseline_report.md`

**关键检查项**：
- kNN AUPRC vs BL0b / NoPAM / PAM 的相对位置
- k=5 vs k=50 的性能差异（是否过平滑）
- 使用不同特征组合（仅 RNA-FM / RNA-FM+Run / RNA-FM+Run+PAM）的 kNN 性能对比

![Fig 13: kNN baseline 对比](../results/figures/bl5_accuracy/knn_baseline_comparison.png)

> 【待插入】X 轴：模型（kNN-k5 / kNN-k10 / kNN-k50 / BL0b / NoPAM / PAM）；Y 轴：AUPRC。需要能看出 kNN 显著低于 BL5-v4-PAM，证明模型不是简单查相似题。也可以画两张并排的 bar chart（左图 AUROC，右图 AUPRC）。

---

#### 3.6 PAM-only baseline

**回答老师可能会问的**：
- 只靠 PAM 本身能不能达到很高性能？
- 你的模型是不是其实只在看 PAM，其他特征都是摆设？

**做什么**：
- 构建一个极简模型：PAM one-hot 或 PAM category embedding → tiny MLP（或直接 logistic regression）。
- **不使用 RNA-FM，不使用 Run，不使用 off-target 序列上下文。**
- 在相同的 formal BL5 split 上训练和评估。

**为什么重要**：
- 如果 PAM-only 很低（例如 AUPRC < 0.10），说明 BL5-v4-PAM 不是只看 PAM，PAM 只是整个证据链中的一环。
- 如果 PAM-only 也很高（例如 AUPRC > 0.30），那就说明 PAM shortcut 风险确实存在，解释时要更谨慎。这不是坏事，反而能让你的分析更可信。

**预期产出**：
- `results/pam_only_baseline_predictions.csv`
- `results/pam_only_baseline_summary.json`
- `results/pam_only_baseline_report.md`

**关键检查项**：
- PAM-only AUPRC 与 Shuffle 控制的对比（Shuffle 也有 PAM 输入，但被打乱）
- PAM-only 在 NGG vs non-NGG 上的表现差异
- 与 per-PAM 分析（3.2）交叉验证：如果 PAM-only 在各 PAM 组上的分布与 full model 一致，说明 PAM 是主导信号

![Fig 14: PAM-only baseline 对比](../results/figures/bl5_accuracy/pam_only_baseline_comparison.png)

> 【待插入】X 轴：模型（PAM-only / BL0b / NoPAM / PAM / Shuffle）；Y 轴：AUPRC。预期 PAM-only 远低于 PAM（例如 <0.15），证明序列上下文（RNA-FM + Run）提供了 PAM 之外的判别力。如果 PAM-only 反而很高，需要承认 PAM shortcut 风险。

---

### 第三优先级：解释性扰动

> 这是"模型是否符合生物直觉"的证据，比再堆一个 BL5-v5 更有解释价值。

---

#### 3.7 in-silico perturbation

**回答老师可能会问的**：
- 模型到底对 PAM、seed、连续错配有没有合理响应？
- 模型是符合生物学直觉，还是只是记住了统计关联？

**做什么**：
设计三类扰动，基于已训练好的 BL5-v4-PAM 模型做前向传播（不需要重新训练）：

**扰动 A：固定 on_seq/off_seq 1-20，只替换 PAM**
- 对 test set 中每个样本，把 PAM 分别改成 AGG/TGG/GGG/CGG/NGG/NNN 等。
- 记录 probability 变化。
- 输出 PAM sensitivity score。

**扰动 B：固定 PAM，只扰动 seed 区 mismatch**
- 在 positions 16-20（hard seed）上系统性地引入/移除 mismatch。
- 观察 probability 是否随 seed mismatch 增加而上升（符合生物学：seed mismatch 越多，脱靶风险越高）。

**扰动 C：固定 mismatch 数量，改变是否连续 run**
- 构造 synthetic 样本：相同 mismatch 数，但一种是 isolated mismatch，一种是 run2/run3+。
- 观察模型是否对连续错配赋予更高权重（符合 ConMismatch9 的先验假设）。

**为什么重要**：
- 当前证据是"Shuffle 控制崩了"（否定性证据：PAM 不能乱）。扰动提供**肯定性证据**：PAM Encoder 具体对哪种信号敏感、对哪种不敏感。
- 如果 seed mismatch 增加 → probability 单调上升，说明模型学到了合理的生物学规律；如果杂乱无章，说明模型可能只是在 memorizing。

**预期产出**：
- `results/insilico_perturbation/pam_sensitivity.csv`
- `results/insilico_perturbation/seed_mismatch_response.csv`
- `results/insilico_perturbation/run_state_response.csv`
- `results/figures/bl5_accuracy/insilico_pam_perturbation_heatmap.png/pdf`
- `results/figures/bl5_accuracy/insilico_seed_run_response.png/pdf`

**关键检查项**：
- PAM 扰动是否单调：NGG 家族 > non-NGG > NNN（或类似合理序）
- Seed mismatch 响应是否单调递增
- Run3+ 的 probability 是否高于 isolated mismatch（相同总 mismatch 数下）

![Fig 10: PAM 扰动热力图](../results/figures/bl5_accuracy/insilico_pam_perturbation_heatmap.png)

> 【待插入】X 轴：原始 PAM；Y 轴：扰动后 PAM；颜色：平均 probability 变化。对角线（原始=扰动后）应为 0，NGG→NGG 附近应为深蓝（高概率），NGG→NNN 应为深红（概率下降）。

![Fig 11: Seed / Run 扰动响应](../results/figures/bl5_accuracy/insilico_seed_run_response.png)

> 【待插入】左图：固定 PAM，seed mismatch 数从 0→5 时 probability 的响应曲线（预期单调上升）；右图：固定总 mismatch 数，isolated vs run2 vs run3+ 的 probability 对比（预期 run3+ > run2 > isolated）。

---

### 第四优先级：训练稳定性

> 这一步需要 GPU，但很重要。现有数据已经提示训练波动：原始 PAM AUPRC=0.5313，两卡重跑 AUPRC=0.5161，差了 0.0152。

---

#### 3.8 BL5-v4-PAM 多 seed 重复

**回答老师可能会问的**：
- 你的结果是撞出来的还是稳出来的？
- 换种子还能复现吗？

**做什么**：
- 用 **3 个不同的随机种子**（seed 42, 43, 44）重新训练 BL5-v4-PAM。
- 保持 config、split、硬件环境完全相同。
- 记录每次的：test AUROC、AUPRC、best epoch、训练时间、最终 loss。
- 计算 mean ± std，画 error bar 图。

**为什么重要**：
- 当前已有两个数据点：原始 run（0.5313）和 2GPU 重跑（0.5161），差值 0.0152。不算崩，但说明存在训练波动。
- 如果 3 个 seed 的 AUPRC 分别是 0.531, 0.528, 0.535（std < 0.004），结果非常稳健。
- 如果是 0.531, 0.516, 0.550（std > 0.015），说明模型对初始化敏感，需要调优（降低学习率、增加 early stopping、调 focal loss gamma）后才能作为"主结果"。

**预期产出**：
- `results/bl5_v4_pam_multi_seed/summary.csv`
- `results/bl5_v4_pam_multi_seed/seed42_summary.json`
- `results/bl5_v4_pam_multi_seed/seed43_summary.json`
- `results/bl5_v4_pam_multi_seed/seed44_summary.json`
- `results/figures/bl5_accuracy/multi_seed_stability.png/pdf`

**关键检查项**：
- 3 seeds AUPRC mean ± std
- AUPRC std < 0.01（通过标准）；0.01~0.02（可接受但需说明）；> 0.02（不稳定，需调优）
- best epoch 是否一致（如果 seed 43 在 epoch 3 就停，seed 42 在 epoch 9，说明早停策略过敏感）

![Fig 12: 多 seed 稳定性 error bar](../results/figures/bl5_accuracy/multi_seed_stability.png)

> 【待插入】X 轴：模型（BL0b / NoPAM / PAM / Shuffle）；Y 轴：AUPRC；每个模型 3 个 seed 的点 + mean error bar。PAM 的 3 个点应密集聚集（std 小），Shuffle 的点也应稳定地低。如果 PAM 的 error bar 很大，说明主结果不稳定，需要调优后再封口。

---

## 4. Phase 2：路线完整性补充（BL4-full）

### 4.1 为什么补 BL4-full

BL4-full 的定义是：**RNA-FM + Region + Run 三者拼接**。当前项目中：
- BL4-Run-only 已跑（AUPRC = 0.206，CCLMoff group-safe），但这是旧结果，且 split 可能不是 formal BL5 split。
- **BL4-full（Region + Run + RNA-FM）从未在 formal BL5 split 上实现和评估**。

BL4-full 的科学价值是：验证"显式生物先验（Region + Run）能否增强 frozen RNA-FM"。如果 BL4-full 的 AUPRC 显著低于 BL5-v4-PAM，说明：
- BL5-v4 的 LearnableRun + PAM 架构优于简单的 Region+Run+FM 拼接；或者
- fine-tuned RNA-FM 比 frozen RNA-FM 强很多（但 BL0b 已经是 fine-tuned，所以这个对照需要仔细设计）。

### 4.2 实施建议（低优先级，视老师要求）

- **如果老师问"你有没有试过把三种特征都拼起来？"** → 需要补 BL4-full。
- **如果老师没问** → 可以暂时不补，先把 BL5-v4-PAM 封口工作做完。
- BL4-full 的实现相对简单（复用现有 Region Encoder、Run Encoder、RNA-FM Encoder，concat 后接 MLP），预计 1~2 天可完成训练和评估。

---

## 5. Phase 3：BL6 的时机与前提

### 5.1 BL6 的准入条件

以下**全部满足**后，方可启动 BL6：

1. ✅ BL5-v4-PAM 封口工作完成（Phase 1 八项全部做完，文档齐备）。
2. ✅ BL4-full 已完成或已明确不需要（老师未质疑路线完整性）。
3. ✅ BL5-v4-PAM 的多 seed 稳定性验证通过（AUPRC std < 0.01）。
4. ✅ 老师在组会上认可"BL5-v4-PAM 是阶段主结果"，并主动提出"下一步能不能做更复杂的交互？"
5. ✅ 有充分的 GPU 资源和开发时间（BL6 的训练和调参成本显著高于 BL5-v4）。

### 5.2 BL6 的风险控制

如果启动 BL6，必须：
- 保留 BL5-v4-PAM 的 checkpoint 和 config 作为强 baseline。
- BL6 的每个子版本（例如单层 Cross-Attn、多层 Cross-Attn、不确定性感知）都要有独立的 NoPAM / shuffle 控制。
- 不跳过 BL5-3（Cross-Attn + Gated），BL6 必须在 BL5-3 的代码基上开发。

---

## 6. 推荐时间线（以 2 周为一个冲刺周期）

| 周次 | 任务 | 优先级 | 预估时间 | 是否需要 GPU |
|:---:|:---|:---:|:---:|:---:|
| W1 | 3.1 per-sgRNA + 3.2 per-PAM | P0 | 2~3 天 | ❌ |
| W1 | 3.3 Bootstrap CI + 3.4 top-k 表 | P0 | 1~2 天 | ❌ |
| W2 | 3.5 kNN baseline + 3.6 PAM-only | P1 | 2~3 天 | ❌（kNN 无训练）/ ⚠️（PAM-only 小训练）|
| W2~W3 | 3.7 in-silico perturbation | P2 | 2~3 天 | ❌（前向传播 only）|
| W3~W4 | 3.8 多 seed 稳定性（seed 42/43/44） | P0 | 4~5 天 | ✅ |
| W4~W5 | 组会汇报、根据反馈调整 | P0 | 2~3 天 | — |
| W5+ | 视老师要求补 BL4-full | P2 | 2~3 天 | ✅ |
| W7+ | 满足 5.1 条件后，启动 BL6 | P3 | — | ✅ |

**总封口周期**：约 4~5 周（含 GPU 训练等待时间）。

---

## 7. 风险与应对

| 风险 | 可能性 | 影响 | 应对 |
|:---|:---:|:---:|:---|
| per-sgRNA AUPRC 波动大 | 中 | 高 | 汇报时强调"整体 AUPRC 稳健，个别 sgRNA 因样本稀疏导致方差大"，并提供 sgRNA-level bootstrap CI。 |
| PAM − NoPAM delta 的 Bootstrap CI 跨 0 | 低 | 高 | 若 CI 包含 0，说明 +0.0289 可能不显著。此时需增加训练数据量、或调整 PAM Encoder 结构、或承认"PAM 贡献较小但方向一致"。 |
| kNN 接近 MLP 性能 | 低 | 中 | 若 kNN AUPRC ≈ 0.50，说明特征空间高度结构化，应强调"特征工程（RNA-FM + Run + PAM）比 MLP 架构更重要"。 |
| PAM-only baseline 也很高 | 中 | 中 | 若 PAM-only AUPRC > 0.30，承认 PAM 是强信号，但强调"完整模型仍显著优于 PAM-only，说明序列上下文提供了额外判别力"。 |
| 多 seed 不稳定（std > 0.02） | 中 | 高 | 优先调优（降低学习率、增加 early stopping、调 focal loss gamma），而非直接进 BL6。 |
| 老师要求立刻做 BL6 | 中 | 高 | 用本文档作为论据："BL5-v4-PAM 的证据链已完整，但封口工作尚未完成。建议先花 2~3 周把主结果讲稳，再进 BL6，否则 BL6 的增益无法与稳定的 BL5-v4 对比。" |

---

## 8. 附录：关键图件索引

| 图号 | 文件名 | 存放路径 | 核心信息 |
|:---:|:---|:---|:---|
| Fig 1 | `fig1_leaderboard_auprc_auroc.png` | `reborn_doc/pic/` | 四模型 AUROC/AUPRC 总览 |
| Fig 2 | `fig2_precision_recall_curves.png` | `reborn_doc/pic/` | PR 曲线标准学术对比 |
| Fig 3 | `fig3_auprc_contribution_waterfall.png` | `reborn_doc/pic/` | **最推荐主图**：AUPRC 增量瀑布 |
| Fig 4 | `fig4_topk_enrichment.png` | `reborn_doc/pic/` | Top-k 召回与 Precision，回答落地价值 |
| Fig 5 | `fig5_stratified_metrics.png` | `reborn_doc/pic/` | NGG / non-NGG 分层指标 |
| Fig 6 | `fig6_paired_delta.png` | `reborn_doc/pic/` | PAM vs Shuffle 的 paired delta，机制证据 |

**待补充图件（本文档 Phase 1 执行后生成，当前为占位符）**：

| 图号 | 预期文件名 | 存放路径 | 状态 | 核心信息 |
|:---:|:---|:---|:---:|:---|
| Fig 7 | `per_sgrna_delta_plot.png` | `results/figures/bl5_accuracy/` | 🔄 待画 | per-sgRNA AUPRC 与 delta 分布 |
| Fig 8 | `per_pam_metric_plot.png` | `results/figures/bl5_accuracy/` | 🔄 待画 | per-PAM motif AUPRC 对比 |
| Fig 9 | `bootstrap_auprc_ci.png` | `results/figures/bl5_accuracy/` | 🔄 待画 | Bootstrap 95% CI |
| Fig 10 | `insilico_pam_perturbation_heatmap.png` | `results/figures/bl5_accuracy/` | 🔄 待画 | PAM 扰动热力图 |
| Fig 11 | `insilico_seed_run_response.png` | `results/figures/bl5_accuracy/` | 🔄 待画 | Seed / Run 扰动响应 |
| Fig 12 | `multi_seed_stability.png` | `results/figures/bl5_accuracy/` | 🔄 待画 | 多 seed AUPRC error bar |
| Fig 13 | `knn_baseline_comparison.png` | `results/figures/bl5_accuracy/` | 🔄 待画 | kNN baseline 对比 |
| Fig 14 | `pam_only_baseline_comparison.png` | `results/figures/bl5_accuracy/` | 🔄 待画 | PAM-only baseline 对比 |

> **说明**：Fig 1~6 已由现有实验数据生成并插入正文。Fig 7~14 需要基于后续分析数据绘制。请将图片放入 `results/figures/bl5_accuracy/` 目录下，文档中的相对路径引用会自动生效。

---

## 9. 结论

当前项目处于"**最强模型已跑出，但故事尚未讲稳**"的关键节点。BL5-v4-PAM（AUPRC=0.5313）已经是 formal BL5 split 上的阶段冠军，且 NoPAM / shuffle / NGG 分层 / paired analysis 形成了完整的证据链。

**最优先行动**：不是继续追模型复杂度，而是把现在这个结果从"看起来赢了"推进到"经得住追问"。

**具体而言**：
1. 先做第一优先级（per-sgRNA、per-PAM、bootstrap CI、top-k 表）——不需要训练，直接增强答辩能力。
2. 再做第二优先级（kNN、PAM-only）——排除质疑，夯实可信度。
3. 接着做第三优先级（in-silico perturbation）——提供机制解释。
4. 最后做第四优先级（多 seed 重复）——验证训练稳定性。
5. 视老师反馈补 BL4-full。
6. 全部满足后再考虑 BL6。

> **中文决策版**：
> 当前不要开 BL6。当前不要继续堆 BL5 新架构。优先封口 BL5-v4-PAM：先做不需要训练的验证分析（per-sgRNA、per-PAM、bootstrap CI、top-k 表），再做小型 baseline（kNN、PAM-only），再做解释性扰动，最后做多 seed 稳定性。然后视老师要求补 BL4-full。最后再考虑 BL6。
