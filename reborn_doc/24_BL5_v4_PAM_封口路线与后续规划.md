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

### 1.2 Test Cohort 核验

- test_samples = 954,326
- test_positive (observed_positive) = 3,057
- test_negative (unobserved_candidate) = 951,269
- positive_rate = 0.003203 (~0.32%)
- 四模型共用同一 test set，`sample_index` 完全对齐。

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
- **Paired delta**：PAM vs Shuffle 时，Observed Positive median Δ = +0.1523，Unobserved Candidate median Δ = -0.1003——正确 PAM **选择性抬高 positive、压低 negative**，不是全员灌水。

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

---

#### 3.4 threshold / top-k operating point 表

**回答老师可能会问的**：
- 实际筛查时好不好用？
- 如果实验室预算只够测前 1000 个位点，能找回多少 true positive？

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

**关键检查项**：
- PAM vs NoPAM 在 top-1000 的 recall 差距
- Shuffle 在 top-10000 是否已经被 PAM 的 top-1000 超越
- enrichment_over_random 是否随 k 增加而单调下降（验证排序质量）

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

**待补充图件（按本文档 Phase 1 执行后生成）**：

| 图号 | 预期文件名 | 存放路径 | 核心信息 |
|:---:|:---|:---|:---|
| Fig 7 | `per_sgrna_delta_plot.png` | `results/figures/bl5_accuracy/` | per-sgRNA AUPRC 与 delta 分布 |
| Fig 8 | `per_pam_metric_plot.png` | `results/figures/bl5_accuracy/` | per-PAM motif AUPRC 对比 |
| Fig 9 | `bootstrap_auprc_ci.png` | `results/figures/bl5_accuracy/` | Bootstrap 95% CI |
| Fig 10 | `insilico_pam_perturbation_heatmap.png` | `results/figures/bl5_accuracy/` | PAM 扰动热力图 |
| Fig 11 | `insilico_seed_run_response.png` | `results/figures/bl5_accuracy/` | Seed / Run 扰动响应 |
| Fig 12 | `multi_seed_stability.png` | `results/figures/bl5_accuracy/` | 多 seed AUPRC error bar |

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
