# 当前优先级快照（2026-06-09）

> 🎯 **本文档是 `reborn_doc/` 目录下最新、最权威的项目状态入口。**
> 所有 AI 助手（Kimi / Codex / Claude）在承接新任务前，应首先阅读本文档以了解当前优先级和进行中的工作。
>
> ⚠️ **禁止引用过期文档**：`reborn_doc/过期/` 路径下的所有文件均为历史归档，信息已部分或全部过时，**不可作为当前事实依据**。
>
> 🌿 **当前 BL5 泛化/消融分支**：`feat/bl5-generalization`。BL5 泛化/消融相关任务必须在该分支执行；BL6 相关文件不得混入本分支的 BL5 提交范围。
>
> 下面的旧记录保留为历史上下文；如果和本节冲突，以本节为准。

## A. 已封口事项

| 事项 | 状态 | 产物 / 结论 |
|:---|:---:|:---|
| BL5-v4-PAM 最小组件消融矩阵 | ✅ 已完成 | RNA-FM / LearnableRun / PAM 单组件、双组件、全模型、NoPAM、PAM shuffle 均已有 formal split 结果；总结见 `reborn_doc/38_BL5_v4_PAM_组件消融总报告.md`。 |
| PAM-holdout feasibility audit | ✅ 已完成 | `results/bl5_generalization/pam_holdout_feasibility/`；feasible PAM = `AGG, CGG, GAG, GGG, TGG`。 |
| External dataset feasibility audit | ✅ 已完成 | `results/bl5_generalization/external_dataset_feasibility/`；`ready_for_strict_external_eval=0`；`SITE/K562` 仅为 provenance-required limited candidates，不能写成正式 external eval。 |
| AGG strict PAM holdout | ✅ 已完成 | AGG test_H 上 NoPAM 显著优于 PAM；ΔAUPRC(NoPAM−PAM)=`+0.176219`，说明 PAM encoder 在 AGG unseen PAM 上负向。 |
| TGG strict PAM holdout | ✅ 已完成 | TGG test_H 上 PAM 显著优于 NoPAM；ΔAUPRC(NoPAM−PAM)=`-0.052940`，说明 AGG 结论不能外推到所有 NGG PAM。 |
| GAG strict PAM holdout | ✅ 已完成 | GAG 是唯一 feasible non-NGG exploratory subset；full test_H 上 PAM vs NoPAM 差异不显著，且存在 per-sgRNA label-composition confounding，不能作为 non-NGG 泛化强证据。 |
| AGG/TGG/GAG paired bootstrap | ✅ 已完成 | AGG/TGG 显著且方向相反；GAG `n_bootstrap=10,000` 后 CI 跨 0。正式口径：PAM encoder 的 strict cross-PAM 行为是 motif-specific / subset-dependent。 |

## B. 当前正在进行

| 优先级 | 任务 | Owner | 状态 | 验收口径 |
|:---:|:---|:---:|:---:|:---|
| P0 | BL5 泛化/消融提交分 scope push | 另一个 Kimi | 🔄 进行中 | 只提交 `feat/bl5-generalization` 上 BL5 泛化/消融相关文件；不要 `git add .`；不要混入 BL6 configs/run/report、data/reference、checkpoint、test_predictions、大型 results 产物或文档重整删除。 |

当前不要改动提交中的 staging/push 工作，除非用户明确要求接手。需要检查结果时只读审计。

## C. 下一步优先级

| 优先级 | 任务 | 条件 | 说明 |
|:---:|:---|:---|:---|
| P1 | `test_seenPAM` sanity eval for AGG/TGG/GAG holdout models | BL5 泛化/消融提交不冲突时 | 不重新训练；对 6 个已训练 holdout 模型做 eval-only，评估各自 `test_seenPAM`，确认 holdout 模型在 seen-PAM formal-test subset 上是否整体正常。 |
| P2 | `CGG` strict PAM holdout | P1 完成且解释链无流程问题 | CGG 是 feasible NGG candidate，test_H 约 46K / 186 observed_positive；用于 NGG 内部第三个 motif 复验。 |
| P3 | `GGG` strict PAM holdout | P1 或 CGG 后 | GGG 是高支持 NGG candidate，test_H 约 203K / 716 observed_positive；训练成本更高，但可补齐 NGG 高支持证据。 |
| P4 | SITE/K562 provenance audit / limited external check | 仅作为附录 / future work | 先确认原始来源、label semantics、candidate generation；当前不能写成 ready external benchmark。 |

## D. 当前禁止误写

- **不要**把 external feasibility audit 写成 cross-dataset model evaluation。
- **不要**把 `SITE/K562` 写成 ready external benchmark。
- **不要**把 `test_20_samples.csv` 写成正式 external eval（仅为 smoke test）。
- **不要**把 GAG full test_H AUPRC≈0.99 写成 non-NGG 泛化很强；GAG 有 per-sgRNA label-composition confounding，只能作为 exploratory evidence。
- **不要**把 AGG/TGG/GAG 写成“三种模式已确认”；稳妥口径是 AGG/TGG 显著且方向相反，GAG 未检测到显著差异但有 composition confounding。
- **不要**把 `test_seenPAM` sanity eval 写成新训练；它应是 eval-only / prediction export / metrics audit。
- `PAM_original` 必须提取自 `off_seq[20:23]`，禁止使用 `off_seq[-3:]`。
- `label=0` 只能写作 `unobserved_candidate`，不能写作 verified safe site。

---

# 历史待办与背景记录

  1. 先把 BL6-1 核验包补齐

  这一步不需要重新训练，优先级最高：

  BL6-1 vs BL5-v4-PAM PR curve
  BL6-1 vs BL5-v4-PAM top-k operating table
  BL6-1 vs BL5-v4-PAM paired probability delta
  BL6-1 NGG-only / non-NGG-only stratified metrics
  BL6-1 per-PAM motif metrics
  BL6-1 per-sgRNA metrics
  BL6-1 bootstrap CI

  尤其是：

  BL6-1 - BL5-v4-PAM AUPRC paired bootstrap CI

  如果 CI 不跨 0，再说 BL6-1 的提升比较稳。

  2. 必须做 gate audit

  BL6-1 是 gated fusion，所以只报 AUROC/AUPRC 不够。要回答：

  gate 到底学了什么？
  有没有 collapse 到某一个 view？
  observed_positive 和 unobserved_candidate 的 gate 权重是否不同？
  NGG / non-NGG 的 gate 权重是否不同？
  top-k 样本里 PAM gate 是否异常偏高？

  理想输出：

  results/bl6_1_pam_gated_fusion/gate_audit.csv
  results/bl6_1_pam_gated_fusion/gate_audit.md
  results/figures/bl6_1/gate_weight_distribution.png

  如果现在 test_predictions.csv 没有 gate 权重，那下一步可以用 best.pt 跑一次
  evaluation/export，把 gate weights 导出来。这不算新训练，但对解释 BL6-1 很关键。

  3. 修正报告里的两个硬伤

  需要让 Kimi 或模型侧修：

  report.md / experiments.csv notes 里写的 "Cross-Attn + Softmax Gate" 是错模板。
  真实机制应写 "PAM-Gated Fusion on BL5-v4-PAM backbone"。

  还有：

  "Test 集全部为 NGG PAM" 改成 "test set contains both NGG and non-NGG PAM; canonical
  PAM distribution matches BL5-v4-PAM."

  这两个不修，后面汇报容易被老师抓住。

  4. 再做 BL6-1 seed repeat

  我建议至少再跑两个 seed：

  BL6-1 seed 43
  BL6-1 seed 44

  目标不是追最高分，而是得到：

  BL6-1 AUPRC mean ± std
  BL6-1 vs BL5-v4-PAM delta mean ± std

  如果三次 BL6-1 都在 0.53+，那基本可以正式把 BL6-1 升级为当前新主模型。

  如果只有这一跑高，其他 seed 掉到 0.51-0.52，那就只能说：

  > BL6-1 是 promising single-run improvement，但稳定性不足。
## 出的任务大纲

# 2
bl4 bl5需要补什么

优先级如下
高优先级
1. bl6-1 audit+report correction
2. bl6=1 vs bl5-v4-pam bootstrap/top-k/stratified/gate audit
3. bl6-1 mutil-seed repeat

中优先级
4. knn baseline
5. in-silicon perturbation

路线完整性优先级
7. bl4-full formal spilt
8. bl5-v4-pam/bl6-1统一总图和总表

低优先级：
9. bl3b
10. tier-aware
11. bl6-3 lightCrossAttn
12. bl6-4 raning loss

# 3
然后才进入bl6-2 

# 4
● 部分做了，但不完整。

   ────────────────────────────────────────────────────────────────────────────────

   消融实验（Ablation）—— 我们做了什么

   ┌───────────────────┬────────┬─────────────────────────────────────────────┐
   │ 消融项            │ 做了？ │ 说明                                        │
   ├───────────────────┼────────┼─────────────────────────────────────────────┤
   │ 去掉 PAM Encoder  │ ✅     │ BL5-v4-NoPAM-control（AUPRC 0.502）         │
   ├───────────────────┼────────┼─────────────────────────────────────────────┤
   │ PAM 打乱          │ ✅     │ BL5-v4-PAM-shuffle-control（AUPRC 0.139）   │
   ├───────────────────┼────────┼─────────────────────────────────────────────┤
   │ 去掉 LearnableRun │ ✅     │ BL5-v4-RNAFM-PAM-noRun（AUPRC 0.2765）     │
   ├───────────────────┼────────┼─────────────────────────────────────────────┤
   │ 去掉 RNA-FM       │ ✅     │ BL5-v4-LearnableRun-PAM-noRNAFM（AUPRC 0.1772） │
   ├───────────────────┼────────┼─────────────────────────────────────────────┤
   │ 只留 PAM          │ ✅     │ PAM-only baseline（AUPRC 0.0592）            │
   └───────────────────┴────────┴─────────────────────────────────────────────┘

   → ✅ BL5-v4-PAM 的最小组件消融矩阵已补齐。各单一组件及组合的贡献均已量化。

   ────────────────────────────────────────────────────────────────────────────────

   泛化实验（Generalization）—— 我们做了什么

   ┌─────────────────┬────────┬────────────────────────────────────────────────────────┐
   │ 泛化维度        │ 做了？ │ 说明                                                   │
   ├─────────────────┼────────┼────────────────────────────────────────────────────────┤
   │ Unseen sgRNA    │ ✅     │ sgrna_safe split，test 的 72 个 sgRNA 在训练集里没见过 │
   ├─────────────────┼────────┼────────────────────────────────────────────────────────┤
   │ Cross-dataset   │ ❌     │ 只在 CCLMoff 一个数据集上训练和测试                    │
   ├─────────────────┼────────┼────────────────────────────────────────────────────────┤
   │ Cross-PAM       │ ❌     │ 未做 PAM-holdout（test 有 14.1% non-NGG，但非严格 PAM 泛化） │
   ├─────────────────┼────────┼────────────────────────────────────────────────────────┤
   │ Cross-cell-line │ ❌     │ 没有验证其他细胞系                                     │
   ├─────────────────┼────────┼────────────────────────────────────────────────────────┤
   │ Cross-species   │ ❌     │ 没有验证其他物种                                       │
   └─────────────────┴────────┴────────────────────────────────────────────────────────┘

   → 我们做了 sgRNA-level 的泛化，但没有做数据集级、PAM 级、细胞系级的泛化。

   ────────────────────────────────────────────────────────────────────────────────

   一句话总结

   ```
     消融：✅ BL5-v4-PAM 组件消融矩阵已完整（7 消融 + 1 shuffle control + 1 gate）。
     泛化：只做了 sgRNA-safe 的，没做跨数据集/跨 PAM/跨条件的泛化。
   ```

---

## 消融与泛化现状修正版（2026-06-06）

### 1. 什么对比可以算消融

消融实验的核心要求是：两个模型除被移除或扰动的组件外，其余数据、split、backbone、训练超参、loss、head 和评估流程尽量一致。否则只能叫 baseline comparison、architecture comparison、variant comparison 或 control experiment。

| 对比 | 是否算严格消融 | 更准确叫法 | 说明 |
|:---|:---:|:---|:---|
| `BL5-v4-PAM` vs `BL5-v4-NoPAM-control` | ✅ | PAM Encoder ablation | 当前最标准的消融；基本只关掉 PAM Encoder。 |
| `BL5-v4-PAM` vs `BL5-v4-PAM-shuffle-control` | ⚠️ | PAM shuffle control / negative control | 不是去掉 PAM，而是保留 PAM 分支并打乱 PAM 与样本对应关系，用来验证正确 PAM 信息是否重要。 |
| `BL0b-on-BL5split` vs `BL5-v4-NoPAM-control` | ❌ | framework baseline comparison | 差异包含 LearnableRun、v4 classifier/head、fusion/实现等，不能说成 LearnableRun 纯贡献。 |
| `BL0b-on-BL5split` vs `BL5-v4-PAM` | ❌ | main baseline comparison | 证明 BL5-v4-PAM 强于 CCLMoff-style RNA-FM baseline，但不是单组件消融。 |
| `BL5-v4-PAM` vs `BL6-1-PAM-Gated-Fusion` | ⚠️ | gate addition / architecture enhancement | 如果只加 sample-wise gate，可以近似看 gate 增益；仍需多 seed、gate audit、bootstrap CI 后再定性。 |
| `BL5-3-LearnableRun` vs `BL5-v4-PAM` | ❌ | architecture variant comparison | Cross-Attn/Gated、PAM、head、pooling 等都不同。 |
| `BL4-finetune` vs `BL5-v4-PAM` | ❌ | stage comparison | 大版本不同，结构差异太多。 |
| GUIDE-seq 的 `P0/BL3` vs CCLMoff formal BL5 split | ❌ | cross-dataset historical comparison | 数据集和 split 不同，不能用于 formal ablation。 |

### 2. 当前已经完成的 targeted ablation/control

| 实验 | 组件状态 | AUROC | AUPRC | 结论 |
|:---|:---|---:|---:|:---|
| `BL0b-on-BL5split` | RNA-FM only baseline | 0.857756 | 0.295678 | CCLMoff-style RNA-FM baseline，同 formal BL5 test set。 |
| `BL5-v4-NoPAM-control` | RNA-FM + LearnableRun，无 PAM | 0.984098 | 0.502389 | v4 无 PAM 框架已经显著强于纯 RNA-FM baseline。 |
| `BL5-v4-PAM` | RNA-FM + LearnableRun + PAM | 0.984194 | 0.531281 | BL5 阶段主模型。 |
| `BL5-v4-PAM-shuffle-control` | RNA-FM + LearnableRun + shuffled PAM | 0.669701 | 0.138883 | PAM 对应关系被破坏后性能崩溃，支持正确 PAM 信息有价值。 |
| `BL6-1-PAM-Gated-Fusion` | BL5-v4-PAM + sample-wise gate | 0.984993 | 0.539917 | single-run strong success，但还需 bootstrap CI、多 seed 和 gate audit。 |

关键解释边界：

```text
NoPAM - BL0b = +0.206711
```

这不是 LearnableRun 的纯贡献，而是 BL5-v4 no-PAM framework 的综合增益。

```text
PAM - NoPAM = +0.028892
```

这个更接近 PAM Encoder 在 v4 框架下的边际贡献。

```text
PAM - Shuffle = +0.392398
```

这个支持 PAM 分支依赖正确 PAM 与样本对应关系，而不是只来自额外参数量。

### 3. 最小完整消融矩阵（✅ 已完成）

| 名称 | 启用组件数 | RNA-FM | LearnableRun | PAM | Gate | 当前状态 | AUROC | AUPRC | 目的 |
|:---|:---:|:---:|:---:|:---:|:---:|:---|---:|---:|:---|
| `BL0b-on-BL5split` | 1 | ✅ | ❌ | ❌ | ❌ | ✅ 已完成 | 0.857756 | 0.295678 | RNA-FM-only baseline，同 formal BL5 test set。 |
| `LearnableRun-only` | 1 | ❌ | ✅ | ❌ | ❌ | ✅ 已完成 | 0.960909 | 0.294920 | 看 LearnableRun 单独贡献。 |
| `PAM-only` | 1 | ❌ | ❌ | ✅ | ❌ | ✅ 已完成 | 0.499426 | 0.059223 | 看 PAM shortcut / PAM 单独能解释多少。 |
| `RNA-FM + LearnableRun, no PAM` | 2 | ✅ | ✅ | ❌ | ❌ | ✅ 已完成（`BL5-v4-NoPAM-control`） | 0.984098 | 0.502389 | 去掉 PAM 的主消融。 |
| `RNA-FM + PAM, no LearnableRun` | 2 | ✅ | ❌ | ✅ | ❌ | ✅ 已完成（`BL5-v4-RNAFM-PAM-noRun`） | 0.837950 | 0.276529 | 看去掉 LearnableRun 后，RNA-FM+PAM 还剩多少性能。 |
| `LearnableRun + PAM, no RNA-FM` | 2 | ❌ | ✅ | ✅ | ❌ | ✅ 已完成（`BL5-v4-LearnableRun-PAM-noRNAFM-control`） | 0.952749 | 0.177171 | 看非 RNA-FM 特征组合能力。 |
| `BL5-v4-PAM` | 3 | ✅ | ✅ | ✅ | ❌ | ✅ 已完成 | 0.984194 | 0.531281 | BL5 full anchor。 |
| `BL5-v4-PAM-shuffle-control` | 3 | ✅ | ✅ | ⚠ shuffled | ❌ | ✅ 已完成（negative control） | 0.669701 | 0.138883 | PAM correspondence negative control。 |
| `BL6-1-PAM-Gated-Fusion` | 4 | ✅ | ✅ | ✅ | ✅ | ✅ 已完成（single-run） | 0.984993 | 0.539917 | BL6-1 full candidate，仍需 gate audit / bootstrap / multi-seed。 |

当前 BL5 最小组件消融矩阵已经补齐。后续优先级不再是继续补 BL5 消融，而是：

1. 统一 BL5 消融总表和总图；
2. 修正各报告中的 PAM 分层口径，统一使用 positions 21-23 / `PAM_original`；
3. 做 BL6-1 gate audit、bootstrap CI、top-k、paired delta、multi-seed；
4. 如老师继续要求路线完整性，再考虑 BL4-full formal split、kNN baseline、in-silico perturbation。

### 3.1 BL5 消融闭环结论

BL5-v4-PAM 的最小组件消融矩阵已经完成。各组件单独及组合的 AUPRC 如下：

| 组件组合 | AUPRC | 核心发现 |
|:---|:---:|:---|
| RNA-FM only | 0.2957 | 单视图基线。RNA-FM 的序列上下文单独能提供约 0.30 的排序能力 |
| LearnableRun only | 0.2949 | 与 RNA-FM only 几乎持平——显式错配模式 + seed 权重的信息量 ≈ RNA-FM 隐式上下文 |
| PAM only | 0.0592 | 单独几乎无用。PAM motif 本身在 NGG 子集内无区分能力（AUROC≈0.5） |
| RNA-FM + LearnableRun | 0.5024 | **核心组合**。两个互补视角融合后 AUPRC 跃升 ~70%，是多视角融合价值的直接证据 |
| RNA-FM + PAM | 0.2765 | 低于 RNA-FM only（−0.0191）。PAM 单独加到 RNA-FM 上无增益 |
| LearnableRun + PAM | 0.1772 | 低于 LearnableRun only（−0.1177）。PAM 在无 RNA-FM 时甚至可能干扰优化 |
| **RNA-FM + LearnableRun + PAM** | **0.5313** | **全模型**。PAM 在强联合上下文中贡献 +0.0289 |
| PAM shuffle-control | 0.1389 | PAM 对应关系被破坏后性能崩溃（−0.3924），证明正确 PAM 信息确实被模型使用 |

> 💡 **严谨解释（一段话版）**：
>
> BL5-v4-PAM 的主性能来自 **RNA-FM + LearnableRun 的强互补**：两个视角单独时 AUPRC 均约 0.29，融合后跃升至 0.50。PAM 单独很弱（0.059），也不能单独补强 RNA-FM（0.2765 < 0.2957）或 LearnableRun（0.1772 < 0.2949）。但在 RNA-FM + LearnableRun 同时存在时，正确 PAM 能带来稳定的边际增益（+0.0289），且该增益依赖正确 PAM 与样本的对应关系——PAM shuffle-control 崩到 0.1389 即为明证。这说明 PAM 是一个「催化剂」而非「燃料」：它本身不驱动性能，但能让已有的强组合跑得更好。
>
> 💡 **比喻理解**：RNA-FM 像一位能读懂整段文字含义的语言学家（理解序列上下文），LearnableRun 像一位专门数错别字个数的校对员（统计连续错配模式），PAM 像一个只认识三个字母（NGG）的路标。语言学家和校对员互补——一个宏观理解、一个微观统计——两人合作已经很强。路标本身告诉不了你文章写得好不好，但当他俩都在工作时，路标能帮他们快速定位到正确的段落开头。如果把路标撕掉（NoPAM），两人依然很强（0.5024）；如果把路标换成假的（shuffle），反而会误导他们，表现暴跌（0.1389）。

### 4. 当前泛化证据边界

| 泛化维度 | 当前状态 | 说明 | 下一步 |
|:---|:---:|:---|:---|
| Same-dataset held-out test | ✅ | formal BL5 test set 有 954,326 个候选位点，3,057 个 observed_positive。 | 继续所有模型都保持同一 test set。 |
| Unseen sgRNA | ✅ | `sgrna_safe` / formal split 按 `sgRNA_type` 分组，test 有 72 个训练阶段未见过的 sgRNA_type。 | 补 per-sgRNA report，看提升是否普遍。 |
| Cross-PAM strict holdout | 🔄 | PAM motif feasibility audit 已完成；AGG strict holdout 成对训练/评估正在由 Kimi 推进。 | 当前优先审核 `AGG`：`BL5-v4-PAM-holdout-AGG` vs `BL5-v4-NoPAM-holdout-AGG`。 |
| NGG / non-NGG stratified evaluation | ✅ | canonical PAM 统计：NGG-only = 819,984；non-NGG = 134,342，non-NGG 约 14.08%，不能写“几乎没有”。 | 报告中明确 PAM 口径为 positions 21-23 / `PAM_original`。 |
| Cross-dataset | ⚠️ | External dataset feasibility audit 已完成：仓库内 `ready_for_strict_external_eval=0`；`SITE/K562` 仅是 provenance-required limited candidates。 | 不直接跑 external eval；如需推进，先做 SITE/K562 provenance audit。 |
| Cross-cell-line | ❌ | CCLMoff `Method` / `Length` 大量为空，细胞系/检测方法元数据不足。 | 只有找到可靠 metadata mapping 后再做。 |
| Cross-species | ❌ | 当前没有可靠可用的跨物种数据。 | 作为 future work，不硬做。 |
| Training seed stability | ⚠️ | BL5-v4-PAM historical best 0.531281，latest rerun 0.516095，存在训练波动；BL6-1 目前 single-run 0.539917。 | 做 BL6-1 seed repeat，并报告 mean +/- std。 |

### 4.1 BL5 泛化补充任务表（不优先新训练）

当前 BL5 已经具备核心泛化证据：`formal_split_bl5_seed42.json` 是 `sgRNA-safe` group split，test set 含 72 个训练阶段未见过的 `sgRNA_type`，共 954,326 个候选位点。下一步不建议继续开 BL5 新模型，优先把现有 `test_predictions.csv` 做成泛化证据包。

| 任务 | 泛化问题 | 是否需要训练 | 当前状态 | 优先级 | 推荐输出 | 备注 |
|:---|:---|:---:|:---:|:---:|:---|:---|
| `per-sgRNA generalization report` | BL5-v4-PAM 的提升是否在 72 个 unseen sgRNA 上普遍存在，而不是少数 sgRNA 驱动。 | ❌ | ❌ 待做 | 高 | `results/bl5_generalization/per_sgrna_metrics.csv` / `.md` | 每个 `sgRNA_type` 报 samples、observed_positive、unobserved_candidate、AUROC、AUPRC、top-k recall；统计 BL5-v4-PAM 胜过 BL0b / NoPAM / shuffle 的 sgRNA 数量。 |
| `per-PAM / NGG / non-NGG stratified report` | BL5-v4-PAM 的提升是否主要来自 non-NGG shortcut，NGG-only 上是否仍有效。 | ❌ | ⚠️ 部分已有，需统一 | 高 | `results/bl5_generalization/per_pam_stratified_metrics.csv` / `.md` | 必须统一口径为 positions 21-23：`PAM_original = off_seq[20:23]`，禁止用 `off_seq[-3:]`。比较 BL0b、NoPAM、PAM、shuffle、PAM-only 等核心模型。 |
| `paired bootstrap CI` | BL5-v4-PAM 相对 NoPAM / shuffle / BL0b 的 AUPRC 差异是否稳定。 | ❌ | ❌ 待做 | 高 | `results/bl5_generalization/bootstrap_ci.csv` / `.md` | 重点做 paired bootstrap：`PAM - NoPAM`、`PAM - shuffle`、`PAM - BL0b`、`NoPAM - BL0b`。如果 CI 不跨 0，PAM 边际贡献更稳。 |
| `top-k operating point table` | 实际筛选时 top-ranked 位点能找回多少 observed_positive。 | ❌ | ⚠️ 部分已有，需统一 | 高 | `results/bl5_generalization/topk_operating_points.csv` / `.md` | 报 Top-100、Top-500、Top-1000、Top-2000、Top-3057、Top-1%、Top-5%、Top-10%；建议同时做 global top-k 和 per-sgRNA macro top-k。 |
| `paired probability delta by subset` | 正确 PAM 是否主要提高 observed_positive 概率，而不是同时抬高大量 unobserved_candidate。 | ❌ | ⚠️ shuffle 已有部分，需扩展 | 中高 | `results/bl5_generalization/paired_delta_by_subset.csv` / `.md` | 分 all、observed_positive、unobserved_candidate、NGG-only、non-NGG、per-sgRNA；重点看 `prob_pam - prob_nopam` 和 `prob_pam - prob_shuffle`。 |
| `PAM-holdout feasibility audit` | 当前 CCLMoff 是否适合做严格 cross-PAM generalization。 | ❌ | ✅ 已完成 | 已封口 | `results/bl5_generalization/pam_holdout_feasibility/` | Feasible: `AGG, CGG, GAG, GGG, TGG`；marginal: `AAG, ATG, CAG, GTG, TAG, TTG`；noncanonical / low-support motifs 不推荐。 |
| `strict PAM-holdout training` | 训练不见某 PAM、测试只看该 PAM 的严格跨 PAM 泛化。 | ✅ | 🔄 AGG 进行中 | 最高 | `results/bl5_generalization/pam_strict_holdout/AGG/` | AGG 是优先级 1，必须成对跑 PAM vs NoPAM；后续候选为 TGG、GAG。 |
| `selected sgRNA LOO` | 比 formal sgRNA-safe 更严格的 leave-one-sgRNA-out 泛化。 | ✅ | ❌ 暂不建议全量 | 低/条件触发 | `results/bl5_generalization/selected_loo/` | 不建议全量 LOO；如老师强要求，可选 5-10 个 positive 足够多的 sgRNA 做小规模 LOO。 |
| `external dataset feasibility audit` | 是否能做 cross-dataset generalization。 | ❌ | ✅ 已完成 | 已封口 | `results/bl5_generalization/external_dataset_feasibility/` | 结论：无 ready strict external benchmark；`SITE/K562` 需 provenance audit 后才可能作为有限候选；GUIDE-seq / CHANGE-seq / Tasi overlap 过高。 |
| `cross-cell-line / cross-species validation` | 是否能跨细胞系或跨物种泛化。 | ✅ | ❌ future work | 低 | 暂无 | 当前 CCLMoff `Method` / `Length` 大量为空，细胞系/物种元数据不足；除非找到可靠外部数据，否则写作 limitation/future work。 |

推荐执行顺序：

1. 等 Kimi 完成并汇报 `AGG` strict PAM-holdout 成对实验。
2. 审核 AGG split / leakage / best checkpoint / paired bootstrap CI。
3. 若 AGG 流程无问题，再推进 `TGG` strict PAM-holdout。
4. 若 AGG/TGG 稳定，再做 `GAG` non-NGG exploratory holdout。
5. SITE/K562 只做 provenance audit，不直接作为 external eval。

稳妥结论：

```text
BL5 已经完成 same-dataset unseen-sgRNA generalization：formal split 按 sgRNA_type 分组，test set 有 72 个训练阶段未见过的 sgRNA_type。PAM-holdout feasibility audit 和 external dataset feasibility audit 已完成；当前泛化主线是 AGG strict PAM-holdout 成对实验。Cross-dataset 当前没有 ready raw external benchmark，SITE/K562 只能作为 provenance 后的有限候选。
```

### 5. 对老师的稳妥表述

如果问“做了消融吗”：

```text
我们已经完成了 BL5-v4-PAM 的完整组件消融矩阵（7 个配置 + 1 个 shuffle control + 1 个 gate variant），覆盖了 RNA-FM / LearnableRun / PAM 三个组件的所有单组件、双组件、三组件组合。核心结论：RNA-FM + LearnableRun 是性能主驱动力（AUPRC 0.5024），PAM 单独几乎无贡献（0.0592），但在完整上下文（RNA-FM + LearnableRun）中贡献稳定的边际增益（+0.0289），且该增益依赖正确 PAM 与样本的对应关系（shuffle 后跌至 0.1389）。
```

如果问“做了泛化吗”：

```text
我们做了 sgRNA-safe generalization。formal split 按 sgRNA_type 分组，test set 中有 72 个训练阶段未见过的 sgRNA_type，共 954,326 个候选位点。因此当前结果不是训练集结果，也不是随机行划分结果，而是在 unseen sgRNA group 上的测试结果。PAM-holdout feasibility audit 已确认 AGG/TGG/GGG/CGG/GAG 可作为候选，目前正在推进 AGG strict holdout 成对实验。Cross-dataset feasibility audit 的结论是仓库内没有 ready raw external benchmark；SITE/K562 需要 provenance audit，不能直接写成 external evaluation。
```
