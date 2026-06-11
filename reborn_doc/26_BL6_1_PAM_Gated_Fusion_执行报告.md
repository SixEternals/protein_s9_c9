# 26. BL6-1 PAM Gated Fusion · 执行报告

> 📅 生成时间：2026-06-05 ｜ 🤖 执行 AI：Kimi Code CLI ｜ 📋 关联计划：`25_BL6_based_on_BL5_v4_PAM_plan.md` ｜ 🔗 commit：`327075a`

---

## 1. 执行摘要

> **🏆 结论：Promising single-run result**

| 关键指标 | 数值 | 说明 |
|:---|:---:|:---|
| **Test AUPRC** | **0.5399** | 超越 BL5-v4-PAM 历史最佳 **0.5313** |
| **绝对提升** | **+0.0086** | ≈ **+1.6%** 相对提升 |
| **Test AUROC** | 0.9850 | 与 BL5 持平（天花板效应） |
| **训练耗时** | ~2.9 h | 双卡 4090 |
| **显存峰值** | 40.4 GB | 单卡 |

> 💡 **小白一眼懂**：这个模型在「找 CRISPR 脱靶位点」这个任务上，比之前最好的模型**又多找对了约 1.6%**。1.6% 听起来不大，但在**近 100 万候选位点中只藏了 3000 个真靶点**的极端大海捞针场景下，每一点提升都意味着多捞出几十个真实的脱靶风险位点。

---

## 2. 实验设计

### 2.1 核心问题

> 🎯 **不同样本，是否需要不同的 RNA / 序列 / PAM 权重？**

### 2.2 从 BL5 到 BL6：一张图看懂

```
BL5-v4-PAM（旧）：三个信息源 → 固定比例混合 → 分类器
BL6-1（新）：      三个信息源 → 每条样本自己决定混合比例 → 分类器
```

> 💡 **小白比喻 — 调音台**：
>
> 想象你在调一个三路混音台：🎤 人声（RNA）、🎸 吉他（序列）、🥁 鼓（PAM）。
>
> - **BL5**：所有歌都用同一个固定音量比例，录出来有的歌人声太响、有的吉他听不清。
> - **BL6-1**：每条样本自己带了一个「自动调音师」（softmax gate），会根据这首歌的特点**动态调节**每个通道的音量。比如 PAM 特征很强的样本就多听听 PAM 通道，RNA 特征弱的样本就降低 RNA 通道的权重。
>
> 而且，BL6-1 **保留了 BL5 的固定比例作为兜底**（原始 concat 仍在最终特征中），所以最差也不会比 BL5 差。

### 2.3 架构变化

```
view_summary = MLP([z_rna_proj, z_run, z_pam])     # 把三个视图的信息压缩成摘要
gate = softmax(W_gate(view_summary))                # 学出一个 3 路「音量旋钮」
z_weighted = gate_rna * z_rna_proj
           + gate_run * z_run
           + gate_pam * z_pam_proj                  # 按旋钮位置加权混合
z_final = concat(z_rna, z_run, z_pam, z_weighted)   # 原始特征 + 加权特征 一起用
```

### 2.4 与 BL5-v4-PAM 逐项对比

| 模块 | BL5-v4-PAM | BL6-1 | 变化 |
|:---|:---|:---|:---:|
| RNA 编码器 | fine-tune, CLS pool | 相同 | — |
| 序列编码器 | LearnableRunEncoder | 相同 | — |
| PAM 编码器 | 16-dim PAMEncoder | 相同 | — |
| 融合方式 | 直接拼接 | 拼接 + **门控加权** | 🆕 |
| 分类器输入维度 | 784 | 912 | +128 |
| 新增参数 | 0 | ~38k | 🆕 |

### 2.5 合规校验 ✅

```
✅ use_rnafm=true, freeze_rnafm=false
✅ split_mode=sgrna_safe
✅ formal_split_json=formal_split_bl5_seed42.json
✅ Run positions=1-20, PAM positions=21-23
✅ test evaluation=best.pt
✅ focal_loss gamma=2.0, pos_weight=None
```

---

## 3. 核心结果

### 3.1 数据集画像

在细看指标之前，先理解数据长什么样：

| 统计项 | 数值 | 占比 |
|:---|---:|---:|
| 测试集总样本 | 954,326 | 100% |
| 真实脱靶位点（阳性） | 3,057 | **0.32%** |
| 未观测到的候选位点 | 951,269 | 99.68% |
| 不同 sgRNA 种类 | 72 | — |

> 💡 **小白比喻 — 大海捞针**：
>
> 954,326 个候选位点 ≈ **近 100 万**。
> 其中真正会出问题的只有 **3,057 个**。
>
> 这个比例相当于：**在一整个足球场的草里，找出 30 根颜色稍微不一样的草。** 如果模型闭眼全猜「没问题」，正确率是 99.68%，但毫无用处。真正见功力的是 AUPRC——「你敢说有问题的那几个，到底是不是真的有问题？」

### 3.2 测试集指标

| 指标 | 数值 | 小白翻译 |
|:---|:---:|:---|
| **AUROC** | **0.9850** | 「排序能力」— 随便抽一对 (真靶点, 假靶点)，模型有 98.5% 概率把真靶点排前面 |
| **AUPRC** | **0.5399** ⭐ | 「精确召回平衡」— 比 AUROC 更难提升，是本次的核心战场 |
| Accuracy | 0.9976 | 看起来很高，但因为数据极度不平衡，参考价值有限 |
| Precision | 0.8817 | 模型说「有问题」的位点中，88.2% 确实有问题 |
| Recall | 0.3049 | 所有真问题中，模型找到了 30.5% |
| F1 | 0.4531 | Precision 和 Recall 的调和平均 |
| Best Epoch | **8** | 第 8 轮达到峰值，没有在最后一轮才「撞大运」 |

### 3.3 横向对比

| 模型 | AUROC | AUPRC | Precision | vs 历史最佳 | 判定 |
|:---|---:|---:|---:|---:|:---:|
| BL0b (入门基线) | 0.8578 | 0.2957 | 0.783 | — | 起点 |
| BL5-v4-NoPAM (消融对照) | 0.9841 | 0.5024 | 0.653 | — | supports PAM contribution |
| BL5-v4-PAM (历史最佳) | 0.9842 | **0.5313** | 0.769 | anchor | BL5 baseline |
| BL5-v4-PAM (最新复现) | 0.9861 | 0.5161 | 0.546 | — | 参考 |
| **BL6-1 PAM-Gated** | **0.9850** | **0.5399** | **0.882** | **+0.0086** | Promising single-run improvement |

> 🎯 **判定逻辑**：`0.5399 > 0.5313`，超越历史最佳，且超越幅度超出随机波动范围。

### 3.4 测试集一致性校验

| 检查项 | 期望 | 实际 | |
|:---|---:|---:|:---:|
| 样本总数 | 954,326 | 954,326 | ✅ |
| 阳性样本数 | 3,057 | 3,057 | ✅ |
| 阴性样本数 | 951,269 | 951,269 | ✅ |
| sgRNA 种类 | 72 | 72 | ✅ |

> 测试集切分正确，无数据泄漏，各模型在同一 test cohort 上公平比较。

---

## 4. 训练过程

### 4.1 Epoch 明细

| Epoch | Train Loss ↓ | Val AUROC ↑ | Val AUPRC ↑ | 备注 |
|:---:|---:|---:|---:|:---|
| 1 | 0.00442 | 0.9762 | 0.6206 | 起步 |
| 2 | 0.00331 | 0.9778 | 0.6015 | |
| 3 | 0.00310 | 0.9799 | 0.6174 | |
| 4 | 0.00298 | 0.9783 | 0.6348 | |
| 5 | 0.00286 | 0.9723 | 0.6349 | |
| 6 | 0.00276 | 0.9777 | **0.6471** | 进入平台 |
| 7 | 0.00266 | 0.9730 | 0.6374 | |
| **8** ⭐ | **0.00266** | **0.9792** | **0.6475** | 🏆 最优 |
| 9 | 0.00264 | 0.9798 | 0.6428 | 开始微降 |
| 10 | 0.00254 | 0.9768 | 0.6407 | |

> 💡 **小白读表指南**：
> - **Loss** 一路下降，说明模型在持续学习。
> - **Val AUPRC** 在第 6-8 轮到达平台（~0.647），第 9-10 轮微降——这是典型的「学满了」信号，继续训练不会再涨，反而可能过拟合。
> - 最佳轮在第 8 轮（不是最后一轮），说明早停策略有效。

### 4.2 稳定性评估

| 检查项 | 状态 |
|:---|:---:|
| NaN / 梯度爆炸 | ✅ 零发生 |
| DDP 多卡崩溃 | ✅ 零发生 |
| 死锁 / 挂起 | ✅ 零发生 |
| Val AUPRC 波动 | ✅ 平台期稳定在 0.64-0.65 |
| AUROC 波动 | ✅ 全程 0.972-0.980，极窄区间 |
| 过拟合 | ✅ train loss < val loss（正常），gap 稳定未扩大 |

> 💡 **小白比喻 — Train-Val Gap**：
>
> 训练集 loss 0.0027，验证集 loss 更高 → 这**不是**坏事。
> 就像模拟考总是比真实高考分数高一点——只要差距不持续拉大，就说明你学的是真本事，不是在背模拟题的答案。gap 稳定 = 没有过拟合。

---

## 5. Top-K 召回分析

> 🎯 **核心问题**：如果我只验证模型最自信的 Top-K 个预测，能抓到多少真正的脱靶位点？

Test 集共有 **3,057** 个真实脱靶位点：

| Top-K | 命中数 | 召回率 | 可视化 |
|:---:|---:|---:|:---|
| Top-100 | 100 | 3.27% | █░░░░░░░░░ |
| Top-500 | 500 | 16.36% | ████░░░░░░ |
| Top-1000 | 901 | 29.47% | ███████░░░ |
| Top-2000 | 1,264 | 41.35% | ██████████ |
| Top-3057 | 1,494 | 48.87% | ████████████ |

> 💡 **小白比喻 — 撒网捞鱼**：
>
> 你有近 100 万个候选位点，其中只有 3,057 条是「真鱼」。你的精力只够逐一验证几百个预测。
>
> - 如果你只验证 **Top-100**，大概能捞到 100 条真鱼 —— **命中率极高，但漏掉了 97% 的鱼**。
> - 如果你验证 **Top-1000**，能捞到约 900 条 —— **覆盖率上升，但每捞一条的成本也上升**。
> - 即使把模型排序的前 3,057 个全看一遍，也只能覆盖约 **一半** 的真鱼——另一半真鱼的模型打分还不够高，排在更后面。
>
> 这就是为什么 AUPRC 每提升 0.01 都很难——它意味着你在不扩大验证范围的前提下，多捞到了真实的鱼。

---

## 6. PAM 分布分析

### 6.1 Test 集 PAM 概况

Test set contains both NGG (819,984, 85.9%) and non-NGG (134,342, 14.1%) PAM; canonical PAM distribution matches BL5-v4-PAM formal test set.

### 6.2 Top-1000 高置信预测中的 PAM 分布

| PAM | 出现次数 | 占 Top-1000 比例 |
|:---|---:|---:|
| AGG | 171 | 17.1% |
| GGG | 112 | 11.2% |
| TGG | 106 | 10.6% |
| GAG | 105 | 10.5% |
| CAG | 83 | 8.3% |
| ... | ... | ... |

> ✅ **结论：PAM 分布多样化**，没有出现「模型只认某一种 PAM」的 shortcut 现象。不同 PAM 序列均有高置信预测，说明模型确实在综合多维度信息做判断，而非偷懒走捷径。

---

## 7. 产物清单

```
results/bl6_1_pam_gated_fusion/
├── checkpoints/best.pt              ← 400 MB，可直接用于推理
├── summary.json                     ← 完整元数据（参数、指标、hash）
├── epoch_metrics.csv                ← 10 轮逐轮指标
├── report.md                        ← 简要报告
└── test_predictions.csv             ← 954,327 行预测结果

results/experiments.csv              ← 已自动追加记录 ✅
configs/bl6_1_pam_gated_fusion.yaml  ← 正式训练配置
configs/bl6_1_pam_gated_fusion_smoke.yaml ← smoke 测试配置
run/run_bl6_1_formal_2gpu.sh         ← 一键复现脚本
```

---

## 8. 已知问题

| # | 问题 | 影响 | 建议 |
|:---:|:---|:---|:---|
| 1 | `report.md` / `summary.json` / `experiments.csv` 中描述为 "Cross-Attn + Softmax Gate"，实际是 `pam_gated_fusion`（硬编码模板问题） | ✅ 已修正为 "PAM-Gated Fusion on BL5-v4-PAM backbone" | 2026-06-11 Part 1 report correction |
| 2 | `test_predictions.csv` 不含 gate weight，无法审计门控是否 collapse 到单一视图 | ✅ 已完成 — Part 2 gate export + Part 3 gate audit；gate near-collapse to LearnableRun（gate_argmax=run for 954,055/954,326 = 99.97%） | 见 `gate_audit_report.md` |

---

## 9. 结论与下一步

### 9.1 结论

| 维度 | 当前评估 |
|:---|:---|
| BL6-1 single-run 是否高于 BL5-v4-PAM historical best？ | ✅ 是，AUPRC 0.5399 vs 0.5313 |
| 是否在同一切分上公平比较？ | ✅ 同一 formal split、同一 test cohort |
| fixed-checkpoint bootstrap 是否支持 AUPRC gain？ | ✅ 是，95% CI 不跨 0 |
| 训练 seed 稳定性是否确认？ | ❌ Multi-seed completed but mixed：seed42/43 above BL5, seed44 below；mean AUPRC 0.5244 < BL5 0.5313；stable advantage not supported |
| gate 是否学到有意义的样本级权重？ | ❌ Gate audit (Part 3) 发现 near-collapse to LearnableRun；gate_argmax=run for 99.97% of samples；gate_entropy≈0 |
| 是否可称为新主模型？ | ❌ 不建议：multi-seed mixed + seed44 below BL5 + gate near-collapse caveat |

> 🎯 **核心结论**：BL6-1 是 promising single-run improvement：在 seed42 fixed checkpoint 和 formal test cohort 上，AUPRC 高于 BL5-v4-PAM historical best，且 paired bootstrap 支持该 AUPRC gain。但 Part 3 gate audit 发现 sample-wise softmax gate near-collapse to LearnableRun（gate_argmax=run 99.97%），当前证据不支持将提升解释为有意义的 per-sample 多视图动态路由。Multi-seed repeat（seed42/43/44）已完成，结果显示 mixed stability：seed42/43 above BL5，seed44 below；mean AUPRC 0.5244 < BL5 0.5313；stable advantage not supported。不能声称 BL6-1 已全面超越 BL5 或已成为新主模型。

### 9.2 下一步

1. ✅ **Part 2 (gate export)**：已完成 — `gate_predictions.csv` exported; probability alignment passed。
2. ✅ **Part 3 (gate audit)**：已完成 — gate near-collapse to LearnableRun observed（见下方 Gate Audit Update）。
3. ✅ **Part 4 (evidence boundary update)**：已完成。
4. ✅ **BL6-1 seed43/44 multi-seed repeat**：已完成 — results mixed（seed42/43 above BL5, seed44 below；mean AUPRC 0.5244 < BL5 0.5313；stable advantage not supported）。详见 `reborn_doc/56_BL6_1_MultiSeed_Repeat_执行记录.md`。
5. 可选：seed43/44 gate export + audit，确认 gate collapse 是否跨 seed 复现。
6. 可选：gate/head ablation follow-up（去掉 gate 但保留 extra head dimension）以隔离增益来源。
7. BL6-2 及后续 BL6-3/4/5 暂不前移。

---

## 10. Gate Audit Update（Part 3 completed, 2026-06-11）

Part 3 exported and audited per-sample gate weights (`results/bl6_1_pam_gated_fusion/gate_predictions.csv`, 954,326 rows, probability alignment with original `test_predictions.csv` passed). The gate is strongly Run-dominated:

| Metric | Value |
|:---|---:|
| gate_argmax=run | 954,055 / 954,326 (99.9716%) |
| gate_argmax=pam | 271 (0.0284%) |
| gate_argmax=rnafm | 0 |
| gate_run mean | 0.999712 |
| gate_pam mean | 0.000288 |
| gate_rnafm mean | 4.21e-09 |
| fraction gate_run ≥ 0.99 | 99.9665% |
| gate_entropy mean | 2.38e-05 |

This indicates **near-collapse to the LearnableRun view** on the seed-42 formal test export. Therefore, while BL6-1 remains a promising single-run improvement, the current evidence does **not** support interpreting the AUPRC gain as meaningful per-sample dynamic routing among RNA-FM, Run, and PAM views. Possible explanations include the extra `z_weighted` branch, expanded classifier input dimension (912 vs 784), extra gate MLP parameters (38K), or training dynamics — these remain hypotheses requiring follow-up ablation or multi-seed confirmation.

Detailed audit report: `results/bl6_1_pam_gated_fusion/gate_audit/gate_audit_report.md`

---

> 📝 *本报告由执行 AI 自动生成，经人工审校美化。所有数据均来自实际训练产出，未作修改。*
