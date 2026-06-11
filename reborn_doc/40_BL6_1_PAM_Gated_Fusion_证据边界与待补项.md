# 40. BL6-1 PAM Gated Fusion · 证据边界与待补项

> 📅 2026-06-08 ｜ 🎯 明确 BL6-1 当前证据边界：哪些已经验证、哪些仍待补、哪些不能声称
>
> **性质**：证据边界文档，不是「BL6-1 已确认全面超越 BL5」的宣告。未训练模型、未运行推理、未调用 GPU。

---

## 1. 当前状态概览

| 项目 | 状态 |
|:---|:---|
| 模型架构 | PAM-Gated Fusion on BL5-v4-PAM backbone（三个 sample-wise softmax gate weights 加权 RNA/Run/PAM） |
| 训练 | ✅ Single-run complete（seed=42, best_epoch=8） |
| 执行报告 | ✅ `reborn_doc/26_BL6_1_PAM_Gated_Fusion_执行报告.md` |
| Test 评估 | ✅ best.pt 评估完成 |
| Multi-seed repeat | ✅ 已完成（seed42/43/44）；mixed stability；mean AUPRC 0.5244 < BL5 0.5313；stable advantage not supported |
| Report correction (Part 1) | ✅ 已完成 — 模板错误修正 |
| Gate export (Part 2) | ✅ 已完成 — `gate_predictions.csv` exported; probability alignment passed |
| Gate audit (Part 3) | ✅ 已完成 — near-collapse to LearnableRun observed |
| Paired bootstrap CI | ✅ 完成（详见下方） |
| Per-sgRNA 分析 | ✅ 完成 |
| Per-PAM 分析 | ✅ 完成 |
| Top-K operating points | ✅ 完成（详见下方） |

---

## 2. 核心指标（Single-Run）

| 指标 | BL5-v4-PAM（历史最佳） | BL6-1（single-run） | Δ |
|:---|---:|---:|:---|
| Test AUROC | 0.9842 | 0.9850 | +0.0008 |
| Test AUPRC | 0.5313 | **0.5399** | **+0.0086** |
| Test Accuracy | 0.9976 | 0.9976 | — |
| Test Precision | 0.8805 | 0.8817 | +0.0012 |
| Test Recall | 0.3065 | 0.3049 | −0.0016 |
| Test F1 | 0.4546 | 0.4531 | −0.0015 |
| Best Epoch | 9 | 8 | — |
| 训练时间 | 176.8 min | 172.5 min | — |
| GPU 显存 | 40.41 GB | 40.41 GB | — |

> 所有比较在同一 formal BL5 split test set（954,326 样本，3,057 observed_positive）上进行。

---

## 3. 已有证据（✅ 已完成）

### 3.1 Paired Bootstrap CI

来源：`results/bl6_1_validation/bootstrap_ci_report.md`

| 比较 | AUPRC Delta 中位数 | 95% CI | CI 跨 0？ |
|:---|---:|:---|:---:|
| BL6-1 − BL5 AUPRC | **+0.0087** | **[+0.0024, +0.0149]** | ❌ **不跨 0** |
| BL6-1 − NoPAM AUPRC | +0.0374 | [+0.0315, +0.0442] | ❌ 不跨 0 |
| BL5 − NoPAM AUPRC | +0.0288 | [+0.0224, +0.0355] | ❌ 不跨 0 |
| BL6-1 − Shuffle AUPRC | +0.4017 | [+0.3876, +0.4162] | ❌ 不跨 0 |
| BL6-1 − BL5 AUROC | +0.0008 | [−0.0002, +0.0018] | ⚠️ **跨 0** |

> 🎯 **Bootstrap 结论**：在当前 test cohort 上，BL6-1 相对 BL5-v4-PAM 的 AUPRC 提升是统计稳定的（CI 不跨 0）。AUROC delta 跨 0 是天花板效应，不说明问题。

⚠️ **Bootstrap 的局限**：Bootstrap 只反映 **test resampling uncertainty**（同一个训练好的模型在不同虚拟测试集上的波动），**不等于 training seed stability**。要证明训练稳定性，需要实际用不同 random seed 重新训练至少 2-3 次。

### 3.2 Per-sgRNA 分析

来源：`results/bl6_1_validation/per_sgrna_report.md`

| 结果 | 详情 |
|:---|:---|
| BL6-1 更优的 sgRNA | 33 / 72（45.8%） |
| BL5 更优的 sgRNA | 31 / 72（43.1%） |
| 持平 | 8 / 72（11.1%） |
| Positive ≥ 20 的 sgRNA 中 BL6-1 优势 | 12 / 21（57.1%） |
| Mean delta AUPRC | +0.0023 |
| Median delta AUPRC | 0.0000 |

> 🎯 **结论**：BL6-1 的提升**不集中在少数 sgRNA**，但也**不是全面压倒性优势**。在 positive 数更多（AUPRC 更稳定）的 sgRNA 上优势更明显。

### 3.3 Per-PAM 分析

来源：`results/bl6_1_validation/per_pam_report.md`

| PAM 分组 | BL5 AUPRC | BL6-1 AUPRC | Δ | 结论 |
|:---|---:|---:|---:|:---|
| All | 0.5313 | 0.5399 | +0.0086 | 提升 |
| NGG-only（819,984） | 0.3562 | 0.3640 | +0.0078 | 提升 |
| non-NGG（134,342） | 0.9514 | 0.9574 | +0.0060 | 微升 |
| AGG | 0.3500 | 0.3557 | +0.0057 | 微升 |
| TGG | 0.3315 | 0.3527 | +0.0212 | 明确提升 |
| GGG | 0.4005 | 0.3930 | −0.0075 | 微降 |
| CGG | 0.3518 | 0.3716 | +0.0197 | 明确提升 |

> 🎯 **结论**：**无 PAM motif shortcut**。NGG 和 non-NGG 均有提升。BL6-1 的提升不是靠某个 PAM motif 的异常高分段拉动的。

### 3.4 Top-K Operating Points

来源：`results/bl6_1_validation/topk_operating_points_report.md`

| K | BL5 命中 | BL6-1 命中 | Δ | 判决 |
|:---:|---:|---:|---:|:---|
| 100 | 100 | 100 | 0 | 持平 |
| 500 | 500 | 500 | 0 | 持平 |
| **1,000** | **924** | 901 | **−23** | 🥇 **BL5 胜** |
| 2,000 | 1,247 | 1,264 | +17 | 🥇 BL6-1 胜 |
| 3,057 | 1,482 | 1,494 | +12 | 🥇 BL6-1 胜 |
| 5,000 | 1,787 | 1,831 | +44 | 🥇 BL6-1 胜 |
| 10,000 | 2,128 | 2,215 | +87 | 🥇 BL6-1 胜 |

> 🎯 **关键发现**：BL6-1 **不是 Top-K 全面胜利**。Top-100/500 持平，**Top-1000 BL5 更优（多找回 23 个真靶点）**，Top-2000+ BL6-1 更优。如果实验预算只够验证 1000 个候选位点，BL5-v4-PAM 目前仍是更好的选择。

---

## 4. 待补证据（❌ 未完成）

### 4.1 Gate Audit — ✅ 已完成 (Part 3, 2026-06-11)

| 问题 | 结果 |
|:---|:---|
| Gate 是否 collapse 到单一 view？ | **Yes** — near-collapse to LearnableRun。gate_argmax=run 占 954,055/954,326 (99.97%)，gate_argmax=pam 仅 271 行 |
| observed_positive vs unobserved_candidate 的 gate 分布是否不同？ | 基本一致 — 两组均为 Run-dominated，无实质性差异 |
| NGG vs non-NGG 的 gate 分布是否不同？ | 基本一致 — 两组 gate 行为相同 |
| Top-K 样本中 gate 行为是否有别？ | Top-100 中 gate_argmax=run 占 96%（vs 整体 99.97%），略低但仍 Run-dominated |

Gate audit details: `results/bl6_1_pam_gated_fusion/gate_audit/gate_audit_report.md`

Evidence impact: the near-collapse to LearnableRun **weakens** the interpretability story of sample-wise multi-view dynamic routing. The current gate audit does **not** support attributing the AUPRC gain to meaningful per-sample dynamic routing among RNA-FM, Run, and PAM views. Possible explanations (extra `z_weighted` branch, expanded head dimension, extra gate MLP parameters, training dynamics) remain hypotheses requiring follow-up ablation or multi-seed confirmation.

### 4.2 Multi-Seed Repeat — 高优先级

| 问题 | 当前状态 | 需要什么 |
|:---|:---|:---|
| BL6-1 的 +0.0086 是否跨 seed 稳定？ | 未知 | 至少再跑 2 个 seed（建议 43, 44） |
| BL6-1 AUPRC mean ± std？ | 未知 | 3 seeds 的汇总统计 |
| BL6-1 是否每个 seed 都 > BL5 historical best 0.5313？ | 未知 | 逐 seed 比较 |

> 注意：BL5-v4-PAM 自己也有训练波动（historical best 0.5313 vs latest rerun 0.5161，差 0.0152）。因此 BL6-1 的 seed repeat 不仅是为了看 BL6-1 自身是否稳定，也是为了判断 +0.0086 是否在 BL5 的训练波动范围之内。

### 4.3 执行报告中的两处硬伤 — ✅ 已修正（2026-06-11 Part 1）

| 硬伤 | 位置 | 修正状态 |
|:---|:---|:---|
| "Cross-Attn + Softmax Gate" | `report.md`, `summary.json`, `experiments.csv` notes | ✅ 已修正为 "PAM-Gated Fusion on BL5-v4-PAM backbone" |
| "Test 集全部为 NGG PAM" | `reborn_doc/26_...执行报告.md` | ✅ 已修正为 "test set contains both NGG and non-NGG PAM..." |

> 详见 `reborn_doc/52_BL6_1_Report_Correction_执行记录.md`。

---

## 5. 当前可以声称的结论

| ✅ 可以声称 | 支撑证据 |
|:---|:---|
| BL6-1 single-run AUPRC（0.5399）高于 BL5-v4-PAM 历史最佳（0.5313） | experiments.csv + 执行报告 |
| Bootstrap 支持该提升在当前 test cohort 上统计稳定（CI 不跨 0） | bootstrap CI [+0.0024, +0.0149] |
| 提升不是 PAM motif shortcut，NGG/non-NGG 均有增益 | per-PAM 分析 |
| BL6-1 在 Top-2000+ 上优于 BL5，但在 Top-1000 上不如 BL5 | top-k operating points |
| Per-sgRNA 提升不是普遍的（33/72 升 vs 31/72 降） | per-sgRNA 分析 |
| Gate audit 发现 sample-wise gate near-collapse to LearnableRun | gate_audit_report.md |

## 6. 当前不能声称的结论

| ❌ 不能声称 | 缺少的证据 |
|:---|:---|
| "BL6-1 已全面超越 BL5-v4-PAM" | Top-1000 BL5 更优；per-sgRNA 仅 45.8% 提升；multi-seed completed; mixed stability; seed44 below BL5 baseline |
| "BL6-1 是新的主模型" | multi-seed mixed；seed44 below BL5；gate near-collapse caveat remains |
| "Gate 学到了有意义的样本级权重" | Gate audit 发现 near-collapse to LearnableRun |
| "提升来自 PAM gate 的有效调制" | Gate audit 不支持 meaningful per-sample multi-view routing |
| "BL6-1 的训练是稳定的" | 仅 single-run |

---

## 7. 建议优先级

| 优先级 | 行动 | 预期产出 | 是否需 GPU |
|:---:|:---|:---|:---:|
| ✅ 已完成 | Gate export + audit（Part 2 + Part 3） | gate_predictions.csv + gate_audit/ + gate_audit_report.md | ✅ (done) |
| ✅ 已完成 | Report correction（Part 1） | 见 `reborn_doc/52_BL6_1_Report_Correction_执行记录.md` | 否 |
| ✅ 已完成 | BL6-1 multi-seed repeat（seed 43, 44） | mixed stability; seed44 below BL5; stable advantage not supported | ✅ (done) |
| 🟡 P1 | Gate-ablation follow-up（optional） | 去掉 gate 但保留 extra head dimension 消融实验 | ✅ 需 GPU 训练 |
| 🟢 P2 | BL5-v4-PAM multi-seed / variance reference（如需要） | BL5 training variance estimate | ✅ 需 GPU 训练 |

---

## 7.5. Part 1-4 + Multi-Seed 后最终证据边界

After Parts 1-4 and multi-seed repeat (seeds 42/43/44), the updated evidence boundary is:

**Evidence summary table:**

| Evidence item | Status | Current interpretation |
|:---|:---:|:---|
| Single-run AUPRC | ✅ | BL6-1 AUPRC 0.5399 > BL5-v4-PAM 0.5313 |
| Fixed-checkpoint bootstrap | ✅ | CI supports fixed-checkpoint AUPRC gap; not training stability |
| Report correction | ✅ | Template wording fixed |
| Gate export | ✅ | `gate_predictions.csv` exported; probability alignment passed |
| Gate audit | ✅ | Gate near-collapsed to LearnableRun; dynamic routing interpretation weakened |
| Multi-seed stability | ✅ | completed; seed42/43 above BL5, seed44 below; mean AUPRC 0.5244 < BL5 0.5313; stable advantage not supported |
| New main model status | ❌ | not approved — multi-seed mixed; can revert to BL5-v4-PAM |

**Final current boundary:**

After Parts 1-3, BL6-1 should be described as a **promising single-run architectural variant** with fixed-checkpoint bootstrap support but weak gate-routing interpretability. The learned gate is near-deterministically Run-dominated on the seed-42 formal test export (gate_argmax=run for 99.97% of samples, gate_entropy≈0). Therefore BL6-1 should **not** be promoted to the project main model until multi-seed repeat and/or follow-up ablation clarify stability and the source of gain.

## 8. 源码与结果索引

| 证据 | 位置 |
|:---|:---|
| 执行报告 | `reborn_doc/26_BL6_1_PAM_Gated_Fusion_执行报告.md` |
| AUPRC 指标解读 | `reborn_doc/27_AUPRC_AUROC_Precision_指标解读与全模型对照报告.md` |
| AUPRC threshold 解疑 | `reborn_doc/28_AUPRC_threshold_解疑实录.md` |
| Bootstrap CI 报告 | `results/bl6_1_validation/bootstrap_ci_report.md` |
| Per-sgRNA 报告 | `results/bl6_1_validation/per_sgrna_report.md` |
| Per-PAM 报告 | `results/bl6_1_validation/per_pam_report.md` |
| Top-K 报告 | `results/bl6_1_validation/topk_operating_points_report.md` |
| 总结报告 | `results/bl6_1_validation/bl6_1_validation_summary.md` |
| Config | `configs/bl6_1_pam_gated_fusion.yaml` |
| 训练结果 | `results/bl6_1_pam_gated_fusion/` |
| Gate predictions | `results/bl6_1_pam_gated_fusion/gate_predictions.csv` |
| Gate export validation | `results/bl6_1_pam_gated_fusion/gate_export_validation.json` |
| Gate audit report | `results/bl6_1_pam_gated_fusion/gate_audit/gate_audit_report.md` |
| Part 1 执行记录 | `reborn_doc/52_BL6_1_Report_Correction_执行记录.md` |
| Part 2 执行记录 | `reborn_doc/53_BL6_1_Gate_Export_执行记录.md` |
| Part 3 执行记录 | `reborn_doc/54_BL6_1_Gate_Audit_执行记录.md` |
| Part 4 执行记录 | `reborn_doc/55_BL6_1_Evidence_Boundary_Update_执行记录.md` |

---

## 9. 合规说明

- 本文档仅整理已有结果，未训练模型，未运行推理，未调用 GPU。
- 未删除或覆盖 data/ / reference/ 下任何文件。
- 未 commit / push。
- PAM 坐标统一为 `off_seq[20:23]`（positions 21-23）。
- label=0 解释为 unobserved_candidate，不是 safe site。
- BL6-1 真实机制是 PAM-Gated Fusion，不是 Cross-Attn + Softmax Gate。
- Test set 包含 NGG 和 non-NGG PAM，不是全部 NGG。
