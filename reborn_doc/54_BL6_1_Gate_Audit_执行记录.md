# 54. BL6-1 Gate Audit 执行记录

> Date: 2026-06-11  
> Phase: Part 3 — Gate Audit Analysis  
> Executor: Claude  
> Plan: `reborn_doc/51_BL6_1_Gate_Audit_Report_Correction_Plan.md`  

---

## 1. 任务范围

对 BL6-1 已导出的 `gate_predictions.csv` 做描述性 gate audit。**未训练、未做模型 forward、未加载 checkpoint、未修改任何实验产物。**

---

## 2. 输入文件

| 文件 | 用途 |
|:---|:---|
| `results/bl6_1_pam_gated_fusion/gate_predictions.csv` | 954,326 行 gate weight 数据 (Part 2 导出, AMP-disabled) |
| `results/bl6_1_pam_gated_fusion/gate_export_validation.json` | Part 2 validation（probability alignment 已确认） |

## 3. 新增脚本

`scripts/audit_bl6_1_gates.py` — 纯 pandas/numpy 分析脚本，含 AGENTS.md compliance header。

运行命令：
```bash
python scripts/audit_bl6_1_gates.py \
  --input results/bl6_1_pam_gated_fusion/gate_predictions.csv \
  --output-dir results/bl6_1_pam_gated_fusion/gate_audit
```

## 4. 输出文件列表

| 文件 | 内容 |
|:---|:---|
| `gate_audit_input_validation.json` | 输入验证结果（全部通过） |
| `gate_audit_overall.json` | 全局 gate 分布统计 |
| `gate_audit_overall_summary.csv` | 全局 gate 分布摘要表 |
| `gate_audit_by_label.csv` | 按 observed_positive / unobserved_candidate 分层 |
| `gate_audit_by_pam_family.csv` | 按 NGG / non-NGG 分层 |
| `gate_audit_by_pam_motif.csv` | 按 63 个 PAM motif 分层 |
| `gate_audit_by_sgrna_type.csv` | 按 72 个 sgRNA_type 分层 |
| `gate_audit_topk.csv` | Top-K 切片 gate 统计 (K=100/500/1000/2000/5000/10000) |
| `gate_audit_probability_bins.csv` | 按 probability 分箱 gate 统计 |
| `gate_audit_extreme_gate_rows.csv` | 极端 gate 样本 (top 100 gate_pam / rnafm / entropy / probability) |
| `gate_audit_report.md` | 审计报告 |

## 5. 核心结果

### 5.1 Overall Gate Distribution

| Statistic | gate_rnafm | gate_run | gate_pam |
|:---|---:|---:|---:|
| mean | 4.21e-09 | 0.999712 | 0.000288 |
| median (p50) | 3.05e-09 | 0.999945 | 0.000055 |
| p99 | 1.83e-08 | 0.999997 | 0.003419 |

| gate_argmax | Count | Fraction |
|:---|---:|---:|
| run | 954,055 | 99.9716% |
| pam | 271 | 0.0284% |
| rnafm | 0 | 0% |

| Threshold fraction | gate_run ≥0.99 | gate_pam ≥0.99 |
|:---|---:|---:|
| | 99.9665% | 0.0273% |

| gate_entropy mean | gate_entropy median |
|---:|---:|
| 2.38e-05 | 4.91e-07 |

### 5.2 主要发现（谨慎措辞）

**BL6-1 sample-wise gate 已近乎完全 collapse 到 LearnableRun 视图。** 99.97% 的 test 样本 gate_argmax=run，gate entropy 接近零，表明 gate 没有进行有意义的 per-sample 多视图动态路由。

这意味着：
- 当前 gate audit **不支持**将 BL6-1 的 +0.0086 AUPRC 提升解释为有意义的 per-sample 多视图动态路由。
- 可能解释包括额外的 `z_weighted` 分支、head 输入维度扩展 (912 vs 784)、gate MLP 额外参数 (38K)、或训练动力学差异；这些只是 hypotheses，需要后续 ablation 或 multi-seed 验证。
- Gate collapse 是 BL6-1 seed 42 上的训练结果，不排除其他 seed 下 gate 行为不同。

### 5.3 当前 BL6-1 状态

| 问题 | 状态 |
|:---|:---|
| Single-run AUPRC > BL5 historical best? | ✅ yes |
| Bootstrap CI supports AUPRC gain? | ✅ yes |
| Gate dynamically routes between views? | ❌ no — near-total collapse to Run |
| Training seed stability? | ❌ single-run only |
| Can become new main model? | ❌ not yet |

## 6. 合规声明

- 未训练 ✅
- 未做模型 forward ✅
- 未加载 checkpoint ✅
- 未覆盖 gate_predictions.csv / test_predictions.csv ✅
- 未改 summary.json / report.md / experiments.csv ✅
- 未改 data/ / reference/ / checkpoints/ ✅
- 未 commit / push ✅
- 所有 `label=0` 写作 unobserved_candidate ✅
- PAM_original 使用 off_seq[20:23] ✅
- 未称 BL6-1 为新主模型 ✅

## 7. 下一步

**Part 4**: 用 gate audit 结果更新 `reborn_doc/40_BL6_1_PAM_Gated_Fusion_证据边界与待补项.md`，将 gate audit 状态从 ❌ 未做 改为 ✅ 已完成，并记录 gate collapse 发现。
