# 55. BL6-1 Evidence Boundary Update 执行记录

> Date: 2026-06-11  
> Phase: Part 4 — Evidence Boundary Update  
> Executor: Claude  
> Input: Part 1 (52), Part 2 (53), Part 3 (54), gate_audit_report.md  

---

## 1. 任务范围

Part 4 evidence boundary update only. 将 Part 1-3 的最终证据边界同步到 BL6-1 主文档。未训练、未推理、未加载 checkpoint、未重跑任何脚本。

## 2. 修改文件

| 文件 | 修改内容 |
|:---|:---|
| `reborn_doc/26_BL6_1_PAM_Gated_Fusion_执行报告.md` | 已知问题 #2 更新为已完成；结论表 gate audit 行更新为 near-collapse 发现；核心结论重写；下一步更新（Part 2-4 done → multi-seed）；新增 §10 Gate Audit Update 小节 |
| `reborn_doc/40_BL6_1_PAM_Gated_Fusion_证据边界与待补项.md` | 状态表拆分 gate export/audit 为 completed；§4.1 重写为已完成 gate audit 总结；§5-6 更新可以/不能声称项；§7 优先级表更新 gate audit 为已完成 + 新增 gate-ablation follow-up；新增 §7.5 最终证据边界；§8 索引新增 Part 1-4 文档和 gate artifacts |
| `reborn_doc/55_BL6_1_Evidence_Boundary_Update_执行记录.md` | 新建（本文档） |

## 3. 更新状态表

| Evidence item | Status | Interpretation |
|:---|:---:|:---|
| Single-run AUPRC | ✅ | BL6-1 AUPRC 0.5399 > BL5-v4-PAM 0.5313 |
| Fixed-checkpoint bootstrap | ✅ | CI supports fixed-checkpoint AUPRC gap; not training stability |
| Report correction | ✅ | Template wording fixed |
| Gate export | ✅ | `gate_predictions.csv` exported; probability alignment passed |
| Gate audit | ✅ | Gate near-collapsed to LearnableRun; dynamic routing interpretation weakened |
| Multi-seed stability | ❌ | pending |
| New main model status | ❌ | not approved |

## 4. 当前 BL6-1 最终证据边界

BL6-1 is a **promising single-run improvement** over BL5-v4-PAM with fixed-checkpoint bootstrap support, but gate audit shows **near-collapse to LearnableRun** and weakens the intended dynamic multi-view routing interpretation. It should **not** be promoted to the current main model before multi-seed validation and/or follow-up ablation.

## 5. 合规声明

- 未训练 ✅
- 未 eval-only inference ✅
- 未加载 checkpoint ✅
- 未重跑 gate export / gate audit ✅
- 未改 prediction CSV ✅
- 未改 checkpoint ✅
- 未改 data/ / reference/ ✅
- 未写 experiments.csv ✅
- 未 commit / push ✅
