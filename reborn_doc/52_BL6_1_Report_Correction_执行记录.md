# 52. BL6-1 Report Correction 执行记录

> Date: 2026-06-11  
> Phase: Part 1 — Preflight + Report Correction  
> Executor: Claude  
> Plan: `reborn_doc/51_BL6_1_Gate_Audit_Report_Correction_Plan.md`

---

## 1. 任务范围

本阶段仅做低风险文字修正和状态核对。**未训练模型，未做 eval-only 推理，未导出 gate weights，未写 gate audit 脚本。**

目标：修正 BL6-1 现有报告/台账中的两处模板错误，并降级过强措辞。

---

## 2. 修改文件清单

| # | 文件 | 修改内容 |
|:---|:---|:---|
| 1 | `results/bl6_1_pam_gated_fusion/report.md` | notes 行 `Cross-Attn + Softmax Gate` → `PAM-Gated Fusion on BL5-v4-PAM backbone`；指标未改 |
| 2 | `results/bl6_1_pam_gated_fusion/summary.json` | `notes` 字段同上修正；`test_metrics`、`status`、`best_epoch`、`train_seconds`、`artifact paths` 均未改；JSON 合法性验证通过 |
| 3 | `results/experiments.csv` | BL6-1 smoke (#20) 和 formal (#21) 两行 notes 中 `Cross-Attn + Softmax Gate` → `PAM-Gated Fusion on BL5-v4-PAM backbone`；AUROC/AUPRC/status 未改；无新增/删除行 |
| 4 | `reborn_doc/26_BL6_1_PAM_Gated_Fusion_执行报告.md` | (a) 结论标签 `Strong Success` → `Promising single-run result`；(b) 横向对比表 `🥇 旧王` / `👑 新王` → `BL5 baseline` / `Promising single-run improvement`；(c) `Test 集全部为 NGG PAM` → `test set contains both NGG (819,984, 85.9%) and non-NGG (134,342, 14.1%) PAM...`；(d) 已知问题表 #1 状态从 `后续修正模板` → `✅ 已修正` |
| 5 | `reborn_doc/40_BL6_1_PAM_Gated_Fusion_证据边界与待补项.md` | (a) 当前状态表 `执行报告中的两个硬伤` 从 `⚠️ 描述错误有待修正` → `✅ 已修正`；(b) §4.3 硬伤详情表改为已修正状态，引用本执行记录 |
| 6 | `reborn_doc/52_BL6_1_Report_Correction_执行记录.md` | 新建（本文档） |
| 7 | `reborn_doc/26_BL6_1_PAM_Gated_Fusion_执行报告.md` §9 | 追加修正：结论与下一步降级为 gate audit / multi-seed pending；移除 BL6-2 作为下一步 |
| 8 | `reborn_doc/40_BL6_1_PAM_Gated_Fusion_证据边界与待补项.md` §7 | 追加修正：建议优先级表将 report correction 从 🔴 P0 待办改为 ✅ 已完成 |

---

## 3. 修正前后对照

| 旧短语 | 新短语 |
|:---|:---|
| `Cross-Attn + Softmax Gate` | `PAM-Gated Fusion on BL5-v4-PAM backbone` |
| `Test 集全部为 NGG PAM` | `test set contains both NGG (819,984, 85.9%) and non-NGG (134,342, 14.1%) PAM; canonical PAM distribution matches BL5-v4-PAM formal test set.` |
| `Strong Success` | `Promising single-run result` |
| `👑 新王` / `🥇 旧王` | `Promising single-run improvement` / `BL5 baseline` |

---

## 4. 验证

### 4.1 旧短语清理

```bash
rg -n "Cross-Attn \+ Softmax Gate|Test 集全部为 NGG|全部为 NGG PAM" \
  results/bl6_1_pam_gated_fusion/report.md \
  results/bl6_1_pam_gated_fusion/summary.json \
  results/experiments.csv \
  reborn_doc/26_BL6_1_PAM_Gated_Fusion_执行报告.md \
  reborn_doc/40_BL6_1_PAM_Gated_Fusion_证据边界与待补项.md
```

结果：无实际旧模板残留。命中项均在「已修正」状态记录或合规声明中，属预期行为。

### 4.2 experiments.csv BL6-1 行

```
20:BL6-1-PAM-Gated-Fusion-Smoke ... PAM-Gated Fusion on BL5-v4-PAM backbone ...
21:BL6-1-PAM-Gated-Fusion      ... PAM-Gated Fusion on BL5-v4-PAM backbone ...
```

- 仍仅 smoke 和 formal 两行 ✅
- AUROC/AUPRC 未变 ✅
- 无新增 eval-only 行 ✅

### 4.3 JSON 合法性

```bash
python -m json.tool results/bl6_1_pam_gated_fusion/summary.json >/tmp/bl6_summary_check.json
# exit code: 0
```

### 4.4 过强措辞清理

```bash
rg -n "新王|旧王|全面超越|Strong Success" reborn_doc/26_BL6_1_PAM_Gated_Fusion_执行报告.md
# exit code: 1 (no matches)
```

### 4.5 audit_compliance.py

```
ERROR: 0
WARNING: 95（全部为历史遗留 Python 文件/未跟踪脚本，非本次修改引入）
```

### 4.6 Git 状态

```
Branch: feat/bl5-generalization
Modified (tracked): results/experiments.csv
Untracked: BL6 configs/scripts/reports（此前遗留，非本次新增）
No data/reference changes
No checkpoint changes
No commit/push
```

---

## 5. 合规声明

- 未训练 ✅
- 未 eval-only 推理 ✅
- 未改 checkpoint ✅
- 未改 prediction CSV ✅
- 未改 data/ / reference/ ✅
- 未 commit / push ✅

---

## 6. 下一步

Part 2: implement gate export script (`scripts/export_bl6_1_gate_predictions.py`) to produce `results/bl6_1_pam_gated_fusion/gate_predictions.csv` from existing `best.pt`，再进入 Part 3 gate audit analysis。
