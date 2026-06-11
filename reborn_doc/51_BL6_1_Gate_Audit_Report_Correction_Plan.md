# 51. BL6-1 Gate Audit + Report Correction Plan

> Date: 2026-06-11  
> Scope: BL6-1-PAM-Gated-Fusion evidence cleanup, gate audit export, and report correction  
> Status: Plan only. This document does not train, evaluate, or modify experiment artifacts.  
> Owner handoff target: Claude / DeepSeek custom model

---

## 1. Objective

BL6-1-PAM-Gated-Fusion is currently a promising single-run improvement over BL5-v4-PAM, but two evidence gaps must be closed before it can be presented cleanly:

1. **Gate audit is missing.**  
   `results/bl6_1_pam_gated_fusion/test_predictions.csv` contains probabilities but no per-sample gate weights. Therefore, we cannot yet answer whether the sample-wise gate actually uses RNA-FM / Run / PAM views in a meaningful way, or whether it collapsed to one view.

2. **Report / ledger wording has template errors.**  
   BL6-1 is **PAM-Gated Fusion on BL5-v4-PAM backbone**, not "Cross-Attn + Softmax Gate". Also, the test set is **not all NGG**; it contains both NGG and non-NGG PAM.

This plan turns BL6-1 from "single-run result with missing interpretability evidence" into a cleaner evidence package:

- corrected report and experiments ledger wording;
- per-sample gate weight export from the existing `best.pt`;
- gate distribution audit by label, PAM family, motif, and Top-K slice;
- a final evidence-boundary report that says what BL6-1 can and cannot claim.

---

## 2. Non-Negotiable Constraints

Follow `AGENTS.md`, `reborn_doc/1. 大纲拟定.md`, and `reborn_doc/todo.md`.

Hard constraints for this BL6-1 audit:

1. Do **not** train a new model in Parts 1-3.
2. Do **not** overwrite checkpoints.
3. Do **not** delete or overwrite `data/`, `reference/`, or existing raw result files.
4. Do **not** run `git commit` or `git push` unless the user explicitly asks.
5. Do **not** use `git add .`.
6. Use `best.pt` only for evaluation/export.
7. Keep `label=1` as `observed_positive`.
8. Keep `label=0` as `unobserved_candidate`; never write verified safe site.
9. Use PAM coordinate `off_seq[20:23]`, not `off_seq[-3:]`.
10. Report AUROC and AUPRC together whenever performance metrics are mentioned.
11. Do not claim BL6-1 is the new main model until gate audit and multi-seed evidence are complete.
12. Do not call bootstrap evidence "training stability"; bootstrap only measures test resampling uncertainty for fixed trained models.

---

## 3. Current Known Facts

### 3.1 Existing BL6-1 Files

Config and run files:

- `configs/bl6_1_pam_gated_fusion.yaml`
- `configs/bl6_1_pam_gated_fusion_smoke.yaml`
- `run/run_bl6_1_formal_2gpu.sh`

Reports:

- `reborn_doc/26_BL6_1_PAM_Gated_Fusion_执行报告.md`
- `reborn_doc/40_BL6_1_PAM_Gated_Fusion_证据边界与待补项.md`

Main result directory:

- `results/bl6_1_pam_gated_fusion/checkpoints/best.pt`
- `results/bl6_1_pam_gated_fusion/summary.json`
- `results/bl6_1_pam_gated_fusion/report.md`
- `results/bl6_1_pam_gated_fusion/epoch_metrics.csv`
- `results/bl6_1_pam_gated_fusion/test_predictions.csv`

Validation directory:

- `results/bl6_1_validation/bootstrap_ci_report.md`
- `results/bl6_1_validation/per_sgrna_report.md`
- `results/bl6_1_validation/per_pam_report.md`
- `results/bl6_1_validation/topk_operating_points_report.md`
- `results/bl6_1_validation/bl6_1_validation_summary.md`

### 3.2 Current BL6-1 Metrics

From existing reports:

| Item | Value |
|:---|---:|
| Test samples | 954,326 |
| observed_positive | 3,057 |
| unobserved_candidate | 951,269 |
| BL6-1 AUROC | 0.984993 |
| BL6-1 AUPRC | 0.539917 |
| BL5-v4-PAM historical AUPRC | 0.531281 |
| BL6-1 - BL5 AUPRC | +0.0086 |
| BL6-1 best epoch | 8 |
| Training status | single-run only |

Bootstrap result already exists:

| Comparison | Delta AUPRC | 95% CI | Interpretation |
|:---|---:|:---|:---|
| BL6-1 - BL5-v4-PAM | +0.0087 | [+0.0024, +0.0149] | CI does not cross 0 for fixed trained checkpoints |

Top-K caveat already exists:

| K | BL5 hits | BL6-1 hits | Delta | Interpretation |
|:---:|---:|---:|---:|:---|
| 100 | 100 | 100 | 0 | tie |
| 500 | 500 | 500 | 0 | tie |
| 1,000 | 924 | 901 | -23 | BL5 better |
| 2,000 | 1,247 | 1,264 | +17 | BL6-1 better |
| 5,000 | 1,787 | 1,831 | +44 | BL6-1 better |
| 10,000 | 2,128 | 2,215 | +87 | BL6-1 better |

### 3.3 Known Report Errors

Wrong text currently appears in BL6-1 artifacts:

- `results/bl6_1_pam_gated_fusion/summary.json`
- `results/bl6_1_pam_gated_fusion/report.md`
- `results/experiments.csv`, rows for:
  - `BL6-1-PAM-Gated-Fusion-Smoke`
  - `BL6-1-PAM-Gated-Fusion`

Wrong wording:

```text
Cross-Attn + Softmax Gate
```

Correct wording:

```text
PAM-Gated Fusion on BL5-v4-PAM backbone
```

Also wrong in the execution report:

```text
Test 集全部为 NGG PAM
```

Correct wording:

```text
test set contains both NGG (819,984, 85.9%) and non-NGG (134,342, 14.1%) PAM; canonical PAM distribution matches BL5-v4-PAM formal test set.
```

---

## 4. Relevant Code Facts

### 4.1 Model Gate Weights Already Exist Internally

File:

- `models/bl5_dynamic_fusion.py`

In `BL5RunOnlyDynamicFusion.forward(..., return_aux=True)`, when `fusion_type == "pam_gated_fusion"`:

```python
z_rna_proj = self.rnafm_proj(z_rna)
z_pam_proj = self.pam_proj(z_pam)
view_summary = torch.cat([z_rna_proj, z_run, z_pam], dim=-1)
gate_logits = self.gate_mlp(view_summary)
gate = torch.softmax(gate_logits, dim=-1)
z_weighted = gate[:, 0:1] * z_rna_proj + gate[:, 1:2] * z_run + gate[:, 2:3] * z_pam_proj
fused = torch.cat([z_rna, z_run, z_pam, z_weighted], dim=-1)
aux = {"gate_weights": gate}
```

Gate column meaning:

| Gate index | Meaning |
|:---:|:---|
| `gate[:, 0]` | RNA-FM view weight |
| `gate[:, 1]` | LearnableRun view weight |
| `gate[:, 2]` | PAM view weight |

Therefore, gate audit does **not** require model architecture invention. It only requires exporting `aux["gate_weights"]` during eval-only inference.

### 4.2 Current Export Path Does Not Write Gate Weights

File:

- `scripts/train_bl5.py`

Current helper:

- `predict_probabilities(...)` calls `model(...)` without `return_aux=True`.
- `write_test_predictions(...)` only writes probability and metadata columns.
- `train_one_epoch(...)` only calls `return_aux=True` when `gate_l2_lambda > 0.0`; this is training-side regularization and should not be repurposed as an export switch.

Current output columns in `test_predictions.csv`:

```text
sample_index, sgRNA_type, on_seq, off_seq, PAM_original, PAM_shuffled, PAM, label, probability, Direction, split
```

Target gate-audit prediction export should preserve these and add:

```text
gate_rnafm, gate_run, gate_pam
```

Optional but useful:

```text
gate_entropy, gate_max, gate_argmax, pam_family
```

Do not overwrite original `test_predictions.csv` during gate export. Use a new file:

```text
results/bl6_1_pam_gated_fusion/gate_predictions.csv
```

---

## 5. Work Breakdown

### Part 1 - Preflight + Report Correction Plan Review

Goal: no GPU work, no model forward pass. Make sure Claude understands the current state and fixes only obvious wording errors.

Allowed actions:

- read required docs;
- inspect `git status`;
- inspect BL6-1 config/report/summary/experiments rows;
- correct wording in reports / summary / experiments ledger;
- write a small correction report.

Forbidden actions:

- no training;
- no eval-only inference;
- no checkpoint loading unless just checking file existence;
- no new gate export script yet;
- no commit / push.

Expected outputs:

- corrected `results/bl6_1_pam_gated_fusion/report.md`;
- corrected `results/bl6_1_pam_gated_fusion/summary.json` notes only;
- corrected BL6-1 rows in `results/experiments.csv` notes only;
- corrected `reborn_doc/26_BL6_1_PAM_Gated_Fusion_执行报告.md`;
- corrected `reborn_doc/40_BL6_1_PAM_Gated_Fusion_证据边界与待补项.md` if it repeats stale wording;
- new short report: `reborn_doc/52_BL6_1_Report_Correction_执行记录.md`;
- no metric changes.

Acceptance checks:

```bash
rg -n "Cross-Attn \\+ Softmax Gate|Test 集全部为 NGG|全部为 NGG PAM" \
  results/bl6_1_pam_gated_fusion/report.md \
  results/bl6_1_pam_gated_fusion/summary.json \
  results/experiments.csv \
  reborn_doc/26_BL6_1_PAM_Gated_Fusion_执行报告.md \
  reborn_doc/40_BL6_1_PAM_Gated_Fusion_证据边界与待补项.md
```

Expected: no matches, except if a document explicitly lists the old phrase as "old wording" in a correction table. Prefer no matches for simplicity.

### Part 2 - Gate Export Code Design

Goal: design and implement the smallest safe path to export per-sample gate weights from the existing `best.pt`.

Preferred implementation:

1. Add a new helper in `scripts/train_bl5.py` or a separate script.
2. Safer option: create a dedicated script:

```text
scripts/export_bl6_1_gate_predictions.py
```

This avoids destabilizing `train_bl5.py`.

The script should:

- include AGENTS.md compliance header;
- load `configs/bl6_1_pam_gated_fusion.yaml`;
- force eval-only semantics;
- load `results/bl6_1_pam_gated_fusion/checkpoints/best.pt`;
- build the same formal test dataset;
- call model forward with `return_aux=True`;
- collect probability and `gate_weights`;
- write `results/bl6_1_pam_gated_fusion/gate_predictions.csv`;
- not append to `results/experiments.csv`;
- not overwrite `test_predictions.csv`, `summary.json`, or `report.md`;
- run on a single GPU by default to avoid DDP row-order complexity.

Required gate export columns:

| Column | Meaning |
|:---|:---|
| `sample_index` | original row index |
| `sgRNA_type` | guide group |
| `on_seq` | sgRNA sequence |
| `off_seq` | off-target sequence |
| `PAM_original` | `off_seq[20:23]` |
| `label` | 0/1 |
| `probability` | model probability |
| `gate_rnafm` | RNA-FM gate weight |
| `gate_run` | LearnableRun gate weight |
| `gate_pam` | PAM gate weight |
| `gate_sum` | sum of three gates, should be near 1 |
| `gate_entropy` | `-sum(g * log(g))` |
| `gate_max` | max gate value |
| `gate_argmax` | one of `rnafm`, `run`, `pam` |
| `pam_family` | `NGG` if pattern N-G-G, otherwise `non-NGG` |
| `split` | `test` |

Validation requirements:

- row count = `954,326`;
- observed_positive = `3,057`;
- unobserved_candidate = `951,269`;
- `PAM_original == off_seq[20:23]`;
- probability in `[0, 1]`;
- all gates in `[0, 1]`;
- `abs(gate_sum - 1) <= 1e-5` for all rows, or explain float tolerance;
- exported probability should match existing `test_predictions.csv` within small tolerance after row alignment.

### Part 3 - Gate Audit Analysis

Goal: use `gate_predictions.csv` to answer whether gate behavior is meaningful or collapsed.

Create:

```text
results/bl6_1_pam_gated_fusion/gate_audit_summary.csv
results/bl6_1_pam_gated_fusion/gate_audit_by_label.csv
results/bl6_1_pam_gated_fusion/gate_audit_by_pam_family.csv
results/bl6_1_pam_gated_fusion/gate_audit_by_pam_motif.csv
results/bl6_1_pam_gated_fusion/gate_audit_topk.csv
results/bl6_1_pam_gated_fusion/gate_audit.md
```

Minimum statistics:

- mean / std / median / p05 / p25 / p75 / p95 for `gate_rnafm`, `gate_run`, `gate_pam`;
- mean / median `gate_entropy`;
- proportion with `gate_argmax == rnafm/run/pam`;
- max gate concentration:
  - `gate_max >= 0.80`;
  - `gate_max >= 0.90`;
  - `gate_max >= 0.95`;
- label-stratified stats:
  - observed_positive vs unobserved_candidate;
- PAM-family stats:
  - NGG vs non-NGG;
- per-motif stats for major motifs:
  - AGG, TGG, GGG, CGG, GAG, CAG, etc.;
- Top-K stats for K:
  - 100, 500, 1000, 2000, 3057, 5000, 10000.

Interpretation rules:

- If one view has mean gate > 0.90 and `gate_argmax` proportion > 95%, say "gate appears collapsed".
- If gates vary by label / PAM family / Top-K but do not collapse, say "gate is sample-dependent".
- Do not claim causal mechanism. Gate audit is descriptive.
- Do not claim PAM gate caused AUPRC gain unless supported by perturbation or ablation.

### Part 4 - Evidence Boundary Update

Goal: update BL6-1 narrative after gate audit.

Update:

- `reborn_doc/40_BL6_1_PAM_Gated_Fusion_证据边界与待补项.md`
- create `reborn_doc/53_BL6_1_Gate_Audit_执行报告.md`

The updated boundary should classify:

| Question | Status after gate audit |
|:---|:---|
| BL6-1 single-run AUPRC > BL5? | yes |
| Bootstrap CI for fixed checkpoints? | yes |
| Gate collapse checked? | yes, after Part 3 |
| Gate mechanism proven? | no, descriptive only |
| Training seed stability? | no, still needs seed 43/44 |
| New main model? | still pending multi-seed unless user decides otherwise |

### Part 5 - Optional Plot Package

Only after Parts 1-4 pass, optionally create figures:

```text
results/figures/bl6_1/gate_weight_distribution.png
results/figures/bl6_1/gate_by_label_boxplot.png
results/figures/bl6_1/gate_by_pam_family_boxplot.png
results/figures/bl6_1/gate_topk_profile.png
```

Figures are optional. CSV + markdown audit is sufficient for the first pass.

### Part 6 - Multi-Seed Repeat Plan

Not part of this immediate task. After gate audit:

- BL6-1 seed 43;
- BL6-1 seed 44;
- compare AUPRC mean ± std;
- compare BL6-1 vs BL5-v4-PAM across seeds;
- do not conflate this with bootstrap.

---

## 6. Part 1 Detailed Prompt for Claude

The prompt below is intentionally detailed. It is designed for a fresh agent that has not worked on this repository.

```text
你现在接手 /data/zwf/code1/reborn_seed 这个 repo 的 BL6-1 第一阶段任务。请严格按步骤执行，不要发挥，不要扩大范围。

任务名称：
BL6-1 Part 1 - Preflight + Report Correction

本阶段目标：
只做低风险文字修正和状态核对。不要训练，不要 eval-only 推理，不要导出 gate weights，不要写 gate audit 脚本。你这一步只是把 BL6-1 现有报告里的明显模板错误修掉，并写一份执行记录。

必须先读：
1. AGENTS.md
2. reborn_doc/1. 大纲拟定.md
3. reborn_doc/todo.md
4. reborn_doc/51_BL6_1_Gate_Audit_Report_Correction_Plan.md
5. reborn_doc/26_BL6_1_PAM_Gated_Fusion_执行报告.md
6. reborn_doc/40_BL6_1_PAM_Gated_Fusion_证据边界与待补项.md

开始前运行：
git status --short --branch
python scripts/audit_compliance.py

你必须在心里确认：
- 当前任务不训练。
- 当前任务不推理。
- 当前任务不改 checkpoint。
- 当前任务不改 prediction CSV。
- 当前任务不改任何 BL5 已封口文件。
- 当前任务不 commit / push。
- 当前任务不 git add .。
- label=1 叫 observed_positive。
- label=0 叫 unobserved_candidate，不是 safe site。
- PAM_original 必须是 off_seq[20:23]。

背景事实：
BL6-1 的真实机制是：
PAM-Gated Fusion on BL5-v4-PAM backbone

它不是：
Cross-Attn + Softmax Gate

原因：
models/bl5_dynamic_fusion.py 中 fusion_type == "pam_gated_fusion" 时，是对 RNA-FM / LearnableRun / PAM 三个 view 做 sample-wise softmax gate。没有 cross-attention。

BL6-1 当前核心数字，不要改：
- AUROC = 0.984993
- AUPRC = 0.539917
- best_epoch = 8
- status = completed
- test samples = 954,326
- observed_positive = 3,057
- unobserved_candidate = 951,269

本阶段要修的错误有两个：

错误 1：
报告或台账里写了：
Cross-Attn + Softmax Gate

应该改成：
PAM-Gated Fusion on BL5-v4-PAM backbone

允许更完整地写：
Fine-tuned RNA-FM + LearnableRunEncoder + PAM-Gated Fusion on BL5-v4-PAM backbone; rna_pooling=cls; use_pam_encoder=True; focal_loss gamma=2.0; gate_l2_lambda=0; early_stopping_patience=None; best.pt test evaluation

错误 2：
执行报告里写了：
Test 集全部为 NGG PAM
或
全部为 NGG PAM

应该改成：
test set contains both NGG (819,984, 85.9%) and non-NGG (134,342, 14.1%) PAM; canonical PAM distribution matches BL5-v4-PAM formal test set.

注意：
不要重新计算这些数字。先使用 reborn_doc/40 和已有 per-PAM 报告中的数字。

允许修改的文件：
1. results/bl6_1_pam_gated_fusion/report.md
   - 只修 notes 文字，不改 AUROC/AUPRC/epoch/status。

2. results/bl6_1_pam_gated_fusion/summary.json
   - 只修 notes 字段里的 "Cross-Attn + Softmax Gate"。
   - 不改 test_metrics。
   - 不改 status。
   - 不改 best_epoch。
   - 不改 train_seconds。
   - 不改 artifact paths。

3. results/experiments.csv
   - 只修 BL6-1 相关两行 notes：
     a. BL6-1-PAM-Gated-Fusion-Smoke
     b. BL6-1-PAM-Gated-Fusion
   - 不改 AUROC。
   - 不改 AUPRC。
   - 不改 train_time。
   - 不改 status。
   - 不新增行。
   - 不删除行。

4. reborn_doc/26_BL6_1_PAM_Gated_Fusion_执行报告.md
   - 修正 "Cross-Attn + Softmax Gate" 描述。
   - 修正 "Test 集全部为 NGG PAM"。
   - 降级 "Strong Success / 新王 / 全面超越" 等过强措辞，如果存在明显越界。
   - 推荐口径：single-run promising improvement；fixed-checkpoint bootstrap supports AUPRC gain；仍待 gate audit 和 multi-seed。

5. reborn_doc/40_BL6_1_PAM_Gated_Fusion_证据边界与待补项.md
   - 如果里面已经写清楚 gate audit 未做、multi-seed 未做，可以少改。
   - 如果仍有旧模板错误，修掉。
   - 保持谨慎边界：不能说 BL6-1 已经是新主模型。

6. 新建执行记录：
   reborn_doc/52_BL6_1_Report_Correction_执行记录.md

执行记录必须包含：
- 本阶段目标
- 修改文件清单
- 每个文件改了什么
- 未改什么
- 验证命令和结果
- 合规声明
- 下一阶段建议：Part 2 gate export script design

具体操作建议：

第一步：定位旧模板错误
运行：
rg -n "Cross-Attn \\+ Softmax Gate|Test 集全部为 NGG|全部为 NGG PAM" \
  results/bl6_1_pam_gated_fusion/report.md \
  results/bl6_1_pam_gated_fusion/summary.json \
  results/experiments.csv \
  reborn_doc/26_BL6_1_PAM_Gated_Fusion_执行报告.md \
  reborn_doc/40_BL6_1_PAM_Gated_Fusion_证据边界与待补项.md

把命中的地方记录下来。

第二步：修 report.md
文件：
results/bl6_1_pam_gated_fusion/report.md

只把 notes 那一行从 Cross-Attn + Softmax Gate 改成 PAM-Gated Fusion on BL5-v4-PAM backbone。
不要改指标。

第三步：修 summary.json
文件：
results/bl6_1_pam_gated_fusion/summary.json

只改 "notes" 字段。
注意 JSON 必须仍然合法。改完后运行：
python -m json.tool results/bl6_1_pam_gated_fusion/summary.json >/tmp/bl6_summary_check.json

第四步：修 experiments.csv
文件：
results/experiments.csv

只改 BL6-1 两行 notes 里的旧短语。
建议用文本编辑或小心的脚本，但不能重排 CSV。
改完检查：
rg -n "BL6-1-PAM-Gated-Fusion" results/experiments.csv

确认：
- 仍然只有 smoke 和 formal 两行 BL6-1。
- 两行 status 仍是 completed。
- 没有新增 eval-only 行。

第五步：修执行报告 26
文件：
reborn_doc/26_BL6_1_PAM_Gated_Fusion_执行报告.md

必须修：
- Cross-Attn + Softmax Gate -> PAM-Gated Fusion on BL5-v4-PAM backbone
- Test 集全部为 NGG PAM -> test set contains both NGG and non-NGG PAM...

建议顺手把过强措辞改谨慎：
- "新王" 改为 "promising single-run improvement"
- "Strong Success" 改为 "Promising single-run result"
- "全面超越" 改为 "AUPRC higher in this run, but Top-K and per-sgRNA evidence are mixed; gate audit and multi-seed pending"

不要改表格里的指标数字。

第六步：检查 40 边界报告
文件：
reborn_doc/40_BL6_1_PAM_Gated_Fusion_证据边界与待补项.md

如果已正确写：
- gate audit 未做
- multi-seed 未做
- 不能声称新主模型
则不要大改。

只修旧模板错误。

第七步：新建执行记录
文件：
reborn_doc/52_BL6_1_Report_Correction_执行记录.md

建议结构：
# 52. BL6-1 Report Correction 执行记录

## 1. 任务范围
说明本次只修文档和台账措辞，未训练，未推理。

## 2. 修改文件
表格列出每个文件和修改内容。

## 3. 修正前后
列出：
Cross-Attn + Softmax Gate -> PAM-Gated Fusion on BL5-v4-PAM backbone
Test 全部 NGG -> test contains both NGG and non-NGG

## 4. 验证
写你实际跑的命令和结果。

## 5. 合规
未训练、未 eval、未改 checkpoint、未改 prediction CSV、未 commit/push。

## 6. 下一步
Part 2: implement gate export script to produce gate_predictions.csv.

第八步：最终验证
运行：
rg -n "Cross-Attn \\+ Softmax Gate|Test 集全部为 NGG|全部为 NGG PAM" \
  results/bl6_1_pam_gated_fusion/report.md \
  results/bl6_1_pam_gated_fusion/summary.json \
  results/experiments.csv \
  reborn_doc/26_BL6_1_PAM_Gated_Fusion_执行报告.md \
  reborn_doc/40_BL6_1_PAM_Gated_Fusion_证据边界与待补项.md \
  reborn_doc/52_BL6_1_Report_Correction_执行记录.md

预期：无匹配。

运行：
python -m json.tool results/bl6_1_pam_gated_fusion/summary.json >/tmp/bl6_summary_check.json

预期：exit code 0。

运行：
python scripts/audit_compliance.py

记录 ERROR/WARNING。不要修历史 warning。

运行：
git status --short --branch

确认：
- 没有 data/reference 改动。
- 没有 checkpoint 改动。
- 没有 prediction CSV 改动。
- 没有 commit/push。

最终回报格式：

1. 修改文件清单
2. 旧短语清理结果
3. experiments.csv BL6-1 行检查结果
4. JSON 合法性检查结果
5. audit_compliance.py 结果
6. git status 摘要
7. 合规声明

合规声明必须包含：
- 未训练
- 未 eval-only 推理
- 未改 checkpoint
- 未改 prediction CSV
- 未改 data/reference
- 未 commit/push

不要做 Part 2。Part 2 是 gate export script，等我确认 Part 1 后再启动。
```

---

## 7. Suggested Part 1 Acceptance Summary

Part 1 is complete only if all are true:

- no old BL6-1 wording remains in the specified files;
- BL6-1 metrics are unchanged;
- `summary.json` remains valid JSON;
- `experiments.csv` still has exactly the original BL6-1 smoke/formal records and no eval-only pollution;
- `reborn_doc/52_BL6_1_Report_Correction_执行记录.md` exists;
- `audit_compliance.py` has `ERROR=0`;
- no commit/push was performed.

After Part 1, hand the updated report back for Codex review before starting Part 2.

