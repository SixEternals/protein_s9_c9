# Cross-PAM Strict Holdout Feasibility Plan

> 目的：先判断 CCLMoff formal BL5 split 是否适合做严格 Cross-PAM 泛化实验。当前阶段只做 feasibility audit，不直接训练 PAM-holdout 模型。

---

## 1. 背景与核心问题

当前 BL5 已经完成了 `sgRNA-safe` formal split 上的主模型、消融和 PAM shuffle control。现有结果能说明：

- test set 是 held-out sgRNA group，不是训练集结果；
- BL5-v4-PAM 在 same-dataset unseen-sgRNA 上强于 BL0b / NoPAM / shuffle；
- PAM shuffle-control 证明正确 PAM 与样本对应关系很重要。

但这仍然不是严格 Cross-PAM generalization。

严格 Cross-PAM holdout 要回答的是：

```text
如果训练阶段完全不见某个 PAM motif H，
模型能否在测试阶段对 PAM=H 的样本仍然保持有效排序？
```

这和 NGG / non-NGG stratified evaluation 不同。NGG/non-NGG 分层只是“测试后切片分析”，训练阶段仍可能见过这些 PAM motif；而 strict PAM-holdout 要求训练和验证阶段彻底排除 heldout PAM。

---

## 2. 绝对坐标约束

所有 PAM 统计必须使用项目约定：

```python
PAM_original = off_seq[20:23]
```

禁止使用：

```python
off_seq[-3:]
```

原因：

- 项目坐标约定中 positions `1-20` 是 protospacer；
- positions `21-23` 是 PAM；
- Python 0-based index 对应 `off_seq[20:23]`；
- 使用 `off_seq[-3:]` 会在含 gap、长度混合或非标准序列时产生错误 PAM 口径。

报告中必须明确写：

```text
PAM motif was extracted from positions 21-23 using off_seq[20:23].
```

---

## 3. 第一阶段：只做 PAM Motif Feasibility Audit

### 3.1 任务名称

```text
PAM-holdout feasibility audit
```

### 3.2 建议新增脚本

```text
scripts/audit_pam_holdout_feasibility.py
```

脚本要求：

- 可重复运行；
- 带 `argparse`；
- 不训练模型；
- 不修改任何现有结果；
- 只读取 formal split 与数据文件；
- 输出 CSV / JSON / Markdown 报告。

### 3.3 建议命令

```bash
python scripts/audit_pam_holdout_feasibility.py \
  --formal_split_json formal_split_bl5_seed42.json \
  --output_dir results/bl5_generalization/pam_holdout_feasibility \
  --pam_start 20 \
  --pam_end 23 \
  --min_test_positive 100 \
  --min_test_unobserved 1000 \
  --min_test_sgrna_types 10 \
  --min_train_positive_after_exclusion 1000 \
  --min_val_positive_after_exclusion 100
```

如果项目中的 formal split json 路径不在根目录，应由脚本自动搜索常见位置，或由参数显式传入。

---

## 4. Audit 需要统计什么

### 4.1 每个 split 的 PAM motif 分布

对 `train` / `val` / `test` 分别统计：

```text
split
PAM_original
samples
observed_positive
unobserved_candidate
positive_ratio
sgRNA_type_count
```

输出：

```text
results/bl5_generalization/pam_holdout_feasibility/pam_motif_by_split_counts.csv
```

### 4.2 每个 PAM motif 的 holdout 可行性

对每个候选 PAM motif `H`，模拟 strict holdout：

```text
train_H_excluded = formal_train where PAM_original != H
val_H_excluded   = formal_val   where PAM_original != H
test_H_only      = formal_test  where PAM_original == H
```

统计：

```text
holdout_pam
train_remaining_samples
train_remaining_observed_positive
train_remaining_unobserved_candidate
train_remaining_sgRNA_type_count
val_remaining_samples
val_remaining_observed_positive
val_remaining_unobserved_candidate
val_remaining_sgRNA_type_count
test_H_samples
test_H_observed_positive
test_H_unobserved_candidate
test_H_positive_ratio
test_H_sgRNA_type_count
test_H_ngg_status
feasibility_status
risk_flags
recommendation
```

输出：

```text
results/bl5_generalization/pam_holdout_feasibility/pam_holdout_candidate_table.csv
```

---

## 5. Feasibility 判定规则

### 5.1 Feasible

满足以下条件才标记为 `feasible`：

```text
test_H_observed_positive >= 100
test_H_unobserved_candidate >= 1000
test_H_sgRNA_type_count >= 10
train_remaining_observed_positive >= 1000
val_remaining_observed_positive >= 100
test_H 同时包含 observed_positive 和 unobserved_candidate
```

含义：

- test set 至少有足够 positive 支撑 AUPRC；
- test set 也有足够 unobserved_candidate，避免单类测试；
- heldout PAM 覆盖多个 sgRNA_type，避免单 sgRNA shortcut；
- 排除该 PAM 后 train/val 仍能正常训练和选 best checkpoint。

### 5.2 Marginal

如果满足双类别，但样本量偏少，标记为 `marginal`：

```text
20 <= test_H_observed_positive < 100
或
200 <= test_H_unobserved_candidate < 1000
或
3 <= test_H_sgRNA_type_count < 10
```

含义：

- 可以做探索性分析；
- 不能作为强泛化证据；
- AUPRC / AUROC 方差会很大；
- 必须配 bootstrap CI。

### 5.3 Infeasible

出现以下情况标记为 `infeasible`：

```text
test_H_observed_positive < 20
或
test_H_unobserved_candidate < 200
或
test_H 只有单一类别
或
test_H_sgRNA_type_count < 3
或
排除 H 后 train/val observed_positive 明显不足
```

含义：

- 不建议训练 strict PAM-holdout；
- 即使跑出指标，也很可能只是数据偏差或单类切片结果；
- 应写作 limitation/future work。

---

## 6. Risk Flags

`pam_holdout_candidate_table.csv` 中的 `risk_flags` 建议用分号拼接，例如：

```text
single_class_test;too_few_observed_positive;too_few_sgRNA_type
```

建议 flags：

| flag | 含义 |
|:---|:---|
| `single_class_test` | test_H 只有 observed_positive 或只有 unobserved_candidate，AUROC/AUPRC 不可正常解释。 |
| `too_few_observed_positive` | test_H positive 太少，AUPRC 不稳定。 |
| `too_few_unobserved_candidate` | test_H unobserved_candidate 太少，排序任务不成立。 |
| `too_few_sgRNA_type` | heldout PAM 只覆盖少数 sgRNA，可能被 sgRNA 特异性主导。 |
| `extreme_positive_ratio_shift` | heldout PAM 的 positive ratio 与 overall test 差异过大。 |
| `train_too_small_after_exclusion` | 排除该 PAM 后 train positive 不足。 |
| `val_too_small_after_exclusion` | 排除该 PAM 后 val positive 不足，best.pt 选择不稳定。 |
| `motif_has_only_observed_positive` | 该 motif 只出现在 observed_positive 中，可能是 shortcut。 |
| `motif_has_only_unobserved_candidate` | 该 motif 只出现在 unobserved_candidate 中，无法评估召回。 |

---

## 7. 推荐输出文件

```text
results/bl5_generalization/pam_holdout_feasibility/
├── pam_motif_by_split_counts.csv
├── pam_holdout_candidate_table.csv
├── recommended_holdout_motifs.json
└── pam_holdout_feasibility_report.md
```

### 7.1 `recommended_holdout_motifs.json`

建议结构：

```json
{
  "coordinate_contract": "PAM_original = off_seq[20:23]",
  "formal_split_json": "formal_split_bl5_seed42.json",
  "recommended_for_training": ["AGG"],
  "marginal_candidates": ["TGG"],
  "not_recommended": ["AAA"],
  "decision": "run_strict_holdout_if_at_least_one_feasible_candidate_exists"
}
```

实际 PAM motif 由 audit 结果决定，不能提前写死。

### 7.2 `pam_holdout_feasibility_report.md`

报告必须包含：

```markdown
# PAM Holdout Feasibility Audit

## 1. Purpose
说明为什么 strict PAM-holdout 是泛化实验，不是普通分层评估。

## 2. Coordinate Contract
说明 PAM_original = off_seq[20:23]，禁止 off_seq[-3:]。

## 3. Formal Split Summary
报告 train / val / test 样本数、observed_positive、unobserved_candidate、sgRNA_type_count。

## 4. PAM Motif Distribution
列出各 split PAM motif 分布。

## 5. Holdout Candidate Table
列出 feasible / marginal / infeasible PAM motif。

## 6. Recommended Next Step
明确是否建议训练 strict PAM-holdout。

## 7. Limitations
说明如果候选 PAM 类别少、positive 少或 sgRNA 覆盖不足，不能把结果解释成强泛化证据。
```

---

## 8. 第二阶段：只有 audit 通过才训练

如果至少有一个 `feasible` PAM motif，才考虑训练 strict PAM-holdout。

### 8.1 严格 split 定义

对 heldout PAM `H`：

```text
train = formal_train rows where PAM_original != H
val   = formal_val   rows where PAM_original != H
test  = formal_test  rows where PAM_original == H
```

禁止：

```text
train 中出现 PAM_original == H
val 中出现 PAM_original == H
test 中出现 PAM_original != H
```

### 8.2 必跑模型

如果要跑 strict PAM-holdout，至少成对跑：

```text
BL5-v4-PAM-holdout-H
BL5-v4-NoPAM-holdout-H
```

原因：

- 单独跑 PAM 模型无法判断 PAM Encoder 是否贡献泛化；
- 必须和 NoPAM 同 holdout split 对照；
- 如果 PAM-holdout-H 低于 NoPAM-holdout-H，说明 heldout PAM 上 PAM Encoder 没有帮助；
- 如果 PAM-holdout-H 高于 NoPAM-holdout-H，才说明 PAM Encoder 对未见 PAM motif 有泛化价值。

### 8.3 可选模型

如果资源允许，再补：

```text
BL0b-on-PAM-holdout-H
BL6-1-PAM-Gated-holdout-H
```

但优先级低于 BL5-v4-PAM / NoPAM 成对实验。

---

## 9. 建议配置字段

如果训练 strict PAM-holdout，需要新增或扩展 config 字段：

```yaml
split_mode: sgrna_safe

split:
  strategy: formal_group_json_pam_holdout
  formal_split_json: formal_split_bl5_seed42.json
  group_column: sgRNA_type
  holdout_pam: AGG
  pam_source: off_seq_positions_21_23
  train_exclude_holdout_pam: true
  val_exclude_holdout_pam: true
  test_only_holdout_pam: true

model:
  use_rnafm: true
  freeze_rnafm: false
```

注意：

- `use_rnafm` 必须显式声明；
- 如果 `use_rnafm: true`，`freeze_rnafm` 必须显式声明；
- `split_mode` 必须显式声明；
- test 必须加载 `checkpoints/best.pt`；
- AUROC 和 AUPRC 必须同时报告。

---

## 10. Holdout Audit 必须写入训练结果

如果进入训练阶段，每个结果目录必须包含：

```text
pam_holdout_audit.json
pam_holdout_audit.md
```

必须检查并报告：

```text
train_H_count = 0
val_H_count = 0
test_non_H_count = 0
train_samples
val_samples
test_samples
train_observed_positive
val_observed_positive
test_observed_positive
train_unobserved_candidate
val_unobserved_candidate
test_unobserved_candidate
test_sgRNA_type_count
```

如果任一硬条件不满足，必须停止训练或停止解释结果。

---

## 11. 评价指标

strict PAM-holdout 训练完成后必须报告：

```text
AUROC
AUPRC
Accuracy
Precision
Recall
F1
best_epoch
best_val_AUPRC
test_samples
test_observed_positive
test_unobserved_candidate
test_sgRNA_type_count
```

如果 heldout test 只有单一类别：

```text
AUROC/AUPRC undefined because only one class is present.
```

不要强行计算或用 1.0 / 0.0 填充。

---

## 12. 决策规则

### 情况 A：存在 feasible PAM motif

推荐下一步：

```text
训练 BL5-v4-PAM-holdout-H 与 BL5-v4-NoPAM-holdout-H。
```

解释边界：

```text
这是 same-dataset strict PAM-motif holdout，不是 cross-dataset、cross-cell-line 或 cross-species 泛化。
```

### 情况 B：只有 marginal PAM motif

推荐下一步：

```text
可以做探索性 holdout，但必须配 bootstrap CI，并在报告中写明样本量限制。
```

解释边界：

```text
只能作为 supplementary / sanity check，不能作为强主结论。
```

### 情况 C：没有可用 PAM motif

推荐下一步：

```text
不做 strict PAM-holdout training。
```

报告表述：

```text
We audited strict PAM-holdout feasibility, but no PAM motif had sufficient class balance, sample size, and sgRNA coverage to support a reliable heldout-PAM generalization experiment. Therefore, Cross-PAM strict holdout is reported as infeasible on the current CCLMoff formal split and left as future work pending a suitable dataset.
```

---

## 13. 给 Kimi / Claude 的执行提示词

可以直接把下面这一段交给 Kimi 或 Claude：

```text
请在 /data/zwf/code1/reborn_seed 中执行 PAM-holdout feasibility audit。严格遵守 AGENTS.md：先读 reborn_doc/1. 大纲拟定.md，检查 git status --short，不要 commit/push，不要训练模型。

任务目标：
新增 scripts/audit_pam_holdout_feasibility.py，用于判断 formal BL5 split 是否适合做 strict Cross-PAM holdout generalization。

硬性坐标：
PAM_original 必须等于 off_seq[20:23]，对应 positions 21-23。禁止使用 off_seq[-3:]。

输入：
formal_split_bl5_seed42.json 以及项目中 BL5 formal split 使用的数据源。若路径不确定，先在代码中查找 train_bl5.py / dataset loader 如何读取 formal split，不要猜。

输出目录：
results/bl5_generalization/pam_holdout_feasibility/

必须输出：
1. pam_motif_by_split_counts.csv
2. pam_holdout_candidate_table.csv
3. recommended_holdout_motifs.json
4. pam_holdout_feasibility_report.md

统计内容：
对 train / val / test 分别统计每个 PAM_original 的 samples、observed_positive、unobserved_candidate、positive_ratio、sgRNA_type_count。
对每个候选 heldout PAM H，模拟：
train = formal_train where PAM_original != H
val = formal_val where PAM_original != H
test = formal_test where PAM_original == H
并统计 train/val 剩余样本和 positive 数，以及 test_H 的 samples、observed_positive、unobserved_candidate、positive_ratio、sgRNA_type_count。

feasibility_status：
feasible：test_H_observed_positive >= 100，test_H_unobserved_candidate >= 1000，test_H_sgRNA_type_count >= 10，train_remaining_observed_positive >= 1000，val_remaining_observed_positive >= 100，且 test_H 双类别齐全。
marginal：双类别齐全但样本量或 sgRNA 覆盖偏少。
infeasible：单类别、positive 太少、unobserved_candidate 太少、sgRNA 覆盖太少或排除 H 后 train/val 不足。

risk_flags 至少包含：
single_class_test、too_few_observed_positive、too_few_unobserved_candidate、too_few_sgRNA_type、extreme_positive_ratio_shift、train_too_small_after_exclusion、val_too_small_after_exclusion、motif_has_only_observed_positive、motif_has_only_unobserved_candidate。

报告必须明确：
1. 这是 feasibility audit，不是训练结果。
2. NGG/non-NGG stratified evaluation 不等于 strict PAM-holdout。
3. strict PAM-holdout 只有在存在 feasible candidate 时才建议训练。
4. 如果没有 feasible PAM motif，应明确写 Cross-PAM strict holdout 在当前 formal split 上不可可靠执行，只能作为 future work。

完成后不要 commit/push。只汇报 changed files、outputs、关键 candidate 表、是否建议进入训练阶段。
```

---

## 14. 一句话结论

当前不应直接启动 Cross-PAM strict holdout 训练。正确顺序是先做 PAM motif feasibility audit：确认每个 PAM motif 在 formal train/val/test 中的样本量、类别平衡和 sgRNA 覆盖是否足够。只有 audit 显示至少一个 PAM motif 满足双类别、足够 positive、足够 unobserved_candidate、足够 sgRNA 覆盖，并且排除该 motif 后 train/val 仍可训练，才进入 BL5-v4-PAM-holdout-H 与 BL5-v4-NoPAM-holdout-H 的成对训练。
