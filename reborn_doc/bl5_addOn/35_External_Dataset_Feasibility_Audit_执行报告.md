# 外部数据集可行性审计执行报告 v2

> **任务**: BL5-v4-PAM 跨数据集泛化 — 外部数据集可行性审计 v2（完整剩余候选审计）
> **时间**: 2026-06-07
> **状态**: ✅ 已完成

---

## 1. 执行摘要

对仓库内**全部**可能作为 external dataset 的候选文件进行扫描和审计，判断是否存在可用于 BL5-v4-PAM 严格跨数据集评估的外部数据。

**核心结论**:
- 当前仓库内**没有可直接支持 strict raw cross-dataset AUROC/AUPRC evaluation 的外部数据集**。
- 旧 five-dataset `predictions.csv` 整体因 73.6% exact pair overlap 且为 old-model prediction artifact，不能作为整体 external benchmark。
- GUIDE-seq / CHANGE-seq / Tasi 子集也因高 pair overlap 不独立。
- **SITE 子集显示 0% exact pair overlap 且样本量充足**；**K562 子集显示 5.4% pair overlap 且 observed_positive / unobserved_candidate 数量达到最低阈值**；二者可作为 `provenance_required_limited_candidate`。
- 正式 external evaluation 前，必须确认这些 rows 的原始数据来源、label semantics、candidate generation 口径，并从 prediction artifact 中剥离 raw sgRNA/dna/label 表。

| 类别 | 数量 |
|:---|:---:|
| 扫描文件总数 | **876** |
| 外部数据集候选 | **38** |
| 严格外部评估可行 (ready_for_strict_external_eval) | **0** |
| 需来源确认有限候选 (provenance_required_limited_candidate) | **2** |
| 有限外部评估可行 (limited_external_eval_candidate) | **0** |
| 仅 smoke test | **1** |
| 不推荐 / 不可行 | **35** |

---

## 2. 扫描范围与审计方法

### 扫描范围（v2 扩大）
仓库内以下目录的 `.csv/.tsv/.parquet/.npz/.json/.jsonl/.fasta/.fa` 文件：
- `data/`
- `reference/`
- `results/`
- `output/`（v2 新增，v1 遗漏）
- `doc/`
- `reborn_doc/`
- `artifacts/`
- `offtarget_fusion_project/`
- 根目录文件（如 `test_20_samples.csv`）

### 排除规则
- 当前输出目录 `results/bl5_generalization/external_dataset_feasibility/`（避免自引用）
- `.git/`、`__pycache__/`
- checkpoint / model weight / `.pt` / `.pth` / `.ckpt`
- 图片、PDF、HTML、日志等明显非数据文件
- 训练 summary、`epoch_metrics.csv`、`config_used.json`
- CCLMoff 本体 `09212024_CCLMoff_dataset.csv` 及其派生 NPZ

### Inventory 分类体系
每个文件被赋予明确的 `inventory_class`：
- `raw_table_candidate` / `derived_result_artifact` / `metadata_only`
- `training_artifact` / `report_artifact` / `baseline_reference_cclmoff`
- `external_npz_candidate` / `sequence_fasta_unpaired`
- `config_or_manifest` / `model_weight_or_checkpoint`
- `too_large_needs_targeted_audit` / `non_data`

### 审计维度（v2 增强）
1. **Schema 兼容性**: sgRNA_seq / off_seq / label 或等价列识别
2. **标签语义**: clear_binary / positive_only / candidate_only / unclear
3. **序列规范性**: canonical 23nt, PAM 可提取性 (`off_seq[20:23]`)
4. **与 CCLMoff 重叠度（按 formal split 细分）**:
   - train exact pair overlap
   - val exact pair overlap
   - test exact pair overlap
   - any exact pair overlap
   - sgRNA overlap（辅助参考，不单独判死）
5. **可行性综合判断**: 样本量、sgRNA 覆盖、artifact 类型、overlap 程度

### Formal Split 基准
使用 `formal_split_bl5_seed42.json`，按 `sgRNA_type` 分组：
- Train: 150 sgRNA types, 3,247,056 unique pairs
- Val: 60 sgRNA types, 660,927 unique pairs
- Test: 72 sgRNA types, 885,864 unique pairs

---

## 3. 重点候选文件审计结果

### 3.1 `output/crispr_dualpred_five_dataset_full_20260507_204525/predictions.csv`

**整体 ALL**：
| 维度 | 结果 |
|:---|:---|
| 行数 | 1,513,878 |
| Schema | ✅ compatible (sgRNA, dna, label 齐全) |
| 标签 | ✅ clear_binary, observed_positive=35,985, unobserved_candidate=1,477,893, positive_ratio=2.4% |
| 序列 | ✅ all_canonical_23nt, 1,513,878 / 1,513,878 = 100% |
| PAM 提取 | ✅ `dna[20:23]` 可提取 |
| Train overlap | 682,127 pairs |
| Val overlap | 158,689 pairs |
| Test overlap | 141,708 pairs |
| Any overlap | 982,524 pairs (73.57%) |
| 可行性 | **overlap_not_independent** |

**Per-dataset 子集审计**:

| Subset | 行数 | Observed positive | Unobserved candidate | Unique sgRNA | Train/V/Test/Any Overlap | Overlap% | 状态 |
|:---|---:|---:|---:|---:|:---|---:|:---|
| GUIDE-seq | 520,281 | 1,123 | 519,158 | 249 | 375,344 / 86,828 / 58,093 / 520,265 | **100.00%** | overlap_not_independent |
| CHANGE-seq | 462,896 | 30,623 | 432,273 | 1,325 | 306,737 / 71,673 / 83,520 / 461,930 | **99.79%** | overlap_not_independent |
| Tasi | 294,534 | 354 | 294,180 | 36 | 142,063 / 0 / 6,712 / 148,775 | **50.51%** | overlap_not_independent |
| K562 | 18,434 | 118 | 18,316 | 12 | 719 / 188 / 95 / 1,002 | **5.44%** | provenance_required_limited_candidate |
| SITE | 217,733 | 3,767 | 213,966 | 9 | 0 / 0 / 0 / 0 | **0.00%** | provenance_required_limited_candidate |

**解读**:
- GUIDE-seq / CHANGE-seq 几乎 100% 重合于 CCLMoff，完全不能作为外部基准。
- Tasi 50.5% 重合，也不能独立。
- K562 仅 5.4% 重合，且 train/val/test 都有少量重叠，但 observed_positive=118、unobserved_candidate=18,316 达到最低阈值。因源自 prediction artifact，需 provenance 确认。
- SITE **0% exact pair overlap**，observed_positive=3,767、unobserved_candidate=213,966 样本充足，但 unique_sgRNA=9 < 10，且同样源自 prediction artifact。需 provenance 确认。

### 3.2 `test_20_samples.csv`

| 维度 | 结果 |
|:---|:---|
| 行数 | 20 |
| Schema | ✅ compatible (sgRNA, dna, label) |
| 标签 | ✅ clear_binary, observed_positive=10, unobserved_candidate=10 |
| 序列 | ✅ all_canonical_23nt |
| PAM 提取 | ✅ 可提取 |
| Train overlap | 0 |
| Val overlap | 1 |
| Test overlap | 0 |
| Any overlap | 1 (5.00%) |
| Unique sgRNA | 9 |
| 可行性 | **smoke_test_only** |

**解读**: 样本量过小，unique_sgRNA < 10，不足以支撑稳定 AUROC/AUPRC。仅可用于 smoke test。

### 3.3 其他模型预测 artifact（results/ 下 test_predictions.csv）

所有 `results/*/test_predictions.csv` 均为**本项目的模型预测输出**，与 CCLMoff test set 的 exact pair overlap = 100%（因为它们就是在 test set 上的预测）。这些文件**不能作为外部数据集**。

---

## 4. 可行性状态定义与本次分布

| 状态 | 定义 | 本次数量 | 代表候选 |
|:---|:---|:---:|:---|
| `ready_for_strict_external_eval` | 原始数据、schema 完整、双标签充足、pair overlap <1% 且 **train overlap=0**、sgRNA≥10 | 0 | — |
| `provenance_required_limited_candidate` | 大部分条件满足，但源自 result artifact 或需要确认数据来源/标签语义 | 2 | predictions.csv::K562, predictions.csv::SITE |
| `limited_external_eval_candidate` | 双标签齐全但样本量或 sgRNA 覆盖不足（非 heavy overlap） | 0 | — |
| `smoke_test_only` | 样本太小，无法稳定估计指标 | 1 | test_20_samples.csv |
| `overlap_not_independent` | pair overlap ≥10%，不能作为 strict external | 13 | predictions.csv::ALL, GUIDE-seq, CHANGE-seq, Tasi 等 |
| `infeasible` | 其他不满足条件（包括 schema incompatible、metadata only、result artifact 无充足样本等） | 22 | tier_labels.csv 等 |

**总计不推荐/不可行**: 35（= 22 infeasible + 13 overlap_not_independent）

---

## 5. 建议与后续工作

### 5.1 当前仓库结论
- **严格跨数据集评估**: ❌ 不可行（0 个 ready candidate）
- **来源确认后有限评估**: ⚠️ 可能（SITE、K562 子集需 provenance audit）
- **Smoke test**: ✅ 可用 `test_20_samples.csv`
- **PAM holdout 评估**: ✅ 可行，见 `reborn_doc/31_PAM_Holdout_Feasibility_Audit_执行报告.md`

### 5.2 SITE / K562 的 provenance audit 清单
如需将这两个子集推进到 external eval，必须确认：
1. **原始数据来源**: 这些 sgRNA/dna/label 是否来自原始实验数据，还是经过旧模型筛选/排序后的子集？
2. **Label 语义**: `label=1` 是否确实是 observed_positive，`label=0` 是否确实是 unobserved_candidate（而不是模型预测阈值后的伪标签）？
3. **Candidate generation 口径**: off-target 候选位点是如何生成的？是否与 CCLMoff 使用相同的 in-silico 搜索策略？
4. **序列一致性**: `dna` 列是否完全对应 `off_seq`，坐标方向是否与 CCLMoff 一致？
5. **去重检查**: 即使 exact pair overlap=0，是否有近邻重复或 sgRNA 序列的变体映射到同一基因组位点？

### 5.3 评估模型集（如未来 provenance 确认后）
- `BL0b-on-BL5split` (fine-tuned RNA-FM baseline)
- `BL5-v4-NoPAM-control` (no PAM)
- `BL5-v4-PAM` (full PAM model)
- `BL6-1-PAM-Gated-Fusion` (optional)

必须报告的指标：AUROC, AUPRC, positive_ratio, test_samples, observed_positive, unobserved_candidate, unique_sgRNA_count, NGG/non-NGG stratified metrics, bootstrap CI。

---

## 6. 输出产物清单

所有产物位于 `results/bl5_generalization/external_dataset_feasibility/`：

| # | 文件名 | 说明 |
|:---:|:---|:---|
| 1 | `external_dataset_inventory.csv` | 仓库内 876 个文件的完整清单与 inventory_class |
| 2 | `external_dataset_schema_audit.csv` | 候选文件的 schema 兼容性审计 |
| 3 | `external_dataset_label_audit.csv` | 标签语义、正负样本量、AUROC/AUPRC 可行性 |
| 4 | `external_dataset_sequence_pam_audit.csv` | 序列长度、canonical 23nt、PAM 提取能力 |
| 5 | `external_dataset_overlap_with_cclmoff.csv` | 与 CCLMoff train/val/test/any 的 exact pair 重叠度 |
| 6 | `external_dataset_feasibility_table.csv` | 综合可行性判定表（含 per-dataset rows，已清理无意义子集） |
| 7 | `recommended_external_eval.json` | JSON 格式推荐结论与下一步决策 |
| 8 | `external_dataset_feasibility_report.md` | 完整 Markdown 审计报告（术语统一为 observed_positive / unobserved_candidate） |

---

## 7. 合规声明

```
AGENTS.md compliance: [use_rnafm=False, freeze_rnafm=False,
                       split_mode=sgrna_safe, pos_weight=None]
确认本文件遵守 AGENTS.md 约束
```

- 本次审计**不涉及模型训练、推理、GPU 调用**
- 未修改任何已有代码、checkpoint、数据文件（仅重写了 audit 脚本自身）
- 仅执行读取扫描与审计分析
- PAM 坐标统一使用 `off_seq[20:23]`，未使用 `off_seq[-3:]`
- **未执行 git commit / push**
- `audit_compliance.py` 结果：0 ERROR，93 WARNING（全部为历史遗留 warning，非本次新增）
- 本任务没有主动修改无关文件；当前仓库仍有大量其他未提交工作（见 `git status --short`），后续提交前需要分 scope 核对
