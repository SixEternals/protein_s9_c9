你现在作为“本地代码审计者 + 实验执行者 + 结果核验者”，请严格执行 **BL5-v4-PAM-shuffle-control**。不要自由发挥，不要跑 Cross-Attn、SeedWeightedRun、BL6 或任何新模型。当前唯一任务是完成 PAM shuffle control，用来验证 BL5-v4-PAM 的性能提升是否真的依赖正确的 PAM 信息，而不是其他实现差异或数据偏差。

# 一、当前背景

目前已经完成：

## 1. BL0b-on-BL5split

正式 RNA-FM baseline：

```text
模型：RNA-FM CLS + official MLP 640 → 64 → 1
split：formal_split_bl5_seed42.json
test_samples = 954,326
test_positive = 3,057
test_negative = 951,269
test_sgRNA_type_count = 72
test AUROC = 0.8578
test AUPRC = 0.2957
```

## 2. BL5-v4-NoPAM-control

无 PAM 的 v4 对照：

```text
模型：RNA-FM CLS + LearnableRun + v4 classifier，无 PAM Encoder
test AUROC = 0.984098
test AUPRC = 0.502389
```

## 3. BL5-v4-PAM

当前主模型：

```text
模型：RNA-FM CLS + LearnableRun + PAM Encoder + v4 classifier
test AUROC = 0.984194
test AUPRC = 0.531281
```

当前三段式贡献拆解为：

```text
NoPAM - BL0b = 0.502389 - 0.2957 = +0.206689
PAM - NoPAM = 0.531281 - 0.502389 = +0.028892
PAM - BL0b = 0.531281 - 0.2957 = +0.235581
```

严谨解释：

```text
NoPAM − BL0b = BL5-v4 无 PAM 框架的综合增益
PAM − NoPAM = PAM Encoder 在 v4 框架下的近似净贡献
PAM − BL0b = 完整 BL5-v4-PAM 框架相对纯 RNA-FM baseline 的总增益
```

注意：不要把 NoPAM − BL0b 简单解释为 LearnableRun 的纯贡献。

---

# 二、本任务的核心问题

当前还没有完成：

```text
BL5-v4-PAM-shuffle-control
```

该实验的目标是回答：

> BL5-v4-PAM 的额外 +0.0289 AUPRC 是否真的来自正确 PAM 与样本之间的对应关系？

也就是说，要验证：

```text
真实 PAM 编码 > 随机打乱 PAM 编码
```

如果真实 PAM 明显优于 shuffle PAM，说明 PAM Encoder 的确利用了正确 PAM 信息。

如果 shuffle PAM 仍接近真实 PAM，说明 PAM 增益可能不是 PAM 本身导致，可能来自 classifier、训练噪声、数据偏差或实现问题。

如果 shuffle PAM 高于真实 PAM，必须立即检查 PAM 实现、split、label 或数据泄漏。

---

# 三、总体原则

请严格遵守：

1. 不改 BL5-v4-PAM 的模型主体结构。
2. 不改 label。
3. 不改 sgRNA / off_seq。
4. 不改 RNA-FM tokens。
5. 不改 LearnableRun / run_features。
6. 不改 formal split。
7. 只改变 PAM feature 与样本之间的对应关系。
8. 使用 formal_split_bl5_seed42.json。
9. 使用 best.pt 做 test evaluation。
10. 必须同时报告 AUROC 和 AUPRC。
11. 必须导出 test_predictions.csv。
12. 必须生成 summary.json、report.md、epoch_metrics.csv。
13. 必须追加 results/experiments.csv。
14. 如果 test_samples、test_positive、test_negative、test_sgRNA_type_count 与 BL5-v4-PAM 不一致，立即停止并报告。
15. 不要把该实验解释成 NoPAM；它不是关闭 PAM，而是保留 PAM Encoder，但输入的 PAM 对应关系被打乱。

---

# 四、第一步：审计现有配置

请先检查是否已有：

```text
configs/bl5_v4_pam_shuffle_control.yaml
```

以及是否已有结果目录：

```text
results/bl5_v4_pam_shuffle_control/
```

如果已有配置但没有结果，请审计配置是否正确。

如果没有配置，请从：

```text
configs/bl5_v4_pam.yaml
```

复制生成：

```text
configs/bl5_v4_pam_shuffle_control.yaml
```

配置必须满足：

```yaml
version: "BL5-v4-PAM-shuffle-control"
output_dir: "results/bl5_v4_pam_shuffle_control"

model:
  use_rnafm: true
  freeze_rnafm: false
  use_learnable_run: true
  use_pam_encoder: true
  pam_dim: 16
  rna_pooling: cls
  fusion_type: simple_concat
  d_model: 128
  rnafm_dim: 640

training:
  focal_loss: true
  focal_gamma: 2.0
```

必须保持与 BL5-v4-PAM 完全一致的训练参数：

```text
batch_size
eval_batch_size
lr_transformer
lr_run_encoder
lr_pam_encoder
lr_mlp
weight_decay
epochs
dropout
dropout2
optimizer
scheduler
loss
gradient_clip
precision
ddp 设置
```

新增或确认：

```yaml
shuffle_pam: true
shuffle_pam_mode: "within_split"
shuffle_pam_seed: 42
formal_group_json: "formal_split_bl5_seed42.json"
```

如果当前代码还没有这些字段，允许你添加，但必须保持向后兼容：普通 BL5-v4-PAM 不受影响。

---

# 五、第二步：实现 PAM shuffle 逻辑

## 5.1 shuffle 的对象

只允许 shuffle：

```text
pam_features
```

也就是由 off_seq positions 21-23 生成的 PAM one-hot 特征。

不允许 shuffle：

```text
label
off_seq
on_seq
sgRNA_type
RNA-FM tokens
run_features
seed_weights
Direction
sample_index
```

## 5.2 shuffle 的方式

请使用：

```text
within_split shuffle
```

即：

```text
train 内部随机打乱 train 的 pam_features
val 内部随机打乱 val 的 pam_features
test 内部随机打乱 test 的 pam_features
```

不要把 train 的 PAM 打乱到 test，也不要跨 split 混合。

原因：

```text
避免引入 split 间信息泄漏，同时保证每个 split 内 PAM 分布不变。
```

## 5.3 shuffle 的可复现性

使用固定随机种子：

```text
shuffle_pam_seed = 42
```

必须在 summary.json 里记录：

```text
shuffle_pam = true
shuffle_pam_mode = within_split
shuffle_pam_seed = 42
```

## 5.4 shuffle 正确性自检

训练前必须输出并保存 shuffle 审计：

```text
results/bl5_v4_pam_shuffle_control/pam_shuffle_audit.json
results/bl5_v4_pam_shuffle_control/pam_shuffle_audit.md
```

审计内容包括：

```text
train 原始 PAM 分布
train shuffle 后 PAM 分布
val 原始 PAM 分布
val shuffle 后 PAM 分布
test 原始 PAM 分布
test shuffle 后 PAM 分布
```

要求：

```text
shuffle 前后 PAM 分布完全一致或仅有浮点/排序差异
```

还要检查：

```text
same_position_ratio_train
same_position_ratio_val
same_position_ratio_test
```

定义：

```text
same_position_ratio = shuffle 后某样本 PAM 与原始 PAM 完全相同的比例
```

由于 PAM 分布本身不均衡，该比例不一定接近 0，但必须报告。

同时报告：

```text
number_of_samples_with_changed_PAM
number_of_samples_with_unchanged_PAM
```

如果 changed 数量异常低，说明 shuffle 可能没有生效，必须停止。

---

# 六、第三步：训练 BL5-v4-PAM-shuffle-control

运行训练：

```bash
torchrun --nproc_per_node=2 scripts/train_bl5.py \
  --config configs/bl5_v4_pam_shuffle_control.yaml \
  --output_dir results/bl5_v4_pam_shuffle_control
```

如果本地项目使用其他启动脚本，请保持与 BL5-v4-PAM 相同的启动方式，只改变 config 和 output_dir。

训练要求：

1. 使用 formal_split_bl5_seed42.json。
2. 使用 best.pt 做最终 test evaluation。
3. 同时报告 AUROC 和 AUPRC。
4. 必须导出 test_predictions.csv。
5. 必须写 summary.json。
6. 必须写 report.md。
7. 必须写 epoch_metrics.csv。
8. 必须追加 results/experiments.csv。
9. 如果出现 NaN，必须停止并从 best.pt 恢复评估，标记 status 为 completed_recovered_best_after_nan。

---

# 七、第四步：输出 test_predictions.csv

文件路径：

```text
results/bl5_v4_pam_shuffle_control/test_predictions.csv
```

必须包含字段：

```text
sample_index
sgRNA_type
on_seq
off_seq
PAM_original
PAM_shuffled
label
probability
Direction
split
```

如果实现上只能保存一个 PAM 字段，请至少保存：

```text
PAM_original
PAM_shuffled
```

原因：后续要检查 shuffle 是否真正改变了样本级 PAM 对应关系。

---

# 八、第五步：正式指标对比

完成后，生成：

```text
results/bl5_v4_pam_shuffle_control/report.md
```

报告必须包含以下表格：

## 8.1 四模型总表

比较：

```text
BL0b-on-BL5split
BL5-v4-NoPAM-control
BL5-v4-PAM
BL5-v4-PAM-shuffle-control
```

表格字段：

```text
model
PAM setting
test AUROC
test AUPRC
Accuracy
Precision
Recall
F1
best_epoch
best_val_AUPRC
test_samples
test_positive
test_negative
```

## 8.2 关键差值

计算：

```text
NoPAM_minus_BL0b = NoPAM_AUPRC - BL0b_AUPRC
PAM_minus_NoPAM = PAM_AUPRC - NoPAM_AUPRC
Shuffle_minus_NoPAM = Shuffle_AUPRC - NoPAM_AUPRC
PAM_minus_Shuffle = PAM_AUPRC - Shuffle_AUPRC
Shuffle_minus_BL0b = Shuffle_AUPRC - BL0b_AUPRC
```

已知值：

```text
BL0b_AUPRC = 0.2957
NoPAM_AUPRC = 0.502389
PAM_AUPRC = 0.531281
```

你需要补：

```text
Shuffle_AUPRC = ?
```

解释逻辑：

```text
如果 PAM_minus_Shuffle 明显 > 0：
    真实 PAM 对应关系有价值。

如果 Shuffle_minus_NoPAM 接近 0：
    shuffle PAM 退化到 NoPAM 附近，说明 PAM 分支需要正确 PAM 才有用。

如果 Shuffle_minus_NoPAM 明显 > 0：
    即使 PAM 被打乱，PAM 分支仍有增益，可能来自 PAM 分布、额外参数量或训练噪声，需要谨慎。

如果 PAM_minus_Shuffle 接近 0：
    真实 PAM 与随机 PAM 没区别，PAM Encoder 的生物学解释很弱。

如果 Shuffle_AUPRC > PAM_AUPRC：
    立即标红，检查实现或数据泄漏。
```

---

# 九、第六步：分层评估

如果已有 test_predictions.csv，请在 shuffle-control 完成后立即做分层评估。

比较模型：

```text
BL0b-on-BL5split
BL5-v4-NoPAM-control
BL5-v4-PAM
BL5-v4-PAM-shuffle-control
```

分层：

```text
All test
NGG-only test: PAM_original ∈ {AGG, TGG, GGG, CGG}
non-NGG-only test: PAM_original not in {AGG, TGG, GGG, CGG}
```

每个 subset 报告：

```text
samples
positive
negative
positive_ratio
AUROC
AUPRC
Accuracy
Precision
Recall
F1
mean probability positive
mean probability negative
median probability positive
median probability negative
probability > 0.5 ratio for positives
probability > 0.5 ratio for negatives
```

注意：如果 non-NGG-only 全是 positive，则 AUROC/AUPRC 不可定义。此时不要强行计算，要写：

```text
AUROC/AUPRC undefined because only one class is present.
```

但仍然报告：

```text
recall
mean probability
median probability
probability > 0.5 ratio
```

输出：

```text
results/stratified_metrics_all_ngg_nongg_with_shuffle.csv
results/stratified_metrics_all_ngg_nongg_with_shuffle.md
```

核心问题：

```text
1. BL5-v4-PAM 在 NGG-only 上是否仍明显优于 NoPAM？
2. shuffle PAM 在 NGG-only 上是否下降？
3. BL5-v4-PAM 的提升是否主要来自 non-NGG shortcut？
```

---

# 十、第七步：paired comparison

把以下模型的 test_predictions 合并：

```text
BL0b-on-BL5split
BL5-v4-NoPAM-control
BL5-v4-PAM
BL5-v4-PAM-shuffle-control
```

生成：

```text
results/paired_comparison_with_shuffle.csv
```

字段：

```text
sample_index
sgRNA_type
on_seq
off_seq
PAM_original
label
prob_bl0b
prob_nopam
prob_pam
prob_shuffle
delta_nopam_minus_bl0b
delta_pam_minus_nopam
delta_shuffle_minus_nopam
delta_pam_minus_shuffle
```

然后生成：

```text
results/paired_comparison_with_shuffle_report.md
```

分别统计：

```text
all samples
positive samples
negative samples
NGG-only samples
non-NGG samples
```

每层报告：

```text
mean delta_pam_minus_shuffle
median delta_pam_minus_shuffle
proportion delta_pam_minus_shuffle > 0
proportion delta_pam_minus_shuffle < 0
```

核心问题：

```text
真实 PAM 是否主要提高 positive 的概率，而不是同时抬高大量 negative 的概率？
```

---

# 十一、最终总报告

生成：

```text
results/bl5_v4_pam_shuffle_control/final_shuffle_control_report.md
```

报告结构：

```markdown
# BL5-v4-PAM Shuffle Control Report

## 1. Executive Summary
用 5-8 句话总结：
- 本实验为什么做
- shuffle PAM 怎么做
- test set 是否一致
- shuffle AUPRC 是多少
- 真实 PAM 是否优于 shuffle PAM
- 是否支持 PAM Encoder 的真实贡献

## 2. Experimental Setup
说明：
- 使用 formal_split_bl5_seed42.json
- 模型结构与 BL5-v4-PAM 一致
- 仅打乱 pam_features
- train/val/test 内部分别 shuffle
- seed=42

## 3. PAM Shuffle Audit
说明：
- shuffle 前后 PAM 分布是否一致
- 有多少样本 PAM 发生改变
- same_position_ratio 是多少

## 4. Main Results
四模型对比：
- BL0b
- NoPAM
- PAM
- PAM-shuffle

## 5. Contribution Analysis
计算：
- NoPAM − BL0b
- PAM − NoPAM
- Shuffle − NoPAM
- PAM − Shuffle

## 6. Stratified Analysis
报告：
- All
- NGG-only
- non-NGG-only

## 7. Paired Probability Analysis
说明真实 PAM 是否主要提高 positive 样本概率。

## 8. Interpretation
分三类写：

### 已经证明
比如：
- formal split 一致
- BL5-v4-PAM 整体优于 BL0b
- NoPAM 证明 v4 无 PAM 框架已经很强

### 本实验支持
比如：
- 如果 PAM > shuffle，则支持正确 PAM 对应关系有价值

### 仍需谨慎
比如：
- non-NGG PAM 100% positive 的 shortcut 风险
- 仍需 per-sgRNA、kNN baseline、in silico perturbation

## 9. Final Conclusion
给出一句严谨结论。
```

---

# 十二、结论措辞模板

如果结果为：

```text
PAM_AUPRC > Shuffle_AUPRC ≈ NoPAM_AUPRC
```

请写：

```text
BL5-v4-PAM-shuffle-control 的结果显示，打乱 PAM 与样本之间的对应关系后，模型性能从真实 PAM 的 AUPRC=0.531281 下降至接近 NoPAM-control 的水平。这说明 PAM Encoder 的增益依赖于正确的 PAM 信息，而不仅仅来自额外参数量或训练噪声。该结果支持 PAM Encoder 在 BL5-v4 框架中提供真实增量信号。
```

如果结果为：

```text
PAM_AUPRC > Shuffle_AUPRC > NoPAM_AUPRC
```

请写：

```text
打乱 PAM 后模型性能低于真实 PAM，但仍高于 NoPAM-control。这说明正确 PAM 信息确实有价值，但 PAM 分支的部分提升可能来自 PAM 分布、额外参数量或其他非特异性因素。因此，PAM Encoder 的生物学解释需要结合 NGG-only、per-PAM 和 paired probability 分析进一步确认。
```

如果结果为：

```text
Shuffle_AUPRC ≈ PAM_AUPRC
```

请写：

```text
PAM shuffle 后性能接近真实 PAM，说明模型性能提升并不依赖 PAM 与样本之间的正确对应关系。因此，不能将 BL5-v4-PAM 的提升解释为模型学习到了真实 PAM 生物学信号。需要优先检查 PAM shuffle 实现、数据泄漏、训练差异和 classifier/parameter-count confounding。
```

如果结果为：

```text
Shuffle_AUPRC > PAM_AUPRC
```

请写：

```text
PAM shuffle-control 反而优于真实 PAM，这是异常结果。必须暂缓所有生物学解释，并优先检查 shuffle 实现、PAM feature 对齐、test set 对齐、label 对齐、random seed 和数据泄漏问题。
```

---

# 十三、执行顺序

请按下面顺序执行，不要跳步：

```text
1. 审计 configs/bl5_v4_pam_shuffle_control.yaml
2. 确认它与 BL5-v4-PAM 除 shuffle_pam 相关字段外完全一致
3. 实现或确认 train_bl5.py 支持 shuffle_pam
4. 生成 pam_shuffle_audit.json / md
5. 训练 BL5-v4-PAM-shuffle-control
6. 使用 best.pt 做 test evaluation
7. 导出 test_predictions.csv，包含 PAM_original 与 PAM_shuffled
8. 与 BL0b、NoPAM、PAM 做主表对比
9. 做 All / NGG-only / non-NGG-only 分层评估
10. 做 paired comparison
11. 写 final_shuffle_control_report.md
12. 追加 results/experiments.csv
```

不要在完成这些之前启动任何新模型。
