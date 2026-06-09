# BL5-v4-LearnableRun-only-control Plan

> Date: 2026-06-06  
> Status: implementation plan, not completed result  
> Owner suggestion: Kimi implements; Codex/model worker reviews model correctness and result alignment.  
> Purpose: fill the missing `LearnableRun-only` row in the component ablation matrix.

---

## 1. Goal

This experiment is a component-level ablation baseline.

Core question:

```text
If the model only uses LearnableRunEncoder and does not use RNA-FM or PAM Encoder,
how strong is the Run prior on the same formal BL5 split?
```

Recommended version name:

```text
BL5-v4-LearnableRun-only-control
```

Recommended paths:

```text
config: configs/bl5_v4_learnablerun_only_control.yaml
output_dir: results/bl5_v4_learnablerun_only_control
smoke_config: configs/bl5_v4_learnablerun_only_control_smoke.yaml
smoke_output_dir: results/smoke_tests/bl5_v4_learnablerun_only_control_smoke
```

This is not BL6, not a new main model, not Cross-Attn, not Gated Fusion, and not a PAM model.

---

## 2. Why This Is Needed

The current ablation matrix is still incomplete.

Already available:

| model | RNA-FM | LearnableRun | PAM | Gate | AUPRC | role |
|:---|:---:|:---:|:---:|:---:|---:|:---|
| BL0b-on-BL5split | yes | no | no | no | 0.295678 | RNA-FM-only baseline |
| BL5-v4-NoPAM-control | yes | yes | no | no | 0.502389 | RNA-FM + Run, no PAM |
| BL5-v4-PAM | yes | yes | yes | no | 0.531281 | BL5 full anchor |
| BL5-v4-PAM-shuffle-control | yes | yes | shuffled | no | 0.138883 | PAM correspondence control |
| BL6-1-PAM-Gated-Fusion | yes | yes | yes | yes | 0.539917 | single-run BL6-1 candidate |

Missing key row:

| model | RNA-FM | LearnableRun | PAM | Gate | status |
|:---|:---:|:---:|:---:|:---:|:---|
| LearnableRun-only | no | yes | no | no | not implemented |

This row answers whether the explicit Run prior is useful by itself and helps separate:

```text
RNA-FM-only
Run-only
RNA-FM + Run
RNA-FM + Run + PAM
RNA-FM + Run + PAM + Gate
```

Important interpretation boundary:

```text
BL5-v4-NoPAM - LearnableRun-only is not a pure RNA-FM contribution.
BL5-v4-NoPAM - BL0b is not a pure LearnableRun contribution.
```

Those gaps also include classifier/head, interaction, implementation, and training effects.

---

## 3. Hard Constraints

Before implementation, the worker must:

```text
1. Read AGENTS.md.
2. Read reborn_doc/1. 大纲拟定.md.
3. Run git status --short.
4. Avoid overwriting any user/Kimi/Codex uncommitted work.
```

Forbidden:

```text
git reset --hard
git clean -f
git rebase
git push --force
commit/push without user approval
deleting data/, results/, reference/
```

Model/data constraints:

```text
Do not modify labels.
Do not modify sgRNA/off_seq.
Do not modify RNA-FM tokens for other models.
Do not modify formal_split_bl5_seed42.json.
Do not run BL6-2, Cross-Attn, SeedWeightedRun, or a new large model in this task.
```

Sequence feature constraints:

```text
Raw sgRNA_seq/off_seq strings must not be fed directly into a neural network.
LearnableRun input must be generated through the approved base-pair/run encoding path.
Run/LearnableRun must only use positions 1-20.
PAM positions 21-23 must not enter Run encoding.
PAM Encoder must be disabled.
RNA-FM must be disabled.
```

Evaluation constraints:

```text
Use formal_split_bl5_seed42.json.
Use validation AUPRC to select best.pt.
Final test evaluation must explicitly load best.pt.
Report AUROC and AUPRC together.
label=0 means unobserved_candidate, not a verified safe site.
```

---

## 4. Config Definition

Create:

```text
configs/bl5_v4_learnablerun_only_control.yaml
```

Use `configs/bl5_v4_nopam_control.yaml` as the closest starting point, then minimally modify it.

Required config semantics:

```yaml
version: "BL5-v4-LearnableRun-only-control"
output_dir: "results/bl5_v4_learnablerun_only_control"
split_mode: "sgrna_safe"
formal_split_json: "formal_split_bl5_seed42.json"

model:
  use_rnafm: false
  freeze_rnafm: false
  use_learnable_run: true
  use_pam_encoder: false
  rna_pooling: none
  fusion_type: run_only
  d_model: 128
  run_dim: 128

training:
  focal_loss: true
  focal_gamma: 2.0
```

Notes:

- If current guardrails require `freeze_rnafm` to be present even when `use_rnafm=false`, keep it explicit and document that RNA-FM is disabled.
- If `rna_pooling: none` is not supported, use the local equivalent but make sure RNA-FM is not initialized or used.
- Keep optimizer/scheduler/epochs/dropout/gradient clipping/precision as close to BL5-v4-NoPAM-control as practical.

Recommended first formal settings:

```text
epochs = 10
best_metric = validation AUPRC
focal_loss gamma = 2.0
batch_size = same as BL5-v4-NoPAM-control for comparability, unless memory/speed justifies a documented increase
eval_batch_size = same as BL5-v4-NoPAM-control or documented larger value
```

Because RNA-FM is disabled, runtime should be much shorter than the 3-hour RNA-FM fine-tuning runs.

---

## 5. Model Definition

Minimal model:

```text
on_seq/off_seq positions 1-20
  -> approved base-pair indices / LearnableRun input
  -> LearnableRunEncoder
  -> pool over 20 positions
  -> classifier
  -> logit
```

Expected tensor flow:

```text
base_pair_indices: [B, 20]
H_run = LearnableRunEncoder(base_pair_indices)  # [B, 20, 128]
z_run = mean_pool(H_run) or existing local pooling  # [B, 128]
logit = MLP(z_run)
```

Suggested classifier:

```text
128 -> 256 -> 64 -> 1
```

or use the local v4 classifier style with input dimension changed to `128`.

Must not include:

```text
RNA-FM CLS
RNA-FM token states
PAM one-hot
PAM embedding
PAM shuffle
Cross-Attn
Gated Fusion
Region encoder
SeedWeightedRun new variant
```

---

## 6. Implementation Boundary

Preferred files to reuse:

```text
scripts/train_bl5.py
models/bl5_dynamic_fusion.py
encoders/learnable_run_encoder.py
```

Do not copy a full training script unless absolutely necessary.

If current code does not support `use_rnafm=false + use_learnable_run=true + use_pam_encoder=false`, implement the smallest compatible change:

```text
1. Skip RNA-FM initialization and forward when use_rnafm=false.
2. Keep LearnableRun initialization and forward.
3. Skip PAM Encoder when use_pam_encoder=false.
4. Set classifier input dimension from active features.
5. Preserve existing BL5-v4-PAM, NoPAM, PAM-shuffle, and BL6-1 behavior.
```

Backward compatibility is mandatory:

```text
BL5-v4-PAM must still work.
BL5-v4-NoPAM-control must still work.
BL5-v4-PAM-shuffle-control must still work.
BL6-1-PAM-Gated-Fusion must still work.
```

If guardrails fail, do not weaken them broadly. Fix config/model semantics narrowly and document why `use_rnafm=false` is valid for this ablation.

---

## 7. Smoke Test

Create optional smoke config:

```text
configs/bl5_v4_learnablerun_only_control_smoke.yaml
```

Smoke output:

```text
results/smoke_tests/bl5_v4_learnablerun_only_control_smoke/
```

Smoke requirements:

```text
small sample or 1 epoch
forward/backward succeeds
no NaN
summary.json generated
epoch_metrics.csv generated
checkpoints/best.pt generated
best.pt test evaluation runs
AUROC and AUPRC both reported
```

Smoke failure response:

```text
Stop. Fix only the smoke-blocking issue. Do not start formal training.
```

---

## 8. Formal Training

Formal command:

```bash
torchrun --nproc_per_node=2 scripts/train_bl5.py \
  --config configs/bl5_v4_learnablerun_only_control.yaml \
  --output_dir results/bl5_v4_learnablerun_only_control
```

Single GPU may be acceptable because RNA-FM is disabled, but use the same two-GPU formal pipeline unless the user explicitly approves single-GPU formal training.

Expected runtime:

```text
Much shorter than RNA-FM fine-tuning.
If it takes close to 3 hours, audit whether RNA-FM was accidentally still enabled.
```

---

## 9. Required Artifacts

Formal run must produce:

```text
results/bl5_v4_learnablerun_only_control/summary.json
results/bl5_v4_learnablerun_only_control/report.md
results/bl5_v4_learnablerun_only_control/epoch_metrics.csv
results/bl5_v4_learnablerun_only_control/checkpoints/best.pt
results/bl5_v4_learnablerun_only_control/test_predictions.csv
```

`test_predictions.csv` must include at least:

```text
sample_index
sgRNA_type
on_seq
off_seq
PAM_original
label
probability
Direction
split
```

Even though the model does not use PAM, keep `PAM_original` for downstream per-PAM analysis.

`summary.json` must include:

```text
version
status
generated_at
commit_hash
split_mode
formal_split_json
use_rnafm: false
use_learnable_run: true
use_pam_encoder: false
model_variant: learnablerun_only
train_seconds
gpu_mem
epochs
planned_epochs
best_epoch
best_metric_name
best_metric_value
test_metrics:
  auroc
  auprc
  accuracy
  precision
  recall
  f1
test_samples
test_observed_positive
test_unobserved_candidate
test_sgRNA_type_count
notes
```

---

## 10. Mandatory Cohort Check

After formal test export, verify:

```text
test_samples = 954,326
test_observed_positive = 3,057
test_unobserved_candidate = 951,269
test_sgRNA_type_count = 72
sample_index aligns with BL5-v4-PAM
label aligns with BL5-v4-PAM
```

If any mismatch occurs:

```text
Stop.
Do not compare metrics.
Audit split loading, sampler order, and prediction export.
```

---

## 11. Report Requirements

Create:

```text
results/bl5_v4_learnablerun_only_control/report.md
```

Required structure:

```markdown
# BL5-v4-LearnableRun-only-control Report

## 1. Purpose
Component-level ablation to measure LearnableRun single-view performance.

## 2. Model Definition
- RNA-FM disabled
- PAM Encoder disabled
- LearnableRun enabled
- Run positions 1-20 only
- PAM positions 21-23 not used by the model

## 3. Split and Cohort Check
- formal_split_bl5_seed42.json
- test_samples
- observed_positive
- unobserved_candidate
- sgRNA_type_count
- sample_index / label alignment

## 4. Main Test Metrics
AUROC, AUPRC, Accuracy, Precision, Recall, F1, best_epoch, best_val_AUPRC.

## 5. Component Ablation Comparison
Compare:
- BL0b-on-BL5split
- LearnableRun-only
- BL5-v4-NoPAM-control
- BL5-v4-PAM
- BL6-1-PAM-Gated-Fusion

## 6. Interpretation
Explain how strong LearnableRun is by itself and how it compares with RNA-FM-only and RNA-FM+Run.

## 7. Limitations
- Run-only does not use RNA-FM context
- Run-only does not use PAM
- label=0 means unobserved_candidate
- still need PAM-only and RNA-FM+PAM no Run to complete the matrix
```

---

## 12. Interpretation Templates

If LearnableRun-only is lower than BL0b:

```text
LearnableRun-only is weaker than the RNA-FM-only baseline, indicating that the Run prior alone is not sufficient to replace RNA-FM sequence context. However, BL5-v4-NoPAM is much stronger than BL0b, supporting that LearnableRun can provide complementary information when combined with RNA-FM.
```

If LearnableRun-only is close to BL0b:

```text
LearnableRun-only approaches RNA-FM-only performance, suggesting that explicit Run prior is a strong single-view signal. BL5-v4-NoPAM further improves over both single-view baselines, supporting complementarity between RNA-FM and Run features.
```

If LearnableRun-only is higher than BL0b:

```text
LearnableRun-only outperforms the RNA-FM-only baseline on the formal BL5 split, indicating that explicit Run prior is a very strong single-view signal. This should still be interpreted cautiously because model capacity and classifier/head details may differ.
```

Avoid:

```text
NoPAM - LearnableRun-only = pure RNA-FM contribution
NoPAM - BL0b = pure LearnableRun contribution
```

---

## 13. experiments.csv

Append to:

```text
results/experiments.csv
```

Recommended row notes:

```text
LearnableRun-only ablation on formal BL5 split; use_rnafm=false; use_pam_encoder=false; Run positions 1-20 only; best.pt test evaluation
```

---

## 14. todo.md Update

After successful formal completion, update:

```text
reborn_doc/todo.md
```

Change the ablation matrix row:

```text
LearnableRun-only | ❌ 未实现（预留）
```

to:

```text
LearnableRun-only | ✅ 已实现
```

and add AUROC/AUPRC either in the row or nearby result summary.

---

## 15. Final Handoff Checklist

After completion, report:

```text
1. Files changed/added.
2. Smoke test status.
3. Formal training status.
4. Whether test cohort matches BL5-v4-PAM.
5. LearnableRun-only AUROC/AUPRC/Accuracy/Precision/Recall/F1.
6. Comparison with BL0b, NoPAM, BL5-v4-PAM, and BL6-1.
7. Whether results/experiments.csv was appended.
8. Whether reborn_doc/todo.md was updated.
9. Any anomalies or caveats.
```

Do not commit and do not push unless explicitly requested.

---

## 16. Priority Note

This experiment should be prioritized before opening BL6-2 if the immediate goal is to close the component ablation gap.

Recommended next ablation order:

```text
1. LearnableRun-only
2. PAM-only
3. RNA-FM + PAM, no LearnableRun
4. Run + PAM, no RNA-FM
```

