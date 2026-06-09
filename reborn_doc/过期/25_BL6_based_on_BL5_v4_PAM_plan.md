# BL6 Plan Based on BL5-v4-PAM

> Date: 2026-06-05  
> Status: planning draft, not a training approval  
> Scope: BL6 must use the current strongest BL5-v4-PAM line as backbone.  
> Coordination note: Kimi can maintain narrative/docs; Codex/model worker should own model implementation, smoke tests, training, and result verification.

---

## 1. Why BL5 Is Stage-Complete

BL5 can be treated as stage-complete for architecture search. The current strongest formal BL5 split model is `BL5-v4-PAM`, not the older Cross-Attn/Gated branch.

The formal comparison uses the same test cohort:

```text
split_file = formal_split_bl5_seed42.json
split_mode = sgrna_safe
test_samples = 954,326
test_observed_positive = 3,057
test_unobserved_candidate = 951,269
test_sgRNA_type_count = 72
positive_rate = 0.003203
```

Main formal results:

| model | setting | test AUROC | test AUPRC |
|:---|:---|---:|---:|
| BL0b-on-BL5split | CCLMoff-style RNA-FM baseline | 0.857756 | 0.295678 |
| BL5-v4-NoPAM-control | RNA-FM + LearnableRun, no PAM | 0.984098 | 0.502389 |
| BL5-v4-PAM | RNA-FM + LearnableRun + correct PAM | 0.984194 | 0.531281 |
| BL5-v4-PAM-shuffle-control | same model, shuffled PAM | 0.669701 | 0.138883 |

Key contribution decomposition:

```text
NoPAM - BL0b = 0.502389 - 0.295678 = +0.206711
PAM - NoPAM = 0.531281 - 0.502389 = +0.028892
PAM - BL0b = 0.531281 - 0.295678 = +0.235603
PAM - Shuffle = 0.531281 - 0.138883 = +0.392398
Shuffle - NoPAM = 0.138883 - 0.502389 = -0.363506
```

Interpretation:

- `NoPAM - BL0b` is the holistic gain of the BL5-v4 no-PAM framework. It must not be described as the pure contribution of LearnableRun.
- `PAM - NoPAM` is the approximate additional contribution of correct PAM encoding inside the v4 framework.
- `PAM - Shuffle` supports that correct PAM-to-sample correspondence matters.

Existing BL5 variants also indicate that the older complex fusion route is not currently the winning line:

| model | core idea | test AUPRC |
|:---|:---|---:|
| BL5-3 | Cross-Attn + Gated, hand-crafted Run | 0.445172 |
| BL5-3-LearnableRun | Cross-Attn + Gated + LearnableRun | 0.518032 |
| BL5-v3-CLS | CLS + LearnableRun, no PAM | 0.483643 |
| BL5-v4-PAM | CLS + LearnableRun + PAM | 0.531281 |

Conclusion:

```text
BL5-v4-PAM should be the BL6 backbone.
BL6 should not restart from the old BL5-3 Cross-Attn/Gated branch.
```

---

## 2. Why BL6 Should Use BL5-v4-PAM As Backbone

BL6 should not mean "make the model more complex by default." The current evidence says:

- Simple `RNA-FM CLS + LearnableRun + PAM + v4 classifier` is stronger than the previous Cross-Attn/Gated branch.
- PAM is the clearest validated additional signal in BL5.
- PAM shuffle-control shows that a misleading PAM branch can damage performance, so BL6 must preserve correct PAM handling and keep the PAM definition explicit.

Therefore BL6 should be defined as:

```text
Incremental optimization on top of BL5-v4-PAM.
```

The BL6 backbone must preserve:

```text
use_rnafm = true
freeze_rnafm = false
RNA-FM pooling = cls
use_learnable_run = true
Run positions = 1-20 only
use_pam_encoder = true
PAM positions = canonical 21-23
formal_split_json = formal_split_bl5_seed42.json
focal_loss = true
focal_gamma = 2.0
best.pt test evaluation
AUROC and AUPRC both reported
```

The default reference representation is:

```text
z_rna: RNA-FM CLS, [B, 640]
z_run: LearnableRun pooled representation, [B, 128]
z_pam: PAM embedding, [B, 16]
z_concat = [z_rna, z_run, z_pam], [B, 784]
classifier = v4 MLP head
```

Every BL6 subversion should change one mechanism at a time. If multiple mechanisms are changed together, the result will be hard to interpret.

---

## 3. BL6 Design Principles

1. **Same data and split**

   All BL6 subversions must use `formal_split_bl5_seed42.json`.

2. **Same evaluation protocol**

   Training selects checkpoint by validation AUPRC. Test evaluation must explicitly load `checkpoints/best.pt`.

3. **Same core feature contract**

   Raw sequence strings must not be fed directly into a neural network without RNA-FM tokenization or the project-approved encoders. Run remains positions `1-20`; PAM remains positions `21-23`.

4. **One mechanism per subversion**

   BL6-1, BL6-2, and BL6-3 must each answer one clear scientific question.

5. **Preserve BL5-v4-PAM fallback**

   BL6 should not discard the known-good BL5-v4-PAM representation. New gates, FiLM, or attention should be residual or fallback-aware.

6. **Do not equate complexity with progress**

   A more complex model is only useful if it beats the BL5-v4-PAM anchor on the same test set.

7. **Respect terminology**

   `label=0` is `unobserved_candidate`, not an experimentally verified safe site.

---

## 4. Candidate Subversions

### 4.1 BL6-0-anchor

Purpose:

```text
Define the BL6 reference anchor.
```

Definition:

```text
Equivalent to BL5-v4-PAM.
No new architecture.
No new claim.
```

Reference metrics:

```text
historical_best:
  AUROC = 0.984194
  AUPRC = 0.531281
  best_epoch = 9

latest_2gpu_rerun:
  AUROC = 0.986086
  AUPRC = 0.516095
  best_epoch = 9
```

Interpretation rules:

- If a BL6 subversion exceeds `0.531281` AUPRC, it can be described as improving over the current BL5-v4-PAM best record.
- If it exceeds `0.516095` but not `0.531281`, it only improves over the latest two-GPU rerun anchor, not the historical best.
- If it does not exceed `0.516095`, it is not an improvement.

Required action:

```text
Do not retrain BL6-0 unless a clean anchor rerun is explicitly requested.
Use it as a reference point in reports and plots.
```

---

### 4.2 BL6-1-PAM-Gated-Fusion

Purpose:

```text
Test whether sample-wise dynamic weighting of RNA-FM, Run, and PAM views improves over fixed simple_concat.
```

Backbone inputs:

```text
z_rna [B, 640]
z_run [B, 128]
z_pam [B, 16]
```

Proposed mechanism:

```text
view_summary = MLP([z_rna_proj, z_run, z_pam])
gate = softmax(W_gate(view_summary))  # [B, 3]
z_weighted = gate_rna * z_rna_proj + gate_run * z_run + gate_pam * z_pam_proj
z_final = concat(original z_rna, z_run, z_pam, z_weighted)
classifier = small MLP
```

Important design detail:

The original BL5-v4-PAM concat representation should remain available as a residual/fallback. The gate must not be allowed to completely erase the baseline features.

Scientific question:

```text
Do different samples require different RNA-FM / Run / PAM weights?
```

Expected risk:

Older Gated Fusion variants did not beat BL5-v4-PAM. This version must stay lightweight and fallback-aware.

Initial config name:

```text
configs/bl6_1_pam_gated_fusion.yaml
```

Output directory:

```text
results/bl6_1_pam_gated_fusion/
```

Keep if:

```text
test AUPRC > 0.531281
or
test AUPRC > 0.516095 with clearly better top-k recall and stable validation behavior
```

Stop condition:

```text
validation AUPRC stays clearly below BL5-v4-PAM anchor for multiple epochs
or
gate collapses to one view for nearly all samples without performance benefit
```

---

### 4.3 BL6-2-PAM-FiLM-Run

Purpose:

```text
Test whether PAM should modulate interpretation of protospacer mismatch / Run features.
```

Rationale:

The same mismatch/run pattern may have different cleavage behavior under different PAM contexts. BL5-v4-PAM treats PAM as an independent branch. BL6-2 tests whether PAM should condition the Run representation.

Proposed mechanism:

```text
gamma = MLP_gamma(z_pam)
beta = MLP_beta(z_pam)
z_run_mod = gamma * z_run + beta
z_final = concat(z_rna, z_run, z_run_mod, z_pam)
classifier = MLP
```

Alternative token-level version:

```text
gamma_token = MLP_gamma(z_pam) -> [B, 1, d_model]
beta_token = MLP_beta(z_pam) -> [B, 1, d_model]
H_run_mod = gamma_token * H_run + beta_token
z_run_mod = pool(H_run_mod)
```

Recommended first implementation:

```text
Use pooled z_run FiLM first.
Do not start with token-level FiLM unless pooled FiLM is promising.
```

Scientific question:

```text
Does PAM context change how the model scores Run/protospacer patterns?
```

Initial config name:

```text
configs/bl6_2_pam_film_run.yaml
```

Output directory:

```text
results/bl6_2_pam_film_run/
```

Keep if:

```text
test AUPRC > 0.531281
or
PAM - NoPAM behavior improves in NGG-only stratified analysis
```

Stop condition:

```text
z_run_mod causes unstable training, NaN, or validation collapse.
```

---

### 4.4 BL6-3-LightCrossAttn-PAM-Residual

Purpose:

```text
Test whether token-level RNA-FM / Run interaction still adds value after BL5-v4-PAM.
```

This is the highest-risk BL6 direction because older Cross-Attn/Gated variants did not beat BL5-v4-PAM.

Proposed mechanism:

```text
H_rna = RNA-FM token states projected to d_model
H_run = LearnableRun token states [B, 20, d_model]
z_pam = PAM embedding

H_run_attn = CrossAttn(query=H_run, key=H_rna, value=H_rna)
H_run_res = H_run + dropout(H_run_attn)

z_run_attn = pool(H_run_res)
z_final = concat(z_rna_cls, z_run, z_run_attn, z_pam)
classifier = MLP
```

Optional PAM conditioning:

```text
Use z_pam as a small bias/gate on the attention output:
attn_gate = sigmoid(MLP(z_pam))
z_run_attn = attn_gate * z_run_attn
```

Required constraints:

- Only one Cross-Attn layer in the first formal version.
- No multi-layer attention in the first version.
- Keep BL5-v4-PAM original concat as residual/fallback.
- Do not copy the old BL5-3 full structure wholesale.

Scientific question:

```text
Does token-level RNA-FM/Run interaction provide marginal gain once CLS + LearnableRun + PAM are already present?
```

Initial config name:

```text
configs/bl6_3_light_crossattn_pam_residual.yaml
```

Output directory:

```text
results/bl6_3_light_crossattn_pam_residual/
```

Keep if:

```text
test AUPRC > 0.531281
and
training remains stable
```

Stop condition:

```text
validation AUPRC does not approach BL5-v4-PAM by mid-training
or
DDP/NaN instability appears repeatedly.
```

---

### 4.5 BL6-4-Rank-Aware-Head

Status:

```text
Optional. Do not start before BL6-1 and BL6-2 are evaluated.
```

Purpose:

```text
Improve AUPRC/top-k behavior directly under extreme class imbalance.
```

Possible mechanisms:

- Keep BL5-v4-PAM backbone unchanged.
- Replace or augment the classifier head.
- Add a small ranking auxiliary loss on sampled observed_positive vs unobserved_candidate pairs.

Scientific question:

```text
Can a ranking-aware objective improve AUPRC and top-k recall without changing the biological feature encoders?
```

Risks:

- Ranking losses are sensitive to sampling.
- AUPRC gains can come at the cost of poor calibration.
- Implementation can easily introduce hidden confounders if paired sampling is not controlled.

Initial config name:

```text
configs/bl6_4_rank_aware_head.yaml
```

Output directory:

```text
results/bl6_4_rank_aware_head/
```

Stop condition:

```text
AUROC or calibration collapses, or top-k improves only by over-scoring too many unobserved_candidate samples.
```

---

### 4.6 BL6-5-Seed-Ensemble

Status:

```text
Optional deployment/stability experiment.
Not a new architecture claim.
```

Purpose:

```text
Reduce training-seed variance and improve stable top-k performance.
```

Definition:

```text
Train multiple seeds of BL5-v4-PAM or the best BL6 subversion.
Average test probabilities.
Evaluate AUROC, AUPRC, top-k recall, and calibration.
```

Scientific question:

```text
Can multi-seed averaging stabilize the current best model?
```

Required reporting:

```text
single-seed mean +/- std
ensemble AUROC/AUPRC
top-k recall change
```

Important interpretation:

```text
This is a stability/deployment optimization, not a biological or architectural mechanism.
```

---

## 5. Execution Order

Do not implement or train all BL6 variants at once.

Recommended order:

```text
Step 1: Confirm BL5-v4-PAM backbone reuse points in code.
Step 2: Add a BL6 variant switch with minimal code changes.
Step 3: Implement BL6-1-PAM-Gated-Fusion.
Step 4: Run smoke test only.
Step 5: If smoke passes, run one formal BL6-1 training.
Step 6: Compare BL6-1 with BL5-v4-PAM historical best and latest rerun.
Step 7: Decide whether BL6-2 is worth implementing.
Step 8: Implement BL6-2 only if BL6-1 does not expose a broader design issue.
Step 9: BL6-3 LightCrossAttn only after BL6-1/2 are understood.
Step 10: BL6-4 and BL6-5 remain optional.
```

First formal target:

```text
BL6-1-PAM-Gated-Fusion
```

Reason:

- It is the smallest architecture change.
- It preserves all BL5-v4-PAM inputs.
- It directly tests whether sample-wise dynamic view weighting helps.
- It is easier to interpret than token-level Cross-Attn.

---

## 6. Implementation Boundary

Preferred implementation path:

```text
Reuse scripts/train_bl5.py if possible.
Extend model/config variant handling carefully.
Avoid copying the entire training script into scripts/train_bl6.py unless absolutely necessary.
```

If a new training script is required:

```text
scripts/train_bl6.py
```

It must preserve these mature behaviors from `train_bl5.py`:

- guardrails checks
- formal split loading
- DDP training
- best.pt checkpointing by validation AUPRC
- explicit best.pt test evaluation
- test prediction export
- summary/report/epoch_metrics generation
- `results/experiments.csv` append

New code should be narrowly scoped. Do not rewrite the data loader, RNA-FM integration, split logic, or evaluation logic unless there is a verified bug.

---

## 7. Smoke Test Protocol

Before any formal BL6 training:

```text
small sample or debug subset
1 epoch
single GPU or 2 GPU DDP quick pass
confirm forward/backward works
confirm no NaN
confirm best.pt writes
confirm summary/report/epoch_metrics writes
confirm test prediction export path works if enabled
```

Smoke output path:

```text
results/smoke_tests/bl6_1_pam_gated_fusion_smoke/
```

Smoke success criteria:

- Training starts and finishes.
- No DDP deadlock.
- No nonfinite probability explosion.
- `summary.json` exists.
- `epoch_metrics.csv` exists.
- If export enabled, prediction row count matches the smoke subset expectation.

Smoke failure response:

```text
Stop. Fix only the smoke-blocking issue. Do not start formal training.
```

---

## 8. Formal Training Protocol

Formal launch template:

```bash
torchrun --nproc_per_node=2 scripts/train_bl5.py \
  --config configs/bl6_1_pam_gated_fusion.yaml \
  --output_dir results/bl6_1_pam_gated_fusion
```

If `scripts/train_bl6.py` is created:

```bash
torchrun --nproc_per_node=2 scripts/train_bl6.py \
  --config configs/bl6_1_pam_gated_fusion.yaml \
  --output_dir results/bl6_1_pam_gated_fusion
```

Required hardware expectation:

```text
2 GPUs for formal runs
expected runtime close to BL5-v4-PAM: around 3 hours for 10 epochs, unless the variant is heavier
```

Do not use single-GPU formal training unless the user explicitly approves a degraded run.

---

## 9. Required Artifacts

Each formal BL6 subversion must produce:

```text
results/<bl6_output_dir>/summary.json
results/<bl6_output_dir>/report.md
results/<bl6_output_dir>/epoch_metrics.csv
results/<bl6_output_dir>/checkpoints/best.pt
results/<bl6_output_dir>/test_predictions.csv
```

Each `summary.json` must include:

```text
version
status
generated_at
commit_hash
split_mode
formal_split_json
use_rnafm
freeze_rnafm
use_learnable_run
use_pam_encoder
model_variant
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

Each `report.md` must include:

```text
1. Model definition
2. Difference from BL5-v4-PAM
3. Split and cohort check
4. Main test metrics
5. Comparison against BL5-v4-PAM
6. Comparison against NoPAM and shuffle controls
7. Training stability
8. Whether this subversion should be kept
9. Failure analysis if no improvement
```

Each `test_predictions.csv` must include at least:

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

Required cohort check:

```text
test_samples = 954,326
test_observed_positive = 3,057
test_unobserved_candidate = 951,269
test_sgRNA_type_count = 72
```

If any of these counts do not match, stop and do not compare the metric with BL5-v4-PAM.

---

## 10. Success Criteria

### 10.1 Strong Success

```text
test AUPRC > 0.531281
AUROC does not materially collapse
formal test set is identical
top-k recall is equal or better
training is stable
```

Allowed conclusion:

```text
This BL6 subversion improves over the current BL5-v4-PAM historical best on the formal BL5 split.
```

### 10.2 Weak Success

```text
0.516095 < test AUPRC <= 0.531281
```

Allowed conclusion:

```text
This BL6 subversion improves over the latest two-GPU BL5-v4-PAM rerun, but does not exceed the historical BL5-v4-PAM best.
```

Not allowed:

```text
Claiming a new overall best.
```

### 10.3 No Improvement

```text
test AUPRC <= 0.516095
```

Allowed conclusion:

```text
This direction did not improve over the BL5-v4-PAM anchor.
```

Required action:

```text
Record as a negative result.
Do not keep expanding the same mechanism unless there is a specific diagnosed bug.
```

### 10.4 Failed Direction

```text
NaN
DDP instability
test set mismatch
AUPRC far below NoPAM
model relies on suspicious shortcut without clear value
```

Required action:

```text
Stop the direction.
Recover best.pt test eval if possible.
Mark status as completed_recovered_best_after_nan or failed.
Write failure analysis.
```

---

## 11. Comparison Table Template

Every BL6 report should include:

| model | variant | AUROC | AUPRC | Accuracy | Precision | Recall | F1 | best_epoch | best_val_AUPRC | keep? |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| BL0b-on-BL5split | baseline | 0.857756 | 0.295678 | TBD | TBD | TBD | TBD | 8 | N/A | reference |
| BL5-v4-NoPAM-control | no PAM | 0.984098 | 0.502389 | TBD | TBD | TBD | TBD | 4 | 0.637471 | reference |
| BL5-v4-PAM | anchor historical best | 0.984194 | 0.531281 | TBD | TBD | TBD | TBD | 9 | 0.638364 | anchor |
| BL5-v4-PAM latest rerun | anchor rerun | 0.986086 | 0.516095 | 0.997021 | 0.545923 | 0.416094 | 0.472248 | 9 | 0.634354 | reference |
| BL6-x | candidate | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

The table must clearly distinguish historical best and latest rerun.

---

## 12. Risks and Stop Conditions

### 12.1 Training Variance

The latest two-GPU BL5-v4-PAM rerun gave AUPRC `0.516095`, lower than the historical best `0.531281`. BL6 improvements must be interpreted against both anchors.

Stop condition:

```text
If BL6 only beats 0.516095 by a very small margin, do not claim robust improvement without seed repeats.
```

### 12.2 Complexity Risk

Older complex fusion variants did not beat BL5-v4-PAM. A more complex BL6 can easily overfit or destabilize training.

Stop condition:

```text
If Cross-Attn/Gated variants repeatedly underperform, stop architecture expansion and return to stability/analysis.
```

### 12.3 PAM Shortcut Risk

PAM is useful, but shortcut risk remains. Any BL6 improvement that relies more heavily on PAM must be checked with:

```text
NGG-only
non-NGG-only
per-PAM motif
per-sgRNA
PAM shuffle or perturbation if needed
```

### 12.4 Test Set Mismatch

Any metric from a different split or different test cohort is not directly comparable.

Stop condition:

```text
If test_samples / observed_positive / unobserved_candidate / sgRNA_type_count mismatch, stop and audit split logic.
```

### 12.5 Evaluation Leakage

Do not tune architecture based on repeated test inspection without recording that risk. Validation AUPRC must drive checkpoint choice.

### 12.6 Label Semantics

Do not write that label `0` means safe. It means `unobserved_candidate`.

---

## 13. Immediate Next Action

Before code changes:

```text
1. Confirm this BL6 plan with the user.
2. Confirm BL6-1-PAM-Gated-Fusion as first candidate.
3. Audit current train_bl5.py and model code to find the minimal variant hook.
4. Prepare config draft only.
5. Run smoke test.
6. Only then launch formal two-GPU training.
```

Do not start BL6-2, BL6-3, BL6-4, or BL6-5 until BL6-1 has a verified result.

Recommended first implementation target:

```text
BL6-1-PAM-Gated-Fusion
```

Recommended first result gate:

```text
If BL6-1 test AUPRC <= 0.516095, stop BL6-1 and do not expand gated variants.
If 0.516095 < AUPRC <= 0.531281, consider a seed repeat before making claims.
If AUPRC > 0.531281, run verification analyses and consider BL6-2.
```

---

## 14. Final Planning Conclusion

BL5-v4-PAM is the stage-complete BL5 winner and should be the BL6 backbone. BL6 should not be a restart from old Cross-Attn/Gated models. The first BL6 step should be a small, residual, interpretable enhancement over BL5-v4-PAM, with BL6-1-PAM-Gated-Fusion as the first candidate. BL6 success requires beating the BL5-v4-PAM historical best AUPRC `0.531281` on the same formal BL5 test set, or it should be recorded as a neutral or negative result rather than promoted as a new best model.
