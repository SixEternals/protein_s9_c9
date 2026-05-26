# External Job Manifest Guardrails

This directory stores task manifests. A manifest is the message body that tells
Codex, Kimi, Claude, or a human runner what is being executed and which
guardrails apply before any training or evaluation command starts.

The mechanism is intentionally external:

- It does not edit model internals.
- It does not touch running BL3A/BL3B jobs.
- It checks the message body and config before a new command starts.
- It can be bypassed if someone runs training directly, so new jobs should use
  `scripts/guarded_run.py`.

## Basic Flow

```bash
python scripts/preflight_guardrails.py run_manifests/examples/bl3_5_full.yaml
python scripts/guarded_run.py --dry-run run_manifests/examples/bl3_5_full.yaml
python scripts/guarded_run.py run_manifests/examples/bl3_5_full.yaml
```

`preflight_guardrails.py` only reads files and validates the manifest. It does
not run training.

`guarded_run.py` runs preflight first. It executes the manifest command only if
preflight passes. Use `--dry-run` to verify the resolved command without
starting the job.

## Required Message Fields

```yaml
task_id: bl3_5_full_seed42
agent: codex
stage: train
bl_version: BL3.5-Full
architecture_layer: middleware
config_path: configs/bl3_5_full_cclmoff.yaml
model_entry: models/bl3_5_fusion.py
command:
  - python
  - scripts/train_bl3.py
  - --config
  - configs/bl3_5_full_cclmoff.yaml

policy:
  use_rnafm: false
  freeze_rnafm: null
  split_mode: sgrna_safe
  pos_weight: 12

midware:
  use_c9: true
  use_r9: true
  fusion_mode: full
  allow_concat_only: false

eval:
  checkpoint_type: best
  report_metrics: [AUROC, AUPRC]
```

## Important Rules

- BL3 and BL3.5 must use `use_rnafm: false`.
- BL3.5 is middleware and must use C9 + R9 dynamic fusion.
- Main-line C9 + R9 concat is blocked; concat is allowed only for explicit
  historical ablation with `allow_concat_only: true`.
- Eval/test manifests must use `checkpoint_type: best` and report both AUROC
  and AUPRC.
- `guarded_run.py` executes without a shell and rejects destructive commands.
