# CRISPR-DualPred

Local implementation scaffold for the R9/DeepFocus and C9/ConMismatch9 pipelines.

> **⚠️ Fallback mode notice**: The default configs (`configs/*.yaml`) do **not** point to pretrained weights. If you start the server with default settings, it will run in **fallback mode** using lightweight built-in models (`model_backend: legacy_json`). For production predictions, you must train or provide your own `.pt` checkpoints and update `weights_path` in the config.

## Quick usage

1. Install the project environment:

   ```bash
   bash scripts/setup_env.sh
   ```

2. Start the server:

   ```bash
   python3 server.py --port 8000 --device cpu
   ```

3. Open the UI at `http://127.0.0.1:8000/`.

4. Use 23nt sgRNA/DNA pairs only. `U` is converted to `T`, and invalid or non-23nt input is rejected.

5. Use the same endpoints from the UI or with `curl`:

   - `/health`
   - `/models`
   - `/predict`
   - `/predict/batch`
   - `/predict/file`
   - `/train`
   - `/jobs/{job_id}`

For a step-by-step guide, see [doc/21_快速使用指南.md](doc/21_快速使用指南.md) and the full reference in [doc/17_使用文档.md](doc/17_使用文档.md).

## What is included

- `encoders/r9_encoder.py`
- `encoders/c9_encoder.py`
- `models/deepfocus_torch.py` (formal PyTorch model)
- `models/conmismatch9_torch.py` (formal PyTorch model)
- `models/deepfocus.py` (lightweight fallback)
- `models/conmismatch9.py` (lightweight fallback)
- `server.py`
- `train.py`
- `configs/`

## Model pairings

| Model | Encoder | Backend when trained |
|-------|---------|---------------------|
| `deepfocus` | `r9` | `torch_deepfocus` |
| `conmismatch9` | `c9` | `torch_conmismatch9` |

Always use the correct encoder for each model. Mismatched pairs will be rejected.

## Prediction input contract

All prediction endpoints (`/predict`, `/predict/batch`, `/predict/file`) enforce the same input contract:

- **sgRNA** and **dna** must each be exactly **23 nucleotides**.
- Allowed characters: `A`, `T`, `C`, `G`, `U`. `U` is internally converted to `T`.
- Empty, shorter, longer, or sequences with illegal characters are rejected with `400` and a clear error message.

## Run the server

Requires **PyYAML** for config loading and **torch** for trained models.

```bash
python3 server.py --port 8000 --device cpu
```

Check whether trained weights are actually loaded:

```bash
curl http://127.0.0.1:8000/health
```

- `status: ok` — all trained checkpoints loaded
- `status: partial` — some checkpoints loaded
- `status: degraded` — no checkpoints loaded (fallback mode)
- `fallback_active: true` — at least one model is using the lightweight fallback

List available models and their load status:

```bash
curl http://127.0.0.1:8000/models
```

## Predict

### Single sequence

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sgRNA":"GAGTCCGAGCAGAAGAAGAAGAA","dna":"GAGTCCGAGCAGAAGAAGAAGAA","model":"deepfocus","encoder":"r9"}'
```

Check `model_backend` in the response to confirm whether the prediction used a trained PyTorch checkpoint (`torch_deepfocus`) or the fallback (`legacy_json`).

### Batch prediction

```bash
curl -X POST http://127.0.0.1:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepfocus",
    "encoder": "r9",
    "pairs": [
      {"sgRNA": "GAGTCCGAGCAGAAGAAGAAGAA", "dna": "GAGTCCGAGCAGAAGAAGAAGAA"}
    ]
  }'
```

### File upload (CSV/TSV)

```bash
curl -X POST http://127.0.0.1:8000/predict/file \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepfocus",
    "encoder": "r9",
    "file_content": "sgRNA,dna\nGAGTCCGAGCAGAAGAAGAAGAA,GAGTCCGAGCAGAAGAAGAAGAA",
    "format": "csv",
    "has_header": true
  }'
```

If rows fail validation, the response includes per-row `errors` with row numbers and reasons.

## Train

```bash
python3 train.py --config configs/r9_deepfocus.yaml
```

Configurations support both JSON and YAML (including comments). See `doc/17_使用文档.md` for full field reference.

Training configs are validated **before** training starts. Invalid configs exit with code 2 and print a clear error list.

Set up a fresh GPU-ready conda env:

```bash
bash scripts/setup_env.sh
```

Train only the smallest local dataset, which is the default quick path:

```bash
bash scripts/train_all_datasets.sh
```

Train every discovered dataset from small to large:

```bash
bash scripts/train_all_datasets.sh --all
```

Run a full GPU training on the smallest dataset:

```bash
bash scripts/train_all_datasets.sh --first-only --device cuda --amp auto
```

## Asynchronous training (API-only)

The `/train` endpoint supports `async_run: true` to queue training in a background thread.

> **Limitations**: Jobs are stored in an in-memory registry and are lost on server restart. Concurrent training jobs may race on shared model state. This API is intended for local experimentation, not production multi-tenant use.

```bash
curl -X POST http://127.0.0.1:8000/train \
  -H "Content-Type: application/json" \
  -d '{"config_path": "configs/r9_deepfocus.yaml", "async_run": true}'
```

Query job status and (on success) a lightweight result summary:

```bash
curl http://127.0.0.1:8000/jobs/{job_id}
```

## Documentation

- [doc/21_快速使用指南.md](doc/21_快速使用指南.md) — quick usage guide for the current UI and API
- `doc/17_使用文档.md` — full usage guide (Chinese)
- `doc/20_深度学习模块清单与接口设计报告.md` — deep learning module inventory and API design
- `doc/18_从零开始理解CRISPR-DualPred.md` — beginner-friendly conceptual guide

## Testing

The full test suite is intended to run in the project conda environment:

```bash
/data/zwf/conda/envs/reborn_seed/bin/python -m unittest discover -s tests -v
```

For a lighter check that does not require `torch`, run:

```bash
/data/zwf/conda/envs/reborn_seed/bin/python -m unittest tests.test_light -v
```

## Notes

- 本项目使用 R9/C9 命名规范，禁止在任何代码中出现裸 `"9bit"`
- Default configs intentionally leave `weights_path` as `null` to avoid silently loading non-existent checkpoints
- PyYAML is required for config loading (installed by `scripts/setup_env.sh`)
