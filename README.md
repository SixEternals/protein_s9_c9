# CRISPR-DualPred

Local implementation scaffold for the R9/DeepFocus and C9/ConMismatch9 pipelines.

## What is included

- `encoders/r9_encoder.py`
- `encoders/c9_encoder.py`
- `models/deepfocus.py`
- `models/conmismatch9.py`
- `server.py`
- `train.py`
- `configs/`

## Run the server

```bash
python3 server.py --port 8000 --device cpu
```

## Predict

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sgRNA":"GAGTCCGAGCAGAAGAAGAA","dna":"GAGTCCGAGCAGAAGAAGAA","model":"deepfocus","encoder":"r9"}'
```

## Train

```bash
python3 train.py --config configs/r9_deepfocus.yaml
```

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

本项目使用 R9/C9 命名规范，禁止在任何代码中出现裸 "9bit"
