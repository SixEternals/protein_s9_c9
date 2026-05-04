#!/usr/bin/env bash
set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-/data/zwf/conda/envs/reborn_seed}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
CONDA_BIN="${CONDA_BIN:-conda}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  "$CONDA_BIN" create -y -p "$ENV_PREFIX" --override-channels -c conda-forge -c defaults "python=${PYTHON_VERSION}" pip
fi

"$ENV_PREFIX/bin/python" -m pip install --upgrade pip
"$ENV_PREFIX/bin/python" -m pip install numpy scikit-learn tqdm pyyaml fastapi uvicorn pydantic
"$ENV_PREFIX/bin/python" -m pip install torch --index-url "$PYTORCH_INDEX_URL"

"$ENV_PREFIX/bin/python" - <<'PY'
import sys

import numpy
import sklearn
import torch

print("python", sys.version.split()[0])
print("numpy", numpy.__version__)
print("sklearn", sklearn.__version__)
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda_device_0", torch.cuda.get_device_name(0))
PY

cat <<EOF

Environment ready:
  conda activate $ENV_PREFIX

Training smoke command:
  PYTHON_BIN=$ENV_PREFIX/bin/python bash scripts/train_all_datasets.sh --dry-run /data/zwf/project/zhb/
EOF
