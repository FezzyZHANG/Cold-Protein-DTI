#!/usr/bin/env bash
set -euo pipefail

echo "[bootstrap] host=$(hostname)"
echo "[bootstrap] date=$(date)"
echo "[bootstrap] cwd=$(pwd)"

echo "[bootstrap] python"
python -V || true

if command -v conda >/dev/null 2>&1; then
  echo "[bootstrap] conda available"
  conda info --envs || true
fi

if command -v micromamba >/dev/null 2>&1; then
  echo "[bootstrap] micromamba available"
  micromamba env list || true
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[bootstrap] gpu"
  nvidia-smi || true
fi

echo "[bootstrap] torch check"
python - <<'PY'
try:
    import torch
    print("torch_version=", torch.__version__)
    print("cuda_available=", torch.cuda.is_available())
    print("device_count=", torch.cuda.device_count())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(i, torch.cuda.get_device_name(i))
except Exception as e:
    print("torch_check_failed=", repr(e))
PY
