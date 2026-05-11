#!/usr/bin/env bash
set -euo pipefail

# AutoGaze extension setup for macOS MPS environments.
#
# Assumptions:
# - Run this script from the repository root.
# - Apple Silicon with MPS-capable PyTorch is recommended for smoke tests.
# - FlashAttention is intentionally not installed on macOS/MPS.
# - MPS runs should use PyTorch SDPA or eager attention fallback in model code.
# - This script does not read, print, or persist Hugging Face tokens.

if [[ ! -f "pyproject.toml" || ! -d "src/autogaze_ext" ]]; then
  echo "Run this script from the AutoGaze repository root." >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_SMOKE_TEST="${RUN_SMOKE_TEST:-0}"

echo "Using Python: ${PYTHON_BIN}"
"${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel

echo "Installing PyTorch for macOS. This should provide MPS support on compatible Apple Silicon systems."
"${PYTHON_BIN}" -m pip install torch torchvision

echo "Installing core AutoGaze extension dependencies."
"${PYTHON_BIN}" -m pip install \
  "hydra-core>=1.3.2" \
  omegaconf \
  numpy \
  pillow \
  matplotlib \
  einops \
  av \
  imageio \
  "timm>=1.0.15" \
  tqdm \
  loguru \
  wandb

echo "Installing Hugging Face dependencies."
"${PYTHON_BIN}" -m pip install \
  "transformers~=4.51" \
  datasets \
  evaluate \
  huggingface_hub \
  accelerate

echo "Installing local package in editable mode without pyproject dependency resolution."
"${PYTHON_BIN}" -m pip install --no-deps -e .

echo "FlashAttention is not installed on macOS/MPS."

echo
echo "MPS smoke-test command:"
echo "PYTHONPATH=src ${PYTHON_BIN} - <<'PY'"
echo "import torch"
echo "print('mps_available:', torch.backends.mps.is_available())"
echo "PY"
echo "PYTHONPATH=src ${PYTHON_BIN} -c \"from autogaze_ext.pipeline.runner import load_config, print_summary; cfg = load_config(config_name='config'); cfg.runtime.device.type = 'mps'; print_summary(cfg)\""

if [[ "${RUN_SMOKE_TEST}" == "1" ]]; then
  PYTHONPATH=src "${PYTHON_BIN}" - <<'PY'
import torch
print("mps_available:", torch.backends.mps.is_available())
PY
  PYTHONPATH=src "${PYTHON_BIN}" -c "from autogaze_ext.pipeline.runner import load_config, print_summary; cfg = load_config(config_name='config'); cfg.runtime.device.type = 'mps'; print_summary(cfg)"
fi
