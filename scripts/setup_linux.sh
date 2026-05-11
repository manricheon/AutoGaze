#!/usr/bin/env bash
set -euo pipefail

# AutoGaze extension setup for Linux CUDA environments.
#
# Assumptions:
# - Run this script from the repository root.
# - Python 3.10+ is recommended.
# - A CUDA-capable NVIDIA GPU and matching driver are available for benchmark runs.
# - This script does not read, print, or persist Hugging Face tokens.
# - FlashAttention is optional because wheel availability depends on CUDA, PyTorch,
#   Python, GPU architecture, and compiler versions.

if [[ ! -f "pyproject.toml" || ! -d "src/autogaze_ext" ]]; then
  echo "Run this script from the AutoGaze repository root." >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}"
RUN_SMOKE_TEST="${RUN_SMOKE_TEST:-0}"

echo "Using Python: ${PYTHON_BIN}"
"${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel

echo "Installing PyTorch for CUDA from: ${TORCH_INDEX_URL}"
"${PYTHON_BIN}" -m pip install --index-url "${TORCH_INDEX_URL}" torch torchvision

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

if [[ "${INSTALL_FLASH_ATTN}" == "1" ]]; then
  echo "Attempting optional FlashAttention install."
  if "${PYTHON_BIN}" -m pip install flash-attn --no-build-isolation; then
    echo "FlashAttention installed."
  else
    echo "FlashAttention install failed or is unsupported in this environment; continuing without it." >&2
  fi
else
  echo "Skipping FlashAttention. Set INSTALL_FLASH_ATTN=1 to attempt optional installation."
fi

echo
echo "Smoke-test command:"
echo "PYTHONPATH=src ${PYTHON_BIN} -m autogaze_ext.pipeline.runner --config-name config"

if [[ "${RUN_SMOKE_TEST}" == "1" ]]; then
  PYTHONPATH=src "${PYTHON_BIN}" -m autogaze_ext.pipeline.runner --config-name config
fi
