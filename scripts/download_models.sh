#!/usr/bin/env bash
# Download model weights required for AutoGaze training.
# Usage:
#   bash scripts/download_models.sh [TARGET_DIR]
#
# Downloads:
#   - nvidia/AutoGaze        : AutoGaze gaze model weights (~50 MB)
#   - bfshi/VideoMAE_AutoGaze: VideoMAE task model weights (~2 GB)

set -euo pipefail

TARGET_DIR="${1:-./weights}"
mkdir -p "$TARGET_DIR"

echo "=== Downloading model weights to $TARGET_DIR ==="

# AutoGaze (gaze model)
echo "[1/2] Downloading nvidia/AutoGaze..."
huggingface-cli download nvidia/AutoGaze \
    --local-dir "$TARGET_DIR/AutoGaze"

# VideoMAE (task model for training)
echo "[2/2] Downloading bfshi/VideoMAE_AutoGaze..."
huggingface-cli download bfshi/VideoMAE_AutoGaze \
    --local-dir "$TARGET_DIR/VideoMAE_AutoGaze"

echo ""
echo "Done! Weights saved to:"
echo "  AutoGaze  : $TARGET_DIR/AutoGaze"
echo "  VideoMAE  : $TARGET_DIR/VideoMAE_AutoGaze"
echo "  videomae.pt path: $TARGET_DIR/VideoMAE_AutoGaze/videomae.pt"
