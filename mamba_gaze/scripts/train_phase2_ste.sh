#!/usr/bin/env bash
# Phase 2 STE fine-tuning from a Phase 1 checkpoint.
#
# Usage:
#   bash mamba_gaze/scripts/train_phase2_ste.sh [CONFIG] [PHASE1_CKPT] [GPUS]

set -euo pipefail

CONFIG="${1:-mamba_gaze/configs/default.yaml}"
RESUME="${2:-$(ls -t checkpoints/phase1_epoch*.pt 2>/dev/null | head -1)}"
NGPU="${3:-8}"
MASTER_PORT="${MASTER_PORT:-29501}"

if [ -z "$RESUME" ]; then
    echo "Error: no Phase-1 checkpoint found. Run train_phase1.sh first."
    exit 1
fi

echo "=== MambaGaze Phase 2 STE ==="
echo "  Config  : $CONFIG"
echo "  Resume  : $RESUME"
echo "  GPUs    : $NGPU"

torchrun \
    --nproc_per_node="$NGPU" \
    --master_port="$MASTER_PORT" \
    -m mamba_gaze.training.phase2_ste \
    --config "$CONFIG" \
    --resume "$RESUME"

echo "Phase 2 STE complete."
