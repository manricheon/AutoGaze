#!/usr/bin/env bash
# Phase 1 training: Focal-BCE + L2 reconstruction loss
#
# Hardware: A100×8 recommended (batch_size=256 total, 32 per GPU with grad accum)
# Runtime:  ~2 weeks on A100×8
#
# Usage:
#   bash mamba_gaze/scripts/train_phase1.sh [CONFIG] [GPUS]

set -euo pipefail

CONFIG="${1:-mamba_gaze/configs/default.yaml}"
NGPU="${2:-8}"
MASTER_PORT="${MASTER_PORT:-29500}"

echo "=== MambaGaze Phase 1 Training ==="
echo "  Config : $CONFIG"
echo "  GPUs   : $NGPU"

# Data prep (skip if already done)
if [ ! -d "/data/autogaze_preprocessed/train" ]; then
    echo "Preparing dataset ..."
    python -m mamba_gaze.scripts.prepare_data \
        --out_dir /data/autogaze_preprocessed \
        --split   train \
        --num_frames 16 \
        --num_workers 16
fi

# Launch distributed training
torchrun \
    --nproc_per_node="$NGPU" \
    --master_port="$MASTER_PORT" \
    -m mamba_gaze.training.phase1_bce \
    --config "$CONFIG"

echo "Phase 1 complete. Checkpoints in: $(python -c "
import yaml
with open('$CONFIG') as f: cfg = yaml.safe_load(f)
print(cfg.get('checkpoint', {}).get('save_dir', 'checkpoints'))
")"
