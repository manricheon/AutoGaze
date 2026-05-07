#!/usr/bin/env bash
# Full evaluation benchmark: latency + reconstruction + downstream.
#
# Usage:
#   bash mamba_gaze/scripts/benchmark.sh [CKPT] [AUTOGAZE_CKPT] [CONFIG]

set -euo pipefail

CKPT="${1:-$(ls -t checkpoints/phase2_ste/phase2ste_epoch*.pt 2>/dev/null | head -1)}"
AG_CKPT="${2:-weights/autogaze.pt}"
CONFIG="${3:-mamba_gaze/configs/default.yaml}"
RESULTS_DIR="results/$(date +%Y%m%d_%H%M%S)"

if [ -z "$CKPT" ]; then
    echo "Error: no checkpoint found. Pass path as first argument."
    exit 1
fi

mkdir -p "$RESULTS_DIR"
echo "=== MambaGaze Benchmark ==="
echo "  Checkpoint : $CKPT"
echo "  Results    : $RESULTS_DIR"

# 1. Latency
echo ""
echo "── 1/3 Latency ──"
python -m mamba_gaze.eval.latency \
    --config "$CONFIG" \
    --ckpt   "$CKPT" \
    --autogaze_ckpt "$AG_CKPT" \
    --out    "$RESULTS_DIR/latency.csv"

# 2. Reconstruction quality
echo ""
echo "── 2/3 Reconstruction ──"
python -m mamba_gaze.eval.reconstruction \
    --config "$CONFIG" \
    --ckpt   "$CKPT" \
    --recon_model weights/videomae_recon.pt \
    --split  val \
    --gazing_ratio 0.5 \
    2>&1 | tee "$RESULTS_DIR/reconstruction.txt"

# 3. Downstream (VideoMME) — requires lmms-eval + LLM weights
if python -c "import lmms_eval" 2>/dev/null; then
    echo ""
    echo "── 3/3 VideoMME ──"
    python -m mamba_gaze.eval.downstream \
        --config "$CONFIG" \
        --ckpt   "$CKPT" \
        --task   videomme \
        --gazing_ratio 0.5 \
        --output "$RESULTS_DIR/videomme" \
        2>&1 | tee "$RESULTS_DIR/videomme.txt"
else
    echo "── 3/3 VideoMME: skipped (lmms-eval not installed) ──"
fi

echo ""
echo "All results saved to: $RESULTS_DIR"
