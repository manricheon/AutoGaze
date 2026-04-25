#!/usr/bin/env bash
# Run AutoGaze inference on a video file or directory.
#
# Usage:
#   bash scripts/run_inference.sh <INPUT> [OUTPUT_DIR] [OPTIONS]
#
# Examples:
#   # Single video, all output formats (json + viz + npy)
#   bash scripts/run_inference.sh assets/example_input.mp4
#
#   # Directory of videos, JSON only (generate NTP training labels)
#   bash scripts/run_inference.sh /data/my_videos/ results/labels/ --output-format json
#
#   # Custom gazing ratio, local model weights
#   bash scripts/run_inference.sh video.mp4 results/ \
#       --model-path weights/AutoGaze \
#       --gazing-ratio 0.5

set -euo pipefail

INPUT="${1:?Usage: $0 <input_video_or_dir> [output_dir] [options...]}"
OUTPUT_DIR="${2:-results}"
shift 2 || true   # remaining args forwarded to infer.py

python -m autogaze.infer \
    "$INPUT" \
    --model-path "${MODEL_PATH:-weights/AutoGaze}" \
    --output-dir "$OUTPUT_DIR" \
    "$@"
