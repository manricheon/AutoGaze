#!/usr/bin/env bash
# Download and extract AutoGaze training data from HuggingFace.
# Usage:
#   bash scripts/download_data.sh [TARGET_DIR] [SUBSET]
#
# Arguments:
#   TARGET_DIR : Directory to download data into (default: ./data/AutoGaze-Training-Data)
#   SUBSET     : Which subset to download (default: all)
#                Options: all | internvid | 100doh | ego4d | scanning_sam | scanning_idl | labels
#
# Data sizes (approximate, compressed):
#   InternVid_res448_250K   :  90 GB
#   100DoH_res448_250K      : 157 GB
#   Ego4D_res448_250K       : 298 GB
#   scanning_SAM_res448_50K :  59 GB
#   scanning_idl_res448_50K :  38 GB
#   gazing_labels.json      :   4 GB
#   Total                   : ~646 GB
#
# For NTP pre-training, InternVid + gazing_labels is a minimal start.

set -euo pipefail

TARGET_DIR="${1:-./data/AutoGaze-Training-Data}"
SUBSET="${2:-all}"
mkdir -p "$TARGET_DIR"

REPO="bfshi/AutoGaze-Training-Data"

download_and_extract() {
    local filename="$1"
    local label="$2"
    echo "--- Downloading $label ($filename) ---"
    huggingface-cli download "$REPO" "$filename" \
        --repo-type dataset \
        --local-dir "$TARGET_DIR"
    echo "--- Extracting $filename ---"
    tar -xzf "$TARGET_DIR/$filename" -C "$TARGET_DIR"
    echo "--- Done: $label ---"
}

download_labels() {
    echo "--- Downloading gazing_labels.json ---"
    huggingface-cli download "$REPO" gazing_labels.json \
        --repo-type dataset \
        --local-dir "$TARGET_DIR"
    echo "--- Done: gazing_labels.json ---"
}

case "$SUBSET" in
    all)
        download_labels
        download_and_extract "InternVid_res448_250K.tar.gz"  "InternVid (~90 GB)"
        download_and_extract "100DoH_res448_250K.tar.gz"     "100DoH (~157 GB)"
        download_and_extract "Ego4D_res448_250K.tar.gz"      "Ego4D (~298 GB)"
        download_and_extract "scanning_SAM_res448_50K.tar.gz" "scanning_SAM (~59 GB)"
        download_and_extract "scanning_idl_res448_50K.tar.gz" "scanning_idl (~38 GB)"
        ;;
    internvid)
        download_labels
        download_and_extract "InternVid_res448_250K.tar.gz" "InternVid (~90 GB)"
        ;;
    100doh)
        download_and_extract "100DoH_res448_250K.tar.gz" "100DoH (~157 GB)"
        ;;
    ego4d)
        download_and_extract "Ego4D_res448_250K.tar.gz" "Ego4D (~298 GB)"
        ;;
    scanning_sam)
        download_and_extract "scanning_SAM_res448_50K.tar.gz" "scanning_SAM (~59 GB)"
        ;;
    scanning_idl)
        download_and_extract "scanning_idl_res448_50K.tar.gz" "scanning_idl (~38 GB)"
        ;;
    labels)
        download_labels
        ;;
    *)
        echo "Unknown subset: $SUBSET"
        echo "Options: all | internvid | 100doh | ego4d | scanning_sam | scanning_idl | labels"
        exit 1
        ;;
esac

echo ""
echo "=== Download complete ==="
echo "Data directory: $TARGET_DIR"
echo ""
echo "Expected structure:"
echo "  $TARGET_DIR/"
echo "  ├── InternVid_res448_250K/{train,val}/"
echo "  ├── 100DoH_res448_250K/{train,val}/"
echo "  ├── Ego4D_res448_250K/{train,val}/"
echo "  ├── scanning_SAM_res448_50K/{train,val}/"
echo "  ├── scanning_idl_res448_50K/{train,val}/"
echo "  └── gazing_labels.json"
