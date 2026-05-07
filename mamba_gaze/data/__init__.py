from .autogaze_dataset import AutoGazeDataset, build_dataloader
from .mask_converter import (
    seq_to_multihot, multihot_to_per_scale, seq_to_per_scale,
    per_scale_to_multihot, per_scale_to_seq, batch_seq_to_multihot,
    SCALES, SCALE_PATCHES, SCALE_HW, SCALE_OFFSETS, N_TOKENS,
)

__all__ = [
    "AutoGazeDataset", "build_dataloader",
    "seq_to_multihot", "multihot_to_per_scale", "seq_to_per_scale",
    "per_scale_to_multihot", "per_scale_to_seq", "batch_seq_to_multihot",
    "SCALES", "SCALE_PATCHES", "SCALE_HW", "SCALE_OFFSETS", "N_TOKENS",
]
