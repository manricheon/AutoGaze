"""Minimal video-folder dataset for Borissal v1 training.

Reuses video_io.load_video (PyAV decode + uniform sampling + resize +
ImageNet normalize) -- no transformers video processor, no legacy AutoGaze
loader coupling. Works with the AutoGaze-Training-Data layout (a directory
of .mp4 files, recursively) as well as any ad-hoc folder of clips.
"""

import glob
import os

import torch
from torch.utils.data import Dataset

from .video_io import load_video


class VideoFolderDataset(Dataset):
    """Yields (T, C, size, size) float32 clips from every .mp4 under root
    (recursive). Default collate stacks to (B, T, C, size, size)."""

    def __init__(self, root: str, num_frames: int = 16, size: int = 384, ext: str = ".mp4"):
        self.paths = sorted(
            glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True)
        )
        if not self.paths:
            raise FileNotFoundError(f"no {ext} files found under {root}")
        self.num_frames = num_frames
        self.size = size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return load_video(self.paths[idx], num_frames=self.num_frames, size=self.size).squeeze(0)
