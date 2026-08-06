"""Minimal video-folder dataset for Borissal v1 training.

Reuses video_io.load_video (PyAV decode + uniform sampling + resize +
ImageNet normalize) -- no transformers video processor, no legacy AutoGaze
loader coupling.

AutoGaze-Training-Data layout note: the HF dataset holds 5 sub-dataset
folders, each with flat train/ and val/ splits. Pass the TRAIN dirs as a
comma-separated root (matching the original repo's `dataset.root`
convention) so val clips are not swept in by the recursive glob:

    --data-root "<D>/InternVid_res448_250K/train,<D>/100DoH_res448_250K/train,..."
"""

import glob
import os
import warnings

import torch
from torch.utils.data import Dataset

from .video_io import load_video


class VideoFolderDataset(Dataset):
    """Yields (T, C, size, size) float32 clips from every .mp4 under one or
    more roots (comma-separated, each globbed recursively). Default collate
    stacks to (B, T, C, size, size).

    Robustness for large real datasets: a clip that fails to decode is
    skipped with a warning (the next index is substituted) instead of
    killing the DataLoader worker; `max_files` caps the index for quick
    subset/debug runs.
    """

    def __init__(self, root: str, num_frames: int = 16, size: int = 384,
                 ext: str = ".mp4", max_files: int = 0):
        self.paths = []
        for r in str(root).split(","):
            r = r.strip()
            if r:
                self.paths.extend(glob.glob(os.path.join(r, "**", f"*{ext}"), recursive=True))
        self.paths = sorted(set(self.paths))
        if not self.paths:
            raise FileNotFoundError(f"no {ext} files found under {root}")
        if max_files > 0:
            self.paths = self.paths[:max_files]
        self.num_frames = num_frames
        self.size = size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # tolerate corrupt/empty files: try successive clips (bounded)
        for attempt in range(10):
            path = self.paths[(idx + attempt) % len(self.paths)]
            try:
                return load_video(path, num_frames=self.num_frames, size=self.size).squeeze(0)
            except Exception as e:  # av decode errors, zero-frame clips, ...
                warnings.warn(f"skipping unreadable clip {path}: {e}")
        raise RuntimeError(f"10 consecutive unreadable clips starting at index {idx}")
