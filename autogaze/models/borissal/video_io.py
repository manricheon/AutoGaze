"""Lightweight video decode + preprocessing for Borissal, independent of any
transformers video processor (Mac/CPU/MPS friendly, no heavy deps beyond
av/numpy/PIL/torch, which the repo already depends on)."""

import numpy as np
import torch
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def sample_frame_indices(total_frames: int, num_frames: int) -> np.ndarray:
    """Uniformly spaced frame indices covering the whole clip (matches
    autogaze/datasets/video_utils.py sample_frame_indices's default behavior)."""
    idx = np.linspace(0, total_frames - 1, num=num_frames)
    return np.round(idx).astype(np.int64)


def load_video(
    path: str,
    num_frames: int = 16,
    size: int = 384,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
) -> torch.Tensor:
    """Decode a video file, sample `num_frames` frames, resize to (size, size),
    and normalize. Returns a (1, T, 3, size, size) float32 tensor."""
    import av  # local import: keep av optional for pure-tensor callers/tests

    container = av.open(path)
    try:
        stream = container.streams.video[0]
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
    finally:
        container.close()

    total = len(frames)
    if total == 0:
        raise ValueError(f"No frames decoded from {path}")

    indices = sample_frame_indices(total, num_frames)
    resized = [
        np.array(Image.fromarray(frames[i]).resize((size, size), Image.BILINEAR))
        for i in indices
    ]
    arr = np.stack(resized).astype(np.float32) / 255.0  # (T, H, W, C)
    arr = (arr - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).unsqueeze(0).contiguous().float()  # (1, T, C, H, W)
    return tensor


def unnormalize(video: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> torch.Tensor:
    """Inverse of load_video's normalization, for rendering. video: (..., 3, H, W)."""
    mean_t = torch.tensor(mean, dtype=video.dtype, device=video.device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=video.dtype, device=video.device).view(1, 3, 1, 1)
    shape = video.shape
    flat = video.reshape(-1, *shape[-3:])
    out = (flat * std_t + mean_t).clamp(0, 1)
    return out.reshape(shape)
