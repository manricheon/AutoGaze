"""Configuration for Borissal, the signal-based feed-forward patch selector (Phase 1)."""

from dataclasses import dataclass, field
from typing import Literal, Union


@dataclass
class BorissalConfig:
    """Config for the non-learned saliency selector.

    grid_thw for a clip of shape (T, H, W) is derived as
    (T // tubelet_size, H // patch_size, W // patch_size).
    """

    scale: int = 384
    patch_size: int = 16
    tubelet_size: int = 2

    motion_weight: Union[float, Literal["auto"]] = 0.5
    """Fixed blend weight in [0, 1], or "auto" to derive it per-clip from the
    clip's own (pre-normalization) motion vs. spatial energy -- still
    non-learned, just data-adaptive: motion_weight = motion_energy /
    (motion_energy + spatial_energy)."""
    spatial_op: Literal["grad", "sobel"] = "grad"
    pooling: Literal["avg", "max"] = "avg"

    gazing_ratio: float = 0.5
    per_frame_allocation: Literal["uniform", "proportional"] = "uniform"

    eps: float = 1e-6

    image_mean: tuple = field(default_factory=lambda: (0.485, 0.456, 0.406))
    image_std: tuple = field(default_factory=lambda: (0.229, 0.224, 0.225))
