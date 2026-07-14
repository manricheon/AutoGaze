"""Configuration for Borissal v0, the non-learned feed-forward patch selector (Phase 1)."""

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


@dataclass
class BorissalV1Config:
    """Config for Borissal v1, the learned selector (Phase 2).

    Shares v0's grid semantics: grid_thw = (T // tubelet_size,
    H // patch_size, W // patch_size). The v0 saliency signal is computed
    internally (non-learned, cheap) whenever `input_mode` includes maps or
    `residual_scoring` is on.
    """

    scale: int = 384
    patch_size: int = 16
    tubelet_size: int = 2

    # What the learned backbone sees, at grid resolution (H_grid x W_grid):
    #   "maps"   -- v0's normalized motion+spatial patch maps only (2 channels)
    #   "pixels" -- grid-downsampled RGB of the tubelet mean only (3 channels)
    #   "both"   -- concat of the two (5 channels)
    input_mode: Literal["maps", "pixels", "both"] = "both"

    hidden_channels: int = 64
    num_blocks: int = 3
    # Fraction of channels shifted forward/backward one tubelet per TSM block.
    shift_fraction: float = 0.25

    # score = v0_score + f_theta(...) instead of score = f_theta(...).
    residual_scoring: bool = False

    gazing_ratio: float = 0.5
    per_frame_allocation: Literal["uniform", "proportional"] = "uniform"

    # Gumbel noise scale for the straight-through training path (0 disables noise).
    gumbel_tau: float = 1.0

    eps: float = 1e-6

    # v0 signal settings (used for maps input / residual scoring).
    motion_weight: Union[float, Literal["auto"]] = 0.5
    spatial_op: Literal["grad", "sobel"] = "grad"
    pooling: Literal["avg", "max"] = "avg"
