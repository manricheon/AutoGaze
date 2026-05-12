from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Sequence

import torch
import torch.nn.functional as F


ScalingMode = Literal["resize", "spatio_temporal"]

DEFAULT_AUTOGAZE_SCALES = (32, 64, 112, 224)
QUICK_START_HIGH_RES_TARGET_SCALES = (56, 112, 196, 392)


@dataclass(frozen=True)
class ScalingPolicy:
    mode: ScalingMode
    requested_resolution: int
    effective_resolution: int
    patch_size: int
    target_scales: tuple[int, ...] | None
    target_patch_size: int | None
    siglip_scales: str
    temporal_chunk_size: int
    spatial_tile_size: int
    source: str
    status: str
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def autogaze_call_kwargs(self) -> dict[str, Any]:
        if self.target_scales is None or self.target_patch_size is None:
            return {}
        return {
            "target_scales": list(self.target_scales),
            "target_patch_size": self.target_patch_size,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SpatioTemporalChunks:
    chunks: torch.Tensor
    metadata: dict[str, Any]


def _as_scales(value: Sequence[int] | str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        parts = [part for part in value.split("+") if part]
    else:
        parts = list(value)
    try:
        scales = tuple(int(part) for part in parts)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"target scales must be integers, got {value!r}") from exc
    if not scales:
        return None
    return scales


def _scale_string(scales: Sequence[int]) -> str:
    return "+".join(str(int(scale)) for scale in scales)


def resolve_autogaze_scaling_policy(
    *,
    mode: ScalingMode = "resize",
    resolution: int = 224,
    patch_size: int = 16,
    target_scales: Sequence[int] | str | None = None,
    target_patch_size: int | None = None,
    temporal_chunk_size: int = 16,
    spatial_tile_size: int | None = None,
    allow_384_to_392: bool = True,
) -> ScalingPolicy:
    """Resolve a QUICK_START.md-compatible scaling policy.

    The original QUICK_START.md documents the default 16-frame 224x224 path,
    a 392x392 patch14 high-resolution target-scale path, and AnyRes-like
    spatio-temporal chunking. This helper validates those policies without
    loading AutoGaze or SigLIP.
    """
    if mode not in {"resize", "spatio_temporal"}:
        raise ValueError("mode must be 'resize' or 'spatio_temporal'")
    if resolution <= 0:
        raise ValueError("resolution must be > 0")
    if patch_size <= 0:
        raise ValueError("patch_size must be > 0")
    if temporal_chunk_size <= 0:
        raise ValueError("temporal_chunk_size must be > 0")

    configured_scales = _as_scales(target_scales)
    notes: list[str] = []
    effective_resolution = int(resolution)
    effective_patch_size = int(target_patch_size or patch_size)
    effective_target_scales = configured_scales
    effective_target_patch_size = int(target_patch_size) if target_patch_size is not None else None
    source = "configured"
    status = "configured_target_scales"

    if configured_scales is None and target_patch_size is None:
        if resolution == 224 and patch_size == 16:
            effective_resolution = 224
            effective_patch_size = 16
            effective_target_scales = None
            effective_target_patch_size = None
            source = "quick_start_default_224_patch16"
            status = "ready"
            notes.append("Default AutoGaze path: 16-frame 224x224 with patch size 16.")
        elif resolution == 384 and patch_size == 14 and allow_384_to_392:
            effective_resolution = 392
            effective_patch_size = 14
            effective_target_scales = QUICK_START_HIGH_RES_TARGET_SCALES
            effective_target_patch_size = 14
            source = "quick_start_high_res_392_patch14_from_384_request"
            status = "normalized_to_392"
            notes.append("QUICK_START uses 392 instead of raw 384 because 384 is not divisible by patch size 14.")
        elif resolution == 392 and patch_size == 14:
            effective_resolution = 392
            effective_patch_size = 14
            effective_target_scales = QUICK_START_HIGH_RES_TARGET_SCALES
            effective_target_patch_size = 14
            source = "quick_start_high_res_392_patch14"
            status = "ready"
        else:
            raise ValueError(
                "Unsupported AutoGaze scaling policy. Use 224/patch16, or the QUICK_START high-res "
                "policy 392/patch14 with target_scales=[56,112,196,392]."
            )

    if effective_target_scales is not None:
        if effective_target_patch_size is None:
            raise ValueError("target_patch_size is required when target_scales are configured")
        if len(effective_target_scales) != len(DEFAULT_AUTOGAZE_SCALES):
            raise ValueError(
                f"target_scales must keep the QUICK_START scale count {len(DEFAULT_AUTOGAZE_SCALES)}; "
                f"got {len(effective_target_scales)}"
            )
        if effective_resolution != max(effective_target_scales):
            raise ValueError("effective resolution must match the largest target scale")
        if effective_resolution % effective_target_patch_size != 0:
            raise ValueError("effective resolution must be divisible by target_patch_size")
        siglip_scales = _scale_string(effective_target_scales)
    else:
        if effective_resolution % effective_patch_size != 0:
            raise ValueError("effective resolution must be divisible by patch_size")
        siglip_scales = _scale_string(DEFAULT_AUTOGAZE_SCALES)

    tile_size = int(spatial_tile_size or effective_resolution)
    if tile_size <= 0:
        raise ValueError("spatial_tile_size must be > 0")
    if mode == "spatio_temporal":
        notes.append("Spatio-temporal mode chunks video into 16-frame spatial tiles like QUICK_START AnyRes.")
    return ScalingPolicy(
        mode=mode,
        requested_resolution=int(resolution),
        effective_resolution=effective_resolution,
        patch_size=effective_patch_size,
        target_scales=effective_target_scales,
        target_patch_size=effective_target_patch_size,
        siglip_scales=siglip_scales,
        temporal_chunk_size=int(temporal_chunk_size),
        spatial_tile_size=tile_size,
        source=source,
        status=status,
        notes=tuple(notes),
    )


def resize_video(video: torch.Tensor, resolution: int) -> torch.Tensor:
    if video.ndim != 5:
        raise ValueError(f"Expected video shape [B, T, C, H, W], got {tuple(video.shape)}")
    if resolution <= 0:
        raise ValueError("resolution must be > 0")
    batch, frames, channels, height, width = video.shape
    if height == resolution and width == resolution:
        return video
    flattened = video.reshape(batch * frames, channels, height, width)
    resized = F.interpolate(flattened, size=(resolution, resolution), mode="bilinear", align_corners=False)
    return resized.reshape(batch, frames, channels, resolution, resolution)


def chunk_video_spatio_temporal(
    video: torch.Tensor,
    *,
    temporal_chunk_size: int = 16,
    spatial_tile_size: int = 224,
    pad: bool = True,
) -> SpatioTemporalChunks:
    """Chunk [B,T,C,H,W] video into [B*Nt*Nh*Nw,t,C,h,w] tiles.

    This mirrors the QUICK_START.md `einops.rearrange` example while adding
    padding and metadata so later code can map chunks back to original frames
    and spatial tiles.
    """
    if video.ndim != 5:
        raise ValueError(f"Expected video shape [B, T, C, H, W], got {tuple(video.shape)}")
    if temporal_chunk_size <= 0:
        raise ValueError("temporal_chunk_size must be > 0")
    if spatial_tile_size <= 0:
        raise ValueError("spatial_tile_size must be > 0")

    batch, frames, channels, height, width = [int(dim) for dim in video.shape]
    num_temporal_chunks = math.ceil(frames / temporal_chunk_size)
    num_tiles_h = math.ceil(height / spatial_tile_size)
    num_tiles_w = math.ceil(width / spatial_tile_size)
    padded_frames = num_temporal_chunks * temporal_chunk_size
    padded_height = num_tiles_h * spatial_tile_size
    padded_width = num_tiles_w * spatial_tile_size

    if not pad and (padded_frames != frames or padded_height != height or padded_width != width):
        raise ValueError(
            "video dimensions must be divisible by temporal_chunk_size and spatial_tile_size when pad=False"
        )

    padded = video
    if padded_frames != frames or padded_height != height or padded_width != width:
        padded = video.new_zeros((batch, padded_frames, channels, padded_height, padded_width))
        padded[:, :frames, :, :height, :width] = video

    chunks = (
        padded.reshape(
            batch,
            num_temporal_chunks,
            temporal_chunk_size,
            channels,
            num_tiles_h,
            spatial_tile_size,
            num_tiles_w,
            spatial_tile_size,
        )
        .permute(0, 1, 4, 6, 2, 3, 5, 7)
        .reshape(
            batch * num_temporal_chunks * num_tiles_h * num_tiles_w,
            temporal_chunk_size,
            channels,
            spatial_tile_size,
            spatial_tile_size,
        )
    )

    records: list[dict[str, int]] = []
    chunk_index = 0
    for batch_index in range(batch):
        for temporal_index in range(num_temporal_chunks):
            for tile_row in range(num_tiles_h):
                for tile_col in range(num_tiles_w):
                    frame_start = temporal_index * temporal_chunk_size
                    height_start = tile_row * spatial_tile_size
                    width_start = tile_col * spatial_tile_size
                    records.append(
                        {
                            "chunk_index": chunk_index,
                            "batch_index": batch_index,
                            "temporal_chunk_index": temporal_index,
                            "spatial_tile_row": tile_row,
                            "spatial_tile_col": tile_col,
                            "frame_start": frame_start,
                            "frame_end_exclusive": min(frame_start + temporal_chunk_size, frames),
                            "height_start": height_start,
                            "height_end_exclusive": min(height_start + spatial_tile_size, height),
                            "width_start": width_start,
                            "width_end_exclusive": min(width_start + spatial_tile_size, width),
                        }
                    )
                    chunk_index += 1

    metadata = {
        "mode": "spatio_temporal",
        "original_shape": [batch, frames, channels, height, width],
        "padded_shape": [batch, padded_frames, channels, padded_height, padded_width],
        "chunks_shape": [int(dim) for dim in chunks.shape],
        "temporal_chunk_size": temporal_chunk_size,
        "spatial_tile_size": spatial_tile_size,
        "num_temporal_chunks": num_temporal_chunks,
        "num_spatial_tiles_h": num_tiles_h,
        "num_spatial_tiles_w": num_tiles_w,
        "padding": {
            "frames": padded_frames - frames,
            "height": padded_height - height,
            "width": padded_width - width,
        },
        "chunk_records": records,
    }
    return SpatioTemporalChunks(chunks=chunks, metadata=metadata)
