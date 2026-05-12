from __future__ import annotations

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from autogaze_ext.scaling import (
    chunk_video_spatio_temporal,
    resize_video,
    resolve_autogaze_scaling_policy,
    scale_video_for_autogaze,
)


ROOT = Path(__file__).resolve().parents[1]


def test_default_224_resize_policy() -> None:
    policy = resolve_autogaze_scaling_policy(mode="resize", resolution=224, patch_size=16)

    assert policy.effective_resolution == 224
    assert policy.target_scales is None
    assert policy.target_patch_size is None
    assert policy.siglip_scales == "32+64+112+224"


def test_raw_384_patch14_request_normalizes_to_quick_start_392() -> None:
    policy = resolve_autogaze_scaling_policy(mode="resize", resolution=384, patch_size=14)

    assert policy.status == "normalized_to_392"
    assert policy.effective_resolution == 392
    assert policy.target_scales == (56, 112, 196, 392)
    assert policy.target_patch_size == 14
    assert policy.siglip_scales == "56+112+196+392"


def test_unsupported_resolution_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported AutoGaze scaling policy"):
        resolve_autogaze_scaling_policy(mode="resize", resolution=448, patch_size=16)


def test_resize_video_preserves_batch_time_layout() -> None:
    video = torch.randn(2, 3, 3, 32, 48)
    resized = resize_video(video, 64)

    assert list(resized.shape) == [2, 3, 3, 64, 64]


def test_scale_none_preserves_resolution() -> None:
    video = torch.randn(1, 2, 3, 32, 48)
    result = scale_video_for_autogaze(video, scaling_mode="none", resolution=64)

    assert list(result.video.shape) == [1, 2, 3, 32, 48]
    assert result.metadata["processed_resolution"] == [32, 48]


def test_scale_fit_short_side_preserves_aspect_ratio() -> None:
    video = torch.randn(1, 2, 3, 32, 64)
    result = scale_video_for_autogaze(video, scaling_mode="fit_short_side", resolution=16)

    assert list(result.video.shape) == [1, 2, 3, 16, 32]
    assert result.metadata["aspect_ratio_preserved"] is True


def test_scale_fit_long_side_preserves_aspect_ratio() -> None:
    video = torch.randn(1, 2, 3, 32, 64)
    result = scale_video_for_autogaze(video, scaling_mode="fit_long_side", resolution=16)

    assert list(result.video.shape) == [1, 2, 3, 8, 16]
    assert result.metadata["aspect_ratio_preserved"] is True


def test_quickstart_metadata_for_supported_policy() -> None:
    video = torch.randn(1, 2, 3, 32, 32)
    result = scale_video_for_autogaze(video, scaling_mode="quickstart", resolution=224, patch_size=16)

    assert result.metadata["quickstart_reference_used"] == "docs/QUICK_START_reference.md"
    assert result.metadata["quickstart_exact_match"] is True
    assert result.metadata["unsupported_reason"] is None
    assert result.metadata["processed_resolution"] == [224, 224]


def test_quickstart_unsupported_policy_raises_clear_error() -> None:
    video = torch.randn(1, 2, 3, 32, 32)

    with pytest.raises(NotImplementedError, match="scaling_mode='quickstart'"):
        scale_video_for_autogaze(video, scaling_mode="quickstart", resolution=448, patch_size=16)


def test_chop_mode_uses_quick_start_spatio_temporal_chunks() -> None:
    video = torch.randn(1, 4, 3, 32, 48)
    result = scale_video_for_autogaze(
        video,
        scaling_mode="chop",
        resolution=16,
        temporal_chunk_size=2,
        spatial_tile_size=16,
    )

    assert list(result.video.shape) == [12, 2, 3, 16, 16]
    assert result.metadata["status"] == "partial_quick_start_chop"
    assert result.metadata["chop"]["num_temporal_chunks"] == 2
    assert result.metadata["chop"]["num_spatial_tiles_w"] == 3


def test_spatio_temporal_chunking_for_quick_start_shape() -> None:
    video = torch.randn(1, 32, 3, 448, 448)
    result = chunk_video_spatio_temporal(video, temporal_chunk_size=16, spatial_tile_size=224)

    assert list(result.chunks.shape) == [8, 16, 3, 224, 224]
    assert result.metadata["num_temporal_chunks"] == 2
    assert result.metadata["num_spatial_tiles_h"] == 2
    assert result.metadata["num_spatial_tiles_w"] == 2
    assert result.metadata["chunk_records"][0]["frame_start"] == 0
    assert result.metadata["chunk_records"][-1]["frame_end_exclusive"] == 32


def test_spatio_temporal_chunking_pads_non_divisible_video() -> None:
    video = torch.randn(1, 17, 3, 300, 300)
    result = chunk_video_spatio_temporal(video, temporal_chunk_size=16, spatial_tile_size=224)

    assert list(result.chunks.shape) == [8, 16, 3, 224, 224]
    assert result.metadata["padded_shape"] == [1, 32, 3, 448, 448]
    assert result.metadata["padding"] == {"frames": 15, "height": 148, "width": 148}


def test_scaling_config_examples_exist() -> None:
    for name in [
        "resize_224",
        "resize_392_patch14",
        "spatio_temporal_224",
        "spatio_temporal_392_patch14",
    ]:
        path = ROOT / "configs" / "scaling" / f"{name}.yaml"
        cfg = OmegaConf.load(path)
        assert cfg.status in {"supported", "utility_supported"}
        assert cfg.temporal_chunk_size == 16
