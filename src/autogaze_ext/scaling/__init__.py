"""Scaling helpers for QUICK_START-aligned AutoGaze experiments."""

from autogaze_ext.scaling.autogaze_scaling import (
    DEFAULT_AUTOGAZE_SCALES,
    QUICK_START_HIGH_RES_TARGET_SCALES,
    ScalingPolicy,
    SpatioTemporalChunks,
    chunk_video_spatio_temporal,
    resize_video,
    resolve_autogaze_scaling_policy,
)

__all__ = [
    "DEFAULT_AUTOGAZE_SCALES",
    "QUICK_START_HIGH_RES_TARGET_SCALES",
    "ScalingPolicy",
    "SpatioTemporalChunks",
    "chunk_video_spatio_temporal",
    "resize_video",
    "resolve_autogaze_scaling_policy",
]
