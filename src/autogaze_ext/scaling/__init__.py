"""Scaling helpers for QUICK_START-aligned AutoGaze experiments."""

from autogaze_ext.scaling.autogaze_scaling import (
    DEFAULT_AUTOGAZE_SCALES,
    QUICK_START_HIGH_RES_TARGET_SCALES,
    ScaledVideo,
    ScalingPolicy,
    SpatioTemporalChunks,
    chunk_video_spatio_temporal,
    resize_video,
    resolve_autogaze_scaling_policy,
    scale_video_for_autogaze,
)

__all__ = [
    "DEFAULT_AUTOGAZE_SCALES",
    "QUICK_START_HIGH_RES_TARGET_SCALES",
    "ScaledVideo",
    "ScalingPolicy",
    "SpatioTemporalChunks",
    "chunk_video_spatio_temporal",
    "resize_video",
    "resolve_autogaze_scaling_policy",
    "scale_video_for_autogaze",
]
