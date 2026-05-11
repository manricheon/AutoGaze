"""Adapter interfaces and shape utilities."""

from autogaze_ext.adapters.base_adapter import BaseAdapter
from autogaze_ext.adapters.compact_token_adapter import CompactTokenAdapter, CompactTokenOutput
from autogaze_ext.adapters.mllm_visual_input_adapter import MLLMVisualInput, MLLMVisualInputAdapter
from autogaze_ext.adapters.patch_grid_mapper import PatchGridMapper, PatchGridMapping
from autogaze_ext.adapters.patch_index_adapter import PatchIndexAdapter
from autogaze_ext.adapters.temporal_adapter import TemporalAdapter, TemporalAdapterOutput
from autogaze_ext.adapters.token_mask_adapter import TokenMaskAdapter, TokenMaskOutput
from autogaze_ext.adapters.vision_feature_adapter import VisionFeatureAdapter

__all__ = [
    "BaseAdapter",
    "CompactTokenAdapter",
    "CompactTokenOutput",
    "MLLMVisualInput",
    "MLLMVisualInputAdapter",
    "PatchGridMapper",
    "PatchGridMapping",
    "PatchIndexAdapter",
    "TemporalAdapter",
    "TemporalAdapterOutput",
    "TokenMaskAdapter",
    "TokenMaskOutput",
    "VisionFeatureAdapter",
]
