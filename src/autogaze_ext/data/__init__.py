"""Data interfaces for the AutoGaze extension PoC."""

from autogaze_ext.data.dummy_video_dataset import (
    DummyActionRecognitionDataset,
    DummyVideoDataset,
    DummyVideoVQADataset,
)
from autogaze_ext.data.frame_selector import FrameSelectionResult, FrameSelector, FrameWindow, select_frame_windows
from autogaze_ext.data.frame_sampler import FrameSampler, FrameSamplerOutput
from autogaze_ext.data.hf_dataset_loader import HFDatasetLoader, LocalListDataset

__all__ = [
    "DummyActionRecognitionDataset",
    "DummyVideoDataset",
    "DummyVideoVQADataset",
    "FrameSelectionResult",
    "FrameSelector",
    "FrameSampler",
    "FrameSamplerOutput",
    "FrameWindow",
    "HFDatasetLoader",
    "LocalListDataset",
    "select_frame_windows",
]
