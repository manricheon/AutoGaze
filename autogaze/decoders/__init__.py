# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from .base import TaskDecoder
from .recognition import RecognitionDecoder
from .detection import DetectionDecoder
from .segmentation import SegmentationDecoder
from .depth import DepthDecoder

__all__ = [
    'TaskDecoder',
    'RecognitionDecoder',
    'DetectionDecoder',
    'SegmentationDecoder',
    'DepthDecoder',
]
