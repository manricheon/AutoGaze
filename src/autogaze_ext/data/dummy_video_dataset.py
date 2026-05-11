from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from autogaze_ext.data.frame_sampler import FrameSampler


class DummyVideoDataset(Dataset):
    """Deterministic dummy video dataset returning samples shaped [T, C, H, W]."""

    def __init__(
        self,
        num_samples: int = 2,
        total_frames: int = 8,
        channels: int = 3,
        height: int = 224,
        width: int = 224,
        frame_sampler: FrameSampler | None = None,
    ) -> None:
        if num_samples <= 0:
            raise ValueError("num_samples must be > 0")
        if total_frames <= 0:
            raise ValueError("total_frames must be > 0")
        if channels <= 0 or height <= 0 or width <= 0:
            raise ValueError("channels, height, and width must be > 0")

        self.num_samples = num_samples
        self.total_frames = total_frames
        self.channels = channels
        self.height = height
        self.width = width
        self.frame_sampler = frame_sampler

    def __len__(self) -> int:
        return self.num_samples

    def _make_video(self, index: int) -> torch.Tensor:
        shape = (self.total_frames, self.channels, self.height, self.width)
        video = torch.arange(self._num_elements, dtype=torch.float32).reshape(shape)
        return video + float(index)

    @property
    def _num_elements(self) -> int:
        return self.total_frames * self.channels * self.height * self.width

    def _base_sample(self, index: int) -> dict[str, Any]:
        video = self._make_video(index)
        metadata: dict[str, Any] = {
            "sample_index": index,
            "source": "dummy",
            "original_frame_count": self.total_frames,
        }

        if self.frame_sampler is not None:
            sampled = self.frame_sampler(video)
            video = sampled.video
            metadata.update(sampled.metadata)
        else:
            metadata.update(
                {
                    "sampling_mode": "none",
                    "input_frame_count": self.total_frames,
                    "sampled_frame_count": self.total_frames,
                    "original_frame_indices": list(range(self.total_frames)),
                }
            )

        return {"video": video, "metadata": metadata}

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= self.num_samples:
            raise IndexError(index)
        return self._base_sample(index)


class DummyVideoVQADataset(DummyVideoDataset):
    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = super().__getitem__(index)
        sample["question"] = f"What is shown in dummy video {index}?"
        sample["answer"] = "dummy"
        return sample


class DummyActionRecognitionDataset(DummyVideoDataset):
    def __init__(self, *args: Any, num_classes: int = 4, **kwargs: Any) -> None:
        if num_classes <= 0:
            raise ValueError("num_classes must be > 0")
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = super().__getitem__(index)
        sample["label"] = index % self.num_classes
        return sample
