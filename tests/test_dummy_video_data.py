from __future__ import annotations

from torch.utils.data import DataLoader

from autogaze_ext.data import (
    DummyActionRecognitionDataset,
    DummyVideoVQADataset,
    FrameSampler,
)


def test_fixed_n_uniform_sampling_preserves_original_indices() -> None:
    sampler = FrameSampler(mode="fixed", num_frames=4)
    indices = sampler.sample_indices(total_frames=10)

    assert indices.tolist() == [0, 3, 6, 9]


def test_max_frame_sampling_keeps_short_video_and_samples_long_video() -> None:
    sampler = FrameSampler(mode="max", max_frames=4)

    assert sampler.sample_indices(total_frames=3).tolist() == [0, 1, 2]
    assert sampler.sample_indices(total_frames=10).tolist() == [0, 3, 6, 9]


def test_dummy_video_vqa_sample_format() -> None:
    dataset = DummyVideoVQADataset(
        num_samples=1,
        total_frames=10,
        height=8,
        width=8,
        frame_sampler=FrameSampler(mode="fixed", num_frames=4),
    )

    sample = dataset[0]

    assert set(sample.keys()) == {"video", "metadata", "question", "answer"}
    assert sample["video"].shape == (4, 3, 8, 8)
    assert sample["question"]
    assert sample["answer"] == "dummy"
    assert sample["metadata"]["original_frame_indices"] == [0, 3, 6, 9]


def test_dummy_action_recognition_sample_format() -> None:
    dataset = DummyActionRecognitionDataset(
        num_samples=2,
        total_frames=10,
        height=8,
        width=8,
        frame_sampler=FrameSampler(mode="max", max_frames=4),
        num_classes=3,
    )

    sample = dataset[1]

    assert set(sample.keys()) == {"video", "metadata", "label"}
    assert sample["video"].shape == (4, 3, 8, 8)
    assert sample["label"] == 1
    assert sample["metadata"]["original_frame_indices"] == [0, 3, 6, 9]


def test_dataloader_batches_video_as_b_t_c_h_w() -> None:
    dataset = DummyVideoVQADataset(
        num_samples=2,
        total_frames=10,
        height=8,
        width=8,
        frame_sampler=FrameSampler(mode="fixed", num_frames=4),
    )
    batch = next(iter(DataLoader(dataset, batch_size=2)))

    assert batch["video"].shape == (2, 4, 3, 8, 8)
