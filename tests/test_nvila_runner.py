import argparse
from pathlib import Path

import torch

from repro.nvila_runner import build_parser, extract_gaze_metrics, processor_kwargs, resolve_video


def make_args(**overrides):
    values = {
        "num_video_frames": 128,
        "num_video_frames_thumbnail": 64,
        "max_tiles_video": 48,
        "autogaze_model": "nvidia/AutoGaze",
        "task_loss_requirement_tile": 0.6,
        "max_batch_size_autogaze": 16,
        "hlvid_repo": "bfshi/HLVid",
        "hlvid_video_root": "data/hlvid/videos",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_processor_kwargs_match_nvila_quickstart_defaults():
    kwargs = processor_kwargs(make_args())

    assert kwargs["num_video_frames"] == 128
    assert kwargs["num_video_frames_thumbnail"] == 64
    assert kwargs["max_tiles_video"] == 48
    assert kwargs["autogaze_model_id"] == "nvidia/AutoGaze"
    assert kwargs["gazing_ratio_tile"] == [0.2] + [0.06] * 15
    assert kwargs["task_loss_requirement_tile"] == 0.6
    assert kwargs["max_batch_size_autogaze"] == 16
    assert kwargs["trust_remote_code"] is True


def test_processor_kwargs_accept_local_autogaze_checkpoint_path():
    kwargs = processor_kwargs(make_args(autogaze_model="/models/autogaze-local"))

    assert kwargs["autogaze_model_id"] == "/models/autogaze-local"


def test_parser_accepts_local_nvila_model_alias():
    args = build_parser().parse_args(["--nvila-model", "/models/nvila-local"])

    assert args.model_path == "/models/nvila-local"


def test_resolve_video_prefers_existing_local_path(tmp_path: Path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")

    assert resolve_video(str(video), make_args()) == str(video)


def test_resolve_video_builds_hf_url_for_relative_hlvid_path():
    resolved = resolve_video("clip_av_video_5_001.mp4", make_args())

    assert resolved == "https://huggingface.co/datasets/bfshi/HLVid/resolve/main/clip_av_video_5_001.mp4"


def test_extract_gaze_metrics_reads_padded_mask_when_exposed():
    metrics = extract_gaze_metrics({"gaze": {"if_padded_gazing": torch.tensor([[False, True, False]])}})

    assert metrics["autogaze_selected_patches"] == 2
    assert metrics["autogaze_padded_patches"] == 1
    assert metrics["autogaze_total_gaze_slots"] == 3
