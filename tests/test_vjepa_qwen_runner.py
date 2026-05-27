from types import SimpleNamespace

import pytest
from PIL import Image

from repro.plugins.autogaze_sparse_selector import AutogazeSelectorRuntimeConfig
from repro.vjepa_qwen_runner import (
    build_parser,
    build_selector_config_from_args,
    pil_frames_to_vjepa_pixel_values,
    vjepa_resize_plan_from_args,
)


def test_vjepa_qwen_runner_defaults_wire_actual_autogaze():
    args = build_parser().parse_args(["--video", "inputs/example.mp4"])

    assert args.video == "inputs/example.mp4"
    assert args.autogaze_model == "nvidia/AutoGaze"
    assert args.vjepa_model == "facebook/vjepa2-vitl-fpc64-256"
    assert args.qwen_model == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert args.frames_per_clip == 16
    assert args.num_video_frames == 16
    assert args.autogaze_chunk_frames == 16
    assert args.autogaze_tile_size == 392
    assert args.autogaze_target_scales == "56+112+196+392"
    assert args.vjepa_selection_policy == "single_scale_union"
    assert args.output_json.endswith("vjepa_qwen_actual.json")


def test_build_selector_config_from_args_matches_video_sampling_and_resize(tmp_path):
    args = SimpleNamespace(
        video="inputs/example.mp4",
        output_json=str(tmp_path / "actual.json"),
        autogaze_selector_output_json=None,
        sparse_selection_plan_json=None,
        autogaze_repo=".",
        autogaze_model="weights/AutoGaze",
        autogaze_device="cuda",
        autogaze_dtype="float16",
        num_video_frames=32,
        num_video_frames_thumbnail=0,
        qwen_thumbnail_mode="none",
        autogaze_chunk_frames=16,
        max_tiles_video=4,
        autogaze_tile_size=224,
        max_batch_size_autogaze=8,
        gazing_ratio=0.1,
        task_loss_requirement=None,
        autogaze_target_scales="112+224",
        autogaze_target_patch_size=16,
        autogaze_encoder_patch_size=16,
        autogaze_generate_only=True,
        video_decode_strategy="seek",
        video_resize_shortest_edge=None,
        video_resize_longest_edge=448,
        video_resize_width=None,
        video_resize_height=None,
    )

    config = build_selector_config_from_args(args)

    assert isinstance(config, AutogazeSelectorRuntimeConfig)
    assert config.autogaze_model == "weights/AutoGaze"
    assert config.num_video_frames == 32
    assert config.chunk_frames == 16
    assert config.max_tiles_video == 4
    assert config.max_batch_size == 8
    assert config.gazing_ratio == 0.1
    assert config.target_scales == [112, 224]
    assert config.generate_only is True
    assert config.video_decode_strategy == "seek"
    assert config.video_resize_longest_edge == 448
    assert config.output_json.endswith("actual_autogaze_sparse_plan.json")


def test_pil_frames_to_vjepa_pixel_values_shape_and_dtype():
    torch = pytest.importorskip("torch")
    frames = [
        Image.new("RGB", (32, 24), color=(255, 0, 0)),
        Image.new("RGB", (24, 32), color=(0, 255, 0)),
    ]

    values = pil_frames_to_vjepa_pixel_values(
        frames,
        crop_size=16,
        dtype=torch.float32,
        device="cpu",
    )

    assert list(values.shape) == [1, 2, 3, 16, 16]
    assert values.dtype == torch.float32
    assert values.device.type == "cpu"


def test_vjepa_resize_plan_prefers_exact_crop_for_encoder_inputs():
    args = SimpleNamespace(
        video_resize_shortest_edge=None,
        video_resize_longest_edge=448,
        video_resize_width=None,
        video_resize_height=None,
        crop_size=224,
    )

    resize = vjepa_resize_plan_from_args(args)

    assert resize == {"width": 224, "height": 224, "mode": "exact"}
