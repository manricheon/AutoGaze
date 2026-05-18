import argparse
from pathlib import Path

import torch

from repro.nvila_runner import (
    StageProfiler,
    apply_resize_to_dimensions,
    build_parser,
    compute_visual_token_metrics,
    estimate_nvila_preflight,
    extract_gaze_metrics,
    model_patches_per_frame,
    parse_int_sequence,
    parse_args,
    processor_kwargs,
    resolve_video,
    spatial_tile_grid,
    uniform_sample_indices,
)


def make_args(**overrides):
    values = {
        "num_video_frames": 128,
        "num_video_frames_thumbnail": 64,
        "max_tiles_video": 48,
        "autogaze_model": "nvidia/AutoGaze",
        "gazing_mode": "autogaze",
        "autogaze_target_scales": None,
        "autogaze_target_patch_size": None,
        "task_loss_requirement_tile": 0.6,
        "max_batch_size_autogaze": 16,
        "hlvid_repo": "bfshi/HLVid",
        "hlvid_video_root": "data/hlvid/videos",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class DummyVisionConfig:
    def __init__(self, scales, patch_size=14):
        self.scales = scales
        self.patch_size = patch_size


class DummyVisionTower:
    def __init__(self, scales, patch_size=14):
        self.config = DummyVisionConfig(scales, patch_size)


class DummyModel:
    def __init__(self, scales, patch_size=14):
        self.vision_tower = DummyVisionTower(scales, patch_size)


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


def test_processor_kwargs_can_run_keep_all_baseline_without_autogaze_selection():
    kwargs = processor_kwargs(make_args(gazing_mode="keep-all"))

    assert kwargs["gazing_ratio_tile"] == 1
    assert kwargs["task_loss_requirement_tile"] is None
    assert kwargs["gazing_ratio_thumbnail"] == 1
    assert kwargs["task_loss_requirement_thumbnail"] is None


def test_processor_kwargs_forwards_autogaze_resize_scales_to_nvila_processor():
    kwargs = processor_kwargs(
        make_args(
            autogaze_target_scales="56+112+196+392",
            autogaze_target_patch_size=14,
        )
    )

    assert kwargs["target_scales"] == [56, 112, 196, 392]
    assert kwargs["target_patch_size"] == 14


def test_parse_args_accepts_gazing_mode_switch():
    args = parse_args(["--gazing-mode", "keep-all"])

    assert args.gazing_mode == "keep-all"


def test_parse_args_accepts_video_and_autogaze_resize_options():
    args = parse_args(
        [
            "--video-resize-shortest-edge",
            "720",
            "--autogaze-resize-scales",
            "56+112+196+392",
            "--autogaze-target-patch-size",
            "14",
        ]
    )

    assert args.video_resize_shortest_edge == 720
    assert args.autogaze_target_scales == "56+112+196+392"
    assert args.autogaze_target_patch_size == 14


def test_parse_int_sequence_accepts_plus_comma_and_bracket_formats():
    assert parse_int_sequence("56+112+196+392") == [56, 112, 196, 392]
    assert parse_int_sequence("[56, 112, 196, 392]") == [56, 112, 196, 392]


def test_apply_resize_to_dimensions_preserves_aspect_for_shortest_edge():
    resized = apply_resize_to_dimensions(
        width=3840,
        height=2160,
        shortest_edge=720,
        longest_edge=None,
        exact_width=None,
        exact_height=None,
    )

    assert resized == {"width": 1280, "height": 720, "mode": "shortest_edge"}


def test_uniform_sample_indices_matches_nvila_processor_round_linspace():
    assert uniform_sample_indices(total_frames=10, sample_count=4) == [0, 3, 6, 9]


def test_parser_accepts_local_nvila_model_alias():
    args = build_parser().parse_args(["--nvila-model", "/models/nvila-local"])

    assert args.model_path == "/models/nvila-local"


def test_parse_args_loads_nvila_runner_preset_config(tmp_path: Path):
    preset = tmp_path / "preset.yaml"
    preset.write_text(
        "\n".join(
            [
                "nvila_runner:",
                "  args:",
                "    video: inputs/hf_space_autogaze/doorbell.mp4",
                "    num_video_frames: 1024",
                "    max_tiles_video: 48",
                "    model_path: /models/nvila-local",
            ]
        )
        + "\n"
    )

    args = parse_args(["--preset-config", str(preset)])

    assert args.video == "inputs/hf_space_autogaze/doorbell.mp4"
    assert args.num_video_frames == 1024
    assert args.max_tiles_video == 48
    assert args.model_path == "/models/nvila-local"


def test_cli_values_override_preset_config(tmp_path: Path):
    preset = tmp_path / "preset.yaml"
    preset.write_text("nvila_runner:\n  args:\n    num_video_frames: 1024\n")

    args = parse_args(["--preset-config", str(preset), "--num-video-frames", "128"])

    assert args.num_video_frames == 128


def test_parse_args_loads_committed_hf_space_preset():
    root = Path(__file__).resolve().parents[1]
    preset = root / "configs" / "repro" / "hf_space_autogaze_examples.yaml"

    args = parse_args(["--preset-config", str(preset)])

    assert args.video == "inputs/hf_space_autogaze/doorbell.mp4"
    assert args.num_video_frames == 128
    assert args.max_tiles_video == 48


def test_spatial_tile_grid_matches_nvila_dynamic_tiling_for_4k_16x9():
    grid = spatial_tile_grid(width=3840, height=2160, max_tiles_video=48, image_size=392)

    assert grid == {"cols": 9, "rows": 5, "tiles": 45}


def test_preflight_estimator_flags_4k_1024_frame_keep_all_context_risk():
    estimate = estimate_nvila_preflight(
        width=3840,
        height=2160,
        source_frames=9000,
        num_video_frames=1024,
        num_video_frames_thumbnail=64,
        max_tiles_video=48,
    )

    assert estimate["sampling"]["requested_frames"] == 1024
    assert estimate["tiling"]["spatial_tiles"] == 45
    assert estimate["chunking"]["temporal_chunks"] == 64
    assert estimate["counts"]["tile_images"] == 46080
    assert estimate["tokens"]["keep_all_projected_tokens"] > estimate["tokens"]["llm_context_limit"]
    assert "context" in estimate["risk_flags"]
    assert "cpu_memory" in estimate["risk_flags"]


def test_parse_args_accepts_preflight_mode_and_output_path():
    args = parse_args(["--mode", "preflight", "--preflight-json", "out/preflight.json"])

    assert args.mode == "preflight"
    assert args.preflight_json == "out/preflight.json"


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


def test_compute_visual_token_metrics_compares_keep_all_and_autogaze_tokens():
    payload = {
        "input_ids": torch.tensor([[11, 32000, 12, 32000]]),
        "pixel_values_videos_tiles": [torch.zeros(2, 2, 3, 4, 4)],
        "pixel_values_videos_thumbnails": [torch.zeros(1, 1, 3, 4, 4)],
        "num_spatial_tiles_each_video": [2],
        "gazing_info": {
            "if_padded_gazing_tiles": [torch.tensor([[False, True, False], [True, False, False]])],
            "if_padded_gazing_thumbnails": [torch.tensor([[False, True]])],
        },
    }

    metrics = compute_visual_token_metrics(
        payload,
        video_token_id=32000,
        patches_per_frame_value=4,
        patches_per_frame_by_scale={"scale_a": 1, "scale_b": 3},
        token_shuffle=2,
    )

    assert metrics["video_sampled_frames"] == 2
    assert metrics["thumbnail_sampled_frames"] == 1
    assert metrics["tile_sequences"] == 2
    assert metrics["spatial_tiles_per_video"] == [2]
    assert metrics["temporal_chunks_per_video"] == [1]
    assert metrics["encoder_patches_per_frame_multiscale"] == 4
    assert metrics["encoder_patches_per_frame_by_scale"] == {"scale_a": 1, "scale_b": 3}
    assert metrics["encoder_raw_tile_patch_tokens"] == 16
    assert metrics["encoder_autogaze_selected_tile_patch_tokens"] == 4
    assert metrics["encoder_tile_token_reduction_ratio"] == 4.0
    assert metrics["encoder_raw_thumbnail_patch_tokens"] == 4
    assert metrics["encoder_autogaze_selected_thumbnail_patch_tokens"] == 1
    assert metrics["encoder_thumbnail_token_reduction_ratio"] == 4.0
    assert metrics["encoder_raw_patch_tokens"] == 20
    assert metrics["encoder_autogaze_selected_patch_tokens"] == 5
    assert metrics["encoder_token_reduction_ratio"] == 4.0
    assert metrics["llm_keep_all_visual_tokens_estimated"] == 10
    assert metrics["llm_actual_visual_tokens"] == 2
    assert metrics["llm_visual_token_reduction_ratio"] == 5.0


def test_model_patches_per_frame_accepts_plus_separated_nvila_scales():
    patches = model_patches_per_frame(DummyModel("56+112+196+392"))

    assert patches == 1060


def test_stage_profiler_records_and_resets_timing():
    profiler = StageProfiler()

    with profiler.measure("video_decode"):
        value = 1 + 1

    timings = profiler.as_dict()
    assert value == 2
    assert timings["video_decode"]["count"] == 1
    assert timings["video_decode"]["total_ms"] >= 0

    profiler.reset()

    assert profiler.as_dict() == {}
