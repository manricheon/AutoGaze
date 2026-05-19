import argparse
from fractions import Fraction
from pathlib import Path

import torch
from PIL import Image

from repro.nvila_runner import (
    StageProfiler,
    apply_resize_to_dimensions,
    autogaze_processor_size_kwargs,
    build_autogaze_effect_metrics,
    build_seek_decode_groups,
    build_keep_all_gazing_info,
    build_parser,
    build_autogaze_token_summary,
    build_video_input_summary,
    build_stream_profile_compute_metrics,
    build_stream_profile_token_metrics,
    compute_visual_token_metrics,
    estimate_siglip_encoder_compute,
    estimate_nvila_preflight,
    estimate_stream_profile_plan,
    extract_gaze_metrics,
    model_patches_per_frame,
    parse_int_sequence,
    parse_args,
    processor_kwargs,
    processor_videos_argument,
    build_single_summary,
    frame_index_to_pts,
    pts_to_frame_index,
    stream_pts_per_frame,
    repeat_last_stream_samples_after_eof,
    resolve_video,
    spatial_tile_grid,
    summarize_repeat_results,
    summarize_token_budget_rows,
    summarize_stream_chunks,
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


class DummyTransformerConfig:
    def __init__(
        self,
        *,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads=None,
        scales="56+112",
        patch_size=14,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.scales = scales
        self.patch_size = patch_size


class DummyFullModel:
    def __init__(self):
        vision_config = DummyTransformerConfig(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            scales=[14, 28],
            patch_size=14,
        )
        text_config = DummyTransformerConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
        self.vision_tower = DummyVisionTower([14, 28], patch_size=14)
        self.vision_tower.config = vision_config
        self.config = argparse.Namespace(text_config=text_config)


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


def test_parse_args_accepts_warmup_and_repeat_runs():
    args = parse_args(["--warmup-runs", "1", "--repeat-runs", "3"])

    assert args.warmup_runs == 1
    assert args.repeat_runs == 3


def test_summarize_repeat_results_collects_latency_memory_token_and_compute_stats():
    summary = summarize_repeat_results(
        [
            {
                "total_ms": 100.0,
                "ttft_ms": 40.0,
                "llm_peak_memory_bytes": 1000,
                "token_metrics": {"llm_visual_token_reduction_ratio": 2.0},
                "compute_metrics": {
                    "siglip_encoder": {"keep_all_to_actual_total_macs_ratio": 3.0},
                    "mllm": {"kv_cache_reduction_ratio": 2.5},
                },
            },
            {
                "total_ms": 80.0,
                "ttft_ms": 30.0,
                "llm_peak_memory_bytes": 800,
                "token_metrics": {"llm_visual_token_reduction_ratio": 2.5},
                "compute_metrics": {
                    "siglip_encoder": {"keep_all_to_actual_total_macs_ratio": 4.0},
                    "mllm": {"kv_cache_reduction_ratio": 3.5},
                },
            },
        ]
    )

    assert summary["total_ms"]["median"] == 90.0
    assert summary["ttft_ms"]["min"] == 30.0
    assert summary["llm_peak_memory_bytes"]["max"] == 1000.0
    assert summary["token_metrics.llm_visual_token_reduction_ratio"]["mean"] == 2.25
    assert summary["compute_metrics.siglip_encoder.keep_all_to_actual_total_macs_ratio"]["median"] == 3.5
    assert summary["compute_metrics.mllm.kv_cache_reduction_ratio"]["median"] == 3.0


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


def test_processor_videos_argument_wraps_preloaded_frames_as_single_video():
    frames = [object(), object()]

    assert processor_videos_argument(frames, {"mode": "preloaded_resized_frames"}) == [frames]


def test_processor_videos_argument_keeps_path_or_url_as_single_input():
    path = "/tmp/video.mp4"

    assert processor_videos_argument(path, {"mode": "path_or_url"}) == path


def test_build_single_summary_extracts_report_ready_metrics_from_single_payload():
    payload = {
        "model_path": "local-nvila",
        "gazing_mode": "autogaze",
        "video": "video.mp4",
        "prompt": "What does the sign say? A. A B. B C. C D. D",
        "question": "What does the sign say?",
        "result": {
            "raw_output": "A",
            "generated_tokens": 4,
            "total_ms": 100.0,
            "ttft_ms": 30.0,
            "video_tiling_ms": 11.0,
            "autogaze_ms": 12.0,
            "autogaze_forward_ms": 10.0,
            "siglip_vision_ms": 20.0,
            "mm_projector_ms": 3.0,
            "llm_forward_ms": 50.0,
            "llm_peak_memory_bytes": 1024,
            "video_input_summary": {
                "source_frames": 240,
                "source_resolution": "3840x2160",
                "requested_video_frames": 8,
                "actual_video_frames": 8,
                "requested_thumbnail_frames": 4,
                "actual_thumbnail_frames": 4,
                "runner_resize_enabled": True,
                "processor_input_resolution": "1280x720",
            },
            "token_metrics": {
                "video_sampled_frames": 8,
                "thumbnail_sampled_frames": 4,
                "spatial_tiles_per_video": [2],
                "temporal_chunks_per_video": [4],
                "encoder_patches_per_frame_multiscale": 4,
                "encoder_patches_per_frame_by_scale": {"scale_a": 1, "scale_b": 3},
                "encoder_raw_patch_tokens": 80,
                "encoder_raw_tile_patch_tokens": 64,
                "encoder_autogaze_selected_tile_patch_tokens": 24,
                "autogaze_input_tile_frame_instances": 16,
                "autogaze_input_patch_tokens": 64,
                "autogaze_selected_patch_tokens": 24,
                "autogaze_removed_patch_tokens": 40,
                "autogaze_patch_reduction_ratio": 64 / 24,
                "encoder_raw_thumbnail_patch_tokens": 16,
                "encoder_autogaze_selected_thumbnail_patch_tokens": 16,
                "encoder_autogaze_selected_patch_tokens": 40,
                "encoder_token_reduction_ratio": 2.0,
                "llm_visual_token_reduction_ratio": 2.0,
                "llm_keep_all_visual_tokens_estimated": 80,
                "llm_actual_visual_tokens": 40,
            },
            "compute_metrics": {
                "siglip_encoder": {"keep_all_to_actual_total_macs_ratio": 4.0},
                "mllm": {
                    "prefill_context_reduction_ratio": 1.5,
                    "kv_cache_reduction_ratio": 2.5,
                },
            },
        },
        "repeat_summary": {
            "total_ms": {"median": 90.0},
            "ttft_ms": {"median": 25.0},
            "autogaze_ms": {"median": 12.0},
            "autogaze_forward_ms": {"median": 10.0},
            "siglip_vision_ms": {"median": 18.0},
            "llm_forward_ms": {"median": 45.0},
            "llm_peak_memory_bytes": {"median": 900.0},
        },
    }

    summary = build_single_summary(payload)

    assert summary["key_autogaze_effect"] == {
        "gazing_mode": "autogaze",
        "total_ms_median": 90.0,
        "ttft_ms_median": 25.0,
        "autogaze_total_ms_median": 12.0,
        "autogaze_forward_ms_median": 10.0,
        "siglip_vision_ms_median": 18.0,
        "llm_forward_ms_median": 45.0,
        "encoder_patch_tokens_before_keep_all": 80,
        "encoder_patch_tokens_after_actual": 40,
        "encoder_patch_tokens_removed": 40,
        "encoder_patch_reduction_ratio": 2.0,
        "encoder_patch_reduction_percent": 50.0,
        "llm_visual_tokens_before_keep_all_estimated": 80,
        "llm_visual_tokens_after_actual": 40,
        "llm_visual_tokens_removed_estimated": 40,
        "llm_visual_token_reduction_ratio": 2.0,
        "llm_visual_token_reduction_percent": 50.0,
        "siglip_total_macs_reduction_ratio": 4.0,
        "mllm_prefill_context_reduction_ratio": 1.5,
        "mllm_kv_cache_reduction_ratio": 2.5,
        "llm_peak_memory_bytes_median": 900.0,
    }
    assert summary["video_input_summary"] == {
        "source_frames": 240,
        "source_resolution": "3840x2160",
        "requested_video_frames": 8,
        "actual_video_frames": 8,
        "requested_thumbnail_frames": 4,
        "actual_thumbnail_frames": 4,
        "runner_resize_enabled": True,
        "processor_input_resolution": "1280x720",
    }
    assert summary["autogaze_token_summary"] == {
        "frame_basis": {
            "video_sampled_frames": 8,
            "thumbnail_sampled_frames": 4,
            "spatial_tiles_per_video": [2],
            "temporal_chunks_per_video": [4],
            "encoder_patches_per_frame_multiscale": 4,
        },
        "autogaze_input_breakdown": {
            "formula": "tile_frame_instances * multiscale_patch_positions_per_tile_frame",
            "expanded_formula": "16 tile-frame instances * 4 multiscale patch positions = 64",
            "video_sampled_frames": 8,
            "spatial_tiles_per_frame": 2,
            "temporal_chunks": 4,
            "tile_frame_instances": 16,
            "multiscale_patch_positions_per_tile_frame": 4,
            "patch_positions_by_scale": {"scale_a": 1, "scale_b": 3},
            "input_patch_tokens": 64,
            "unit_note": (
                "These are encoder patch positions before SigLIP/TokenShuffle/MLLM, not final LLM visual tokens."
            ),
            "why_it_can_be_large": (
                "A resized video can still be split into multiple spatial tiles. "
                "For example, 128 frames * 8 tiles/frame * 1060 multiscale patches = 1085440."
            ),
        },
        "autogaze_selection_patch_tokens": {
            "input_patch_tokens": 64,
            "selected_patch_tokens": 24,
            "removed_patch_tokens": 40,
            "reduction_ratio": 64 / 24,
            "reduction_percent": 62.5,
            "scope": (
                "Tiled-video encoder patch positions passed to AutoGaze before selection. "
                "Thumbnail patches are not included because thumbnails are keep-all in this runner."
            ),
        },
        "encoder_patch_tokens_before_siglip": {
            "raw_tile_patch_tokens": 64,
            "selected_tile_patch_tokens": 24,
            "removed_tile_patch_tokens": 40,
            "raw_thumbnail_patch_tokens": 16,
            "selected_thumbnail_patch_tokens": 16,
            "removed_thumbnail_patch_tokens": 0,
            "raw_total_patch_tokens": 80,
            "selected_total_patch_tokens": 40,
            "removed_total_patch_tokens": 40,
            "reduction_ratio": 2.0,
            "reduction_percent": 50.0,
            "selected_token_definition": (
                "Non-padded AutoGaze-selected encoder patch positions before TokenShuffle, "
                "SigLIP, and the MLLM projector. Thumbnails are keep-all in this runner."
            ),
        },
        "llm_visual_tokens_after_token_shuffle": {
            "keep_all_visual_tokens_estimated": 80,
            "actual_visual_tokens": 40,
            "removed_visual_tokens_estimated": 40,
            "reduction_ratio": 2.0,
            "reduction_percent": 50.0,
            "token_definition": (
                "Visual placeholder tokens consumed by the LLM after TokenShuffle/projector input "
                "packing; this is the token count that drives LLM prefill/KV cache estimates."
            ),
        },
    }
    assert summary["answer"] == "A"
    assert summary["prompt"] == "What does the sign say? A. A B. B C. C D. D"
    assert summary["question"] == "What does the sign say?"
    assert summary["latency_ms"]["total_median"] == 90.0
    assert summary["latency_ms"]["ttft_median"] == 25.0
    assert summary["latency_ms"]["autogaze_total_median"] == 12.0
    assert summary["latency_ms"]["autogaze_forward_median"] == 10.0
    assert summary["memory_bytes"]["llm_peak_median"] == 900.0
    assert summary["tokens"]["encoder_token_reduction_ratio"] == 2.0
    assert summary["tokens"]["llm_actual_visual_tokens"] == 40
    assert summary["compute"]["siglip_total_macs_reduction_ratio"] == 4.0
    assert summary["compute"]["mllm_kv_cache_reduction_ratio"] == 2.5


def test_build_autogaze_token_summary_separates_encoder_patches_and_llm_tokens():
    token_metrics = {
        "video_sampled_frames": 32,
        "thumbnail_sampled_frames": 16,
        "spatial_tiles_per_video": [8],
        "temporal_chunks_per_video": [2],
        "encoder_patches_per_frame_multiscale": 1060,
        "encoder_patches_per_frame_by_scale": {"56": 16, "112": 64, "196": 196, "392": 784},
        "autogaze_input_tile_frame_instances": 256,
        "encoder_raw_tile_patch_tokens": 271360,
        "encoder_autogaze_selected_tile_patch_tokens": 27136,
        "autogaze_input_patch_tokens": 271360,
        "autogaze_selected_patch_tokens": 27136,
        "autogaze_removed_patch_tokens": 244224,
        "autogaze_patch_reduction_ratio": 10.0,
        "encoder_raw_thumbnail_patch_tokens": 16960,
        "encoder_autogaze_selected_thumbnail_patch_tokens": 16960,
        "encoder_raw_patch_tokens": 288320,
        "encoder_autogaze_selected_patch_tokens": 44096,
        "encoder_token_reduction_ratio": 288320 / 44096,
        "llm_keep_all_visual_tokens_estimated": 32096,
        "llm_actual_visual_tokens": 4896,
        "llm_visual_token_reduction_ratio": 32096 / 4896,
    }

    summary = build_autogaze_token_summary(token_metrics)

    assert summary["frame_basis"]["video_sampled_frames"] == 32
    assert summary["autogaze_selection_patch_tokens"]["input_patch_tokens"] == 271360
    assert summary["autogaze_selection_patch_tokens"]["selected_patch_tokens"] == 27136
    assert summary["autogaze_selection_patch_tokens"]["removed_patch_tokens"] == 244224
    assert summary["autogaze_selection_patch_tokens"]["reduction_ratio"] == 10.0
    assert summary["autogaze_input_breakdown"] == {
        "formula": "tile_frame_instances * multiscale_patch_positions_per_tile_frame",
        "expanded_formula": "256 tile-frame instances * 1060 multiscale patch positions = 271360",
        "video_sampled_frames": 32,
        "spatial_tiles_per_frame": 8,
        "temporal_chunks": 2,
        "tile_frame_instances": 256,
        "multiscale_patch_positions_per_tile_frame": 1060,
        "patch_positions_by_scale": {"56": 16, "112": 64, "196": 196, "392": 784},
        "input_patch_tokens": 271360,
        "unit_note": (
            "These are encoder patch positions before SigLIP/TokenShuffle/MLLM, not final LLM visual tokens."
        ),
        "why_it_can_be_large": (
            "A resized video can still be split into multiple spatial tiles. "
            "For example, 128 frames * 8 tiles/frame * 1060 multiscale patches = 1085440."
        ),
    }
    assert summary["encoder_patch_tokens_before_siglip"]["raw_total_patch_tokens"] == 288320
    assert summary["encoder_patch_tokens_before_siglip"]["selected_total_patch_tokens"] == 44096
    assert summary["encoder_patch_tokens_before_siglip"]["removed_total_patch_tokens"] == 244224
    assert summary["encoder_patch_tokens_before_siglip"]["reduction_percent"] == 84.705882
    assert summary["llm_visual_tokens_after_token_shuffle"]["keep_all_visual_tokens_estimated"] == 32096
    assert summary["llm_visual_tokens_after_token_shuffle"]["actual_visual_tokens"] == 4896
    assert summary["llm_visual_tokens_after_token_shuffle"]["removed_visual_tokens_estimated"] == 27200


def test_summarize_token_budget_rows_reports_benchmark_medians():
    rows = [
        {
            "status": "ok",
            "token_metrics": {
                "video_sampled_frames": 32,
                "encoder_raw_patch_tokens": 100,
                "encoder_autogaze_selected_patch_tokens": 25,
                "autogaze_input_tile_frame_instances": 20,
                "autogaze_input_patch_tokens": 80,
                "autogaze_selected_patch_tokens": 20,
                "encoder_token_reduction_ratio": 4.0,
                "llm_keep_all_visual_tokens_estimated": 40,
                "llm_actual_visual_tokens": 10,
                "llm_visual_token_reduction_ratio": 4.0,
            },
        },
        {
            "status": "ok",
            "token_metrics": {
                "video_sampled_frames": 32,
                "encoder_raw_patch_tokens": 200,
                "encoder_autogaze_selected_patch_tokens": 100,
                "autogaze_input_tile_frame_instances": 40,
                "autogaze_input_patch_tokens": 160,
                "autogaze_selected_patch_tokens": 80,
                "encoder_token_reduction_ratio": 2.0,
                "llm_keep_all_visual_tokens_estimated": 80,
                "llm_actual_visual_tokens": 40,
                "llm_visual_token_reduction_ratio": 2.0,
            },
        },
        {"status": "failed", "error": "oom"},
    ]

    summary = summarize_token_budget_rows(rows)

    assert summary["rows_with_token_metrics"] == 2
    assert summary["median"]["video_sampled_frames"] == 32
    assert summary["median"]["encoder_raw_patch_tokens"] == 150
    assert summary["median"]["encoder_autogaze_selected_patch_tokens"] == 62.5
    assert summary["median"]["encoder_removed_patch_tokens"] == 87.5
    assert summary["median"]["autogaze_input_tile_frame_instances"] == 30
    assert summary["median"]["autogaze_input_patch_tokens"] == 120
    assert summary["median"]["autogaze_selected_patch_tokens"] == 50
    assert summary["median"]["autogaze_removed_patch_tokens"] == 70
    assert summary["median"]["encoder_token_reduction_ratio"] == 3.0
    assert summary["median"]["llm_visual_token_reduction_ratio"] == 3.0


def test_build_video_input_summary_reports_source_sample_and_resize_context():
    args = make_args(
        num_video_frames=128,
        num_video_frames_thumbnail=64,
        video_resize_shortest_edge=720,
        video_resize_longest_edge=None,
        video_resize_width=None,
        video_resize_height=None,
    )
    source_metadata = {
        "width": 3840,
        "height": 2160,
        "frames": 240,
        "fps": 30.0,
        "duration_seconds": 8.0,
        "codec": "h264",
    }
    token_metrics = {
        "video_sampled_frames": 128,
        "thumbnail_sampled_frames": 64,
        "spatial_tiles_per_video": [8],
        "temporal_chunks_per_video": 8,
    }

    summary = build_video_input_summary(
        args=args,
        resolved_video="/videos/sample.mp4",
        source_metadata=source_metadata,
        video_input_info={
            "mode": "preloaded_resized_frames",
            "resize": {
                "enabled": True,
                "shortest_edge": 720,
                "longest_edge": None,
                "width": None,
                "height": None,
                "effective": {"width": 1280, "height": 720, "mode": "shortest_edge"},
            },
            "frames_loaded": 128,
        },
        token_metrics=token_metrics,
    )

    assert summary == {
        "resolved_video": "/videos/sample.mp4",
        "source_frames": 240,
        "source_resolution": "3840x2160",
        "source_width": 3840,
        "source_height": 2160,
        "source_fps": 30.0,
        "source_duration_seconds": 8.0,
        "source_codec": "h264",
        "requested_video_frames": 128,
        "actual_video_frames": 128,
        "requested_thumbnail_frames": 64,
        "actual_thumbnail_frames": 64,
        "sampled_frame_start": 0,
        "sampled_frame_end": 239,
        "runner_resize_enabled": True,
        "runner_resize_request": {
            "shortest_edge": 720,
            "longest_edge": None,
            "width": None,
            "height": None,
        },
        "processor_input_width": 1280,
        "processor_input_height": 720,
        "processor_input_resolution": "1280x720",
        "processor_video_input_mode": "preloaded_resized_frames",
        "frames_loaded_for_processor": 128,
        "spatial_tiles_per_video": [8],
        "temporal_chunks_per_video": 8,
    }


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


def test_parse_args_accepts_stream_profile_mode_and_chunk_options():
    args = parse_args(
        [
            "--mode",
            "stream-profile",
            "--stream-chunk-frames",
            "16",
            "--stream-profile-json",
            "out/stream.json",
        ]
    )

    assert args.mode == "stream-profile"
    assert args.stream_chunk_frames == 16
    assert args.stream_profile_json == "out/stream.json"


def test_parse_args_accepts_seek_stream_decode_strategy():
    args = parse_args(["--mode", "stream-profile", "--stream-decode-strategy", "seek"])

    assert args.stream_decode_strategy == "seek"


def test_parse_args_accepts_optional_stream_siglip_stage():
    args = parse_args(
        [
            "--mode",
            "stream-profile",
            "--stream-run-siglip",
            "--stream-siglip-mode",
            "gazed",
            "--stream-siglip-model",
            "google/siglip2-base-patch16-224",
            "--stream-siglip-max-embed-batch-size",
            "4",
        ]
    )

    assert args.stream_run_siglip is True
    assert args.stream_siglip_mode == "gazed"
    assert args.stream_siglip_model == "google/siglip2-base-patch16-224"
    assert args.stream_siglip_max_embed_batch_size == 4


def test_frame_index_pts_roundtrip_uses_stream_time_base_and_start_time():
    pts_per_frame = stream_pts_per_frame(average_rate=Fraction(30, 1), time_base=Fraction(1, 15360))

    assert pts_per_frame == Fraction(512, 1)
    assert frame_index_to_pts(599, pts_per_frame=pts_per_frame, start_time=1000) == 307688
    assert pts_to_frame_index(307688, pts_per_frame=pts_per_frame, start_time=1000) == 599


def test_build_seek_decode_groups_uses_previous_keyframe_and_groups_targets():
    groups = build_seek_decode_groups(
        target_indices=[0, 1, 11, 12, 20, 24],
        keyframe_indices=[0, 12, 24],
    )

    assert groups == [
        {"seek_frame_index": 0, "target_indices": [0, 1, 11]},
        {"seek_frame_index": 12, "target_indices": [12, 20]},
        {"seek_frame_index": 24, "target_indices": [24]},
    ]


def test_stream_profile_plan_describes_chunked_hlvid_like_work():
    plan = estimate_stream_profile_plan(
        width=3840,
        height=2160,
        source_frames=9000,
        num_video_frames=128,
        num_video_frames_thumbnail=64,
        max_tiles_video=48,
        chunk_frames=16,
        max_batch_size_autogaze=4,
        scales=[56, 112, 196, 392],
        patch_size=14,
    )

    assert plan["sampling"]["requested_frames"] == 128
    assert plan["sampling"]["thumbnail_frames"] == 64
    assert plan["tiling"]["spatial_tiles"] == 45
    assert plan["chunking"]["temporal_chunks"] == 8
    assert plan["chunking"]["tile_sequences"] == 360
    assert plan["tokens"]["encoder_raw_tile_patch_tokens"] == 128 * 45 * 1060
    assert plan["tokens"]["encoder_raw_thumbnail_patch_tokens"] == 64 * 1060
    assert plan["memory"]["streaming_raw_frame_buffer_bytes"] == 16 * 3840 * 2160 * 3
    assert plan["memory"]["autogaze_batch_tile_sequences"] == 4
    assert plan["memory"]["streaming_autogaze_tile_tensor_bytes_per_batch"] == 16 * 4 * 3 * 392 * 392 * 4
    assert plan["streaming_boundary"]["pre_llm_stages_can_stream"] is True
    assert plan["streaming_boundary"]["llm_generation_requires_collected_visual_tokens"] is True


def test_build_stream_profile_token_metrics_compares_autogaze_and_keep_all_tokens():
    plan = estimate_stream_profile_plan(
        width=1280,
        height=720,
        source_frames=300,
        num_video_frames=32,
        num_video_frames_thumbnail=16,
        max_tiles_video=1,
        chunk_frames=16,
        scales=[56, 112],
        patch_size=14,
    )
    tile_summary = {
        "raw_patch_budget": 32 * 1 * 80,
        "selected_non_padded_patches": 640,
        "padded_gazing_positions": 0,
        "total_gaze_slots": 640,
    }

    metrics = build_stream_profile_token_metrics(plan, tile_summary)

    assert metrics["video_sampled_frames"] == 32
    assert metrics["thumbnail_sampled_frames"] == 16
    assert metrics["encoder_raw_tile_patch_tokens"] == 2560
    assert metrics["encoder_autogaze_selected_tile_patch_tokens"] == 640
    assert metrics["autogaze_input_tile_frame_instances"] == 32
    assert metrics["encoder_tile_token_reduction_ratio"] == 4.0
    assert metrics["encoder_raw_thumbnail_patch_tokens"] == 1280
    assert metrics["encoder_autogaze_selected_thumbnail_patch_tokens"] == 1280
    assert metrics["encoder_token_reduction_ratio"] == 2.0


def test_build_stream_profile_compute_metrics_reports_siglip_estimated_costs():
    plan = estimate_stream_profile_plan(
        width=1280,
        height=720,
        source_frames=300,
        num_video_frames=32,
        num_video_frames_thumbnail=16,
        max_tiles_video=1,
        chunk_frames=16,
        scales=[56, 112],
        patch_size=14,
    )
    tile_summary = {
        "raw_patch_budget": 32 * 80,
        "selected_non_padded_patches": 640,
        "padded_gazing_positions": 0,
        "total_gaze_slots": 640,
        "siglip_gazed_sequence_slots_sum": 640,
        "siglip_gazed_sequence_slots_squared_sum": 2 * 320 * 320,
    }
    token_metrics = build_stream_profile_token_metrics(plan, tile_summary)

    metrics = build_stream_profile_compute_metrics(
        plan,
        tile_summary,
        token_metrics,
        siglip_info={
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
        },
        dtype_bytes=2,
    )

    assert metrics["siglip_encoder"]["actual"]["sequence_tokens"] == 640 + 16 * 80
    assert metrics["siglip_encoder"]["keep_all"]["sequence_tokens"] == 32 * 80 + 16 * 80
    assert metrics["siglip_encoder"]["keep_all_to_actual_attention_macs_ratio"] > 1
    assert metrics["mllm"]["full_llm_not_run_in_stream_profile"] is True


def test_build_keep_all_gazing_info_selects_every_patch_for_siglip():
    info = build_keep_all_gazing_info(
        batch_size=2,
        frames=3,
        patches_per_frame_value=4,
        device=torch.device("cpu"),
    )

    assert info["gazing_pos"].shape == (2, 12)
    assert info["if_padded_gazing"].shape == (2, 12)
    assert info["num_gazing_each_frame"].tolist() == [4, 4, 4]
    assert info["gazing_pos"][0].tolist() == list(range(12))
    assert info["if_padded_gazing"].any().item() is False


def test_summarize_stream_chunks_sums_siglip_time_but_peaks_siglip_memory():
    summary = summarize_stream_chunks(
        [
            {
                "tile_sequences": 1,
                "raw_patch_budget": 10,
                "selected_non_padded_patches": 5,
                "padded_gazing_positions": 0,
                "total_gaze_slots": 5,
                "siglip_gazed_forward_ms": 10.0,
                "siglip_gazed_hidden_bytes_peak": 100,
                "siglip_gazed_sequence_slots_sum": 5,
                "siglip_gazed_sequence_slots_squared_sum": 25,
            },
            {
                "tile_sequences": 1,
                "raw_patch_budget": 20,
                "selected_non_padded_patches": 10,
                "padded_gazing_positions": 0,
                "total_gaze_slots": 10,
                "siglip_gazed_forward_ms": 12.0,
                "siglip_gazed_hidden_bytes_peak": 80,
                "siglip_gazed_sequence_slots_sum": 10,
                "siglip_gazed_sequence_slots_squared_sum": 100,
            },
        ]
    )

    assert summary["siglip_gazed_forward_ms"] == 22.0
    assert summary["siglip_gazed_hidden_bytes_peak"] == 100
    assert summary["siglip_gazed_sequence_slots_sum"] == 15
    assert summary["siglip_gazed_sequence_slots_squared_sum"] == 125


def test_repeat_last_stream_samples_after_eof_fills_missing_decoded_tail_samples():
    last_frame = Image.new("RGB", (4, 4), "white")
    current_frames = [last_frame.copy()]
    thumbnails = []

    summary = repeat_last_stream_samples_after_eof(
        current_frames=current_frames,
        thumbnails=thumbnails,
        last_selected_frame=last_frame,
        missing_sampled_frames=2,
        missing_thumbnail_frames=1,
        tile_size=2,
    )

    assert summary == {
        "padded_sampled_frames_after_eof": 2,
        "padded_thumbnail_frames_after_eof": 1,
    }
    assert len(current_frames) == 3
    assert len(thumbnails) == 1
    assert thumbnails[0].size == (2, 2)


def test_autogaze_processor_size_kwargs_match_largest_target_scale():
    kwargs = autogaze_processor_size_kwargs([56, 112, 196, 392])

    assert kwargs == {
        "size": {"shortest_edge": 392},
        "crop_size": {"height": 392, "width": 392},
    }


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
    assert metrics["autogaze_input_tile_frame_instances"] == 4
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


def test_estimate_siglip_encoder_compute_splits_attention_and_mlp_costs():
    metrics = estimate_siglip_encoder_compute(
        sequence_lengths=[4, 2],
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        dtype_bytes=2,
    )

    assert metrics["sequence_count"] == 2
    assert metrics["sequence_tokens"] == 6
    assert metrics["dense_attention_pairs"] == 20
    assert metrics["attention_projection_macs_estimated"] == 3072
    assert metrics["attention_quadratic_macs_estimated"] == 640
    assert metrics["mlp_macs_estimated"] == 3072
    assert metrics["total_macs_estimated"] == 6784
    assert metrics["hidden_state_bytes_estimated"] == 96
    assert metrics["attention_score_bytes_estimated"] == 80
    assert metrics["mlp_intermediate_bytes_estimated"] == 192


def test_build_autogaze_effect_metrics_reports_siglip_and_mllm_reductions():
    payload = {
        "input_ids": torch.tensor([[11, 32000, 12, 32000, 13]]),
        "pixel_values_videos_tiles": [torch.zeros(2, 2, 3, 4, 4)],
        "pixel_values_videos_thumbnails": [torch.zeros(1, 1, 3, 4, 4)],
        "num_spatial_tiles_each_video": [2],
        "gazing_info": {
            "if_padded_gazing_tiles": [torch.tensor([[False, True, False], [True, False, False]])],
            "if_padded_gazing_thumbnails": [torch.tensor([[False, True]])],
        },
    }
    token_metrics = compute_visual_token_metrics(
        payload,
        video_token_id=32000,
        patches_per_frame_value=4,
        patches_per_frame_by_scale={"14": 1, "28": 4},
        token_shuffle=2,
    )

    metrics = build_autogaze_effect_metrics(
        payload,
        model=DummyFullModel(),
        token_metrics=token_metrics,
        input_token_count=5,
        dtype_bytes=2,
        patches_per_frame_value=4,
        token_shuffle=2,
    )

    assert metrics["siglip_encoder"]["actual"]["sequence_tokens"] == 8
    assert metrics["siglip_encoder"]["keep_all"]["sequence_tokens"] == 20
    assert metrics["siglip_encoder"]["keep_all_to_actual_total_macs_ratio"] > 1
    assert metrics["mllm"]["actual_prefill_context_tokens"] == 5
    assert metrics["mllm"]["actual_visual_tokens"] == 2
    assert metrics["mllm"]["text_tokens_estimated"] == 3
    assert metrics["mllm"]["keep_all_prefill_context_tokens_estimated"] == 13
    assert metrics["mllm"]["prefill_context_reduction_ratio"] == 2.6
    assert metrics["mllm"]["actual_kv_cache_bytes_after_prefill_estimated"] == 320


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
