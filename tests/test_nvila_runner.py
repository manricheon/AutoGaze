import argparse
from fractions import Fraction
from pathlib import Path

import pytest
import torch
from PIL import Image

from repro.nvila_runner import (
    DEFAULT_BASELINE_MODEL,
    StageProfiler,
    apply_processor_autogaze_generate_only,
    apply_resize_to_dimensions,
    autogaze_processor_size_kwargs,
    build_autogaze_effect_metrics,
    build_h100_decision_row,
    build_latency_accounting,
    build_processing_budget_summary,
    build_run_identity,
    build_seek_decode_groups,
    build_keep_all_gazing_info,
    build_parser,
    build_autogaze_token_summary,
    build_patch_space_metadata,
    build_video_input_summary,
    build_stream_profile_compute_metrics,
    build_stream_profile_token_metrics,
    compute_visual_token_metrics,
    estimate_siglip_encoder_compute,
    effective_gazing_ratio_tile,
    effective_stream_gazing_ratio,
    estimate_h100_preflight_config,
    h100_risk_band,
    estimate_nvila_preflight,
    estimate_stream_profile_plan,
    model_load_kwargs,
    extract_gaze_metrics,
    MODEL_FAMILY_HD_AUTOGAZE,
    MODEL_FAMILY_VIDEO_BASELINE,
    PAPER_PRESET_BASELINE,
    PAPER_PRESET_HD,
    model_patches_per_frame,
    parse_float_sequence,
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
        "model_path": "nvidia/NVILA-8B-HD-Video",
        "model_family": "auto",
        "paper_preset": None,
        "dtype": None,
        "device_map": "auto",
        "video_resize_longest_edge": None,
        "autogaze_model": "nvidia/AutoGaze",
        "gazing_mode": "autogaze",
        "gazing_ratio_tile": None,
        "autogaze_target_scales": None,
        "autogaze_target_patch_size": None,
        "task_loss_requirement_tile": 0.6,
        "max_batch_size_autogaze": 16,
        "max_batch_size_siglip": 32,
        "hlvid_repo": "bfshi/HLVid",
        "hlvid_video_root": "data/hlvid/videos",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_build_latency_accounting_separates_additive_total_from_nested_breakdown():
    accounting = build_latency_accounting(
        {
            "total_ms": 100.0,
            "video_preprocess_ms": 30.0,
            "video_preprocess_without_autogaze_ms": 23.0,
            "autogaze_total_ms": 7.0,
            "generate_ms": 70.0,
            "ttft_ms": 12.0,
            "video_decode_ms": 5.0,
            "video_tiling_ms": 20.0,
            "gazing_info_total_ms": 7.0,
            "autogaze_model_forward_ms": 5.0,
            "generation_decode_after_ttft_estimated_ms": 58.0,
        }
    )

    assert accounting["additive_total_ms"] == {
        "formula": "total_ms = video_preprocess_without_autogaze_ms + autogaze_total_ms + generate_ms",
        "total_ms": 100.0,
        "video_preprocess_without_autogaze_ms": 23.0,
        "autogaze_total_ms": 7.0,
        "generate_ms": 70.0,
        "recomputed_total_ms": 100.0,
        "delta_ms": 0.0,
        "ttft_ms_excluded_from_total": 12.0,
    }
    assert accounting["legacy_inclusive_total_ms"] == {
        "formula": "total_ms = video_preprocess_ms + generate_ms",
        "total_ms": 100.0,
        "video_preprocess_ms": 30.0,
        "generate_ms": 70.0,
        "recomputed_total_ms": 100.0,
        "delta_ms": 0.0,
        "ttft_ms_excluded_from_total": 12.0,
    }
    assert accounting["nested_preprocess_breakdown_ms"]["video_decode_ms"] == {
        "value": 5.0,
        "included_in": "video_preprocess_without_autogaze_ms",
        "add_to_total_ms": False,
    }
    assert accounting["nested_preprocess_breakdown_ms"]["autogaze_model_forward_ms"] == {
        "value": 5.0,
        "included_in": "autogaze_total_ms",
        "add_to_total_ms": False,
    }
    hierarchy = accounting["hierarchy"]
    assert (
        hierarchy["total_formula"]
        == "total_ms = video_preprocess_without_autogaze_ms + autogaze_total_ms + generate_ms"
    )
    assert hierarchy["quick_answers"]["is_autogaze_in_generate_ms"] is False
    assert hierarchy["quick_answers"]["where_is_autogaze_ms_included"] == "autogaze_total_ms"
    assert hierarchy["quick_answers"]["legacy_inclusive_preprocess_field"] == "video_preprocess_ms"
    assert hierarchy["quick_answers"]["is_video_decode_in_preprocess_ms"] is True
    assert hierarchy["nodes"]["video_decode_ms"]["included_in"] == "video_preprocess_without_autogaze_ms"
    assert hierarchy["nodes"]["autogaze_total_ms"]["included_in"] == "total_ms"
    assert hierarchy["nodes"]["generate_ms"]["includes"] == [
        "vision_encoder_ms",
        "llm_forward_ms",
        "generation_decode_after_ttft_estimated_ms",
    ]
    assert "video_preprocess_without_autogaze_ms + autogaze_total_ms + generate_ms" in hierarchy["ascii_tree"]
    assert "video_decode_ms" in accounting["do_not_sum_with_total_ms"]


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


class DummyProcessor:
    def __init__(self, target_scales=None, target_patch_size=None):
        if target_scales is not None:
            self.target_scales = target_scales
        if target_patch_size is not None:
            self.target_patch_size = target_patch_size


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


def test_processor_kwargs_can_override_gazing_ratio_tile_for_timing_audit():
    kwargs = processor_kwargs(make_args(gazing_ratio_tile="0.75", task_loss_requirement_tile=0.7))

    assert kwargs["gazing_ratio_tile"] == 0.75
    assert kwargs["task_loss_requirement_tile"] == 0.7


def test_parse_args_accepts_autogaze_generate_only():
    args = parse_args(["--autogaze-generate-only"])

    assert args.autogaze_generate_only is True


def test_apply_processor_autogaze_generate_only_injects_forward_kwarg():
    class FakeAutoGaze(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen_kwargs = None

        def forward(self, *args, **kwargs):
            self.seen_kwargs = kwargs
            return {"ok": True}

    processor = argparse.Namespace(_autogaze_model=FakeAutoGaze())

    applied = apply_processor_autogaze_generate_only(processor, enabled=True)
    result = processor._autogaze_model({"video": torch.zeros(1)}, gazing_ratio=0.5)

    assert applied is True
    assert result == {"ok": True}
    assert processor.autogaze_generate_only is True
    assert processor._autogaze_model.seen_kwargs["generate_only"] is True
    assert processor._autogaze_model.seen_kwargs["gazing_ratio"] == 0.5


def test_processor_kwargs_accepts_sequence_gazing_ratio_tile():
    kwargs = processor_kwargs(make_args(gazing_ratio_tile="0.2,0.06,0.06"))

    assert kwargs["gazing_ratio_tile"] == [0.2, 0.06, 0.06]


def test_effective_gazing_ratio_tile_defaults_to_nvila_processor_policy():
    args = make_args()

    assert effective_gazing_ratio_tile(args) == [0.2] + [0.06] * 15


def test_parse_args_applies_paper_baseline_preset_defaults():
    args = parse_args(["--paper-preset", PAPER_PRESET_BASELINE])

    assert args.paper_preset == PAPER_PRESET_BASELINE
    assert args.model_family == MODEL_FAMILY_VIDEO_BASELINE
    assert args.model_path == DEFAULT_BASELINE_MODEL
    assert args.num_video_frames == 256
    assert args.num_video_frames_thumbnail == 0
    assert args.max_tiles_video == 1
    assert args.video_resize_longest_edge == 448
    assert args.gazing_mode == "keep-all"


def test_parse_args_applies_paper_hd_preset_defaults():
    args = parse_args(["--paper-preset", PAPER_PRESET_HD])

    assert args.paper_preset == PAPER_PRESET_HD
    assert args.model_family == MODEL_FAMILY_HD_AUTOGAZE
    assert args.model_path == "nvidia/NVILA-8B-HD-Video"
    assert args.num_video_frames == 1024
    assert args.num_video_frames_thumbnail == 128
    assert args.max_tiles_video == 48
    assert args.video_resize_longest_edge == 3584
    assert args.gazing_mode == "autogaze"


def test_parse_args_rejects_thumbnail_zero_for_hd_single_generate_path():
    with pytest.raises(ValueError, match="num-video-frames-thumbnail"):
        parse_args(
            [
                "--mode",
                "single",
                "--model-family",
                MODEL_FAMILY_HD_AUTOGAZE,
                "--num-video-frames-thumbnail",
                "0",
            ]
        )


def test_parse_args_rejects_thumbnail_zero_for_hd_hlvid_generate_path():
    with pytest.raises(ValueError, match="num-video-frames-thumbnail"):
        parse_args(
            [
                "--mode",
                "hlvid",
                "--model-family",
                MODEL_FAMILY_HD_AUTOGAZE,
                "--num-video-frames-thumbnail",
                "0",
            ]
        )


def test_parse_args_allows_thumbnail_zero_for_stream_profile():
    args = parse_args(["--mode", "stream-profile", "--num-video-frames-thumbnail", "0"])

    assert args.mode == "stream-profile"
    assert args.num_video_frames_thumbnail == 0


def test_parse_args_allows_thumbnail_zero_for_video_baseline_family():
    args = parse_args(
        [
            "--mode",
            "hlvid",
            "--model-family",
            MODEL_FAMILY_VIDEO_BASELINE,
            "--num-video-frames-thumbnail",
            "0",
        ]
    )

    assert args.model_family == MODEL_FAMILY_VIDEO_BASELINE
    assert args.num_video_frames_thumbnail == 0


def test_cli_values_override_paper_preset_defaults():
    args = parse_args(
        [
            "--paper-preset",
            PAPER_PRESET_BASELINE,
            "--num-video-frames",
            "128",
            "--model-path",
            "/models/local-nvila-video",
        ]
    )

    assert args.num_video_frames == 128
    assert args.model_path == "/models/local-nvila-video"
    assert args.model_family == MODEL_FAMILY_VIDEO_BASELINE


def test_baseline_processor_kwargs_omit_autogaze_specific_fields():
    args = make_args(
        model_family=MODEL_FAMILY_VIDEO_BASELINE,
        model_path=DEFAULT_BASELINE_MODEL,
        num_video_frames=256,
        num_video_frames_thumbnail=0,
        gazing_mode="keep-all",
    )

    kwargs = processor_kwargs(args)

    assert kwargs == {
        "num_video_frames": 256,
        "trust_remote_code": True,
    }
    assert "autogaze_model_id" not in kwargs
    assert "gazing_ratio_tile" not in kwargs
    assert "max_batch_size_autogaze" not in kwargs


def test_baseline_model_load_kwargs_omit_hd_siglip_batch_kwarg():
    args = make_args(model_family=MODEL_FAMILY_VIDEO_BASELINE, device_map="auto")

    kwargs = model_load_kwargs(args)

    assert kwargs == {"trust_remote_code": True, "device_map": "auto"}
    assert "max_batch_size_siglip" not in kwargs


def test_model_load_kwargs_uses_requested_torch_dtype():
    args = make_args(dtype="float16", device_map="auto")

    kwargs = model_load_kwargs(args)

    assert kwargs["torch_dtype"] == torch.float16


def test_run_identity_marks_paper_baseline_as_not_applicable():
    args = parse_args(["--paper-preset", PAPER_PRESET_BASELINE])

    identity = build_run_identity(args)

    assert identity["model_family"] == MODEL_FAMILY_VIDEO_BASELINE
    assert identity["paper_preset"] == PAPER_PRESET_BASELINE
    assert identity["paper_reference_score"] == 42.5
    assert identity["is_paper_baseline_candidate"] is True
    assert identity["autogaze_applicability"] == "not_applicable"
    assert identity["adapters"]["token_selector"]["name"] == "not_applicable"
    assert identity["adapters"]["vision_encoder"]["name"] == "nvila_video_baseline_vision_metadata"
    assert identity["adapters"]["mllm"]["name"] == "nvila_video_baseline"
    assert identity["components"]["token_selector"] == {
        "adapter": "none",
        "name": "not_applicable",
        "path": None,
        "applicability": "not_applicable",
    }
    assert identity["components"]["vision_encoder"] == {
        "adapter": "nvila-video-vision",
        "name": "nvila-8b-video-vision",
        "path": "auto",
    }
    assert identity["components"]["mllm"] == {
        "adapter": "nvila-video",
        "name": DEFAULT_BASELINE_MODEL,
        "path": DEFAULT_BASELINE_MODEL,
    }


def test_run_identity_separates_hd_keep_all_ablation_from_paper_baseline():
    args = parse_args(
        [
            "--model-family",
            MODEL_FAMILY_HD_AUTOGAZE,
            "--gazing-mode",
            "keep-all",
        ]
    )

    identity = build_run_identity(args)

    assert identity["model_family"] == MODEL_FAMILY_HD_AUTOGAZE
    assert identity["paper_reference_score"] is None
    assert identity["is_paper_baseline_candidate"] is False
    assert identity["autogaze_applicability"] == "hd_keep_all_ablation"
    assert identity["adapters"]["token_selector"]["name"] == "keep_all"
    assert identity["adapters"]["vision_encoder"]["name"] == "nvila_hd_siglip"
    assert identity["adapters"]["mllm"]["name"] == "nvila_hd"
    assert identity["components"]["token_selector"]["adapter"] == "keep-all"
    assert identity["components"]["vision_encoder"]["adapter"] == "nvila-hd-siglip"
    assert identity["components"]["mllm"]["adapter"] == "nvila-hd"


def test_hd_paper_preset_with_keep_all_does_not_claim_hd_autogaze_reference():
    args = parse_args(
        [
            "--paper-preset",
            PAPER_PRESET_HD,
            "--token-selector-adapter",
            "keep-all",
        ]
    )

    identity = build_run_identity(args)

    assert identity["paper_preset"] == PAPER_PRESET_HD
    assert identity["paper_reference_score"] is None
    assert identity["autogaze_applicability"] == "hd_keep_all_ablation"


def test_pipeline_preset_alias_and_component_cli_overrides_model_paths():
    args = parse_args(
        [
            "--pipeline-preset",
            PAPER_PRESET_BASELINE,
            "--mllm-path",
            "weight/NVILA-8B-Video",
            "--mllm-name",
            "local NVILA-8B-Video",
            "--token-selector-adapter",
            "none",
            "--token-selector-name",
            "not_applicable",
            "--vision-encoder-adapter",
            "nvila-video-vision",
            "--vision-encoder-name",
            "NVILA-8B-Video vision",
        ]
    )

    assert args.paper_preset == PAPER_PRESET_BASELINE
    assert args.pipeline_preset == PAPER_PRESET_BASELINE
    assert args.model_path == "weight/NVILA-8B-Video"
    assert args.mllm_path == "weight/NVILA-8B-Video"
    assert args.mllm_name == "local NVILA-8B-Video"
    assert args.token_selector_adapter == "none"
    assert args.token_selector_name == "not_applicable"
    assert args.vision_encoder_adapter == "nvila-video-vision"
    assert args.vision_encoder_name == "NVILA-8B-Video vision"


def test_run_identity_records_component_level_names_and_paths():
    args = parse_args(
        [
            "--model-family",
            MODEL_FAMILY_VIDEO_BASELINE,
            "--mllm-path",
            "weight/NVILA-8B-Video",
            "--mllm-adapter",
            "nvila-video",
            "--mllm-name",
            "local NVILA baseline",
            "--token-selector-adapter",
            "none",
            "--token-selector-name",
            "paper baseline none",
            "--vision-encoder-adapter",
            "nvila-video-vision",
            "--vision-encoder-name",
            "baseline vision metadata",
            "--vision-encoder-path",
            "auto",
        ]
    )

    identity = build_run_identity(args)

    assert identity["components"] == {
        "token_selector": {
            "adapter": "none",
            "name": "paper baseline none",
            "path": None,
            "applicability": "not_applicable",
        },
        "vision_encoder": {
            "adapter": "nvila-video-vision",
            "name": "baseline vision metadata",
            "path": "auto",
        },
        "mllm": {
            "adapter": "nvila-video",
            "name": "local NVILA baseline",
            "path": "weight/NVILA-8B-Video",
        },
    }


def test_component_keep_all_adapter_disables_autogaze_selection():
    args = parse_args(["--token-selector-adapter", "keep-all"])

    kwargs = processor_kwargs(args)
    identity = build_run_identity(args)

    assert args.gazing_mode == "keep-all"
    assert kwargs["gazing_ratio_tile"] == 1
    assert kwargs["task_loss_requirement_tile"] is None
    assert identity["autogaze_applicability"] == "hd_keep_all_ablation"
    assert identity["components"]["token_selector"]["adapter"] == "keep-all"


def test_component_autogaze_adapter_forwards_selector_checkpoint_path():
    args = parse_args(
        [
            "--token-selector-adapter",
            "autogaze",
            "--token-selector-path",
            "/models/local-autogaze",
        ]
    )

    kwargs = processor_kwargs(args)
    identity = build_run_identity(args)

    assert args.gazing_mode == "autogaze"
    assert args.autogaze_model == "/models/local-autogaze"
    assert kwargs["autogaze_model_id"] == "/models/local-autogaze"
    assert identity["components"]["token_selector"]["path"] == "/models/local-autogaze"


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


def test_processor_kwargs_uses_nvila_hd_siglip_aligned_autogaze_grid_by_default():
    kwargs = processor_kwargs(make_args())

    assert kwargs["target_scales"] == [56, 112, 196, 392]
    assert kwargs["target_patch_size"] == 14


def test_patch_space_metadata_separates_autogaze_target_from_vision_tower():
    metadata = build_patch_space_metadata(
        DummyModel("56+112+196+392", patch_size=14),
        DummyProcessor(target_scales=[56, 112, 196, 392], target_patch_size=14),
    )

    assert metadata["autogaze_target_patch_size"] == 14
    assert metadata["autogaze_coordinate_patches_per_frame_multiscale"] == 1060
    assert metadata["vision_encoder_patch_size"] == 14
    assert metadata["vision_encoder_patches_per_frame_multiscale"] == 1060
    assert metadata["patch_space_mismatch"] is False


def test_parse_float_sequence_accepts_fractional_reduction_ratios():
    assert parse_float_sequence("1,2.5,4") == [1.0, 2.5, 4.0]


def test_parse_args_accepts_gazing_mode_switch():
    args = parse_args(["--gazing-mode", "keep-all"])

    assert args.gazing_mode == "keep-all"


def test_stream_profile_can_override_gazing_ratio_for_quickstart_comparison():
    args = parse_args(["--mode", "stream-profile", "--stream-gazing-ratio", "0.75"])

    assert args.stream_gazing_ratio == "0.75"
    assert effective_stream_gazing_ratio(args) == 0.75


def test_parse_args_accepts_single_mode_gazing_ratio_tile_override():
    args = parse_args(["--gazing-ratio-tile", "0.75"])

    assert args.gazing_ratio_tile == "0.75"
    assert effective_gazing_ratio_tile(args) == 0.75


def test_stream_profile_uses_nvila_gazing_ratio_when_not_overridden():
    args = parse_args(["--mode", "stream-profile"])

    assert effective_stream_gazing_ratio(args) == [0.2] + [0.06] * 15


def test_parse_args_accepts_gaze_visualization_options():
    args = parse_args(
        [
            "--visualization-output-dir",
            "outputs/viz",
            "--visualization-fps",
            "6",
            "--visualization-selected-max-long-side",
            "720",
        ]
    )

    assert args.visualization_output_dir == "outputs/viz"
    assert args.visualization_fps == 6
    assert args.visualization_selected_max_long_side == 720


def test_parse_args_accepts_warmup_and_repeat_runs():
    args = parse_args(["--warmup-runs", "1", "--repeat-runs", "3"])

    assert args.warmup_runs == 1
    assert args.repeat_runs == 3


def test_parse_args_accepts_single_model_dtype():
    args = parse_args(["--dtype", "float16"])

    assert args.dtype == "float16"


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
            "generate_ms": 87.0,
            "ttft_ms": 30.0,
            "video_preprocess_ms": 13.0,
            "video_preprocess_without_autogaze_ms": 1.0,
            "autogaze_total_ms": 12.0,
            "video_decode_ms": 2.0,
            "video_tiling_ms": 11.0,
            "autogaze_ms": 12.0,
            "autogaze_forward_ms": 10.0,
            "gazing_info_total_ms": 12.0,
            "autogaze_model_forward_ms": 10.0,
            "generation_decode_after_ttft_estimated_ms": 57.0,
            "siglip_vision_ms": 20.0,
            "mm_projector_ms": 3.0,
            "llm_forward_ms": 50.0,
            "processor_peak_memory_bytes": 1500,
            "ttft_peak_memory_bytes": 1200,
            "llm_peak_memory_bytes": 1024,
            "peak_memory_bytes": 1600,
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
            "generate_ms": {"median": 77.0},
            "ttft_ms": {"median": 25.0},
            "video_preprocess_ms": {"median": 13.0},
            "video_preprocess_without_autogaze_ms": {"median": 1.0},
            "autogaze_total_ms": {"median": 12.0},
            "video_decode_ms": {"median": 2.0},
            "autogaze_ms": {"median": 12.0},
            "autogaze_forward_ms": {"median": 10.0},
            "gazing_info_total_ms": {"median": 12.0},
            "autogaze_model_forward_ms": {"median": 10.0},
            "generation_decode_after_ttft_estimated_ms": {"median": 52.0},
            "siglip_vision_ms": {"median": 18.0},
            "llm_forward_ms": {"median": 45.0},
            "processor_peak_memory_bytes": {"median": 1400.0},
            "ttft_peak_memory_bytes": {"median": 1100.0},
            "llm_peak_memory_bytes": {"median": 900.0},
            "peak_memory_bytes": {"median": 1500.0},
        },
    }

    summary = build_single_summary(payload)

    assert summary["key_autogaze_effect"] == {
        "gazing_mode": "autogaze",
        "total_ms_median": 90.0,
        "ttft_ms_median": 25.0,
        "autogaze_total_ms_median": 12.0,
        "autogaze_forward_ms_median": 10.0,
        "gazing_info_total_ms_median": 12.0,
        "autogaze_model_forward_ms_median": 10.0,
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
        "patch_space_basis": {
            "autogaze_target_scales": None,
            "autogaze_target_patch_size": None,
            "autogaze_coordinate_patches_per_frame_multiscale": None,
            "autogaze_coordinate_patches_per_frame_by_scale": None,
            "vision_encoder_scales": None,
            "vision_encoder_patch_size": None,
            "vision_encoder_patches_per_frame_multiscale": None,
            "vision_encoder_patches_per_frame_by_scale": None,
            "patch_space_mismatch": None,
            "note": None,
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
                "For example, 128 frames * 8 tiles/frame * multiscale patch positions can exceed one million positions."
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
    assert summary["module_latency_ms"] == {
        "total_median": 90.0,
        "generate_median": 77.0,
        "preprocess_without_autogaze_median": 1.0,
        "preprocess_total_median": 13.0,
        "autogaze_median": 12.0,
        "autogaze_total_median": 12.0,
        "gazing_info_total_median": 12.0,
        "autogaze_model_forward_median": 10.0,
        "vit_encoder_median": 18.0,
        "llm_median": 45.0,
        "field_note": (
            "Summary-level module latency is intentionally coarse. "
            "preprocess_without_autogaze=video_preprocess_without_autogaze_ms, "
            "preprocess_total=legacy inclusive video_preprocess_ms, autogaze=autogaze_total_ms, "
            "gazing_info_total=gazing_info_total_ms, "
            "autogaze_model_forward=autogaze_model_forward_ms, "
            "vit_encoder=siglip_vision_ms, llm=llm_forward_ms. "
            "The primary additive formula is preprocess_without_autogaze + autogaze_total + generate."
        ),
    }
    assert summary["latency_accounting"]["additive_total_ms"] == {
        "formula": "total_ms = video_preprocess_without_autogaze_ms + autogaze_total_ms + generate_ms",
        "total_ms": 90.0,
        "video_preprocess_without_autogaze_ms": 1.0,
        "autogaze_total_ms": 12.0,
        "generate_ms": 77.0,
        "recomputed_total_ms": 90.0,
        "delta_ms": 0.0,
        "ttft_ms_excluded_from_total": 25.0,
    }
    assert summary["latency_accounting"]["nested_preprocess_breakdown_ms"]["video_decode_ms"] == {
        "value": 2.0,
        "included_in": "video_preprocess_without_autogaze_ms",
        "add_to_total_ms": False,
    }
    assert summary["key_metrics_summary"] == {
        "latency_ms": {
            "total_median": 90.0,
            "generate_median": 77.0,
            "preprocess_without_autogaze_median": 1.0,
            "preprocess_total_median": 13.0,
            "autogaze_median": 12.0,
            "autogaze_total_median": 12.0,
            "gazing_info_total_median": 12.0,
            "autogaze_model_forward_median": 10.0,
            "vit_encoder_median": 18.0,
            "llm_median": 45.0,
        },
        "latency_accounting": summary["latency_accounting"],
        "tokens": {
            "video_sampled_frames": 8,
            "thumbnail_sampled_frames": 4,
            "encoder_patch_tokens_before_keep_all_or_raw": 80,
            "encoder_patch_tokens_after_autogaze": 40,
            "encoder_token_reduction_ratio": 2.0,
            "autogaze_input_tile_patch_tokens": 64,
            "autogaze_selected_tile_patch_tokens": 24,
            "autogaze_patch_reduction_ratio": 64 / 24,
            "llm_visual_tokens_before_keep_all_estimated": 80,
            "llm_visual_tokens_after_actual": 40,
            "llm_visual_token_reduction_ratio": 2.0,
        },
        "memory_bytes": {
            "processor_peak_median": 1400.0,
            "ttft_peak_median": 1100.0,
            "llm_peak_median": 900.0,
            "overall_peak_median": 1500.0,
        },
    }
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
            "For example, 128 frames * 8 tiles/frame * multiscale patch positions can exceed one million positions."
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
            "decode": {
                "requested_decode_strategy": "auto",
                "decode_strategy": "seek",
                "decode_frames_read": 256,
                "decode_seek_groups": 64,
                "decode_keyframes_indexed": 120,
                "decode_packets_scanned_for_keyframes": 9000,
            },
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
        "video_decode_requested_strategy": "auto",
        "video_decode_strategy": "seek",
        "video_decode_strategy_fallback_error": None,
        "video_decode_frames_read": 256,
        "video_decode_seek_groups": 64,
        "video_decode_keyframes_indexed": 120,
        "video_decode_packets_scanned_for_keyframes": 9000,
        "spatial_tiles_per_video": [8],
        "temporal_chunks_per_video": 8,
    }


def test_build_processing_budget_summary_explains_resize_tiling_thumbnail_and_patch_budget():
    video_input_summary = {
        "source_resolution": "3840x2160",
        "source_width": 3840,
        "source_height": 2160,
        "processor_input_resolution": "1280x720",
        "processor_input_width": 1280,
        "processor_input_height": 720,
        "runner_resize_enabled": True,
        "runner_resize_request": {
            "shortest_edge": 720,
            "longest_edge": None,
            "width": None,
            "height": None,
        },
        "requested_video_frames": 128,
        "actual_video_frames": 128,
        "requested_thumbnail_frames": 64,
        "actual_thumbnail_frames": 64,
        "spatial_tiles_per_video": [8],
        "temporal_chunks_per_video": [8],
    }
    token_metrics = {
        "video_sampled_frames": 128,
        "thumbnail_sampled_frames": 64,
        "spatial_tiles_per_video": [8],
        "temporal_chunks_per_video": [8],
        "autogaze_input_tile_frame_instances": 1024,
        "encoder_patches_per_frame_multiscale": 1060,
        "encoder_patches_per_frame_by_scale": {"56": 16, "112": 64, "196": 196, "392": 784},
        "encoder_raw_tile_patch_tokens": 1085440,
        "encoder_autogaze_selected_tile_patch_tokens": 108544,
        "encoder_raw_thumbnail_patch_tokens": 67840,
        "encoder_autogaze_selected_thumbnail_patch_tokens": 67840,
        "encoder_raw_patch_tokens": 1153280,
        "encoder_autogaze_selected_patch_tokens": 176384,
        "encoder_token_reduction_ratio": 1153280 / 176384,
        "autogaze_input_patch_tokens": 1085440,
        "autogaze_selected_patch_tokens": 108544,
        "autogaze_patch_reduction_ratio": 10.0,
        "llm_keep_all_visual_tokens_estimated": 128512,
        "llm_actual_visual_tokens": 19632,
        "llm_visual_token_reduction_ratio": 128512 / 19632,
        "token_shuffle": 9,
    }

    summary = build_processing_budget_summary(
        video_input_summary=video_input_summary,
        token_metrics=token_metrics,
        runner="nvila_runner",
    )

    assert summary["video"]["source_resolution"] == "3840x2160"
    assert summary["video"]["processor_input_resolution"] == "1280x720"
    assert summary["tiling"]["spatial_tiles_per_frame"] == 8
    assert summary["tiling"]["tile_frame_instances"] == 1024
    assert summary["thumbnail"]["enabled"] is True
    assert summary["thumbnail"]["actual_frames"] == 64
    assert summary["multiscale_patch_space"]["patch_positions_per_tile_frame"] == 1060
    assert summary["multiscale_patch_space"]["patch_positions_by_scale"]["392"] == 784
    assert summary["patch_budget_before_siglip"]["keep_all_total_patch_tokens"] == 1153280
    assert summary["patch_budget_before_siglip"]["autogaze_selected_total_patch_tokens"] == 176384
    assert summary["patch_budget_before_siglip"]["autogaze_selected_tile_patch_tokens"] == 108544
    assert summary["patch_budget_before_siglip"]["thumbnail_policy"] == "keep_all"
    assert summary["single_scale_dense_vision_budget"]["comparison_scope"] == "siglip_392px_single_scale_reference"
    assert summary["single_scale_dense_vision_budget"]["patch_positions_per_tile_frame"] == 784
    assert summary["single_scale_dense_vision_budget"]["tile_patch_tokens"] == 802816
    assert summary["single_scale_dense_vision_budget"]["thumbnail_patch_tokens"] == 50176
    assert summary["single_scale_dense_vision_budget"]["total_patch_tokens"] == 852992
    assert summary["single_scale_dense_vision_budget"]["llm_visual_tokens_estimated"] == 94848
    assert summary["single_scale_dense_vision_budget"]["ratio_over_autogaze_selected_total_patch_tokens"] == 852992 / 176384
    assert summary["llm_visual_budget"]["keep_all_visual_tokens_estimated"] == 128512
    assert summary["llm_visual_budget"]["actual_visual_tokens"] == 19632


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


def test_h100_risk_band_uses_default_70gib_budget_and_context_limit():
    assert h100_risk_band(54.9) == "green"
    assert h100_risk_band(55.0) == "yellow"
    assert h100_risk_band(70.0) == "red"
    assert h100_risk_band(1.0, context_exceeded=True) == "context_red"


def test_estimate_h100_preflight_config_reports_memory_and_context_risk_for_hd():
    estimate = estimate_h100_preflight_config(
        width=3840,
        height=2160,
        source_frames=9000,
        model_family=MODEL_FAMILY_HD_AUTOGAZE,
        num_video_frames=1024,
        num_video_frames_thumbnail=512,
        max_tiles_video=48,
        resize_shortest_edge=None,
        token_reduction_ratio=1.0,
        h100_budget_gib=70.0,
    )

    assert estimate["model_family"] == MODEL_FAMILY_HD_AUTOGAZE
    assert estimate["config"]["num_video_frames"] == 1024
    assert estimate["tokens"]["keep_all_llm_visual_tokens_estimated"] >= estimate["tokens"]["actual_llm_visual_tokens_estimated"]
    assert estimate["memory"]["h100_budget_gib"] == 70.0
    assert estimate["risk"]["band"] in {"green", "yellow", "red", "context_red"}
    assert "recommended_role" in estimate


def test_h100_preflight_distinguishes_full_video_and_streamed_autogaze_working_set():
    full_video = estimate_h100_preflight_config(
        width=3584,
        height=2016,
        source_frames=9000,
        model_family=MODEL_FAMILY_HD_AUTOGAZE,
        num_video_frames=1024,
        num_video_frames_thumbnail=128,
        max_tiles_video=48,
        resize_shortest_edge=None,
        token_reduction_ratio=400.0,
        h100_budget_gib=80.0,
    )
    streamed = estimate_h100_preflight_config(
        width=3584,
        height=2016,
        source_frames=9000,
        model_family=MODEL_FAMILY_HD_AUTOGAZE,
        num_video_frames=1024,
        num_video_frames_thumbnail=128,
        max_tiles_video=48,
        resize_shortest_edge=None,
        token_reduction_ratio=400.0,
        h100_budget_gib=80.0,
        stream_chunk_frames=16,
        max_batch_size_autogaze=16,
    )

    assert full_video["memory"]["autogaze_working_mode"] == "full_video_tensor"
    assert streamed["memory"]["autogaze_working_mode"] == "stream_chunk"
    assert streamed["config"]["stream_chunk_frames"] == 16
    assert streamed["config"]["max_batch_size_autogaze"] == 16
    assert (
        streamed["memory"]["autogaze_working_bytes_estimated"]
        < full_video["memory"]["autogaze_working_bytes_estimated"]
    )
    assert streamed["memory"]["autogaze_tile_tensor_full_video_bytes_estimated"] == full_video["memory"][
        "autogaze_working_bytes_estimated"
    ]
    assert (
        streamed["memory"]["autogaze_forward_batch_tensor_bytes_estimated"]
        < streamed["memory"]["autogaze_tensor_residency_bytes_estimated"]
    )


def test_h100_preflight_estimates_vit_and_llm_bottlenecks_after_streamed_autogaze():
    estimate = estimate_h100_preflight_config(
        width=3584,
        height=2016,
        source_frames=9000,
        model_family=MODEL_FAMILY_HD_AUTOGAZE,
        num_video_frames=1024,
        num_video_frames_thumbnail=128,
        max_tiles_video=48,
        resize_shortest_edge=None,
        token_reduction_ratio=400.0,
        h100_budget_gib=80.0,
        stream_chunk_frames=16,
        max_batch_size_autogaze=16,
        max_batch_size_siglip=32,
    )

    assert estimate["vision_encoder"]["actual"]["max_sequence_tokens_per_batch"] == 1060
    assert estimate["vision_encoder"]["actual"]["tile_sequence_tokens"] == 43
    assert estimate["tokens"]["actual_llm_visual_tokens_estimated"] == 28836
    assert estimate["tokens"]["actual_context_tokens_estimated"] == 29092
    assert estimate["mllm"]["actual"]["context_tokens"] == 29092
    assert estimate["bottlenecks"]["dominant_memory_stage_estimated"] == "mllm_prefill"
    assert estimate["bottlenecks"]["stage_memory_gib_estimated"]["mllm_prefill"] > estimate["bottlenecks"][
        "stage_memory_gib_estimated"
    ]["vision_encoder"]


def test_h100_decision_row_highlights_resolution_frames_tokens_and_bottleneck():
    estimate = estimate_h100_preflight_config(
        width=3840,
        height=2160,
        source_frames=9000,
        model_family=MODEL_FAMILY_HD_AUTOGAZE,
        num_video_frames=1024,
        num_video_frames_thumbnail=128,
        max_tiles_video=48,
        resize_shortest_edge=None,
        resize_longest_edge=3584,
        token_reduction_ratio=400.0,
        h100_budget_gib=80.0,
        stream_chunk_frames=16,
        max_batch_size_autogaze=16,
        max_batch_size_siglip=32,
    )

    row = build_h100_decision_row(estimate)

    assert row["source_resolution"] == "3840x2160"
    assert row["effective_resolution"] == "3584x2016"
    assert row["frames"] == 1024
    assert row["thumbnail_frames"] == 128
    assert row["spatial_tiles"] == 45
    assert row["llm_actual_visual_tokens"] == 28836
    assert row["llm_actual_context_tokens"] == 29092
    assert row["dominant_memory_stage"] == "mllm_prefill"
    assert row["risk_band"] == "yellow"


def test_h100_decision_row_exposes_llm_context_capacity_requirements():
    estimate = estimate_h100_preflight_config(
        width=3584,
        height=2016,
        source_frames=9000,
        model_family=MODEL_FAMILY_HD_AUTOGAZE,
        num_video_frames=1024,
        num_video_frames_thumbnail=128,
        max_tiles_video=48,
        resize_shortest_edge=None,
        token_reduction_ratio=200.0,
        h100_budget_gib=80.0,
        stream_chunk_frames=16,
        max_batch_size_autogaze=16,
        max_batch_size_siglip=32,
    )

    row = build_h100_decision_row(estimate)

    assert row["llm_actual_context_tokens"] == 42532
    assert row["llm_context_margin_tokens"] == -1572
    assert row["llm_context_fits"] is False
    assert row["llm_context_utilization_percent"] == 103.84
    assert row["min_tile_reduction_ratio_for_context"] == 212.0
    assert row["max_tile_sequence_tokens_for_context"] == 80


def test_h100_preflight_can_add_resident_autogaze_model_memory():
    base = estimate_h100_preflight_config(
        width=3584,
        height=2016,
        source_frames=9000,
        model_family=MODEL_FAMILY_HD_AUTOGAZE,
        num_video_frames=1024,
        num_video_frames_thumbnail=128,
        max_tiles_video=48,
        resize_shortest_edge=None,
        token_reduction_ratio=400.0,
        h100_budget_gib=80.0,
        stream_chunk_frames=16,
        max_batch_size_autogaze=16,
        max_batch_size_siglip=32,
    )
    resident = estimate_h100_preflight_config(
        width=3584,
        height=2016,
        source_frames=9000,
        model_family=MODEL_FAMILY_HD_AUTOGAZE,
        num_video_frames=1024,
        num_video_frames_thumbnail=128,
        max_tiles_video=48,
        resize_shortest_edge=None,
        token_reduction_ratio=400.0,
        h100_budget_gib=80.0,
        stream_chunk_frames=16,
        max_batch_size_autogaze=16,
        max_batch_size_siglip=32,
        autogaze_model_resident_gib=5.0,
    )

    assert resident["memory"]["autogaze_residency_policy"] == "resident"
    assert resident["memory"]["autogaze_model_resident_gib_assumed"] == 5.0
    assert resident["memory"]["estimated_vram_gib"] == base["memory"]["estimated_vram_gib"] + 5.0
    assert build_h100_decision_row(resident)["autogaze_model_resident_gib"] == 5.0


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


def test_parse_args_accepts_general_video_decode_strategy():
    args = parse_args(["--mode", "hlvid", "--video-decode-strategy", "seek"])

    assert args.video_decode_strategy == "seek"


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
        "size": {"height": 392, "width": 392},
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
