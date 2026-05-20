from pathlib import Path
import sys

import pytest

from repro.autogaze_timing_compare import (
    CompareConfig,
    build_quickstart_command,
    build_single_command,
    build_stream_profile_command,
    build_sweep_configs,
    config_from_args,
    parse_args,
    parse_float_sweep,
    check_target_runtime,
    run_comparison,
    run_sweep,
    summarize_comparison,
    validate_compare_config,
    write_markdown_report,
)


def test_default_subprocess_python_uses_current_interpreter():
    args = parse_args([])

    assert args.python == Path(sys.executable)


def test_default_paths_are_portable_repo_relative_paths():
    args = parse_args([])

    assert args.autogaze_repo == Path("external/AutoGaze")
    assert args.weights_root == Path("weights")


def test_default_thumbnail_frames_keep_nvila_single_compatible():
    args = parse_args([])

    assert args.thumbnail_frames == 1


def test_validate_compare_config_rejects_thumbnail_zero_for_single_lane():
    config = CompareConfig(thumbnail_frames=0, run_single=True)

    with pytest.raises(ValueError, match="thumbnail"):
        validate_compare_config(config)


def test_validate_compare_config_allows_thumbnail_zero_when_single_lane_is_skipped():
    config = CompareConfig(thumbnail_frames=0, run_single=False)

    validate_compare_config(config)


def test_config_from_args_can_skip_stream_profile_lane():
    args = parse_args(["--skip-stream-profile"])
    config = config_from_args(args)

    assert config.run_stream_profile is False
    assert config.run_single is True


def test_missing_subprocess_python_reports_cli_fix():
    config = CompareConfig(python=Path("/definitely/missing/python"), require_mps=False)

    with pytest.raises(RuntimeError, match="--python"):
        check_target_runtime(config)


def test_build_commands_use_requested_mps_venv_and_local_weights(tmp_path):
    config = CompareConfig(
        python=Path("/Users/mrc/myresearch/AutoGaze/.venv/bin/python"),
        workspace_root=Path("/workspace/autogaze-repro"),
        autogaze_repo=Path("/Users/mrc/myresearch/AutoGaze"),
        weights_root=Path("/Users/mrc/myresearch/AutoGaze/weights"),
        output_dir=tmp_path,
        device="mps",
        dtype="float32",
        frames=16,
        quickstart_batch_size=2,
        stream_chunk_frames=16,
        max_tiles_video=1,
        warmup=1,
        repeat=3,
        autogaze_target_scales="56+112+196+392",
        autogaze_target_patch_size=14,
    )

    quickstart = build_quickstart_command(config)
    stream_profile = build_stream_profile_command(config)
    single = build_single_command(config)

    assert quickstart[:3] == [
        "/Users/mrc/myresearch/AutoGaze/.venv/bin/python",
        "-m",
        "repro.autogaze_bench",
    ]
    assert stream_profile[:3] == [
        "/Users/mrc/myresearch/AutoGaze/.venv/bin/python",
        "-m",
        "repro.nvila_runner",
    ]
    assert quickstart[quickstart.index("--autogaze-repo") + 1] == "/Users/mrc/myresearch/AutoGaze"
    assert quickstart[quickstart.index("--autogaze-model") + 1] == "/Users/mrc/myresearch/AutoGaze/weights/AutoGaze"
    assert quickstart[quickstart.index("--siglip-model") + 1] == "/Users/mrc/myresearch/AutoGaze/weights/siglip2-base-patch16-224"
    assert quickstart[quickstart.index("--batch-size") + 1] == "2"
    assert "--skip-siglip" in quickstart
    assert quickstart[quickstart.index("--target-scales") + 1] == "56+112+196+392"
    assert quickstart[quickstart.index("--target-patch-size") + 1] == "14"
    assert stream_profile[stream_profile.index("--autogaze-model") + 1] == "/Users/mrc/myresearch/AutoGaze/weights/AutoGaze"
    assert stream_profile[stream_profile.index("--device") + 1] == "mps"
    assert stream_profile[stream_profile.index("--stream-chunk-frames") + 1] == "16"
    assert stream_profile[stream_profile.index("--max-tiles-video") + 1] == "1"
    assert stream_profile[stream_profile.index("--stream-gazing-ratio") + 1] == "0.75"
    assert stream_profile[stream_profile.index("--task-loss-requirement-tile") + 1] == "0.7"
    assert stream_profile[stream_profile.index("--autogaze-target-scales") + 1] == "56+112+196+392"
    assert single[:3] == [
        "/Users/mrc/myresearch/AutoGaze/.venv/bin/python",
        "-m",
        "repro.nvila_runner",
    ]
    assert single[single.index("--mode") + 1] == "single"
    assert single[single.index("--gazing-ratio-tile") + 1] == "0.75"
    assert single[single.index("--task-loss-requirement-tile") + 1] == "0.7"
    assert single[single.index("--num-video-frames") + 1] == "16"
    assert single[single.index("--num-video-frames-thumbnail") + 1] == "1"
    assert single[single.index("--max-tiles-video") + 1] == "1"
    assert "--measure-ttft" in single


def test_run_comparison_dry_run_can_compare_quickstart_to_single_without_stream(tmp_path):
    config = CompareConfig(
        python=Path("/Users/mrc/myresearch/AutoGaze/.venv/bin/python"),
        workspace_root=Path("/workspace/autogaze-repro"),
        output_dir=tmp_path,
        run_stream_profile=False,
    )

    payload = run_comparison(config, dry_run=True)

    assert set(payload["commands"]) == {"quickstart", "single"}
    assert payload["commands"]["single"][payload["commands"]["single"].index("--mode") + 1] == "single"


def test_summarize_comparison_separates_quickstart_and_stream_profile_metrics(tmp_path):
    quickstart_payload = {
        "metadata": {"device": "mps"},
        "input": {"video": "example_input.mp4", "frames": 16, "dtype": "float32"},
        "autogaze_latency_options": {
            "batch_size": 1,
            "gazing_ratio": 0.75,
            "task_loss_requirement": 0.7,
            "target_scales": None,
            "target_patch_size": None,
            "siglip_enabled": False,
        },
        "latency_ms": {
            "autogaze": {"median": 100.0},
        },
        "gaze": {
            "raw_patch_budget": 4240,
            "selected_non_padded_patches": 212,
            "total_gaze_slots": 348,
            "token_reduction_ratio": 20.0,
        },
    }
    stream_payload = {
        "metadata": {"device": "mps"},
        "mode": "stream-profile",
        "timing_ms": {
            "video_decode_scan": 9.0,
            "spatial_tile_build": 3.0,
            "tile_autogaze_tensorize": 4.0,
            "tile_autogaze_forward": 300.0,
            "pre_llm_stream_total_measured": 320.0,
        },
        "sampling": {"num_video_frames": 16, "stream_chunk_frames": 16},
        "gaze": {
            "raw_patch_budget": 4240,
            "selected_non_padded_patches": 212,
            "total_gaze_slots": 348,
            "token_reduction_ratio": 20.0,
            "tile_sequences": 1,
        },
        "token_metrics": {
            "encoder_raw_patch_tokens": 4240,
            "encoder_autogaze_selected_patch_tokens": 212,
            "encoder_token_reduction_ratio": 20.0,
        },
    }
    single_payload = {
        "metadata": {"device": "mps"},
        "video": "example_input.mp4",
        "result": {
            "autogaze_runtime_config": {
                "gazing_ratio_tile": 0.75,
                "task_loss_requirement_tile": 0.7,
                "max_batch_size_autogaze": 16,
            },
            "autogaze_model_forward_ms": 310.0,
            "autogaze_total_ms": 340.0,
            "video_preprocess_without_autogaze_ms": 60.0,
            "video_preprocess_ms": 400.0,
            "generate_ms": 900.0,
            "total_ms": 1300.0,
            "ttft_ms": 500.0,
            "siglip_vision_ms": 200.0,
            "llm_forward_ms": 600.0,
            "token_metrics": {
                "video_sampled_frames": 16,
                "autogaze_input_patch_tokens": 4240,
                "autogaze_selected_patch_tokens": 212,
                "autogaze_patch_reduction_ratio": 20.0,
            },
            "stage_timings_ms": {
                "processor": {
                    "autogaze_forward_batched": {
                        "total_ms": 310.0,
                        "count": 1,
                    }
                }
            },
        },
    }

    summary = summarize_comparison(
        quickstart_payload,
        stream_payload,
        single_payload,
        quickstart_json=tmp_path / "quickstart.json",
        stream_json=tmp_path / "stream.json",
        single_json=tmp_path / "single.json",
    )

    assert summary["quickstart_direct"]["autogaze_ms"] == 100.0
    assert summary["quickstart_direct"]["batch_size"] == 1
    assert summary["quickstart_direct"]["siglip_full_ms"] is None
    assert summary["autogaze_latency_options"]["quickstart_direct"]["batch_size"] == 1
    assert summary["autogaze_latency_options"]["current_implementation_single"]["max_batch_size_autogaze"] == 16
    assert summary["current_implementation_stream_profile"]["autogaze_model_forward_ms"] == 300.0
    assert summary["current_implementation_stream_profile"]["pre_llm_stream_total_ms"] == 320.0
    assert summary["current_implementation_single"]["autogaze_model_forward_ms"] == 310.0
    assert summary["current_implementation_single"]["autogaze_total_ms"] == 340.0
    assert summary["current_implementation_single"]["generate_ms"] == 900.0
    assert summary["comparison"]["stream_autogaze_forward_vs_quickstart_ratio"] == 3.0
    assert summary["comparison"]["single_autogaze_forward_vs_quickstart_ratio"] == 3.1
    assert summary["comparison"]["single_autogaze_total_vs_quickstart_ratio"] == 3.4
    assert summary["comparison"]["stream_total_vs_quickstart_autogaze_ratio"] == 3.2
    assert summary["comparison"]["raw_patch_budget_ratio"] == 1.0
    assert summary["comparison"]["single_raw_patch_budget_ratio"] == 1.0


def test_summarize_comparison_allows_quickstart_to_single_without_stream(tmp_path):
    quickstart_payload = {
        "metadata": {"device": "cuda"},
        "input": {"video": "example_input.mp4", "frames": 16, "batch_size": 1, "dtype": "float16"},
        "latency_ms": {"autogaze": {"median": 10.0}},
        "gaze": {
            "raw_patch_budget": 4240,
            "selected_non_padded_patches": 424,
            "total_gaze_slots": 424,
            "token_reduction_ratio": 10.0,
        },
    }
    single_payload = {
        "metadata": {"device": "cuda"},
        "video": "example_input.mp4",
        "result": {
            "autogaze_model_forward_ms": 20.0,
            "autogaze_total_ms": 25.0,
            "total_ms": 80.0,
            "autogaze_runtime_config": {
                "gazing_ratio_tile": 0.75,
                "task_loss_requirement_tile": 0.7,
                "max_batch_size_autogaze": 16,
            },
            "token_metrics": {
                "video_sampled_frames": 16,
                "thumbnail_sampled_frames": 1,
                "autogaze_input_patch_tokens": 4240,
                "autogaze_selected_patch_tokens": 424,
                "autogaze_patch_reduction_ratio": 10.0,
            },
            "video_input_summary": {"spatial_tiles_per_video": 1},
        },
    }

    summary = summarize_comparison(
        quickstart_payload,
        None,
        single_payload,
        quickstart_json=tmp_path / "quickstart.json",
        stream_json=None,
        single_json=tmp_path / "single.json",
    )
    report = tmp_path / "report.md"
    write_markdown_report(summary, report, {"quickstart": ["quick"], "single": ["single"]})

    assert summary["current_implementation_stream_profile"] is None
    assert summary["comparison"]["single_autogaze_forward_vs_quickstart_ratio"] == 2.0
    assert "Current stream-profile" not in report.read_text()


def test_parse_float_sweep_uses_comma_separated_values_or_default():
    assert parse_float_sweep("0.2, 0.75,1", default=[0.75]) == [0.2, 0.75, 1.0]
    assert parse_float_sweep("", default=[0.75]) == [0.75]
    assert parse_float_sweep(None, default=[0.7]) == [0.7]


def test_build_sweep_configs_sets_policy_and_separate_output_dirs(tmp_path):
    config = CompareConfig(output_dir=tmp_path, gazing_ratio=0.75, task_loss_requirement=0.7)

    configs = build_sweep_configs(config, gazing_ratios=[0.2, 0.75], task_loss_requirements=[0.6])

    assert [item.gazing_ratio for item in configs] == [0.2, 0.75]
    assert [item.task_loss_requirement for item in configs] == [0.6, 0.6]
    assert configs[0].output_dir == tmp_path / "gazing_0p2__loss_0p6"
    assert configs[1].output_dir == tmp_path / "gazing_0p75__loss_0p6"


def test_run_sweep_dry_run_builds_policy_matrix(tmp_path):
    config = CompareConfig(
        python=Path("/Users/mrc/myresearch/AutoGaze/.venv/bin/python"),
        workspace_root=Path("/workspace/autogaze-repro"),
        output_dir=tmp_path,
        quickstart_batch_size=2,
        run_single=False,
        autogaze_target_scales="56+112+196+392",
        autogaze_target_patch_size=14,
    )

    payload = run_sweep(config, gazing_ratios=[0.2, 0.75], task_loss_requirements=[0.6, 0.7], dry_run=True)

    assert payload["sweep"]["num_runs"] == 4
    assert payload["sweep"]["gazing_ratios"] == [0.2, 0.75]
    assert payload["sweep"]["task_loss_requirements"] == [0.6, 0.7]
    assert len(payload["runs"]) == 4
    first = payload["runs"][0]
    assert first["gazing_ratio"] == 0.2
    assert first["task_loss_requirement"] == 0.6
    assert first["output_dir"].endswith("gazing_0p2__loss_0p6")
    quickstart = first["commands"]["quickstart"]
    stream = first["commands"]["stream_profile"]
    assert quickstart[quickstart.index("--gazing-ratio") + 1] == "0.2"
    assert quickstart[quickstart.index("--task-loss-requirement") + 1] == "0.6"
    assert quickstart[quickstart.index("--batch-size") + 1] == "2"
    assert quickstart[quickstart.index("--target-scales") + 1] == "56+112+196+392"
    assert stream[stream.index("--stream-gazing-ratio") + 1] == "0.2"
    assert stream[stream.index("--task-loss-requirement-tile") + 1] == "0.6"
    assert "single" not in first["commands"]
    assert (tmp_path / "autogaze_policy_sweep_summary.json").exists()
    assert (tmp_path / "autogaze_policy_sweep_report.md").exists()
