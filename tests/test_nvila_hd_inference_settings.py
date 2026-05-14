from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import infer_autogaze
import infer_full
from poc_infer_utils import apply_nvila_hd_overrides, load_config, nvila_hd_effective_settings


def _cfg(name: str) -> Path:
    return ROOT / "configs" / "poc_inference" / name


def test_nvila_hd_presets_load() -> None:
    smoke = load_config(_cfg("nvila_hd_smoke.yaml"))
    default = load_config(_cfg("nvila_hd_default.yaml"))
    memory_safe = load_config(_cfg("nvila_hd_memory_safe.yaml"))

    assert smoke["mllm"]["processor_from_pretrained_kwargs"]["num_video_frames"] == 16
    assert smoke["mllm"]["processor_from_pretrained_kwargs"]["num_video_frames_thumbnail"] == 8
    assert smoke["mllm"]["processor_from_pretrained_kwargs"]["max_tiles_video"] == 8
    assert smoke["mllm"]["from_pretrained_kwargs"]["max_batch_size_siglip"] == 4
    assert smoke["video_input"]["read_mode"] == "full"
    assert smoke["streaming"]["enabled"] is False

    assert default["mllm"]["processor_from_pretrained_kwargs"]["num_video_frames"] == 128
    assert default["mllm"]["processor_from_pretrained_kwargs"]["num_video_frames_thumbnail"] == 64
    assert default["mllm"]["processor_from_pretrained_kwargs"]["max_tiles_video"] == 48
    assert default["runtime"]["device"] == "auto"
    assert default["runtime"]["dtype"] == "float16"
    assert default["video_input"]["read_mode"] == "full"
    assert default["streaming"]["enabled"] is False

    assert memory_safe["mllm"]["processor_from_pretrained_kwargs"]["num_video_frames"] == 64
    assert memory_safe["mllm"]["processor_from_pretrained_kwargs"]["num_video_frames_thumbnail"] == 32
    assert memory_safe["mllm"]["processor_from_pretrained_kwargs"]["max_batch_size_autogaze"] == 4
    assert memory_safe["mllm"]["from_pretrained_kwargs"]["max_batch_size_siglip"] == 8
    assert memory_safe["video_input"]["read_mode"] == "full"
    assert memory_safe["streaming"]["enabled"] is False


def test_infer_full_full_mode_comparison_configs_load() -> None:
    resize_short = load_config(_cfg("infer_full_full_resize_short.yaml"))
    chop_short = load_config(_cfg("infer_full_full_resize_then_chop_short.yaml"))
    resize_long = load_config(_cfg("infer_full_full_resize_long.yaml"))
    chop_long = load_config(_cfg("infer_full_full_resize_then_chop_long.yaml"))

    for cfg in (resize_short, chop_short, resize_long, chop_long):
        assert cfg["video_input"]["read_mode"] == "full"
        assert cfg["memory"]["fail_on_full_video_load"] is False
        assert cfg["mllm"]["name"] == "nvila"
        assert cfg["mllm"]["video_input_source"] == "processed_tensor"
        assert cfg["autogaze"]["enabled"] is True
        assert cfg["autogaze"]["gaze_ratio"] == 0.75
        assert cfg["autogaze"]["task_loss_requirement"] == 0.7
        assert cfg["mllm"]["sync_autogaze_controls_from_config"] is True
        assert cfg["mllm"]["from_pretrained_kwargs"]["max_batch_size_siglip"] == 32
        assert cfg["mllm"]["processor_from_pretrained_kwargs"]["num_video_frames_thumbnail"] == 16
        assert cfg["mllm"]["processor_from_pretrained_kwargs"]["max_batch_size_autogaze"] == 16
        assert "gazing_ratio_tile" not in cfg["mllm"]["processor_from_pretrained_kwargs"]
        assert "task_loss_requirement_tile" not in cfg["mllm"]["processor_from_pretrained_kwargs"]

    assert resize_short["scaling"]["mode"] == "resize"
    assert chop_short["scaling"]["mode"] == "resize_then_chop"
    assert resize_long["video_input"]["max_decode_frames"] == 96
    assert resize_long["frame_selection"]["mode"] == "all"
    assert chop_long["video_input"]["max_decode_frames"] == 96
    assert chop_long["scaling"]["max_chops"] == 8


def test_nvila_hd_cli_parses_and_overrides_config() -> None:
    args = infer_full.parse_args(
        [
            "--config",
            str(_cfg("nvila_hd_smoke.yaml")),
            "--video-path",
            "dummy",
            "--query-text",
            "Describe.",
            "--num-video-frames",
            "32",
            "--num-video-frames-thumbnail",
            "12",
            "--max-tiles-video",
            "10",
            "--gazing-ratio-tile",
            "0.2,0.06,0.06",
            "--task-loss-requirement-tile",
            "0.6",
            "--gazing-ratio-thumbnail",
            "1.0",
            "--task-loss-requirement-thumbnail",
            "none",
            "--max-batch-size-autogaze",
            "3",
            "--max-batch-size-siglip",
            "5",
        ]
    )
    cfg = apply_nvila_hd_overrides(load_config(args.config), args)
    settings = nvila_hd_effective_settings(cfg)

    assert args.num_video_frames == 32
    assert args.num_video_frames_thumbnail == 12
    assert args.max_tiles_video == 10
    assert settings["num_video_frames"] == 32
    assert settings["num_video_frames_thumbnail"] == 12
    assert settings["max_tiles_video"] == 10
    assert settings["gazing_ratio_tile"] == [0.2, 0.06, 0.06]
    assert settings["task_loss_requirement_tile"] == 0.6
    assert settings["gazing_ratio_thumbnail"] == 1.0
    assert settings["task_loss_requirement_thumbnail"] is None
    assert settings["max_batch_size_autogaze"] == 3
    assert settings["max_batch_size_siglip"] == 5
    assert cfg["frame_selection"]["num_frames"] == 32
    assert cfg["streaming"]["window_size"] == 32


def test_infer_autogaze_saves_nvila_hd_runtime_metadata(tmp_path: Path) -> None:
    output_dir = tmp_path / "autogaze_nvila_hd"
    summary = infer_autogaze.run(
        infer_autogaze.parse_args(
            [
                "--config",
                str(_cfg("nvila_hd_smoke.yaml")),
                "--video-path",
                "dummy",
                "--output-dir",
                str(output_dir),
                "--num-video-frames",
                "4",
                "--num-video-frames-thumbnail",
                "2",
                "--max-tiles-video",
                "3",
                "--gazing-ratio-tile",
                "0.25,0.125",
                "--task-loss-requirement-tile",
                "0.5",
                "--max-batch-size-autogaze",
                "2",
                "--no-progress",
            ]
        )
    )
    assert summary["metrics"]["num_video_frames"] == 4
    assert summary["metrics"]["num_video_frames_thumbnail"] == 2
    assert summary["metrics"]["max_tiles_video"] == 3
    assert summary["metrics"]["effective_gazing_ratio_tile"] == [0.25, 0.125]
    runtime = json.loads((output_dir / "autogaze" / "runtime_metadata.json").read_text(encoding="utf-8"))
    assert runtime["nvila_hd"]["num_video_frames"] == 4
    assert runtime["nvila_hd"]["max_batch_size_autogaze"] == 2


def test_infer_full_saves_nvila_hd_metrics_in_dummy_mode(tmp_path: Path) -> None:
    output_dir = tmp_path / "full_nvila_hd"
    summary = infer_full.run(
        infer_full.parse_args(
            [
                "--config",
                str(_cfg("nvila_hd_smoke.yaml")),
                "--video-path",
                "dummy",
                "--query-text",
                "Describe the video.",
                "--output-dir",
                str(output_dir),
                "--num-video-frames",
                "4",
                "--num-video-frames-thumbnail",
                "2",
                "--max-tiles-video",
                "3",
                "--gazing-ratio-tile",
                "0.25,0.125",
                "--max-batch-size-siglip",
                "6",
                "--allow-dummy-weights",
                "--no-progress",
            ]
        )
    )
    assert summary["metrics"]["num_video_frames"] == 4
    assert summary["metrics"]["num_video_frames_thumbnail"] == 2
    assert summary["metrics"]["max_batch_size_siglip"] == 6
    answer = json.loads((output_dir / "predictions" / "answer.json").read_text(encoding="utf-8"))
    assert answer["query_text"] == "Describe the video."
    assert answer["query_text_used"] is True


def test_infer_full_full_mode_comparison_visualizations(tmp_path: Path) -> None:
    cases = [
        ("infer_full_full_resize_short.yaml", "processed_frames", 16, None),
        ("infer_full_full_resize_then_chop_short.yaml", "merged_chop_source_frames", 16, 64),
        ("infer_full_full_resize_long.yaml", "processed_frames", 96, None),
        ("infer_full_full_resize_then_chop_long.yaml", "merged_chop_source_frames", 96, 384),
    ]
    for config_name, expected_mode, expected_frames, expected_crop_frames in cases:
        output_dir = tmp_path / config_name.removesuffix(".yaml")
        args = [
            "--config",
            str(_cfg(config_name)),
            "--video-path",
            "dummy",
            "--query-text",
            "Describe the video.",
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--mllm-dtype",
            "float32",
            "--resolution",
            "32",
            "--allow-dummy-weights",
            "--save-overlay-video",
            "--save-side-by-side-video",
            "--save-scale-panel-video",
            "--no-progress",
        ]
        if "resize_then_chop" in config_name:
            args.extend(["--chop-size", "32", "--max-chops", "4"])

        summary = infer_full.run(infer_full.parse_args(args))
        assert summary["status"] in {"partial", "ok"}

        metadata_path = output_dir / "visualizations" / "autogaze" / "metadata" / "visualization_metadata.json"
        viz = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert viz["visualization_mode"] == expected_mode
        assert viz["frame_count"] == expected_frames
        assert viz["rendered_frame_count"] == expected_frames
        assert viz["video_errors"] == {}
        if expected_crop_frames is not None:
            assert viz["processed_crop_frame_count"] == expected_crop_frames

        metrics = json.loads((output_dir / "logs" / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["autogaze_latency_includes_preprocessing"] is True
        assert metrics["autogaze_preprocessing_latency_ms"] == metrics["preprocessing_latency_ms"]
        assert metrics["autogaze_latency_ms"] >= metrics["autogaze_stage_latency_ms"] >= metrics["autogaze_result_build_latency_ms"] >= 0.0
        assert metrics["autogaze_latency_ms"] >= metrics["autogaze_preprocessing_latency_ms"] >= 0.0
        assert metrics["module_processing_latency_ms"] >= metrics["autogaze_latency_ms"]
        if expected_crop_frames is not None:
            assert metrics["number_of_processed_frames"] == expected_crop_frames
            assert metrics["mllm_input_frame_count"] == expected_crop_frames
            assert metrics["mllm_input_tensor_shape"][0] * metrics["mllm_input_tensor_shape"][1] == expected_crop_frames
        else:
            assert metrics["mllm_input_frame_count"] == expected_frames

        videos_dir = output_dir / "visualizations" / "autogaze" / "videos"
        assert (videos_dir / "autogaze_overlay.mp4").exists()
        assert (videos_dir / "autogaze_side_by_side.mp4").exists()
        assert (videos_dir / "autogaze_scale_panels.mp4").exists()


def test_a1_a2_share_nvila_processor_shape_settings_except_autogaze_flag() -> None:
    a1 = load_config(_cfg("A1_modified_siglip_nvila_off.yaml"))
    a2 = load_config(_cfg("A2_modified_siglip_nvila_on.yaml"))
    assert a1["autogaze"]["enabled"] is False
    assert a2["autogaze"]["enabled"] is True
    for key in ("num_video_frames", "num_video_frames_thumbnail", "max_batch_size_autogaze"):
        assert a1["mllm"]["processor_from_pretrained_kwargs"][key] == a2["mllm"]["processor_from_pretrained_kwargs"][key]
    assert a1["mllm"]["from_pretrained_kwargs"]["max_batch_size_siglip"] == a2["mllm"]["from_pretrained_kwargs"]["max_batch_size_siglip"]
