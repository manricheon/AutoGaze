import json
import os
from pathlib import Path

from repro.plugin_hlvid_benchmark import (
    _flatten_key_metrics,
    build_mode_runner_args,
    resolve_hlvid_video_path,
    run_plugin_hlvid_benchmark,
)


def test_resolve_hlvid_video_path_falls_back_to_basename_when_video_root_is_flat(tmp_path):
    video_root = tmp_path / "videos"
    video_root.mkdir()
    flat_video = video_root / "clip_001.mp4"
    flat_video.write_text("fake")

    resolved = resolve_hlvid_video_path(video_root, "nested/path/clip_001.mp4")

    assert resolved == flat_video


def test_build_mode_runner_args_for_nvila_video_off():
    row = {
        "video_path": "clip.mp4",
        "question": "Question? A. one B. two C. three D. four",
        "answer": "A",
    }

    args = build_mode_runner_args(
        mode="nvila-video-off",
        row=row,
        video_path=Path("/data/clip.mp4"),
        output_json=Path("/tmp/run.json"),
        models={
            "nvila-video": "weight/NVILA-8B-Video",
            "longvila": "weight/LongVILA",
            "internvl3": "weight/InternVL3",
            "qwen3-vl": "weight/Qwen3-VL",
        },
        external_mllm_command="/opt/vila/bin/vila-infer",
        num_video_frames=256,
        max_tiles_video=8,
        max_new_tokens=16,
    )

    assert args[:8] == [
        "--mode",
        "single",
        "--model-family",
        "nvila-video-plugin",
        "--model-path",
        "weight/NVILA-8B-Video",
        "--token-selector-adapter",
        "keep-all",
    ]
    assert "--external-mllm-command" in args
    assert "/opt/vila/bin/vila-infer" in args
    assert "Question? A. one B. two C. three D. four" in args


def test_build_mode_runner_args_for_qwen3_pixelprune_pre_vit():
    row = {
        "video_path": "clip.mp4",
        "question": "Question? A. one B. two C. three D. four",
        "answer": "A",
    }

    args = build_mode_runner_args(
        mode="qwen3-vl-pixelprune-pre-vit",
        row=row,
        video_path=Path("/data/clip.mp4"),
        output_json=Path("/tmp/run.json"),
        models={"qwen3-vl": "weight/Qwen3-VL"},
        external_mllm_command="vila-infer",
        num_video_frames=128,
        max_tiles_video=1,
        max_new_tokens=8,
    )

    assert "--model-family" in args
    assert args[args.index("--model-family") + 1] == "qwen3-vl"
    assert args[args.index("--autogaze-integration-level") + 1] == "pre_encoder_sparse"
    assert args[args.index("--pre-encoder-prune-adapter") + 1] == "pixelprune"
    assert args[args.index("--token-selector-adapter") + 1] == "keep-all"


def test_build_mode_runner_args_for_priority_autogaze_poc_modes():
    row = {
        "video_path": "clip.mp4",
        "question": "Question? A. one B. two C. three D. four",
        "answer": "A",
    }
    common = {
        "row": row,
        "video_path": Path("/data/clip.mp4"),
        "output_json": Path("/tmp/run.json"),
        "models": {
            "nvila-video": "weight/NVILA-8B-Video",
            "longvila": "weight/LongVILA",
            "qwen3-vl": "weight/Qwen3-VL",
        },
        "external_mllm_command": "vila-infer",
        "num_video_frames": 128,
        "max_tiles_video": 4,
        "max_new_tokens": 8,
    }

    nvila_args = build_mode_runner_args(mode="nvila-video-autogaze-probe", **common)
    longvila_args = build_mode_runner_args(mode="longvila-autogaze-probe", **common)
    qwen_args = build_mode_runner_args(mode="qwen3-vl-autogaze-poc", **common)

    assert nvila_args[nvila_args.index("--model-family") + 1] == "nvila-video-plugin"
    assert nvila_args[nvila_args.index("--token-selector-adapter") + 1] == "autogaze"
    assert nvila_args[nvila_args.index("--autogaze-integration-level") + 1] == "post_encoder_token_prune"
    assert longvila_args[longvila_args.index("--model-family") + 1] == "longvila"
    assert longvila_args[longvila_args.index("--token-selector-adapter") + 1] == "autogaze"
    assert qwen_args[qwen_args.index("--model-family") + 1] == "qwen3-vl"
    assert qwen_args[qwen_args.index("--gazing-ratio") + 1] == "0.1"


def test_build_mode_runner_args_for_qwen3_autogaze_prune_generate_mode():
    row = {
        "video_path": "clip.mp4",
        "question": "Question? A. one B. two C. three D. four",
        "answer": "A",
    }

    args = build_mode_runner_args(
        mode="qwen3-vl-autogaze-prune-generate",
        row=row,
        video_path=Path("/data/clip.mp4"),
        output_json=Path("/tmp/run.json"),
        models={"qwen3-vl": "weight/Qwen3-VL"},
        external_mllm_command="vila-infer",
        num_video_frames=128,
        max_tiles_video=1,
        max_new_tokens=8,
    )

    assert args[args.index("--model-family") + 1] == "qwen3-vl"
    assert args[args.index("--token-selector-adapter") + 1] == "autogaze"
    assert args[args.index("--autogaze-integration-level") + 1] == "post_encoder_token_prune"
    assert "--enable-qwen-prune-generate" in args


def test_build_mode_runner_args_for_qwen3_direct_autogaze_prune_generate_mode():
    row = {
        "video_path": "clip.mp4",
        "question": "Question? A. one B. two C. three D. four",
        "answer": "A",
    }

    args = build_mode_runner_args(
        mode="qwen3-vl-autogaze-direct-prune-generate",
        row=row,
        video_path=Path("/data/clip.mp4"),
        output_json=Path("/tmp/run.json"),
        models={"qwen3-vl": "weight/Qwen3-VL"},
        external_mllm_command="vila-infer",
        num_video_frames=128,
        max_tiles_video=1,
        max_new_tokens=8,
    )

    assert "--enable-qwen-prune-generate" in args
    assert "--run-autogaze-selector" in args
    assert "--autogaze-generate-only" in args


def test_build_mode_runner_args_for_qwen3_direct_autogaze_pre_vit_sparse_mode():
    row = {
        "video_path": "clip.mp4",
        "question": "Question? A. one B. two C. three D. four",
        "answer": "A",
    }

    args = build_mode_runner_args(
        mode="qwen3-vl-autogaze-direct-pre-vit-sparse",
        row=row,
        video_path=Path("/data/clip.mp4"),
        output_json=Path("/tmp/run.json"),
        models={"qwen3-vl": "weight/Qwen3-VL"},
        external_mllm_command="vila-infer",
        num_video_frames=128,
        max_tiles_video=1,
        max_new_tokens=8,
        qwen_video_max_pixels=200704,
    )

    assert args[args.index("--autogaze-integration-level") + 1] == "pre_encoder_sparse"
    assert args[args.index("--pre-encoder-prune-adapter") + 1] == "autogaze-sparse"
    assert "--enable-qwen-prune-generate" in args
    assert "--run-autogaze-selector" in args
    assert args[args.index("--qwen-video-max-pixels") + 1] == "200704"


def test_build_mode_runner_args_for_qwen_vit_comparison_modes():
    row = {
        "video_path": "clip.mp4",
        "question": "Question? A. one B. two C. three D. four",
        "answer": "A",
    }
    common = {
        "row": row,
        "video_path": Path("/data/clip.mp4"),
        "output_json": Path("/tmp/run.json"),
        "models": {"qwen3-vl": "weight/Qwen3-VL"},
        "external_mllm_command": "vila-infer",
        "num_video_frames": 128,
        "max_tiles_video": 1,
        "max_new_tokens": 8,
        "qwen_vit_chunk_frames": 16,
        "qwen_vit_max_spatial_chunks": 4,
    }

    full_args = build_mode_runner_args(mode="qwen_full_vit", **common)
    chunked_args = build_mode_runner_args(mode="qwen_chunked_vit", **common)
    sparse_args = build_mode_runner_args(mode="qwen_chunked_vit_autogaze_sparse", **common)

    assert full_args[full_args.index("--qwen-vit-mode") + 1] == "qwen_full_vit"
    assert full_args[full_args.index("--token-selector-adapter") + 1] == "keep-all"
    assert chunked_args[chunked_args.index("--qwen-vit-mode") + 1] == "qwen_chunked_vit"
    assert chunked_args[chunked_args.index("--qwen-vit-chunk-frames") + 1] == "16"
    assert chunked_args[chunked_args.index("--qwen-vit-max-spatial-chunks") + 1] == "4"
    assert sparse_args[sparse_args.index("--qwen-vit-mode") + 1] == "qwen_chunked_vit_autogaze_sparse"
    assert sparse_args[sparse_args.index("--pre-encoder-prune-adapter") + 1] == "autogaze-sparse"
    assert "--run-autogaze-selector" in sparse_args
    assert "--enable-qwen-prune-generate" in sparse_args


def test_flatten_key_metrics_includes_qwen_vit_comparison_fields():
    flattened = _flatten_key_metrics(
        {
            "latency_ms": {"total": 100.0, "generate": 40.0, "qwen_vit_prepare": 30.0},
            "memory_bytes": {"peak_cuda_allocated": 10, "peak_cuda_reserved": 20},
            "tokens": {
                "visual_tokens_before_prune": 1000,
                "visual_tokens_after_prune": 100,
                "visual_token_reduction_ratio": 10.0,
                "llm_context_tokens": 120,
            },
            "qwen_vit": {
                "mode": "qwen_chunked_vit_autogaze_sparse",
                "raw_patch_tokens_before_vit": 4000,
                "chunk_count": 4,
                "executed_chunk_count": 3,
                "spatial_chunking": {"tile_grid": {"tiles": 4}},
            },
        }
    )

    assert flattened["qwen_vit_prepare_ms"] == 30.0
    assert flattened["qwen_vit_mode"] == "qwen_chunked_vit_autogaze_sparse"
    assert flattened["visual_tokens_before_prune"] == 1000
    assert flattened["visual_tokens_after_prune"] == 100
    assert flattened["visual_token_reduction_ratio"] == 10.0
    assert flattened["qwen_vit_raw_patch_tokens_before_vit"] == 4000
    assert flattened["qwen_vit_executed_chunk_count"] == 3
    assert flattened["qwen_vit_spatial_tiles"] == 4


def test_run_plugin_hlvid_benchmark_writes_predictions_summary_and_markdown(tmp_path):
    manifest = tmp_path / "manifest.json"
    video_root = tmp_path / "videos"
    output_dir = tmp_path / "out"
    video_root.mkdir()
    (video_root / "clip_001.mp4").write_text("fake video")
    manifest.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "category": "toy",
                    "video_path": "nested/clip_001.mp4",
                    "question": "Question? A. one B. two C. three D. four",
                    "answer": "B",
                }
            ]
        )
    )
    fake_runner = tmp_path / "fake-vila-infer"
    fake_runner.write_text("#!/bin/sh\necho 'Assistant: B'\n")
    os.chmod(fake_runner, 0o755)

    payload = run_plugin_hlvid_benchmark(
        manifest=manifest,
        video_root=video_root,
        output_dir=output_dir,
        modes=["nvila-video-off"],
        models={"nvila-video": "weight/NVILA-8B-Video"},
        external_mllm_command=str(fake_runner),
        limit=3,
        num_video_frames=8,
        max_tiles_video=1,
        max_new_tokens=4,
    )

    assert payload["summary"]["modes"]["nvila-video-off"]["correct"] == 1
    assert payload["summary"]["modes"]["nvila-video-off"]["accuracy_total"] == 1.0
    assert (output_dir / "plugin_hlvid_predictions.jsonl").is_file()
    assert (output_dir / "plugin_hlvid_summary.json").is_file()
    report = (output_dir / "plugin_hlvid_report.md").read_text()
    assert "nvila-video-off" in report
    assert "accuracy_total" in report


def test_plugin_hlvid_summary_reports_probe_and_poc_statuses(tmp_path):
    manifest = tmp_path / "manifest.json"
    video_root = tmp_path / "videos"
    output_dir = tmp_path / "out"
    nvila_model = tmp_path / "NVILA-8B-Video"
    longvila_model = tmp_path / "LongVILA"
    video_root.mkdir()
    nvila_model.mkdir()
    longvila_model.mkdir()
    (nvila_model / "config.json").write_text('{"model_type": "llava", "vision_tower": "siglip"}')
    (longvila_model / "config.json").write_text('{"model_type": "llava", "vision_tower": "siglip"}')
    (video_root / "clip_001.mp4").write_text("fake video")
    manifest.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "category": "toy",
                    "video_path": "clip_001.mp4",
                    "question": "Question? A. one B. two C. three D. four",
                    "answer": "B",
                }
            ]
        )
    )

    payload = run_plugin_hlvid_benchmark(
        manifest=manifest,
        video_root=video_root,
        output_dir=output_dir,
        modes=["nvila-video-autogaze-probe", "longvila-autogaze-probe", "qwen3-vl-autogaze-poc"],
        models={
            "nvila-video": str(nvila_model),
            "longvila": str(longvila_model),
            "qwen3-vl": "weight/Qwen3-VL",
        },
        limit=1,
        num_video_frames=8,
        max_tiles_video=1,
        max_new_tokens=4,
    )

    summary = payload["summary"]["modes"]
    assert summary["nvila-video-autogaze-probe"]["status_counts"] == {"probe_collected": 1}
    assert summary["longvila-autogaze-probe"]["status_counts"] == {"probe_collected": 1}
    assert summary["qwen3-vl-autogaze-poc"]["status_counts"] == {"poc_ready": 1}
    assert summary["nvila-video-autogaze-probe"]["next_action"] == "instrument_vila_remote_code_feature_packing"
    assert summary["longvila-autogaze-probe"]["next_action"] == "instrument_vila_remote_code_feature_packing"
    assert summary["qwen3-vl-autogaze-poc"]["next_action"] == "implement_qwen_visual_feature_prune_generate"
    report = (output_dir / "plugin_hlvid_report.md").read_text()
    assert "status_counts" in report
    assert "instrument_vila_remote_code_feature_packing" in report
    assert "implement_qwen_visual_feature_prune_generate" in report
