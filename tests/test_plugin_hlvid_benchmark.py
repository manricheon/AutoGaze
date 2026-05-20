import json
import os
from pathlib import Path

from repro.plugin_hlvid_benchmark import (
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
