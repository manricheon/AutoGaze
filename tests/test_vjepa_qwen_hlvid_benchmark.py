from pathlib import Path

import json

from repro.vjepa_qwen_hlvid_benchmark import build_parser, build_runner_args_for_row, resolve_modes, run_benchmark


def test_vjepa_qwen_hlvid_defaults_are_smoke_safe():
    args = build_parser().parse_args(
        [
            "--manifest",
            "data/hlvid/test.jsonl",
            "--video-root",
            "data/hlvid/videos",
            "--output-dir",
            "outputs/vjepa_qwen_hlvid",
        ]
    )

    assert args.limit is None
    assert args.continue_on_error is True
    assert args.num_video_frames == 16
    assert args.frames_per_clip == 16
    assert args.autogaze_tile_size == 224
    assert args.autogaze_target_scales == "32+64+112+224"
    assert args.vjepa_qwen_modes == ""
    assert args.vjepa_selection_policies == "single_scale_union"
    assert args.visualization_max_frames == 16
    assert resolve_modes(args) == ["autogaze_single_grid"]


def test_build_runner_args_for_row_uses_question_as_prompt(tmp_path):
    row = {
        "question_id": "q1",
        "video_path": "clip.mp4",
        "question": "What is happening?",
        "answer": "A",
    }
    output_json = tmp_path / "run.json"

    argv = build_runner_args_for_row(
        row=row,
        video_path=Path("/videos/clip.mp4"),
        output_json=output_json,
        mode="autogaze_scale_aware",
        benchmark_args=build_parser().parse_args(
            [
                "--manifest",
                "manifest.jsonl",
                "--video-root",
                "/videos",
                "--output-dir",
                str(tmp_path),
                "--autogaze-model",
                "weights/AutoGaze",
                "--vjepa-model",
                "weights/VJEPA",
                "--qwen-model",
                "weights/Qwen",
                "--device",
                "cuda",
                "--num-video-frames",
                "32",
                "--frames-per-clip",
                "32",
            ]
        ),
    )

    assert argv[argv.index("--video") + 1] == "/videos/clip.mp4"
    assert argv[argv.index("--prompt") + 1] == "What is happening?"
    assert argv[argv.index("--autogaze-mode") + 1] == "on"
    assert argv[argv.index("--vjepa-selection-policy") + 1] == "scale_aware_multi_pass"
    assert argv[argv.index("--autogaze-model") + 1] == "weights/AutoGaze"
    assert argv[argv.index("--num-video-frames") + 1] == "32"
    assert argv[argv.index("--visualization-max-frames") + 1] == "16"


def test_build_runner_args_for_dense_off_disables_autogaze(tmp_path):
    row = {
        "question_id": "q1",
        "video_path": "clip.mp4",
        "question": "What is happening?",
        "answer": "A",
    }
    output_json = tmp_path / "run.json"

    argv = build_runner_args_for_row(
        row=row,
        video_path=Path("/videos/clip.mp4"),
        output_json=output_json,
        mode="dense_off",
        benchmark_args=build_parser().parse_args(
            [
                "--manifest",
                "manifest.jsonl",
                "--video-root",
                "/videos",
                "--output-dir",
                str(tmp_path),
            ]
        ),
    )

    assert argv[argv.index("--autogaze-mode") + 1] == "off"
    assert argv[argv.index("--vjepa-selection-policy") + 1] == "single_scale_union"


def test_vjepa_qwen_hlvid_dry_run_writes_dense_and_autogaze_plan(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    video_root = tmp_path / "videos"
    video_root.mkdir()
    (video_root / "clip.mp4").write_bytes(b"fake")
    manifest.write_text(
        json.dumps(
                {
                    "question_id": "q1",
                    "category": "smoke",
                    "video_path": "clip.mp4",
                    "question": "What happens?",
                "answer": "A",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    args = build_parser().parse_args(
        [
            "--manifest",
            str(manifest),
            "--video-root",
            str(video_root),
            "--output-dir",
            str(tmp_path / "out"),
            "--dry-run",
            "--vjepa-qwen-modes",
            "dense_off,autogaze_single_grid",
            "--autogaze-model",
            "weights/AutoGaze",
            "--vjepa-model",
            "weights/VJEPA",
            "--qwen-model",
            "weights/Qwen",
            "--require-cuda",
        ]
    )

    payload = run_benchmark(args)
    plan_path = Path(payload["artifacts"]["dry_run_plan"])
    modes = [row["mode"] for row in payload["plan"]]

    assert payload["summary"]["dry_run"] is True
    assert payload["summary"]["planned_run_count"] == 2
    assert modes == ["dense_off", "autogaze_single_grid"]
    assert payload["plan"][0]["autogaze_mode"] == "off"
    assert payload["plan"][1]["autogaze_mode"] == "on"
    assert payload["plan"][1]["requires_cuda"] is True
    assert plan_path.exists()
