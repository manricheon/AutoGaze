from pathlib import Path

from repro.vjepa_qwen_hlvid_benchmark import build_parser, build_runner_args_for_row


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
    assert args.vjepa_selection_policies == "single_scale_union"


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
        policy="scale_aware_multi_pass",
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
    assert argv[argv.index("--vjepa-selection-policy") + 1] == "scale_aware_multi_pass"
    assert argv[argv.index("--autogaze-model") + 1] == "weights/AutoGaze"
    assert argv[argv.index("--num-video-frames") + 1] == "32"
