from pathlib import Path

from scripts.run_colab_autogaze_cuda_smoke import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_WEIGHTS_ROOT,
    build_parser,
    compact_results,
    render_colab_verification_markdown,
    result_failures,
    run_smoke,
    vjepa_qwen_command,
)


def test_colab_cuda_smoke_defaults_are_cuda_actual_smoke_oriented():
    args = build_parser().parse_args([])

    assert args.output_root == DEFAULT_OUTPUT_ROOT
    assert args.weights_root == DEFAULT_WEIGHTS_ROOT
    assert args.require_cuda is True
    assert args.run_entrypoint_verifier is True
    assert args.run_vjepa_qwen is True
    assert args.run_dense_off is True
    assert args.run_autogaze_on is True
    assert args.autogaze_target_scales == "32+64+112+224"


def test_colab_cuda_smoke_dry_run_builds_verifier_dense_and_autogaze_commands(tmp_path):
    args = build_parser().parse_args(
        [
            "--dry-run",
            "--python-executable",
            "python",
            "--repo-root",
            str(tmp_path),
            "--output-root",
            str(tmp_path / "outputs"),
            "--weights-root",
            str(tmp_path / "weights"),
            "--video",
            "inputs/example.mp4",
        ]
    )

    payload = run_smoke(args)
    names = [item["name"] for item in payload["command_plan"]]

    assert payload["summary"]["passed"] is True
    assert names == [
        "download_missing_weights",
        "download_missing_video",
        "entrypoint_verifier",
        "vjepa_qwen_dense_off",
        "autogaze_vjepa_qwen_on",
    ]


def test_vjepa_qwen_dense_command_uses_off_without_autogaze_model(tmp_path):
    args = build_parser().parse_args(
        [
            "--python-executable",
            "python",
            "--weights-root",
            str(tmp_path / "weights"),
            "--video",
            "inputs/example.mp4",
        ]
    )

    command = vjepa_qwen_command(
        args,
        mode="off",
        output_json=Path("dense.json"),
        output_md=Path("dense.md"),
    )

    assert "--autogaze-mode" in command
    assert command[command.index("--autogaze-mode") + 1] == "off"
    assert "--autogaze-model" not in command
    assert "--require-cuda" in command


def test_vjepa_qwen_autogaze_command_forwards_selector_options(tmp_path):
    args = build_parser().parse_args(
        [
            "--python-executable",
            "python",
            "--weights-root",
            str(tmp_path / "weights"),
            "--video",
            "inputs/example.mp4",
            "--autogaze-target-scales",
            "32+64+112+224",
            "--max-batch-size-autogaze",
            "4",
        ]
    )

    command = vjepa_qwen_command(
        args,
        mode="on",
        output_json=Path("ag.json"),
        output_md=Path("ag.md"),
    )

    assert command[command.index("--autogaze-mode") + 1] == "on"
    assert command[command.index("--autogaze-model") + 1].endswith("nvidia__AutoGaze")
    assert command[command.index("--autogaze-target-scales") + 1] == "32+64+112+224"
    assert command[command.index("--max-batch-size-autogaze") + 1] == "4"


def test_result_failures_distinguishes_missing_verifier_and_pipeline_failures():
    failures = result_failures(
        {
            "entrypoint_verifier": {"summary": {"passed": False}},
            "vjepa_qwen_dense_off": None,
            "autogaze_vjepa_qwen_on": {"status": "failed", "failure": {"stage": "qwen_generate"}},
        }
    )

    assert {item["kind"] for item in failures} == {
        "verification_failed",
        "missing_result",
        "pipeline_failed",
    }


def test_compact_results_exposes_entrypoint_verifier_script_matrix():
    compact = compact_results(
        {
            "entrypoint_verifier": {
                "summary": {"passed": True, "command_count": 2, "check_count": 1},
                "script_matrix": [
                    {
                        "id": "nvila_single_autogaze",
                        "entrypoint": "python -m repro.nvila_runner --mode single --gazing-mode autogaze",
                        "selector": "AutoGaze on",
                        "vit": "NVILA-HD SigLIP",
                        "mllm": "NVILA-HD",
                    },
                    {
                        "id": "vjepa_qwen_hlvid",
                        "entrypoint": "python -m repro.vjepa_qwen_hlvid_benchmark",
                        "selector": "dense_off, autogaze_single_grid",
                        "vit": "V-JEPA2",
                        "mllm": "Qwen bridge/generate",
                    },
                ],
                "commands": [
                    {"name": "download_qwen_dry_run"},
                    {"name": "vjepa_qwen_hlvid_dry_run"},
                    {"name": "nvila_runner_help"},
                ],
            }
        }
    )

    verifier = compact["entrypoint_verifier"]
    assert verifier["status"] is True
    assert verifier["verified_script_ids"] == ["nvila_single_autogaze", "vjepa_qwen_hlvid"]
    assert verifier["dry_run_commands"] == ["download_qwen_dry_run", "vjepa_qwen_hlvid_dry_run"]


def test_render_colab_verification_markdown_compares_answers_tokens_and_visuals(tmp_path):
    payload = {
        "summary": {"passed": True, "command_count": 3, "failed_count": 0, "elapsed_ms": 1234.0},
        "paths": {
            "repo_root": "/content/AutoGaze",
            "output_root": str(tmp_path),
            "weights_root": "/content/autogaze_weights",
            "video": "inputs/example.mp4",
        },
        "prompt": "What is happening?",
        "results": {
            "vjepa_qwen_dense_off": {
                "status": "passed",
                "autogaze_mode": "off",
                "generated_text": "A person is cooking.",
                "tokens": {
                    "vjepa_raw_tokens": 1568,
                    "vjepa_selected_tokens": 1568,
                    "qwen_visual_tokens_inserted": 1568,
                },
                "latency_ms": {"total": 100.0, "qwen_generate": 20.0},
                "memory_bytes": {"cuda_peak_total": 1024},
                "visualizations": {
                    "selected_frames_grid_image": str(tmp_path / "dense_frames.png"),
                    "vjepa_token_mask_image": str(tmp_path / "dense_mask.png"),
                },
            },
            "autogaze_vjepa_qwen_on": {
                "status": "passed",
                "autogaze_mode": "on",
                "generated_text": "A person prepares food.",
                "tokens": {
                    "autogaze_raw_patch_tokens": 4240,
                    "autogaze_selected_patch_tokens": 16,
                    "vjepa_raw_tokens": 1568,
                    "vjepa_selected_tokens": 8,
                    "qwen_visual_tokens_inserted": 8,
                },
                "latency_ms": {"total": 120.0, "autogaze_selector_total": 10.0, "qwen_generate": 5.0},
                "memory_bytes": {"cuda_peak_total": 2048},
                "visualizations": {
                    "selected_frames_grid_image": str(tmp_path / "ag_frames.png"),
                    "vjepa_token_mask_image": str(tmp_path / "ag_mask.png"),
                    "autogaze_overlay_image": str(tmp_path / "ag_overlay.png"),
                },
            },
            "entrypoint_verifier": {
                "status": True,
                "summary": {"passed": True, "command_count": 18, "check_count": 26},
                "verified_script_ids": ["nvila_single_autogaze", "vjepa_qwen_single"],
            },
        },
    }

    markdown = render_colab_verification_markdown(payload, output_md=tmp_path / "colab_verification.md")

    assert "# Colab Verification" in markdown
    assert "What is happening?" in markdown
    assert "A person is cooking." in markdown
    assert "A person prepares food." in markdown
    assert "1568" in markdown
    assert "8" in markdown
    assert "ag_overlay.png" in markdown
    assert "nvila_single_autogaze" in markdown
