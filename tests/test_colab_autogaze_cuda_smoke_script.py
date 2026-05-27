from pathlib import Path

from scripts.run_colab_autogaze_cuda_smoke import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_WEIGHTS_ROOT,
    build_parser,
    compact_results,
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
