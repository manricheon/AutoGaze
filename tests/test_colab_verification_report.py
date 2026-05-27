import json
from pathlib import Path

from repro.colab_verification_report import (
    build_colab_verification_payload,
    main,
    parse_case_arg,
    render_colab_verification_markdown,
)


def test_parse_case_arg_requires_name_and_path():
    name, path = parse_case_arg("nvila_autogaze=/tmp/nvila.json")

    assert name == "nvila_autogaze"
    assert path == Path("/tmp/nvila.json")


def test_build_payload_reads_direct_runner_jsons_and_missing_artifacts(tmp_path):
    dense_json = tmp_path / "dense.json"
    dense_json.write_text(
        json.dumps(
            {
                "status": "passed",
                "generated_text": "A person is cooking.",
                "tokens": {
                    "vjepa_raw_tokens": 1568,
                    "vjepa_selected_tokens": 1568,
                    "qwen_visual_tokens_inserted": 1568,
                },
                "latency_ms": {"total": 100.0, "qwen_generate": 20.0},
                "memory_bytes": {"cuda_peak_total": 1024},
                "visualizations": {"selected_frames_grid_image": str(tmp_path / "dense_frames.png")},
            }
        ),
        encoding="utf-8",
    )
    autogaze_json = tmp_path / "autogaze.json"
    autogaze_json.write_text(
        json.dumps(
            {
                "summary": {"passed": True},
                "result": {"generated_text": "A person prepares food."},
                "token_metrics": {
                    "autogaze_raw_patch_tokens": 4240,
                    "autogaze_selected_patch_tokens": 16,
                    "vjepa_raw_tokens": 1568,
                    "vjepa_selected_tokens": 8,
                    "qwen_visual_tokens_inserted": 8,
                },
                "latency_ms": {"total": 120.0, "autogaze_selector_total": 10.0, "qwen_generate": 5.0},
                "memory": {"cuda_peak_total": 2048},
                "visualizations": {
                    "autogaze_overlay_image": str(tmp_path / "autogaze_overlay.png"),
                    "vjepa_token_mask_image": str(tmp_path / "autogaze_mask.png"),
                },
            }
        ),
        encoding="utf-8",
    )
    verifier_json = tmp_path / "verifier.json"
    verifier_json.write_text(
        json.dumps(
            {
                "summary": {"passed": True, "command_count": 18, "check_count": 26},
                "script_matrix": [{"id": "nvila_single_autogaze"}, {"id": "vjepa_qwen_single"}],
                "commands": [{"name": "vjepa_qwen_hlvid_dry_run"}],
            }
        ),
        encoding="utf-8",
    )

    payload = build_colab_verification_payload(
        title="Direct Colab Check",
        video="inputs/example.mp4",
        query="What is happening?",
        cases=[
            ("vjepa_qwen_dense_off", dense_json),
            ("autogaze_vjepa_qwen_on", autogaze_json),
            ("missing_case", tmp_path / "missing.json"),
        ],
        entrypoint_verification_json=verifier_json,
        output_md=tmp_path / "colab_verification.md",
    )

    markdown = render_colab_verification_markdown(payload, output_md=tmp_path / "colab_verification.md")

    assert "Direct Colab Check" in markdown
    assert "What is happening?" in markdown
    assert "A person is cooking." in markdown
    assert "A person prepares food." in markdown
    assert "vjepa_qwen_dense_off" in markdown
    assert "autogaze_vjepa_qwen_on" in markdown
    assert "missing_case" in markdown
    assert "missing_artifact" in markdown
    assert "4240" in markdown
    assert "8" in markdown
    assert "autogaze_overlay.png" in markdown
    assert "nvila_single_autogaze" in markdown


def test_colab_verification_report_cli_writes_markdown(tmp_path):
    case_json = tmp_path / "case.json"
    case_json.write_text(
        json.dumps(
            {
                "status": "passed",
                "answer": "A vehicle moves down the road.",
                "tokens": {"encoder_input_tokens": 128, "llm_visual_tokens": 128},
                "latency_ms": {"total": 33.0},
            }
        ),
        encoding="utf-8",
    )
    output_md = tmp_path / "report.md"

    main(
        [
            "--output-md",
            str(output_md),
            "--title",
            "CLI Report",
            "--video",
            "inputs/example.mp4",
            "--query",
            "Describe the video.",
            "--case",
            f"nvila_keep_all_single={case_json}",
        ]
    )

    markdown = output_md.read_text(encoding="utf-8")
    assert "# CLI Report" in markdown
    assert "nvila_keep_all_single" in markdown
    assert "A vehicle moves down the road." in markdown
