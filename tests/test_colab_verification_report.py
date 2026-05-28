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
    assert "## V2 Pipeline Evidence Matrix" in markdown


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


def test_colab_report_normalizes_nvila_single_payload_with_nested_visualization(tmp_path):
    selected = tmp_path / "nvila_selected.mp4"
    overlay = tmp_path / "nvila_overlay.mp4"
    case_json = tmp_path / "nvila.json"
    case_json.write_text(
        json.dumps(
            {
                "gazing_mode": "autogaze",
                "summary": {
                    "answer": "The road sign says Hampden Ave.",
                    "tokens": {
                        "encoder_raw_patch_tokens": 33920,
                        "encoder_selected_patch_tokens": 17024,
                        "llm_actual_visual_tokens": 1904,
                    },
                    "latency_ms": {
                        "total_median": 10828.0,
                        "autogaze_total_median": 1591.9,
                        "vision_encoder_median": 1831.84,
                    },
                    "memory_bytes": {"overall_peak_median": 8428875776},
                },
                "result": {
                    "visualization": {
                        "selected_frames_video": str(selected),
                        "overlay_video": str(overlay),
                        "sampled_frame_count": 16,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    payload = build_colab_verification_payload(
        title="V2",
        cases=[("nvila_hd_autogaze_single", case_json)],
        output_md=tmp_path / "report.md",
    )
    result = payload["results"]["nvila_hd_autogaze_single"]
    markdown = render_colab_verification_markdown(payload, output_md=tmp_path / "report.md")

    assert result["status"] == "passed"
    assert result["generated_text"] == "The road sign says Hampden Ave."
    assert result["tokens"]["encoder_raw_patch_tokens"] == 33920
    assert result["latency_ms"]["vision_encoder_median"] == 1831.84
    assert result["visualizations"]["selected_frames_video"] == str(selected)
    assert "NVILA-HD" in markdown
    assert "nvila_selected.mp4" in markdown
    assert "sampled_frame_count" in markdown


def test_colab_report_normalizes_qwen_plugin_run_payload_and_sparse_plan(tmp_path):
    plan = tmp_path / "qwen_sparse_plan.json"
    case_json = tmp_path / "qwen_run.json"
    case_json.write_text(
        json.dumps(
            {
                "implementation_status": "executed",
                "direct_autogaze_selector": {
                    "status": "executed",
                    "sparse_selection_plan_json": str(plan),
                },
                "generation": {
                    "status": "executed",
                    "text": "A",
                    "metrics": {
                        "tokens": {
                            "visual_tokens_before_prune": 256,
                            "visual_tokens_after_prune": 140,
                            "qwen_context_tokens": 209,
                        },
                        "latency_ms": {
                            "total": 12722.54,
                            "qwen_vit_prepare": 183.13,
                            "generate": 251.05,
                        },
                        "memory_bytes": {"peak_cuda_allocated": 1234},
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    payload = build_colab_verification_payload(
        title="V2",
        cases=[("qwen_chunked_vit_autogaze_sparse", case_json)],
        output_md=tmp_path / "report.md",
    )
    result = payload["results"]["qwen_chunked_vit_autogaze_sparse"]
    markdown = render_colab_verification_markdown(payload, output_md=tmp_path / "report.md")

    assert result["status"] == "executed"
    assert result["generated_text"] == "A"
    assert result["tokens"]["visual_tokens_after_prune"] == 140
    assert result["latency_ms"]["qwen_vit_prepare"] == 183.13
    assert result["visualizations"]["sparse_selection_plan_json"] == str(plan)
    assert "Qwen" in markdown
    assert "qwen_sparse_plan.json" in markdown
