import csv
import json

from repro.aggregate_reports import aggregate_report_roots, normalize_report_file


def test_normalize_single_report_extracts_latency_tokens_memory_and_failure(tmp_path):
    report = tmp_path / "failed.json"
    report.write_text(
        json.dumps(
            {
                "runner": "flexible_runner",
                "mode": "single",
                "model_path": "weight/Qwen3-VL",
                "failure": {
                    "kind": "oom",
                    "stage": "qwen_vit_prepare",
                    "exception_type": "OutOfMemoryError",
                    "message": "CUDA out of memory",
                },
                "generation": {
                    "status": "oom",
                    "metrics": {
                        "latency_ms": {"total": 1200, "qwen_vit_prepare": 700},
                        "tokens": {
                            "visual_tokens_before_prune": 1000,
                            "visual_tokens_after_prune": 100,
                            "visual_token_reduction_ratio": 10,
                            "llm_context_tokens": 220,
                        },
                        "memory_bytes": {"peak_cuda_allocated": 80_000_000_000},
                    },
                },
            }
        )
    )

    rows = normalize_report_file(report)

    assert rows == [
        {
            "source_path": str(report),
            "report_kind": "single_inference",
            "mode": "single",
            "model_path": "weight/Qwen3-VL",
            "status": "oom",
            "oom": True,
            "oom_stage": "qwen_vit_prepare",
            "failure_kind": "oom",
            "failure_message": "CUDA out of memory",
            "total_ms": 1200.0,
            "preprocess_ms": None,
            "autogaze_ms": None,
            "vision_encoder_ms": 700.0,
            "llm_ms": None,
            "single_scale_dense_patch_tokens": None,
            "full_or_raw_patch_tokens": 1000.0,
            "autogaze_selected_patch_tokens": 100.0,
            "llm_visual_tokens": 220.0,
            "token_reduction_ratio": 10.0,
            "peak_memory_bytes": 80000000000.0,
            "accuracy_total": None,
            "accuracy_scored": None,
            "failed": None,
            "parse_failed": None,
            "frames": None,
            "thumbnail_frames": None,
            "source_resolution": None,
            "processor_input_resolution": None,
            "max_tiles_video": None,
            "gazing_mode": None,
        }
    ]


def test_aggregate_report_roots_writes_markdown_csv_json_and_svg(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    (root / "ok.json").write_text(
        json.dumps(
            {
                "key_metrics_summary": {
                    "latency_ms": {"total_median": 9000, "autogaze_total_median": 800},
                    "tokens": {
                        "single_scale_dense_siglip_reference_patch_tokens": 800,
                        "hd_multiscale_keep_all_patch_tokens": 900,
                        "autogaze_selected_total_patch_tokens": 300,
                        "llm_visual_tokens_actual_from_budget": 40,
                    },
                    "memory_bytes": {"overall_peak_median": 4_000_000_000},
                },
                "processing_budget_summary": {
                    "video": {
                        "requested_video_frames": 128,
                        "processor_input_resolution": "1280x720",
                    },
                    "tiling": {"spatial_tiles_per_frame": 8},
                },
            }
        )
    )
    out = tmp_path / "trend"

    artifacts = aggregate_report_roots([root], out)

    assert artifacts["csv"].is_file()
    assert artifacts["json"].is_file()
    assert artifacts["markdown"].is_file()
    assert (out / "assets" / "latency_by_config.svg").is_file()
    assert (out / "assets" / "token_reduction_by_config.svg").is_file()
    assert (out / "assets" / "memory_peak_by_config.svg").is_file()
    assert (out / "assets" / "status_by_config.svg").is_file()
    markdown = artifacts["markdown"].read_text()
    assert "# AutoGaze Experiment Trend Report" in markdown
    assert "Latency By Config" in markdown
    with artifacts["csv"].open() as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["processor_input_resolution"] == "1280x720"
    assert rows[0]["autogaze_selected_patch_tokens"] == "300.0"
