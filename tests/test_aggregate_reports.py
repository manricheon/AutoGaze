import csv
import json

from repro.aggregate_reports import aggregate_report_roots, normalize_report_file, sort_rows


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
                        "latency_ms": {
                            "total": 1200,
                            "preprocess_without_autogaze_ms": 1000,
                            "video_decode_ms": 300,
                            "preprocess_total_median": 300,
                            "qwen_vit_prepare": 700,
                        },
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
            "comparison_pair": None,
            "baseline_mode": None,
            "candidate_mode": None,
            "model_path": "weight/Qwen3-VL",
            "status": "oom",
            "integration_level": None,
            "execution_claim": None,
            "actual_pruning_applied": None,
            "vit_latency_reduction_claim": None,
            "mllm_context_reduction_claim": None,
            "oom": True,
            "oom_stage": "qwen_vit_prepare",
            "failure_kind": "oom",
            "failure_message": "CUDA out of memory",
                "total_ms": 1200.0,
                "latency_speedup": None,
                "preprocess_ms": 1000.0,
                "video_decode_read_ms": 300.0,
                "video_prepare_total_ms": None,
                "video_frame_resize_ms": None,
                "video_tiling_ms": None,
                "selector_input_build_ms": None,
                "preprocess_rest_ms": 700.0,
                "autogaze_ms": None,
            "vision_encoder_ms": 700.0,
            "mm_projector_ms": None,
            "generate_ms": None,
            "llm_generation_ms": None,
            "llm_forward_ms": None,
            "generation_rest_ms": None,
            "llm_ms": None,
            "single_scale_dense_patch_tokens": None,
            "full_or_raw_patch_tokens": 1000.0,
            "autogaze_selected_patch_tokens": 100.0,
            "llm_visual_tokens": 220.0,
            "token_reduction_ratio": 10.0,
            "llm_visual_token_reduction_ratio": None,
            "memory_reduction_ratio": None,
            "accuracy_total_delta": None,
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
                    "latency_ms": {
                        "total_median": 9000,
                        "video_decode_read_ms": 700,
                        "video_frame_resize_ms": 300,
                        "video_tiling_ms": 1200,
                        "selector_input_build_ms": 100,
                        "autogaze_total_median": 800,
                        "vit_encoder_median": 1200,
                        "mm_projector_ms": 200,
                        "llm_median": 4000,
                    },
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
    assert (out / "assets" / "latency_attribution_by_config.svg").is_file()
    assert (out / "assets" / "token_reduction_by_config.svg").is_file()
    assert (out / "assets" / "memory_peak_by_config.svg").is_file()
    assert (out / "assets" / "status_by_config.svg").is_file()
    markdown = artifacts["markdown"].read_text()
    assert "# AutoGaze Experiment Trend Report" in markdown
    assert "Latency By Config" in markdown
    assert "Latency Attribution By Config" in markdown
    with artifacts["csv"].open() as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["processor_input_resolution"] == "1280x720"
    assert rows[0]["autogaze_selected_patch_tokens"] == "300.0"


def test_sort_rows_comparison_keeps_baseline_before_autogaze_and_failures_last():
    rows = [
        {"mode": "autogaze", "status": "ok", "frames": 128, "processor_input_resolution": "720p", "total_ms": 7000},
        {"mode": "keep_all", "status": "ok", "frames": 128, "processor_input_resolution": "720p", "total_ms": 9000},
        {
            "mode": "single_scale_dense",
            "status": "ok",
            "frames": 128,
            "processor_input_resolution": "720p",
            "total_ms": 8000,
        },
        {"mode": "qwen_sparse", "status": "ok", "frames": 128, "processor_input_resolution": "720p", "total_ms": 6000},
        {"mode": "keep_all", "status": "oom", "frames": 256, "processor_input_resolution": "1080p", "total_ms": None},
    ]

    sorted_rows = sort_rows(rows, sort="comparison")

    assert [row["mode"] for row in sorted_rows] == [
        "keep_all",
        "single_scale_dense",
        "autogaze",
        "qwen_sparse",
        "keep_all",
    ]
    assert sorted_rows[-1]["status"] == "oom"


def test_normalize_hlvid_gain_report_includes_single_scale_dense_mode(tmp_path):
    report = tmp_path / "hlvid_gain_report.json"
    report.write_text(
        json.dumps(
            {
                "keep_all": {"accuracy": {"accuracy_scored": 0.4, "failed": 0, "parse_failed": 0}},
                "single_scale_dense": {
                    "accuracy": {"accuracy_scored": 0.6, "failed": 0, "parse_failed": 0}
                },
                "autogaze": {"accuracy": {"accuracy_scored": 0.8, "failed": 0, "parse_failed": 0}},
                "readable_summary": {
                    "key_metrics_median": {
                        "latency_ms": {
                            "total_ms": {
                                "keep_all": 10000,
                                "single_scale_dense": 8000,
                                "autogaze": 6000,
                            }
                        },
                        "tokens": {
                            "vit_encoder_input_patch_tokens_before_autogaze": {
                                "keep_all": 1060,
                                "single_scale_dense": 784,
                                "autogaze": 1060,
                            },
                            "vit_encoder_input_patch_tokens_after_autogaze": {
                                "keep_all": 1060,
                                "single_scale_dense": 784,
                                "autogaze": 212,
                            },
                            "patch_reduction_ratio_full_or_raw_over_autogaze": {
                                "keep_all": 1,
                                "single_scale_dense": 1,
                                "autogaze": 5,
                            },
                        },
                        "memory_bytes": {},
                    }
                },
            }
        )
    )

    rows = normalize_report_file(report)

    assert [row["mode"] for row in rows] == ["keep_all", "single_scale_dense", "autogaze"]
    single_scale = rows[1]
    assert single_scale["accuracy_scored"] == 0.6
    assert single_scale["total_ms"] == 8000.0
    assert single_scale["full_or_raw_patch_tokens"] == 784.0
    assert single_scale["autogaze_selected_patch_tokens"] == 784.0
    assert single_scale["token_reduction_ratio"] == 1.0


def test_normalize_video_task_summary_reports_task_kind_and_scores(tmp_path):
    report = tmp_path / "action_classification_summary.json"
    report.write_text(
        json.dumps(
            {
                "task_type": "action_classification",
                "modes": {
                    "qwen3_full_vit": {
                        "total": 2,
                        "correct": 1,
                        "failed": 0,
                        "parse_failed": 0,
                        "accuracy_total": 0.5,
                        "accuracy_scored": 0.5,
                        "latency_ms": {"median": 123.0},
                        "peak_memory_bytes": {"median": 4096.0},
                        "visual_tokens_before_prune": {"median": 100.0},
                        "visual_tokens_after_prune": {"median": 40.0},
                        "status_counts": {"executed": 2},
                    }
                },
            }
        )
    )

    rows = normalize_report_file(report)

    assert rows[0]["report_kind"] == "video_task_summary"
    assert rows[0]["mode"] == "qwen3_full_vit"
    assert rows[0]["accuracy_total"] == 0.5
    assert rows[0]["total_ms"] == 123.0
    assert rows[0]["full_or_raw_patch_tokens"] == 100.0
    assert rows[0]["autogaze_selected_patch_tokens"] == 40.0
    assert rows[0]["peak_memory_bytes"] == 4096.0


def test_normalize_plugin_summary_preserves_integration_claim_fields(tmp_path):
    report = tmp_path / "plugin_hlvid_summary.json"
    report.write_text(
        json.dumps(
            {
                "modes": {
                    "qwen3_chunked_vit_autogaze_sparse": {
                        "total": 1,
                        "correct": 1,
                        "failed": 0,
                        "parse_failed": 0,
                        "accuracy_total": 1.0,
                        "accuracy_scored": 1.0,
                        "status_counts": {"executed": 1},
                        "integration_summary": {
                            "integration_level": "pre_encoder_sparse",
                            "execution_claim": "actual_pre_encoder_sparse",
                            "actual_pruning_applied_claim": "yes",
                            "vision_encoder_latency_reduction_claim": "yes",
                            "mllm_context_reduction_claim": "yes",
                        },
                        "processing_budget_summary": {
                            "mode_median": {
                                "patch_budget_before_vit.actual_raw_patch_tokens_before_vit": 1000,
                                "patch_budget_before_vit.estimated_visual_tokens_after_prune": 100,
                                "patch_budget_before_vit.estimated_visual_token_reduction_ratio": 10,
                                "llm_visual_budget.actual_visual_tokens": 100,
                            }
                        },
                    }
                },
                "pairwise_comparisons": [
                    {
                        "pair": "qwen3_full_vit -> qwen3_chunked_vit_autogaze_sparse",
                        "latency_speedup": 2.0,
                        "patch_or_visual_token_reduction_ratio": 10.0,
                    }
                ],
            }
        )
    )

    rows = normalize_report_file(report)

    assert rows[0]["integration_level"] == "pre_encoder_sparse"
    assert rows[0]["execution_claim"] == "actual_pre_encoder_sparse"
    assert rows[0]["actual_pruning_applied"] == "yes"
    assert rows[0]["vit_latency_reduction_claim"] == "yes"
    assert rows[0]["mllm_context_reduction_claim"] == "yes"
    assert rows[0]["token_reduction_ratio"] == 10.0
    pairwise = rows[1]
    assert pairwise["report_kind"] == "plugin_pairwise_comparison"
    assert pairwise["comparison_pair"] == "qwen3_full_vit -> qwen3_chunked_vit_autogaze_sparse"
    assert pairwise["baseline_mode"] == "qwen3_full_vit"
    assert pairwise["candidate_mode"] == "qwen3_chunked_vit_autogaze_sparse"
    assert pairwise["latency_speedup"] == 2.0
    assert pairwise["token_reduction_ratio"] == 10.0


def test_aggregate_report_roots_accepts_sort_mode(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    for name, mode, total in [
        ("autogaze.json", "autogaze", 5000),
        ("keep_all.json", "keep_all", 9000),
    ]:
        (root / name).write_text(
            json.dumps(
                {
                    "mode": mode,
                    "key_metrics_summary": {"latency_ms": {"total_median": total}},
                }
            )
        )
    out = tmp_path / "trend"

    artifacts = aggregate_report_roots([root], out, sort="latency")

    with artifacts["csv"].open() as f:
        rows = list(csv.DictReader(f))
    assert [row["mode"] for row in rows] == ["autogaze", "keep_all"]
