import json

from repro.markdown_report import render_markdown_report, write_markdown_report


def test_render_single_markdown_report_includes_pipeline_and_key_metrics(tmp_path):
    payload = {
        "model_path": "local-nvila",
        "autogaze_model": "local-autogaze",
        "gazing_mode": "autogaze",
        "video": "clip.mp4",
        "prompt": "Question? A. one B. two C. three D. four",
        "answer": "A",
        "video_input_summary": {
            "source_frames": 240,
            "source_resolution": "3840x2160",
            "requested_video_frames": 128,
            "actual_video_frames": 128,
            "requested_thumbnail_frames": 64,
            "actual_thumbnail_frames": 64,
            "processor_input_resolution": "1280x720",
            "video_decode_strategy": "seek",
            "video_decode_frames_read": 812,
            "spatial_tiles_per_video": [8],
            "temporal_chunks_per_video": 8,
        },
        "processing_budget_summary": {
            "model_processing_unit": {"name": "nvila_392px_spatial_tile_sequence", "tile_size_px": 392},
            "tiling": {"spatial_tiles_per_frame": 8, "tile_frame_instances": 1024},
            "thumbnail": {"enabled": True, "actual_frames": 64, "policy": "keep_all"},
            "multiscale_patch_space": {
                "patch_positions_per_tile_frame": 1060,
                "patch_positions_by_scale": {"56": 16, "112": 64, "196": 196, "392": 784},
            },
            "patch_budget_before_siglip": {
                "keep_all_total_patch_tokens": 1153280,
                "autogaze_selected_total_patch_tokens": 176384,
                "total_patch_reduction_ratio": 6.538,
            },
            "llm_visual_budget": {
                "keep_all_visual_tokens_estimated": 128512,
                "actual_visual_tokens": 19632,
                "visual_token_reduction_ratio": 6.546,
            },
        },
        "key_metrics_summary": {
            "latency_ms": {
                "total_median": 9000,
                "preprocess_without_autogaze_median": 2200,
                "preprocess_total_median": 3000,
                "autogaze_median": 800,
                "autogaze_total_median": 800,
                "vit_encoder_median": 1200,
                "llm_median": 4000,
            },
            "tokens": {
                "video_sampled_frames": 128,
                "thumbnail_sampled_frames": 64,
                "encoder_patch_tokens_before_keep_all_or_raw": 1085440,
                "encoder_patch_tokens_after_autogaze": 250000,
                "encoder_token_reduction_ratio": 4.34,
                "autogaze_input_tile_patch_tokens": 1085440,
                "autogaze_selected_tile_patch_tokens": 220000,
                "autogaze_patch_reduction_ratio": 4.93,
                "llm_visual_tokens_before_keep_all_estimated": 122880,
                "llm_visual_tokens_after_actual": 36000,
                "llm_visual_token_reduction_ratio": 3.41,
            },
            "memory_bytes": {
                "processor_peak_median": 1_500_000_000,
                "ttft_peak_median": 2_500_000_000,
                "llm_peak_median": 3_500_000_000,
                "overall_peak_median": 4_000_000_000,
            },
            "latency_accounting": {
                "additive_total_ms": {
                    "formula": (
                        "total_ms = video_preprocess_without_autogaze_ms + "
                        "autogaze_total_ms + generate_ms"
                    ),
                    "total_ms": 9000,
                    "video_preprocess_without_autogaze_ms": 2200,
                    "autogaze_total_ms": 800,
                    "generate_ms": 6000,
                    "recomputed_total_ms": 9000,
                    "delta_ms": 0,
                    "ttft_ms_excluded_from_total": 1800,
                },
                "nested_preprocess_breakdown_ms": {
                    "video_decode_ms": {
                        "value": 700,
                        "included_in": "video_preprocess_without_autogaze_ms",
                        "add_to_total_ms": False,
                    },
                },
                "do_not_sum_with_total_ms": ["video_decode_ms"],
                "hierarchy": {
                    "total_formula": (
                        "total_ms = video_preprocess_without_autogaze_ms + "
                        "autogaze_total_ms + generate_ms"
                    ),
                    "ascii_tree": (
                        "total_ms = video_preprocess_without_autogaze_ms + autogaze_total_ms + generate_ms\n"
                        "|-- video_preprocess_without_autogaze_ms\n"
                        "|-- autogaze_total_ms\n"
                        "`-- generate_ms"
                    ),
                    "quick_answers": {
                        "what_total_ms_is": "End-to-end latency.",
                        "what_generate_ms_includes": (
                            "NVILA model.generate after preprocessing: vision_encoder_ms plus llm_forward_ms."
                        ),
                        "where_is_autogaze_ms_included": "autogaze_total_ms",
                    },
                },
            },
        },
    }

    markdown = render_markdown_report(payload, source_path="single.json")

    assert "# AutoGaze Reproduction Report" in markdown
    assert "## Model Pipeline" in markdown
    assert "Video decode/sample" in markdown
    assert "AutoGaze ON/OFF" in markdown
    assert "SigLIP / ViT Encoder" in markdown
    assert "LLM prefill/generation" in markdown
    assert "## Video And Experiment Info" in markdown
    assert "3840x2160" in markdown
    assert "1280x720" in markdown
    assert "## Key Metrics" in markdown
    assert "encoder_patch_tokens_before_keep_all_or_raw" in markdown
    assert "llm_peak_median" in markdown
    assert "## Step-by-step Pipeline Metrics" in markdown
    assert "## Processing Budget Summary" in markdown
    assert "nvila_392px_spatial_tile_sequence" in markdown
    assert "keep_all_total_patch_tokens" in markdown
    assert "## AutoGaze Token And Patch Flow" in markdown
    assert "full_multiscale_patch_budget_before_autogaze" in markdown
    assert "encoder_input_patch_tokens_after_autogaze" in markdown
    assert "llm_input_visual_tokens_after_token_shuffle_projector" in markdown
    assert "## Latency Accounting" in markdown
    assert "### Time Hierarchy" in markdown
    assert "total_ms = video_preprocess_without_autogaze_ms + autogaze_total_ms + generate_ms" in markdown
    assert "where_is_autogaze_ms_included" in markdown
    assert "video_decode_ms" in markdown

    output = tmp_path / "report.md"
    input_json = tmp_path / "single.json"
    input_json.write_text(json.dumps(payload))
    write_markdown_report(input_json, output)
    assert output.read_text() == render_markdown_report(payload, source_path=str(input_json))


def test_render_benchmark_markdown_report_includes_scores_and_comparison_tables():
    payload = {
        "dataset": {
            "dataset_dir": "/data/HLVid",
            "video_root": "/data/HLVid/videos",
            "manifest": "/data/HLVid/data/test.parquet",
        },
        "keep_all": {
            "accuracy": {"accuracy_scored": 0.50, "correct": 5, "scored": 10, "failed": 0},
        },
        "autogaze": {
            "accuracy": {"accuracy_scored": 0.60, "correct": 6, "scored": 10, "failed": 1},
        },
        "readable_summary": {
            "latency_ms_detail_median": {
                "total_ms": {
                    "keep_all": 10000,
                    "autogaze": 7000,
                    "speedup_ratio_keep_all_over_autogaze": 1.43,
                    "reduction_percent_of_keep_all": 30,
                },
                "video_decode_ms": {
                    "keep_all": 1200,
                    "autogaze": 1100,
                    "speedup_ratio_keep_all_over_autogaze": 1.09,
                    "reduction_percent_of_keep_all": 8.33,
                },
                "gazing_info_total_ms": {
                    "keep_all": 0,
                    "autogaze": 900,
                    "speedup_ratio_keep_all_over_autogaze": 0,
                    "reduction_percent_of_keep_all": None,
                },
            },
            "key_metrics_median": {
                "latency_ms": {
                    "total_ms": {
                        "keep_all": 10000,
                        "autogaze": 7000,
                        "speedup_ratio_keep_all_over_autogaze": 1.43,
                        "reduction_percent_of_keep_all": 30,
                    },
                    "vit_encoder_ms": {
                        "keep_all": 2000,
                        "autogaze": 500,
                        "speedup_ratio_keep_all_over_autogaze": 4,
                        "reduction_percent_of_keep_all": 75,
                    },
                },
                "tokens": {
                    "llm_visual_tokens": {
                        "before_keep_all_estimated": 120000,
                        "after_autogaze_actual": 40000,
                        "reduction_ratio_before_over_after": 3,
                        "reduction_percent_of_before": 66.67,
                    },
                },
                "memory_bytes": {
                    "llm_peak": {
                        "keep_all": 80_000_000_000,
                        "autogaze": 50_000_000_000,
                        "reduction_ratio_keep_all_over_autogaze": 1.6,
                        "reduction_percent_of_keep_all": 37.5,
                    },
                },
            },
            "latency_accounting": {
                "additive_formula": "total_ms = video_preprocess_ms + generate_ms",
                "additive_fields": ["video_preprocess_ms", "generate_ms"],
                "do_not_sum_with_total_ms": ["video_decode_ms", "autogaze_ms", "ttft_ms"],
            },
            "processing_budget_summary": {
                "keep_all_median": {
                    "video.source_resolution": "3840x2160",
                    "video.processor_input_resolution": "1280x720",
                    "tiling.spatial_tiles_per_frame": 8,
                    "thumbnail.actual_frames": 64,
                    "multiscale_patch_space.patch_positions_per_tile_frame": 1060,
                    "patch_budget_before_siglip.keep_all_total_patch_tokens": 900,
                    "patch_budget_before_siglip.autogaze_selected_total_patch_tokens": 900,
                    "llm_visual_budget.keep_all_visual_tokens_estimated": 100,
                    "llm_visual_budget.actual_visual_tokens": 100,
                },
                "autogaze_median": {
                    "video.source_resolution": "3840x2160",
                    "video.processor_input_resolution": "1280x720",
                    "tiling.spatial_tiles_per_frame": 8,
                    "thumbnail.actual_frames": 64,
                    "multiscale_patch_space.patch_positions_per_tile_frame": 1060,
                    "patch_budget_before_siglip.keep_all_total_patch_tokens": 900,
                    "patch_budget_before_siglip.autogaze_selected_total_patch_tokens": 300,
                    "llm_visual_budget.keep_all_visual_tokens_estimated": 100,
                    "llm_visual_budget.actual_visual_tokens": 40,
                },
            },
        },
        "correctness_comparison": {
            "counts": {
                "total_unique": 4,
                "paired": 4,
                "both_correct": 1,
                "keep_all_only_correct": 1,
                "autogaze_only_correct": 1,
                "both_wrong": 1,
                "keep_all_missing": 0,
                "autogaze_missing": 0,
            },
            "paired_rates": {
                "both_correct": 0.25,
                "keep_all_only_correct": 0.25,
                "autogaze_only_correct": 0.25,
                "both_wrong": 0.25,
            },
            "samples": [
                {
                    "target_video": "clip2.mp4",
                    "question": "What happened?",
                    "correct_answer": "B",
                    "keep_all_answer": "B",
                    "keep_all_correct": True,
                    "autogaze_answer": "A",
                    "autogaze_correct": False,
                    "bucket": "keep_all_only_correct",
                },
                {
                    "target_video": "clip3.mp4",
                    "question": "Where is it?",
                    "correct_answer": "C",
                    "keep_all_answer": "A",
                    "keep_all_correct": False,
                    "autogaze_answer": "C",
                    "autogaze_correct": True,
                    "bucket": "autogaze_only_correct",
                },
            ],
        },
        "benchmark_samples": {
            "autogaze": [
                {
                    "target_video": "clip.mp4",
                    "question": "What happened?",
                    "model_answer": "A",
                    "correct_answer": "A",
                    "correct": True,
                    "status": "ok",
                }
            ]
        },
    }

    markdown = render_markdown_report(payload, source_path="hlvid_gain.json")

    assert "## Benchmark Score" in markdown
    assert "accuracy_scored" in markdown
    assert "keep_all" in markdown
    assert "autogaze" in markdown
    assert "## Key Metrics" in markdown
    assert "llm_visual_tokens" in markdown
    assert "llm_peak" in markdown
    assert "## Benchmark Samples" in markdown
    assert "clip.mp4" in markdown
    assert "## Latency Accounting" in markdown
    assert "## Module Detail Metrics" in markdown
    assert "| Metric | Keep-all | AutoGaze | Speedup | Reduction % |" in markdown
    assert "| total_ms | 10,000 | 7,000 | 1.43 | 30 |" in markdown
    assert "| gazing_info_total_ms | 0 | 900 | 0 | - |" in markdown
    assert "## Benchmark Correctness Comparison" in markdown
    assert "| keep_all_only_correct | 1 | 0.25 |" in markdown
    assert "| autogaze_only_correct | 1 | 0.25 |" in markdown
    assert "| clip2.mp4 | What happened? | B | B | true | A | false | keep_all_only_correct |" in markdown
    assert "## Processing Budget Summary" in markdown
    assert "patch_budget_before_siglip.autogaze_selected_total_patch_tokens" in markdown
    assert "## AutoGaze Token And Patch Flow" in markdown
    assert "| Full multiscale patch budget before AutoGaze | 900 | 900 | 300 | 3 | 66.666667 |" in markdown
    assert "| LLM visual tokens after TokenShuffle/projector | 100 | 100 | 40 | 2.5 | 60 |" in markdown


def test_render_benchmark_markdown_keeps_detail_latency_when_autogaze_is_skipped():
    payload = {
        "keep_all": {
            "accuracy": {"accuracy_scored": 0.50, "correct": 5, "scored": 10, "failed": 0},
        },
        "autogaze": {
            "accuracy": {"accuracy_scored": 0.0, "correct": 0, "scored": 0, "failed": 0},
        },
        "readable_summary": {
            "mode_status": {"keep_all": "available", "autogaze": "skipped_or_missing"},
            "latency_ms_detail_median": {
                "total_ms": {
                    "keep_all": 10000,
                    "autogaze": None,
                    "speedup_ratio_keep_all_over_autogaze": None,
                    "reduction_percent_of_keep_all": None,
                },
                "video_decode_ms": {
                    "keep_all": 1200,
                    "autogaze": None,
                    "speedup_ratio_keep_all_over_autogaze": None,
                    "reduction_percent_of_keep_all": None,
                },
            },
            "key_metrics_median": {
                "latency_ms": {
                    "total_ms": {
                        "keep_all": 10000,
                        "autogaze": None,
                        "speedup_ratio_keep_all_over_autogaze": None,
                        "reduction_percent_of_keep_all": None,
                    },
                },
            },
        },
        "correctness_comparison": {
            "counts": {
                "total_unique": 1,
                "paired": 0,
                "both_correct": 0,
                "keep_all_only_correct": 0,
                "autogaze_only_correct": 0,
                "both_wrong": 0,
                "keep_all_missing": 0,
                "autogaze_missing": 1,
            },
            "paired_rates": {
                "both_correct": None,
                "keep_all_only_correct": None,
                "autogaze_only_correct": None,
                "both_wrong": None,
            },
            "samples": [
                {
                    "target_video": "clip-only.mp4",
                    "question": "What happened?",
                    "correct_answer": "A",
                    "keep_all_answer": "A",
                    "keep_all_correct": True,
                    "autogaze_answer": None,
                    "autogaze_correct": None,
                    "bucket": "autogaze_missing",
                }
            ],
        },
    }

    markdown = render_markdown_report(payload, source_path="hlvid_gain_keep_all_only.json")

    assert "## Module Detail Metrics" in markdown
    assert "| Metric | Keep-all | AutoGaze | Speedup | Reduction % |" in markdown
    assert "| total_ms | 10,000 | - | - | - |" in markdown
    assert "| video_decode_ms | 1,200 | - | - | - |" in markdown
    assert "| autogaze_missing | 1 | - |" in markdown
    assert "| clip-only.mp4 | What happened? | A | A | true | - | - | autogaze_missing |" in markdown


def test_render_stream_profile_markdown_keeps_source_and_effective_resolution_separate():
    payload = {
        "mode": "stream-profile",
        "model_path": "nvidia/NVILA-8B-HD-Video",
        "source_metadata": {
            "frames": 8992,
            "width": 3840,
            "height": 2160,
            "fps": 30,
        },
        "effective_video": {
            "width": 1280,
            "height": 720,
            "resize_mode": "shortest_edge",
        },
        "sampling": {
            "decode_strategy": "seek",
            "decode_frames_read": 812,
            "num_video_frames": 128,
            "num_video_frames_thumbnail": 64,
        },
        "stream_plan": {
            "tokens": {
                "encoder_raw_patch_tokens": 50880,
                "llm_keep_all_visual_tokens_estimated": 5760,
            }
        },
        "gaze": {
            "raw_patch_budget": 33920,
            "selected_non_padded_patches": 2113,
            "token_reduction_ratio": 16.05,
        },
        "stage_timings_ms": {
            "video_decode_seek": {"total_ms": 100, "count": 10},
            "tile_autogaze_forward": {"total_ms": 200, "count": 2},
            "siglip_gazed_forward": {"total_ms": 50, "count": 2},
        },
    }

    markdown = render_markdown_report(payload, source_path="stream.json")

    assert "source_resolution | 3840x2160" in markdown
    assert "processor_input_resolution | 1280x720" in markdown


def test_render_flexible_single_markdown_reads_generation_metrics_processing_budget():
    payload = {
        "runner": "flexible_runner",
        "mode": "single",
        "model_path": "weight/Qwen3-VL",
        "video": "clip.mp4",
        "generation": {
            "text": "B",
            "metrics": {
                "latency_ms": {"total": 1200, "generate": 700},
                "tokens": {
                    "visual_tokens_before_prune": 1000,
                    "visual_tokens_after_prune": 100,
                    "visual_token_reduction_ratio": 10.0,
                    "llm_context_tokens": 220,
                },
                "memory_bytes": {"peak_cuda_allocated": 10_000},
                "processing_budget_summary": {
                    "runner": "flexible_runner",
                    "video": {
                        "source_resolution": "3840x2160",
                        "processor_input_resolution": "1280x720",
                        "requested_video_frames": 128,
                    },
                    "thumbnail": {"enabled": True, "effective_frames": 16},
                    "patch_budget_before_vit": {
                        "actual_raw_patch_tokens_before_vit": 1000,
                        "estimated_visual_tokens_after_prune": 100,
                        "estimated_visual_token_reduction_ratio": 10.0,
                    },
                },
            },
        },
    }

    markdown = render_markdown_report(payload, source_path="flexible_single.json")

    assert "## Processing Budget Summary" in markdown
    assert "3840x2160" in markdown
    assert "## AutoGaze Token And Patch Flow" in markdown
    assert "full_patch_budget_before_selector" in markdown
    assert "encoder_input_patch_tokens_after_autogaze" in markdown
    assert "llm_input_context_tokens" in markdown
