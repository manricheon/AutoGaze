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
        "key_metrics_summary": {
            "latency_ms": {
                "total_median": 9000,
                "preprocess_total_median": 3000,
                "autogaze_median": 800,
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
            }
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
