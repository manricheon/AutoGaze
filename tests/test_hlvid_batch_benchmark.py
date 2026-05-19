import argparse
import json
from pathlib import Path

from repro.hlvid_batch_benchmark import (
    build_prepare_report,
    build_gain_report,
    build_runner_command,
    discover_dataset_layout,
    flatten_metric_row,
)


def test_discover_dataset_layout_finds_manifest_and_video_root(tmp_path: Path):
    dataset = tmp_path / "hlvid"
    videos = dataset / "videos"
    videos.mkdir(parents=True)
    (videos / "clip.mp4").write_bytes(b"video")
    manifest = dataset / "manifest_test.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "question_id": 1,
                    "category": "av",
                    "video_path": "clip.mp4",
                    "question": "Q? A. a B. b C. c D. d",
                    "answer": "A",
                }
            ]
        )
    )

    layout = discover_dataset_layout(dataset)

    assert layout["manifest"] == manifest
    assert layout["video_root"] == videos


def test_discover_dataset_layout_supports_huggingface_snapshot_layout(tmp_path: Path):
    dataset = tmp_path / "hlvid"
    data = dataset / "data"
    videos = dataset / "videos"
    data.mkdir(parents=True)
    videos.mkdir()
    manifest = data / "test-00000-of-00001.parquet"
    video = videos / "clip_av_video_3_000.mp4"
    video.write_bytes(b"video")

    import pandas as pd

    pd.DataFrame(
        [
            {
                "question_id": 1,
                "category": "av",
                "video_path": video.name,
                "question": "Q? A. a B. b C. c D. d",
                "answer": "A",
            }
        ]
    ).to_parquet(manifest)

    layout = discover_dataset_layout(dataset)

    assert layout["manifest"] == manifest
    assert layout["video_root"] == videos


def test_prepare_report_detects_missing_videos_and_archives(tmp_path: Path):
    dataset = tmp_path / "hlvid"
    data = dataset / "data"
    data.mkdir(parents=True)
    (dataset / "videos_part_0001.tar").write_bytes(b"tar")
    manifest = data / "test-00000-of-00001.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "question_id": 1,
                "category": "av",
                "video_path": "clip_av_video_3_000.mp4",
                "question": "Q? A. a B. b C. c D. d",
                "answer": "A",
            }
        )
        + "\n"
    )

    report = build_prepare_report(dataset)

    assert report["hf_layout_detected"] is True
    assert report["video_archive_count"] == 1
    assert report["ready_for_full_benchmark"] is False
    assert report["missing_video_samples"] == ["clip_av_video_3_000.mp4"]


def test_build_runner_command_includes_local_manifest_and_measurement_flags(tmp_path: Path):
    args = argparse.Namespace(
        model_path="local-nvila",
        autogaze_model="local-autogaze",
        device="cuda",
        device_map="auto",
        num_video_frames=128,
        num_video_frames_thumbnail=64,
        max_tiles_video=8,
        max_batch_size_autogaze=16,
        max_batch_size_siglip=32,
        max_new_tokens=4,
        warmup_runs=1,
        measure_ttft=True,
        video_resize_shortest_edge=720,
        video_resize_longest_edge=None,
        video_resize_width=None,
        video_resize_height=None,
        video_decode_strategy="auto",
        autogaze_target_scales="56+112+196+392",
        autogaze_target_patch_size=14,
        visualization_output_dir=str(tmp_path / "viz"),
        visualization_fps=5,
        visualization_alpha=0.25,
        visualization_selected_max_long_side=720,
        task_loss_requirement_tile=0.7,
        continue_on_error=True,
        limit=3,
        split="test",
        config="default",
        extra_runner_args=[],
    )

    command = build_runner_command(
        args,
        gazing_mode="autogaze",
        manifest=tmp_path / "manifest.jsonl",
        video_root=tmp_path / "videos",
        predictions=tmp_path / "pred.jsonl",
        summary=tmp_path / "summary.json",
        scored_predictions=tmp_path / "scored.jsonl",
    )

    assert command[:3] == ["python", "-m", "repro.nvila_runner"]
    assert "--manifest" in command
    assert str(tmp_path / "manifest.jsonl") in command
    assert "--hlvid-video-root" in command
    assert "--gazing-mode" in command
    assert "autogaze" in command
    assert "--measure-ttft" in command
    assert "--continue-on-error" in command
    assert "--warmup-runs" in command
    assert "1" in command
    assert "--video-decode-strategy" in command
    assert "auto" in command
    assert "--visualization-output-dir" in command
    assert str(tmp_path / "viz") in command
    assert "--visualization-selected-max-long-side" in command
    assert "720" in command


def test_build_gain_report_compares_accuracy_latency_tokens_and_memory():
    keep_all_rows = [
        {
            "question_id": 1,
            "video_path": "clip.mp4",
            "question": "Q? A. a B. b C. c D. d",
            "answer": "A",
            "raw_output": "A",
            "total_ms": 100.0,
            "generate_ms": 80.0,
            "video_preprocess_ms": 20.0,
            "autogaze_ms": 1.0,
            "gazing_info_total_ms": 1.0,
            "siglip_vision_ms": 40.0,
            "llm_forward_ms": 30.0,
            "ttft_ms": 50.0,
            "generation_decode_after_ttft_estimated_ms": 30.0,
            "processor_peak_memory_bytes": 2000,
            "ttft_peak_memory_bytes": 1500,
            "llm_peak_memory_bytes": 1000,
            "peak_memory_bytes": 2500,
            "token_metrics": {
                "llm_actual_visual_tokens": 100,
                "llm_keep_all_visual_tokens_estimated": 100,
                    "encoder_raw_patch_tokens": 900,
                    "encoder_autogaze_selected_patch_tokens": 900,
                    "autogaze_input_tile_frame_instances": 80,
                    "autogaze_input_patch_tokens": 800,
                    "autogaze_selected_patch_tokens": 800,
                },
            }
    ]
    autogaze_rows = [
        {
            "question_id": 1,
            "video_path": "clip.mp4",
            "question": "Q? A. a B. b C. c D. d",
            "answer": "A",
            "raw_output": "A",
            "total_ms": 60.0,
            "generate_ms": 35.0,
            "video_preprocess_ms": 25.0,
            "autogaze_ms": 12.0,
            "gazing_info_total_ms": 12.0,
            "autogaze_model_forward_ms": 10.0,
            "siglip_vision_ms": 15.0,
            "llm_forward_ms": 20.0,
            "ttft_ms": 25.0,
            "generation_decode_after_ttft_estimated_ms": 10.0,
            "processor_peak_memory_bytes": 1500,
            "ttft_peak_memory_bytes": 1200,
            "llm_peak_memory_bytes": 500,
            "peak_memory_bytes": 1600,
            "token_metrics": {
                "llm_actual_visual_tokens": 40,
                "llm_keep_all_visual_tokens_estimated": 100,
                "llm_visual_token_reduction_ratio": 2.5,
                "encoder_raw_patch_tokens": 900,
                "encoder_autogaze_selected_patch_tokens": 300,
                "autogaze_input_tile_frame_instances": 80,
                "autogaze_input_patch_tokens": 800,
                "autogaze_selected_patch_tokens": 200,
                "encoder_token_reduction_ratio": 3.0,
            },
            "compute_metrics": {
                "siglip_encoder": {"keep_all_to_actual_total_macs_ratio": 4.0},
                "mllm": {"kv_cache_reduction_ratio": 2.5},
            },
        }
    ]

    report = build_gain_report(keep_all_rows=keep_all_rows, autogaze_rows=autogaze_rows)

    assert report["keep_all"]["accuracy"]["accuracy_scored"] == 1.0
    assert report["benchmark_samples"]["autogaze"][0]["target_video"] == "clip.mp4"
    assert report["benchmark_samples"]["autogaze"][0]["question"] == "Q? A. a B. b C. c D. d"
    assert report["benchmark_samples"]["autogaze"][0]["model_answer"] == "A"
    assert report["benchmark_samples"]["autogaze"][0]["correct_answer"] == "A"
    assert report["readable_summary"]["run_counts"] == {
        "keep_all_rows": 1,
        "autogaze_rows": 1,
        "count_note": (
            "Counts are prediction rows per mode. With --limit 3 and both modes enabled, "
            "expect keep_all_rows=3 and autogaze_rows=3; warmup runs are not counted."
        ),
    }
    assert report["readable_summary"]["latency_ms_median"]["total_ms"] == {
        "keep_all": 100.0,
        "autogaze": 60.0,
        "speedup_ratio_keep_all_over_autogaze": 100.0 / 60.0,
        "reduction_percent_of_keep_all": 40.0,
    }
    assert report["readable_summary"]["latency_accounting"]["additive_formula"] == (
        "total_ms = video_preprocess_ms + generate_ms"
    )
    assert "video_decode_ms" in report["readable_summary"]["latency_accounting"]["do_not_sum_with_total_ms"]
    assert report["readable_summary"]["latency_ms_detail_median"]["generate_ms"] == {
        "keep_all": 80.0,
        "autogaze": 35.0,
        "speedup_ratio_keep_all_over_autogaze": 80.0 / 35.0,
        "reduction_percent_of_keep_all": 56.25,
    }
    assert report["readable_summary"]["latency_ms_median"]["preprocess_total_ms"] == {
        "keep_all": 20.0,
        "autogaze": 25.0,
        "speedup_ratio_keep_all_over_autogaze": 20.0 / 25.0,
        "reduction_percent_of_keep_all": -25.0,
    }
    assert report["readable_summary"]["latency_ms_median"]["autogaze_ms"] == {
        "keep_all": 1.0,
        "autogaze": 12.0,
        "speedup_ratio_keep_all_over_autogaze": 1.0 / 12.0,
        "reduction_percent_of_keep_all": -1100.0,
    }
    assert report["readable_summary"]["latency_ms_median"]["vit_encoder_ms"] == {
        "keep_all": 40.0,
        "autogaze": 15.0,
        "speedup_ratio_keep_all_over_autogaze": 40.0 / 15.0,
        "reduction_percent_of_keep_all": 62.5,
    }
    assert report["readable_summary"]["latency_ms_median"]["llm_ms"] == {
        "keep_all": 30.0,
        "autogaze": 20.0,
        "speedup_ratio_keep_all_over_autogaze": 30.0 / 20.0,
        "reduction_percent_of_keep_all": 100.0 / 3.0,
    }
    assert report["readable_summary"]["latency_ms_detail_median"]["siglip_vision_ms"] == {
        "keep_all": 40.0,
        "autogaze": 15.0,
        "speedup_ratio_keep_all_over_autogaze": 40.0 / 15.0,
        "reduction_percent_of_keep_all": 62.5,
    }
    assert report["readable_summary"]["key_metrics_median"]["latency_ms"]["total_ms"] == {
        "keep_all": 100.0,
        "autogaze": 60.0,
        "speedup_ratio_keep_all_over_autogaze": 100.0 / 60.0,
        "reduction_percent_of_keep_all": 40.0,
    }
    assert report["readable_summary"]["key_metrics_median"]["tokens"]["llm_visual_tokens"] == {
        "before_keep_all_estimated": 100.0,
        "after_autogaze_actual": 40.0,
        "reduction_ratio_before_over_after": 2.5,
        "reduction_percent_of_before": 60.0,
    }
    assert report["readable_summary"]["key_metrics_median"]["memory_bytes"]["llm_peak"] == {
        "keep_all": 1000.0,
        "autogaze": 500.0,
        "reduction_ratio_keep_all_over_autogaze": 2.0,
        "reduction_percent_of_keep_all": 50.0,
    }
    assert report["readable_summary"]["key_metrics_median"]["memory_bytes"]["overall_peak"] == {
        "keep_all": 2500.0,
        "autogaze": 1600.0,
        "reduction_ratio_keep_all_over_autogaze": 2500.0 / 1600.0,
        "reduction_percent_of_keep_all": 36.0,
    }
    assert report["readable_summary"]["memory_bytes_median"]["llm_peak_memory_bytes"] == {
        "keep_all": 1000.0,
        "autogaze": 500.0,
        "reduction_ratio_keep_all_over_autogaze": 2.0,
        "reduction_percent_of_keep_all": 50.0,
    }
    assert report["readable_summary"]["tokens_median"]["encoder_patches"] == {
        "before_keep_all_or_raw": 900.0,
        "after_autogaze": 300.0,
        "reduction_ratio_before_over_after": 3.0,
        "reduction_percent_of_before": 100.0 * (900.0 - 300.0) / 900.0,
    }
    assert report["readable_summary"]["tokens_median"]["autogaze_input_tile_patches"] == {
        "before_autogaze_selection": 800.0,
        "after_autogaze_selection": 200.0,
        "tile_frame_instances": 80.0,
        "reduction_ratio_before_over_after": 4.0,
        "reduction_percent_of_before": 75.0,
    }
    assert report["readable_summary"]["tokens_median"]["llm_visual_tokens"] == {
        "before_keep_all_estimated": 100.0,
        "after_autogaze_actual": 40.0,
        "reduction_ratio_before_over_after": 2.5,
        "reduction_percent_of_before": 60.0,
    }
    assert report["gains"]["reduction_percent_median"]["latency_ms"]["total_ms"] == 40.0
    assert report["gains"]["reduction_percent_median"]["memory_bytes"]["llm_peak_memory_bytes"] == 50.0
    assert report["gains"]["reduction_percent_median"]["tokens"]["llm_visual_tokens"] == 60.0
    assert report["autogaze"]["latency_ms"]["total_ms"]["median"] == 60.0
    assert report["gains"]["latency_speedup_median"]["total_ms"] == 100.0 / 60.0
    assert report["gains"]["memory_reduction_ratio_median"]["llm_peak_memory_bytes"] == 2.0
    assert report["gains"]["autogaze_token_reduction_median"]["llm_visual_token_reduction_ratio"] == 2.5
    assert report["autogaze"]["tokens"]["token_metrics.encoder_raw_patch_tokens"]["median"] == 900.0
    assert report["autogaze"]["tokens"]["token_metrics.encoder_autogaze_selected_patch_tokens"]["median"] == 300.0
    assert report["autogaze"]["tokens"]["token_metrics.autogaze_input_tile_frame_instances"]["median"] == 80.0
    assert report["autogaze"]["tokens"]["token_metrics.autogaze_input_patch_tokens"]["median"] == 800.0
    assert report["autogaze"]["tokens"]["token_metrics.autogaze_selected_patch_tokens"]["median"] == 200.0
    assert report["gains"]["autogaze_token_reduction_median"]["encoder_raw_patch_tokens"] == 900.0
    assert report["gains"]["autogaze_token_reduction_median"]["encoder_autogaze_selected_patch_tokens"] == 300.0
    assert report["gains"]["autogaze_token_reduction_median"]["autogaze_input_tile_frame_instances"] == 80.0
    assert report["gains"]["autogaze_token_reduction_median"]["autogaze_input_patch_tokens"] == 800.0
    assert report["gains"]["autogaze_token_reduction_median"]["autogaze_selected_patch_tokens"] == 200.0
    assert report["gains"]["compute_reduction_median"]["siglip_total_macs"] == 4.0


def test_build_gain_report_marks_keep_all_as_missing_when_skipped():
    autogaze_rows = [
        {
            "question_id": 1,
            "answer": "A",
            "raw_output": "A",
            "total_ms": 60.0,
            "llm_peak_memory_bytes": 500,
            "token_metrics": {
                "llm_actual_visual_tokens": 40,
                "llm_keep_all_visual_tokens_estimated": 100,
                "encoder_raw_patch_tokens": 900,
                "encoder_autogaze_selected_patch_tokens": 300,
                "autogaze_input_patch_tokens": 800,
                "autogaze_selected_patch_tokens": 200,
            },
        }
    ]

    report = build_gain_report(keep_all_rows=[], autogaze_rows=autogaze_rows)

    assert report["keep_all"]["accuracy"]["total"] == 0
    assert report["autogaze"]["accuracy"]["total"] == 1
    assert report["readable_summary"]["mode_status"] == {
        "keep_all": "skipped_or_missing",
        "autogaze": "available",
        "note": "A skipped/missing mode is still shown, but cross-mode ratios are null because no baseline rows exist.",
    }
    assert report["readable_summary"]["run_counts"]["keep_all_rows"] == 0
    assert report["readable_summary"]["latency_ms_median"]["total_ms"] == {
        "keep_all": 0.0,
        "autogaze": 60.0,
        "speedup_ratio_keep_all_over_autogaze": None,
        "reduction_percent_of_keep_all": None,
    }
    assert report["readable_summary"]["memory_bytes_median"]["llm_peak_memory_bytes"] == {
        "keep_all": 0.0,
        "autogaze": 500.0,
        "reduction_ratio_keep_all_over_autogaze": None,
        "reduction_percent_of_keep_all": None,
    }
    assert report["readable_summary"]["tokens_median"]["llm_visual_tokens"] == {
        "before_keep_all_estimated": 100.0,
        "after_autogaze_actual": 40.0,
        "reduction_ratio_before_over_after": 2.5,
        "reduction_percent_of_before": 60.0,
    }
    assert report["gains"]["latency_speedup_median"]["total_ms"] is None
    assert report["gains"]["memory_reduction_ratio_median"]["llm_peak_memory_bytes"] is None
    assert report["gains"]["reduction_percent_median"]["latency_ms"]["total_ms"] is None


def test_build_gain_report_splits_paired_correctness_between_modes():
    keep_all_rows = [
        {
            "question_id": 1,
            "video_path": "clip1.mp4",
            "question": "Q1? A. a B. b C. c D. d",
            "answer": "A",
            "raw_output": "A",
        },
        {
            "question_id": 2,
            "video_path": "clip2.mp4",
            "question": "Q2? A. a B. b C. c D. d",
            "answer": "B",
            "raw_output": "B",
        },
        {
            "question_id": 3,
            "video_path": "clip3.mp4",
            "question": "Q3? A. a B. b C. c D. d",
            "answer": "C",
            "raw_output": "A",
        },
        {
            "question_id": 4,
            "video_path": "clip4.mp4",
            "question": "Q4? A. a B. b C. c D. d",
            "answer": "D",
            "raw_output": "A",
        },
        {
            "question_id": 6,
            "video_path": "clip6.mp4",
            "question": "Q6? A. a B. b C. c D. d",
            "answer": "B",
            "raw_output": "B",
        },
    ]
    autogaze_rows = [
        {
            "question_id": 1,
            "video_path": "clip1.mp4",
            "question": "Q1? A. a B. b C. c D. d",
            "answer": "A",
            "raw_output": "A",
        },
        {
            "question_id": 2,
            "video_path": "clip2.mp4",
            "question": "Q2? A. a B. b C. c D. d",
            "answer": "B",
            "raw_output": "A",
        },
        {
            "question_id": 3,
            "video_path": "clip3.mp4",
            "question": "Q3? A. a B. b C. c D. d",
            "answer": "C",
            "raw_output": "C",
        },
        {
            "question_id": 4,
            "video_path": "clip4.mp4",
            "question": "Q4? A. a B. b C. c D. d",
            "answer": "D",
            "raw_output": "A",
        },
        {
            "question_id": 5,
            "video_path": "clip5.mp4",
            "question": "Q5? A. a B. b C. c D. d",
            "answer": "A",
            "raw_output": "A",
        },
    ]

    report = build_gain_report(keep_all_rows=keep_all_rows, autogaze_rows=autogaze_rows)

    comparison = report["correctness_comparison"]
    assert comparison["counts"] == {
        "total_unique": 6,
        "paired": 4,
        "both_correct": 1,
        "keep_all_only_correct": 1,
        "autogaze_only_correct": 1,
        "both_wrong": 1,
        "keep_all_missing": 1,
        "autogaze_missing": 1,
    }
    assert comparison["paired_rates"] == {
        "both_correct": 0.25,
        "keep_all_only_correct": 0.25,
        "autogaze_only_correct": 0.25,
        "both_wrong": 0.25,
    }
    assert [sample["bucket"] for sample in comparison["samples"][:6]] == [
        "both_correct",
        "keep_all_only_correct",
        "autogaze_only_correct",
        "both_wrong",
        "keep_all_missing",
        "autogaze_missing",
    ]
    assert comparison["samples"][1]["keep_all_correct"] is True
    assert comparison["samples"][1]["autogaze_correct"] is False
    assert comparison["samples"][2]["keep_all_answer"] == "A"
    assert comparison["samples"][2]["autogaze_answer"] == "C"
    assert report["benchmark_samples"]["correctness_comparison"] == comparison["samples"]


def test_flatten_metric_row_creates_csv_friendly_summary():
    report = {
        "gains": {"latency_speedup_median": {"total_ms": 2.0}},
        "autogaze": {"accuracy": {"accuracy_scored": 0.5}},
        "keep_all": {"accuracy": {"accuracy_scored": 0.4}},
        "correctness_comparison": {
            "counts": {
                "paired": 10,
                "both_correct": 4,
                "keep_all_only_correct": 2,
                "autogaze_only_correct": 1,
                "both_wrong": 3,
            }
        },
    }

    row = flatten_metric_row(report)

    assert row["autogaze_accuracy_scored"] == 0.5
    assert row["keep_all_accuracy_scored"] == 0.4
    assert row["gain_latency_total_ms_speedup_median"] == 2.0
    assert row["correctness_paired"] == 10
    assert row["correctness_both_correct"] == 4
    assert row["correctness_keep_all_only_correct"] == 2
    assert row["correctness_autogaze_only_correct"] == 1
    assert row["correctness_both_wrong"] == 3
