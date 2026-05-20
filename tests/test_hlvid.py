from repro.hlvid import (
    REQUIRED_COLUMNS,
    parse_choice,
    read_manifest_file,
    score_predictions,
    validate_manifest_rows,
    viewer_row_to_manifest,
)


def test_parse_choice_accepts_direct_letters_and_prefixed_text():
    assert parse_choice("A") == "A"
    assert parse_choice("Answer: c.") == "C"
    assert parse_choice("The correct answer is D because the sign says Duke.") == "D"


def test_parse_choice_returns_none_for_ambiguous_output():
    assert parse_choice("A or B") is None
    assert parse_choice("No idea") is None


def test_validate_manifest_rows_requires_official_columns():
    row = {
        "question_id": 1,
        "category": "av",
        "video_path": "clip_av_video_5_001.mp4",
        "question": "Question? A. One B. Two C. Three D. Four",
        "answer": "A",
    }
    validate_manifest_rows([row])
    assert set(REQUIRED_COLUMNS).issubset(row)


def test_score_predictions_tracks_parse_failures_separately():
    rows = [
        {"answer": "A", "raw_output": "A"},
        {"answer": "B", "raw_output": "Answer: C"},
        {"answer": "D", "raw_output": "A or D"},
    ]
    summary, scored = score_predictions(rows)
    assert summary["total"] == 3
    assert summary["scored"] == 2
    assert summary["correct"] == 1
    assert summary["parse_failed"] == 1
    assert summary["accuracy_scored"] == 0.5
    assert scored[2]["parse_status"] == "failed"


def test_score_predictions_counts_model_failures_without_parse_failures():
    rows = [
        {"answer": "A", "raw_output": "A", "status": "ok"},
        {"answer": "B", "raw_output": None, "status": "failed", "error": "OOM"},
    ]

    summary, scored = score_predictions(rows)

    assert summary["total"] == 2
    assert summary["failed"] == 1
    assert summary["parse_failed"] == 0
    assert summary["scored"] == 1
    assert scored[1]["parse_status"] == "failed_run"


def test_score_predictions_summarizes_questions_without_dropping_row_details():
    rows = [
        {
            "question_id": 1,
            "video_path": "clip_1.mp4",
            "question": "What is shown? A. One B. Two C. Three D. Four",
            "answer": "A",
            "raw_output": "A",
        },
        {
            "question_id": 2,
            "video_path": "clip_2.mp4",
            "question": "What text appears? A. Left B. Right C. Up D. Down",
            "answer": "B",
            "raw_output": "B",
        },
    ]

    summary, scored = score_predictions(rows)

    assert summary["question_count"] == 2
    assert summary["question_samples"] == [
        {
            "question_id": 1,
            "video_path": "clip_1.mp4",
            "question": "What is shown? A. One B. Two C. Three D. Four",
            "answer": "A",
        },
        {
            "question_id": 2,
            "video_path": "clip_2.mp4",
            "question": "What text appears? A. Left B. Right C. Up D. Down",
            "answer": "B",
        },
    ]
    assert "predictions" in summary["question_note"]
    assert scored[0]["question"] == rows[0]["question"]


def test_score_predictions_adds_readable_benchmark_samples():
    rows = [
        {
            "question_id": 1,
            "video_path": "clip_1.mp4",
            "question": "What is shown? A. One B. Two C. Three D. Four",
            "answer": "A",
            "raw_output": "Answer: A",
            "status": "ok",
        },
        {
            "question_id": 2,
            "video": "clip_2.mp4",
            "prompt": "What text appears? A. Left B. Right C. Up D. Down",
            "answer": "B",
            "raw_output": None,
            "status": "failed",
            "error": "OOM",
        },
    ]

    summary, _ = score_predictions(rows)

    assert summary["benchmark_samples"] == [
        {
            "question_id": 1,
            "target_video": "clip_1.mp4",
            "question": "What is shown? A. One B. Two C. Three D. Four",
            "model_answer": "Answer: A",
            "parsed_model_answer": "A",
            "correct_answer": "A",
            "ground_truth_answer": "A",
            "correct": True,
            "status": "ok",
            "parse_status": "parsed",
        },
        {
            "question_id": 2,
            "target_video": "clip_2.mp4",
            "question": "What text appears? A. Left B. Right C. Up D. Down",
            "model_answer": None,
            "parsed_model_answer": None,
            "correct_answer": "B",
            "ground_truth_answer": "B",
            "correct": False,
            "status": "failed",
            "parse_status": "failed_run",
        },
    ]
    assert "model_answer" in summary["benchmark_sample_note"]


def test_score_predictions_includes_latency_memory_token_and_compute_summaries():
    rows = [
        {
            "answer": "A",
            "raw_output": "A",
            "total_ms": 100.0,
            "generate_ms": 88.0,
            "video_preprocess_ms": 20.0,
            "video_preprocess_without_autogaze_ms": 6.0,
            "autogaze_total_ms": 14.0,
            "video_decode_ms": 10.0,
            "autogaze_ms": 14.0,
            "gazing_info_total_ms": 14.0,
            "autogaze_model_forward_ms": 11.0,
            "siglip_vision_ms": 40.0,
            "llm_forward_ms": 30.0,
            "ttft_ms": 50.0,
            "generation_decode_after_ttft_estimated_ms": 38.0,
            "processor_peak_memory_bytes": 1300,
            "ttft_peak_memory_bytes": 1200,
            "llm_peak_memory_bytes": 1000,
            "peak_memory_bytes": 1500,
            "token_metrics": {
                "video_sampled_frames": 128,
                "thumbnail_sampled_frames": 64,
                "encoder_raw_patch_tokens": 900,
                "encoder_autogaze_selected_patch_tokens": 300,
                "encoder_token_reduction_ratio": 3.0,
                "autogaze_input_patch_tokens": 800,
                "autogaze_selected_patch_tokens": 200,
                "autogaze_patch_reduction_ratio": 4.0,
                "llm_keep_all_visual_tokens_estimated": 100,
                "llm_actual_visual_tokens": 40,
                "llm_visual_token_reduction_ratio": 2.5,
            },
            "compute_metrics": {
                "siglip_encoder": {"keep_all_to_actual_total_macs_ratio": 4.0},
                "mllm": {"kv_cache_reduction_ratio": 2.5},
            },
            "stage_timings_ms": {
                "processor": {
                    "autogaze_forward_batched": {"total_ms": 11.0, "count": 3, "mean_ms": 11.0 / 3.0},
                    "autogaze_total": {"total_ms": 14.0, "count": 1, "mean_ms": 14.0},
                }
            },
        },
        {
            "answer": "B",
            "raw_output": "B",
            "total_ms": 60.0,
            "generate_ms": 52.0,
            "video_preprocess_ms": 14.0,
            "video_preprocess_without_autogaze_ms": 4.0,
            "autogaze_total_ms": 10.0,
            "video_decode_ms": 6.0,
            "autogaze_ms": 10.0,
            "gazing_info_total_ms": 10.0,
            "autogaze_model_forward_ms": 8.0,
            "siglip_vision_ms": 20.0,
            "llm_forward_ms": 18.0,
            "ttft_ms": 30.0,
            "generation_decode_after_ttft_estimated_ms": 22.0,
            "processor_peak_memory_bytes": 900,
            "ttft_peak_memory_bytes": 800,
            "llm_peak_memory_bytes": 700,
            "peak_memory_bytes": 1000,
            "token_metrics": {
                "video_sampled_frames": 96,
                "thumbnail_sampled_frames": 48,
                "encoder_raw_patch_tokens": 800,
                "encoder_autogaze_selected_patch_tokens": 200,
                "encoder_token_reduction_ratio": 4.0,
                "autogaze_input_patch_tokens": 700,
                "autogaze_selected_patch_tokens": 175,
                "autogaze_patch_reduction_ratio": 4.0,
                "llm_keep_all_visual_tokens_estimated": 90,
                "llm_actual_visual_tokens": 30,
                "llm_visual_token_reduction_ratio": 3.0,
            },
            "compute_metrics": {
                "siglip_encoder": {"keep_all_to_actual_total_macs_ratio": 3.0},
                "mllm": {"kv_cache_reduction_ratio": 3.0},
            },
            "stage_timings_ms": {
                "processor": {
                    "autogaze_forward_batched": {"total_ms": 8.0, "count": 2, "mean_ms": 4.0},
                    "autogaze_total": {"total_ms": 10.0, "count": 1, "mean_ms": 10.0},
                }
            },
        },
    ]

    summary, _ = score_predictions(rows)

    assert summary["latency_ms"]["total_ms"]["median"] == 80.0
    assert summary["latency_ms"]["generate_ms"]["median"] == 70.0
    assert summary["latency_ms"]["video_decode_ms"]["median"] == 8.0
    assert summary["latency_ms"]["autogaze_model_forward_ms"]["median"] == 9.5
    assert summary["latency_ms"]["siglip_vision_ms"]["median"] == 30.0
    assert summary["latency_ms"]["llm_forward_ms"]["median"] == 24.0
    assert summary["memory_bytes"]["llm_peak_memory_bytes"]["median"] == 850.0
    assert summary["tokens"]["token_metrics.encoder_raw_patch_tokens"]["median"] == 850.0
    assert summary["tokens"]["token_metrics.encoder_autogaze_selected_patch_tokens"]["median"] == 250.0
    assert summary["compute"]["compute_metrics.siglip_encoder.keep_all_to_actual_total_macs_ratio"]["median"] == 3.5
    assert summary["readable_performance_summary"]["latency_ms_median"] == {
        "total_ms": 80.0,
        "preprocess_without_autogaze_ms": 5.0,
        "preprocess_total_ms": 17.0,
        "autogaze_ms": 12.0,
        "autogaze_total_ms": 12.0,
        "vit_encoder_ms": 30.0,
        "llm_ms": 24.0,
    }
    assert summary["readable_performance_summary"]["key_metrics_median"] == {
        "latency_ms": {
            "total_ms": 80.0,
            "preprocess_without_autogaze_ms": 5.0,
            "preprocess_total_ms": 17.0,
            "autogaze_ms": 12.0,
            "autogaze_total_ms": 12.0,
            "vit_encoder_ms": 30.0,
            "llm_ms": 24.0,
        },
        "tokens": {
            "video_sampled_frames": 112.0,
            "thumbnail_sampled_frames": 56.0,
            "encoder_patch_tokens_before_keep_all_or_raw": 850.0,
            "encoder_patch_tokens_after_autogaze": 250.0,
            "encoder_token_reduction_ratio": 3.5,
            "autogaze_input_tile_patch_tokens": 750.0,
            "autogaze_selected_tile_patch_tokens": 187.5,
            "autogaze_patch_reduction_ratio": 4.0,
            "llm_visual_tokens_before_keep_all_estimated": 95.0,
            "llm_visual_tokens_after_actual": 35.0,
            "llm_visual_token_reduction_ratio": 2.75,
        },
        "memory_bytes": {
            "processor_peak": 1100.0,
            "ttft_peak": 1000.0,
            "llm_peak": 850.0,
            "overall_peak": 1250.0,
        },
    }
    assert summary["readable_performance_summary"]["latency_accounting"]["additive_formula"] == (
        "total_ms = video_preprocess_without_autogaze_ms + autogaze_total_ms + generate_ms"
    )
    assert (
        summary["readable_performance_summary"]["latency_accounting"]["hierarchy"]["quick_answers"][
            "where_is_video_decode_ms_included"
        ]
        == "video_preprocess_without_autogaze_ms"
    )
    assert summary["readable_performance_summary"]["latency_ms_detail_median"][
        "video_preprocess_without_autogaze_ms"
    ] == 5.0
    assert "video_decode_ms" in summary["readable_performance_summary"]["latency_accounting"]["do_not_sum_with_total_ms"]
    assert summary["readable_performance_summary"]["latency_ms_detail_median"]["generate_ms"] == 70.0
    assert summary["readable_performance_summary"]["latency_ms_detail_median"]["video_decode_ms"] == 8.0
    assert summary["stage_timings_ms"]["processor.autogaze_forward_batched.total_ms"]["median"] == 9.5
    assert summary["stage_timings_ms"]["processor.autogaze_forward_batched.count"]["median"] == 2.5
    assert summary["readable_performance_summary"]["stage_timings_ms_median"] == {
        "processor_autogaze_forward_batched_total_ms": 9.5,
        "processor_autogaze_forward_batched_count": 2.5,
        "processor_autogaze_forward_batched_mean_ms": ((11.0 / 3.0) + 4.0) / 2.0,
        "processor_autogaze_total_total_ms": 12.0,
        "processor_autogaze_total_count": 1.0,
    }
    assert summary["readable_performance_summary"]["tokens_median"]["llm_visual_tokens_after_actual"] == 35.0


def test_read_manifest_file_supports_jsonl_and_validates_rows(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        '{"question_id": 1, "category": "av", "video_path": "clip.mp4", '
        '"question": "Q? A. a B. b C. c D. d", "answer": "A"}\n'
    )

    rows = read_manifest_file(manifest)

    assert rows == [
        {
            "question_id": 1,
            "category": "av",
            "video_path": "clip.mp4",
            "question": "Q? A. a B. b C. c D. d",
            "answer": "A",
        }
    ]


def test_viewer_row_to_manifest_uses_dataset_viewer_row_payload():
    payload = {
        "row_idx": 0,
        "row": {
            "question_id": 7,
            "category": "av",
            "video_path": "clip_av_video_5_001.mp4",
            "question": "What text is visible? A. A B. B C. C D. D",
            "answer": "C",
            "extra": "ignored",
        },
    }

    assert viewer_row_to_manifest(payload) == {
        "question_id": 7,
        "category": "av",
        "video_path": "clip_av_video_5_001.mp4",
        "question": "What text is visible? A. A B. B C. C D. D",
        "answer": "C",
    }
