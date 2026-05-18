import subprocess

from repro.stream_profile_sweep import (
    SweepCandidate,
    build_recommendation_rows,
    default_candidate_matrix,
    rank_recommendations,
    run_candidates,
)


def test_default_candidate_matrix_includes_latency_and_quality_profiles():
    candidates = default_candidate_matrix()

    assert any(candidate.name == "fast_448p_1tile_64f" for candidate in candidates)
    assert any(candidate.name == "balanced_720p_8tile_128f" for candidate in candidates)
    assert any(candidate.name == "quality_native_48tile_256f" for candidate in candidates)


def test_build_recommendation_rows_estimates_tokens_memory_and_command_args():
    candidate = SweepCandidate(
        name="balanced_720p_8tile_128f",
        num_video_frames=128,
        num_video_frames_thumbnail=64,
        max_tiles_video=8,
        stream_chunk_frames=16,
        max_batch_size_autogaze=8,
        max_batch_size_siglip=16,
        video_resize_shortest_edge=720,
    )

    rows = build_recommendation_rows(
        video="clip.mp4",
        width=3840,
        height=2160,
        source_frames=9000,
        candidates=[candidate],
        device="cuda",
        stream_dtype="float16",
        out_dir="outputs/sweep",
    )

    row = rows[0]
    assert row["candidate"] == "balanced_720p_8tile_128f"
    assert row["effective_width"] == 1280
    assert row["effective_height"] == 720
    assert row["spatial_tiles"] == 8
    assert row["num_video_frames"] == 128
    assert row["num_video_frames_thumbnail"] == 64
    assert row["stream_chunk_frames"] == 16
    assert row["max_batch_size_autogaze"] == 8
    assert row["max_batch_size_siglip"] == 16
    assert row["llm_keep_all_visual_tokens_estimated"] == 128256
    assert row["encoder_raw_patch_tokens"] == 1153280
    assert "--mode" in row["command"]
    assert "stream-profile" in row["command"]


def test_rank_recommendations_filters_context_and_memory_then_prefers_latency():
    rows = [
        {
            "candidate": "too_many_tokens",
            "llm_keep_all_visual_tokens_estimated": 1000000,
            "streaming_raw_frame_buffer_bytes": 1,
            "streaming_autogaze_tile_tensor_bytes_per_batch": 1,
            "latency_proxy": 1.0,
        },
        {
            "candidate": "balanced",
            "llm_keep_all_visual_tokens_estimated": 30000,
            "streaming_raw_frame_buffer_bytes": 1024,
            "streaming_autogaze_tile_tensor_bytes_per_batch": 2048,
            "latency_proxy": 20.0,
        },
        {
            "candidate": "fast",
            "llm_keep_all_visual_tokens_estimated": 20000,
            "streaming_raw_frame_buffer_bytes": 1024,
            "streaming_autogaze_tile_tensor_bytes_per_batch": 1024,
            "latency_proxy": 10.0,
        },
    ]

    ranked = rank_recommendations(rows, max_visual_tokens=40000, max_stream_memory_bytes=4096)

    assert [row["candidate"] for row in ranked] == ["fast", "balanced"]


def test_run_candidates_records_timeout_as_failed_row(monkeypatch):
    def timeout_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=["python"], timeout=1)

    monkeypatch.setattr(subprocess, "run", timeout_run)

    rows = [
        {
            "candidate": "too_slow",
            "command": "python -m repro.nvila_runner",
            "output_json": "outputs/missing.json",
        }
    ]

    results = run_candidates(rows, timeout_seconds=1)

    assert results[0]["candidate"] == "too_slow"
    assert results[0]["status"] == "timeout"
    assert results[0]["timeout_seconds"] == 1
