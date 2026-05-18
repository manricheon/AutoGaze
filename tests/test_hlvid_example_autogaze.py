from repro.hlvid_example_autogaze import (
    chunked,
    nvila_thumbnail_indices,
    summarize_chunks,
    uniform_sample_indices,
)


def test_uniform_sample_indices_matches_nvila_round_linspace_policy():
    assert uniform_sample_indices(total_frames=10, sample_count=4) == [0, 3, 6, 9]


def test_uniform_sample_indices_repeats_single_available_frame():
    assert uniform_sample_indices(total_frames=1, sample_count=4) == [0, 0, 0, 0]


def test_chunked_splits_values_without_padding():
    assert list(chunked([0, 1, 2, 3, 4], 2)) == [[0, 1], [2, 3], [4]]


def test_nvila_thumbnail_indices_match_sampled_frame_step_policy():
    sampled = list(range(128))

    thumbnails = nvila_thumbnail_indices(sampled, thumbnail_count=64)

    assert thumbnails == list(range(0, 128, 2))


def test_summarize_chunks_accumulates_gaze_counts():
    summary = summarize_chunks(
        [
            {
                "tile_sequences": 10,
                "raw_patch_budget": 160,
                "selected_non_padded_patches": 40,
                "padded_gazing_positions": 4,
                "total_gaze_slots": 44,
                "autogaze_forward_ms": 10.0,
            },
            {
                "tile_sequences": 5,
                "raw_patch_budget": 80,
                "selected_non_padded_patches": 20,
                "padded_gazing_positions": 2,
                "total_gaze_slots": 22,
                "autogaze_forward_ms": 7.0,
            },
        ]
    )

    assert summary["tile_sequences"] == 15
    assert summary["raw_patch_budget"] == 240
    assert summary["selected_non_padded_patches"] == 60
    assert summary["token_reduction_ratio"] == 4.0
    assert summary["autogaze_forward_ms"] == 17.0
