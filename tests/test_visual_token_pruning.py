from repro.plugins.visual_token_pruning import (
    build_post_encoder_prune_result,
    build_pre_encoder_sparse_probe,
    prune_sequence_by_indices,
)


def test_prune_sequence_by_indices_keeps_selected_visual_tokens():
    features = ["v0", "v1", "v2", "v3", "v4"]

    assert prune_sequence_by_indices(features, [0, 2, 4]) == ["v0", "v2", "v4"]


def test_post_encoder_prune_result_reports_mllm_only_gain():
    result = build_post_encoder_prune_result(raw_visual_tokens=1000, selected_indices=list(range(100)))

    assert result.raw_visual_tokens == 1000
    assert result.selected_visual_tokens == 100
    assert result.reduction_ratio == 10.0
    assert result.vision_encoder_latency_reduced is False
    assert result.mllm_context_reduced is True
    assert result.to_dict()["expected_gain"] == "mllm_context_only"


def test_pre_encoder_sparse_probe_requires_position_grid_semantics():
    result = build_pre_encoder_sparse_probe(
        model_family="qwen3-vl",
        raw_patch_tokens=1000,
        selected_indices=list(range(90)),
        position_grid_fields=["video_grid_thw", "mm_token_type_ids"],
    )

    assert result.model_family == "qwen3-vl"
    assert result.selected_patch_tokens == 90
    assert result.reduction_ratio == 1000 / 90
    assert result.status == "requires_model_specific_probe"
    assert "video_grid_thw" in result.required_semantics
