from repro.qwen_sparse_preflight import estimate_qwen_sparse_tokens, estimate_qwen_plugin_preflight


def test_estimate_qwen_sparse_tokens_matches_patch_and_merged_grid_counts():
    estimate = estimate_qwen_sparse_tokens(
        num_frames=16,
        height=224,
        width=224,
        patch_size=14,
        spatial_merge_size=2,
        autogaze_reduction_ratio=4.0,
        chunk_frames=8,
    )

    assert estimate["video_grid_thw"] == [16, 16, 16]
    assert estimate["raw_patch_tokens_before_vit"] == 4096
    assert estimate["visual_tokens_before_prune"] == 1024
    assert estimate["selected_patch_tokens_after_autogaze"] == 1024
    assert estimate["visual_tokens_after_prune"] == 256
    assert estimate["chunk_count"] == 2
    assert estimate["vit_token_reduction_ratio"] == 4.0
    assert estimate["llm_visual_token_reduction_ratio"] == 4.0


def test_estimate_qwen_plugin_preflight_flags_context_and_memory_risk():
    estimate = estimate_qwen_plugin_preflight(
        model_family="qwen3-vl",
        num_frames=128,
        height=720,
        width=1280,
        autogaze_reduction_ratio=10.0,
        prompt_tokens=512,
        max_new_tokens=128,
        context_limit=4096,
        h100_budget_gib=20.0,
        resident_model_gib=18.0,
    )

    assert estimate["model_family"] == "qwen3-vl"
    assert estimate["token_estimate"]["raw_patch_tokens_before_vit"] > estimate["token_estimate"]["selected_patch_tokens_after_autogaze"]
    assert estimate["llm_context_tokens_estimated"] > 4096
    assert estimate["risk"]["context"] == "red"
    assert estimate["risk"]["memory"] in {"yellow", "red"}
