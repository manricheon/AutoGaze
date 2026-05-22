from repro.plugins.pre_vit_sparse import (
    build_llava_onevision_pre_vit_candidate,
    build_pre_vit_sparse_contract,
    pre_vit_sparse_model_matrix,
)


def test_pre_vit_sparse_contract_marks_qwen_as_ready_first():
    contract = build_pre_vit_sparse_contract("qwen3-vl")

    assert contract["model_family"] == "qwen3-vl"
    assert contract["difficulty"] == "low"
    assert contract["status"] == "implemented_pending_cuda"
    assert contract["selector_plan_format"] == "SparseSelectionPlan"
    assert contract["model_grid_fields"] == ["pixel_values_videos", "video_grid_thw", "spatial_merge_size"]
    assert contract["expected_gain"] == ["vision_encoder", "mllm_prefill", "kv_cache"]


def test_pre_vit_sparse_contract_marks_vila_and_internvl_as_probe_required():
    vila = build_pre_vit_sparse_contract("longvila")
    internvl = build_pre_vit_sparse_contract("internvl3")

    assert vila["difficulty"] == "medium_high"
    assert vila["status"] == "in_process_probe_required"
    assert "vision_tower.forward input tensor" in vila["required_hooks"]
    assert internvl["difficulty"] == "medium"
    assert internvl["status"] == "dynamic_tile_probe_required"
    assert "num_patches_list" in internvl["model_grid_fields"]


def test_pre_vit_sparse_model_matrix_orders_next_work():
    matrix = pre_vit_sparse_model_matrix()

    assert [row["model_family"] for row in matrix[:4]] == [
        "qwen2.5-vl",
        "qwen3-vl",
        "nvila-video-plugin",
        "longvila",
    ]
    assert matrix[0]["priority"] == 1
    assert matrix[-1]["difficulty"] == "high"


def test_llava_onevision_candidate_prefers_frame_or_tile_level_before_patch_level():
    candidate = build_llava_onevision_pre_vit_candidate()

    assert candidate["model_family"] == "llava-onevision"
    assert candidate["recommended_first_pre_vit_unit"] == "frame_or_tile"
    assert candidate["patch_level_status"] == "hard_due_to_video_pooling"
    assert "do not claim ViT patch-level gain until pooling boundary is bypassed" in candidate["guardrails"]
