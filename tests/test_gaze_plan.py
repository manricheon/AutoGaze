from repro.plugins.gaze_plan import (
    EncoderMapping,
    MllmMapping,
    PatchSpace,
    PreprocessSpace,
    SelectedPatch,
    SourceVideo,
    SparseSelectionPlan,
    qwen_visual_indices_from_sparse_plan,
    sparse_selection_plan_from_dict,
)


def test_sparse_selection_plan_serializes_patch_encoder_and_mllm_mapping():
    plan = SparseSelectionPlan(
        selector_name="autogaze",
        source_video=SourceVideo(
            path="inputs/example.mp4",
            source_width=1920,
            source_height=1080,
            sampled_frame_indices=[0, 8, 15],
            sampled_fps=1.5,
        ),
        preprocess_space=PreprocessSpace(
            resize_policy="shortest_edge=720",
            resized_width=1280,
            resized_height=720,
            tile_grid=[2, 1],
            tile_size=392,
        ),
        patch_space=PatchSpace(
            autogaze_patch_size=16,
            encoder_patch_size=14,
            scale_ids=[0, 1],
            scale_sizes=[196, 392],
        ),
        selected_patches=[
            SelectedPatch(
                frame_index=8,
                frame_order=1,
                tile_id=0,
                scale_id=1,
                scale_size=392,
                patch_index=17,
                bbox_resized_xyxy=[10, 20, 30, 40],
                bbox_original_xyxy=[15.0, 30.0, 45.0, 60.0],
                autoregressive_order=3,
            )
        ],
        encoder_mapping=EncoderMapping(
            status="approximate",
            encoder_grid_thw=[3, 28, 28],
            encoder_patch_indices=[33],
            position_ids={"t": [1], "h": [2], "w": [5]},
            reason="patch size mismatch requires bbox overlap mapping",
        ),
        mllm_mapping=MllmMapping(
            status="probe_required",
            visual_feature_indices=[33],
            projected_token_indices=None,
            llm_context_indices=None,
            reason="Qwen visual insertion boundary must be probed",
        ),
        raw_patch_tokens=3000,
        selected_patch_tokens=300,
    )

    payload = plan.to_dict()

    assert payload["selector_name"] == "autogaze"
    assert payload["patch_space"]["patch_size_mismatch"] is True
    assert payload["token_accounting"]["raw_patch_tokens"] == 3000
    assert payload["token_accounting"]["selected_patch_tokens"] == 300
    assert payload["token_accounting"]["reduction_ratio"] == 10.0
    assert payload["selected_patches"][0]["bbox_resized_xyxy"] == [10, 20, 30, 40]
    assert payload["encoder_mapping"]["encoder_grid_thw"] == [3, 28, 28]
    assert payload["mllm_mapping"]["status"] == "probe_required"


def test_sparse_selection_plan_can_build_empty_autogaze_placeholder():
    plan = SparseSelectionPlan.placeholder(
        selector_name="autogaze",
        source_path="inputs/example.mp4",
        raw_patch_tokens=1000,
        selected_patch_tokens=100,
        frame_indices=[0, 1],
        reason="standalone AutoGaze selector has not emitted concrete patch coordinates yet",
    )

    payload = plan.to_dict()

    assert payload["selector_name"] == "autogaze"
    assert payload["source_video"]["sampled_frame_indices"] == [0, 1]
    assert payload["selected_patches"] == []
    assert payload["encoder_mapping"]["status"] == "not_mapped"
    assert payload["mllm_mapping"]["status"] == "not_mapped"
    assert payload["quality_control"]["reason"] == "standalone AutoGaze selector has not emitted concrete patch coordinates yet"


def test_qwen_visual_mapping_uses_exact_frame_grid_patch_order():
    plan = SparseSelectionPlan(
        selector_name="autogaze",
        source_video=SourceVideo(path="inputs/example.mp4", sampled_frame_indices=[0, 8]),
        preprocess_space=PreprocessSpace(resized_width=32, resized_height=32),
        patch_space=PatchSpace(
            autogaze_patch_size=16,
            encoder_patch_size=16,
            scale_ids=[0],
            scale_sizes=[32],
        ),
        selected_patches=[
            SelectedPatch(
                frame_index=0,
                frame_order=0,
                tile_id=0,
                scale_id=0,
                scale_size=32,
                patch_index=1,
                bbox_resized_xyxy=[16, 0, 32, 16],
                bbox_original_xyxy=[16.0, 0.0, 32.0, 16.0],
                autoregressive_order=1,
            ),
            SelectedPatch(
                frame_index=8,
                frame_order=1,
                tile_id=0,
                scale_id=0,
                scale_size=32,
                patch_index=2,
                bbox_resized_xyxy=[0, 16, 16, 32],
                bbox_original_xyxy=[0.0, 16.0, 16.0, 32.0],
                autoregressive_order=2,
            ),
        ],
        encoder_mapping=EncoderMapping(status="not_mapped"),
        mllm_mapping=MllmMapping(status="not_mapped"),
    )

    mapping = qwen_visual_indices_from_sparse_plan(plan, video_grid_thw=[2, 2, 2])

    assert mapping.status == "exact_grid"
    assert mapping.visual_feature_indices == [1, 6]
    assert mapping.reason == "mapped 2 AutoGaze patches to 2 Qwen visual feature indices"


def test_qwen_visual_mapping_uses_bbox_centers_when_patch_spaces_differ():
    plan = SparseSelectionPlan(
        selector_name="autogaze",
        source_video=SourceVideo(path="inputs/example.mp4", sampled_frame_indices=[0]),
        preprocess_space=PreprocessSpace(resized_width=64, resized_height=64),
        patch_space=PatchSpace(
            autogaze_patch_size=16,
            encoder_patch_size=14,
            scale_ids=[0],
            scale_sizes=[64],
        ),
        selected_patches=[
            SelectedPatch(
                frame_index=0,
                frame_order=0,
                tile_id=0,
                scale_id=0,
                scale_size=64,
                patch_index=5,
                bbox_resized_xyxy=[16, 16, 32, 32],
                bbox_original_xyxy=[16.0, 16.0, 32.0, 32.0],
                autoregressive_order=1,
            )
        ],
        encoder_mapping=EncoderMapping(status="not_mapped"),
        mllm_mapping=MllmMapping(status="not_mapped"),
    )

    mapping = qwen_visual_indices_from_sparse_plan(plan, video_grid_thw=[1, 4, 4])

    assert mapping.status == "approximate_bbox"
    assert mapping.visual_feature_indices == [5]
    assert "bbox center" in mapping.reason


def test_sparse_selection_plan_round_trips_from_dict_for_qwen_mapping():
    original = SparseSelectionPlan.placeholder(
        selector_name="autogaze",
        source_path="inputs/example.mp4",
        raw_patch_tokens=16,
        selected_patch_tokens=1,
        frame_indices=[0],
        reason="round trip",
    ).to_dict()
    original["preprocess_space"]["resized_width"] = 64
    original["preprocess_space"]["resized_height"] = 64
    original["patch_space"]["autogaze_patch_size"] = 16
    original["patch_space"]["encoder_patch_size"] = 16
    original["patch_space"]["scale_sizes"] = [64]
    original["selected_patches"] = [
        {
            "frame_index": 0,
            "frame_order": 0,
            "tile_id": 0,
            "scale_id": 0,
            "scale_size": 64,
            "patch_index": 10,
            "bbox_resized_xyxy": [32, 32, 48, 48],
            "bbox_original_xyxy": [32.0, 32.0, 48.0, 48.0],
            "autoregressive_order": 1,
        }
    ]

    plan = sparse_selection_plan_from_dict(original)
    mapping = qwen_visual_indices_from_sparse_plan(plan, video_grid_thw=[1, 4, 4])

    assert plan.selected_patches[0].patch_index == 10
    assert mapping.visual_feature_indices == [10]
