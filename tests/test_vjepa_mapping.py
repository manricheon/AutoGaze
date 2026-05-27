from repro.plugins.gaze_plan import (
    EncoderMapping,
    MllmMapping,
    PatchSpace,
    PreprocessSpace,
    SelectedPatch,
    SourceVideo,
    SparseSelectionPlan,
)
from repro.plugins.vjepa_mapping import (
    VjepaGridConfig,
    scale_aware_vjepa_selection_from_sparse_plan,
    vjepa_token_selection_from_sparse_plan,
)


def _plan_with_patches(patches):
    return SparseSelectionPlan(
        selector_name="autogaze-direct",
        source_video=SourceVideo(
            path="inputs/example.mp4",
            source_width=224,
            source_height=224,
            sampled_frame_indices=[0, 1, 2, 3],
        ),
        preprocess_space=PreprocessSpace(
            resize_policy="unit-test",
            resized_width=224,
            resized_height=224,
        ),
        patch_space=PatchSpace(
            autogaze_patch_size=16,
            encoder_patch_size=16,
            scale_ids=[0, 1],
            scale_sizes=[112, 224],
        ),
        selected_patches=patches,
        encoder_mapping=EncoderMapping(status="not_mapped"),
        mllm_mapping=MllmMapping(status="not_mapped"),
        raw_patch_tokens=4 * ((112 // 16) ** 2 + (224 // 16) ** 2),
        selected_patch_tokens=len(patches),
    )


def test_vjepa_mapping_unions_autogaze_patches_inside_same_tubelet():
    plan = _plan_with_patches(
        [
            SelectedPatch(
                frame_index=0,
                frame_order=0,
                tile_id=0,
                scale_id=1,
                scale_size=224,
                patch_index=0,
                bbox_resized_xyxy=[0, 0, 16, 16],
                bbox_original_xyxy=[0.0, 0.0, 16.0, 16.0],
                autoregressive_order=0,
            ),
            SelectedPatch(
                frame_index=1,
                frame_order=1,
                tile_id=0,
                scale_id=1,
                scale_size=224,
                patch_index=1,
                bbox_resized_xyxy=[16, 0, 32, 16],
                bbox_original_xyxy=[16.0, 0.0, 32.0, 16.0],
                autoregressive_order=1,
            ),
            SelectedPatch(
                frame_index=1,
                frame_order=1,
                tile_id=0,
                scale_id=1,
                scale_size=224,
                patch_index=0,
                bbox_resized_xyxy=[0, 0, 16, 16],
                bbox_original_xyxy=[0.0, 0.0, 16.0, 16.0],
                autoregressive_order=2,
            ),
        ]
    )

    selection = vjepa_token_selection_from_sparse_plan(
        plan,
        VjepaGridConfig(frames_per_clip=4, tubelet_size=2, crop_size=224, patch_size=16),
    )

    assert selection.status == "mapped"
    assert selection.grid_thw == [2, 14, 14]
    assert selection.raw_token_count == 392
    assert selection.selected_token_indices == [0, 1]
    assert selection.selected_token_count == 2
    assert selection.mapping_policy["tubelet"] == "any_frame_selected"


def test_vjepa_mapping_expands_coarse_autogaze_bbox_by_overlap():
    plan = _plan_with_patches(
        [
            SelectedPatch(
                frame_index=0,
                frame_order=0,
                tile_id=0,
                scale_id=0,
                scale_size=112,
                patch_index=0,
                bbox_resized_xyxy=[0, 0, 32, 32],
                bbox_original_xyxy=[0.0, 0.0, 32.0, 32.0],
                autoregressive_order=0,
            )
        ]
    )

    selection = vjepa_token_selection_from_sparse_plan(
        plan,
        VjepaGridConfig(frames_per_clip=4, tubelet_size=2, crop_size=224, patch_size=16),
    )

    assert selection.selected_token_indices == [0, 1, 14, 15]
    assert selection.selected_token_count == 4
    assert selection.selected_tokens_by_scale == {"0": 4}
    assert selection.to_dict()["vjepa"]["reduction_ratio"] == 98.0


def test_scale_aware_vjepa_mapping_keeps_coarse_patch_in_coarse_grid():
    plan = _plan_with_patches(
        [
            SelectedPatch(
                frame_index=0,
                frame_order=0,
                tile_id=0,
                scale_id=0,
                scale_size=112,
                patch_index=0,
                bbox_resized_xyxy=[0, 0, 32, 32],
                bbox_original_xyxy=[0.0, 0.0, 32.0, 32.0],
                autoregressive_order=0,
            ),
            SelectedPatch(
                frame_index=0,
                frame_order=0,
                tile_id=0,
                scale_id=1,
                scale_size=224,
                patch_index=0,
                bbox_resized_xyxy=[0, 0, 16, 16],
                bbox_original_xyxy=[0.0, 0.0, 16.0, 16.0],
                autoregressive_order=1,
            ),
        ]
    )

    selection = scale_aware_vjepa_selection_from_sparse_plan(
        plan,
        frames_per_clip=4,
        tubelet_size=2,
        patch_size=16,
    )

    assert selection.status == "mapped"
    assert selection.raw_token_count == 490
    assert selection.selected_token_count == 2
    assert selection.selected_tokens_by_scale == {"0": 1, "1": 1}
    assert selection.to_dict()["vjepa"]["scale_passes"]["0"]["grid_thw"] == [2, 7, 7]
    assert selection.to_dict()["vjepa"]["scale_passes"]["1"]["grid_thw"] == [2, 14, 14]


def test_vjepa_mapping_handles_non_square_resize_into_square_crop():
    plan = SparseSelectionPlan(
        selector_name="autogaze-direct",
        source_video=SourceVideo(path="inputs/wide.mp4", sampled_frame_indices=[0, 1]),
        preprocess_space=PreprocessSpace(resized_width=448, resized_height=224),
        patch_space=PatchSpace(autogaze_patch_size=16, encoder_patch_size=16, scale_ids=[1], scale_sizes=[224]),
        selected_patches=[
            SelectedPatch(
                frame_index=0,
                frame_order=0,
                tile_id=0,
                scale_id=1,
                scale_size=224,
                patch_index=0,
                bbox_resized_xyxy=[224, 0, 448, 224],
                bbox_original_xyxy=[224.0, 0.0, 448.0, 224.0],
                autoregressive_order=0,
            )
        ],
        encoder_mapping=EncoderMapping(status="not_mapped"),
        mllm_mapping=MllmMapping(status="not_mapped"),
    )

    selection = vjepa_token_selection_from_sparse_plan(
        plan,
        VjepaGridConfig(frames_per_clip=2, tubelet_size=2, crop_size=224, patch_size=16),
    )

    assert selection.grid_thw == [1, 14, 14]
    assert selection.selected_token_count == 98
    assert selection.selected_token_indices[0] == 7
    assert selection.selected_token_indices[-1] == 195
