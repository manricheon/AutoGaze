from PIL import Image

from repro.plugins.gaze_plan import (
    EncoderMapping,
    MllmMapping,
    PatchSpace,
    PreprocessSpace,
    SelectedPatch,
    SourceVideo,
    SparseSelectionPlan,
)
from repro.plugins.materialized_sparse_video import build_materialized_sparse_frames


def make_plan() -> SparseSelectionPlan:
    return SparseSelectionPlan(
        selector_name="autogaze-direct",
        source_video=SourceVideo(path="clip.mp4", sampled_frame_indices=[10, 20, 30]),
        preprocess_space=PreprocessSpace(resized_width=100, resized_height=80),
        patch_space=PatchSpace(autogaze_patch_size=16, encoder_patch_size=16, scale_sizes=[64]),
        selected_patches=[
            SelectedPatch(
                frame_index=20,
                frame_order=1,
                tile_id=0,
                scale_id=0,
                scale_size=64,
                patch_index=0,
                bbox_resized_xyxy=[10, 10, 40, 50],
                bbox_original_xyxy=[10.0, 10.0, 40.0, 50.0],
                autoregressive_order=0,
            ),
            SelectedPatch(
                frame_index=30,
                frame_order=2,
                tile_id=0,
                scale_id=0,
                scale_size=64,
                patch_index=1,
                bbox_resized_xyxy=[60, 20, 90, 70],
                bbox_original_xyxy=[60.0, 20.0, 90.0, 70.0],
                autoregressive_order=1,
            ),
        ],
        encoder_mapping=EncoderMapping(status="not_mapped"),
        mllm_mapping=MllmMapping(status="not_mapped"),
        raw_patch_tokens=300,
        selected_patch_tokens=2,
    )


def test_build_materialized_sparse_frames_keeps_selected_orders_and_crops_to_union_boxes():
    frames = [
        Image.new("RGB", (100, 80), "red"),
        Image.new("RGB", (100, 80), "green"),
        Image.new("RGB", (100, 80), "blue"),
    ]

    materialized = build_materialized_sparse_frames(frames, make_plan(), crop_to_selection=True)

    assert materialized.metadata["integration_claim"] == "materialized_sparse_video"
    assert materialized.metadata["original_sampled_frame_count"] == 3
    assert materialized.metadata["kept_frame_orders"] == [1, 2]
    assert materialized.metadata["kept_source_frame_indices"] == [20, 30]
    assert materialized.metadata["coarse_pre_vit_input_reduced"] is True
    assert len(materialized.frames) == 2
    assert materialized.frames[0].size == materialized.frames[1].size
    assert materialized.metadata["crop_boxes_resized_xyxy"] == [[10, 10, 40, 50], [60, 20, 90, 70]]


def test_build_materialized_sparse_frames_falls_back_to_first_frame_when_plan_has_no_patches():
    empty = SparseSelectionPlan.placeholder(
        selector_name="autogaze-direct",
        source_path="clip.mp4",
        raw_patch_tokens=100,
        selected_patch_tokens=0,
        frame_indices=[5, 6],
        reason="empty selector output",
    )
    frames = [Image.new("RGB", (32, 32), "white"), Image.new("RGB", (32, 32), "black")]

    materialized = build_materialized_sparse_frames(frames, empty, crop_to_selection=True)

    assert len(materialized.frames) == 1
    assert materialized.metadata["kept_frame_orders"] == [0]
    assert materialized.metadata["fallback_reason"] == "no_selected_patches"
