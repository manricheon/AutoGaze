from pathlib import Path

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
from scripts.build_colab_v2_visualization_assets import make_frame_grid, make_sparse_overlay_grid


def test_make_frame_grid_renders_all_16_frames():
    frames = [Image.new("RGB", (16, 16), color=(index, index, index)) for index in range(16)]

    image = make_frame_grid(frames)

    assert image.size == (64, 64)


def test_make_sparse_overlay_grid_uses_plan_patch_boxes(tmp_path):
    frames = [Image.new("RGB", (32, 32), color=(255, 255, 255)) for _ in range(16)]
    plan = SparseSelectionPlan(
        selector_name="autogaze",
        source_video=SourceVideo(path="video.mp4", sampled_frame_indices=list(range(16))),
        preprocess_space=PreprocessSpace(resized_width=32, resized_height=32, tile_size=32),
        patch_space=PatchSpace(autogaze_patch_size=16, encoder_patch_size=16, scale_ids=[0], scale_sizes=[32]),
        selected_patches=[
            SelectedPatch(
                frame_index=0,
                frame_order=0,
                tile_id=0,
                scale_id=0,
                scale_size=32,
                patch_index=0,
                bbox_resized_xyxy=[0, 0, 16, 16],
                bbox_original_xyxy=[0.0, 0.0, 16.0, 16.0],
                autoregressive_order=0,
            )
        ],
        encoder_mapping=EncoderMapping(status="mapped"),
        mllm_mapping=MllmMapping(status="mapped"),
        raw_patch_tokens=64,
        selected_patch_tokens=1,
    )

    image = make_sparse_overlay_grid(frames, plan)
    path = tmp_path / "overlay.png"
    image.save(path)

    assert path.exists()
    assert image.size == (128, 128)
