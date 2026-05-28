from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from repro.plugins.autogaze_sparse_selector import AutogazeSelectorRuntimeConfig
from repro.plugins.gaze_plan import (
    EncoderMapping,
    MllmMapping,
    PatchSpace,
    PreprocessSpace,
    SelectedPatch,
    SourceVideo,
    SparseSelectionPlan,
)
from repro.plugins.vjepa_mapping import VjepaGridConfig, VjepaTokenSelection
from repro.vjepa_qwen_runner import (
    _vjepa_patch_embeddings,
    build_parser,
    build_selector_config_from_args,
    pil_frames_to_vjepa_pixel_values,
    vjepa_resize_plan_from_args,
    write_vjepa_qwen_visualization_artifacts,
)


def test_vjepa_qwen_runner_defaults_wire_actual_autogaze():
    args = build_parser().parse_args(["--video", "inputs/example.mp4"])

    assert args.video == "inputs/example.mp4"
    assert args.autogaze_mode == "on"
    assert args.autogaze_model == "nvidia/AutoGaze"
    assert args.vjepa_model == "facebook/vjepa2-vitl-fpc64-256"
    assert args.qwen_model == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert args.frames_per_clip == 16
    assert args.num_video_frames == 16
    assert args.autogaze_chunk_frames == 16
    assert args.autogaze_tile_size == 224
    assert args.autogaze_target_scales == "32+64+112+224"
    assert args.vjepa_selection_policy == "single_scale_union"
    assert args.visualization_max_frames == 16
    assert args.output_json.endswith("vjepa_qwen_actual.json")


def test_vjepa_qwen_runner_accepts_dense_off_mode_without_changing_vjepa_defaults():
    args = build_parser().parse_args(["--video", "inputs/example.mp4", "--autogaze-mode", "off"])

    assert args.autogaze_mode == "off"
    assert args.vjepa_selection_policy == "single_scale_union"
    assert args.frames_per_clip == 16
    assert args.crop_size == 224


def test_build_selector_config_from_args_matches_video_sampling_and_resize(tmp_path):
    args = SimpleNamespace(
        video="inputs/example.mp4",
        output_json=str(tmp_path / "actual.json"),
        autogaze_selector_output_json=None,
        sparse_selection_plan_json=None,
        autogaze_repo=".",
        autogaze_model="weights/AutoGaze",
        autogaze_device="cuda",
        autogaze_dtype="float16",
        num_video_frames=32,
        num_video_frames_thumbnail=0,
        qwen_thumbnail_mode="none",
        autogaze_chunk_frames=16,
        max_tiles_video=4,
        autogaze_tile_size=224,
        max_batch_size_autogaze=8,
        gazing_ratio=0.1,
        task_loss_requirement=None,
        autogaze_target_scales="112+224",
        autogaze_target_patch_size=16,
        autogaze_encoder_patch_size=16,
        autogaze_generate_only=True,
        video_decode_strategy="seek",
        video_resize_shortest_edge=None,
        video_resize_longest_edge=448,
        video_resize_width=None,
        video_resize_height=None,
    )

    config = build_selector_config_from_args(args)

    assert isinstance(config, AutogazeSelectorRuntimeConfig)
    assert config.autogaze_model == "weights/AutoGaze"
    assert config.num_video_frames == 32
    assert config.chunk_frames == 16
    assert config.max_tiles_video == 4
    assert config.max_batch_size == 8
    assert config.gazing_ratio == 0.1
    assert config.target_scales == [112, 224]
    assert config.generate_only is True
    assert config.video_decode_strategy == "seek"
    assert config.video_resize_longest_edge == 448
    assert config.output_json.endswith("actual_autogaze_sparse_plan.json")


def test_pil_frames_to_vjepa_pixel_values_shape_and_dtype():
    torch = pytest.importorskip("torch")
    frames = [
        Image.new("RGB", (32, 24), color=(255, 0, 0)),
        Image.new("RGB", (24, 32), color=(0, 255, 0)),
    ]

    values = pil_frames_to_vjepa_pixel_values(
        frames,
        crop_size=16,
        dtype=torch.float32,
        device="cpu",
    )

    assert list(values.shape) == [1, 2, 3, 16, 16]
    assert values.dtype == torch.float32
    assert values.device.type == "cpu"


def test_runner_vjepa_patch_embedding_boundary_supports_wrapper_embeddings():
    torch = pytest.importorskip("torch")

    class FakeEmbeddings:
        patch_embeddings = object()

        def __init__(self):
            self.seen_shape = None

        def __call__(self, values):
            self.seen_shape = list(values.shape)
            return values

    embeddings = FakeEmbeddings()
    model = SimpleNamespace(encoder=SimpleNamespace(embeddings=embeddings))
    values = torch.zeros((1, 3, 16, 8, 8), dtype=torch.float32)

    output = _vjepa_patch_embeddings(model, values)

    assert embeddings.seen_shape == [1, 16, 3, 8, 8]
    assert list(output.shape) == [1, 16, 3, 8, 8]


def test_runner_vjepa_patch_embedding_boundary_supports_direct_patch_embeddings():
    torch = pytest.importorskip("torch")

    class FakePatchEmbeddings:
        def __init__(self):
            self.seen_shape = None

        def __call__(self, values):
            self.seen_shape = list(values.shape)
            return values

    embeddings = FakePatchEmbeddings()
    model = SimpleNamespace(encoder=SimpleNamespace(embeddings=embeddings))
    values = torch.zeros((1, 16, 3, 8, 8), dtype=torch.float32)

    output = _vjepa_patch_embeddings(model, values)

    assert embeddings.seen_shape == [1, 3, 16, 8, 8]
    assert list(output.shape) == [1, 3, 16, 8, 8]


def test_vjepa_resize_plan_prefers_exact_crop_for_encoder_inputs():
    args = SimpleNamespace(
        video_resize_shortest_edge=None,
        video_resize_longest_edge=448,
        video_resize_width=None,
        video_resize_height=None,
        crop_size=224,
    )

    resize = vjepa_resize_plan_from_args(args)

    assert resize == {"width": 224, "height": 224, "mode": "exact"}


def test_write_vjepa_qwen_visualization_artifacts_saves_grid_mask_and_overlay(tmp_path):
    frames = [
        Image.new("RGB", (64, 64), color=(255, 255, 255)),
        Image.new("RGB", (64, 64), color=(220, 220, 220)),
    ]
    plan = SparseSelectionPlan(
        selector_name="autogaze",
        source_video=SourceVideo(path="video.mp4", sampled_frame_indices=[0, 1]),
        preprocess_space=PreprocessSpace(resized_width=64, resized_height=64, tile_size=64),
        patch_space=PatchSpace(autogaze_patch_size=16, encoder_patch_size=16, scale_ids=[0], scale_sizes=[64]),
        selected_patches=[
            SelectedPatch(
                frame_index=0,
                frame_order=0,
                tile_id=0,
                scale_id=0,
                scale_size=64,
                patch_index=0,
                bbox_resized_xyxy=[0, 0, 32, 32],
                bbox_original_xyxy=[0.0, 0.0, 32.0, 32.0],
                autoregressive_order=0,
            )
        ],
        encoder_mapping=EncoderMapping(status="mapped"),
        mllm_mapping=MllmMapping(status="mapped"),
        raw_patch_tokens=16,
        selected_patch_tokens=1,
    )
    selection = VjepaTokenSelection(
        status="mapped",
        grid_config=VjepaGridConfig(frames_per_clip=2, tubelet_size=2, crop_size=64, patch_size=16),
        selected_token_indices=[0, 1],
        selected_tokens_by_scale={"0": 2},
    )

    artifacts = write_vjepa_qwen_visualization_artifacts(
        frames=frames,
        sparse_plan=plan,
        selection=selection,
        output_dir=tmp_path,
        run_label="ag_on",
        autogaze_mode="on",
        crop_size=64,
        patch_size=16,
        max_frames=2,
    )

    assert artifacts["status"] == "written"
    assert Path(artifacts["selected_frames_grid_image"]).exists()
    assert Path(artifacts["vjepa_token_mask_image"]).exists()
    assert Path(artifacts["autogaze_overlay_image"]).exists()
    assert artifacts["autogaze_overlay_status"] == "written"
    assert artifacts["token_mask"]["selected_tokens"] == 2
