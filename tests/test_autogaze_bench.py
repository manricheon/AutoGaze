import torch
from pathlib import Path

from repro.autogaze_bench import (
    add_external_autogaze,
    autogaze_transform_kwargs,
    flatten_video_batch_for_siglip_baseline,
    move_model_to_device_dtype,
    repeat_video_batch,
    select_siglip_vision_model_class,
    summarize_gaze,
)


def test_official_autogaze_tree_is_available_at_repository_root():
    root = Path(__file__).resolve().parents[1]

    assert (root / "autogaze").is_dir()
    assert (root / "assets" / "example_input.mp4").is_file()


def test_add_external_autogaze_accepts_repository_root_layout():
    root = Path(__file__).resolve().parents[1]

    add_external_autogaze(str(root))


def test_summarize_gaze_counts_selected_and_padded_positions_from_lists():
    gaze_outputs = {
        "if_padded_gazing": [[False, True, False, False]],
        "num_gazing_each_frame": [2, 2],
    }

    summary = summarize_gaze(gaze_outputs, raw_patch_budget=16)

    assert summary["raw_patch_budget"] == 16
    assert summary["selected_non_padded_patches"] == 3
    assert summary["padded_gazing_positions"] == 1
    assert summary["total_gaze_slots"] == 4
    assert summary["token_reduction_ratio"] == 16 / 3
    assert summary["num_gazing_each_frame"] == [2, 2]


def test_summarize_gaze_flattens_batched_num_gazing_each_frame():
    gaze_outputs = {
        "if_padded_gazing": [[False, True], [False, False]],
        "num_gazing_each_frame": [[2, 3], [4, 5]],
    }

    summary = summarize_gaze(gaze_outputs, raw_patch_budget=32)

    assert summary["selected_non_padded_patches"] == 3
    assert summary["num_gazing_each_frame"] == [2, 3, 4, 5]


def test_flatten_video_batch_for_siglip_baseline_turns_frames_into_batch():
    video = torch.zeros(2, 3, 4, 5, 6)

    flattened = flatten_video_batch_for_siglip_baseline(video)

    assert list(flattened.shape) == [6, 4, 5, 6]


def test_repeat_video_batch_repeats_single_quickstart_clip():
    video = torch.zeros(1, 2, 3, 4, 5)

    repeated = repeat_video_batch(video, batch_size=3)

    assert list(repeated.shape) == [3, 2, 3, 4, 5]


def test_autogaze_transform_kwargs_force_square_largest_target_scale():
    assert autogaze_transform_kwargs([56, 112, 196, 392]) == {
        "size": {"height": 392, "width": 392},
        "crop_size": {"height": 392, "width": 392},
    }
    assert autogaze_transform_kwargs(None) == {}


def test_move_model_to_device_dtype_passes_dtype_to_torch_module_to():
    class FakeModel:
        def __init__(self):
            self.to_kwargs = None

        def to(self, **kwargs):
            self.to_kwargs = kwargs
            return self

    model = FakeModel()

    moved = move_model_to_device_dtype(model, torch.device("cpu"), torch.float16)

    assert moved is model
    assert model.to_kwargs == {"device": torch.device("cpu"), "dtype": torch.float16}


def test_select_siglip_vision_model_class_uses_config_model_type():
    assert select_siglip_vision_model_class("siglip_vision_model", "siglip", "siglip2") == "siglip"
    assert select_siglip_vision_model_class("siglip2_vision_model", "siglip", "siglip2") == "siglip2"
