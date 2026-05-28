import json
from types import SimpleNamespace

import torch

from repro.plugins.autogaze_sparse_selector import (
    AutogazeSelectorRuntimeConfig,
    autogaze_selection_policy_summary,
    build_autogaze_selector_video_plan,
    build_sparse_selection_plan_from_autogaze_outputs,
    ensure_transformers_tied_weight_compat,
    patch_autogaze_inputs_embeds_generate_compat,
    runtime_config_from_args,
)
from repro.plugins.gaze_plan import qwen_visual_indices_from_sparse_plan
from repro.nvila_runner import spatial_tile_grid


def test_build_sparse_plan_decodes_autogaze_multiscale_positions():
    outputs = {
        "gazing_pos": torch.tensor([[0, 17, 20, 35]]),
        "if_padded_gazing": torch.tensor([[False, False, True, False]]),
        "num_gazing_each_frame": torch.tensor([2, 2]),
    }

    plan = build_sparse_selection_plan_from_autogaze_outputs(
        outputs,
        source_path="inputs/example.mp4",
        frame_indices=[10, 20],
        target_scales=[32, 64],
        target_patch_size=16,
        encoder_patch_size=16,
        resized_width=64,
        resized_height=64,
    )

    assert plan.selector_name == "autogaze-direct"
    assert plan.raw_patch_tokens == 40
    assert plan.selected_patch_tokens == 3
    assert [patch.frame_order for patch in plan.selected_patches] == [0, 0, 1]
    assert [patch.frame_index for patch in plan.selected_patches] == [10, 10, 20]
    assert [patch.scale_id for patch in plan.selected_patches] == [0, 1, 1]
    assert [patch.patch_index for patch in plan.selected_patches] == [0, 13, 11]
    assert plan.selected_patches[0].bbox_resized_xyxy == [0, 0, 32, 32]
    assert plan.selected_patches[1].bbox_resized_xyxy == [16, 48, 32, 64]


def test_build_sparse_plan_offsets_patch_bbox_by_spatial_tile_id():
    outputs = {
        "gazing_pos": torch.tensor([[0], [0]]),
        "if_padded_gazing": torch.tensor([[False], [False]]),
        "num_gazing_each_frame": torch.tensor([1]),
    }

    plan = build_sparse_selection_plan_from_autogaze_outputs(
        outputs,
        source_path="inputs/example.mp4",
        frame_indices=[0],
        target_scales=[64],
        target_patch_size=16,
        encoder_patch_size=16,
        resized_width=64,
        resized_height=64,
        tile_grid=[2, 1],
        tile_size=64,
    )

    assert [patch.tile_id for patch in plan.selected_patches] == [0, 1]
    assert plan.selected_patches[0].bbox_resized_xyxy == [0, 0, 16, 16]
    assert plan.selected_patches[1].bbox_resized_xyxy == [64, 0, 80, 16]


def test_qwen_mapping_spreads_autogaze_frames_over_qwen_temporal_grid():
    outputs = {
        "gazing_pos": torch.tensor([[0, 16, 32, 48]]),
        "if_padded_gazing": torch.tensor([[False, False, False, False]]),
        "num_gazing_each_frame": torch.tensor([1, 1, 1, 1]),
    }
    plan = build_sparse_selection_plan_from_autogaze_outputs(
        outputs,
        source_path="inputs/example.mp4",
        frame_indices=[0, 8, 16, 24],
        target_scales=[64],
        target_patch_size=16,
        encoder_patch_size=16,
        resized_width=64,
        resized_height=64,
    )

    mapping = qwen_visual_indices_from_sparse_plan(plan, video_grid_thw=[2, 4, 4])

    assert mapping.visual_feature_indices == [0, 16]


def test_sparse_plan_can_be_written_for_qwen_adapter(tmp_path):
    outputs = {
        "gazing_pos": [[5]],
        "if_padded_gazing": [[False]],
        "num_gazing_each_frame": [1],
    }
    plan = build_sparse_selection_plan_from_autogaze_outputs(
        outputs,
        source_path="inputs/example.mp4",
        frame_indices=[0],
        target_scales=[64],
        target_patch_size=16,
        encoder_patch_size=16,
        resized_width=64,
        resized_height=64,
    )
    target = tmp_path / "sparse_plan.json"
    target.write_text(json.dumps(plan.to_dict()))

    payload = json.loads(target.read_text())

    assert payload["selector_name"] == "autogaze-direct"
    assert payload["selected_patches"][0]["patch_index"] == 5
    assert payload["token_accounting"]["reduction_ratio"] == 16.0


def test_runtime_config_from_args_carries_runner_video_resize_options():
    config = runtime_config_from_args(
        SimpleNamespace(
            video="inputs/example.mp4",
            output_json="outputs/run.json",
            autogaze_target_scales="32+64+112+224",
            video_resize_shortest_edge=None,
            video_resize_longest_edge=448,
            video_resize_width=None,
            video_resize_height=None,
        )
    )

    assert config.video_resize_shortest_edge is None
    assert config.video_resize_longest_edge == 448
    assert config.video_resize_width is None
    assert config.video_resize_height is None


def test_runtime_config_keeps_gazing_ratio_none_for_checkpoint_default():
    config = runtime_config_from_args(
        SimpleNamespace(
            video="inputs/example.mp4",
            output_json="outputs/run.json",
            autogaze_target_scales="32+64+112+224",
        )
    )

    assert config.gazing_ratio is None


def test_autogaze_selection_policy_explains_checkpoint_default_task_loss():
    class FakeModel:
        gazing_ratio_config = {"fixed": {"gazing_ratio": 0.75}}
        has_task_loss_requirement_during_inference = True
        task_loss_requirement_config = {"fixed": {"task_loss_requirement": 0.7}}

    summary = autogaze_selection_policy_summary(
        requested_gazing_ratio=None,
        requested_task_loss_requirement=None,
        model=FakeModel(),
    )

    assert summary["policy"] == "model_default"
    assert summary["model_gazing_ratio_config"]["fixed"]["gazing_ratio"] == 0.75
    assert summary["model_has_task_loss_requirement_during_inference"] is True
    assert "task-loss early stop" in summary["note"]


def test_autogaze_selection_policy_explains_explicit_ratio_disables_task_loss():
    summary = autogaze_selection_policy_summary(
        requested_gazing_ratio=0.1,
        requested_task_loss_requirement=None,
    )

    assert summary["policy"] == "fixed_ratio_no_task_loss"
    assert summary["requested_gazing_ratio"] == 0.1
    assert "disables task-loss" in summary["note"]


def test_autogaze_selector_video_plan_uses_resized_dimensions_for_tile_grid():
    config = AutogazeSelectorRuntimeConfig(
        video="inputs/example.mp4",
        output_json="outputs/plan.json",
        max_tiles_video=8,
        tile_size=224,
        video_resize_longest_edge=448,
    )

    plan = build_autogaze_selector_video_plan(
        config,
        {"width": 3840, "height": 2160, "frames": 100, "fps": 30.0},
    )

    assert plan["resize"]["enabled"] is True
    assert plan["resize"]["effective"] == {"width": 448, "height": 252, "mode": "longest_edge"}
    assert plan["effective_width"] == 448
    assert plan["effective_height"] == 252
    assert plan["grid"] == spatial_tile_grid(width=448, height=252, max_tiles_video=8, image_size=224)


def test_autogaze_selector_video_plan_uses_source_dimensions_without_resize():
    config = AutogazeSelectorRuntimeConfig(
        video="inputs/example.mp4",
        output_json="outputs/plan.json",
        max_tiles_video=8,
        tile_size=224,
    )

    plan = build_autogaze_selector_video_plan(
        config,
        {"width": 3840, "height": 2160, "frames": 100, "fps": 30.0},
    )

    assert plan["resize"]["enabled"] is False
    assert plan["resize"]["effective"] == {"width": 3840, "height": 2160, "mode": "none"}
    assert plan["grid"] == spatial_tile_grid(width=3840, height=2160, max_tiles_video=8, image_size=224)


def test_ensure_transformers_tied_weight_compat_adds_newer_transformers_attrs():
    class LegacyAutoGaze:
        pass

    ensure_transformers_tied_weight_compat(LegacyAutoGaze)

    assert LegacyAutoGaze.all_tied_weights_keys == {}
    assert LegacyAutoGaze._tied_weights_keys == []


def test_patch_autogaze_inputs_embeds_generate_compat_adds_and_strips_dummy_prefix():
    class FakeGenerateOutput:
        def __init__(self, sequences):
            self.sequences = sequences

    class FakeDecoder:
        def __init__(self):
            self.calls = []

        def generate(self, *args, **kwargs):
            self.calls.append(kwargs)
            prefix = kwargs["input_ids"]
            new_tokens = torch.tensor([[7, 8, 9]], device=prefix.device, dtype=prefix.dtype)
            return FakeGenerateOutput(torch.cat([prefix, new_tokens], dim=1))

    class FakeGazingModel:
        def __init__(self):
            self.gaze_decoder = FakeDecoder()

    class FakeAutoGaze:
        def __init__(self):
            self.gazing_model = FakeGazingModel()

    model = FakeAutoGaze()
    patched = patch_autogaze_inputs_embeds_generate_compat(model)
    embeds = torch.zeros((1, 5, 4))

    output = model.gazing_model.gaze_decoder.generate(inputs_embeds=embeds, max_new_tokens=3)

    assert patched is True
    assert model.gazing_model.gaze_decoder.calls[-1]["input_ids"].shape == (1, 5)
    assert model.gazing_model.gaze_decoder.calls[-1]["max_new_tokens"] == 8
    assert output.sequences.tolist() == [[7, 8, 9]]
