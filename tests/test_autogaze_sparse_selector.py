import json

import torch

from repro.plugins.autogaze_sparse_selector import (
    build_sparse_selection_plan_from_autogaze_outputs,
)
from repro.plugins.gaze_plan import qwen_visual_indices_from_sparse_plan


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
