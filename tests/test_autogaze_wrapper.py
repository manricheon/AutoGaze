from __future__ import annotations

import pytest
import torch

from autogaze_ext.models import AutoGazeOutput, AutoGazeWrapper


def test_off_mode_output_shape() -> None:
    visual_tokens = torch.zeros(2, 3, 4, 8)
    wrapper = AutoGazeWrapper(enabled=False)

    output = wrapper(visual_tokens=visual_tokens)

    assert isinstance(output, AutoGazeOutput)
    assert output.selected_patch_indices.shape == (2, 3, 4)
    assert output.selected_patch_indices[0, 0].tolist() == [0, 1, 2, 3]
    assert output.attention_maps is None


def test_on_mode_stub_behavior_without_original_model() -> None:
    wrapper = AutoGazeWrapper(enabled=True)
    video = torch.zeros(1, 2, 3, 8, 8)

    with pytest.raises(NotImplementedError, match="requires an explicit original_model"):
        wrapper({"video": video})


def test_metadata_preservation() -> None:
    visual_tokens = torch.zeros(1, 2, 3, 4)
    metadata = {
        "frame_indices": [2, 5],
        "patch_indices": [10, 11, 12],
        "scales": [224, 224, 448],
        "custom": "kept",
    }
    wrapper = AutoGazeWrapper(enabled=False)

    output = wrapper(visual_tokens=visual_tokens, metadata=metadata)

    assert output.metadata["frame_indices"] == [2, 5]
    assert output.metadata["patch_indices"] == [10, 11, 12]
    assert output.metadata["scales"] == [224, 224, 448]
    assert output.metadata["custom"] == "kept"
    assert output.selected_scales is not None
    assert output.selected_scales.shape == (1, 2, 3)


def test_token_count_reporting() -> None:
    visual_tokens = torch.zeros(2, 3, 4, 8)
    wrapper = AutoGazeWrapper(enabled=False, token_budget=12)

    output = wrapper(visual_tokens=visual_tokens)

    assert output.token_budget == 12
    assert output.metadata["original_visual_token_count"] == 12
    assert output.metadata["selected_visual_token_count"] == 12
    assert output.metadata["original_visual_token_count_total"] == 24
    assert output.metadata["selected_visual_token_count_total"] == 24
