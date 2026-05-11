from __future__ import annotations

import pytest
import torch

from autogaze_ext.models.vision import (
    BaseVisionEncoder,
    GenericViTAdapter,
    ModifiedSigLIPAdapter,
    VJEPA2Adapter,
    VanillaSigLIPAdapter,
)


def test_adapter_interface_consistency() -> None:
    adapters = [
        GenericViTAdapter(patch_size=8, hidden_dim=12, resolution=(32, 32)),
        VanillaSigLIPAdapter(patch_size=8, hidden_dim=12, mode="dummy", resolution=(32, 32)),
        VJEPA2Adapter(mode="full", patch_size=8, hidden_dim=12, resolution=(32, 32)),
        ModifiedSigLIPAdapter(patch_size=8, output_dim=12, resolution=(32, 32)),
    ]

    for adapter in adapters:
        assert isinstance(adapter, BaseVisionEncoder)
        for method in ["prepare_inputs", "forward", "count_visual_tokens", "get_patch_grid", "get_output_dim"]:
            assert callable(getattr(adapter, method))
        assert adapter.get_patch_grid() == (4, 4)
        assert adapter.get_output_dim() == 12


def test_generic_vit_dummy_forward_shape() -> None:
    video = torch.zeros(2, 3, 3, 32, 32)
    adapter = GenericViTAdapter(patch_size=8, hidden_dim=10)

    output = adapter(video)

    assert output.visual_tokens.shape == (2, 3, 16, 10)
    assert adapter.count_visual_tokens(output.visual_tokens) == 48
    assert output.metadata["patch_grid"] == (4, 4)


def test_vanilla_siglip_dummy_forward_shape() -> None:
    video = torch.zeros(1, 2, 3, 32, 32)
    adapter = VanillaSigLIPAdapter(patch_size=16, hidden_dim=6, mode="dummy")

    output = adapter(video)

    assert output.visual_tokens.shape == (1, 2, 4, 6)
    assert output.metadata["vision_encoder_type"] == "vanilla_siglip"
    assert output.metadata["vision_encoder_mode"] == "dummy"


def test_vjepa2_preserves_video_shape_metadata_and_modes() -> None:
    video = torch.zeros(1, 2, 3, 32, 32)

    for mode in ["full", "crop", "mask", "compact"]:
        output = VJEPA2Adapter(mode=mode, patch_size=16, hidden_dim=5)(video)
        assert output.visual_tokens.shape == (1, 2, 4, 5)
        assert output.metadata["vision_encoder_type"] == "vjepa2"
        assert output.metadata["vision_encoder_mode"] == mode
        assert output.metadata["preserves_video_shape"] == (1, 2, 3, 32, 32)


def test_patch_grid_reporting_with_arbitrary_resolution() -> None:
    adapter = GenericViTAdapter(patch_size=14, hidden_dim=8, resolution=(56, 70))

    assert adapter.get_patch_grid() == (4, 5)
    assert adapter.count_visual_tokens(torch.zeros(2, 3, 3, 56, 70)) == 60


def test_unsupported_mode_errors() -> None:
    with pytest.raises(NotImplementedError, match="explicit original modified SigLIP"):
        ModifiedSigLIPAdapter()(torch.zeros(1, 1, 3, 32, 32))

    with pytest.raises(NotImplementedError, match="hf/local modes require"):
        VanillaSigLIPAdapter(mode="hf")

    with pytest.raises(ValueError, match="Unsupported VJEPA2Adapter mode"):
        VJEPA2Adapter(mode="unsupported")

    with pytest.raises(ValueError, match="resolution must be divisible"):
        GenericViTAdapter(patch_size=7)(torch.zeros(1, 1, 3, 32, 32))
