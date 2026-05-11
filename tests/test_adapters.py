from __future__ import annotations

import pytest
import torch

from autogaze_ext.adapters import (
    CompactTokenAdapter,
    PatchGridMapper,
    TemporalAdapter,
    TokenMaskAdapter,
    VisionFeatureAdapter,
)


def test_patch_grid_remapping() -> None:
    mapper = PatchGridMapper(
        source_grid=(2, 2),
        target_grid=(4, 4),
        source_resolution=(224, 224),
        target_resolution=(224, 224),
        source_patch_size=112,
        target_patch_size=56,
    )

    output = mapper(torch.tensor([0, 1, 2, 3]))

    assert output.mapped_patch_indices.tolist() == [5, 7, 13, 15]
    assert output.metadata["patch_size_mismatch"] is True


def test_token_mask_creation() -> None:
    output = TokenMaskAdapter()(torch.tensor([[0, 2], [1, 3]]), num_tokens=4)

    assert output.mask.dtype is torch.bool
    assert output.mask.tolist() == [
        [True, False, True, False],
        [False, True, False, True],
    ]


def test_compact_token_gathering() -> None:
    tokens = torch.arange(1 * 2 * 4 * 1, dtype=torch.float32).reshape(1, 2, 4, 1)
    indices = torch.tensor([[0, 3], [1, 2]])

    output = CompactTokenAdapter()(tokens, indices)

    assert output.tokens.shape == (1, 2, 2, 1)
    assert output.tokens.flatten().tolist() == [0.0, 3.0, 5.0, 6.0]
    assert output.metadata["original_token_shape"] == (1, 2, 4, 1)


def test_temporal_mean_pooling() -> None:
    features = torch.tensor([[[[1.0], [3.0]], [[5.0], [7.0]]]])

    output = TemporalAdapter("mean_pool")(features)

    assert output.features.shape == (1, 2, 1)
    assert output.features.squeeze(-1).tolist() == [[3.0, 5.0]]


def test_temporal_max_pooling() -> None:
    features = torch.tensor([[[[1.0], [9.0]], [[5.0], [7.0]]]])

    output = TemporalAdapter("max_pool")(features)

    assert output.features.squeeze(-1).tolist() == [[5.0, 9.0]]


def test_concat_token_behavior() -> None:
    features = torch.arange(1 * 2 * 3 * 1, dtype=torch.float32).reshape(1, 2, 3, 1)

    output = TemporalAdapter("concat_tokens")(features)

    assert output.features.shape == (1, 6, 1)
    assert output.features.flatten().tolist() == [0, 1, 2, 3, 4, 5]


def test_unsupported_shape_errors() -> None:
    with pytest.raises(ValueError, match="Expected temporal features shape"):
        TemporalAdapter("mean_pool")(torch.zeros(2, 3, 4))

    with pytest.raises(ValueError, match="outside token dimension"):
        CompactTokenAdapter()(torch.zeros(1, 4, 2), torch.tensor([4]))

    with pytest.raises(ValueError, match="multi-scale patch metadata"):
        PatchGridMapper(
            source_grid=(2, 2),
            target_grid=(2, 2),
            source_resolution=(224, 224),
            target_resolution=(224, 224),
            source_patch_size=112,
            target_patch_size=112,
        )(torch.tensor([0]), metadata={"scales": [224]})

    with pytest.raises(ValueError, match="Expected feature dim"):
        VisionFeatureAdapter(input_dim=4, output_dim=2)(torch.zeros(1, 3, 5))
