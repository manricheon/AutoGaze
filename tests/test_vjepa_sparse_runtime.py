import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers.models.vjepa2.configuration_vjepa2 import VJEPA2Config
from transformers.models.vjepa2.modeling_vjepa2 import VJEPA2Encoder

from repro.plugins.vjepa_sparse_runtime import (
    gather_vjepa_hidden_states,
    run_vjepa_encoder_on_selected_embeddings,
)


def test_gather_vjepa_hidden_states_preserves_original_position_ids():
    hidden_states = torch.arange(1 * 4 * 3, dtype=torch.float32).reshape(1, 4, 3)

    gathered, position_mask = gather_vjepa_hidden_states(hidden_states, [0, 2])

    assert gathered.shape == (1, 2, 3)
    assert gathered.tolist() == [[[0.0, 1.0, 2.0], [6.0, 7.0, 8.0]]]
    assert position_mask.tolist() == [[0, 2]]


def test_run_vjepa_encoder_on_selected_embeddings_runs_tiny_encoder():
    config = VJEPA2Config(
        crop_size=32,
        patch_size=16,
        frames_per_clip=2,
        tubelet_size=2,
        hidden_size=72,
        num_attention_heads=3,
        num_hidden_layers=1,
        mlp_ratio=2.0,
    )
    config._attn_implementation = "eager"
    encoder = VJEPA2Encoder(config)
    patch_embeddings = torch.randn(1, 4, 72)

    result = run_vjepa_encoder_on_selected_embeddings(
        encoder,
        patch_embeddings,
        selected_token_indices=[0, 3],
    )

    assert result["last_hidden_state"].shape == (1, 2, 72)
    assert result["position_mask"].tolist() == [[0, 3]]
    assert result["metrics"]["raw_token_count"] == 4
    assert result["metrics"]["selected_token_count"] == 2
    assert result["metrics"]["encoder_token_reduction_ratio"] == 2.0
    timings = result["metrics"]["stage_timings_ms"]
    assert set(timings) >= {
        "gather_selected_embeddings",
        "encoder_layers_total",
        "layernorm",
        "encoder_total",
    }
    assert result["metrics"]["sparse_execution_policy"] == {
        "patch_embedding_scope": "dense_all_vjepa_tokens",
        "encoder_scope": "selected_vjepa_tokens_only",
        "position_policy": "original_vjepa_position_mask",
    }
