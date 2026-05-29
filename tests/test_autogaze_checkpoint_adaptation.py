import torch

from autogaze.trainer import (
    adapt_autogaze_state_dict_for_partial_load,
    adapt_task_state_dict_for_partial_load,
)


def test_adapt_autogaze_state_dict_remaps_official_four_scale_vocab_to_single_224():
    hidden = 3
    source_vocab = 266
    target_vocab = 197
    num_multi_token_pred = 2
    embed_key = "gazing_model.gaze_decoder.model.embed_tokens.weight"
    lm_head_key = "gazing_model.gaze_decoder.lm_head.weight"
    source_state = {
        embed_key: torch.arange(source_vocab * hidden).reshape(source_vocab, hidden),
        lm_head_key: torch.arange(source_vocab * num_multi_token_pred * hidden).reshape(
            source_vocab * num_multi_token_pred, hidden
        ),
        "compatible.weight": torch.ones(2, 2),
        "mismatched.weight": torch.ones(4, 4),
    }
    target_state = {
        embed_key: torch.zeros(target_vocab, hidden),
        lm_head_key: torch.zeros(target_vocab * num_multi_token_pred, hidden),
        "compatible.weight": torch.zeros(2, 2),
        "mismatched.weight": torch.zeros(1, 1),
    }

    adapted_state, report = adapt_autogaze_state_dict_for_partial_load(
        source_state,
        target_state,
        source_scales="32+64+112+224",
        target_scales="224",
        source_num_vision_tokens=265,
        target_num_vision_tokens=196,
    )

    source_224_offset = 4 + 16 + 49
    assert torch.equal(adapted_state[embed_key][0], source_state[embed_key][source_224_offset])
    assert torch.equal(adapted_state[embed_key][195], source_state[embed_key][source_224_offset + 195])
    assert torch.equal(adapted_state[embed_key][196], source_state[embed_key][265])
    assert torch.equal(adapted_state[lm_head_key][0], source_state[lm_head_key][source_224_offset])
    assert torch.equal(
        adapted_state[lm_head_key][target_vocab],
        source_state[lm_head_key][source_vocab + source_224_offset],
    )
    assert "mismatched.weight" not in adapted_state
    assert embed_key in report["adapted_keys"]
    assert lm_head_key in report["adapted_keys"]
    assert "mismatched.weight" in report["skipped_shape_mismatch_keys"]


def test_adapt_task_state_dict_keeps_224_scale_embedding_row_for_single_scale_task():
    vit_key = "mae.vit.embeddings.scale_embed"
    decoder_key = "mae.decoder.decoder_scale_embed"
    source_state = {
        vit_key: torch.arange(4 * 3).reshape(4, 3),
        decoder_key: torch.arange(4 * 5).reshape(4, 5),
    }
    target_state = {
        vit_key: torch.zeros(1, 3),
        decoder_key: torch.zeros(1, 5),
    }

    adapted_state, report = adapt_task_state_dict_for_partial_load(
        source_state,
        target_state,
        target_scales="224",
    )

    assert torch.equal(adapted_state[vit_key], source_state[vit_key][3:4])
    assert torch.equal(adapted_state[decoder_key], source_state[decoder_key][3:4])
    assert vit_key in report["adapted_keys"]
    assert decoder_key in report["adapted_keys"]
