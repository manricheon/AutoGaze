"""Borissal v0.8 -> Gemma 4 encoder-stage attach tests.

Runs against `transformers.models.gemma4`'s real (installed) modeling code with
a tiny hand-built config -- no HF Hub checkpoint needed, no network dependency.
Skipped if `transformers` has no `gemma4` model (older/pinned versions).

The one thing these tests exist to catch: `Gemma4VisionPooler`'s kernel-index
math derives the pooled grid's width from whichever patches are physically
present in a call, which silently corrupts (or outright crashes on) a
scattered, non-contiguous patch subset -- exactly what a saliency selection
produces. `attach_gemma4.py`'s `_pool_true_grid` is the fix; these tests prove
end to end that a real Gemma4VisionModel + Gemma4MultimodalEmbedder can be
trained through v0.8's selection without it.
"""
import pytest
import torch

pytest.importorskip("transformers")
gemma4_configuration = pytest.importorskip("transformers.models.gemma4.configuration_gemma4")
gemma4_modeling = pytest.importorskip("transformers.models.gemma4.modeling_gemma4")

from autogaze.models.borissal.adapters import to_gemma4_video_tokens        # noqa: E402
from autogaze.models.borissal.attach_gemma4 import (                        # noqa: E402
    build_pruned_video_patches, gated_video_features)
from autogaze.models.borissal.modeling_borissal_v08 import BorissalV08      # noqa: E402

RAW_PATCH, POOL_K = 4, 3           # small stand-ins for Gemma4's real 16 / 3
V08_PATCH = RAW_PATCH * POOL_K     # v0.8 must run AT the pooled-cell grid
T_FRAMES, SIZE = 2, 96             # -> 24x24 raw patches, 8x8 pooled cells


def _tiny_gemma4_vision():
    vcfg = gemma4_configuration.Gemma4VisionConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, head_dim=8,
        patch_size=RAW_PATCH, pooling_kernel_size=POOL_K, position_embedding_size=64)
    tcfg = gemma4_configuration.Gemma4TextConfig(hidden_size=24)
    vision_tower = gemma4_modeling.Gemma4VisionModel(vcfg).eval()
    embed_vision = gemma4_modeling.Gemma4MultimodalEmbedder(vcfg, tcfg).eval()
    return vision_tower, embed_vision


def _patchify(video, patch_size):
    B, T, C, H, W = video.shape
    Hg, Wg = H // patch_size, W // patch_size
    x = video.reshape(B * T, C, Hg, patch_size, Wg, patch_size)
    return x.permute(0, 2, 4, 3, 5, 1).reshape(B * T, Hg * Wg, -1)


def _dense_video_inputs(video):
    """Mirrors `Gemma4VideoProcessor`'s own patchify + position-id construction
    (`convert_video_to_patches` + `meshgrid(..., indexing="xy")`), traced from
    `video_processing_gemma4.py` -- not assumed."""
    pixel_values = _patchify(video, RAW_PATCH)
    T, H, W = video.shape[1], video.shape[-2], video.shape[-1]
    Hg, Wg = H // RAW_PATCH, W // RAW_PATCH
    ys, xs = torch.meshgrid(torch.arange(Hg), torch.arange(Wg), indexing="ij")
    pos = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)
    return pixel_values, pos.unsqueeze(0).repeat(T, 1, 1)


def _selection(ratio=0.4, learnable=True, learn_signal=True):
    torch.manual_seed(0)
    video = torch.rand(1, T_FRAMES, 3, SIZE, SIZE)
    m = BorissalV08(tubelet_size=1, patch_size=V08_PATCH,
                    learnable=learnable, learn_signal=learn_signal)
    sel, aux = m(video, gazing_ratio=ratio)
    return m, video, sel, aux


def test_gemma4_sparse_selection_does_not_collide():
    """Regression for the crash this file exists to avoid: two non-adjacent
    3x3 cells on a small pooled grid used to raise
    'Class values must be smaller than num_classes' via the vanilla pooler's
    subset-derived kernel index. `to_gemma4_video_tokens` + `_pool_true_grid`
    must not."""
    m, video, sel, aux = _selection()
    tokens = to_gemma4_video_tokens(sel, pooling_kernel_size=POOL_K)
    assert tokens["pooled_grid"] == tuple(int(x) for x in sel.grid_thw[0].tolist())
    assert int(tokens["n_kept_per_frame"].sum()) == int(sel.num_keep)


def test_gemma4_pruned_matches_num_keep():
    vision_tower, embed_vision = _tiny_gemma4_vision()
    m, video, sel, aux = _selection()
    tokens = to_gemma4_video_tokens(sel, pooling_kernel_size=POOL_K)
    pixel_values_dense, pos_dense = _dense_video_inputs(video)
    pruned = build_pruned_video_patches(pixel_values_dense, pos_dense, tokens)

    out = gated_video_features(vision_tower, embed_vision, pruned, tokens["pooled_grid"])
    assert out.shape == (int(sel.num_keep), embed_vision.text_hidden_size)


def test_gemma4_gate_is_near_identity():
    """Gated and ungated forwards must match to float32 precision -- st_gate's
    forward value is 1.0 up to rounding, not bit-exact (an identical tensor
    minus its own .detach(), see modeling_borissal_v08.st_gate's docstring)."""
    vision_tower, embed_vision = _tiny_gemma4_vision()
    m, video, sel, aux = _selection()
    tokens = to_gemma4_video_tokens(sel, pooling_kernel_size=POOL_K)
    pixel_values_dense, pos_dense = _dense_video_inputs(video)
    pruned = build_pruned_video_patches(pixel_values_dense, pos_dense, tokens)

    gated = gated_video_features(vision_tower, embed_vision, pruned, tokens["pooled_grid"],
                                 sel=sel, aux=aux)
    ungated = gated_video_features(vision_tower, embed_vision, pruned, tokens["pooled_grid"])
    assert torch.allclose(gated, ungated, atol=1e-5)


def test_gemma4_gradient_reaches_all_nine_knobs():
    """The end-to-end joint-training smoke test: backward through a REAL
    Gemma4VisionModel + Gemma4MultimodalEmbedder must reach every BorissalV08
    knob, same bar as test_v08_st_gate_knob_coverage but through an actual
    downstream encoder instead of the toy one."""
    vision_tower, embed_vision = _tiny_gemma4_vision()
    m, video, sel, aux = _selection()
    tokens = to_gemma4_video_tokens(sel, pooling_kernel_size=POOL_K)
    pixel_values_dense, pos_dense = _dense_video_inputs(video)
    pruned = build_pruned_video_patches(pixel_values_dense, pos_dense, tokens)

    out = gated_video_features(vision_tower, embed_vision, pruned, tokens["pooled_grid"],
                               sel=sel, aux=aux)
    (out * torch.randn_like(out)).sum().backward()
    for name, p in m.params.named_parameters():
        assert p.grad is not None and float(p.grad.abs()) > 0, f"{name}: no grad"


def test_gemma4_mask_mismatch_raises():
    """A Selection that did NOT produce `pruned` must be rejected, not silently
    misalign gate rows with feature rows."""
    vision_tower, embed_vision = _tiny_gemma4_vision()
    m, video, sel, aux = _selection(ratio=0.4)
    tokens = to_gemma4_video_tokens(sel, pooling_kernel_size=POOL_K)
    pixel_values_dense, pos_dense = _dense_video_inputs(video)
    pruned = build_pruned_video_patches(pixel_values_dense, pos_dense, tokens)

    _, _, other_sel, other_aux = _selection(ratio=0.6)   # different ratio -> different mask
    with pytest.raises(RuntimeError):
        gated_video_features(vision_tower, embed_vision, pruned, tokens["pooled_grid"],
                             sel=other_sel, aux=other_aux)
