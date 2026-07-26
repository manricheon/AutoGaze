"""Borissal -> Qwen3-VL true-token-drop attach tests.

Runs on the tiny random checkpoint `optimum-intel-internal-testing/tiny-random-qwen3-vl`
(~32 MB, same geometry as the real models: patch16 / temporal_patch2 /
spatial_merge2, and deepstack_visual_indexes=[1,3,5] so the deepstack path is
exercised). Skipped when it is not in the local HF cache.
"""
import numpy as np
import pytest
import torch

from autogaze.models.borissal import Borissal, BorissalConfig
from autogaze.models.borissal.adapters import to_qwen3vl_video_tokens

MODEL_ID = "optimum-intel-internal-testing/tiny-random-qwen3-vl"
T_FRAMES, SIZE = 8, 64          # -> grid (4, 4, 4), merged (4, 2, 2) = 16 LLM tokens


@pytest.fixture(scope="module")
def qwen():
    transformers = pytest.importorskip("transformers")
    try:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        proc = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID, dtype=torch.float32, local_files_only=True).eval()
    except Exception as e:                                   # not cached / offline
        pytest.skip(f"{MODEL_ID} unavailable locally: {type(e).__name__}: {e}")
    return proc, model


@pytest.fixture(scope="module")
def clip():
    torch.manual_seed(0)
    video = torch.rand(1, T_FRAMES, 3, SIZE, SIZE)
    return video


def _processor_inputs(proc, video):
    frames = (video[0].permute(0, 2, 3, 1).numpy()).astype("float32")   # (T, H, W, C)
    msgs = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": "Describe."}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return proc(text=[text], videos=[frames], return_tensors="pt",
                do_sample_frames=False, do_resize=False)


def _selection(video, preset="v0_5", ratio=0.25, **over):
    cfg = getattr(BorissalConfig, preset)(scale=SIZE, per_frame_allocation="uniform", **over)
    return Borissal(cfg).select(video, gazing_ratio=ratio)


def _all_tokens_patch_index(model, inputs):
    m = model.config.vision_config.spatial_merge_size
    n_tok = int(inputs["video_grid_thw"].prod(-1).sum() // (m * m))
    return (torch.arange(n_tok)[:, None] * m * m + torch.arange(m * m)).reshape(1, -1)


# --- the primary correctness gate ---------------------------------------------

@pytest.mark.parametrize("stage", ["llm", "encoder"])
def test_keep_all_reproduces_vanilla_forward(qwen, clip, stage):
    """Dropping nothing must be indistinguishable from the model's own forward.

    This is the gate that catches every plumbing error at once: placeholder
    accounting, mrope column selection, deepstack wiring, and (for the encoder
    stage) the reimplemented vision forward.
    """
    from autogaze.models.borissal.attach_qwen3vl import build_pruned_inputs, pruned_forward
    proc, model = qwen
    inputs = _processor_inputs(proc, clip)
    kw = {"qwen_patch_index": _all_tokens_patch_index(model, inputs)} if stage == "encoder" else {}
    with torch.no_grad():
        ref = model(**inputs).logits
        pruned = build_pruned_inputs(model, inputs, None, prune_stage=stage, **kw)
        logits, _ = pruned_forward(model, pruned)
    assert pruned.n_vision_tokens == pruned.n_vision_tokens_dense
    assert logits.shape == ref.shape
    assert torch.allclose(logits, ref, atol=1e-5, rtol=1e-5), \
        f"max|diff| = {float((logits - ref).abs().max()):.3e}"


def test_deepstack_is_actually_wired(qwen, clip):
    """Guard the silent-failure mode: passing inputs_embeds to the public forward
    would drop deepstack injection entirely. If deepstack were being ignored,
    zeroing it would change nothing -- assert that it does change the logits."""
    from autogaze.models.borissal.attach_qwen3vl import build_pruned_inputs, pruned_forward
    proc, model = qwen
    assert model.config.vision_config.deepstack_visual_indexes, "fixture must exercise deepstack"
    inputs = _processor_inputs(proc, clip)
    with torch.no_grad():
        pruned = build_pruned_inputs(model, inputs, None)
        with_deep, _ = pruned_forward(model, pruned)
        pruned.deepstack_visual_embeds = [torch.zeros_like(d) for d in pruned.deepstack_visual_embeds]
        without_deep, _ = pruned_forward(model, pruned)
    assert not torch.allclose(with_deep, without_deep), \
        "zeroing deepstack changed nothing -- deepstack features are not reaching the LLM"


# --- pruning accounting -------------------------------------------------------

@pytest.mark.parametrize("stage", ["llm", "encoder"])
def test_pruned_sequence_drops_exactly_the_unselected_tokens(qwen, clip, stage):
    from autogaze.models.borissal.attach_qwen3vl import build_pruned_inputs, pruned_forward
    proc, model = qwen
    inputs = _processor_inputs(proc, clip)
    sel = _selection(clip, "v0_5", ratio=0.5)
    tok = to_qwen3vl_video_tokens(sel, model.config.vision_config.spatial_merge_size, "strict")
    kw = {"qwen_patch_index": tok["qwen_patch_index"]} if stage == "encoder" else {}
    with torch.no_grad():
        pruned = build_pruned_inputs(model, inputs, tok["keep_token_index"], prune_stage=stage, **kw)
        logits, _ = pruned_forward(model, pruned)

    k = int(tok["num_keep_tokens"][0])
    assert pruned.n_vision_tokens == k < pruned.n_vision_tokens_dense
    # placeholder count in the pruned prompt must equal the kept-token count,
    # otherwise transformers' own get_placeholder_mask would have raised
    assert int((pruned.input_ids[0] == model.config.video_token_id).sum()) == k
    assert int(pruned.visual_pos_masks.sum()) == k
    for d in pruned.deepstack_visual_embeds:
        assert d.shape[0] == k
    dropped = pruned.n_vision_tokens_dense - k
    assert pruned.input_ids.shape[1] == inputs["input_ids"].shape[1] - dropped
    assert logits.shape[1] == pruned.input_ids.shape[1]
    # non-vision scaffolding (timestamps, vision_start/end, text) is untouched
    keep = pruned.seq_keep[0]
    is_vid = inputs["input_ids"][0] == model.config.video_token_id
    assert bool(keep[~is_vid].all()), "pruning must not remove any non-vision token"


def test_position_ids_are_the_dense_ones_minus_dropped_columns(qwen, clip):
    """mrope coordinates must be the dense values gathered at kept positions, and
    every position AFTER the vision block must be unchanged by pruning -- that is
    what makes a pruned run positionally comparable to the dense run."""
    from autogaze.models.borissal.attach_qwen3vl import build_pruned_inputs
    proc, model = qwen
    inputs = _processor_inputs(proc, clip)
    sel = _selection(clip, "v0_5", ratio=0.5)
    tok = to_qwen3vl_video_tokens(sel, model.config.vision_config.spatial_merge_size, "strict")
    with torch.no_grad():
        dense = build_pruned_inputs(model, inputs, None)
        pruned = build_pruned_inputs(model, inputs, tok["keep_token_index"])
    if dense.position_ids is None:
        pytest.skip("model returned no 3d position ids")
    keep = pruned.seq_keep[0]
    assert torch.equal(pruned.position_ids, dense.position_ids[..., keep])
    # text tail: same ids in both runs -> same positions
    ids = inputs["input_ids"][0]
    last_vid = int(torch.nonzero(ids == model.config.video_token_id)[-1])
    tail_dense = dense.position_ids[..., last_vid + 1:]
    tail_pruned = pruned.position_ids[..., keep.cumsum(0)[last_vid]:]
    assert torch.equal(tail_dense, tail_pruned), "post-vision text positions shifted under pruning"


# --- adapter index math -------------------------------------------------------

def test_adapter_indices_match_processor_patch_order(clip):
    """Cross-check the merged-token remap against an INDEPENDENT replay of the
    video processor's permute -- no shared arithmetic with the adapter."""
    m, tp, p = 2, 2, 16
    Tg, Hg, Wg = T_FRAMES // tp, SIZE // p, SIZE // p
    Hm, Wm = Hg // m, Wg // m
    borissal_flat = torch.arange(Tg * Hg * Wg).reshape(1, Tg, Hg, Wg)
    # processor: permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9) => groups of m*m per 2x2 block,
    # blocks raster over (t, hm, wm)
    processor_order = borissal_flat.reshape(1, Tg, Hm, m, Wm, m).permute(0, 1, 2, 4, 3, 5).reshape(-1)

    sel = _selection(clip, "v0_5", ratio=0.5)
    out = to_qwen3vl_video_tokens(sel, m, "strict")
    qpi = out["qwen_patch_index"][0]
    qpi = qpi[qpi >= 0]
    tokens = out["keep_token_index"][0]
    tokens = tokens[tokens >= 0]

    expected_patches = sel.keep_index[0][sel.keep_index[0] >= 0]
    assert torch.equal(processor_order[qpi].sort().values, expected_patches.sort().values)
    assert torch.equal(qpi, (tokens[:, None] * m * m + torch.arange(m * m)).reshape(-1))
    coords = out["token_coords"][0][: tokens.numel()]
    assert torch.equal(coords[:, 0] * Hm * Wm + coords[:, 1] * Wm + coords[:, 2], tokens)


def test_cube_coherence_yields_no_partial_blocks(clip):
    """v0.5/v0.6-default map exactly into merged-token space; v0.3 need not."""
    for preset in ("v0_5", "v0_6"):
        alloc = "uniform" if preset == "v0_5" else "global"
        cfg = getattr(BorissalConfig, preset)(scale=SIZE, per_frame_allocation=alloc)
        sel = Borissal(cfg).select(clip, gazing_ratio=0.5)
        assert to_qwen3vl_video_tokens(sel, 2, "any")["n_partial_blocks"] == 0, preset


def test_partial_block_policies(clip):
    sel = _selection(clip, "v0_3", ratio=0.3)           # score_coarsen=1 -> no block structure
    n_partial = to_qwen3vl_video_tokens(sel, 2, "any")["n_partial_blocks"]
    if n_partial == 0:
        pytest.skip("this clip/ratio happened to select whole blocks under v0.3")
    with pytest.raises(ValueError, match="partially selected"):
        to_qwen3vl_video_tokens(sel, 2, "strict")
    any_k = int(to_qwen3vl_video_tokens(sel, 2, "any")["num_keep_tokens"][0])
    full_k = int(to_qwen3vl_video_tokens(sel, 2, "full")["num_keep_tokens"][0])
    assert full_k < any_k, "'full' must under-keep and 'any' over-keep relative to each other"


def test_adapter_rejects_bad_arguments(clip):
    sel = _selection(clip, "v0_5")
    with pytest.raises(ValueError, match="partial_blocks"):
        to_qwen3vl_video_tokens(sel, 2, "nonsense")
    with pytest.raises(ValueError, match="divisible"):
        to_qwen3vl_video_tokens(sel, 3)                  # 4x4 grid not divisible by 3


# --- guardrails ---------------------------------------------------------------

def test_encoder_stage_derives_patch_index_and_rejects_partial_blocks(qwen, clip):
    """`encoder` is the default stage, so the whole-block patch index is derived
    from the kept tokens -- the derived path must match the adapter's explicit one
    exactly, and a short/partial patch index must still be rejected."""
    from autogaze.models.borissal.attach_qwen3vl import build_pruned_inputs, pruned_forward
    proc, model = qwen
    inputs = _processor_inputs(proc, clip)
    sel = _selection(clip, "v0_5", ratio=0.5)
    tok = to_qwen3vl_video_tokens(sel, 2, "strict")
    with torch.no_grad():
        derived = build_pruned_inputs(model, inputs, tok["keep_token_index"])       # stage defaults to encoder
        explicit = build_pruned_inputs(model, inputs, tok["keep_token_index"],
                                       prune_stage="encoder",
                                       qwen_patch_index=tok["qwen_patch_index"])
        assert derived.prune_stage == "encoder", "encoder must be the default stage"
        assert torch.equal(pruned_forward(model, derived)[0], pruned_forward(model, explicit)[0])
    with pytest.raises(ValueError, match="whole"):
        build_pruned_inputs(model, inputs, tok["keep_token_index"], prune_stage="encoder",
                            qwen_patch_index=tok["qwen_patch_index"][:, :-4])


def test_rejects_batched_and_unsorted_and_out_of_range(qwen, clip):
    from autogaze.models.borissal.attach_qwen3vl import build_pruned_inputs
    proc, model = qwen
    inputs = _processor_inputs(proc, clip)
    with pytest.raises(ValueError, match="ascending"):
        build_pruned_inputs(model, inputs, torch.tensor([[3, 1, 2]]))
    with pytest.raises(ValueError, match="out of range"):
        build_pruned_inputs(model, inputs, torch.tensor([[9999]]))
    with pytest.raises(ValueError, match="prune_stage"):
        build_pruned_inputs(model, inputs, None, prune_stage="middle")
    batched = {k: (torch.cat([v, v]) if k in ("input_ids", "attention_mask", "mm_token_type_ids") else v)
               for k, v in inputs.items()}
    with pytest.raises(NotImplementedError, match="batch size"):
        build_pruned_inputs(model, batched, None)


def test_pruning_actually_changes_the_prediction(qwen, clip):
    """Sanity: a pruned run must not coincidentally equal the dense run."""
    from autogaze.models.borissal.attach_qwen3vl import build_pruned_inputs, pruned_forward
    proc, model = qwen
    inputs = _processor_inputs(proc, clip)
    sel = _selection(clip, "v0_5", ratio=0.25)
    tok = to_qwen3vl_video_tokens(sel, 2, "strict")
    with torch.no_grad():
        dense_logits, _ = pruned_forward(model, build_pruned_inputs(model, inputs, None))
        pruned = build_pruned_inputs(model, inputs, tok["keep_token_index"])
        pruned_logits, _ = pruned_forward(model, pruned)
    assert pruned_logits.shape[1] < dense_logits.shape[1]
    # compare the shared text tail (last token's prediction)
    assert not torch.allclose(pruned_logits[:, -1], dense_logits[:, -1])


# --- patch-14 native-resolution families (Mistral3 / GLM-4V / InternVL) --------

@pytest.mark.parametrize("scale,expect_grid", [(336, 24), (392, 28), (448, 32)])
def test_patch14_native_resolution_recipe(scale, expect_grid):
    """Most non-Qwen towers use patch 14, not 16. Native dynamic-resolution
    families round the image to a multiple of `patch*merge = 28`, so running the
    selector at patch_size=14 with a 28-multiple scale yields an EVEN patch grid,
    which keeps cube coherence (and therefore whole merged tokens) valid. The
    merged-token arithmetic is patch-size agnostic, so the same adapter applies.
    """
    video = torch.rand(1, 8, 3, scale, scale)
    cfg = BorissalConfig.v0_5(scale=scale, patch_size=14, per_frame_allocation="uniform")
    sel = Borissal(cfg).select(video, gazing_ratio=0.25)
    assert [int(x) for x in sel.grid_thw[0]] == [4, expect_grid, expect_grid]
    out = to_qwen3vl_video_tokens(sel, 2, "strict")          # raises if any block is split
    assert out["n_partial_blocks"] == 0
    assert out["merged_grid"] == (4, expect_grid // 2, expect_grid // 2)
    assert int(out["num_keep_tokens"][0]) == out["num_tokens_total"] // 4


def test_fixed_resolution_patch14_towers_still_blocked():
    """The recorded OneVision `so400m-patch14-384` limitation, pinned as a test so
    it is not mistaken for solved: 384 is not a multiple of 14, and the 378
    workaround gives an odd 27x27 grid that cube coherence cannot use."""
    with pytest.raises(ValueError, match="divisible by patch_size"):
        Borissal(BorissalConfig.v0_5(scale=384, patch_size=14)).select(
            torch.rand(1, 8, 3, 384, 384), gazing_ratio=0.25)
    with pytest.raises(ValueError, match="score_coarsen=2 requires grid 27x27"):
        Borissal(BorissalConfig.v0_5(scale=378, patch_size=14)).select(
            torch.rand(1, 8, 3, 378, 378), gazing_ratio=0.25)
