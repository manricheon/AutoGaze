#!/usr/bin/env python
"""Semantic-coverage gate — the description-task-aligned judgment axis.

Scores a selection with a LANGUAGE-ALIGNED semantic encoder (SigLIP2,
`google/siglip2-base-patch16-384`: 384/patch16 -> 24x24 patch tokens,
EXACTLY 1:1 with Borissal's main-target grid; AutoGaze precedent — the
original repo used siglip2 as a VideoMAE loss head). Two metrics per
selection, both needed (a pooled-mean metric alone can be gamed by
selecting average-looking background):

- **gist** (higher=better): per-frame cosine(mean-pool(selected patch
  tokens), mean-pool(all patch tokens)) — "does the selection alone carry
  the frame's semantic summary?"
- **recall** (higher=better): per-frame recall of the encoder's most
  important patches (importance = cosine(token, frame pooled embedding),
  top 10%) — "did we miss the patches a caption would be written about?"

Unlike the reconstruction-family gates (V-JEPA coverage, VideoMAE recon),
content-concentrated selections are EXPECTED to beat random here; if
random wins on this axis too, that is a design-review signal (recorded
expectation, design.md). Caveat: single-encoder bias — the final referee
remains the downstream captioner.

Usage:
    uv run python scripts/eval_borissal_semantic.py \
        --selectors random v0.2 v1:weights/<run>/checkpoint_final.pt \
        --ratios 0.25 0.5 --spreads 0 0.25 0.5
"""

import argparse
import json
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
ENCODER_ID = "google/siglip2-base-patch16-384"


def build_encoder(device: torch.device):
    # same class choice as the original AutoGaze task (fixed-res siglip2
    # checkpoints load through the siglip classes; naflex needs Siglip2*)
    from transformers import AutoImageProcessor
    from transformers.models.siglip.modeling_siglip import SiglipVisionModel
    model = SiglipVisionModel.from_pretrained(ENCODER_ID, attn_implementation="sdpa")
    processor = AutoImageProcessor.from_pretrained(ENCODER_ID)
    model.eval().requires_grad_(False)
    return model.to(device), processor


def expand_selection_2x(sel):
    """patch-32 Selection (grid T x 12 x 12) -> patch-16 Selection (T x 24 x 24).

    The user-proposed "compute AND select at the coarse grid, then just expand"
    variant: each selected 32px patch becomes its 2x2 block of 16px children,
    so the FINAL mask stays on the patch-16 grid (downstream contract unchanged)
    and the budget is exactly 4x the coarse count -- same-ratio comparisons
    against native patch-16 selectors are apples to apples. Eval-only helper.
    """
    from autogaze.models.borissal.modeling_borissal import Selection, _pack_gazing_mask
    B = sel.keep_mask.shape[0]
    T, Hc, Wc = (int(x) for x in sel.grid_thw[0])
    H, W = Hc * 2, Wc * 2
    N_pf = H * W

    def up(x):
        return (x.view(B, T, Hc, 1, Wc, 1).expand(B, T, Hc, 2, Wc, 2)
                .reshape(B, T * N_pf))

    keep_mask = up(sel.keep_mask)
    scores = up(sel.scores)
    keep_index, is_padded = _pack_gazing_mask(keep_mask)
    idx = keep_index.clamp(min=0)
    t_c = idx // N_pf
    rem = idx % N_pf
    coords = torch.stack([t_c, rem // W, rem % W], dim=-1)
    coords = coords.masked_fill(is_padded.unsqueeze(-1), -1)
    km3 = keep_mask.view(B, T, N_pf)
    grid = torch.tensor([T, H, W], dtype=sel.grid_thw.dtype,
                        device=sel.grid_thw.device).unsqueeze(0).expand(B, 3).clone()
    return Selection(grid_thw=grid, scores=scores, keep_mask=keep_mask,
                     keep_index=keep_index, keep_coords=coords,
                     num_keep=km3.sum(dim=(1, 2)), per_frame_keep=km3.sum(dim=-1))


def build_selection_coarse(base: str, video: torch.Tensor, ratio: float, spread: float):
    """Run a preset at patch_size=32 (native 12x12 grid). score_coarsen presets
    would need a 6x6-divisible grid times 2 -- 12 works for c=2."""
    from autogaze.models.borissal import Borissal, BorissalConfig
    scale = video.shape[-1]
    preset = base.replace(".", "_")
    if not (base.startswith("v0.") and hasattr(BorissalConfig, preset)):
        raise ValueError(f"coarse: unsupported base {base!r}")
    cfg = getattr(BorissalConfig, preset)(scale=scale, patch_size=32)
    if cfg.selection_mode == "anchor_novelty":
        if spread > 0:
            raise ValueError("coarse anchor preset owns allocation; --spread does not apply")
        return Borissal(cfg).select(video, gazing_ratio=ratio)
    cfg = getattr(BorissalConfig, preset)(scale=scale, patch_size=32,
                                          per_frame_allocation="uniform", block_size=1)
    return Borissal(cfg).select(video, gazing_ratio=ratio, spread_fraction=spread)


def build_selection(spec: str, video: torch.Tensor, ratio: float, spread: float):
    from autogaze.models.borissal import Borissal, BorissalConfig, BorissalV1, BorissalV1Config
    from autogaze.models.borissal.modeling_borissal_v1 import _selection_from_scores
    scale = video.shape[-1]
    if spec == "random":
        cfg = BorissalConfig(scale=scale)
        g = torch.Generator().manual_seed(0)
        S = torch.rand(video.shape[0], video.shape[1] // cfg.tubelet_size,
                       scale // cfg.patch_size, scale // cfg.patch_size, generator=g)
        return _selection_from_scores(S, ratio, "uniform", cfg.eps)  # spread is moot on noise
    if spec.startswith("v1:"):
        ckpt = torch.load(spec[len("v1:"):], map_location="cpu", weights_only=False)
        ckpt_cfg = dict(ckpt["config"])
        ckpt_cfg.setdefault("cosine_scores", False)
        ckpt_cfg.setdefault("global_context", False)
        model = BorissalV1(BorissalV1Config(**ckpt_cfg))
        model.load_state_dict(ckpt["state_dict"])
        return model.eval().select(video, gazing_ratio=ratio,
                                   per_frame_allocation="uniform", spread_fraction=spread)
    if spec == "v0.1":
        return Borissal(BorissalConfig(scale=scale)).select(video, gazing_ratio=ratio,
                                                            spread_fraction=spread)
    if spec == "v0.2":
        cfg = BorissalConfig.v0_2(scale=scale, per_frame_allocation="uniform", block_size=1)
        return Borissal(cfg).select(video, gazing_ratio=ratio, spread_fraction=spread)
    if spec.startswith("coarse:"):
        # signals AND selection at the 12x12 grid (patch 32), expanded 2x after
        inner = build_selection_coarse(spec[len("coarse:"):], video, ratio, spread)
        return expand_selection_2x(inner)
    base, _, ov = spec.partition(",")
    overrides = {}
    for kv in filter(None, ov.split(",")):
        k, val = kv.split("=", 1)
        try:
            overrides[k.strip()] = float(val) if "." in val else int(val)
        except ValueError:
            overrides[k.strip()] = val.strip()
    preset = base.replace(".", "_")
    if base.startswith("v0.") and hasattr(BorissalConfig, preset):
        # generic preset dispatch (v0.3 .. v0.7). anchor_novelty presets own
        # their allocation and reject spread -- pass overrides only where legal.
        cfg = getattr(BorissalConfig, preset)(scale=scale, **overrides)
        if cfg.selection_mode == "anchor_novelty":
            if spread > 0:
                raise ValueError(f"{spec} owns its allocation; --spread does not apply")
            return Borissal(cfg).select(video, gazing_ratio=ratio)
        # NOTE eval convention (pre-existing, from the v0.2 gate): uniform
        # allocation + block_size=1 -- so "v0.3" here is the v0.3 SIGNAL stack
        # under the eval-uniform convention, not the preset verbatim.
        cfg = getattr(BorissalConfig, preset)(scale=scale, per_frame_allocation="uniform",
                                              block_size=1, **overrides)
        return Borissal(cfg).select(video, gazing_ratio=ratio, spread_fraction=spread)
    raise ValueError(f"unknown selector spec: {spec}")


def probe_pool(encoder, tokens: torch.Tensor, need_weights: bool = False):
    """SigLIP's attention-pooling (MAP) head over an arbitrary token subset:
    the learned probe attends to the given tokens -- the encoder's OWN notion
    of 'what this image is about', valid for subsets too. Returns
    (pooled (T, D), attn (T, N) or None)."""
    head = encoder.vision_model.head
    probe = head.probe.expand(tokens.shape[0], -1, -1)
    attn_out, attn_w = head.attention(probe, tokens, tokens, need_weights=need_weights)
    res = attn_out
    attn_out = head.layernorm(attn_out)
    pooled = (res + head.mlp(attn_out))[:, 0]
    return pooled, (attn_w[:, 0] if need_weights else None)


def semantic_metrics(encoder, tokens: torch.Tensor, frame_mask: torch.Tensor,
                     top_frac: float = 0.1):
    """tokens (T, N, D) siglip patch tokens; frame_mask (T, N) bool with the
    SAME count per frame (uniform allocation). Returns (gist, recall).

    Metric-design note (first attempt failed, recorded in design.md): naive
    mean-pool gist is won by random sampling (sample mean -> population
    mean), and cosine-to-mean 'importance' marks TYPICAL patches, not
    description-relevant ones (random recall landed exactly at chance =
    ratio, saliency below chance). Both metrics therefore use the MAP
    head's own attention: importance = where the language-aligned encoder
    actually LOOKS when summarizing the frame."""
    T, N, D = tokens.shape
    counts = frame_mask.sum(dim=-1)
    pooled_all, attn = probe_pool(encoder, tokens, need_weights=True)      # (T, D), (T, N)
    if bool(counts.eq(counts[0]).all()):
        # uniform counts: one batched gather (the original fast path)
        k = int(counts[0].item())
        sel_idx = frame_mask.nonzero(as_tuple=False)[:, 1].reshape(T, k)
        sel_tokens = tokens.gather(1, sel_idx.unsqueeze(-1).expand(-1, -1, D))
        pooled_sel, _ = probe_pool(encoder, sel_tokens)
    else:
        # variable per-frame counts (v0.7 anchor-novelty): the MAP head takes
        # arbitrary-length token subsets, so pool each frame's own subset.
        # Eval-only code -- the python loop over T frames is acceptable here.
        pooled_rows = []
        for t in range(T):
            sub = tokens[t][frame_mask[t]]                                 # (k_t, D)
            if sub.shape[0] == 0:
                sub = tokens[t][:1] * 0.0
            pooled_rows.append(probe_pool(encoder, sub.unsqueeze(0))[0][0])
        pooled_sel = torch.stack(pooled_rows, dim=0)

    pa = torch.nn.functional.normalize(pooled_all.float(), dim=-1)
    ps = torch.nn.functional.normalize(pooled_sel.float(), dim=-1)
    gist = (pa * ps).sum(-1).mean().item()

    n_top = max(1, int(round(top_frac * N)))
    top_idx = attn.topk(n_top, dim=-1).indices                             # (T, n_top)
    recall = frame_mask.gather(1, top_idx).float().mean().item()
    return gist, recall


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", default=str(REPO_ROOT / "assets" / "example_input.mp4"))
    p.add_argument("--selectors", nargs="+", default=["random", "v0.2"],
                   help="random | v0.1 | v0.2 | v1:<checkpoint.pt>")
    p.add_argument("--ratios", nargs="+", type=float, default=[0.25])
    p.add_argument("--spreads", nargs="+", type=float, default=[0.0],
                   help="spread_fraction sweep (applied to non-random selectors)")
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=str(REPO_ROOT / "outputs" / "borissal" / "semantic_gate.json"))
    args = p.parse_args()

    device = torch.device(args.device)
    encoder, processor = build_encoder(device)
    scale = encoder.config.image_size          # 384
    grid = scale // encoder.config.patch_size  # 24

    from autogaze.models.borissal.video_io import load_video, IMAGENET_MEAN, IMAGENET_STD
    video = load_video(args.video, num_frames=args.num_frames, size=scale)
    ours_mean = torch.tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1)
    ours_std = torch.tensor(IMAGENET_STD).view(1, 1, 3, 1, 1)
    enc_mean = torch.tensor(processor.image_mean).view(1, 1, 3, 1, 1)
    enc_std = torch.tensor(processor.image_std).view(1, 1, 3, 1, 1)
    frames = (((video * ours_std + ours_mean) - enc_mean) / enc_std)[0].to(device)  # (T, 3, H, W)

    with torch.no_grad():
        tokens = encoder(pixel_values=frames).last_hidden_state  # (T, N=grid^2, D)
    assert tokens.shape[1] == grid * grid, f"unexpected token count {tokens.shape}"

    results = []
    header = f"{'selector':44} {'ratio':6} {'spread':7} {'gist(>)':>9} {'recall(>)':>10}"
    print(header)
    print("-" * len(header))
    for spec in args.selectors:
        spreads = [0.0] if spec == "random" else args.spreads
        for ratio in args.ratios:
            for s in spreads:
                sel = build_selection(spec, video, ratio, s)
                T_grid = sel.grid_thw[0, 0].item()
                mask_tub = sel.keep_mask[0].reshape(T_grid, grid * grid)
                frame_mask = mask_tub.repeat_interleave(args.num_frames // T_grid, dim=0).to(device)
                gist, recall = semantic_metrics(encoder, tokens, frame_mask)
                results.append({"selector": spec, "ratio": ratio, "spread": s,
                                "gist": gist, "recall": recall})
                print(f"{spec:44} {ratio:<6} {s:<7} {gist:9.4f} {recall:10.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
