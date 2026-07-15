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
    if spec in ("v0.1", "v0.2"):
        cfg = BorissalConfig.v0_2(scale=scale, per_frame_allocation="uniform", block_size=1) \
            if spec == "v0.2" else BorissalConfig(scale=scale)
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
    k = int(frame_mask[0].sum().item())
    assert frame_mask.sum(dim=-1).eq(k).all(), "requires uniform per-frame keep count"

    pooled_all, attn = probe_pool(encoder, tokens, need_weights=True)      # (T, D), (T, N)
    sel_idx = frame_mask.nonzero(as_tuple=False)[:, 1].reshape(T, k)
    sel_tokens = tokens.gather(1, sel_idx.unsqueeze(-1).expand(-1, -1, D))
    pooled_sel, _ = probe_pool(encoder, sel_tokens)

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
