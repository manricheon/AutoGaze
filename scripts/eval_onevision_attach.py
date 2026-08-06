#!/usr/bin/env python
"""Verify the Borissal selector attaches to a LLaVA-OneVision-2 style per-frame
SigLIP vision path (selected tokens -> per-frame SigLIP tower -> later Qwen LLM).

Mac/CPU-friendly. Runs the selector at the ENCODER's own patch size so its
fine grid matches SigLIP's per-frame patch grid (24x24 for patch16-384, 27x27
for the true OneVision patch14-384 tower), bridges tubelet->frame via
`to_onevision_frame_indices`, gathers the encoder's pre-merge patch tokens by
the selected indices, and CROSS-CHECKS that against the known-good SigLIP2
semantic-gate gather path (`keep_mask.reshape().repeat_interleave`). Also
renders overlays proving the tubelet->frame duplication.

Examples:
    # default (cached stand-in encoder, no download):
    uv run python scripts/eval_onevision_attach.py

    # true OneVision patch14 tower (opt-in, ~1.6GB download, 27x27):
    uv run python scripts/eval_onevision_attach.py \
        --encoder google/siglip-so400m-patch14-384 \
        --out outputs/borissal/onevision_attach_patch14.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from autogaze.models.borissal import Borissal, BorissalConfig, resolve_device
from autogaze.models.borissal.adapters import to_onevision_frame_indices
from autogaze.models.borissal.video_io import load_video, unnormalize, IMAGENET_MEAN, IMAGENET_STD
from autogaze.models.borissal.viz import render_overlay

REPO_ROOT = Path(__file__).resolve().parent.parent


def build_encoder(encoder_id: str, device: torch.device):
    from transformers import AutoImageProcessor
    from transformers.models.siglip.modeling_siglip import SiglipVisionModel
    model = SiglipVisionModel.from_pretrained(encoder_id, attn_implementation="sdpa")
    processor = AutoImageProcessor.from_pretrained(encoder_id)
    model.eval().requires_grad_(False)
    return model.to(device), processor


def build_selector(spec: str, scale: int, patch: int):
    if spec == "v0.3":
        cfg = BorissalConfig.v0_3(scale=scale, patch_size=patch, per_frame_allocation="uniform")
    elif spec == "v0.5":
        cfg = BorissalConfig.v0_5(scale=scale, patch_size=patch, per_frame_allocation="uniform")
    else:
        raise ValueError(f"unknown selector spec: {spec} (use v0.3 or v0.5)")
    return Borissal(cfg), cfg


def render_perframe_strip(video_disp, frame_mask, out_path, title):
    """video_disp (T,C,H,W) in [0,1]; frame_mask (T, H_grid, W_grid) bool.
    One row, all T frames, kept patches outlined red. Adjacent frames of a
    tubelet show an IDENTICAL mask -- that is the tubelet->frame duplication."""
    T = video_disp.shape[0]
    H, W = video_disp.shape[-2], video_disp.shape[-1]
    H_grid, W_grid = frame_mask.shape[-2], frame_mask.shape[-1]
    ph, pw = H // H_grid, W // W_grid
    fig, axes = plt.subplots(1, T, figsize=(2 * T, 2.4), squeeze=False)
    for f in range(T):
        frame = video_disp[f].permute(1, 2, 0).numpy()
        m = frame_mask[f].float().numpy()
        m_full = m.repeat(ph, axis=0).repeat(pw, axis=1)
        axes[0, f].imshow((frame * (0.8 * m_full[..., None] + 0.2)).clip(0, 1))
        for i in range(H_grid):
            for j in range(W_grid):
                if m[i, j] > 0.5:
                    axes[0, f].add_patch(plt.Rectangle(
                        (j * pw - 0.5, i * ph - 0.5), pw, ph,
                        linewidth=0.6, edgecolor="red", facecolor="none"))
        axes[0, f].set_title(f"frame {f}", fontsize=8)
        axes[0, f].axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", default=str(REPO_ROOT / "assets" / "example_input.mp4"))
    p.add_argument("--encoder", default="google/siglip2-base-patch16-384",
                   help="cached stand-in default; true tower: google/siglip-so400m-patch14-384")
    p.add_argument("--selector", default="v0.3", choices=["v0.3", "v0.5"])
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--gazing-ratio", type=float, default=0.25)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--out", default=str(REPO_ROOT / "outputs" / "borissal" / "onevision_attach_patch16.png"))
    args = p.parse_args()

    device = resolve_device(args.device)
    encoder, processor = build_encoder(args.encoder, device)
    scale = encoder.config.image_size
    patch = encoder.config.patch_size
    grid = scale // patch
    print(f"encoder={args.encoder}  scale={scale}  patch={patch}  grid={grid}x{grid}  device={device}")

    video = load_video(args.video, num_frames=args.num_frames, size=scale)  # (1,T,3,H,W) imagenet-normed
    # renormalize imagenet -> encoder stats (same as the semantic gate)
    ours_mean = torch.tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1)
    ours_std = torch.tensor(IMAGENET_STD).view(1, 1, 3, 1, 1)
    enc_mean = torch.tensor(processor.image_mean).view(1, 1, 3, 1, 1)
    enc_std = torch.tensor(processor.image_std).view(1, 1, 3, 1, 1)
    frames = (((video * ours_std + ours_mean) - enc_mean) / enc_std)[0].to(device)  # (T,3,H,W)

    with torch.no_grad():
        tokens = encoder(pixel_values=frames).last_hidden_state  # (T, grid^2, D) PRE-merge
    assert tokens.shape[1] == grid * grid, f"unexpected token count {tokens.shape}"

    # run selector at the ENCODER's patch size so its fine grid matches SigLIP's
    model, cfg = build_selector(args.selector, scale, patch)
    model = model.to(device)
    sel = model.select(video.to(device), gazing_ratio=args.gazing_ratio)

    # grid-match assertion (fail loudly if patch_size != encoder's)
    gthw = sel.grid_thw[0].tolist()
    T_grid = gthw[0]
    assert gthw[1:] == [grid, grid], (
        f"selector grid {gthw[1:]} != encoder grid [{grid},{grid}]; "
        f"set the selector patch_size to the encoder's ({patch})")

    info = to_onevision_frame_indices(sel, tubelet_size=cfg.tubelet_size)
    N_pf = info["num_tokens_each_frame"]
    num_frames = info["num_frames"]
    frame_mask = info["frame_mask"][0]  # (num_frames, N_pf)

    # --- cross-check against the known-good semantic-gate path ---
    mask_tub = sel.keep_mask[0].reshape(T_grid, grid * grid)
    frame_mask_ref = mask_tub.repeat_interleave(num_frames // T_grid, dim=0).to(frame_mask.device)
    assert torch.equal(frame_mask, frame_mask_ref), "adapter mask disagrees with semantic-gate path"

    # gather encoder tokens by both index derivations; must match token-for-token
    k = info["num_keep_each_frame"][0].item()
    D = tokens.shape[-1]
    idx_adapter = info["frame_keep_index"][0].to(tokens.device)             # (T, k)
    idx_ref = frame_mask_ref.nonzero(as_tuple=False)[:, 1].reshape(num_frames, k)
    gathered_adapter = tokens.gather(1, idx_adapter.unsqueeze(-1).expand(-1, -1, D))
    gathered_ref = tokens.gather(1, idx_ref.unsqueeze(-1).expand(-1, -1, D))
    assert torch.equal(gathered_adapter, gathered_ref), "gathered tokens disagree"

    print(f"selector={args.selector}  grid_thw={gthw}  ratio={args.gazing_ratio}")
    print(f"per_frame_keep(tubelet)={sel.per_frame_keep[0].tolist()}  k/frame={k}  N_pf={N_pf}")
    print(f"num_frames={num_frames}  selected tokens/frame in [0,{N_pf})  "
          f"bounds ok={bool((idx_adapter >= 0).all() and (idx_adapter < N_pf).all())}")
    print("CROSS-CHECK PASS: adapter indices == semantic-gate gather (mask + gathered tokens)")

    # --- visualization ---
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    video_disp = unnormalize(video[0]).cpu()  # (T,C,H,W) in [0,1]
    keep_mask_grid = sel.keep_mask[0].reshape(T_grid, grid, grid).cpu()
    render_overlay(video_disp, keep_mask_grid, cfg.tubelet_size, str(out))
    perframe_out = out.with_name(out.stem + "_perframe" + out.suffix)
    render_perframe_strip(
        video_disp, frame_mask.reshape(num_frames, grid, grid).cpu(), str(perframe_out),
        title=f"OneVision per-frame selection ({args.encoder}, {grid}x{grid}) — "
              f"tubelet pairs share a mask")
    print(f"saved tubelet overlay: {out}")
    print(f"saved per-frame strip: {perframe_out}")


if __name__ == "__main__":
    main()
