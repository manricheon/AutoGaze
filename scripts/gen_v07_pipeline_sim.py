#!/usr/bin/env python
"""v0.7 "Datdol" pipeline-page payload, computed by the REAL selector.

Every map comes from the same functions `_anchor_novelty_select` uses; the
generator REPLICATES the selection math and then ASSERTS its keep mask equals
`Borissal.select()`'s bit-for-bit for every clip x ratio -- if the viz and the
model ever diverge, generation fails instead of lying.

Tier attribution (the new visual): each selected cube is labeled
  0 anchor   -- in the anchor mask (best-appearance moment of its site)
  1 floor    -- kept by the per-tubelet guarantee (not an anchor)
  2 novelty  -- N-dominant among the rest (N_c >= w_res * A_c)
  3 residual -- appearance-dominant surplus
"""
import base64, io, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from autogaze.models.borissal import Borissal, BorissalConfig                  # noqa: E402
from autogaze.models.borissal.modeling_borissal import _minmax_norm_global     # noqa: E402
from autogaze.models.borissal.signals_v03 import (                             # noqa: E402
    appearance_novelty, cube_best_time, dog_blob, laplacian_energy,
    temporal_median_grid,
)
from autogaze.models.borissal.video_io import load_video, unnormalize          # noqa: E402

SCALE, FRAMES, PATCH, TUB = 384, 32, 16, 2
RATIOS = [0.25, 0.5, 0.75]
FRAME_PX = 120
GAMMA, PCT = 0.62, (5, 95)

TIER_RGBA = {                       # categorical, validated family hues
    0: (13, 116, 144),              # anchor  -> teal
    1: (138, 63, 143),              # floor   -> plum
    2: (37, 99, 168),               # novelty -> blue
    3: (168, 98, 13),               # residual-> ochre
}


def _ramp(lo, hi):
    t = np.linspace(0, 1, 256)[:, None]
    return (np.array(lo) * (1 - t) + np.array(hi) * t).astype(np.uint8)

RAMPS = {
    "blue":  _ramp((244, 248, 251), (11, 40, 70)),
    "ochre": _ramp((251, 247, 240), (66, 38, 4)),
    "teal":  _ramp((247, 250, 250), (6, 48, 62)),
    "gray":  _ramp((250, 250, 249), (28, 32, 34)),
}


def _png(img):
    b = io.BytesIO(); img.save(b, "PNG", optimize=True); return b.getvalue()


def _jpg(img, q=50):
    b = io.BytesIO()
    img.convert("RGB").save(b, "JPEG", quality=q, optimize=True, progressive=True)
    return b.getvalue()


def b64(raw, mime="image/png"):
    return f"data:{mime};base64," + base64.b64encode(raw).decode()


def heat(a, ramp):
    a = np.asarray(a, np.float32)
    lo, hi = np.percentile(a, PCT)
    n = np.clip((a - lo) / (hi - lo + 1e-9), 0, 1) ** GAMMA
    rgb = RAMPS[ramp][np.clip((n * 255).astype(np.int32), 0, 255)]
    return b64(_png(Image.fromarray(rgb, "RGB").convert("P", palette=Image.ADAPTIVE, colors=32)))


def keepmask(keep2d):
    a = np.where(np.asarray(keep2d, bool), 255, 76).astype(np.uint8)
    img = Image.fromarray(np.repeat(a[..., None], 3, -1), "RGB").convert("P", palette=Image.ADAPTIVE, colors=2)
    return b64(_png(img))


def tier_png(tier2d):
    """(Hc,Wc) int in {-1,0..3} -> RGBA palette PNG; -1 transparent."""
    h, w = tier2d.shape
    rgba = np.zeros((h, w, 4), np.uint8)
    for k, c in TIER_RGBA.items():
        m = tier2d == k
        rgba[m, :3] = c
        rgba[m, 3] = 255
    return b64(_png(Image.fromarray(rgba, "RGBA")))


def build_clip(path, label):
    video = load_video(str(path), num_frames=FRAMES, size=SCALE)
    disp = unnormalize(video)[0].permute(0, 2, 3, 1).clamp(0, 1).numpy()

    m = Borissal(BorissalConfig.v0_7(scale=SCALE))
    cfg = m.config
    eps = cfg.eps
    sal = m._saliency_scores(video, TUB, PATCH, 0.0)
    lg, sp, mp = sal["luma_grid"], sal["spatial_p"], sal["motion_p"]
    B, T_grid, Hg, Wg = lg.shape
    c = cfg.score_coarsen
    Hc, Wc = Hg // c, Wg // c
    Sc, n_cubes, L = Hc * Wc, T_grid * Hc * Wc, T_grid * Hg * Wg

    # --- signal maps, exactly as the branch builds them --------------------
    edge_g = _minmax_norm_global(sp, eps)
    dog_g = _minmax_norm_global(dog_blob(lg), eps)
    lap_g = _minmax_norm_global(laplacian_energy(lg), eps)
    A = edge_g + cfg.dog_blob_weight * dog_g + cfg.anchor_lap_weight * lap_g
    canonical = temporal_median_grid(lg)
    nov_med = _minmax_norm_global(appearance_novelty(lg, canonical), eps)
    nov_st = _minmax_norm_global(mp, eps)
    N = nov_med + cfg.novelty_shortterm_weight * nov_st

    def to_cube(x):
        return F.avg_pool2d(x.reshape(B * T_grid, 1, Hg, Wg), c, c).view(B, T_grid, Sc)

    A_c, N_c = to_cube(A), to_cube(N)
    anchor_rank = A_c - cfg.anchor_novelty_lambda * N_c
    best_val, best_t = cube_best_time(anchor_rank)
    R = N_c + cfg.residual_appearance_weight * A_c
    r_max = ((1.0 + max(0.0, cfg.novelty_shortterm_weight))
             + max(0.0, cfg.residual_appearance_weight)
             * (1.0 + max(0.0, cfg.dog_blob_weight) + max(0.0, cfg.anchor_lap_weight)))
    ab, fb = r_max + 1.0, 2 * r_max + 2.0

    per_ratio = {}
    for ratio in RATIOS:
        K_patch = min(max(1, round(ratio * L)), L)
        K_cubes = min(max(T_grid, int(round(K_patch / (c * c)))), n_cubes)
        K_a = min(int(round(cfg.anchor_fraction * K_cubes)), Sc)
        anchor_mask = torch.zeros(B, T_grid, Sc, dtype=torch.bool)
        if K_a > 0:
            _, site_idx = best_val.topk(K_a, dim=-1)
            t_at = best_t.gather(1, site_idx)
            flat = anchor_mask.reshape(B, n_cubes)
            flat.scatter_(1, t_at * Sc + site_idx, torch.ones_like(site_idx, dtype=torch.bool))
            anchor_mask = flat.view(B, T_grid, Sc)
        C0 = R + ab * anchor_mask.to(R.dtype)
        _, fl = C0.topk(1, dim=-1)
        floor_mask = torch.zeros_like(anchor_mask)
        floor_mask.scatter_(-1, fl, torch.ones_like(fl, dtype=torch.bool))
        Cs = (C0 + fb * floor_mask.to(R.dtype)).reshape(B, n_cubes)
        _, ki = Cs.topk(K_cubes, dim=-1)
        keep_c = torch.zeros(B, n_cubes, dtype=torch.bool)
        keep_c.scatter_(1, ki, torch.ones_like(ki, dtype=torch.bool))
        keep_c = keep_c.view(B, T_grid, Sc)

        # ASSERT: replicated math == the model's own selection, bit for bit
        sel = m.select(video, gazing_ratio=ratio)
        model_keep = sel.keep_mask[0].view(T_grid, Hg, Wg)
        mine = (keep_c.view(B, T_grid, Hc, 1, Wc, 1)
                .expand(B, T_grid, Hc, c, Wc, c).reshape(B, T_grid, Hg, Wg))[0]
        assert torch.equal(mine, model_keep), f"viz drifted from model at ratio {ratio}"

        tiers = torch.full((T_grid, Sc), -1, dtype=torch.long)
        kc, am, fm = keep_c[0], anchor_mask[0], floor_mask[0]
        nov_dom = N_c[0] >= cfg.residual_appearance_weight * A_c[0]
        tiers[kc & am] = 0
        tiers[kc & fm & ~am] = 1
        tiers[kc & ~am & ~fm & nov_dom] = 2
        tiers[kc & ~am & ~fm & ~nov_dom] = 3
        counts = [int((tiers == k).sum()) for k in range(4)]
        per_ratio[str(ratio)] = {
            "tiers": tiers.view(T_grid, Hc, Wc), "keep": model_keep,
            "counts": counts, "K_cubes": K_cubes, "K_a": K_a,
            "pf": sel.per_frame_keep[0].tolist(), "num_keep": int(sel.num_keep[0]),
        }

    tubelets = []
    for t in range(T_grid):
        im = Image.fromarray((disp[t * TUB] * 255).astype(np.uint8), "RGB")
        entry = {
            "frame": b64(_jpg(im.resize((FRAME_PX, FRAME_PX), Image.BILINEAR)), "image/jpeg"),
            "imgs": {
                "edge": heat(edge_g[0, t].numpy(), "ochre"),
                "dog": heat(dog_g[0, t].numpy(), "ochre"),
                "lap": heat(lap_g[0, t].numpy(), "ochre"),
                "A": heat(A[0, t].numpy(), "ochre"),
                "novmed": heat(nov_med[0, t].numpy(), "blue"),
                "N": heat(N[0, t].numpy(), "blue"),
                "arank": heat(anchor_rank[0, t].view(Hc, Wc).numpy(), "teal"),
            },
            "masks": {r: keepmask(per_ratio[r]["keep"][t].numpy()) for r in per_ratio},
            "tiers": {r: tier_png(per_ratio[r]["tiers"][t].numpy()) for r in per_ratio},
            "kept": {r: int(per_ratio[r]["pf"][t]) for r in per_ratio},
        }
        tubelets.append(entry)

    return {
        "name": Path(path).name, "label": label,
        "canonical": heat(canonical[0, 0].numpy(), "gray"),
        "geom": {"T_grid": T_grid, "Hg": Hg, "Wg": Wg, "Sc": Sc, "L": L},
        "alloc": {r: {k: v for k, v in per_ratio[r].items() if k not in ("tiers", "keep")}
                  for r in per_ratio},
        "tubelets": tubelets,
    }


def main(out_json):
    specs = [
        ("0TjQiQFeum0_t0.1-4.0_fps15.9.mp4", "움직임 많음"),
        ("A9J1gkw9BI0_t1.2-4.8_fps17.8.mp4", "장면 전환 있음"),
        ("385Yc-AJOeg_t5.0-11.6_fps9.6.mp4", "식사 · 잔동작"),
        ("l9080Uwsw8s_t24.0-30.6_fps9.6.mp4", "스트리머 · 고정 테두리"),
        ("sMWfQv1ERGs_t45.2-78.0_fps1.9.mp4", "저fps 급변 + 자막"),
        ("gSH74lYC7lI_t10.8-15.6_fps13.2.mp4", "거의 정지"),
    ]
    clips = [build_clip(REPO / "videos/internvid_eval16" / n, lab) for n, lab in specs]
    payload = {"ratios": [str(r) for r in RATIOS], "clips": clips}
    Path(out_json).write_text(json.dumps(payload, separators=(",", ":")))
    print(f"wrote {out_json}  {Path(out_json).stat().st_size/1024:.0f} KB")
    for cl in clips:
        a = cl["alloc"]["0.25"]
        print(f"  {cl['label']:10s} K_cubes={a['K_cubes']} K_a={a['K_a']} "
              f"tiers(anchor/floor/nov/res)={a['counts']} pf[:6]={a['pf'][:6]}")


if __name__ == "__main__":
    main(sys.argv[1])
