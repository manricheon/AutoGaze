#!/usr/bin/env python
"""Build the v0.6 pipeline page's simulator payload, computed by the REAL selector.

    uv run python scripts/gen_v06_pipeline_sim.py out.json
    # then inject out.json into docs/borissal/v06-pipeline.html at the
    # <script type="application/json" id="simdata"> placeholder

The page itself is untracked (base64 images + an internal comparison section); this
script is what makes it reproducible. See docs/borissal/v06-pipeline.md 7.

Payload, computed by the REAL selector.

No JS reimplementation: every map is what modeling_borissal.py produced. The
refinement stages are built by enabling the v0.6 knobs CUMULATIVELY on top of the
v0.5-equivalent score, so stepping between two stages shows exactly what that one
knob changed.

Fixes over the first version:
  * several clips, chosen for actual motion (the first clip was the LOWEST-motion
    one in the set, which is why its motion map was blank);
  * 32 frames -> 16 tubelets, all of them;
  * gaze ratios 0.25 / 0.5 / 0.75 -- the maps are ratio-independent, so only the
    keep masks and the per-moment counts vary;
  * the score-family stages share ONE colour scale per tubelet, so an additive
    stage shows as an actual change instead of being normalised away;
  * keep masks ship as 24x24 two-colour PNGs composited in CSS (multiply), not as
    a full photo per (tubelet, ratio) -- ~200 B instead of ~6 KB each.
"""
import base64, io, json, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from autogaze.models.borissal import Borissal, BorissalConfig            # noqa: E402
from autogaze.models.borissal.video_io import load_video, unnormalize    # noqa: E402

SCALE, FRAMES, PATCH, TUB = 384, 32, 16, 2
RATIOS = [0.25, 0.5, 0.75]
FRAME_PX = 112
GAMMA = 0.62                       # <1 darkens mid-low values so structure reads
PCT = (5, 95)                      # robust per-map scale; tighter window = more contrast


def _ramp(lo, hi):
    t = np.linspace(0, 1, 256)[:, None]
    return (np.array(lo) * (1 - t) + np.array(hi) * t).astype(np.uint8)

RAMPS = {
    "motion": _ramp((244, 248, 251), (11, 40, 70)),    # 움직임 -> 파랑
    "edges":  _ramp((251, 247, 240), (66, 38, 4)),     # 윤곽/영역 -> 오커
    "score":  _ramp((247, 250, 250), (6, 48, 62)),     # 점수 계열 -> 청록
}
# diverging pair for "이 단계가 무엇을 바꿨는가": ochre = 올라감, teal = 내려감, 중앙은 회색
_DIV = np.concatenate([_ramp((6, 72, 92), (232, 234, 233)), _ramp((232, 234, 233), (140, 74, 6))])


def heat(a, ramp, lo=None, hi=None):
    """Robust per-map scale. A shared scale across stages was tried first and made
    every early stage unreadable: the additive refinements (+0.5, +0.5, +0.3) own
    the top of the range, squeezing `fused` to near-white. Change is shown by the
    diff map instead, which is the honest way to keep both contrast and truth."""
    a = np.asarray(a, dtype=np.float32)
    if lo is None or hi is None:
        lo, hi = (float(x) for x in np.percentile(a, PCT))
    n = np.clip((a - lo) / (hi - lo + 1e-9), 0, 1) ** GAMMA
    rgb = RAMPS[ramp][np.clip((n * 255).astype(np.int32), 0, 255)]
    img = Image.fromarray(rgb, "RGB").convert("P", palette=Image.ADAPTIVE, colors=32)
    return _b64(_png(img))


def diffmap(after, before):
    """Signed change, symmetric around zero so the neutral midpoint means 'no change'."""
    d = np.asarray(after, np.float32) - np.asarray(before, np.float32)
    s = float(np.percentile(np.abs(d), 98)) or 1e-6
    n = np.clip(d / s, -1, 1)
    n = np.sign(n) * (np.abs(n) ** 0.8)                 # lift small changes
    idx = np.clip(((n + 1) * 0.5 * 511).astype(np.int32), 0, 511)
    img = Image.fromarray(_DIV[idx], "RGB").convert("P", palette=Image.ADAPTIVE, colors=32)
    return _b64(_png(img))


def keepmask(keep2d):
    """White where kept, dark where dropped. CSS multiplies it over the frame."""
    a = np.where(np.asarray(keep2d, bool), 255, 76).astype(np.uint8)   # ~0.30 multiply: dropped stays faintly visible for context
    img = Image.fromarray(np.repeat(a[..., None], 3, -1), "RGB").convert(
        "P", palette=Image.ADAPTIVE, colors=2)
    return _b64(_png(img))


def _png(img):
    b = io.BytesIO(); img.save(b, "PNG", optimize=True); return b.getvalue()


def _b64(raw, mime="image/png"):
    return f"data:{mime};base64," + base64.b64encode(raw).decode()


def _jpg(img, q=55):
    b = io.BytesIO()
    img.convert("RGB").save(b, "JPEG", quality=q, optimize=True, progressive=True)
    return b.getvalue()


V6 = ["laplacian_gate", "static_guard", "keyframe_prior", "center_bias"]

def cfg_at(n):
    """v0.5 behaviour (bt601 kept on) + the first n v0.6 refinement knobs."""
    o = dict(luma_mode="bt601", laplacian_gate=False, static_guard=False,
             keyframe_prior=False, center_bias=0.0, per_frame_allocation="global")
    for k in V6[:n]:
        o[k] = 0.3 if k == "center_bias" else True
    return BorissalConfig.v0_6(scale=SCALE, **o)


STAGE_META = [
    ("input",    "들어온 화면",        "frame",  "이 순간(튜블렛)이 담고 있는 프레임. 여기서부터 모든 신호가 나온다."),
    ("motion",   "움직임 (모션)",      "motion", "프레임 사이가 달라진 양. 잡음 바닥을 깎아낸 뒤라 실제로 움직인 데만 남는다."),
    ("edges",    "윤곽 (엣지)",        "edges",  "밝기가 급하게 바뀌는 선. 반복 무늬는 이미 눌러놨다. 물체 경계에서만 서고 안쪽은 빈다."),
    ("blobs",    "영역 (region)",      "edges",  "윤곽이 비워둔 물체 '본체(면)'를 채우는 세 번째 층. 균일한 색의 몸통은 모션도 윤곽도 0이라 이 채널이 필요하다."),
    ("fused",    "세 층을 합친 점수",   "score",  "움직임·윤곽·영역을 비율대로 섞은 기본 점수. 화면 전체에 고르게 퍼진 지도는 가중치가 깎인다."),
    ("gate",     "겹무늬 한 번 더 깎기", "score", "움직임 대비 잔무늬가 촘촘한 곳을 곱셈으로 누른다."),
    ("static",   "멈춘 곳 살려주기",    "score",  "그 순간이 멈춰 있을 때만 윤곽 힘을 되돌려준다. 글자·문서·고정 샷이 살아남는 이유."),
    ("keyframe", "장면 시작점 밀어주기", "score", "8프레임 주기 기준점과 장면이 바뀌는 지점에 점수를 더한다."),
    ("center",   "가운데 살짝 우대",    "score",  "완만한 가우시안 가점. 마지막 손질."),
    ("cube",     "2×2 덩어리로 묶기",   "score",  "2×2 네 칸이 같은 점수를 갖게 만든다. 흩어진 낱칸이 아니라 덩어리째 골라진다."),
    ("selected", "고른 결과",          "keep",   "예산만큼 실제로 남긴 칸. 어두운 부분은 버려진 칸."),
]
SCORE_KEYS = {"fused", "gate", "static", "keyframe", "center", "cube"}


def build_clip(path, label):
    video = load_video(str(path), num_frames=FRAMES, size=SCALE)
    disp = unnormalize(video)[0].permute(0, 2, 3, 1).clamp(0, 1).numpy()

    full = Borissal(BorissalConfig.v0_6(scale=SCALE))
    _, inter = full.select_with_intermediates(video, gazing_ratio=RATIOS[0])
    motion = inter["motion_norm"][0].numpy()
    edges = inter["spatial_norm"][0].numpy()
    T_grid, Hg, Wg = motion.shape[0], motion.shape[1], motion.shape[2]

    tub = video[0].mean(1).view(1, T_grid, TUB, SCALE, SCALE).mean(2)
    loc, _ = full._extra_channels(video, tub, TUB, PATCH)
    blobs = loc[0][1][0].numpy() if loc else np.zeros_like(motion)

    maps = {"motion": motion, "edges": edges, "blobs": blobs}
    for i in range(len(V6) + 1):
        c = cfg_at(i)
        s = Borissal(c)._saliency_scores(video, TUB, PATCH, c.motion_weight)["score"][0]
        maps[["fused", "gate", "static", "keyframe", "center"][i]] = s.numpy()
    sf = full._saliency_scores(video, TUB, PATCH, "auto")["score"][0]
    maps["cube"] = torch.nn.functional.avg_pool2d(sf.unsqueeze(1), 2, 2) \
        .repeat_interleave(2, -1).repeat_interleave(2, -2)[:, 0].numpy()

    sels, alloc = {}, {}
    for r in RATIOS:
        sel = full.select(video, gazing_ratio=r)
        sels[r] = sel.keep_mask[0].view(T_grid, Hg, Wg).numpy()
        N_pf, L = Hg * Wg, T_grid * Hg * Wg
        K = min(max(T_grid, round(r * L)), L)
        m = min(max(1, int(round(0.25 * K / T_grid))), K // T_grid)
        alloc[str(r)] = {"K_total": K, "m_floor": m,
                         "k_gate": min(N_pf, 2 * ((K + T_grid - 1) // T_grid)),
                         "free": K - T_grid * m, "uniform_share": K // T_grid,
                         "per_frame_keep": sel.per_frame_keep[0].tolist(),
                         "num_keep": int(sel.num_keep[0])}

    tubelets = []
    for t in range(T_grid):
        im = Image.fromarray((disp[t * TUB] * 255).astype(np.uint8), "RGB")
        imgs = {"input": _b64(_jpg(im.resize((FRAME_PX, FRAME_PX), Image.BILINEAR)), "image/jpeg")}
        for key, _, kind, _ in STAGE_META:
            if kind in ("frame", "keep"):
                continue
            imgs[key] = heat(maps[key][t], kind)
        # what each refinement actually changed, vs the stage before it
        diffs = {}
        for a, b in (("gate", "fused"), ("static", "gate"), ("keyframe", "static"),
                     ("center", "keyframe"), ("cube", "center")):
            diffs[a] = diffmap(maps[a][t], maps[b][t])
        tubelets.append({
            "frame": imgs["input"],
            "imgs": imgs,
            "diffs": diffs,
            "masks": {str(r): keepmask(sels[r][t]) for r in RATIOS},
            "kept": {str(r): int(alloc[str(r)]["per_frame_keep"][t]) for r in RATIOS},
            "is_keyframe": (t * TUB) % 8 == 0,
        })
    return {"name": Path(path).name, "label": label, "alloc": alloc,
            "tubelets": tubelets,
            "geom": {"T_grid": T_grid, "Hg": Hg, "Wg": Wg, "N_pf": Hg * Wg,
                     "L": T_grid * Hg * Wg}}


def main(out_json, specs):
    clips = [build_clip(REPO / "videos/internvid_eval16" / n, lab) for n, lab in specs]
    payload = {
        "geom": {"frames": FRAMES, "scale": SCALE, "patch": PATCH, "tubelet": TUB},
        "ratios": RATIOS,
        "stages": [{"key": k, "title": ti, "kind": kd, "note": nt} for k, ti, kd, nt in STAGE_META],
        "clips": clips,
    }
    Path(out_json).write_text(json.dumps(payload, separators=(",", ":")))
    print(f"wrote {out_json}  {Path(out_json).stat().st_size/1024:.0f} KB")
    for c in clips:
        print(f"  {c['label']:14s} {c['name'][:34]:34s} T_grid={c['geom']['T_grid']} "
              f"kept@0.25={c['alloc']['0.25']['num_keep']} pf0={c['alloc']['0.25']['per_frame_keep'][:4]}")


if __name__ == "__main__":
    SPECS = [
        ("0TjQiQFeum0_t0.1-4.0_fps15.9.mp4",   "움직임 많음"),
        ("A9J1gkw9BI0_t1.2-4.8_fps17.8.mp4",   "장면 전환 있음"),
        ("gSH74lYC7lI_t10.8-15.6_fps13.2.mp4", "거의 정지"),
    ]
    main(sys.argv[1], SPECS)
