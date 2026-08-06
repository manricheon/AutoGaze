#!/usr/bin/env python
"""Qualitative selector report: patch-overlay strips + caption diffs + verdicts.

A file-based, model-agnostic inspection harness. Inputs are artifacts the
quantitative pipeline already produces (captions.json from eval_mllm_attach,
verdict/job JSONL from eval_caption_judge, the frozen judge-frame manifest);
output is ONE self-contained HTML (all images base64) that shows, per clip:

  1. the frozen judge frames (ground truth),
  2. per-spec/per-ratio overlays -- dropped patches dimmed, kept patches at
     full brightness -- rendered by running the REAL selector at report time,
  3. every caption side by side (dense + each spec@ratio),
  4. blinded-judge outcomes and one-line reasons, when verdict files given.

Because captions are read from plain captions.json ({"clip":..., "dense":...,
"<spec>@<ratio>": ...}), any generator can feed this -- including company-side
models: drop in their captions.json and the same report renders.

Usage:
  uv run python scripts/make_qual_report.py \
    --captions outputs/borissal/v08_sweep/stageb_025/captions.json \
    --clips 02k0dxleL5A_t0.1-3.9_fps16.7.mp4,... \
    --specs "v0.7,signal_grid=fine,anchor_novelty_lambda=0.75;v0.7;random" \
    --ratios 0.25,0.5 \
    --verdicts outputs/borissal/v08_sweep/stageb_verdicts.jsonl \
    --jobs outputs/borissal/v08_sweep/stageb_jobs.jsonl \
    --out outputs/borissal/v08_sweep/qual_report.html
"""

import argparse
import base64
import html as html_mod
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

MANIFEST = REPO_ROOT / "docs" / "borissal" / "evalset_manifest.json"
DIM = 0.22          # dropped-patch luminance (0.22: structure faintly visible)


def _b64(img, width, quality):
    from PIL import Image
    if img.width > width:
        img = img.resize((width, int(img.height * width / img.width)), Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _short(spec):
    """Compact display label for a selector spec."""
    if spec in ("dense", "random", "orig"):
        return spec
    parts = [p for p in spec.split(",")[1:]]
    return spec.split(",")[0] + ("+" + "+".join(p.split("=")[0].replace("signal_grid", "sg")
                                                .replace("anchor_novelty_lambda", "λ")
                                                .replace("novelty_shortterm_weight", "wst")
                                                + ("=" + p.split("=")[1] if "=" in p and
                                                   p.split("=")[0] != "signal_grid" else
                                                   ("=" + p.split("=")[1] if "=" in p else ""))
                                                for p in parts) if parts else "")


def first_outcome(job, verdict):
    if verdict["overall"] == "tie":
        return "tie"
    a_first = job["a_is"] == job["first"]
    return "win" if (a_first == (verdict["overall"] == "A")) else "loss"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--captions", nargs="+", required=True,
                   help="one or more captions.json (merged per clip)")
    p.add_argument("--clips", required=True,
                   help="comma-separated clip names, or a file with one per line")
    p.add_argument("--specs", required=True, help="';'-separated selector specs")
    p.add_argument("--ratios", default="0.25")
    p.add_argument("--verdicts", default=None, help="verdict JSONL (optional)")
    p.add_argument("--jobs", default=None, help="judge jobs JSONL (needed with --verdicts)")
    p.add_argument("--videos-dir", default=str(REPO_ROOT / "videos" / "internvid_pilot"))
    p.add_argument("--manifest", default=str(MANIFEST),
                   help="eval-set manifest holding the frozen frame paths "
                        "(dev-60/holdout-120 by default; pass evalset_dev60b.json "
                        "for the v0.9 set)")
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--scale", type=int, default=384)
    p.add_argument("--frames-per-clip", type=int, default=4)
    p.add_argument("--img-width", type=int, default=200)
    p.add_argument("--jpeg-quality", type=int, default=60)
    p.add_argument("--title", default="Selector qualitative report")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    from PIL import Image
    from autogaze.models.borissal.video_io import load_video          # noqa: E402
    from eval_borissal_semantic import build_selection                # noqa: E402

    manifest = {c["name"]: c
                for c in json.loads(Path(args.manifest).read_text())["clips"]}
    if Path(args.clips).exists():
        clips = [l.strip() for l in Path(args.clips).read_text().splitlines() if l.strip()]
    else:
        clips = [c.strip() for c in args.clips.split(",") if c.strip()]
    specs = [s for s in args.specs.split(";") if s]
    ratios = [float(r) for r in args.ratios.split(",")]

    caps = defaultdict(dict)
    for cf in args.captions:
        for row in json.load(open(cf)):
            caps[row["clip"]].update({k: v for k, v in row.items() if k != "clip"})

    # verdict lookup: (tag, clip, first) -> {"outcome": pair outcome, "reasons": [...]}
    judge = {}
    if args.verdicts and args.jobs:
        jobs = {json.loads(l)["id"]: json.loads(l)
                for l in Path(args.jobs).read_text().splitlines()}
        pairs, reasons = defaultdict(dict), defaultdict(list)
        for line in Path(args.verdicts).read_text().splitlines():
            v = json.loads(line)
            j = jobs.get(v["id"])
            if not j:
                continue
            k = (j["tag"], j["clip"], j["first"])
            pairs[k][j["order"]] = first_outcome(j, v)
            reasons[k].append(v["reason"])
        for k, d in pairs.items():
            out = d.get(0) if d.get(0) == d.get(1) else "tie"
            judge[k] = {"outcome": out, "reasons": reasons[k]}

    OUT_KO = {"win": "승", "loss": "패", "tie": "무"}
    sections = []
    for ci, name in enumerate(clips):
        video = load_video(str(Path(args.videos_dir) / name),
                           num_frames=args.num_frames, size=args.scale)
        frames_meta = manifest[name]["frames"]
        n8 = len(frames_meta)
        pick = np.linspace(0, n8 - 1, args.frames_per_clip).round().astype(int).tolist()
        origs = [Image.open(REPO_ROOT / frames_meta[k]["file"]) for k in pick]
        ts = [frames_meta[k]["file"].rsplit("_t", 1)[-1].replace(".jpg", "") for k in pick]

        rows = [("원본 (심판 프레임)", "", [_b64(im, args.img_width, args.jpeg_quality)
                                           for im in origs])]
        for ratio in ratios:
            for spec in specs:
                sel = build_selection(spec, video, ratio, 0.0)
                km = sel.keep_mask[0]
                Hg = Wg = args.scale // 16
                Tg = km.numel() // (Hg * Wg)
                km = km.view(Tg, Hg, Wg).float().numpy()
                imgs = []
                for k, im in zip(pick, origs):
                    arr = np.asarray(im).astype(np.float32)
                    tub = min(Tg - 1, int(k / n8 * Tg))
                    H = arr.shape[0]
                    up = np.kron(km[tub], np.ones((H // Hg, H // Wg), np.float32))[..., None]
                    over = (arr * (up + (1 - up) * DIM)).clip(0, 255).astype(np.uint8)
                    imgs.append(_b64(Image.fromarray(over), args.img_width,
                                     args.jpeg_quality))
                cov = float(km.mean())
                rows.append((f"{_short(spec)} @{ratio}",
                             f"keep {cov:.0%}", imgs))

        strip = "\n".join(
            f'<div class="row"><div class="lbl">{html_mod.escape(lbl)}'
            f'<span class="cov">{cov}</span></div>' +
            "".join(f'<img src="{u}" alt="">' for u in urls) + "</div>"
            for lbl, cov, urls in rows)

        cap_rows = []
        c = caps.get(name, {})
        order = (["dense"] +
                 [f"{s}@{r}" for r in ratios for s in specs if f"{s}@{r}" in c])
        for key in order:
            if key not in c:
                continue
            cap_rows.append(f"<tr><th>{html_mod.escape(_short(key.split('@')[0]) + ('@' + key.split('@')[1] if '@' in key else ''))}</th>"
                            f"<td>{html_mod.escape(c[key])}</td></tr>")
        cap_html = ("<table class='caps'>" + "".join(cap_rows) + "</table>"
                    if cap_rows else "<p class='muted'>캡션 없음</p>")

        jd_html = ""
        if judge:
            items = []
            for (tag, clip2, first), info in sorted(judge.items()):
                if clip2 != name:
                    continue
                items.append(
                    f"<li><b>{html_mod.escape(_short(first))}</b> {tag}: "
                    f"<span class='o-{info['outcome']}'>{OUT_KO[info['outcome']]}</span>"
                    "<ul>" + "".join(f"<li class='muted'>{html_mod.escape(r)}</li>"
                                     for r in info["reasons"][:2]) + "</ul></li>")
            if items:
                jd_html = "<details open><summary>심판 판정 (블라인드, 스왑 2회)</summary><ul class='jd'>" \
                          + "".join(items) + "</ul></details>"

        frame_hdr = "".join(f"<span class='ts'>t{t}</span>" for t in ts)
        sections.append(
            f"<section><h2>{ci+1}. {html_mod.escape(name)}</h2>"
            f"<div class='row hdr'><div class='lbl'></div>{frame_hdr}</div>"
            f"{strip}{cap_html}{jd_html}</section>")

    css = """
:root{--bg:#fff;--ink:#1a1c1e;--mut:#6a6f75;--line:#e3e5e8;--acc:#0b57d0;
--win:#1a7f37;--loss:#c0392b;--tie:#8a6d1a}
@media(prefers-color-scheme:dark){:root{--bg:#131416;--ink:#e6e8ea;--mut:#9aa0a6;
--line:#2a2d31;--acc:#8ab4f8;--win:#7ee2a8;--loss:#f28b82;--tie:#e2c96f}}
:root[data-theme=dark]{--bg:#131416;--ink:#e6e8ea;--mut:#9aa0a6;--line:#2a2d31;
--acc:#8ab4f8;--win:#7ee2a8;--loss:#f28b82;--tie:#e2c96f}
:root[data-theme=light]{--bg:#fff;--ink:#1a1c1e;--mut:#6a6f75;--line:#e3e5e8;
--acc:#0b57d0;--win:#1a7f37;--loss:#c0392b;--tie:#8a6d1a}
body{background:var(--bg);color:var(--ink);font:14px/1.55 -apple-system,system-ui,
'Apple SD Gothic Neo',sans-serif;max-width:1080px;margin:0 auto;padding:24px}
h1{font-size:20px}h2{font-size:15px;border-top:1px solid var(--line);
padding-top:18px;margin-top:28px}
.row{display:flex;gap:4px;align-items:center;margin:2px 0;overflow-x:auto}
.row img{border-radius:3px;display:block}
.lbl{flex:0 0 190px;font-size:12px;color:var(--ink)}
.lbl .cov{display:block;color:var(--mut);font-size:11px}
.hdr .ts{width:200px;text-align:center;color:var(--mut);font-size:11px}
table.caps{border-collapse:collapse;margin:12px 0;font-size:13px}
.caps th{text-align:left;vertical-align:top;padding:6px 10px 6px 0;
white-space:nowrap;color:var(--acc);font-weight:600}
.caps td{padding:6px 0;border-top:1px solid var(--line)}
.caps tr:first-child td{border-top:none}
.jd{font-size:13px}.jd ul{margin:2px 0 8px}
.muted{color:var(--mut)}.o-win{color:var(--win);font-weight:700}
.o-loss{color:var(--loss);font-weight:700}.o-tie{color:var(--tie);font-weight:700}
summary{cursor:pointer;color:var(--acc);font-size:13px;margin-top:8px}
p.note{color:var(--mut);font-size:12px}
"""
    doc = (f"<meta charset='utf-8'><title>{html_mod.escape(args.title)}</title>"
           f"<style>{css}</style><h1>{html_mod.escape(args.title)}</h1>"
           "<p class='note'>밝은 패치 = 셀렉터가 유지, 어두운 패치 = 드롭. "
           "오버레이는 리포트 생성 시점에 실제 셀렉터를 실행해 렌더(픽셀 마스킹은 "
           "시각화일 뿐, 실제 스택은 토큰 드롭). 심판 판정은 블라인드·순서스왑 2회, "
           "불일치=무승부.</p>"
           + "\n".join(sections))
    Path(args.out).write_text(doc)
    print(f"wrote {args.out} ({len(doc)//1024} KB, {len(clips)} clips)")


if __name__ == "__main__":
    main()
