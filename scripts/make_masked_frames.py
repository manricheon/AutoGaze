#!/usr/bin/env python
"""Render selector-masked judge frames for the Claude-relative-description screen.

For each (clip, selector spec, ratio): run the REAL selector on the clip,
take the SAME 8 timestamps as the frozen judge frames (manifest), and save a
copy of each frame where dropped patches are darkened to near-black.

The screen this feeds (proposed 2026-07-29, user idea): a strong VLM captions
the masked frames and the ORIGINAL frames separately; the existing judge
harness scores masked-caption vs dense-caption pairs. Because both captions
come from the same captioner, model-taste bias cancels -- the only varying
factor is the selection. This is an INFORMATION-SUFFICIENCY proxy (pixel
masking != true token drop; the encoder never sees "holes"), so it screens
knob candidates; it never replaces on-stack judgement (Stage B/C).

Frames land in outputs/borissal/claude_screen/<clip_stem>/<safe_spec>/f*.jpg
plus originals copied once per clip under .../<clip_stem>/orig/.
Frame k (uniform over 8) maps to tubelet floor(k/8 * T_grid) of the 16-frame
selector run -- the same uniform sampling both pipelines use.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from autogaze.models.borissal import Borissal, BorissalConfig            # noqa: E402
from autogaze.models.borissal.video_io import load_video                 # noqa: E402
from eval_borissal_semantic import build_selection                       # noqa: E402

MANIFEST = REPO_ROOT / "docs" / "borissal" / "evalset_manifest.json"
OUT = REPO_ROOT / "outputs" / "borissal" / "claude_screen"
DIM = 0.06          # dropped-patch luminance multiplier (near-black, structure gone)


def safe(spec):
    return spec.replace(",", "+").replace("=", "-").replace(".", "_")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clips-file", required=True)
    p.add_argument("--videos-dir", default=str(REPO_ROOT / "videos" / "internvid_pilot"))
    p.add_argument("--specs", required=True, help="';'-separated selector specs")
    p.add_argument("--ratios", default="0.25")
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--scale", type=int, default=384)
    args = p.parse_args()

    from PIL import Image
    manifest = {c["name"]: c for c in json.loads(MANIFEST.read_text())["clips"]}
    clips = [n.strip() for n in Path(args.clips_file).read_text().splitlines() if n.strip()]
    specs = [s for s in args.specs.split(";") if s]
    ratios = [float(r) for r in args.ratios.split(",")]

    for ci, name in enumerate(clips):
        video = load_video(str(Path(args.videos_dir) / name), num_frames=args.num_frames,
                           size=args.scale)
        frames_meta = manifest[name]["frames"]
        stem = Path(name).stem
        # copy originals once (reuse frozen bytes -- do not re-encode)
        orig_dir = OUT / stem / "orig"
        orig_dir.mkdir(parents=True, exist_ok=True)
        for fr in frames_meta:
            src = REPO_ROOT / fr["file"]
            dst = orig_dir / Path(fr["file"]).name
            if not dst.exists():
                dst.write_bytes(src.read_bytes())
        for spec in specs:
            for ratio in ratios:
                sel = build_selection(spec, video, ratio, 0.0)
                km = sel.keep_mask[0]  # (L,) flat over T_grid*Hg*Wg
                Hg = Wg = args.scale // 16
                Tg = km.numel() // (Hg * Wg)
                km = km.view(Tg, Hg, Wg).float()
                out_dir = OUT / stem / f"{safe(spec)}@{ratio}"
                out_dir.mkdir(parents=True, exist_ok=True)
                n8 = len(frames_meta)
                for k, fr in enumerate(frames_meta):
                    src = REPO_ROOT / fr["file"]
                    img = np.asarray(Image.open(src)).astype(np.float32)
                    tub = min(Tg - 1, int(k / n8 * Tg))
                    mask = km[tub].numpy()                       # (Hg, Wg) 1=keep
                    H = img.shape[0]
                    up = np.kron(mask, np.ones((H // Hg, H // Wg), np.float32))[..., None]
                    dimmed = (img * (up + (1 - up) * DIM)).clip(0, 255).astype(np.uint8)
                    Image.fromarray(dimmed).save(out_dir / Path(fr["file"]).name,
                                                 "JPEG", quality=90, optimize=True)
        if (ci + 1) % 5 == 0:
            print(f"  {ci + 1}/{len(clips)} clips", flush=True)
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
