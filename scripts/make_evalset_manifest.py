#!/usr/bin/env python
"""Build the frozen dev/holdout eval sets for the v0.8 round (pre-registered).

Stratified sample from the 1K InternVid pilot pool (videos/internvid_pilot),
excluding the 16 held-out clips already burned for smoke tests
(videos/internvid_eval16). Stratification axes:

  - MOTION tercile: mean |gray frame diff| at 112px over 8 uniform frames --
    cheap, no optical flow;
  - SCENE cluster: SigLIP2 embedding of the middle frame -> seeded torch
    k-means (k=20) -- scene-type diversity without labels.

Degenerate filter (conservative, logged): near-static bottom tail AND
near-uniform frames (slideshow/blank).

Outputs:
  - docs/borissal/evalset_manifest.json  (COMMITTED: seed, method, clip list
    with split/motion/cluster, per-frame sha256 -- the integrity record)
  - outputs/borissal/judge_frames/<clip>/f<k>_t<sec>.jpg  (gitignored: the 8
    frozen judge frames per selected clip; byte-frozen, never re-encoded --
    re-encoding breaks judge prompt caching and integrity hashes)
  - outputs/borissal/evalset_scan.json  (cache: motion scores, resumable)

Run (scan is the slow part, ~20-40 min on M1):
  uv run python scripts/make_evalset_manifest.py            # all stages
  uv run python scripts/make_evalset_manifest.py --stage scan
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from autogaze.models.borissal.video_io import sample_frame_indices  # noqa: E402

POOL = REPO_ROOT / "videos" / "internvid_pilot"
EXCLUDE_DIR = REPO_ROOT / "videos" / "internvid_eval16"
SCAN_CACHE = REPO_ROOT / "outputs" / "borissal" / "evalset_scan.json"
EMB_CACHE = REPO_ROOT / "outputs" / "borissal" / "evalset_embeddings.pt"
FRAMES_DIR = REPO_ROOT / "outputs" / "borissal" / "judge_frames"
MANIFEST = REPO_ROOT / "docs" / "borissal" / "evalset_manifest.json"

SEED = 20260728
N_DEV, N_HOLDOUT = 60, 120
K_CLUSTERS = 20
N_JUDGE_FRAMES = 8
JUDGE_FRAME_PX = 384
JPEG_QUALITY = 90
SCAN_FRAMES, SCAN_PX = 8, 112


def _decode_frames(path, num_frames, size):
    """-> (frames uint8 (T,H,W,3), timestamps_sec list, ok flag)."""
    import av
    from PIL import Image
    try:
        container = av.open(str(path))
        stream = container.streams.video[0]
        raw = [f.to_ndarray(format="rgb24") for f in container.decode(stream)]
        duration = float(stream.duration * stream.time_base) if stream.duration else 0.0
        container.close()
    except Exception as e:  # noqa: BLE001 -- corrupt clip = filtered, not fatal
        return None, None, f"decode error: {e}"
    if len(raw) < 2:
        return None, None, "under 2 frames"
    idx = sample_frame_indices(len(raw), num_frames)
    if duration <= 0:
        duration = len(raw) / 25.0
    ts = [float(i) / max(1, len(raw) - 1) * duration for i in idx]
    frames = np.stack([
        np.array(Image.fromarray(raw[i]).resize((size, size), Image.BILINEAR))
        for i in idx])
    return frames, ts, None


def stage_scan(pool_clips):
    SCAN_CACHE.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if SCAN_CACHE.exists():
        done = {r["name"]: r for r in json.loads(SCAN_CACHE.read_text())}
    rows = []
    for i, p in enumerate(pool_clips):
        if p.name in done:
            rows.append(done[p.name])
            continue
        frames, ts, err = _decode_frames(p, SCAN_FRAMES, SCAN_PX)
        if err:
            rows.append({"name": p.name, "error": err})
        else:
            gray = frames.astype(np.float32).mean(axis=-1) / 255.0
            motion = float(np.abs(np.diff(gray, axis=0)).mean())
            frame_std = float(gray.std(axis=(1, 2)).mean())
            rows.append({"name": p.name, "motion": motion, "frame_std": frame_std,
                         "duration": round(ts[-1], 2)})
        if (i + 1) % 50 == 0:
            SCAN_CACHE.write_text(json.dumps(rows))
            print(f"  scanned {i + 1}/{len(pool_clips)}", flush=True)
    SCAN_CACHE.write_text(json.dumps(rows))
    n_err = sum(1 for r in rows if "error" in r)
    print(f"scan done: {len(rows)} clips, {n_err} decode errors")
    return rows


@torch.no_grad()
def stage_embed(names, device):
    """SigLIP2 embedding of each clip's middle frame -> (N, D) unit-norm."""
    if EMB_CACHE.exists():
        blob = torch.load(EMB_CACHE, weights_only=False)
        if blob["names"] == names:
            return blob["emb"]
    from transformers import AutoImageProcessor
    from transformers.models.siglip.modeling_siglip import SiglipVisionModel
    from PIL import Image
    enc_id = "google/siglip2-base-patch16-384"  # same encoder as eval_borissal_semantic
    model = SiglipVisionModel.from_pretrained(enc_id, attn_implementation="sdpa").to(device).eval()
    proc = AutoImageProcessor.from_pretrained(enc_id)
    embs = []
    for i, name in enumerate(names):
        frames, _, err = _decode_frames(POOL / name, 3, JUDGE_FRAME_PX)
        assert not err, f"{name} decoded in scan but not now: {err}"
        px = proc(images=Image.fromarray(frames[1]), return_tensors="pt").to(device)
        out = model(**px).pooler_output.float().cpu()
        embs.append(torch.nn.functional.normalize(out, dim=-1))
        if (i + 1) % 100 == 0:
            print(f"  embedded {i + 1}/{len(names)}", flush=True)
    emb = torch.cat(embs)
    torch.save({"names": names, "emb": emb}, EMB_CACHE)
    return emb


def kmeans(emb, k, seed, iters=100):
    """Seeded torch k-means with k-means++ init -> (N,) labels."""
    g = torch.Generator().manual_seed(seed)
    n = emb.shape[0]
    centers = emb[torch.randint(n, (1,), generator=g)]
    for _ in range(k - 1):
        d2 = torch.cdist(emb, centers).min(dim=1).values ** 2
        probs = d2 / d2.sum().clamp_min(1e-12)
        centers = torch.cat([centers, emb[torch.multinomial(probs, 1, generator=g)]])
    for _ in range(iters):
        labels = torch.cdist(emb, centers).argmin(dim=1)
        new = torch.stack([
            emb[labels == j].mean(dim=0) if (labels == j).any() else centers[j]
            for j in range(k)])
        if torch.allclose(new, centers, atol=1e-6):
            break
        centers = new
    return torch.cdist(emb, centers).argmin(dim=1).tolist()


def stage_sample(rows, device):
    exclude = {p.name for p in EXCLUDE_DIR.glob("*.mp4")}
    ok = [r for r in rows if "error" not in r and r["name"] not in exclude]
    motions = np.array([r["motion"] for r in ok])
    stds = np.array([r["frame_std"] for r in ok])
    # degenerate filter: bottom-5% motion (near-static) or near-uniform frames
    m_lo, s_lo = np.quantile(motions, 0.05), 0.02
    eligible = [r for r in ok if r["motion"] > m_lo and r["frame_std"] > s_lo]
    print(f"pool {len(rows)} -> ok {len(ok)} -> eligible {len(eligible)} "
          f"(motion>{m_lo:.4f}, frame_std>{s_lo})")

    names = [r["name"] for r in eligible]
    emb = stage_embed(names, device)
    clusters = kmeans(emb, K_CLUSTERS, SEED)
    q1, q2 = np.quantile([r["motion"] for r in eligible], [1 / 3, 2 / 3])
    for r, c in zip(eligible, clusters):
        r["cluster"] = int(c)
        r["tercile"] = 0 if r["motion"] <= q1 else (1 if r["motion"] <= q2 else 2)

    # proportional allocation over 3x20 cells, largest remainder to exact totals
    rng = np.random.default_rng(SEED)
    cells = {}
    for r in eligible:
        cells.setdefault((r["tercile"], r["cluster"]), []).append(r)
    total_take = N_DEV + N_HOLDOUT
    quotas = {c: total_take * len(v) / len(eligible) for c, v in cells.items()}
    take = {c: int(q) for c, q in quotas.items()}
    for c in sorted(quotas, key=lambda c: quotas[c] - take[c], reverse=True):
        if sum(take.values()) >= total_take:
            break
        take[c] += 1
    picked = []
    for c, members in sorted(cells.items()):
        order = rng.permutation(len(members))
        picked += [members[i] for i in order[: min(take[c], len(members))]]
    # top up if small cells under-filled their quota
    if len(picked) < total_take:
        rest = [r for r in eligible if r not in picked]
        picked += list(rng.permutation(np.array(rest, dtype=object))[: total_take - len(picked)])
    # dev/holdout split: interleave within shuffled pick so both inherit strata
    order = rng.permutation(len(picked))
    dev = {picked[i]["name"] for i in order[:N_DEV]}
    for r in picked:
        r["split"] = "dev" if r["name"] in dev else "holdout"
    print(f"picked {len(picked)}: dev {sum(r['split'] == 'dev' for r in picked)}, "
          f"holdout {sum(r['split'] == 'holdout' for r in picked)}")
    return picked


def stage_freeze(picked):
    from PIL import Image
    for i, r in enumerate(picked):
        frames, ts, err = _decode_frames(POOL / r["name"], N_JUDGE_FRAMES, JUDGE_FRAME_PX)
        assert not err, f"{r['name']}: {err}"
        stem = Path(r["name"]).stem
        clip_dir = FRAMES_DIR / stem
        clip_dir.mkdir(parents=True, exist_ok=True)
        r["frames"] = []
        for k, (frame, t) in enumerate(zip(frames, ts)):
            fp = clip_dir / f"f{k:02d}_t{t:06.2f}s.jpg"
            if not fp.exists():  # byte-freeze: NEVER re-encode an existing frame
                Image.fromarray(frame).save(fp, "JPEG", quality=JPEG_QUALITY, optimize=True)
            r["frames"].append({
                "file": str(fp.relative_to(REPO_ROOT)), "t_sec": round(t, 2),
                "sha256": hashlib.sha256(fp.read_bytes()).hexdigest()[:16]})
        if (i + 1) % 30 == 0:
            print(f"  froze {i + 1}/{len(picked)}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["scan", "all"], default="all")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    pool_clips = sorted(POOL.glob("*.mp4"))
    rows = stage_scan(pool_clips)
    if args.stage == "scan":
        return
    picked = stage_sample(rows, args.device)
    stage_freeze(picked)

    manifest = {
        "seed": SEED, "pool": str(POOL.name), "excluded": str(EXCLUDE_DIR.name),
        "method": {
            "motion": f"mean |gray frame diff|, {SCAN_FRAMES}f @ {SCAN_PX}px, terciles",
            "scene": f"siglip2-base-patch16-384 middle-frame k-means k={K_CLUSTERS} (seeded torch)",
            "filter": "bottom-5% motion or frame_std<=0.02 or decode error",
            "judge_frames": f"{N_JUDGE_FRAMES} uniform (incl first/last) @ {JUDGE_FRAME_PX}px "
                            f"JPEG q{JPEG_QUALITY}, byte-frozen (sha256-16)",
        },
        "counts": {"dev": N_DEV, "holdout": N_HOLDOUT},
        "clips": sorted(picked, key=lambda r: (r["split"], r["name"])),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=1))
    print(f"wrote {MANIFEST} ({len(picked)} clips)")


if __name__ == "__main__":
    main()
