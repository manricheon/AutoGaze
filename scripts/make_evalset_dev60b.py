#!/usr/bin/env python
"""Build the v0.9 confirmation set dev-60b (pre-registered, seed 20260731).

Same stratification as the v0.8 sets (motion tercile x SigLIP2 k-means k=20)
over the 1K InternVid pool, EXCLUDING eval16 and every clip already used in
dev-60 / holdout-120 (disjointness is the point: the lambda0.75 confirmation
duel must not touch the burned holdout). Reuses the v0.8 scan + embedding
caches; k-means is re-run on the remaining pool with the new seed.

Outputs:
- docs/borissal/evalset_dev60b.json (committed integrity record)
- outputs/borissal/dev60b.txt (clip list for the eval scripts)
- outputs/borissal/judge_frames/<clip>/f*.jpg (byte-frozen, never re-encoded)
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from make_evalset_manifest import (  # noqa: E402
    EXCLUDE_DIR, MANIFEST, SCAN_CACHE, EMB_CACHE, K_CLUSTERS,
    kmeans, stage_freeze,
)

SEED_B = 20260731
N_DEV_B = 60
OUT_MANIFEST = REPO_ROOT / "docs" / "borissal" / "evalset_dev60b.json"
OUT_LIST = REPO_ROOT / "outputs" / "borissal" / "dev60b.txt"


def main():
    rows = json.loads(SCAN_CACHE.read_text())
    used = {c["name"] for c in json.loads(MANIFEST.read_text())["clips"]}
    exclude = {p.name for p in EXCLUDE_DIR.glob("*.mp4")} | used
    ok = [r for r in rows if "error" not in r and r["name"] not in exclude]
    motions = np.array([r["motion"] for r in ok])
    m_lo = np.quantile(motions, 0.05)
    eligible = [r for r in ok if r["motion"] > m_lo and r["frame_std"] > 0.02]
    print(f"pool {len(rows)} -> unused ok {len(ok)} -> eligible {len(eligible)}")

    cache = torch.load(EMB_CACHE)
    emb_by_name = dict(zip(cache["names"], cache["emb"]))
    names = [r["name"] for r in eligible]
    missing = {n for n in names if n not in emb_by_name}
    if missing:
        # Clips whose eligibility flipped vs the v0.8 run (quantile thresholds
        # shift with the exclusion set) have no cached embedding; dropping them
        # keeps the cache authoritative and costs <2% of the pool. Recorded.
        print(f"dropping {len(missing)} clips absent from the v0.8 embedding cache")
        eligible = [r for r in eligible if r["name"] not in missing]
        names = [r["name"] for r in eligible]
    emb = torch.stack([torch.as_tensor(emb_by_name[n]) for n in names])
    clusters = kmeans(emb, K_CLUSTERS, SEED_B)
    q1, q2 = np.quantile([r["motion"] for r in eligible], [1 / 3, 2 / 3])
    for r, c in zip(eligible, clusters):
        r["cluster"] = int(c)
        r["tercile"] = 0 if r["motion"] <= q1 else (1 if r["motion"] <= q2 else 2)

    rng = np.random.default_rng(SEED_B)
    cells = {}
    for r in eligible:
        cells.setdefault((r["tercile"], r["cluster"]), []).append(r)
    quotas = {c: N_DEV_B * len(v) / len(eligible) for c, v in cells.items()}
    take = {c: int(q) for c, q in quotas.items()}
    for c in sorted(quotas, key=lambda c: quotas[c] - take[c], reverse=True):
        if sum(take.values()) >= N_DEV_B:
            break
        take[c] += 1
    picked = []
    for c, members in sorted(cells.items()):
        order = rng.permutation(len(members))
        picked += [members[i] for i in order[: min(take[c], len(members))]]
    if len(picked) < N_DEV_B:
        rest = [r for r in eligible if r not in picked]
        picked += list(rng.permutation(np.array(rest, dtype=object))[: N_DEV_B - len(picked)])
    picked = picked[:N_DEV_B]
    for r in picked:
        r["split"] = "dev60b"
    assert not {r["name"] for r in picked} & used, "overlap with v0.8 sets"
    print(f"picked {len(picked)} (terciles: "
          f"{[sum(r['tercile'] == t for r in picked) for t in (0, 1, 2)]})")

    stage_freeze(picked)
    OUT_MANIFEST.write_text(json.dumps({
        "seed": SEED_B, "purpose": "v0.9 lambda0.75 confirmation duel + ACR arms",
        "disjoint_from": "evalset_manifest.json (dev-60, holdout-120) + eval16",
        "clips": [{k: r[k] for k in ("name", "motion", "tercile", "cluster",
                                     "split", "frames")} for r in picked],
    }, indent=1))
    OUT_LIST.write_text("\n".join(r["name"] for r in picked) + "\n")
    print(f"wrote {OUT_MANIFEST.name} + {OUT_LIST}")


if __name__ == "__main__":
    main()
