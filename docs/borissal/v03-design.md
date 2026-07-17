# Borissal v0.3 — Design Spec (2026-07-17)

Approved design for the next non-learned selector iteration. This spec is
the contract for implementation planning; measurement results and
adopt/reject verdicts will be recorded in `design.md` (dev log) as they
land, per the v0.2 precedent. Companion docs: `reference.md` (user-facing
knobs, updated on adoption), `approach-ko.md` (Korean explainer, updated
on adoption).

## 1. Positioning and goals

v0.3 is **not a single mechanism — it is a candidate bank + quantitative
gate + combination search**, the scaled-up version of the process that
produced v0.2 (seven elements implemented, each gated, some rejected as
negative results). Every candidate is an independent config knob; with all
knobs off the pipeline is **bit-identical to v0.2** (locked by a
regression test). Only gate-passing combinations are admitted to the v0.3
preset.

User-set priorities (2026-07-17), in order:

1. **Semantic recall** — the description-aligned axis (SigLIP2 MAP-head
   recall on the held-out 16-clip set; v0.2 baseline 0.325 ± 0.022).
2. **Edge/texture bias mitigation** — luma gradient fires on repetitive
   background texture and only on object boundaries, never interiors.
3. **Temporal selection stability** — previously excluded as a
   streaming-UI nicety; now an explicit target, but admitted only under a
   recall-non-degradation condition (see §5).
4. **Camera-motion robustness** (subsumed under 1: pan/zoom makes
   frame-diff fire everywhere, wasting budget).

Enriching v1's input feature bank is an expected side effect, not a gate.

Constraints (unchanged from the v0 line):

- **Pure classical signals only** — zero learned weights, no frozen
  pretrained models (a hybrid/frozen path is a possible v0.4+ direction,
  explicitly out of scope here).
- **Mobile-safe ops only**: conv/pool/elementwise/topk. No FFT, no
  general sort in new code paths, no connected components, no sequential
  raster scans. Fixed small matmuls (e.g. DCT as a constant matrix) are
  allowed — a matmul is a delegate-native op.
- **Latency budget ≤ 25 ms/clip** on the dev-box CPU (v0.2 preset is at
  15.4 ms; new-mechanism headroom ≈ 10 ms). v0.3's cost also lands inside
  v1's `_grid_inputs`, so cheap matters twice.
- **The `Selection` output contract and the canonical ascending
  `keep_index` convention are immutable** (downstream attachability).
- Fully vectorized torch — no Python loops over batch/frame/patch.

## 2. Candidate bank

Three tiers. Tier 1 is implemented unconditionally; Tier 2 is implemented
only for axes where the Tier-1 combination stalls (§5); Tier 3 is
recorded for completeness and deliberately not implemented now.

### Tier 1 (implement now — highest value per millisecond)

| id | mechanism | principle & key reference | targets | est. cost |
|---|---|---|---|---|
| `motion_cs` | **Motion center-surround**: replace pooled diff energy `D` with `relu(D − avgpool_large(D))` | Uniform ego-motion produces a flat diff field whose center-surround difference ≈ 0; an independently moving object survives as a local peak (Itti motion conspicuity 1998; Mahadevan & Vasconcelos 2010, simplified) | camera | <0.5 ms |
| `coherence_gate` | **Structure-tensor coherence gating** of the existing gradient channel: `grad × (1 − coherence)^γ`, closed-form coherence `((a−c)² + 4b²)/(a+c+ε)²` from Gaussian-smoothed gradient products | Repetitive gratings and long straight edges are maximally coherent (λ1≫λ2) and get suppressed; multi-orientation object micro-structure (λ1≈λ2) survives (Harris 1988; Förstner 1987; Weickert 1999) | texture | ~1.5 ms |
| `signature` | **Image signature** at grid/low res: DCT as a fixed matmul (`D @ X @ D.T`), `sign()`, inverse matmul, square, small blur — *not* FFT | Sign-only reconstruction concentrates energy on spatially sparse foreground and kills spectrally sparse (periodic) background; fires on object support, not just boundaries (Hou, Harel & Koch, TPAMI 2012) | recall, texture | ~0.5 ms |
| `color_rarity` | **Global color rarity**: per-patch mean color soft-binned onto K≈32 fixed centers (one matmul), rarity = distance-weighted inverse histogram mass; degenerate fallback = Mahalanobis distance to global mean color (Achanta 2009) | Rare colors mark description-relevant objects and fire on **interiors** uniformly (whole red car, not its silhouette) (Cheng et al., CVPR 2011, HC variant — no segmentation) | recall, texture | ~0.5 ms |
| `dog_blob` | **Multi-scale DoG blob channel**: avgpool pyramid differences on tubelet-mean luma, \|·\|, max over scales (2–6 grid cells) | Objects are blobs at some scale; coarse DoG extrema fire on interiors (Lindeberg 1998) — the cheapest interior filler | texture | <1 ms |
| `fusion_norm` | **Content-adaptive channel fusion**: Itti's N(·) peak-promotion (scale each map by `(M − m̄_localmax)²`) and/or bounded entropy gating (softmax-entropy of a map ↓ ⇒ weight ↓, floor 0.3) | A map that fires everywhere (gradient on texture; motion during a pan) gets its *fusion weight* crushed before blending — map-level texture suppression plus a free camera fallback (Itti 1998; vid-TLDR CVPR 2024 for the entropy variant) | texture, camera, fusion | ~0 ms |
| `score_ema` | **Temporal score EMA**: `S̄_t = α·S̄_{t−1} + (1−α)·S_t`, unrolled within a clip as one lower-triangular decay-matrix matmul (no loop); streaming carries one grid-shaped state tensor | Leaky-integrator evidence accumulation; classical temporal filtering | stability | ~0 ms |
| `select_hysteresis` | **Selection hysteresis**: additive bonus ε (pre-topk) to patches kept in the previous tubelet | Discrete cousin of EMA — stabilizes the kept set without smoothing scores | stability | ~0 ms |

### Tier 2 (implement only where Tier 1 stalls — see §5 trigger rule)

| id | mechanism | principle & reference | targets | est. cost |
|---|---|---|---|---|
| `distinctness` | Patch distinctness via global self-similarity: per-cell descriptor (Lab mean + tiny gradient stats), 576×576 Gram matmul, distinctness = 1 − mean of top-k similarities, with a precomputed spatial-discount mask (compare only cells >r apart) | Repetitive texture has many near-duplicates → distinctness ≈ 0 across the whole texture; object patches are globally rare (Goferman 2010; Margolin 2013) | recall, texture | ~0.5 ms |
| `border_prior` | Boundary-connectivity background prior: similarity of every cell to the ~92 border cells (one small matmul); bgness gates the fused score multiplicatively | Background texture connects to the frame border in appearance; kills it *even when high-gradient* (Zhu et al., CVPR 2014, soft variant — no superpixels) | texture | ~0.1 ms |
| `mbd` | Minimum-barrier distance, parallel approximation: per-cell (hi, lo) path extrema seeded at the border, K≈24 Bellman–Ford iterations of shift (`roll`/pad) + elementwise min/max at grid res | Salient regions have a high barrier on every path to the border; the strongest classical interior-filler, robust to smooth illumination ramps (Zhang et al., ICCV 2015 — published raster-scan version is a showstopper; this is the fixed-iteration parallel form) | recall, texture | ~1–2 ms |
| `gme` | Integral-projection global motion estimation + compensated diff: row/col mean profiles per frame, conv1d correlation over shifts ∈ [−16, 16], winner via topk-1, shift applied through a 33-way one-hot pad+slice (static-graph-safe, no dynamic `roll`) | The accurate classical pan/tilt fix (Alliney & Morandi 1986; Dufaux & Konrad 2000). Translation-only; zoom/rotation decorrelate the profiles → estimator returns ~0 shift → graceful fallback to `motion_cs` | camera | ~1–2 ms |

### Tier 3 (recorded, deliberately not implemented)

- **BMS approximation** (border-seeded iterative maxpool propagation) —
  overlaps `mbd`, which is cheaper per unit of interior-filling.
- **Gabor orientation-contrast bank** — 4× channel cost; `coherence_gate`
  covers the anti-texture role at a fraction of the price.
- **Coarse block-matching dominant-motion field** — ~5–10 ms even
  downsampled; `gme` + `motion_cs` cover ~80% at a quarter of the cost.
- **Sharpness/defocus prior** — needs bimodality confidence gating and is
  domain-fragile (deep-DOF phone footage); revisit per-domain like
  `center_bias`.
- **Spectral residual / PQFT via DFT-matmul** — allowed as an
  *experimental knob* if someone wants it (`double_diff` precedent), but
  `signature` occupies the same niche with better op safety.
- **GBVS, RARE2012, Kadir–Brady, radial symmetry, BING, background
  subtraction GMMs, phase correlation** — surveyed and rejected
  (duplicative, scatter-op-bound, learned weights, static-camera
  assumption, or FFT-bound, respectively).

## 3. Pipeline integration

Signal flow (bracketed items are the new knobs; everything else is v0.2
unchanged):

```
decode → luma (existing) + grid/half-res RGB or Lab (NEW, feeds color candidates)

motion path:   frame diff → [gme compensation] → pool → [motion_cs]
               → noise floor (order: floor AFTER motion_cs)
spatial path:  gradient → [coherence_gate] → pool
new channels:  [signature] [color_rarity] [dog_blob]      (grid resolution)

fusion:        per-channel normalization → [fusion_norm weighting]
               → N-channel blend (generalizes motion_weight="auto" to
                 per-channel energy weights over 2–5 channels)
temporal:      [score_ema] → center bias → allocation (global + floor,
               spread_fraction) → coherent-region block gate
selection:     [select_hysteresis] → topk → _pack_gazing_mask (UNCHANGED)
```

Notes:

- **The only structural change is color**: the pipeline is luma-only
  today. RGB/Lab is kept at grid (24×24) or half resolution only, so the
  added decode/memory cost is negligible. Lab conversion is a fixed 3×3
  matmul + elementwise (mobile-safe); start with RGB and adopt Lab only if
  rarity quality demands it.
- **Ordering rules are part of the spec** (they encode known
  interactions, learned from the v0.2 double_diff negative result):
  quantile noise floor runs *after* `motion_cs` (center-surround first,
  then dead-zone); per-channel normalization runs *before* `score_ema`
  (never EMA raw-scale maps); `fusion_norm` runs after the quantile floor
  (so a single noise spike can't win peak promotion); heavy-tailed
  channels (`color_rarity`, `distinctness`) get a log/sqrt compression
  before min-max and prefer clip-global over per-tubelet normalization.
- New-channel weights fold into the existing blend machinery: the
  2-channel `motion_weight` becomes an N-channel weight vector with an
  `"auto"` mode that generalizes the current energy-ratio rule
  (`w_i ∝ energy_i`, computed pre-normalization, optionally modulated by
  `fusion_norm`).
- `score_ema` and `select_hysteresis` must keep the streaming hook
  viable: within-clip they are loop-free (decay matmul / shifted mask);
  across clips each carries exactly one state tensor.

## 4. Evaluation gates (unchanged infrastructure, fixed decision rules)

| gate | script | rule |
|---|---|---|
| **Semantic recall (PRIMARY)** | `scripts/eval_borissal_semantic.py`, held-out `videos/internvid_eval16/`, ratio 0.25 (spot-check 0.5) | higher is better; differences within ±0.02 (≈ the set's observed std) are ties — resolve ties by paired per-clip comparison, not means |
| Coverage / uniqueness | `scripts/eval_borissal_coverage.py`, frozen V-JEPA2 | **must not degrade BOTH** (v0.2 rule; never coverage alone — scatter bias) |
| Latency | `scripts/borissal_benchmark.py` | combined preset ≤ 25 ms CPU; report per-knob deltas |
| Export | `scripts/export_borissal_check.py` | jit trace + ONNX opset-17 must pass with each adopted knob on |
| Visual | overlay dumps (`borissal_dump_outputs.py`) | qualitative sanity, not a pass/fail gate |
| VideoMAE recon / gist | existing scripts | reference axes only, reported but never optimized toward |

## 5. Combination search protocol

Exhaustive search over 8–12 binary knobs (plus their hyperparameters) is
intractable and unnecessary. Three stages, mirroring v0.2's process:

1. **Solo screening.** Each Tier-1 candidate alone on top of the v0.2
   preset, default hyperparameters, all four gates. Verdicts: KEEP
   (recall up or tied with another axis clearly up), KILL (recall down
   beyond tie range, or both cov/uniq degraded, or export fails), TUNE
   (one retry with a single obvious hyperparameter change, e.g. surround
   radius, K bins, α). Kills are recorded as negative results with the
   mechanism kept as an off-by-default experimental knob (`double_diff`
   precedent).
2. **Greedy forward combination.** Start from the v0.2 preset; add
   survivors in descending solo-recall order; keep an addition only if
   the combined config passes all gates and does not drop recall vs. the
   previous step (ties resolved per-clip). Stop when no remaining
   survivor helps. Two targeted interaction checks regardless of greedy
   order: (a) `fusion_norm` × each adopted new channel (fusion weighting
   is the mechanism most likely to change another knob's verdict), and
   (b) `score_ema` × `select_hysteresis` (redundant stabilizers — adopt
   at most one unless both independently pass the recall condition).
3. **Preset admission.** The winning combination becomes the v0.3 preset
   (a named config, like v0.2's) once it passes: Pareto rule vs. v0.2 on
   (recall, cov/uniq), latency ≤ 25 ms, export check, ascending-index
   test, overlay review. Stability candidates (`score_ema`,
   `select_hysteresis`) are admitted **only if recall is non-degraded**
   (they trade recall of late-appearing objects for stability; the
   description task pays for recall). Tier-2 trigger: implement a Tier-2
   candidate only for an axis where the Tier-1 combination shows no gain
   (e.g. texture bias unmoved → `distinctness`/`border_prior`/`mbd`;
   pan clips still flooded → `gme`).

A sweep runner (`scripts/sweep_borissal_v03.py`) automates stages 1–2:
takes a candidate list, runs gates per config, dumps a results table
(JSON + markdown) under `outputs/borissal/v03_sweep/` (gitignored), and
never decides adoption by itself — verdicts are recorded manually in
`design.md` after reading the table (same discipline as v0.2).

## 6. Deliverables

- `autogaze/models/borissal/modeling_borissal.py` — new knobs, all
  off by default; `configuration_borissal.py` — config fields.
- `scripts/sweep_borissal_v03.py` — stage-1/2 sweep runner.
- Tests (`tests/test_borissal.py` extensions):
  - all-knobs-off output equals v0.2 exactly (tensor-level regression);
  - each knob changes selection on a synthetic clip built to trigger it
    (e.g. panning texture for `motion_cs`, picket-fence + colored blob
    for `coherence_gate`/`color_rarity`);
  - ascending `keep_index` invariant with every knob on;
  - trace/export smoke for the adopted preset;
  - streaming-state equivalence: EMA/hysteresis over a clip equals the
    two-half-clips run with carried state.
- Docs on adoption: `reference.md` §2/§3 (mechanisms + knobs),
  `design.md` (measurements, adopt/reject verdicts, negative results),
  `approach-ko.md` (explainer update).

## 7. Risks and mitigations

- **Held-out eval variance** (per-clip recall spread 0.23–0.45 at n=16):
  the ±0.02 tie rule + paired per-clip comparison in §4/§5 exist for
  this; if verdicts stay ambiguous, extend the eval set before extending
  the candidate bank.
- **Color is a structural change**: contained by keeping color at grid
  resolution and behind knobs; the all-off regression test guarantees
  the luma-only path is untouched.
- **Stability vs. recall tension**: explicit admission condition (§5.3).
- **Knob explosion**: v0.3 adds ~8 knobs to an already knob-rich config.
  Mitigation: the preset is the product; knobs are the search space.
  Anything KILLed stays off-by-default and documented as experimental,
  and Tier-3 items are not implemented at all.
- **Hyperparameter overfitting to the example clip** (v0.2 lesson):
  stage-1/2 verdicts use the held-out 16-clip set, never the example
  clip alone; the example clip is for overlays/debugging.

## 8. Out of scope

- Any learned weights or frozen pretrained models (v0.4+ candidate
  direction, per user: "일단 순수 고전 가자").
- Changes to v1's architecture or training recipe (v1 picks up v0.3
  maps automatically via `input_mode=maps|both`; retraining/re-gating v1
  on the richer bank is a separate follow-up).
- The `Selection` contract, packing, adapters, or downstream interfaces.
- Multi-scale / summary-token output (downstream-contract change,
  explicitly parked in `progress.md` 2026-07-16).
