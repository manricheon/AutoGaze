# Borissal selector — state of the model (2026-07-31)

One page, authoritative. `design.md` is the running lab notebook (132 KB, every
round in order); this file is what you read first: what ships, what it is known
to do, what was tried and rejected, and what is still open. If the two ever
disagree, `design.md` is the record of what happened and this file is stale —
fix this file.

---

## 1. What ships

**`BorissalConfig.v0_7()` — the "Datdol" anchor-novelty preset.** Training-free,
runs on Mac (MPS) and CPU, trace-safe and mobile-safe (no data-dependent shapes).

```python
from autogaze.models.borissal import Borissal, BorissalConfig

sel = Borissal(BorissalConfig.v0_7()).select(video, gazing_ratio=0.25)
# sel.keep_mask -> (B, T_grid*H_grid*W_grid) bool, exact budget
```

Mechanism, in one paragraph: the cube budget splits into an **anchor pool**
(each spatial site is selected once, at the tubelet where its appearance is
best — a temporal downstream encoder does not need static content re-sent every
tubelet), a **novelty pool** (deviation from the clip's temporal-median
canonical state, so it is frame-rate independent unlike consecutive-frame
diffs), and a **residual appearance tier** that absorbs surplus budget at high
ratios. Motion and appearance never compete inside one score. Three pins the
mode depends on: `motion_weight=0.0`, `block_size=1`, bt601 luma. Allocation is
architecture-owned — per-tubelet counts follow from where anchors and novelty
land, and incompatible knobs (spread / hysteresis / keyframe_prior / block gate
/ per_frame_counts) raise at select time rather than silently composing.

Default `signal_grid="cube"` (12x12-native signals, user decision 2026-07-28).

Operating points: **ratio 0.25 is the main battleground, 0.5 the secondary.**

## 2. Evidence ledger — what is actually established

| Claim | How it was tested | Result |
|---|---|---|
| Beats random at 0.25 | holdout-120, 7,200 blind judgments, 3 repeats, single aggregation | 52.5% win [.44,.61], **p = 3e-5** |
| Beats random at 0.5 | same | 39.2% win [.31,.48], **p = .027** |
| Loses less to dense than random | same | vs-dense loss 75.0% vs random 83.3% @0.25; 45.8% vs 53.3% @0.5 |
| Budget is exact at every ratio | `tests/test_borissal_v07.py`, ratio-1.0 keep-all fix (2026-07-26) | green |
| Trace-safe / mobile-safe | export + trace tests across the suite | green |
| Judge harness is not gamed by length | length-bias monitor every round | r = −0.049 (v0.8), −0.016 (v0.9) |

Full test suite: **208 passed** (2026-07-31).

The vs-dense numbers are the honest cost line: at 0.25 the small stack still
loses to dense on ~3 of 4 clips. The selector's job is to lose *less* than a
random budget of the same size, and that is what is demonstrated.

## 3. Negative results — tried, did not change the preset

Kept here so nobody re-runs them.

- **Allocation policy** (2026-07-19): uniform beats content-adaptive; preset
  default flipped to uniform.
- **E5 oracle allocation ceiling** (2026-07-19): negative even with an oracle —
  and the negative extends to multi-scene, ultra-sparse and clip-global judging.
- **Per-tubelet floor** (2026-07-27): inert at deployment ratios.
- **v0.6 saliency-v3.1 knob family** (2026-07-24): no admission.
- **`signal_grid=fine` + `anchor_novelty_lambda=0.75`**: three independent looks.
  Won the dev-60 Stage B at 0.25; lost the A7 both-ratio tie-break to the
  default; then *dominated* the v0.8 holdout observationally (+14.1 pp vs-random
  at 0.5); then, in the v0.9 head-to-head on fresh dev-60b, came out 24/9/27 at
  0.25 (p = .78) and 20/20/20 at 0.5 (p = 1.00). **Not adopted.** Three flips of
  a close call is the signature of a difference below the instrument's
  resolution. The lambda/signal_grid line is closed.

## 4. Experimental — implemented, off by default, NOT validated

**ACR — Adaptive Checkerboard Refresh** (`keyframe_refresh`, `keyframe_keep`,
`keyframe_dynamic`). Generalizes the user's GOP idea: `j` refresh tubelets
(one per window), alternating checkerboard parity so two consecutive refreshes
cover the whole grid, dynamic placement at the window's mean-novelty argmax,
budget clamped so non-refresh tubelets keep their floor, anchors excluded from
refresh tubelets. Fixed GOP is the `keyframe_dynamic=False` special case;
the company agent's `{4,8,16}` maps to `j ∈ {4,2,1}` at 16 frames.

- Default is **`keyframe_refresh=0` (off)**. Nothing in the shipping path changes.
- 6 contract tests in `tests/test_borissal_keyframe.py` (budget invariance,
  checkerboard parity + complementarity, low-budget clamp, determinism,
  partial keep, mode guard).
- **NLL preview only** (proxy, not decisive), dev-60b vs dense:
  `keyframe_refresh=2` dynamic is best at both ratios — +0.241 nats @0.25
  (default +0.309) and +0.086 @0.5 (default +0.109); dynamic > static.
- Captions for all four arms are **already generated**; only judging is left:
  1,680 jobs sit in `outputs/borissal/v09_round/judge_parts_jobs/`
  (acr-duel 600 + vs-dense 1,080). Resume procedure:
  `outputs/borissal/v09_round/RESUME.md`.

Do not quote the NLL preview as evidence of quality — NLL has already been
caught disagreeing with judged outcomes once this project.

## 5. The measurement instrument, and its ceiling

Harness: selector → Qwen 2B captioner (`eval_mllm_attach.py --generate`) → blind
pairwise judging against frozen 8-frame strips (`eval_caption_judge.py`), 5 axes
(objects / actions / scene / temporal / hallucination), A/B order swapped, swap
disagreement counted as a tie. Pre-registration before every round
(`v08-prereg.md`, `v09-prereg.md`); holdout aggregated exactly once.

Two limits, both now measured rather than suspected:

1. **Power.** At n = 60 clips a sign test resolves win rates of roughly ≥67%.
   Knob-level differences land at 47–53%. Two consecutive rounds of "no
   adoption" are the predictable consequence, not bad luck.
2. **The actions axis is nearly blind.** In the v0.9 round, 46/60 (@0.25) and
   49/60 (@0.5) clips were ties on actions — the 2B captioner barely
   differentiates movement at all. That is the axis the downstream goal (action
   recognition, risk detection) cares about most.

Conclusion carried forward: **further caption-level knob search cannot promote
anything.** The next real move is either a mechanism large enough to clear the
resolution floor (ACR is the candidate on the table) or a higher-power
instrument — a linear probe on V-JEPA2.1-L features against labeled action data,
which measures the target task directly and extracts far more signal per clip.

## 6. Where things live

```
autogaze/models/borissal/
  configuration_borissal.py     presets v0_2..v0_7, all knobs + ACR
  modeling_borissal.py          selection modes incl. anchor_novelty + ACR
  signals_v03.py                appearance/motion signal stack
tests/test_borissal_v07.py      preset contracts        (+ _keyframe.py for ACR)
scripts/
  eval_mllm_attach.py           caption generation (--generate), spec overrides
  eval_caption_judge.py         judge harness + validation suite
  make_evalset_manifest.py      dev-60 / holdout-120 + frozen frames
  make_evalset_dev60b.py        dev-60b (v0.9 set, seed 20260731)
  make_qual_report.py           overlay + caption + verdict HTML report
docs/borissal/
  STATUS.md                     this file
  design.md                     full chronological record
  tuning-guide.md               how to reproduce the tuning loop elsewhere
  v08-prereg.md / v09-prereg.md pre-registrations
  evalset_manifest.json / evalset_dev60b.json
outputs/borissal/               all round artifacts (gitignored)
```

Eval sets are disjoint by construction: `eval16` is smoke-only (never for
decisions), dev-60 and holdout-120 are burned, dev-60b is the current fresh set.

Reproduce a qualitative snapshot of what the shipping preset selects:

```bash
uv run python scripts/make_qual_report.py \
  --captions outputs/borissal/v09_round/gen/captions.json \
  --manifest docs/borissal/evalset_dev60b.json \
  --clips <comma-separated> --specs "v0.7" --ratios 0.25,0.5 \
  --out outputs/borissal/v09_round/state_qual_report.html
```

## 7. Open items

1. **Instrument upgrade** (recommended next): V-JEPA2.1-L features + linear
   probe on labeled action data, so the selector is scored on the task it is
   meant to serve. Mac-local, no paid API.
2. **ACR judging**: 240-judgment lite version (`keyframe_refresh=2` dynamic vs
   v0.7 default, both ratios) is one session's work and uses captions that
   already exist; the full 1,680 needs a fresh sub-agent budget.
3. **CUDA track** (task #19, optional): FA2-vs-SDPA diagnostic and fast sweeps
   on free CUDA (Kaggle / Modal credits).
4. **Rename**: `borissal` → a real name. Recommendation on the table is *Skim*
   (`skim.select(video, skim_ratio=...)`), with *Glimpse* second. Must happen
   between rounds, never mid-round, so prereg documents stay comparable.
