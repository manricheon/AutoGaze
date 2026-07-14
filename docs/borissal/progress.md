# Borissal — progress / handoff log

Read this file first when resuming work on the patch-selector line in a new
session. See `design.md` in this directory for the full design rationale.

Versioning: the current non-learned selector is **Borissal v0**
(`autogaze.models.borissal.MODEL_TAG == "borissal-v0"`); the learned
successor (Phase 2) will be **Borissal v1**. Earlier entries below predate
this naming and call v0 "Borissal-signal" — left as written, since this file
is a historical log, not a living reference (see `design.md`/`reference.md`
for the current name).

---

## 2026-07-14 — Phase 1 (Borissal-signal) implemented and verified

**Branch**: `feat/selector` at the time (later renamed to `feat/borissal` the
same day — see next entry; `feat/borissal` is the durable target branch name
for all of this patch-selector work going forward).

**Status**: Phase 1 complete. Non-learned, feed-forward, grid_thw-native
saliency selector implemented, environment set up on Mac via uv, and
qualitative verification passed.

**What was done:**
- `pyproject.toml`: dropped `flash_attn` from base deps (moved to new
  `[project.optional-dependencies].cuda`), bumped `transformers` to
  `>=5.5,<6`, `requires-python` to `>=3.11`.
- `uv venv --python 3.11 && uv pip install -e .` — installed cleanly on
  macOS. Resolved: `torch==2.13.0` (CPU/MPS wheel), `transformers==5.13.1`.
  `uv pip install -e '.[dev]'` for pytest.
- New package `autogaze/models/borissal/`:
  `configuration_borissal.py`, `modeling_borissal.py` (`Borissal`,
  `Selection`), `adapters.py` (`to_vjepa2`, `to_autogaze_gazing_info`),
  `video_io.py` (PyAV decode, no transformers video processor).
- `scripts/eval_borissal_qualitative.py` — standalone overlay renderer.
- `tests/test_borissal.py` — 7 tests, all passing.
- `docs/borissal/{design.md,progress.md}` (this pair of files).
- No existing repo files touched other than `pyproject.toml`. `train.py`,
  `trainer.py`, legacy `models/autogaze/`, `tasks/` untouched.

**Verified (commands + results):**
```
uv run python -c "import torch, transformers; print(transformers.__version__, torch.backends.mps.is_available())"
# -> 5.13.1 True

uv run pytest tests/test_borissal.py -v
# -> 7 passed

uv run python scripts/eval_borissal_qualitative.py \
  --video assets/example_input.mp4 --gazing-ratio 0.5 --motion-weight 0.5 \
  --tubelet-size 2 --scale 384 --patch 16 --out /tmp/borissal_eval/sal_r50_m50.png
# -> grid_thw = [8, 24, 24]; num_keep = 2304/4608; per_frame_keep uniform 288 each
```

**Qualitative result** (visual inspection of the overlay PNGs, `gazing_ratio`
swept to 0.3-0.5, `motion_weight` swept 0.0/0.5/1.0 on
`assets/example_input.mp4`, a screen-recording-style clip with slides/diagram
+ scrolling subtitle bar):
- Selected patches consistently concentrate on informative regions (diagram
  lines, subtitle text, speaker silhouette edges) and avoid flat/background
  regions (blurred stage background, blank slide whitespace) — sanity check
  passed for both motion-only (`motion_weight=1.0`) and spatial-only
  (`motion_weight=0.0`) extremes.
- The two extremes select visibly different (not identical) patch sets in
  the diagram region across frames (where node highlight colors change frame
  to frame), confirming the motion and spatial terms are contributing
  distinct signal, not one dominating trivially. Also confirmed
  programmatically in `test_motion_weight_changes_selection` with a
  synthetic video (isolated moving block vs. static high-contrast edge).
- This video's content has text/graphics that are both edge-rich *and*
  changing, so motion-only and spatial-only overlaps are large — a more
  motion-distinct test video (e.g. static background + one moving object)
  would show starker separation; not needed for Phase 1 sign-off given the
  synthetic unit test already isolates this.

**Known trade-off accepted (see design.md):** bumping `transformers` to
`>=5.5,<6` repo-wide likely breaks importing the legacy AutoGaze custom
modeling classes (`autogaze/models/autogaze/*`, `vision_encoders/siglip/*`,
`tasks/video_mae_reconstruction/*`) which target `~=4.51`. Not verified
either way in this session (no reason to import them for Phase 1). If the
legacy NTP/GRPO training path needs to run again, it will need its own
environment (pin transformers back to `~=4.51` there) or a migration pass —
out of scope for Phase 1.

**Open / not yet done:**
- Checkpoints: none required for Phase 1 (fully non-learned). No weights
  downloaded this session. `weights/` dir exists in the repo (untracked) but
  is empty/unused by Phase 1.
- V-JEPA2.1L exact repo id / checkpoint and its native token flatten order:
  **not yet confirmed** — needed before Phase 3 (and before treating
  `adapters.to_vjepa2` as more than a documented stub). Action item carried
  from the design doc.
- Optional dense-vs-sparse reconstruction sanity (`--recon` flag mentioned in
  planning) was **not implemented** — Phase 1 sign-off criteria was
  qualitative-only per explicit direction, so this was skipped as
  unnecessary scope. Revisit only if a quantitative pre-Phase-2 baseline is
  wanted.
- Committed to git as of this entry (commit `86c3a69`, on what was then
  `feat/selector`), local only, not pushed.

**Next up (Phase 2, not started):** learned selector (TSM or conv3d backbone
+ scoring head + straight-through top-k), using Borissal-signal's
motion/spatial maps as candidate input features or initialization. See
design.md's "Open items for Phase 2/3".

---

## 2026-07-14 (same day) — ratio 0.5/0.25 examples, real latency benchmark, mobile readiness gate

Pre-Phase-2 checkpoint requested by the user: show 0.5/0.25 ratio examples,
measure actual speed, and review operator/burden concerns for the eventual
mobile target — all **before** starting Phase 2. See design.md's new
"Mobile readiness review" section for the full write-up; this entry is the
session log / numbers.

**What was done:**
- Regenerated `outputs/borissal/{r50_m50_grad_uniform,r25_m50_grad_uniform}/`
  via the existing `scripts/borissal_dump_outputs.py` (no script changes
  needed). `num_keep` scaled exactly with ratio: 2304 (r=0.5) vs 1152
  (r=0.25) out of L=4608 — precisely half, as expected. Overlay comparison
  confirms r=0.25 stays concentrated on the same high-saliency regions
  (subtitle bar, diagram edges) just more sparsely.
- New `scripts/borissal_benchmark.py`: measures `Borissal.select()` latency
  (B=1,T=16,384x384 clip, no video decode) across
  device×gazing_ratio×per_frame_allocation, plus a `torch.profiler` op
  breakdown. Results land in `outputs/borissal/benchmark/` (gitignored):
  `latency_cpu.json`, `latency_mps.json`, `profiler_cpu.txt`.
  - CPU: 6.7-11.6ms mean per clip (86-150 clips/sec).
  - MPS: 150-210ms mean (4.8-6.5 clips/sec) — **slower than CPU**, a known
    MPS per-op-dispatch-overhead quirk for small tensors, not representative
    of real mobile hardware.
  - Profiler: sort-family ops (`aten::sort` + `aten::topk` combined) are now
    under 6% of self-CPU-time; dominant cost is ordinary
    elementwise/reduction/pooling arithmetic over the raw pixel grid
    (`aten::mean`, `aten::sum`, `aten::avg_pool2d`, etc.) — all
    universally mobile-supported ops.
- `autogaze/models/borissal/modeling_borissal.py` — two fixes, both
  behavior-preserving for eager execution (all 7 tests still pass):
  1. Replaced the selector's own double-`argsort` rank-based top-k with
     `torch.topk(k_max) + scatter_` (`torch.topk` has first-class mobile
     support — TFLite `TopKV2`, CoreML `top_k` — general `sort`/`argsort`
     does not). Verified semantically equivalent (both select exactly the
     top-k highest-scoring patches per tubelet).
  2. Fixed a **real `torch.jit.trace` failure**: `B, T, C, H, W =
     video.shape` unpacking didn't yield plain Python ints under tracing,
     breaking `round()` calls downstream. Fixed with an explicit
     `int(x) for x in video.shape` cast — one line, no eager-mode behavior
     change.
- Empirically traced the full model (not just reasoned about it) and found
  a **real correctness trap**: `per_frame_allocation="proportional"` bakes
  its data-dependent per-tubelet split into the traced graph, so re-running
  the same traced graph on different video content of the same shape gives
  a silently wrong, stale result (verified: `num_keep=2293` traced vs.
  correct `2304` fresh eager, on new random content). `"uniform"` allocation
  has no such issue (config/shape-derived, not data-dependent) — verified
  correct across repeated trials with fresh content.
- `docs/borissal/design.md`: added "## Mobile readiness review" section
  with the full latency table, profiler summary, the two fixes, the trace
  findings, and concrete constraints carried into Phase 2 (prefer
  `topk`-style ops; treat data-independent allocation as the mobile-export
  -safe default; export one artifact per fixed input shape; real on-device
  latency still needs measuring once mobile export tooling exists).

**Verified (commands):**
```
uv run pytest tests/test_borissal.py -q                     # 7 passed (before and after both fixes)
uv run python scripts/borissal_dump_outputs.py --video assets/example_input.mp4 --gazing-ratio 0.25 --motion-weight 0.5
uv run python scripts/borissal_benchmark.py                 # latency + profiler, see numbers above
git status --short                                           # outputs/ still fully untracked
```

**Not done / explicitly out of scope this round:** no ONNX/CoreML export
attempt (used `torch.jit.trace` only — zero extra dependencies, already
installed with torch); no on-device (phone) latency measurement; the
"proportional" trace-correctness trap was found and documented but NOT
fixed (fixing it would mean either forcing "uniform"-only for any traced/
exported path, or adding a genuinely dynamic export mechanism — deferred as
a Phase 2/3 design decision, not a Phase 1 blocker since Phase 1 itself is
never traced/exported, only Python-called).

**Next up (Phase 2, unchanged):** learned selector (TSM or conv3d backbone +
scoring head + straight-through top-k). New mobile-readiness constraints to
carry in: prefer topk-style ops, keep allocation policy mobile-export-safe
(or explicitly scope data-dependent allocation out of any traced/exported
path), export against a fixed input shape.

---

## 2026-07-14 (same day) — Branch renamed to feat/borissal; stage-by-stage outputs/ dump

**Branch**: `feat/selector` renamed in place to **`feat/borissal`**
(`git branch -m feat/selector feat/borissal`) — this is now the confirmed
durable branch name for the whole patch-selector line (Phases 1-3), per
direct instruction. Use `feat/borissal` going forward; don't recreate
`feat/selector`.

**What was done:**
- `autogaze/models/borissal/modeling_borissal.py`: added
  `Borissal.select_with_intermediates(...)` (returns `(Selection,
  intermediates)` where `intermediates = {motion_norm, spatial_norm, score}`,
  each `(B, T_grid, H_grid, W_grid)`, pre-top-k). Implemented via a shared
  private `_select_impl(..., want_intermediates)` so `select()`'s existing
  signature/behavior/tests are unchanged (verified: same 7 tests still pass,
  plus a manual check that `select()` and `select_with_intermediates()` agree
  on `grid_thw`/`keep_index`).
- New `autogaze/models/borissal/viz.py`: shared rendering helpers extracted
  from the eval script — `render_frame_strip` (all-frames thumbnail strip),
  `render_overlay` (moved as-is from `eval_borissal_qualitative.py`),
  `render_heatmap_grid` (per-tubelet heatmap + colorbar, used for
  motion/spatial/score), `render_allocation_bar` (per-tubelet kept-count bar
  chart). `tubelet_title(t, tubelet_size)` labels each tubelet column with
  its covered frame range, e.g. "t=3 (frames 6-7)".
- `scripts/eval_borissal_qualitative.py`: refactored to import
  `render_overlay` from `viz.py` instead of defining it locally (behavior
  unchanged, re-ran and confirmed identical output).
- New `scripts/borissal_dump_outputs.py`: runs the full pipeline and writes,
  per run, to `outputs/borissal/<run_name>/` (default `run_name` derived from
  config, e.g. `r50_m50_grad_uniform`):
  `00_input_frames.png` (all `num_frames` raw decoded frames in one strip —
  not just one representative per tubelet), `01_motion.png`, `02_spatial.png`,
  `03_score.png` (per-tubelet heatmaps, `T_grid`=8 columns for the default
  config), `04_overlay.png` (final selection), `05_allocation.png`
  (`per_frame_keep` bar chart — the direct visual answer to "is the
  per-frame/per-tubelet selection-count allocation policy visible"), and
  `summary.json` (full config + `grid_thw`/`num_keep`/`per_frame_keep`).

**Clarification (user question, direct answer for the record):** the
per-frame (per-tubelet) selection-count allocation policy
(`BorissalConfig.per_frame_allocation`: `uniform` | `proportional`) was
**already implemented** in Phase 1 (see the previous entry / design.md §4) —
this session only added the `05_allocation.png` visualization of its effect;
no allocation logic changed.

**Verified (commands + results):**
```
uv run python scripts/borissal_dump_outputs.py --video assets/example_input.mp4 --gazing-ratio 0.5 --motion-weight 0.5
# -> outputs/borissal/r50_m50_grad_uniform/  (grid_thw=[8,24,24], num_keep=2304/4608, per_frame_keep all 288)

uv run python scripts/borissal_dump_outputs.py --video assets/example_input.mp4 --gazing-ratio 0.3 --motion-weight 0.0
# -> outputs/borissal/r30_m0_grad_uniform/   (num_keep=1384/4608, per_frame_keep all 173)

uv run python scripts/borissal_dump_outputs.py --video assets/example_input.mp4 --gazing-ratio 0.3 --motion-weight 1.0
# -> outputs/borissal/r30_m100_grad_uniform/ (num_keep=1384/4608, per_frame_keep all 173)

uv run python scripts/borissal_dump_outputs.py --video assets/example_input.mp4 --gazing-ratio 0.5 --motion-weight 0.5 --per-frame-allocation proportional
# -> outputs/borissal/r50_m50_grad_proportional/ (per_frame_keep = [318,253,304,317,256,245,279,332] -- visibly non-uniform)

uv run pytest tests/test_borissal.py -q
# -> 7 passed (unchanged)

git status --short   # outputs/ does not appear at all (gitignore:2 "outputs/")
git check-ignore -v outputs/borissal/r50_m50_grad_uniform/summary.json
# -> confirms .gitignore:2:outputs/ matches
```

**Qualitative confirmation:** opened `00_input_frames.png` (all 16 raw
frames), `01_motion.png`/`03_score.png` (motion heatmap lights up on the
scrolling subtitle bar and the diagram region, matching the earlier overlay
finding), and the two `05_allocation.png` bar charts side by side — `uniform`
shows a flat 288-per-tubelet bar chart, `proportional` shows visibly varying
bars (245-332) that track each tubelet's saliency energy. This is the
intended visual proof that the allocation policy has a real, visible effect.

**Not done / explicitly out of scope this round:** no reconstruction/encoder
sanity added (still qualitative-only per standing direction); V-JEPA2.1L repo
id still unconfirmed (unchanged open item from the previous entry).

**Repro note for a fresh session:** `outputs/` is gitignored and this
session's generated runs live only on this machine — if you need them again,
just re-run the four commands above (they're deterministic given the same
`assets/example_input.mp4` and config).

---

## 2026-07-14 (same day) — Reference doc added; content-adaptive auto-tuning scoped then deferred

Follow-up to the mobile-readiness round: the user clarified that Borissal's
selected patches feed a **description task** downstream (patch → encoder →
LLM → description), and asked for (a) a polished, non-verbose reference doc
on the selector's operations/policies framed around that, and (b) an
"auto"-tuning option for the various knobs, prioritized speed > exact
`gazing_ratio` compliance > fit-for-purpose selection.

**What was done:**
- New `docs/borissal/reference.md` — a concise "what and why" reference,
  deliberately distinct from this file's dev-log and `design.md`'s
  decision-rationale style. Sections: (1) **Why saliency** — the core
  motivation, framed around the description-task pipeline and a
  video-codec analogy (codecs encode what *changed*/is salient rather than
  every pixel; Borissal approximates that directly on decoded frames
  instead of using real motion vectors), plus why feed-forward/single-scale/
  top-k was chosen over AutoGaze's autoregressive/multi-scale approach; (2)
  algorithm walkthrough; (3) config-knob table with defaults + rationale;
  (4) `Selection` output schema; (5) a pointer-only performance/mobile
  summary (no duplication of design.md's numbers); (6) a "not yet built"
  note for deferred auto-tuning.
- **Scoped, then explicitly deferred**: investigated auto-tuning the
  selector's knobs. Finding: `spatial_op="grad"`, `pooling="avg"`, and
  `per_frame_allocation="uniform"` are *already* the fast+exact-ratio-safe
  preset (confirmed against the current `BorissalConfig` defaults), so the
  only knob whose optimal value genuinely varies per-clip is `motion_weight`
  (fixed at 0.5). Designed a concrete, cheap, non-learned approach —
  `motion_weight="auto"` computing `motion_energy / (motion_energy +
  spatial_energy)` from the already-computed `motion_p`/`spatial_p` tensors
  (near-zero added cost) — but the user then said to **hold off**: "추후 더
  고도화해보자. 학습 기반 모델도 추가로 있을 거니까" (revisit alongside/after
  the learned selector, Phase 2). **No code was changed for this** — it's a
  deliberate deferral, not an oversight; don't re-propose the plain
  energy-ratio version without checking whether Phase 2's learned selector
  changes what "auto" should mean.
- Follow-up consistency check (user asked to read `reference.md` against
  the rest of the docs): found and fixed two real staleness issues in
  `design.md` — (1) its own "Saliency algorithm" section still described
  the pre-swap `argsort`-twice top-k, contradicting its own later "Mobile
  readiness review" section (which correctly describes the `torch.topk`
  swap) and the actual code; fixed to describe `torch.topk`, pointing to
  the Mobile readiness review section for detail. (2) its "Files" list
  hadn't been updated since Phase 1's initial commit — added `viz.py`,
  `scripts/borissal_dump_outputs.py`, `scripts/borissal_benchmark.py`, and
  `docs/borissal/reference.md`. `reference.md` itself was accurate — it was
  `design.md` that had drifted.

**Verified:** re-read both `design.md` sections after the edit to confirm
they no longer contradict each other; no code/tests touched this round, so
no need to re-run `pytest`.

**Not done / explicitly out of scope this round:** `motion_weight="auto"`
implementation itself (deliberately deferred, see above); any other
auto-tuning knob.

**Next up (Phase 2, unchanged):** learned selector. When it lands, revisit
whether/how to auto-tune `motion_weight` (or fold it into the learned
scoring head entirely, making the question moot) — this is an open design
question for Phase 2, not a Phase 1 task.

---

## 2026-07-14 (same day) — Standalone core, device auto-select, canonical keep-index interface verified

User asked three things: (1) confirm the model is mostly/entirely standard
PyTorch ops and will "just work" on a Linux/CUDA box, (2) add an option to
auto-pick the fastest device per machine, (3) make the model implementation
standalone-portable. Also, mid-round, the user pasted back a canonical
selector-output interface spec from another agent's investigation of the
real downstream pipeline (Qwen-VL-style sparse encoder over V-JEPA2) and
asked to check Borissal's output against it.

**(1) Answered (no code needed), verified by grep, not just recalled:**
confirmed the entire `autogaze/models/borissal/` package had exactly one
`autogaze.*` import (`autogaze.utils.get_gazing_pos_from_gazing_mask`),
zero custom CUDA/C++ extensions, zero `torch.jit.script`, zero hardcoded
`"cpu"`/`"mps"`/`"cuda"` branches — every op used (mean/abs/sub/pad/sqrt/
avg_pool2d/max_pool2d/conv2d/topk/arange/scatter_/clamp/floor/argsort) is
standard `aten` with existing CUDA kernels. Conclusion: runs on Linux/CUDA
with zero code changes, just `.to("cuda")`.

**(2) Device auto-select** — asked the user to pick a mechanism (static
priority vs. live self-benchmark); they chose static. New
`autogaze/models/borissal/device.py`:
`resolve_device(mode="auto")` → `cuda` if available, else `cpu` (**not**
`mps` — this session's own benchmark data showed mps slower than cpu for
Borissal's small-tensor workload; `mps` is still usable via
`mode="mps"`), and `available_devices()` (enumeration, for
`borissal_benchmark.py`'s "test every device" use case, which is a
different job from "pick one" and was kept separate rather than forced
into `resolve_device`). Replaced three near-identical local
`resolve_device`/`available_devices` copies in
`eval_borissal_qualitative.py`, `borissal_dump_outputs.py`, and
`borissal_benchmark.py` with imports from the new module. Re-ran all
three scripts to confirm identical behavior.

**(3) Standalone** — inlined the one remaining `autogaze.utils` import
(`get_gazing_pos_from_gazing_mask`, verbatim, ported as `_pack_gazing_mask`
inside `modeling_borissal.py`). Result: `configuration_borissal.py` +
`modeling_borissal.py` + `adapters.py` + `device.py` now depend on
**`torch` only** — confirmed via `grep` (zero `autogaze.*` imports
remaining). `video_io.py`/`viz.py` stay as-is (already had no
cross-dependency on the core files; they're this repo's dev-tooling, not
needed by the model). `docs/borissal/reference.md` gained a "Standalone"
section (§6) documenting exactly this.

**(Canonical downstream interface, added mid-round)** — user's pasted spec:
selector output must be a flat, per-video, ascending list of kept patch
indices, `idx = t*N + n` (`n` = row-major within-frame), sorted by
(frame, row, col), because the real downstream encoder's mask-gather +
RoPE position recovery depends on that exact order. **Checked this against
`Selection.keep_index` before writing any code** (a quick in-memory
verification script, batch of 4, both `uniform` and `proportional`
allocation) and found it **already matches exactly** — Borissal's native
flatten order (`t*(H_grid*W_grid)+h*W_grid+w`) *is* `t*N+n`, and the
packer's stable sort already preserves ascending order among kept entries.
No algorithm change was needed. What *was* added: a new
`adapters.to_canonical_keep_indices(selection) -> list[Tensor]` (per-video
1-D ascending tensor, `-1` padding stripped — the exact shape a
`keep_indices_per_video`-style handoff expects), and two new tests
(`test_keep_index_is_ascending_per_row`, `test_to_canonical_keep_indices`)
to lock this contract down so a future top-k change can't silently break
it. Steps 3-6 of the user's pasted spec (processor/model/encoder/LLM
internals) are explicitly out of scope for Borissal — noted, not
implemented.

**Verified (commands):**
```
uv run pytest tests/test_borissal.py -v          # 9 passed (7 previous + 2 new)
grep -rln "^from autogaze\|^import autogaze" autogaze/models/borissal/{configuration_borissal,modeling_borissal,adapters,device}.py
                                                   # -> no matches (standalone confirmed)
uv run python scripts/eval_borissal_qualitative.py --video assets/example_input.mp4 ...
uv run python scripts/borissal_dump_outputs.py --video assets/example_input.mp4 ...
uv run python scripts/borissal_benchmark.py       # all three re-verified after the device.py refactor
```
Also spot-checked `to_canonical_keep_indices` against the real
`assets/example_input.mp4` clip (not just synthetic tensors): ascending,
correct length, matches `num_keep`.

**Docs touched:** `design.md` — new "Canonical downstream interface"
section, a new "Key design decisions" row for standalone-portability, and
fixed several now-inaccurate "reused legacy `autogaze/utils.py`" phrasings
left over from before the inlining (found while updating, not left for
later). `reference.md` — new "Canonical downstream interface" (§5) and
"Standalone" (§6) sections; renumbered §6→§8 ("Not yet built") and fixed
its internal cross-reference.

**Not done / explicitly out of scope:** the actual processor/model/
encoder/LLM integration code (steps 3-6 of the user's pasted spec) —
Borissal only needed to satisfy the *selector's* half of the contract,
confirmed done. No ONNX/CoreML export attempted this round either (still
just `torch.jit.trace`, from the previous round).

**Next up (Phase 2, unchanged):** learned selector — carries forward all
prior mobile-readiness constraints, plus: keep the canonical ascending
`idx=t*N+n` output contract intact regardless of how the scoring/top-k
mechanism changes (there's now a test guarding it).

---

## 2026-07-14 (same day) — motion_weight="auto" implemented, before Phase 2/3

User decided to complete the previously-deferred `motion_weight` auto-tuning
now, before moving on to Phase 2/3, and asked for a before/after comparison
shown separately.

**What was done:**
- `configuration_borissal.py`: `motion_weight: Union[float, Literal["auto"]]`
  (default unchanged, `0.5`).
- `modeling_borissal.py`: when `motion_weight_setting == "auto"`,
  `w = motion_energy / (motion_energy + spatial_energy + eps)` computed from
  `motion_p`/`spatial_p` **before** `_minmax_norm` (using the post-normalization
  maps would erase the absolute-magnitude signal needed here — min-max
  normalization always rescales to `[0,1]` regardless of original energy).
  Per-video (`(B,1,1,1)`, broadcasts against the `(B,T_grid,H_grid,W_grid)`
  score maps). Exposed via a new `motion_weight_used` `(B,)` field in the
  `select_with_intermediates` intermediates dict — deliberately *not* added
  to `Selection` itself, to avoid touching the output contract that was just
  locked down with tests/the canonical-interface work last round.
- `scripts/{borissal_dump_outputs.py,eval_borissal_qualitative.py}`:
  `--motion-weight` now accepts the literal string `"auto"` (custom argparse
  type; `float()` otherwise) and both scripts print/record the *resolved*
  value alongside the requested setting (`borissal_dump_outputs.py`'s
  `summary.json` gained a `motion_weight_used` field; its combined-score
  heatmap title shows both).
- `tests/test_borissal.py`: two new tests —
  `test_motion_weight_auto_adapts_to_content` (static synthetic clip →
  `w<0.2`, high-motion synthetic clip → `w>0.5`, motion > static) and
  `test_motion_weight_fixed_unaffected_by_auto_support` (explicit float
  behavior unchanged by "auto" existing). 11/11 tests pass (9 previous + 2
  new).

**Before/after comparison (shown to the user directly, also saved to
`outputs/borissal/`, gitignored):**
- Real clip (`assets/example_input.mp4`): fixed `motion_weight=0.5`
  (`r50_m50_grad_uniform/`) vs. `motion_weight="auto"` → resolved `w≈0.354`
  (`r50_mauto_grad_uniform/`). Overlays are visually similar for this
  particular clip — expected, since its content (on-screen text + diagram)
  is both edge-rich *and* changing, so the auto value lands close to but
  below the neutral 0.5 rather than at an extreme.
- Synthetic contrast pair (ad hoc, not a permanent script — reused
  `render_overlay` directly): a fully static clip (identical frame repeated)
  resolves to **`w=0.000`** and the overlay shows selection concentrated
  purely on the static block's edge; a clip with a small bright block
  sweeping across low-texture background resolves to **`w≈0.697`** and the
  overlay visibly tracks the block's per-tubelet position. Saved to
  `outputs/borissal/{auto_demo_static,auto_demo_motion}/04_overlay.png`.
  This pair demonstrates the actual adaptive range much more clearly than
  the real clip does.

**Verified (commands):**
```
uv run pytest tests/test_borissal.py -v          # 11 passed
uv run python scripts/eval_borissal_qualitative.py --video assets/example_input.mp4 --motion-weight auto --out ...
uv run python scripts/borissal_dump_outputs.py --video assets/example_input.mp4 --motion-weight auto
                                                   # both scripts print "motion_weight = auto (resolved = 0.354)"
```

**Docs touched:** `reference.md` §3 (motion_weight row + explanation) and §8
(moved from "not yet built" to "implemented, other knobs intentionally not
auto-tuned"); `design.md` gained a "Content-adaptive `motion_weight`"
section (algorithm, rationale for using pre-normalization energy, where the
resolved value is exposed, verification numbers) and a small step-6 update
in the algorithm walkthrough.

**Not done / explicitly out of scope:** auto-tuning any other knob
(`spatial_op`/`pooling`/`per_frame_allocation`) — each already defaults to
the fast+ratio-safe choice regardless of content, so there was nothing to
gain; revisit only if a real future need appears.

**Aside (not a code decision):** user asked whether to start using the
"Fable" model now or later. Recommended waiting for a natural scope
boundary (e.g. the start of Phase 2) rather than switching mid-task, mainly
so this session's accumulated context isn't lost — but this is the user's
call, not something resolved here.

**Next up (Phase 2, unchanged):** learned selector. `motion_weight="auto"`
being done removes it from Phase 2's "carry forward" list — Phase 2 can
decide independently whether its scoring head needs anything like it at
all (a learned head may fold this decision in implicitly).

---

## 2026-07-14 (same day) — Phase 2+3 implemented: Borissal v1 + V-JEPA2 SSL training

Combined plan approved with 10 explicit user requirements (transformers
==5.5.0 pin, differentiability first, random gazing_ratio during training,
peak-memory tracking, DDP support, composable losses, self-inference/viz/
profiling for v0+v1, minimal-data local training check, a training-method
doc, and expanded v0-leverage options). All implemented this session.
Canonical training doc: `docs/borissal/training.md`. Engineering summary:
design.md's "Borissal v1 + SSL training" section.

**New files:** `modeling_borissal_v1.py` (BorissalV1: TSM-style 2D CNN,
~114K params, `input_mode: maps|pixels|both`, `residual_scoring`; same
Selection contract + canonical ascending keep_index as v0),
`vjepa2_sparse.py` (functional sparse-encoder forward over the stock HF
encoder + frozen `VJEPA2Teacher` wrapper — the only core file importing
transformers), `losses.py` (predictor-coverage / dense-sparse-match /
entropy / v0-distill, weight-composable), `data.py` (VideoFolderDataset
reusing video_io), `scripts/train_borissal_v1.py` (DDP-under-torchrun or
single-device, per-batch ratio sampling rank-synced, jsonl logging incl.
peak memory), `tests/test_borissal_v1.py` (9 tests),
`docs/borissal/training.md`. Extended: dump/benchmark scripts got
`--model v0|v1 --checkpoint` + peak-memory column; `BorissalV1Config`
added; `weights/` added to .gitignore (was merely untracked).

**Verified (highlights, all on this Mac):**
- transformers==5.5.0 reinstalled; vjepa2 sparse primitives re-verified
  identical to the 5.13.1 exploration; all pre-existing tests green.
- **Two real bugs found empirically during smoke** (would have silently
  broken training): (1) operator precedence made the Gumbel term
  constant-NaN → topk ignored scores entirely (selection frozen); (2) raw
  softmax probs (~1/N_pf) starved the ST gradient ~1000x. Both fixed +
  regression-tested. 20/20 tests pass.
- Plumbing proof: v0-distill-only smoke run collapsed loss 2.26 → 0.03 in
  30 steps with healthy gradients.
- **V-JEPA 2.1 checkpoints do not load into native transformers vjepa2**
  (confirmed via load report on `davevanveen/vjepa2.1-vitb-fpc64-384`:
  dual patch embeds / modality embeds / distillation norms / predictor
  proj 1664). True-2.1 teachers (the team's torch.hub vjepa2.1l/b) need a
  small adapter behind the teacher wrapper — interface already designed
  for it; grid convention identical so nothing else changes.
- **Real-teacher local check** (user req 8): official
  `facebook/vjepa2-vitl-fpc64-256` (326M, patch16/tubelet2 asserted,
  native load) on MPS, 30 steps, batch 1, scale 256, duplicated example
  clips: ~8s/step, peak 1.4GB, gradients strong (norm 24–244 — vs ~0 with
  a random tiny teacher, showing the real feature space genuinely
  discriminates selections). **Honest caveat:** predictor-coverage loss
  did NOT visibly decrease in 30 steps on this degenerate single-clip
  dataset (8.18→8.28 fluctuation) — expected at this scale; the
  optimization mechanism itself is proven by the v0-distill collapse
  above. Real learning-curve claims need the Linux/CUDA run on
  AutoGaze-Training-Data.
- Before/after overlays (`outputs/borissal/v1_real_{before,after}/`):
  the 30-step checkpoint visibly changes selection behavior vs random
  init; checkpoint save→reload→inference path works
  (`--model v1 --checkpoint`).
- v0 vs v1 benchmark (`--model both`): v1 CPU ~17-18ms/clip (~2.7x v0,
  includes internal v0 signal computation in `both` mode), peak-mem column
  now reported.

**Not done / open:** torch.hub V-JEPA2.1 teacher adapter (three methods,
when scale training starts); loss-combination/input_mode experiment matrix
at scale (training.md §3); predictor fine-tuning option.

**Next up:** push, then Linux/CUDA scale training on AutoGaze-Training-Data
(torchrun recipe in training.md §6), teacher = team's vjepa2.1l via adapter
or HF ViT-L as-is.

---

## 2026-07-14 (same day) — V-JEPA 2.1 torch.hub teacher + oracle-reference coverage loss

User asked to fetch V-JEPA 2.1 from the official torch.hub and to check
whether its predictor is usable. Both done; one upstream bug and one design
adaptation along the way.

**Findings:**
- Official `facebookresearch/vjepa2` torch.hub exposes 2.1 entrypoints
  (`vjepa2_1_vit_base_384`, `vjepa2_1_vit_large_384`, ...), all returning
  `(encoder, predictor)` — **predictor included** (depth 12 for B/L),
  img384/patch16/tubelet2 (grid matches ours).
- **Upstream bug**: the repo's main hardcodes `VJEPA_BASE_URL =
  "http://localhost:8300"` (dev leftover; the real
  `https://dl.fbaipublicfiles.com/vjepa2` is commented out). Workaround:
  pre-download the .pt into `~/.cache/torch/hub/checkpoints/` (done for
  `vjepa2_1_vitb_dist_vitG_384.pt`, 1.66GB).
- The original encoder supports sparse natively (`masks=[keep_index]`) and
  its token order is the same canonical t-major — verified shapes:
  dense (1,4608,768), sparse (1,1382,768) at ratio 0.3/384.
- **2.1's predictor projects to the DISTILLATION-teacher space (1664-d,
  ViT-gigantic; `return_all_tokens=True` → (x_pred, x_context))**, not the
  encoder's own 768-d. So plain "predict vs own dense features" coverage
  can't be wired. User asked if 2.1-L/B alone (no gigantic) is workable →
  **oracle-reference coverage** designed and implemented: run the SAME
  predictor head twice — reference pass (no-grad) with context = ALL dense
  tokens, student pass with context = selected only — and MSE the two in
  the same 1664-d space, same positions. "Selection should make the
  sparse-context prediction match the full-information prediction." Our
  adaptation, not a published recipe (flagged in training.md §5);
  user approved direction explicitly.

**Implementation:** `autogaze/models/borissal/vjepa21_hub.py`
(`VJEPA21HubTeacher`, same 3-method interface; ST gate applied at encoder
OUTPUT features there — no clean input-embedding injection point in the
original code; equally valid gradient conduit). Trainer: `--teacher
hub:<entrypoint>` scheme + automatic oracle-reference targets for hub
teachers. Verified: gate gradient flows, teacher params isolated, 20/20
tests green.

**Local training run (2.1-B, MPS, 30 steps, batch 1, scale 384, fixed
ratio 0.3, duplicated example clips):** ran end-to-end, checkpoint saved,
peak 602MB, ~125s/step (dense + oracle predictor + sparse + student
predictor + backward). Loss scale ~0.05-0.08 (same-head comparison, small
by construction). **Honest caveats:** loss did not decrease on this
degenerate single-clip setup, and grad_norm decayed 0.17→0.0002 over 30
steps — consistent with score-head saturation (softmax sharpening kills
the ST soft path). For scale runs, start with `--w-entropy > 0` and/or a
lower lr; this is exactly what the training.md §3 experiment matrix is
for. Mechanism validation (the purpose of this run) is complete.

---

## 2026-07-14 (same day) — Borissal v0.2: seven elements, quantitative gate, preset finalized

User specified v0.2's element list (frame-diff motion, clustering, noise/
consistency penalty, global top-k + minimum floor, edge blending, optional
center bias, local/global score blending) with hard requirements: every
element justified, fast, fully vectorized, judged against the video-
description task. A skeleton review (requested by the user, "Fable 입장에서
고도로 검토") added the key process change: a label-free quantitative
adoption gate instead of eyeball-only tuning.

**Implemented** (all config-gated, defaults byte-identical to v0.1;
`BorissalConfig.v0_2()` preset): `motion_diff=frame|tubelet` +
`frame_diff_agg`, resize-based coarse-to-fine block gate (`block_size`),
`motion_noise_floor` (topk-quantile dead-zone), `motion_consistency=
double_diff` (experimental, excluded — see below), `per_frame_allocation=
"global"` + `min_keep_per_frame_ratio`, `score_norm_blend`, `center_bias`
(off by default). New `scripts/eval_borissal_coverage.py` (coverage +
uniqueness metrics, random-selection anchor). Dump script gained
`--preset/--block-size/--noise-floor/...` flags, `07_coarse.png`, and a
`contiguity` metric in summary.json. Tests 20 → 33, all green throughout.

**Final preset** (gate-verified Pareto-best vs v0.1 at ratio 0.25 on the
example clip: coverage 8.226<8.249 AND uniqueness 8.370>8.300; contiguity
2.31→2.94; latency 5.9→12.2ms CPU): frame-diff + quantile floor + global
allocation (25% floor) + blend 0.7 + block gate b=2.

**Key findings recorded in design.md's "Borissal v0.2" section** (the
substance of this round — read it when resuming):
1. Coverage-alone is scatter-biased — random beat all saliency configs on
   it; gate rule is therefore two-metric; the SAME bias affects the v1 SSL
   objective (Phase 3 scale runs should add an anti-scatter term).
2. Under global allocation the c2f gate was silently fully-open (worst-
   case sizing) — the metric exposed it; fixed with 2x-share sizing.
3. double_diff: negative result (normalization cancels its benefit;
   suppresses untextured movers) — excluded from preset, knob retained.
4. Corrected double-diff semantics claim (removes ghosting/uncorrelated
   noise, NOT single-frame events).

**Verified:** 33/33 tests; budget exactness for global(+c2f) across
ratios; concentration on high-energy tubelets (synthetic); center-bias
tie-break; blend beta=1 identity; qualitative overlays
(`outputs/borissal/v02_final_preset/`); gate JSONs
(`outputs/borissal/gate_v02*.json`).

**Open:** validate the preset's gate numbers on more clips than
example_input.mp4 (single-clip numbers are indicative, not conclusive);
`center_bias` remains per-domain; low-ratio (<0.1) callers should set
block_size=1.

---

## 2026-07-14 (same day) — v0.2 signals wired into v1 training; collapse monitoring added

User asked whether the training setup is complete and v0.2-based, how the
loss is composed, whether gazing_ratio varies during training, and whether
the "always selects the same patches" collapse is a live risk. Gap analysis
answered honestly (infra complete; v1's internal v0 signals were still
v0.1; collapse risk real — its signature was observed twice) and the gaps
were closed this round:

1. **v0.2 → v1 signal wiring**: `BorissalV1Config.v0_preset` (default
   "v0.2") — the learned scorer's input maps / residual base now come from
   the gate-validated v0.2 SIGNAL knobs (frame-diff, noise floor, score
   blend); selection-stage knobs (global/block) are internally overridden
   to cheap defaults since v1 does its own selection. `--input-v0-preset`
   in the trainer for ablation.
2. **v1 global-allocation inference**: `_selection_from_scores` gained the
   global branch (exact K_total + per-tubelet floor + canonical ascending —
   contract-tested), so a trained v1 can serve selections with the same
   allocation policy as the v0.2 preset.
3. **Anti-scatter loss option**: `uniqueness_reward` in losses.py (capped
   negative MSE of rest→selected reconstruction; gradient reaches the
   selector via an INVERSE ST gate on the unselected side — verified by
   test). Counters the measured scatter bias of pure coverage.
4. **Collapse monitoring + safer default**: trainer logs
   `probe_overlap_prev` (fixed-clip eval-mode selection IoU across log
   points; pinned 1.0 + dying grad = frozen selector) and
   `score_entropy_mean`; `--w-entropy` default changed 0 → 0.01 (both
   real-teacher smokes had shown grad death from score saturation).

**Verified:** 36/36 tests (3 new: preset paths, v1 global contract,
uniqueness gradient flow). Smoke (tiny teacher, entropy default +
`--w-uniqueness 0.1`): all three loss terms active,
`score_entropy_mean` RISES 3.90→4.15 (saturation counteracted),
`probe_overlap_prev` fluctuates 0.09–0.30 (no freeze; random-selection
reference ≈ 0.3 at ratio 0.3).

**Loss composition summary as of now** (the user's question, for the
record): main = predictor coverage (oracle-reference variant for hub-2.1
teachers); optional weighted terms = dense-sparse match (aux only),
score entropy (default 0.01), v0-distill warmup (decayed), uniqueness
reward (anti-scatter, default 0); gazing_ratio sampled per batch
(uniform 0.15–0.75, rank-synced) so the selector trains budget-general.

---

## 2026-07-14 (same day) — gradient-reach diagnostics for unselected/low-score patches

User relayed a team concern: do unselected (or very low-probability)
patches receive gradient during training, or do they freeze out? Answered
with analysis + live instrumentation:

- **Channels** (documented in training.md §2): softmax Jacobian coupling
  (∝ p_j — vanishes as p_j→0), Gumbel exploration (the real revival
  channel; protected by the entropy default), the entropy term itself,
  and the uniqueness inverse gate. The concern is legitimate in the limit
  — hence measurement, not just argument.
- **Instrumentation**: `forward_train` now retains the logits' grad
  (`out["logits"].grad` readable after backward); the trainer logs
  `lgrad_sel_mean` / `lgrad_unsel_mean` / `lgrad_low_decile_mean` /
  `lgrad_unsel_zero_frac` every log point.
- **Measured (smoke, entropy 0.01 + uniqueness 0.1)**: zero-grad fraction
  among unselected = 0.0; unselected mean ≈ 0.5–0.6× selected mean;
  lowest-prob decile same order — no freeze-out under current defaults.
  Alarm rule for scale runs recorded in training.md (low-decile collapsing
  orders below selected ⇒ raise --w-entropy / gumbel_tau).
- New test `test_gradient_reaches_unselected_and_low_prob_patches`
  (37/37 green).
