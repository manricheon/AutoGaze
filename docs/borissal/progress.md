# Borissal — progress / handoff log

Read this file first when resuming work on the patch-selector line in a new
session. See `design.md` in this directory for the full design rationale.

---

## 2026-07-14 — Phase 1 (Borissal-signal) implemented and verified

**Branch**: `feat/selector` (new, off `main`).

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
- Nothing committed to git yet as of writing this entry — pending user's
  go-ahead on commit/push conventions for this branch.

**Next up (Phase 2, not started):** learned selector (TSM or conv3d backbone
+ scoring head + straight-through top-k), using Borissal-signal's
motion/spatial maps as candidate input features or initialization. See
design.md's "Open items for Phase 2/3".
