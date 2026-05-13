# Branch Status Audit

Audit date: 2026-05-13

Scope: audit-only validation of the current `bench` branch for the NVILA-HD-Video PoC feature set.

This audit did not modify original AutoGaze source files, original `INTEGRATION.md`, original `QUICK_START.md`, or `docs/nvila-hd-video-readme.md`.
No large benchmark, model download, dataset download, or paper-reproduction run was performed.

Post-audit fix note: after this report was generated, the first small fixes were applied:
`--strict-autogaze-params` was added to `scripts/poc_nvila_hd_video.py`, PoC benchmark presets were normalized with `run_by_default=false`, and the PoC feature matrix/guide were updated to reflect the config-driven component override policy. Historical PASS/PARTIAL/STUB rows below reflect the branch at audit time.

## 1. Branch Summary

| Item | Result |
|---|---|
| Current branch | `bench` |
| Latest commit | `ad4201b Add PoC AutoGaze impact benchmark wrapper` |
| Tracked diff before audit report | none |
| Protected original file tracked diff | none |
| Generated outputs tracked | none found under `outputs/`, `results/`, or `weights/` |
| Large tracked files added in current diff | none |
| Large local untracked files | `weights/` is present locally and about 27G, but not tracked |

Commands run:

```bash
git branch --show-current
git log -1 --oneline
git status --short
git diff --stat
git diff --name-only
git diff --name-only -- INTEGRATION.md QUICK_START.md docs/nvila-hd-video-readme.md autogaze
git ls-files | rg '^(outputs|results|weights|notebooks|.*\.mp4|.*\.pt|.*\.pth|.*\.safetensors|.*\.bin|.*\.ckpt)'
find . -type f -size +50M -not -path './.git/*' -print
```

Untracked local files reported by `git status --short`:

```text
.DS_Store
.claude/
AGENTS.md
CLUADE.md
CODEX.md
GEMINI.md
autogaze/.DS_Store
docs/PROJECT_REQUEST.md
docs/mamba_gaze_plan.md
docs/mamba_gaze_plan_v2.md
docs/mamba_gaze_ref_v0.md
mamba_gaze/
notebooks/
results/
scripts/.gitkeep
weights/
```

Protected file check:

| Protected target | Status |
|---|---|
| `INTEGRATION.md` | PASS, no tracked modification |
| `QUICK_START.md` | PASS, no tracked modification |
| `docs/nvila-hd-video-readme.md` | PASS, no tracked modification |
| original `autogaze/` source files | PASS, no tracked modification; only `autogaze/.DS_Store` is untracked |

Large file check:

| Area | Status |
|---|---|
| tracked large checkpoint/model files in current diff | PASS, none |
| untracked local weights | NOT TRACKED, `weights/` contains large checkpoint/model files |
| tracked generated outputs | PASS, none found |
| tracked media | NOTE, `assets/example_input.mp4` is tracked and appears pre-existing |

## 2. Feature Matrix Status

Status labels:

```text
PASS     implemented and covered by tests or safe smoke behavior
PARTIAL  constrained implementation, guarded real path, or missing requested details
STUB     explicit skip or NotImplementedError path
BLOCKED  requires external checkpoint/API/hardware validation
NOT TESTED not exercised in this audit
```

### Modes

| Feature | Status | Evidence / note |
|---|---|---|
| `check` | PASS | `outputs/audit_check` status `passed`; import/path checks reported |
| `autogaze_only` | PARTIAL | Safe smokes run and emit metadata; real AutoGaze skipped without `--allow-checkpoint-load` |
| `full_pipeline` | PARTIAL | Query text preserved; AutoGaze/SigLIP/NVILA stages skipped clearly when checkpoint loading disabled |

### Frame Selection

| Feature | Status | Evidence / note |
|---|---|---|
| `sample` | PASS | Smoke `outputs/audit_autogaze_sample`; tests cover frame selector |
| `chunk` | PASS | Smoke `outputs/audit_autogaze_chunk`; multiple windows reported |
| `interval` | PASS | Smoke `outputs/audit_autogaze_interval`; interval metadata saved |
| `all` | PASS | Unit tests cover alias behavior as chunked processing |
| overlapping stride windows | STUB/FUTURE | intentionally not implemented |

### Scaling / Chop

| Feature | Status | Evidence / note |
|---|---|---|
| `none` | PARTIAL | implemented as no resize, but fixed-resolution model compatibility is not enforced |
| `resize` | PASS | smoke uses `resize`; tests cover square output |
| `fit_short_side` | PASS | tests cover aspect-preserving short-side scaling |
| `fit_long_side` | PASS | tests cover aspect-preserving long-side scaling |
| `quickstart` | PARTIAL | supports documented 224/patch16 and 392/patch14 policies; other requests report unsupported |
| `chop` | PARTIAL | smoke writes chop info inside `scaling/scaling_metadata.json`; full `chops/chop_metadata.json` requires AutoGaze output |
| non-zero chop overlap | STUB | raises `NotImplementedError` |
| custom chop stride | STUB | raises `NotImplementedError` for unsupported non-overlap semantics |

### AutoGaze Runtime Controls

| Feature | Status | Evidence / note |
|---|---|---|
| `gaze_ratio` / `--gaze-ratio` | PARTIAL | passed into runtime metadata and calls; real effect requires checkpoint execution |
| `task_loss_requirement` / `--task-loss-requirement` | PARTIAL | passed into runtime metadata and calls; real effect requires checkpoint execution |
| `strict_autogaze_params` / `--strict-autogaze-params` | STUB | requested but CLI flag is missing |
| target scales / target patch size | PARTIAL | CLI exists; quickstart 392/patch14 path covered, broader real validation pending |

### Full Pipeline Controls

| Feature | Status | Evidence / note |
|---|---|---|
| query text | PASS | full-pipeline smoke preserves query text and reports skipped generation explicitly |
| vision encoder selection | PARTIAL | config-driven via `--config`; no direct CLI override |
| vision encoder checkpoint/config override | PARTIAL | config-driven via experiment YAML; no direct CLI override |
| MLLM selection | PARTIAL | config-driven via `--config`; no direct CLI override |
| MLLM checkpoint/config override | PARTIAL | config-driven via experiment YAML; no direct CLI override |
| processor/tokenizer path | PARTIAL | config-driven; no `--processor-path` / `--tokenizer-path` CLI |
| generated answer | PARTIAL | written only if NVILA generation succeeds; skipped clearly in guarded smoke |

### Visualization

| Feature | Status | Evidence / note |
|---|---|---|
| mask overlay | PASS | unit tests cover overlay output |
| box overlay | PASS | unit tests cover box/both style |
| patch index visualization | PASS | unit tests cover disabled default and enabled labels |
| multi-scale overlay | PASS | unit tests cover scale colors and missing-scale fallback |
| scale panel video | PASS | unit tests cover 2x2 scale-panel output |
| external info panel | PARTIAL | bottom external panel implemented; `right` is accepted by CLI but raises unsupported |
| side-by-side video | PASS | unit tests cover `processed_overlay` |
| chop visualization | PARTIAL | chop-local and non-overlap paths covered; overlap/custom stride unsupported |
| full-length video export | PASS | unit tests cover metadata and output frame count |
| original-space overlay | PASS for affine modes | tests cover resize/fit mappings; chop original-space remains unsupported |
| merged chop overlay | PARTIAL | non-overlap `overlay_union` implemented; overlap/custom stride unsupported |

### Reporting

| Feature | Status | Evidence / note |
|---|---|---|
| `logs/poc_summary.json` | PASS | emitted for every smoke |
| `logs/metrics.json` / `metrics.csv` | PASS | emitted for every smoke |
| token counts | PARTIAL | emitted as summary; real counts are `None` when AutoGaze is skipped |
| selected token counts | PARTIAL | real counts require AutoGaze execution |
| token reduction ratio | PARTIAL | real ratio requires AutoGaze execution |
| selected patches per frame | PARTIAL | real values require AutoGaze execution |
| selected patches per scale | PARTIAL | real values require AutoGaze execution |
| latency | PASS | stage and end-to-end fields present; skipped stages use `N/A` |
| memory / VRAM | PARTIAL | CUDA peak VRAM supported; CPU/MPS unavailable marked `N/A` |
| skipped stages | PASS | explicit stage/reason entries present |
| generated answer status | PASS | query text accepted and generation skip is explicit |

### Benchmark Configs

| Config | Status | Note |
|---|---|---|
| `poc_feature_matrix_smoke.yaml` | PASS | loads; `heavy_benchmark=false` |
| `poc_autogaze_only_visualization.yaml` | PASS | loads; `heavy_benchmark=false` |
| `poc_full_pipeline_visualization.yaml` | PASS | loads; `heavy_benchmark=false` |
| `poc_chop_mode_smoke.yaml` | PASS | loads; `heavy_benchmark=false` |
| `poc_multiscale_visualization.yaml` | PASS | loads; `heavy_benchmark=false` |
| `poc_scale_panel_video.yaml` | PASS | loads; `heavy_benchmark=false` |
| `poc_high_resolution_chop_smoke.yaml` | PASS | loads; `run_by_default=false` |
| `poc_high_resolution_chop_medium.yaml` | PASS | loads; `run_by_default=false` |
| `poc_full_length_video_export_smoke.yaml` | PASS | loads; `run_by_default=false` |
| `poc_autogaze_impact_full_pipeline.yaml` | PASS | loads; dry-run plan by default |

Observation: several smoke configs have `run_by_default` omitted, but all have `heavy_benchmark=false`.

## 3. CLI Audit

CLI source: `python scripts/poc_nvila_hd_video.py --help`

### Supported CLI Args

| Area | Supported args |
|---|---|
| general | `--mode`, `--config`, `--video`, `--video-path`, `--output-dir`, `--device`, `--dtype`, `--json` |
| frame selection | `--frame-selection-mode`, `--num-frames`, `--frame-interval`, `--max-windows`, `--drop-last`, `--pad-last` |
| scaling/chop | `--scaling-mode`, `--resolution`, `--patch-size`, `--target-scales`, `--target-patch-size`, `--spatial-tile-size`, `--chop-size`, `--chop-overlap`, `--chop-stride`, `--max-chops`, `--chop-merge-mode`, `--save-chop-frames`, `--save-chop-overlay-video` |
| AutoGaze runtime | `--gaze-ratio`, `--task-loss-requirement` |
| visualization | `--overlay-style`, `--overlay-alpha`, `--overlay-line-width`, `--multi-scale-overlay`, `--no-multi-scale-overlay`, `--scale-color-mode`, `--show-patch-index`, `--show-scale-label`, `--hide-patch-boxes`, `--hide-patch-indices`, `--metadata-placement`, `--info-panel-position`, `--info-panel-size`, `--info-panel-mode`, `--save-overlay-video`, `--save-side-by-side-video`, `--save-scale-panel-video`, `--video-export-mode`, `--video-fps`, `--comparison-layout` |
| full pipeline | `--query-text`, `--max-new-tokens`, `--allow-checkpoint-load`, `--no-checkpoint-load`, `--checkpoint-metadata-only` |

### Missing Requested CLI Args

| Missing arg | Current status | Recommended fix |
|---|---|---|
| `--strict-autogaze-params` | missing | add strict validation flag for AutoGaze callable parameter acceptance |
| `--vision-encoder` | missing | add if runtime component override is required outside config YAML |
| `--vision-encoder-module` | missing | add config override wiring or document config-only policy |
| `--vision-encoder-class` | missing | add config override wiring or document config-only policy |
| `--vision-encoder-ckpt` | missing | add config override wiring or document config-only policy |
| `--vision-encoder-config` | missing | add config override wiring or document config-only policy |
| `--mllm` | missing | add if runtime MLLM selection is required outside config YAML |
| `--mllm-module` | missing | add config override wiring or document config-only policy |
| `--mllm-class` | missing | add config override wiring or document config-only policy |
| `--mllm-ckpt` | missing | add config override wiring or document config-only policy |
| `--mllm-config` | missing | add config override wiring or document config-only policy |
| `--processor-path` | missing | add config override wiring or document config-only policy |
| `--tokenizer-path` | missing | add config override wiring or document config-only policy |

### Args Present But Stubbed or Guarded

| Arg | Behavior |
|---|---|
| `--video-export-mode hold_last` | raises `NotImplementedError` when export is requested |
| `--info-panel-position right` | accepted by parser but raises `NotImplementedError`; only bottom supported |
| `--comparison-layout chop_overlay` | accepted by parser but raises `NotImplementedError`; use `--chop-merge-mode overlay_union` |
| non-zero `--chop-overlap` | raises `NotImplementedError` |
| custom `--chop-stride` | raises `NotImplementedError` unless equal to tile size |
| `--allow-checkpoint-load` absent | model stages are skipped with explicit reasons |

No silent query-text no-op was observed. In full-pipeline smoke, query text was preserved and MLLM generation was marked skipped because checkpoint loading was disabled.

## 4. Test Results

Full safe suite:

```bash
pytest -q
```

Result:

```text
194 passed, 1 skipped in 12.11s
```

The skipped test is the canonical real-path dry-run that requires local real checkpoint paths. It is environment-dependent, not an implementation failure.

Targeted PoC tests included in the full suite:

```text
tests/test_poc_nvila_hd_video.py
tests/test_frame_selector.py
tests/test_autogaze_scaling.py
tests/test_visualization.py
tests/test_poc_autogaze_impact_benchmark.py
tests/test_inference_guide_alignment.py
```

## 5. Safe Smoke Tests

All smokes used CPU, dummy input, small settings, and no checkpoint loading.

### Commands Run

```bash
python scripts/poc_nvila_hd_video.py --mode check --config configs/experiment/A2_real.yaml --device cpu --output-dir outputs/audit_check
```

```bash
python scripts/poc_nvila_hd_video.py --mode autogaze_only --video dummy --frame-selection-mode sample --num-frames 2 --scaling-mode resize --resolution 32 --device cpu --config configs/experiment/A2_real.yaml --output-dir outputs/audit_autogaze_sample
```

```bash
python scripts/poc_nvila_hd_video.py --mode autogaze_only --video dummy --frame-selection-mode chunk --num-frames 2 --max-windows 2 --scaling-mode resize --resolution 32 --device cpu --config configs/experiment/A2_real.yaml --output-dir outputs/audit_autogaze_chunk
```

```bash
python scripts/poc_nvila_hd_video.py --mode autogaze_only --video dummy --frame-selection-mode interval --num-frames 2 --frame-interval 2 --scaling-mode resize --resolution 32 --device cpu --config configs/experiment/A2_real.yaml --output-dir outputs/audit_autogaze_interval
```

```bash
python scripts/poc_nvila_hd_video.py --mode full_pipeline --video dummy --query-text "Question: What is happening? Please answer directly." --frame-selection-mode sample --num-frames 2 --scaling-mode resize --resolution 32 --device cpu --max-new-tokens 1 --config configs/experiment/A2_real.yaml --output-dir outputs/audit_full_pipeline_sample
```

```bash
python scripts/poc_nvila_hd_video.py --mode autogaze_only --video dummy --frame-selection-mode sample --num-frames 2 --scaling-mode chop --resolution 32 --spatial-tile-size 16 --chop-overlap 0 --max-chops 2 --save-chop-frames --device cpu --config configs/experiment/A2_real.yaml --output-dir outputs/audit_chop_smoke
```

```bash
python scripts/benchmark_poc_autogaze_impact.py --mode full_pipeline --num-frames 2 --resolution 32 --device cpu --dtype float32 --output-dir outputs/audit_poc_autogaze_impact
```

### Smoke Results

| Smoke | Status | Notes |
|---|---|---|
| check mode | PASS | imports/classes/path checks passed, no heavy loading |
| autogaze_only sample | PARTIAL | metadata saved; AutoGaze skipped due checkpoint loading disabled |
| autogaze_only chunk | PARTIAL | two windows reported; AutoGaze skipped clearly |
| autogaze_only interval | PARTIAL | interval metadata saved; AutoGaze skipped clearly |
| full_pipeline sample + query | PARTIAL | query text preserved; AutoGaze, SigLIP, NVILA skipped clearly |
| chop mode smoke | PARTIAL | scaling/chop metadata saved; top-level `chops/chop_metadata.json` not produced because AutoGaze skipped |
| PoC impact benchmark wrapper | PASS dry-run | wrote `benchmark_plan.json` and `commands.sh` only |

## 6. Output / Metadata Validation

Validated for each smoke:

```text
logs/poc_summary.json
logs/metrics.json
logs/metrics.csv
```

Validated for inference smokes:

```text
autogaze/frame_selection_metadata.json
autogaze/runtime_metadata.json
autogaze/token_counts_summary.json
scaling/scaling_metadata.json
```

Not produced in guarded no-checkpoint smokes:

```text
predictions/answer.json
visualization videos/images
top-level chops/chop_metadata.json
```

Reasons:

- `predictions/answer.json` requires successful NVILA generation.
- visualization videos/images require selected AutoGaze outputs.
- top-level chop artifacts require AutoGaze-selected patch metadata in addition to scaling/chop metadata.

Validation conclusions:

| Check | Result |
|---|---|
| query text silently ignored | PASS, not ignored; skip reason records accepted query text |
| skipped MLLM generation reported | PASS |
| AutoGaze metadata saved when downstream fails | PASS for runtime/frame/scaling/token-summary metadata; selected patch outputs require AutoGaze execution |
| sampled-only mislabeled as full-length | NOT OBSERVED |
| original-space overlay claim without mapping | NOT OBSERVED in smoke; tests cover affine mappings |
| merged chop overlay claim without valid mapping | NOT OBSERVED; overlap/custom stride remains unsupported |
| CPU/MPS memory unavailable flag | PASS, CPU smoke reports `peak_vram_mb=N/A` and unavailable flag |

## 7. Benchmark / Doc Audit

Benchmark config load command:

```bash
python - <<'PY'
from pathlib import Path
from omegaconf import OmegaConf
for path in sorted(Path('configs/benchmark').glob('poc*.yaml')):
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    print(path, cfg['benchmark'].get('mode'), cfg['benchmark'].get('heavy_benchmark'))
PY
```

Result:

- all `configs/benchmark/poc*.yaml` files loaded.
- all PoC benchmark configs report `heavy_benchmark=false`.
- high-resolution preparation configs set `run_by_default=false`.
- some visualization smoke configs omit `run_by_default`, but they remain safe templates and set `heavy_benchmark=false`.

Docs checked:

| Doc | Status |
|---|---|
| `docs/INFERENCE_GUIDE.md` | PASS, points to focused PoC guide and labels stub/future work |
| `docs/NVILA_HD_VIDEO_REFERENCE.md` | PASS, documents processor-first NVILA reference and PoC differences |
| `docs/QUICK_START_reference.md` | PASS, documents QUICK_START Python API and scaling policy |
| `docs/POC_NVILA_HD_VIDEO_FEATURE_MATRIX.md` | PARTIAL, broadly accurate but one row still says requested `--chop-*` flags are missing even though they now exist; behavior status remains mostly correct |
| `docs/POC_NVILA_HD_VIDEO_GUIDE.md` | PASS, documents CLI, A1/A2 configs, benchmark wrapper, outputs, and unsupported modes |
| `docs/benchmark.md` | PASS, documents PoC A1/A2 impact wrapper and safe defaults |
| `docs/benchmark_analysis.md` | PASS, includes A1/A2 sanity checklist and summary path |

## 8. Implemented / Partial / Stub / Blocked Lists

### Implemented

- Check mode import/path validation.
- Dummy input path.
- Frame selection: sample, chunk, interval, all.
- Resize scaling and aspect-preserving fit modes.
- Basic quickstart exact policies.
- Guarded checkpoint loading.
- Query text preservation in full pipeline.
- Metadata/metrics output.
- Mask/box/both overlays, patch labels, scale labels, multi-scale color metadata.
- Scale-panel video and side-by-side video under tested/mock conditions.
- Full-length video export under tested/mock conditions.
- Non-overlap chop overlay union under tested/mock conditions.
- PoC A1/A2 impact benchmark dry-run wrapper.

### Partial

- `autogaze_only` and `full_pipeline` real execution are guarded and depend on checkpoints/memory.
- AutoGaze runtime controls are passed through but not strictly validated against callable signatures.
- Vision encoder and MLLM selection are config-driven only, not direct CLI overrides.
- `none`, `fit_short_side`, and `fit_long_side` real model compatibility is not fully validated.
- Chop mode metadata exists in scaling metadata without checkpoint loading; richer top-level chop artifacts require AutoGaze output.
- Memory reporting is CUDA-focused; CPU/MPS unavailable metrics are `N/A`.
- MLLM prefill/decode separation is not implemented beyond a coarse generation latency.

### Stub / Unsupported

- `--strict-autogaze-params`.
- Direct per-component CLI overrides for vision encoder, MLLM, checkpoints, configs, processor/tokenizer.
- `hold_last` video export.
- `info_panel_position=right`.
- `comparison_layout=chop_overlay`.
- Non-zero chop overlap and custom chop stride.
- Direct visual token injection into arbitrary MLLMs.

### Blocked

- Real NVILA generation on limited hardware unless checkpoints and sufficient memory are available.
- Official public benchmark reproduction.
- Remote video URL path outside official processor-first mode.
- A0/A3 vanilla SigLIP benchmark readiness beyond prior feasibility/config layers.

## 9. Highest-Priority Fixes

1. Add `--strict-autogaze-params` or explicitly downgrade it to future work in the feature matrix.
2. Decide whether full-pipeline component switching should remain config-only or add requested CLI override flags.
3. Update the feature matrix row that still implies requested `--chop-*` flags are missing; the flags exist, but overlap/custom stride remain unsupported.
4. Add `run_by_default=false` to all PoC smoke configs for consistent safety labeling.
5. Add a no-checkpoint visualization skip test or report field that explicitly explains why no visualization videos are produced when AutoGaze is skipped.
6. Add a small real or mocked CLI smoke that produces overlay/side-by-side videos through the command line, not only unit tests.

## 10. Overall Summary

| Area | Summary |
|---|---|
| Git/protected files | PASS |
| Unit tests | PASS, 194 passed / 1 skipped |
| Safe no-checkpoint smokes | PARTIAL by design; explicit skip reasons present |
| Feature matrix completeness | PARTIAL; core feature states covered, some status text stale |
| CLI completeness | PARTIAL; core PoC CLI present, strict/autogaze and per-component override flags missing |
| Visualization/video export | PASS in tests, PARTIAL in no-checkpoint smoke |
| Full pipeline | PARTIAL; query handling works, real downstream execution is guarded |
| Benchmark docs/configs | PASS with minor safety-label consistency issue |

The branch is safe for continued PoC work. It should not be described as paper reproduction or as confirmed encoder-side acceleration until real A1/A2 full-pipeline runs demonstrate that A2 reduces tokens before the intended encoder compute stage.
