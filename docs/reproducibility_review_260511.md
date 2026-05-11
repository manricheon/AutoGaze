# AutoGaze Branch Review — 260511 Reproducibility

This note records the 260511 review request: before new advanced-model development, the branch should support a reproducible PoC from open checkpoints and paper/code materials, with reliable AutoGaze ON/OFF comparison.

## Current Status

- The branch is primarily an evaluation and integration branch on top of the original AutoGaze code.
- Stable core components remain the original AutoGaze model, NTP/GRPO training pipeline, and SigLIP-compatible gaze integration.
- Added branch components include video QA benchmarks, MLLM runner registry, CV task visualizations, V-JEPA2/Qwen integration paths, and inference notebooks.
- `autogaze.eval.run_benchmark` is the most reliable benchmark entry point for repeatable AutoGaze ON/OFF comparisons.
- `autogaze.infer` is the gaze-only inspection entry point.
- `autogaze.infer_full` is the interactive full-pipeline entry point and now follows the same current runner keys as the benchmark runner.

## Inference Pipeline Review

### `autogaze/infer.py`

Purpose: run AutoGaze only, generate gaze maps, videos, frame overlays, `.npz`, and optional `gazing_labels.json`.

Recommended PoC commands:

```bash
python -m autogaze.infer \
  assets/example_input.mp4 \
  --model-path weights/AutoGaze \
  --output-dir results/repro_260511/gaze \
  --output-format viz,npy,video \
  --gazing-ratio 0.75
```

For full-frame qualitative inspection:

```bash
python -m autogaze.infer \
  assets/example_input.mp4 \
  --model-path weights/AutoGaze \
  --output-dir results/repro_260511/gaze_all_frames \
  --all-frames \
  --output-format frames,video \
  --gazing-ratio 0.75
```

Notes:

- `--compare-autogaze` runs the requested ratio and ratio `1.0`; for gaze-only inspection this is an all-patch reference rather than a separate downstream model baseline.
- `--sweep-ratio` is useful for visualizing token-budget effects before expensive MLLM runs.
- Use `--no-task-loss-requirement` if the PoC must isolate `--gazing-ratio` without early-stop behavior.

### `autogaze/infer_full.py`

Purpose: run AutoGaze + ViT + MLLM or feature extraction for interactive inspection.

Current primary runner keys:

| Runner | Default mode | Purpose |
| :--- | :--- | :--- |
| `nvila` | `native` | Main open NVILA full-pipeline path |
| `siglip_qwen25` | `hook` | Qwen2.5-VL compatibility/full-mode experiments |
| `vjepa2_nvila` | `full` | V-JEPA2 encoder + NVILA language model |
| `vjepa2_qwen25` | `full` | V-JEPA2 encoder + Qwen2.5 language model |
| `vjepa2` | `hook` | V-JEPA2 feature extraction only |
| `siglip` | `hook` | SigLIP feature extraction only |

Recommended NVILA ON/OFF PoC:

```bash
python autogaze/infer_full.py \
  assets/example_input.mp4 \
  --mllm nvila \
  --model-path weights/NVILA-8B-HD-Video \
  --autogaze-path weights/AutoGaze \
  --gazing-ratio 0.75 \
  --compare-autogaze \
  --max-new-tokens 64
```

Recommended Qwen compatibility PoC:

```bash
python autogaze/infer_full.py \
  assets/example_input.mp4 \
  --mllm siglip_qwen25 \
  --integration hook \
  --model-path weights/Qwen2.5-VL-7B-Instruct \
  --autogaze-path weights/AutoGaze \
  --compare-autogaze \
  --max-new-tokens 64
```

Recommended V-JEPA2 + Qwen PoC:

```bash
python autogaze/infer_full.py \
  assets/example_input.mp4 \
  --mllm vjepa2_qwen25 \
  --integration full \
  --vjepa2-path weights/vjepa2-vitl-fpc64-256 \
  --lm-path weights/Qwen2.5-7B-Instruct \
  --autogaze-path weights/AutoGaze \
  --compare-autogaze \
  --max-new-tokens 64
```

Notes:

- Native NVILA baseline needs the AutoGaze config path even when comparing against all-patch behavior; the script now keeps `--autogaze-path` and forces ratio `1.0` for that baseline.
- `--vjepa2-path` is the user-facing flag for V-JEPA2 encoder weights.
- Deprecated runner aliases still work through the shared runner registry, but new work should use the primary keys.

## Benchmark Reproduction

Use `autogaze.eval.run_benchmark` for recordable benchmark runs.

Smoke test:

```bash
python -m autogaze.eval.run_benchmark \
  --task videomme \
  --mllm nvila \
  --model-path weights/NVILA-8B-HD-Video \
  --autogaze-path weights/AutoGaze \
  --max-samples 5 \
  --output results/repro_260511/videomme_nvila_ag075.json
```

Matching baseline:

```bash
python -m autogaze.eval.run_benchmark \
  --task videomme \
  --mllm nvila \
  --model-path weights/NVILA-8B-HD-Video \
  --autogaze-path weights/AutoGaze \
  --no-autogaze \
  --max-samples 5 \
  --output results/repro_260511/videomme_nvila_baseline.json
```

For offline datasets, prefer `--hf-data-dir` with locally downloaded dataset repos.

## Review Outcome

Implemented during this review:

- `infer.py` now has a local device resolver, so `python -m autogaze.infer --help` and gaze-only inference no longer fail on a missing `autogaze.utils.get_device` import.
- `infer_full.py` now exposes current primary runner keys and `--integration`.
- `infer_full.py` now supports `--vjepa2-path` consistently with benchmark commands.
- Native NVILA ON/OFF comparison now preserves AutoGaze config loading while forcing ratio `1.0` for baseline.
- Added tests for full-pipeline CLI argument resolution and V-JEPA2 path handling.

Remaining work:

- Run real GPU PoC commands above with local open checkpoints and record outputs under `results/repro_260511/`.
- Validate projector-dependent V-JEPA2+LLM paths with trained projector checkpoints before treating their answers as quality benchmarks.
- Keep notebooks synchronized with the current runner keys after the CLI changes.
