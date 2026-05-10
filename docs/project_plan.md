# AutoGaze Project Review And Reproduction Plan

This plan documents the current project state, how to review the branch, how to reproduce the main workflows, and where to improve the repository next. It focuses on the original AutoGaze project and the current `eval` branch updates.

## Summary

AutoGaze is an autoregressive gaze model for efficient video understanding. It selects informative video patches so downstream ViTs and MLLMs can process fewer visual tokens without major information loss. The original branch centers on the model, training pipeline, SigLIP integration, and quick-start usage.

The current `eval` branch extends that base with evaluation and integration infrastructure:

- Video QA benchmark runner and task registry.
- MLLM runner registry with `native`, `hook`, and `full` integration modes.
- Additional ViT/MLLM integration paths for SigLIP, Qwen2.5-VL, and V-JEPA2.
- Inference scripts for gaze visualization and full MLLM pipelines.
- CV task comparison scripts for depth, detection, classification, segmentation, SigLIP, and action-recognition models.
- Expanded documentation and Korean notebooks for operational use.

The branch should be reviewed as an evaluation and integration project layered on top of the original AutoGaze core.

## Implementation Snapshot

Stable project layers:

- Core model and training: original AutoGaze model, VideoMAE reconstruction task, NTP/GRPO algorithms, Hydra-driven training entrypoint.
- Evaluation layer: `autogaze/eval/tasks.py`, `autogaze/eval/models.py`, and `autogaze/eval/run_benchmark.py`.
- Integration layer: customized SigLIP, Qwen2.5-VL, and V-JEPA2 paths plus `native`, `hook`, and `full` integration mode documentation.
- Workflow layer: inference scripts, benchmark scripts, CV task scripts, notebooks, and guides.

Current primary runner keys:

| Runner | Default mode | Purpose | Required extra paths |
| :--- | :--- | :--- | :--- |
| `nvila` | `native` | SigLIP + NVILA video QA | `--autogaze-path` |
| `siglip_qwen25` | `hook` | SigLIP + Qwen2.5-VL video QA | none beyond model/AutoGaze paths |
| `vjepa2_nvila` | `full` | V-JEPA2 + NVILA video QA | `--vjepa2-path` |
| `vjepa2_qwen25` | `full` | V-JEPA2 + Qwen2.5-7B video QA | `--vjepa2-path`, `--lm-path` |
| `vjepa2` | `hook` | V-JEPA2 feature extraction | `--vjepa2-path` |
| `siglip` | `hook` | SigLIP feature extraction | none beyond model/AutoGaze paths |

Deprecated aliases such as `qwen25vl`, `qwen25vl_full`, `vjepa2_llm`, and `nvila_vjepa2` may still work, but new examples should use the primary keys above.

## Collaboration Context

Claude work logs requested:

- More MLLM runner options and clearer runner naming.
- Explicit support and explanation for `native`, `hook`, and `full` integration modes.
- Visual and more detailed integration documentation.
- A dedicated AutoGaze model guide explaining architecture, training, inference, and autoregressive decoding.
- Video-related benchmarks, with action recognition as a specific example.

Gemini guidance contributes:

- A concise AI collaboration guide in `GEMINI.md`.
- Operational commands for downloads, inference, benchmarks, and training.
- A documentation map for `README.md`, `QUICK_START.md`, `TRAIN.md`, `docs/eval_guide.md`, `docs/integration_guide.md`, `docs/benchmark_guide.md`, `docs/inference_guide.md`, and `docs/guide_ko.md`.
- Working expectations: preserve original core logic, use config-first updates, validate with scripts or notebooks, and keep docs synchronized.

Known documentation drift to handle carefully:

- `GEMINI.md` is useful for workflow context but lists older runner aliases in its benchmark section.
- `docs/eval_guide.md` and `autogaze/eval/models.py` are the current runner-name authorities.
- `docs/inference_guide.md` overlaps with `docs/eval_guide.md`; keep new operational examples in sync with both only when they affect inference.

## Review Plan

Review the repository in this order:

1. Original project baseline:
   - Read `README.md`, `QUICK_START.md`, `TRAIN.md`, and root `INTEGRATION.md`.
   - Confirm the original purpose: patch selection for efficient video ViT/MLLM processing.
   - Confirm the original training story: NTP pre-training followed by GRPO with reconstruction reward.

2. Eval branch additions:
   - Read `docs/eval_guide.md` for task names, runner keys, CLI flags, and dataset handling.
   - Read `autogaze/eval/tasks.py` for benchmark schema assumptions.
   - Read `autogaze/eval/models.py` for runner registry, aliases, integration support, and mode limitations.
   - Read `autogaze/eval/run_benchmark.py` for evaluation flow, video loading, output format, and resume behavior.

3. Integration docs and implementation:
   - Read `docs/integration_guide.md` and `docs/benchmark_guide.md`.
   - Compare the documented integration modes with runner behavior.
   - Verify which runners support `native`, `hook`, and `full`, and where unsupported modes raise explicit errors.

4. Inference and CV tasks:
   - Read `docs/inference_guide.md`, `autogaze/infer.py`, `autogaze/infer_full.py`, and `scripts/run_cv_tasks.py`.
   - Confirm supported output formats, default model paths, frame sampling behavior, and local/offline expectations.
   - Check that CV task docs match the script's `ALL_TASKS`.

5. Notebooks and docs consistency:
   - Review notebook names and referenced commands for consistency with scripts.
   - Keep `docs/guide_ko.md` aligned with operational changes.
   - Treat `weights/`, `results/`, and generated media as local artifacts, not documentation sources.

6. Acceptance criteria for a review pass:
   - Current runner names and deprecated aliases are clearly separated.
   - Every documented command uses an existing script/module and current flag names.
   - AutoGaze ON/OFF comparison is reproducible for at least one small benchmark smoke test.
   - Hook mode is not described as providing latency speedup.
   - Any unsupported `full` mode path fails explicitly rather than silently falling back.

## Reproduction Plan

Set up the environment:

```bash
conda create -n autogaze python=3.11
conda activate autogaze
conda install -c nvidia cuda-toolkit=12.8
pip install uv
uv pip install -e .
```

Download required models:

```bash
bash scripts/download_models.sh
bash scripts/download_models.sh weights mllm
bash scripts/download_models.sh weights cv
```

Download evaluation datasets as needed:

```bash
bash scripts/download_data_eval.sh data/eval hf_bytes
bash scripts/download_hlvid.sh data/HLVid
```

Run a single-video gaze visualization:

```bash
python -m autogaze.infer \
  assets/example_input.mp4 \
  --model-path weights/AutoGaze \
  --output-format viz,npy
```

Run a small VideoMME AutoGaze ON benchmark:

```bash
python -m autogaze.eval.run_benchmark \
  --task videomme \
  --mllm nvila \
  --model-path weights/NVILA-8B-HD-Video \
  --autogaze-path weights/AutoGaze \
  --gazing-ratio 0.75 \
  --max-samples 5
```

Run the matching AutoGaze OFF baseline:

```bash
python -m autogaze.eval.run_benchmark \
  --task videomme \
  --mllm nvila \
  --model-path weights/NVILA-8B-HD-Video \
  --autogaze-path weights/AutoGaze \
  --no-autogaze \
  --max-samples 5
```

Run a hook-mode compatibility smoke test:

```bash
python -m autogaze.eval.run_benchmark \
  --task videomme \
  --mllm siglip_qwen25 \
  --integration hook \
  --model-path weights/Qwen2.5-VL-7B-Instruct \
  --autogaze-path weights/AutoGaze \
  --max-samples 5
```

Run a full-mode smoke test only when the required weights are present:

```bash
python -m autogaze.eval.run_benchmark \
  --task videomme \
  --mllm vjepa2_qwen25 \
  --integration full \
  --vjepa2-path weights/vjepa2-vitl-fpc64-256 \
  --lm-path weights/Qwen2.5-7B-Instruct \
  --autogaze-path weights/AutoGaze \
  --max-samples 5
```

Run a CV task visualization:

```bash
python scripts/run_cv_tasks.py \
  --input assets/example_input.mp4 \
  --tasks depth yolos siglip videomae_cls xclip \
  --ag-ratio 0.5
```

Use `--hf-data-dir` for offline benchmark runs after dataset repositories are downloaded locally:

```bash
python -m autogaze.eval.run_benchmark \
  --task videomme \
  --hf-data-dir data/eval/Video-MME \
  --mllm nvila \
  --model-path weights/NVILA-8B-HD-Video \
  --autogaze-path weights/AutoGaze \
  --max-samples 5
```

## Test Plan

Start with low-cost checks:

```bash
python -m autogaze.eval.run_benchmark --help
python scripts/run_cv_tasks.py --help
python - <<'PY'
from autogaze.eval.tasks import TASKS
from autogaze.eval.models import RUNNERS
print(sorted(TASKS))
print(sorted(RUNNERS))
PY
```

Then run smoke workflows:

- Single-video inference on `assets/example_input.mp4`.
- CV tasks on one short video and a small task subset.
- Benchmark with `--max-samples 5` before any full run.
- AutoGaze ON and OFF benchmark comparison for the same task, runner, and sample limit.

For GPU and integration validation:

- Test `nvila` native mode first, because it is the main supported path.
- Test `hook` mode for a compatibility-focused runner before relying on `full`.
- Test `full` mode only for runners that document support in `docs/eval_guide.md` and do not raise `NotImplementedError`.
- Record latency, VRAM, visual token counts, and accuracy together; latency alone is not enough to validate AutoGaze quality.

## Improvement Roadmap

Documentation cleanup:

- Keep `CODEX.md`, `GEMINI.md`, and `docs/project_plan.md` aligned as role-specific docs.
- Fix outdated runner names in module docstrings if they diverge from `docs/eval_guide.md`.
- Keep all benchmark examples using the same current runner convention.
- Clarify which commands require downloaded local weights and which can use HuggingFace IDs.
- Add a short "current vs deprecated runner names" note to any guide that still shows old names.

Runner and benchmark reliability:

- Add small unit tests for `TaskConfig` prompt building, answer parsing, and video path resolution.
- Add registry tests for runner aliases, default integration modes, and unsupported mode errors.
- Add a tiny fixture or mock runner so benchmark output and resume behavior can be tested without loading large models.
- Validate `--hf-data-dir` behavior for each HF-bytes task.
- Add a smoke test that imports `TASKS` and `RUNNERS` and asserts the primary runner keys are present.

Inference and CV workflow:

- Add smoke tests for `autogaze.infer` argument parsing and output path creation.
- Add explicit documentation for expected output directories under `results/`.
- Confirm CV task documentation matches `scripts/run_cv_tasks.py` task keys and defaults.

Notebook reproducibility:

- Add a notebook checklist: required weights, expected runtime, required GPU memory, offline flags, and output artifacts.
- Keep notebooks as demonstrations, with scripts as the authoritative reproducible workflows.

Packaging and repository hygiene:

- Avoid tracking generated results, model weights, caches, and platform files.
- Keep optional heavy dependencies documented by workflow where possible.
- Prefer small deterministic smoke checks before large benchmark runs.

## Update Policy

When implementing future updates from this plan:

- Make code and docs changes in the same branch when CLI behavior changes.
- Add the smallest useful test before broad benchmark runs.
- Keep old aliases working only for compatibility; do not use them in new examples.
- Record benchmark output paths and exact model/data paths in any result summary.
- Keep large artifacts under `weights/`, `results/`, or external storage rather than committing them.
