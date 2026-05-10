# AutoGaze Codex Working Guide

This file is the working guide for Codex sessions in this repository. It summarizes the project purpose, the current branch intent, and the safest way to review, update, reproduce, and test the project.

## Project Purpose

AutoGaze, or Autoregressive Gazing, predicts informative video patches before a downstream ViT or MLLM processes the video. The original project goal is to reduce visual token load for high-resolution, high-FPS, long-form videos while preserving downstream task quality.

The original model stack is:

- A shallow video convolutional encoder for lightweight spatio-temporal features.
- A connector with learned patch-position embeddings.
- A LLaMA-style autoregressive gaze decoder that emits patch IDs.
- Two-stage training: NTP pre-training on pseudo gaze labels, then GRPO reinforcement learning with VideoMAE reconstruction reward.

The current `eval` branch extends the original project with evaluation, inference, benchmark, and integration tooling. It should be treated as an evaluation and integration branch built on top of the original AutoGaze core.

## Current Branch Context

The branch is `eval`. Compared with `main`, this branch adds evaluation infrastructure and documentation around:

- Video QA benchmark tasks in `autogaze/eval/`.
- MLLM runner registry and integration modes: `native`, `hook`, and `full`.
- Additional ViT and MLLM paths including SigLIP, Qwen2.5-VL, and V-JEPA2.
- Single-video and full-pipeline inference scripts.
- CV task comparisons for depth, detection, classification, segmentation, SigLIP, VideoMAE-CLS, and X-CLIP.
- Korean notebooks and expanded documentation guides.

Use this branch as a practical evaluation branch, not just a paper-code snapshot. The main implementation surface is now split into three layers:

- Original AutoGaze core: `autogaze/models/`, `autogaze/tasks/video_mae_reconstruction/`, `autogaze/algorithms/`, and `autogaze/train.py`.
- Evaluation and integration layer: `autogaze/eval/`, `autogaze/vision_encoders/qwen25vl/`, `autogaze/vision_encoders/vjepa2/`, and benchmark scripts.
- User-facing workflows: `autogaze/infer.py`, `autogaze/infer_full.py`, `scripts/run_cv_tasks.py`, notebooks, and docs.

Local collaboration records are important source material:

- `.claude/260508_request.md`: requested expanded MLLM runners, integration-mode explanation, model guide, and video/action-recognition benchmarks.
- `.claude/260508_guide.md`: recommended reading order for eval-branch work.
- `GEMINI.md`: AI collaboration guide, workflow commands, and documentation map.
- `CLUADE.md`: older brief Claude-facing guide; keep the existing filename unless the user asks to rename it.

`GEMINI.md` still contains some deprecated runner aliases. Treat `docs/eval_guide.md` and `autogaze/eval/models.py` as the current source of truth for runner keys.

## Source Of Truth Docs

Use these docs first when orienting, in this order:

- `docs/eval_guide.md`: benchmark tasks, runner names, dataset prep, flags, and metrics.
- `docs/integration_guide.md`: integration modes and new backend guidance.
- `INTEGRATION.md`: root integration overview for ViTs and MLLMs.
- `docs/benchmark_guide.md`: ViT/MLLM and CV task benchmark interpretation.
- `docs/guide_ko.md`: broad Korean project guide and end-to-end operational reference.
- `README.md`: original project overview, installation, and high-level architecture.
- `QUICK_START.md`: basic AutoGaze use, SigLIP application, any-resolution use, and streaming.
- `TRAIN.md`: NTP and GRPO training workflow.
- `docs/action_recognition_guide.md`: action-recognition benchmark notes.
- `docs/autogaze_model_guide.md`: model architecture, training, and inference walkthrough.

`docs/inference_guide.md` is useful for inference-specific detail, but much of its operational content overlaps with `docs/eval_guide.md`.

## Current Runner Keys

Primary runner keys follow the `{vit}_{lm}` convention:

| Runner | ViT | LLM / output | Default mode | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `nvila` | SigLIP custom | NVILA | `native` | Main production-style path; native baseline still needs `--autogaze-path`. |
| `siglip_qwen25` | SigLIP | Qwen2.5-VL | `hook` | Supports hook and full paths. |
| `vjepa2_nvila` | V-JEPA2 | NVILA | `full` | Requires `--vjepa2-path`. |
| `vjepa2_qwen25` | V-JEPA2 | Qwen2.5-7B | `full` | Requires `--vjepa2-path` and `--lm-path`. |
| `vjepa2` | V-JEPA2 | features | `hook` | Feature extraction only, not MCQ generation. |
| `siglip` | SigLIP | features | `hook` | Feature extraction only. |

Deprecated aliases still work but should not be used in new docs or scripts:

- `nvila_vjepa2` -> `vjepa2_nvila`
- `qwen25vl` -> `siglip_qwen25`
- `qwen25vl_full` -> `siglip_qwen25 --integration full`
- `vjepa2_llm` -> `vjepa2_qwen25`
- `vjepa2_full` -> `vjepa2 --integration full`

## Working Rules

- Preserve the original AutoGaze core unless the user explicitly asks for model or training changes.
- Prefer config-first changes over hardcoded parameters.
- Do not modify `weights/`, `results/`, or generated artifacts unless the task explicitly requires it.
- Keep evaluation docs, runner names, CLI flags, and notebooks synchronized when changing benchmark behavior.
- Treat `native`, `hook`, and `full` as distinct integration modes:
  - `native`: model-specific baked-in integration, currently NVILA-oriented.
  - `hook`: zero-shot masking with unchanged sequence length; useful for compatibility checks.
  - `full`: physical token removal; required for real latency and VRAM reduction.
- Use local paths and offline loading behavior where supported. Several scripts are designed to work with `weights/` and pre-downloaded datasets.
- When reviewing branch changes, separate stable eval infrastructure from untracked local outputs and large artifacts.
- Prefer updating docs and smoke tests together when changing user-facing CLI behavior.
- If a guide and code disagree, inspect code first, then update the guide to match the current implementation.

## Common Commands

Install:

```bash
conda create -n autogaze python=3.11
conda activate autogaze
conda install -c nvidia cuda-toolkit=12.8
pip install uv
uv pip install -e .
```

Download models:

```bash
bash scripts/download_models.sh
bash scripts/download_models.sh weights mllm
bash scripts/download_models.sh weights cv
```

Download evaluation data:

```bash
bash scripts/download_data_eval.sh data/eval hf_bytes
bash scripts/download_hlvid.sh data/HLVid
```

Run single-video AutoGaze inference:

```bash
python -m autogaze.infer assets/example_input.mp4 --model-path weights/AutoGaze --output-format viz,npy
```

Run full inference:

```bash
python autogaze/infer_full.py assets/example_input.mp4 --mllm nvila
```

Run a benchmark smoke test:

```bash
python -m autogaze.eval.run_benchmark \
  --task videomme \
  --mllm nvila \
  --model-path weights/NVILA-8B-HD-Video \
  --autogaze-path weights/AutoGaze \
  --max-samples 5
```

Run AutoGaze OFF baseline:

```bash
python -m autogaze.eval.run_benchmark \
  --task videomme \
  --mllm nvila \
  --model-path weights/NVILA-8B-HD-Video \
  --autogaze-path weights/AutoGaze \
  --no-autogaze \
  --max-samples 5
```

Run CV task comparison:

```bash
python scripts/run_cv_tasks.py \
  --input assets/example_input.mp4 \
  --tasks depth yolos siglip videomae_cls xclip \
  --ag-ratio 0.5
```

Inspect runner and task code:

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

## Review Checklist

Before changing behavior, check:

- Does the requested change belong to original AutoGaze core, eval infrastructure, docs, notebooks, or scripts?
- Are runner names still aligned with the `{vit}_{lm}` convention in `docs/eval_guide.md` and `autogaze/eval/models.py`?
- Does the change affect `native`, `hook`, or `full` mode semantics?
- Does it require updating dataset download instructions or offline `--hf-data-dir` behavior?
- Is a low-cost smoke test available before running large GPU benchmarks?
- Are generated results, model weights, and local caches excluded from the intended edit?
- Have deprecated runner aliases been kept out of new examples?
- Does any new benchmark path support both AutoGaze ON and OFF comparison?
- Is the claimed speed benefit tied to `full` or `native` mode, not `hook` mode?
