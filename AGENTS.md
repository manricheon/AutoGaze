# AGENTS.md

## Project Rule

- Always read `docs/PROJECT_REQUEST.md` before making architectural decisions.
- Target branch for this work: `bench`.

## Scope Control

- Do not implement the full project at once.
- Implement only the task explicitly requested in the current prompt.
- If a requested part is blocked by missing checkpoints, missing original code, or unsupported APIs, add a clear stub and document the blocker.

## Implementation Principles

- Preserve the original AutoGaze code as much as possible.
- Prefer wrappers, adapters, registries, and config-driven design over direct modification.
- Do not modify the original `INTEGRATION.md`.
- Do not hardcode SigLIP, NVILA, Qwen, dataset paths, frame counts, resolutions, or token budgets.
- Add tests for every new module.
- Keep CUDA, MPS, and CPU compatibility in mind.
- Do not claim encoder-side acceleration unless encoder computation is actually reduced.

## Original QUICK_START.md Policy

If a task involves real model loading, real inference, checkpoint paths, runtime arguments, or resolution scaling, read the original `QUICK_START.md` first.

Use:

- `INTEGRATION.md` for architecture and integration behavior.

- `QUICK_START.md` for real inference commands, checkpoint layout, runtime arguments, and resolution scaling behavior.

Do not modify the original `QUICK_START.md`.

If the implemented command or config differs from `QUICK_START.md`, document the difference and the reason.

## Inference Guide Policy

If a task involves inference commands, runtime arguments, query text, output paths, or resolution scaling, read the original `QUICK_START.md` first.

Create and maintain `docs/INFERENCE_GUIDE.md` as the user-facing inference guide.

`INFERENCE_GUIDE.md` should document:
- AutoGaze-only inference
- full pipeline inference
- query text-based MLLM inference
- dummy video inference
- local video inference
- Hugging Face-based inference, if supported
- resolution scaling behavior
- output and visualization paths
Do not claim that an inference mode is supported unless there is a runnable command or a clearly marked stub.