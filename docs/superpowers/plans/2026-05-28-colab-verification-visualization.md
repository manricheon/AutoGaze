# Colab Verification Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Colab-ready verification artifacts for AutoGaze off/on V-JEPA+Qwen runs, including visualizations, answers, latency, memory, token summaries, and a generated `colab_verification.md`.

**Architecture:** Keep the actual model runner responsible for per-case visual artifacts because it already has sampled frames, sparse plans, and V-JEPA token selections in memory. Keep the Colab wrapper responsible for combining dense/off and AutoGaze/on JSON outputs into one verification markdown report.

**Tech Stack:** Python, PIL, pytest, existing `repro.vjepa_qwen_runner`, existing `scripts/run_colab_autogaze_cuda_smoke.py`.

---

### Task 1: Runner Visualization Artifacts

**Files:**
- Modify: `repro/vjepa_qwen_runner.py`
- Test: `tests/test_vjepa_qwen_runner.py`

- [ ] Add failing tests for selected-frame grid and AutoGaze overlay artifact metadata.
- [ ] Add `--visualization-output-dir` and `--visualization-max-frames` CLI flags.
- [ ] Save selected-frame grid, V-JEPA token mask, and AutoGaze patch overlay PNGs when requested.
- [ ] Record artifact paths in result JSON and markdown.

### Task 2: Colab Verification Markdown

**Files:**
- Modify: `scripts/run_colab_autogaze_cuda_smoke.py`
- Test: `tests/test_colab_autogaze_cuda_smoke_script.py`

- [ ] Add failing tests for combined markdown report content and artifact links.
- [ ] Add `--verification-md` CLI flag with default `<output-root>/colab_verification.md`.
- [ ] Generate a compact report with environment, query/video, off/on answers, token/latency/memory tables, and visual artifacts.

### Task 3: Verification

**Files:**
- Test: focused pytest files plus full suite.

- [ ] Run focused tests for the changed files.
- [ ] Run entrypoint verifier.
- [ ] Run full pytest.
- [ ] Re-run Colab CUDA smoke and inspect `colab_verification.md` output.
