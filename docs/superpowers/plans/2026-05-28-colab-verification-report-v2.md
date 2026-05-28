# Colab Verification Report V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete `docs/COLAB_VERIFICATION_REPORT_V2_KO.md` so it verifies AutoGaze on/off single inference and benchmark paths for NVILA-HD, Qwen, and V-JEPA2+Qwen, with 16-frame visual artifacts recorded in the report.

**Architecture:** Keep CUDA execution in the existing notebook and runner scripts, but make their outputs audit-friendly. Add normalization to the report builder so NVILA runner JSON, Qwen plugin/flexible-runner JSON, and V-JEPA2+Qwen JSON are summarized through one V2 report. Add lightweight visualization helpers where a runner already has sampled frames or an AutoGaze sparse plan, and keep missing CUDA artifacts explicit instead of pretending they exist locally.

**Tech Stack:** Python, pytest, PIL, existing `repro.nvila_runner`, `repro.flexible_runner`, `repro.vjepa_qwen_runner`, `scripts/run_hlvid_folder_benchmark.py`, `scripts/write_external_cuda_verification_notebook.py`.

---

### Task 1: V2 Report Normalization

**Files:**
- Modify: `repro/colab_verification_report.py`
- Test: `tests/test_colab_verification_report.py`

- [ ] Add tests showing NVILA single JSON normalizes `summary.answer`, `summary.tokens`, `summary.latency_ms`, `summary.memory_bytes`, and nested `result.visualization`.
- [ ] Add tests showing Qwen plugin run JSON normalizes `generation.text`, `generation.metrics.tokens`, `generation.metrics.latency_ms`, `generation.metrics.memory_bytes`, and direct AutoGaze sparse plan artifact paths.
- [ ] Add a V2 evidence matrix section with pipeline, single status, benchmark status, AutoGaze applicability, token reduction, and artifact status.
- [ ] Keep the existing V1 renderer behavior compatible.

### Task 2: 16-Frame Visualization Defaults

**Files:**
- Modify: `repro/vjepa_qwen_runner.py`
- Modify: `scripts/run_colab_autogaze_cuda_smoke.py`
- Modify: `scripts/write_external_cuda_verification_notebook.py`
- Test: `tests/test_vjepa_qwen_runner.py`
- Test: `tests/test_colab_autogaze_cuda_smoke_script.py`
- Test: `tests/test_external_cuda_verification_notebook.py`

- [ ] Set V-JEPA2+Qwen visualization max frames to 16 by default.
- [ ] Ensure the Colab/Kaggle smoke wrapper passes `--visualization-max-frames 16`.
- [ ] Ensure the external verification notebook passes NVILA visualization flags for keep-all-single and AutoGaze runs.
- [ ] Ensure the notebook records Qwen sparse plan artifacts and report paths.

### Task 3: V2 Static Report

**Files:**
- Create: `docs/COLAB_VERIFICATION_REPORT_V2_KO.md`
- Test: `tests/test_colab_verification_report.py`

- [ ] Write a Korean V2 report that separates verified CUDA evidence from local static/report generation evidence.
- [ ] Include NVILA-HD single on/off, NVILA HLVid mini, Qwen single/plugin or plugin benchmark, Qwen plugin HLVid mini, and V-JEPA2+Qwen single on/off.
- [ ] Include visualization artifact sections for selected frames, AutoGaze overlay, processor/resize-space overlays, and V-JEPA token masks.
- [ ] Explicitly mark any still-missing CUDA artifact as `needs_rerun` with the exact rerun cell/command.

### Task 4: Verification

**Files:**
- Test: focused pytest and entrypoint verifier.

- [ ] Run focused tests for the changed report/notebook/visualization files.
- [ ] Run `scripts/verify_autogaze_entrypoints.py`.
- [ ] Run `git diff --check`.
- [ ] Confirm upstream official files remain untouched.
