# AutoGaze Reproduction Runbook

This is the English companion to the Korean runbook. The Korean docs are the primary working docs for this branch:

- [INDEX_KO.md](INDEX_KO.md)
- [AUTOGAZE_REPRO_RUNBOOK_KO.md](AUTOGAZE_REPRO_RUNBOOK_KO.md)

Root `README.md`, `QUICK_START.md`, `TRAIN.md`, `INTEGRATION.md`, `LICENSE`, and upstream `autogaze/` materials are treated as official upstream files and are not edited by this branch.

## Current Scope

| Area | Status | Entry point |
| --- | --- | --- |
| NVILA-HD single inference | stable | `python -m repro.nvila_runner --mode single` |
| Visualization | stable | `--visualization-output-dir` |
| Basic HLVid benchmark | stable | `python scripts/run_hlvid_folder_benchmark.py` |
| Paper baseline comparison | ready | `--paper-baseline --paper-hd-autogaze` |
| Streaming/profile/H100 preflight | ready | `repro.nvila_runner`, `repro.hlvid_batch_benchmark` |
| Markdown/chart report | ready | `python -m repro.markdown_report` |
| Aggregate trend report | ready | `python -m repro.aggregate_reports` |
| Plugin experiments | PoC/probe | `repro.flexible_runner`, `repro.plugin_hlvid_benchmark` |

Basic HLVid benchmark and plugin HLVid benchmark are intentionally separate. Use the basic wrapper for NVILA-HD keep-all/autogaze, paper baseline, and H100 preflight. Use plugin benchmark only for Qwen/LongVILA/NVILA-Video extension experiments.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -U pip setuptools wheel
.venv/bin/python -m pip install -r requirements-repro.txt
bash scripts/bootstrap_official_repos.sh
.venv/bin/python -m pip install -e . --no-deps --no-build-isolation
```

On CUDA machines, place local model checkpoints under `weight/` when possible. On Apple MPS/macOS, use the editable install with `--no-deps --no-build-isolation` because some upstream optional dependencies are CUDA-specific.

## 1. NVILA-HD Single Inference

```bash
.venv/bin/python -m repro.nvila_runner \
  --mode single \
  --device cuda \
  --model-path nvidia/NVILA-8B-HD-Video \
  --autogaze-model nvidia/AutoGaze \
  --video inputs/hlvid_example/clip_av_video_5_001.mp4 \
  --prompt "Question: What is happening in the video? Please answer briefly." \
  --num-video-frames 128 \
  --num-video-frames-thumbnail 64 \
  --max-tiles-video 8 \
  --video-resize-longest-edge 720 \
  --gazing-mode autogaze \
  --measure-ttft \
  --warmup-runs 1 \
  --repeat-runs 3 \
  --print-summary \
  --summary-json outputs/autogaze_repro/nvila_single_summary.json \
  --output-json outputs/autogaze_repro/nvila_single.json
```

For the keep-all ablation, keep every other option fixed and change only:

```bash
--gazing-mode keep-all
```

## 2. Visualization

```bash
.venv/bin/python -m repro.nvila_runner \
  --mode single \
  --device cuda \
  --video inputs/hlvid_example/clip_av_video_5_001.mp4 \
  --num-video-frames 128 \
  --num-video-frames-thumbnail 64 \
  --max-tiles-video 8 \
  --video-resize-longest-edge 720 \
  --gazing-mode autogaze \
  --visualization-output-dir outputs/autogaze_repro/visualizations \
  --visualization-fps 4 \
  --visualization-selected-max-long-side 1280 \
  --output-json outputs/autogaze_repro/nvila_single_with_viz.json
```

Expected outputs include selected-frame video, processor-resolution video, AutoGaze overlay video, and `gazing_info.json`. In keep-all mode, selected-frame videos are still useful, but overlay is skipped.

## 3. Basic HLVid Benchmark

`--video-root` should point to the directory containing mp4 files.

```bash
.venv/bin/python scripts/run_hlvid_folder_benchmark.py \
  --dataset-dir /data/HLVid \
  --video-root /data/HLVid/videos \
  --output-dir outputs/autogaze_repro/hlvid_limit3_128f_720 \
  --limit 3 \
  --num-video-frames 128 \
  --num-video-frames-thumbnail 64 \
  --max-tiles-video 8 \
  --video-resize-longest-edge 720 \
  --measure-ttft \
  --continue-on-error
```

Main artifacts:

```text
hlvid_keep_all_predictions.jsonl
hlvid_autogaze_predictions.jsonl
hlvid_keep_all_summary.json
hlvid_autogaze_summary.json
hlvid_autogaze_gain_report.json
hlvid_autogaze_gain_report.csv
```

## 4. Paper Baseline Comparison

The AutoGaze paper baseline is treated as a separate `NVILA-8B-Video` model row, not as NVILA-HD keep-all.

```bash
.venv/bin/python scripts/run_hlvid_folder_benchmark.py \
  --dataset-dir /data/HLVid \
  --video-root /data/HLVid/videos \
  --output-dir outputs/autogaze_repro/hlvid_paper_comparison \
  --paper-baseline \
  --paper-hd-autogaze \
  --paper-comparison-report \
  --limit 3 \
  --continue-on-error
```

Paper references:

| Row | Reference |
| --- | --- |
| `NVILA-8B-Video` baseline | HLVid 42.5 |
| `NVILA-8B-HD-Video` AutoGaze | HLVid 52.6 |

## 5. Plugin HLVid Benchmark

Use this path for extension experiments only.

```bash
.venv/bin/python -m repro.plugin_hlvid_benchmark \
  --manifest /data/HLVid/manifest.json \
  --video-root /data/HLVid/videos \
  --output-dir outputs/autogaze_repro/plugin_hlvid_qwen_limit3 \
  --modes qwen_full_vit,qwen_chunked_vit,qwen_chunked_vit_autogaze_sparse \
  --model qwen3-vl=weight/Qwen3-VL-8B-Instruct \
  --limit 3 \
  --num-video-frames 32 \
  --num-video-frames-thumbnail 8 \
  --max-tiles-video 4 \
  --qwen-vit-chunk-frames 16 \
  --qwen-vit-max-spatial-chunks 4 \
  --qwen-thumbnail-mode append-video \
  --video-resize-longest-edge 448 \
  --max-new-tokens 8
```

## 6. Stream Profile / H100 Preflight

Stream profile checks decode, tiling, AutoGaze, and optional SigLIP without running the full LLM.

```bash
.venv/bin/python -m repro.nvila_runner \
  --mode stream-profile \
  --device cuda \
  --video /data/HLVid/videos/example.mp4 \
  --num-video-frames 128 \
  --num-video-frames-thumbnail 64 \
  --stream-chunk-frames 16 \
  --max-tiles-video 8 \
  --video-resize-longest-edge 720 \
  --gazing-mode autogaze \
  --stream-run-siglip \
  --stream-siglip-mode both \
  --stream-profile-json outputs/autogaze_repro/stream_profile.json
```

H100 preflight:

```bash
.venv/bin/python scripts/run_hlvid_folder_benchmark.py \
  --dataset-dir /data/HLVid \
  --video-root /data/HLVid/videos \
  --output-dir outputs/autogaze_repro/hlvid_preflight \
  --h100-preflight \
  --h100-budget-gib 70 \
  --allow-missing-videos
```

## 7. Markdown And Trend Reports

```bash
.venv/bin/python -m repro.markdown_report \
  --input-json outputs/autogaze_repro/hlvid_limit3_128f_720/hlvid_autogaze_gain_report.json \
  --output-md outputs/autogaze_repro/hlvid_limit3_128f_720/hlvid_autogaze_gain_report.md
```

```bash
.venv/bin/python -m repro.aggregate_reports \
  --input-root outputs/autogaze_repro \
  --output-dir outputs/autogaze_repro/trend_report
```

Aggregate outputs:

```text
aggregate_rows.csv
aggregate_summary.json
aggregate_report.md
assets/latency_by_config.svg
assets/token_reduction_by_config.svg
assets/memory_peak_by_config.svg
assets/status_by_config.svg
```

## Key Metrics

| Area | Fields to check first |
| --- | --- |
| Latency | `total_ms`, `video_decode_read_ms`, `preprocess_rest_without_decode_autogaze_ms`, `selector_input_build_ms`, `autogaze_total_ms`, `vision_input_build_ms`, `siglip_vision_ms`, `vision_encoder_ms`, `generate_ms`, `llm_forward_ms`, `ttft_ms` |
| Token/patch | full/off expected patch, multiscale candidate patch, selected patch, encoder input token, LLM visual token |
| Memory | processor/autogaze/vision/LLM/overall peak memory |
| Benchmark | `accuracy_total`, `accuracy_scored`, `failed`, `parse_failed`, `oom`, `skipped` |
| OOM | `failure.kind`, `failure.stage`, `failure.message` |

Primary latency accounting:

```text
total_ms = video_preprocess_without_autogaze_ms + autogaze_total_ms + generate_ms
```

Do not add legacy inclusive preprocess fields again when using this total.

`selector_input_build_ms` is derived from measured timers as `autogaze_total_ms - autogaze_model_forward_ms`. `vision_input_build_ms` is derived as `vision_encoder_ms - siglip_vision_ms - mm_projector_ms`. Treat both as residual breakdown fields, not extra total terms.

## Related Docs

- HLVid details: [AUTOGAZE_HLVID_BENCHMARK_GUIDE_KO.md](AUTOGAZE_HLVID_BENCHMARK_GUIDE_KO.md)
- Reporting details: [AUTOGAZE_REPORTING_GUIDE_KO.md](AUTOGAZE_REPORTING_GUIDE_KO.md)
- Plugin plan: [AUTOGAZE_PLUGIN_IMPLEMENTATION_PLAN_KO.md](AUTOGAZE_PLUGIN_IMPLEMENTATION_PLAN_KO.md)
- Selector/ViT/MLLM connection: [AUTOGAZE_SELECTOR_VIT_MLLM_CONNECTION_REPORT_KO.md](AUTOGAZE_SELECTOR_VIT_MLLM_CONNECTION_REPORT_KO.md)
- Timing validation: [AUTOGAZE_TIMING_VALIDATION_REPORT_KO.md](AUTOGAZE_TIMING_VALIDATION_REPORT_KO.md)
- Streaming/H100 configs: [STREAMING_PIPELINE_CONFIG_RECOMMENDATIONS_KO.md](STREAMING_PIPELINE_CONFIG_RECOMMENDATIONS_KO.md)
- CUDA result capture: [CUDA_RESULTS_TEMPLATE.md](CUDA_RESULTS_TEMPLATE.md)
