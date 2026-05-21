# CUDA AutoGaze / HLVid Results Template

Use this template when collecting H100/CUDA results for leader-facing comparison. The Korean runbook and reporting guide are the source of truth for execution flow:

- [AUTOGAZE_REPRO_RUNBOOK_KO.md](AUTOGAZE_REPRO_RUNBOOK_KO.md)
- [AUTOGAZE_HLVID_BENCHMARK_GUIDE_KO.md](AUTOGAZE_HLVID_BENCHMARK_GUIDE_KO.md)
- [AUTOGAZE_REPORTING_GUIDE_KO.md](AUTOGAZE_REPORTING_GUIDE_KO.md)

## 1. Environment

| Item | Value |
| --- | --- |
| Date |  |
| Host |  |
| GPU |  |
| GPU memory |  |
| Driver |  |
| CUDA |  |
| PyTorch |  |
| Transformers |  |
| Branch / commit |  |
| AutoGaze checkpoint |  |
| NVILA-HD checkpoint |  |
| NVILA-8B-Video checkpoint |  |
| Qwen/LongVILA checkpoint |  |

## 2. Input / Config

| Item | Value |
| --- | --- |
| Dataset root |  |
| Video root |  |
| Manifest |  |
| Limit / split |  |
| Video count available |  |
| Frames |  |
| Thumbnail frames |  |
| Max tiles video |  |
| Resize policy |  |
| AutoGaze target scales |  |
| AutoGaze patch size |  |
| Gazing ratio |  |
| Max batch size AutoGaze |  |
| Max batch size SigLIP |  |
| Warmup / repeat |  |

## 3. Single Inference Smoke

| Mode | Status | Answer | Total ms | Preprocess ms | AutoGaze ms | ViT ms | LLM/generate ms | TTFT ms | Peak GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| keep-all |  |  |  |  | n/a |  |  |  |  |
| AutoGaze |  |  |  |  |  |  |  |  |  |

Notes:

- Video:
- Prompt:
- Output JSON:
- Markdown report:
- Visualization directory:

## 4. Token / Patch Accounting

| Mode | Full/off patch | Multiscale candidate patch | Selected patch | Encoder input token | LLM visual token | Patch reduction | LLM token reduction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| keep-all |  |  |  |  |  |  |  |
| AutoGaze |  |  |  |  |  |  |  |
| paper baseline |  | n/a | n/a |  |  | n/a |  |

Interpretation:

- Full/off denominator:
- Thumbnail handling:
- TokenShuffle/projector behavior:
- Any metric fields that were null:

## 5. Basic HLVid Benchmark

| Mode | Samples attempted | Scored | Correct | Accuracy total | Accuracy scored | Failed | OOM | Parse failed | Skipped |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| keep-all |  |  |  |  |  |  |  |  |  |
| AutoGaze |  |  |  |  |  |  |  |  |  |

Artifacts:

- Output dir:
- `hlvid_keep_all_summary.json`:
- `hlvid_autogaze_summary.json`:
- `hlvid_autogaze_gain_report.json`:
- Markdown report:

## 6. Paper Baseline Comparison

| Row | Model | Paper reference | Local measured | Delta | Failed | OOM | Parse failed | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NVILA-8B-Video baseline |  | 42.5 |  |  |  |  |  | AutoGaze not applicable |
| NVILA-8B-HD-Video AutoGaze |  | 52.6 |  |  |  |  |  | HD AutoGaze |
| NVILA-HD keep-all optional |  | n/a |  |  |  |  |  | ablation only |

Artifacts:

- `hlvid_paper_comparison_report.json`:
- Markdown report:

## 7. Plugin Experiments

| Mode | Model | Token selector | ViT path | MLLM path | Status | Accuracy | Total ms | Selected patch | LLM visual token | Peak GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen_full_vit |  | off | full | Qwen |  |  |  | n/a |  |  |
| qwen_chunked_vit |  | off | chunked | Qwen |  |  |  | n/a |  |  |
| qwen_chunked_vit_autogaze_sparse |  | AutoGaze | sparse/chunked | Qwen |  |  |  |  |  |  |

Artifacts:

- Plugin output dir:
- Summary:
- Markdown report:

## 8. OOM / Failure Log

| Run | Kind | Stage | Message summary | Config | Expected tokens | Selected tokens | Action |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  |

Typical stages:

- `video_decode`
- `video_preprocess`
- `autogaze`
- `vision_encoder`
- `llm_prefill`
- `generate`

## 9. Aggregate Trend

Artifacts:

- `aggregate_rows.csv`:
- `aggregate_summary.json`:
- `aggregate_report.md`:
- `assets/latency_by_config.svg`:
- `assets/token_reduction_by_config.svg`:
- `assets/memory_peak_by_config.svg`:
- `assets/status_by_config.svg`:

High-level conclusion:

```text
Across the tested configs, AutoGaze reduced encoder patch tokens by __x median and
LLM visual tokens by __x median. The best successful config was __.
The main remaining bottleneck is __.
The main OOM risk appears at __.
```

## 10. Decision Notes

- Best current config:
- Configs to avoid:
- Evidence for token/compute reduction:
- Evidence for latency gain or loss:
- Evidence for memory gain or OOM avoidance:
- Accuracy risk:
- Next experiment:
