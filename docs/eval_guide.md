# Video QA Benchmark Guide (비디오 QA 벤치마크 가이드)

This guide explains how to evaluate **AutoGaze + MLLM** pipelines on standard video QA benchmarks, including dataset preparation, running comparisons, and interpreting results.

이 가이드는 표준 비디오 QA 벤치마크에서 **AutoGaze + MLLM** 파이프라인을 평가하는 방법을 설명합니다 — 데이터셋 준비, 비교 실행, 결과 해석 포함.

---

## 1. Supported Tasks & Runners

### Tasks (--task)

| Task name | Dataset (HF repo) | Video source | Split |
| :--- | :--- | :--- | :--- |
| `videomme` | `lmms-lab/Video-MME` | HF bytes | test |
| `videomme_w_sub` | `lmms-lab/Video-MME` | HF bytes | test |
| `mvbench` | `OpenGVLab/MVBench` | HF bytes | test |
| `nextqa` | `lmms-lab/NExTQA` | HF bytes | val |
| `egoschema` | `lmms-lab/EgoSchema` | HF bytes | test |
| `mlvu` | `MLVU/MLVU` | HF bytes | test |
| `longvideobench` | `longvideobench/LongVideoBench` | HF bytes | val |
| `hlvid` | `bfshi/HLVid` | local `--video-dir` | test |

Tasks marked "HF bytes" embed video data directly in the dataset parquet — no `--video-dir` is needed unless the bytes cache is incomplete (see [Section 3](#3-dataset-preparation)).

### MLLM Runners (--mllm)

| `--mllm` | Model | AutoGaze integration |
| :--- | :--- | :--- |
| `nvila` | NVILA-8B-HD-Video | processor-integrated (full) |
| `qwen25vl` | Qwen2.5-VL-7B | zero-shot token selector |
| `qwen25vl_full` | Qwen2.5-VL-7B | zero-shot token selector (full video) |
| `vjepa2_llm` | V-JEPA2 ViT + Qwen2.5-7B LLM | zero-shot token selector |
| `nvila_vjepa2` | V-JEPA2 ViT + NVILA LLM | zero-shot token selector |

---

## 2. Quick Start (빠른 시작)

### Automated comparison (AutoGaze ON vs OFF)

```bash
# VideoMME — NVILA, both modes, first 100 samples
bash scripts/run_benchmarks.sh --tasks videomme --max-samples 100

# Multiple tasks at once
bash scripts/run_benchmarks.sh --tasks videomme,mvbench,nextqa --max-samples 200
```

### Manual — AutoGaze ON

```bash
python -m autogaze.eval.run_benchmark \
    --task videomme \
    --mllm nvila \
    --model-path weights/NVILA-8B-HD-Video \
    --autogaze-path weights/AutoGaze \
    --gazing-ratio 0.75 \
    --output results/videomme_nvila_ag075.json
```

### Manual — AutoGaze OFF (baseline)

```bash
python -m autogaze.eval.run_benchmark \
    --task videomme \
    --mllm nvila \
    --model-path weights/NVILA-8B-HD-Video \
    --no-autogaze \
    --output results/videomme_nvila_baseline.json
```

### Qwen2.5-VL zero-shot runner

```bash
python -m autogaze.eval.run_benchmark \
    --task mvbench \
    --mllm qwen25vl \
    --model-path weights/Qwen2.5-VL-7B-Instruct \
    --autogaze-path weights/AutoGaze \
    --gazing-ratio 0.75
```

### V-JEPA2 + NVILA LLM runner

```bash
python -m autogaze.eval.run_benchmark \
    --task videomme \
    --mllm nvila_vjepa2 \
    --model-path weights/NVILA-8B-HD-Video \
    --vjepa2-path weights/vjepa2-vitl-fpc64-256 \
    --autogaze-path weights/AutoGaze \
    --gazing-ratio 0.75
```

---

## 3. Dataset Preparation (데이터셋 준비)

### 3-A. Default: rely on HF cache (권장)

For HF-bytes tasks (`videomme`, `mvbench`, etc.) the benchmark calls `load_dataset` at runtime and streams bytes from the HF hub cache. This works out of the box if you have internet access and enough disk space.

If you see `Video bytes missing … — skipping` warnings for many samples, it means the HF cache is incomplete. Use Option B or C below.

### 3-B. Download dataset repo locally then run offline

Download the full dataset repository once with `huggingface-cli`. This fetches all parquet shards including the binary video data.

```bash
# Download (run once)
huggingface-cli download lmms-lab/Video-MME            --repo-type dataset --local-dir data/Video-MME
huggingface-cli download OpenGVLab/MVBench             --repo-type dataset --local-dir data/MVBench
huggingface-cli download lmms-lab/NExTQA               --repo-type dataset --local-dir data/NExTQA
huggingface-cli download lmms-lab/EgoSchema            --repo-type dataset --local-dir data/EgoSchema
huggingface-cli download MLVU/MLVU                     --repo-type dataset --local-dir data/MLVU
huggingface-cli download longvideobench/LongVideoBench --repo-type dataset --local-dir data/LongVideoBench
```

Then point the benchmark at the local directory with `--hf-data-dir`:

```bash
python -m autogaze.eval.run_benchmark \
    --task videomme \
    --hf-data-dir data/Video-MME \
    --mllm nvila \
    --model-path weights/NVILA-8B-HD-Video \
    --autogaze-path weights/AutoGaze
```

`--hf-data-dir` overrides the remote HF repo ID for `load_dataset`, so the benchmark runs fully offline.

### 3-C. Extract videos to individual mp4 files

If bytes are still `None` after a full repo download (e.g. the dataset stores videos as external binary blobs), extract them to individual files and pass `--video-dir` as a fallback.

```bash
# Extract (one-time)
python scripts/extract_hf_videos.py \
    --task videomme \
    --hf-data-dir data/Video-MME \
    --out data/videomme_videos

# Run benchmark — bytes are preferred; --video-dir is used when bytes are missing
python -m autogaze.eval.run_benchmark \
    --task videomme \
    --hf-data-dir data/Video-MME \
    --video-dir data/videomme_videos \
    --mllm nvila \
    --model-path weights/NVILA-8B-HD-Video \
    --autogaze-path weights/AutoGaze
```

The script reports how many samples were saved and how many still had no bytes (those will need a separate manual download if any).

### 3-D. HLVid (no embedded bytes)

HLVid stores no video bytes in the HF dataset. You must download the videos locally first:

```bash
bash scripts/download_hlvid.sh data/HLVid

python -m autogaze.eval.run_benchmark \
    --task hlvid \
    --video-dir data/HLVid \
    --mllm nvila \
    --model-path weights/NVILA-8B-HD-Video \
    --autogaze-path weights/AutoGaze
```

---

## 4. All CLI Flags (전체 플래그 목록)

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--task` | required | Benchmark task name (see Section 1) |
| `--mllm` | `nvila` | MLLM runner (see Section 1) |
| `--model-path` | `weights/NVILA-8B-HD-Video` | Path to MLLM weights or HF repo ID |
| `--autogaze-path` | `weights/AutoGaze` | Path to AutoGaze weights |
| `--no-autogaze` | off | Disable AutoGaze (full-patch baseline) |
| `--gazing-ratio` | `0.75` | Fraction of patches to keep (0–1) |
| `--num-frames` | `16` | Frames sampled per video |
| `--max-new-tokens` | `16` | Max generation tokens (MCQ needs very few) |
| `--video-dir` | `None` | Local video directory (required for hlvid; fallback for others) |
| `--hf-data-dir` | `None` | Local pre-downloaded HF dataset repo (overrides remote repo ID) |
| `--output` | auto | Output JSON path (`results/{task}_{mllm}_{tag}.json`) |
| `--max-samples` | `None` | Cap dataset size (for smoke tests) |
| `--resume` | off | Skip samples already written in `--output` |
| `--log-level` | `INFO` | Logging verbosity |

---

## 5. Measuring the AutoGaze Effect (효과 측정 지표)

| Metric | Goal | Notes |
| :--- | :--- | :--- |
| **Accuracy (%)** | Maintain | Should stay within ~0.5 pp of baseline |
| **Latency (ms/frame)** | Reduce | Full integration gives 2–4× speedup |
| **VRAM (GB)** | Reduce | Fewer tokens → smaller KV cache |

### Expected accuracy (paper reference)

| Benchmark | AutoGaze OFF | AutoGaze ON (r=0.75) | Delta |
| :--- | :---: | :---: | :---: |
| VideoMME | ~72.0% | ~72.1% | +0.1 |
| MVBench | ~76.2% | ~76.0% | −0.2 |
| EgoSchema | ~72.5% | ~72.5% | 0.0 |

### Gazing ratio guide

| Scenario | Recommended ratio |
| :--- | :---: |
| Real-time streaming (latency-first) | 0.25–0.40 |
| Balanced (speed + quality) | 0.50–0.65 |
| Quality-first (long video) | 0.70–0.80 |
| Baseline (no AutoGaze effect) | 1.00 |

---

## 6. Troubleshooting (문제 해결)

**`Video bytes missing for … and no --video-dir provided — skipping`**
The HF bytes column returned `None` for some samples. Fix: download the dataset locally (Section 3-B) and use `--hf-data-dir`, or extract videos to files (Section 3-C) and use `--video-dir`.

**`Video not found: … — skipping`**
A sample's video file could not be located under `--video-dir`. Check that filenames match the `video_id` (or the HF path field) and the expected extension (`.mp4` by default).

**Out of Memory (OOM)**
Reduce `--num-frames`. AutoGaze ON uses far fewer tokens than the baseline, so the baseline run is typically the memory bottleneck.

**Accuracy drop > 2%**
Verify that `--model-path` and `--autogaze-path` point to the correct weight directories, and that `--gazing-ratio` is not set too low (< 0.25).

**`AssertionError` in processing_nvila.py (NVILA runner)**
The NVILA processor requires a `<video>` token in the prompt. This is handled automatically by `NVILARunner.run()` — if you see this error, ensure you are using the runner via `load_runner()` rather than calling the processor directly.

**`nvila_vjepa2` runner fails to load**
This runner requires both `--model-path` (NVILA) and `--vjepa2-path`. If using `load_runner()` in Python, pass `vjepa2_path=` as a keyword argument.

---

## 7. Key Source Files (주요 소스 파일)

| Purpose | Path |
| :--- | :--- |
| Benchmark entry point | `autogaze/eval/run_benchmark.py` |
| Task definitions | `autogaze/eval/tasks.py` |
| MLLM runner registry | `autogaze/eval/models.py` |
| Video extraction script | `scripts/extract_hf_videos.py` |
| HLVid download script | `scripts/download_hlvid.sh` |
| Master benchmark script | `scripts/run_benchmarks.sh` |
| Inference notebook | `notebooks/12_inference_full_ko.ipynb` |
