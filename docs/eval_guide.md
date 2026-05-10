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

Naming convention: **`{vit}_{lm}`** (ViT first).  Use `--integration` to select the mode.

**Integration mode support matrix**:

| `--mllm` | ViT | LLM | `native` | `hook` | `full` | Extra flags |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| `nvila` | SigLIP (custom) | NVILA | ✅ default | ✅ | — | `--autogaze-path` required |
| `siglip_qwen25` | SigLIP | Qwen2.5-VL | — | ✅ default | ✅ | |
| `vjepa2_nvila` | V-JEPA2 | NVILA | — | ✅ | ✅ default | `--vjepa2-path` required |
| `vjepa2_qwen25` | V-JEPA2 | Qwen2.5-7B | — | ✅ | ✅ default | `--vjepa2-path`, `--lm-path` |
| `vjepa2` | V-JEPA2 | — | — | ✅ default | ✅ | feature extraction only |
| `siglip` | SigLIP (HF) | — | — | ✅ | — | feature extraction only |

- **native**: AutoGaze fully baked into the model processor — deepest integration, best efficiency, NVILA-specific.
- **hook**: Gaze mask zeroes non-selected tokens via a forward hook/method patch — zero-shot, easy to add to any model, no latency benefit (sequence length unchanged).
- **full**: Tokens physically removed inside the ViT forward pass — real latency/VRAM reduction, requires ViT modification.

**Deprecated aliases** (still work, emit warning): `nvila_vjepa2` → `vjepa2_nvila`, `qwen25vl` → `siglip_qwen25`, `qwen25vl_full` → `siglip_qwen25 --integration full`, `vjepa2_llm` → `vjepa2_qwen25`.

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

> **Note (NVILA native)**: the NVILA processor loads the AutoGaze config on `__init__`, so `--autogaze-path` is required even with `--no-autogaze`.  The flag forces `gazing_ratio=1.0` (all patches), which is equivalent to AutoGaze OFF.

```bash
python -m autogaze.eval.run_benchmark \
    --task videomme \
    --mllm nvila \
    --model-path weights/NVILA-8B-HD-Video \
    --autogaze-path weights/AutoGaze \
    --no-autogaze \
    --output results/videomme_nvila_baseline.json
```

### Qwen2.5-VL hook runner (zero-shot)

```bash
python -m autogaze.eval.run_benchmark \
    --task mvbench \
    --mllm siglip_qwen25 \
    --model-path weights/Qwen2.5-VL-7B-Instruct \
    --autogaze-path weights/AutoGaze \
    --gazing-ratio 0.75
```

### Qwen2.5-VL full integration (higher efficiency)

```bash
python -m autogaze.eval.run_benchmark \
    --task mvbench \
    --mllm siglip_qwen25 \
    --model-path weights/Qwen2.5-VL-7B-Instruct \
    --autogaze-path weights/AutoGaze \
    --integration full \
    --gazing-ratio 0.75
```

### V-JEPA2 + NVILA LLM runner

```bash
python -m autogaze.eval.run_benchmark \
    --task videomme \
    --mllm vjepa2_nvila \
    --model-path weights/NVILA-8B-HD-Video \
    --vjepa2-path weights/vjepa2-vitl-fpc64-256 \
    --autogaze-path weights/AutoGaze \
    --integration full \
    --gazing-ratio 0.75
```

### V-JEPA2 + Qwen2.5-7B LLM runner

```bash
python -m autogaze.eval.run_benchmark \
    --task videomme \
    --mllm vjepa2_qwen25 \
    --model-path weights/NVILA-8B-HD-Video \
    --vjepa2-path weights/vjepa2-vitl-fpc64-256 \
    --lm-path weights/Qwen2.5-7B-Instruct \
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
| `--mllm` | `nvila` | Runner key — `{vit}_{lm}` convention (see Section 1) |
| `--integration` | runner default | Override integration mode: `native`, `hook`, or `full` |
| `--model-path` | `weights/NVILA-8B-HD-Video` | Primary model weights path (ViT or ViT+LLM) |
| `--vjepa2-path` | `None` | V-JEPA2 encoder weights (required for `vjepa2_nvila`) |
| `--lm-path` | `None` | LLM weights (required for `vjepa2_qwen25`) |
| `--projector-path` | `None` | Trained ViT→LLM projector (optional for LLM runners) |
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

**`vjepa2_nvila` runner fails to load** (previously `nvila_vjepa2`)
This runner requires both `--model-path` (NVILA) and `--vjepa2-path`. If using `load_runner()` in Python, pass `vjepa2_path=` as a keyword argument.

**`--no-autogaze` with `--mllm nvila` (native)**
NVILA's processor reads the AutoGaze config on init even for the baseline, so `--autogaze-path` must always be provided.  When you pass `--no-autogaze`, the benchmark automatically keeps the path and forces `--gazing-ratio 1.0` (all patches), which is equivalent to AutoGaze OFF.  You must still supply `--autogaze-path weights/AutoGaze`.
For hook mode (`--integration hook`), `--no-autogaze` works normally (`autogaze_path=None` → no masking).

**Deprecation warnings for old runner keys**
Keys `nvila_vjepa2`, `qwen25vl`, `qwen25vl_full`, `vjepa2_llm`, `vjepa2_full` still work but emit a `DeprecationWarning`. Update scripts to use the new `{vit}_{lm}` keys.

**Metrics section shows `avg_latency_ms` / `avg_peak_vram_mb` as 0 or missing**
VRAM metrics require a CUDA GPU. On CPU/MPS, `peak_vram_mb` is omitted per sample and the aggregate is not computed. `n_tokens_visual` is an estimate; exact counts require overriding `n_visual_tokens()` in the runner class.

---

## 7. Video Action Recognition CV Tasks (동작 인식 CV 태스크)

Beyond the VQA benchmarks above, `scripts/run_cv_tasks.py` supports **video action recognition** as a CV-level comparison task — showing how AutoGaze token selection affects classification outputs frame-by-frame.

### Supported action recognition tasks (`--tasks`)

| Task key | Model | Dataset | Mode |
| :--- | :--- | :--- | :--- |
| `videomae_cls` | `MCG-NJU/videomae-base-finetuned-kinetics` | Kinetics-400 (400 classes) | supervised classification |
| `xclip` | `microsoft/xclip-base-patch32` | text-guided, zero-shot | zero-shot with custom labels |

### How AutoGaze is applied

**VideoMAE-CLS** — The gaze mask (14×14 spatial) is broadcast across all 8 temporal positions (16 frames / tubelet_size 2 = 1568 tokens), then zeroed via a forward hook on `model.videomae.embeddings`:

```text
 AutoGaze gaze mask (14×14)
         │
         │  broadcast to 8 temporal positions
         ▼
 token mask (8 × 196 = 1568)
         │
         │  hook on VideoMAEEmbeddings output
         ▼
 non-gaze tokens zeroed → VideoMAE-CLS inference
```

**X-CLIP** — The spatial gaze mask is applied per-frame inside the shared CLIP vision encoder.  Since all frames share the same spatial ViT, one (196,) mask applies to every frame:

```text
 AutoGaze gaze mask (196,)
         │
         │  broadcast over B×T frames
         ▼
 hook on CLIPVisionEmbeddings output (CLS kept, spatial zeroed)
         │
         ▼
 X-CLIP text-video similarity → top-K labels
```

### Quick start

```bash
# Image mode — run videomae_cls and xclip on a single frame
python scripts/run_cv_tasks.py \
    --input assets/example.jpg \
    --ag-path weights/AutoGaze \
    --tasks videomae_cls xclip \
    --ratios 0.75 0.5 0.25

# Video mode — action recognition per chunk across the full video
python scripts/run_cv_tasks.py \
    --input assets/example.mp4 \
    --ag-path weights/AutoGaze \
    --tasks videomae_cls xclip \
    --ag-ratio 0.5 \
    --temporal-window 16
```

In **image mode** the single frame is repeated to fill the model's required temporal window (16 frames for VideoMAE, 8 for X-CLIP).  In **video mode** the action is classified once per temporal chunk and the label is overlaid on every frame in that chunk.

### Customizing X-CLIP labels

Pass custom action descriptions via Python:

```python
from scripts.run_cv_tasks import run_xclip, _load_ag

ag_model, ag_proc, _ = _load_ag("weights/AutoGaze", "cuda")
# ag_video = prep_for_autogaze(pil_img, ag_proc, "cuda")
top, ratio_top, metrics = run_xclip(
    [pil_img] * 8, ag_video, ag_model, [0.75, 0.5],
    device="cuda",
    texts=["a goalkeeper saving a penalty", "a crowd cheering"],
)
```

---

## 8. Key Source Files (주요 소스 파일)

| Purpose | Path |
| :--- | :--- |
| Benchmark entry point | `autogaze/eval/run_benchmark.py` |
| Task definitions | `autogaze/eval/tasks.py` |
| MLLM runner registry | `autogaze/eval/models.py` |
| CV task comparison script | `scripts/run_cv_tasks.py` |
| Video extraction script | `scripts/extract_hf_videos.py` |
| HLVid download script | `scripts/download_hlvid.sh` |
| Master benchmark script | `scripts/run_benchmarks.sh` |
| Inference notebook | `notebooks/12_inference_full_ko.ipynb` |
