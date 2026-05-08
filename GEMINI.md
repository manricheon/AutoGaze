# AutoGaze: AI Collaboration Guide (AI 협업 가이드)

This document serves as a foundational guide for AI agents and human developers collaborating on the AutoGaze project. It outlines the project's architecture, key workflows, and constraints for maintaining system integrity.

이 문서는 AutoGaze 프로젝트에서 협업하는 AI 에이전트와 개발자를 위한 기초 가이드입니다. 프로젝트 아키텍처, 주요 워크플로우, 시스템 무결성 유지를 위한 제약 사항을 설명합니다.

---

## 1. Project Overview (프로젝트 개요)

**AutoGaze (Autoregressive Gazing)** is a model that selects informative patches in videos to optimize downstream processing for Vision Transformers (ViTs) and MLLMs.

**AutoGaze (자기회귀적 응시)**는 비디오에서 정보가 많은 패치를 선택하여 Vision Transformer(ViT) 및 MLLM의 후속 처리를 최적화하는 모델입니다.

### Key Capabilities (주요 기능)
- **Efficient Video Understanding**: Processes high-resolution, long-form videos by attending only to 10–25% of patches.
- **Modular Design**: Easy to integrate new models, tasks, and algorithms.
- **Two-Stage Training**: NTP pre-training followed by GRPO RL for strategy refinement.

---

## 2. Architecture at a Glance (아키텍처 요약)

| Directory | Description |
| :--- | :--- |
| `autogaze/models/` | Gaze model definitions (AutoGaze, etc.) |
| `autogaze/tasks/` | Training objectives (VideoMAE reconstruction, etc.) |
| `autogaze/algorithms/` | Training logic (NTP, GRPO RL) |
| `autogaze/vision_encoders/` | Customized encoders (SigLIP, etc.) compatible with AutoGaze |
| `autogaze/datasets/` | Video loading and preprocessing utilities |
| `autogaze/eval/` | Benchmark runners, task configs, evaluation loop |
| `configs/` | YAML configurations for all components |
| `scripts/` | Entry points for training, inference, data download, and benchmarks |

---

## 3. Core Workflows (핵심 워크플로우)

### Model Download (모델 다운로드)
```bash
# AutoGaze + VideoMAE (기본, 항상 필요)
bash scripts/download_models.sh

# MLLM 벤치마크 전체 (nvila, vjepa2, qwen25vl, qwen25)
bash scripts/download_models.sh weights mllm

# CV 태스크 소형 모델
bash scripts/download_models.sh weights cv
```

### Dataset Download (데이터셋 다운로드)
```bash
# 평가 데이터셋 (HF-bytes 전체, ~158 GB)
bash scripts/download_data_eval.sh data/eval hf_bytes

# 학습 데이터
bash scripts/download_data.sh data/AutoGaze-Training-Data
```

### Inference (추론)
```bash
# 가이즈 맵 시각화
python -m autogaze.infer assets/example_input.mp4 --gazing-ratio 0.75

# 전체 MLLM 파이프라인
python autogaze/infer_full.py assets/example_input.mp4 --mllm nvila
```

See `autogaze/infer.py`, `autogaze/infer_full.py`, and `notebooks/12_inference_full_ko.ipynb`.

### Benchmark (벤치마크)
```bash
# 기준선 (AutoGaze OFF)
python -m autogaze.eval.run_benchmark \
    --task videomme --mllm nvila --no-autogaze \
    --hf-data-dir data/eval/Video-MME

# AutoGaze ON vs OFF 자동 비교
bash scripts/run_benchmarks.sh \
    --tasks videomme,mvbench --hf-data-dir data/eval
```

Supported runners: `nvila`, `qwen25vl`, `qwen25vl_full`, `vjepa2_llm`, `nvila_vjepa2`.

### Training (학습)
Training is orchestrated via `autogaze/train.py` using Hydra configs.

- **Stage 1 (NTP)**: `scripts/train_ntp_multi_gpu.sh`
- **Stage 2 (RL)**: `scripts/train_rl_multi_gpu.sh`

### Integration (통합)
To make a ViT compatible, wrap it with `mask_with_gazing` and update the attention mask logic. See `docs/integration_guide.md` and `docs/benchmark_guide.md`.

---

## 4. AI Interaction Guidelines (AI 상호작용 지침)

1. **Preserve Originality (원본 유지)**: Avoid modifying core logic in `autogaze/models/` or `autogaze/algorithms/` unless explicitly requested. Prefer creating new subclasses or tasks.
2. **Config-First (설정 우선)**: Use and extend YAML files in `configs/` instead of hardcoding parameters in Python.
3. **Validation (검증)**: Always verify changes by running existing scripts or checking notebooks.
4. **Documentation (문서화)**: Keep `GEMINI.md` and `docs/guide_ko.md` updated with any significant structural changes.

---

## 5. Reference Documentation (참조 문서)

| Document | Description |
| :--- | :--- |
| `README.md` | General overview and installation |
| `QUICK_START.md` | Basic and advanced Python usage examples |
| `TRAIN.md` | Detailed training instructions and parameter explanation |
| `docs/guide_ko.md` | Comprehensive Korean guide (models, datasets, benchmarks, training) |
| `docs/eval_guide.md` | Video QA benchmark guide (tasks, runners, dataset prep, CLI flags) |
| `docs/inference_guide.md` | Master guide for all inference workflows |
| `docs/integration_guide.md` | Detailed guide for integrating AutoGaze into new ViTs/MLLMs |
| `docs/benchmark_guide.md` | ViT/MLLM benchmark guide (Hook vs Full mode, CV tasks) |
