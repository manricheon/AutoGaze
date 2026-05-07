# AutoGaze 벤치마크 평가 가이드

> NVILA + AutoGaze를 표준 비디오 QA 벤치마크에서 평가하는 방법을 설명합니다.

---

## 목차

1. [개요](#1-개요)
2. [환경 설정](#2-환경-설정)
3. [벤치마크별 데이터 다운로드](#3-벤치마크별-데이터-다운로드)
4. [평가 실행](#4-평가-실행)
5. [결과 분석](#5-결과-분석)
6. [AutoGaze ON/OFF 비교](#6-autogaze-onoff-비교)
7. [전체 스윕 스크립트](#7-전체-스윕-스크립트)
8. [결과 해석 가이드](#8-결과-해석-가이드)

---

## 1. 개요

### 평가 스택

```
NVILA-8B-HD-Video (LLM + SigLIP ViT)
    └── NVILAProcessor
            └── AutoGaze (선택적 토큰 선택)
```

| 구성 요소 | 역할 |
|----------|------|
| **AutoGaze** | 비디오 프레임에서 중요 패치 예측 (weights/AutoGaze) |
| **NVILAProcessor** | AutoGaze를 포함한 멀티스케일 타일링 + 프리프로세싱 |
| **NVILA** | 시각-언어 추론 (weights/NVILA-8B-HD-Video) |
| **run_benchmark.py** | 벤치마크 루프, 메트릭 계산, JSON 저장 |

### 지원 벤치마크

| task 이름 | 전체 이름 | 샘플 수 | 답변 형식 |
|-----------|----------|--------|----------|
| `videomme` | VideoMME (자막 없음) | ~2,700 | A/B/C/D |
| `videomme_w_sub` | VideoMME (자막 포함) | ~2,700 | A/B/C/D |
| `mvbench` | MVBench | 4,000 | A/B/C/D |
| `nextqa` | NExT-QA (MC) | 4,996 | A/B/C/D/E |
| `egoschema` | EgoSchema | 5,031 | A/B/C/D/E |
| `mlvu` | MLVU | ~1,400 | A/B/C/D |
| `longvideobench` | LongVideoBench | ~1,337 | A/B/C/D |
| `hlvid` | HLVid | 268 | A/B/C/D |

---

## 2. 환경 설정

### 필수 패키지 설치

```bash
# 기본 의존성 (이미 설치된 경우 skip)
pip install av datasets transformers accelerate

# 가속 백엔드 (선택)
pip install flash-attn --no-build-isolation   # CUDA 환경 권장
```

### 모델 가중치 확인

```bash
ls weights/
# 반드시 다음 두 디렉토리가 있어야 합니다:
# ├── AutoGaze/                   (~50 MB)
# │   ├── config.json
# │   ├── model.safetensors
# │   └── preprocessor_config.json
# └── NVILA-8B-HD-Video/          (~16 GB)
#     ├── config.json
#     ├── model-0000X-of-00004.safetensors
#     └── processing_nvila.py
```

가중치가 없으면:
```bash
bash scripts/download_models.sh weights autogaze   # AutoGaze (~50 MB)
bash scripts/download_models.sh weights nvila      # NVILA (~16 GB)
```

---

## 3. 벤치마크별 데이터 다운로드

### 비디오 파일이 필요한 벤치마크 (HuggingFace에 bytes 포함)

대부분의 벤치마크는 `lmms-lab/*` 또는 해당 HuggingFace 레포지토리에 **비디오 bytes가 직접 저장**되어 있습니다.  
이 경우 별도 다운로드 없이 `--video-dir` 없이 바로 실행할 수 있습니다.

| 벤치마크 | 비디오 소스 | `--video-dir` 필요 여부 |
|---------|-----------|----------------------|
| VideoMME | `lmms-lab/Video-MME` HF bytes | ✗ 불필요 |
| MVBench | `OpenGVLab/MVBench` HF bytes | ✗ 불필요 |
| NExT-QA | `lmms-lab/NExTQA` HF bytes | ✗ 불필요 |
| EgoSchema | `lmms-lab/EgoSchema` HF bytes | ✗ 불필요 |
| MLVU | `MLVU/MLVU` HF bytes | ✗ 불필요 |
| LongVideoBench | `longvideobench/LongVideoBench` HF bytes | ✗ 불필요 |
| **HLVid** | 별도 다운로드 필요 | **✓ 필요** |

> **HuggingFace 캐시**: 첫 실행 시 `~/.cache/huggingface/datasets/`에 다운로드됩니다.  
> 용량이 클 수 있으니 `HF_DATASETS_CACHE` 환경변수로 경로를 변경할 수 있습니다.
> ```bash
> export HF_DATASETS_CACHE=/data/hf_cache
> ```

### HLVid 다운로드 (유일하게 별도 다운로드 필요)

### HLVid 다운로드

```bash
# 프로젝트 내 다운로드 스크립트 사용
bash scripts/download_hlvid.sh data/HLVid               # 전체 (~152 GB)
bash scripts/download_hlvid.sh --parts 1-4 data/HLVid   # 부분 다운로드
bash scripts/download_hlvid.sh --annotations-only data/HLVid  # QA만 (스모크 테스트용)
```

> **나머지 벤치마크**: `datasets` 라이브러리가 첫 실행 시 자동으로 HuggingFace에서 다운로드합니다.  
> 캐시 경로 변경: `export HF_DATASETS_CACHE=/data/hf_cache`

---

## 4. 평가 실행

### 기본 명령어

```bash
python -m autogaze.eval.run_benchmark \
    --task <TASK_NAME> \
    --video-dir <VIDEO_DIR> \
    [--output <OUTPUT_JSON>] \
    [--gazing-ratio 0.75] \
    [--num-frames 16] \
    [--no-autogaze]
```

### 전체 벤치마크 명령어

```bash
# ── VideoMME (자막 없음) — HF bytes, --video-dir 불필요 ──────────────────
python -m autogaze.eval.run_benchmark \
    --task videomme \
    --output results/videomme_ag075.json

# ── VideoMME (자막 포함) ──────────────────────────────────────────────────
python -m autogaze.eval.run_benchmark \
    --task videomme_w_sub \
    --output results/videomme_sub_ag075.json

# ── MVBench — HF bytes ───────────────────────────────────────────────────
python -m autogaze.eval.run_benchmark \
    --task mvbench \
    --output results/mvbench_ag075.json

# ── NExT-QA — HF bytes ───────────────────────────────────────────────────
python -m autogaze.eval.run_benchmark \
    --task nextqa \
    --output results/nextqa_ag075.json

# ── EgoSchema — HF bytes ─────────────────────────────────────────────────
python -m autogaze.eval.run_benchmark \
    --task egoschema \
    --output results/egoschema_ag075.json

# ── MLVU — HF bytes ──────────────────────────────────────────────────────
python -m autogaze.eval.run_benchmark \
    --task mlvu \
    --output results/mlvu_ag075.json

# ── LongVideoBench — HF bytes ────────────────────────────────────────────
python -m autogaze.eval.run_benchmark \
    --task longvideobench \
    --output results/longvideobench_ag075.json

# ── HLVid — 로컬 다운로드 필요 (--video-dir 필수) ─────────────────────────
bash scripts/download_hlvid.sh data/HLVid   # 먼저 실행
python -m autogaze.eval.run_benchmark \
    --task hlvid \
    --video-dir data/HLVid/videos \
    --output results/hlvid_ag075.json
```

### AutoGaze OFF 기준선

`--no-autogaze` 플래그를 추가하면 AutoGaze 없이 전체 패치를 사용합니다.

```bash
python -m autogaze.eval.run_benchmark \
    --task videomme \
    --no-autogaze \
    --output results/videomme_baseline.json
```

### 스모크 테스트 (50샘플)

```bash
python -m autogaze.eval.run_benchmark \
    --task videomme \
    --max-samples 50 \
    --output results/videomme_smoke.json
```

### 중단 후 재개

```bash
python -m autogaze.eval.run_benchmark \
    --task mvbench \
    --output results/mvbench_ag075.json \
    --resume    # 이미 처리된 샘플 skip
```

---

## 5. 결과 분석

### 출력 파일 구조

```json
{
  "task"         : "videomme",
  "autogaze"     : true,
  "gazing_ratio" : 0.75,
  "num_frames"   : 16,
  "n_skipped"    : 3,
  "metrics": {
    "overall_accuracy"    : 72.4,
    "n_total"             : 2700,
    "n_correct"           : 1955,
    "accuracy_by_duration": {
      "long"  : 68.1,
      "medium": 72.5,
      "short" : 76.6
    },
    "accuracy_by_category": {
      "Sports"        : 74.2,
      "Knowledge"     : 70.1,
      ...
    }
  },
  "per_sample": [
    {
      "sample_id"   : "q0001",
      "video_id"    : "4RYCC4HFmT8",
      "question"    : "What is happening?",
      "ground_truth": "A",
      "predicted"   : "A",
      "generated"   : "A",
      "correct"     : true,
      "latency_s"   : 2.34,
      "duration"    : "short",
      "category"    : "Sports"
    },
    ...
  ]
}
```

### Python에서 결과 로드

```python
import json

with open("results/videomme_ag075.json") as f:
    result = json.load(f)

print(f"Overall: {result['metrics']['overall_accuracy']:.1f}%")
print(f"By duration: {result['metrics'].get('accuracy_by_duration', {})}")

# 오답 분석
errors = [s for s in result["per_sample"] if not s["correct"]]
print(f"오답: {len(errors)}개")
```

---

## 6. AutoGaze ON/OFF 비교

두 결과 파일을 비교하는 유틸리티 스크립트:

```python
# compare_results.py
import json, sys

ag_file, base_file = sys.argv[1], sys.argv[2]

with open(ag_file)   as f: ag   = json.load(f)
with open(base_file) as f: base = json.load(f)

print(f"{'':30} {'AutoGaze ON':>12} {'OFF (baseline)':>15} {'차이':>8}")
print("-" * 67)
print(f"{'전체 정확도':30} {ag['metrics']['overall_accuracy']:>11.1f}% "
      f"{base['metrics']['overall_accuracy']:>14.1f}% "
      f"{ag['metrics']['overall_accuracy']-base['metrics']['overall_accuracy']:>+7.1f}%")

# Duration별 비교 (VideoMME)
for dur, val in ag["metrics"].get("accuracy_by_duration", {}).items():
    bval = base["metrics"].get("accuracy_by_duration", {}).get(dur, 0)
    print(f"  {dur:28} {val:>11.1f}% {bval:>14.1f}% {val-bval:>+7.1f}%")
```

```bash
python compare_results.py results/videomme_ag075.json results/videomme_baseline.json
```

---

## 7. 전체 스윕 스크립트

모든 벤치마크를 순차 실행:

```bash
#!/usr/bin/env bash
# scripts/run_all_benchmarks.sh
set -euo pipefail

GAZING_RATIO="${1:-0.75}"
RESULTS_DIR="results/$(date +%Y%m%d)_ag${GAZING_RATIO/./}"

mkdir -p "$RESULTS_DIR"

run_hf() {
    local task="$1"
    echo ""; echo "══ $task ════════════════════════════════════════"
    python -m autogaze.eval.run_benchmark \
        --task "$task" \
        --gazing-ratio "$GAZING_RATIO" \
        --output "$RESULTS_DIR/${task}_ag.json" \
        --resume
}

run() {
    local task="$1" vid_dir="$2"
    echo ""; echo "══ $task ════════════════════════════════════════"
    python -m autogaze.eval.run_benchmark \
        --task "$task" \
        --video-dir "$vid_dir" \
        --gazing-ratio "$GAZING_RATIO" \
        --output "$RESULTS_DIR/${task}_ag.json" \
        --resume
}

# HF bytes — no --video-dir needed
run_hf videomme
run_hf videomme_w_sub
run_hf mvbench
run_hf nextqa
run_hf egoschema
run_hf mlvu
run_hf longvideobench

# Local download required
run hlvid data/HLVid/videos

echo ""
echo "══ 결과 요약 ════════════════════════════════════════"
python - "$RESULTS_DIR" <<'EOF'
import json, glob, sys
from pathlib import Path

results_dir = sys.argv[1]
files = sorted(glob.glob(f"{results_dir}/*.json"))
print(f"{'Task':25} {'Accuracy':>10} {'N':>7} {'Ratio':>7}")
print("-" * 52)
for f in files:
    with open(f) as fp:
        r = json.load(fp)
    task  = r.get("task", Path(f).stem)
    acc   = r["metrics"].get("overall_accuracy", 0)
    n     = r["metrics"].get("n_total", 0)
    ratio = r.get("gazing_ratio", "—")
    print(f"{task:25} {acc:>9.2f}% {n:>7} {ratio:>7}")
EOF
```

```bash
bash scripts/run_all_benchmarks.sh 0.75
```

---

## 8. 결과 해석 가이드

### Gazing Ratio 선택

| Ratio | 처리 토큰 | 추천 시나리오 |
|-------|----------|--------------|
| 0.25  | 25%      | 속도 최우선 (짧은 비디오) |
| 0.50  | 50%      | 균형 |
| **0.75** | **75%** | **기본값** — NVILA 논문 기준 |
| 1.00  | 100%     | AutoGaze OFF 기준선 |

### 기대 성능 (NVILA 논문 Table 1 기준)

| 벤치마크 | AutoGaze OFF | AutoGaze ON (0.75) | 차이 |
|---------|-------------|-------------------|------|
| VideoMME (w/o sub) | ~72% | ~72% | ≈0 |
| VideoMME (w/ sub) | ~76% | ~76% | ≈0 |
| MVBench | ~76% | ~76% | ≈0 |
| EgoSchema | ~72% | ~72% | ≈0 |
| MLVU | ~68% | ~68% | ≈0 |

> AutoGaze는 **정확도를 유지하면서** KV cache와 ViT 연산을 줄이는 것이 목표입니다.  
> ratio=0.75에서 정확도 하락 없이 ViT 처리량을 25% 감소시킵니다.

### 주의사항

- **VideoMME 비디오 다운로드**: YouTube 정책상 일부 영상이 삭제되거나 지역 제한될 수 있음. 누락된 비디오는 skip되고 `n_skipped`에 기록됨.
- **EgoSchema**: Ego4D 접근 권한 신청 필요 (심사에 수일 소요).
- **MLVU / HLVid**: 긴 비디오 (5분~수십 분). `--num-frames 16`은 전체 영상에서 균등 샘플링.
- **GPU 메모리**: NVILA-8B는 bfloat16 기준 약 18 GB VRAM 필요. `device_map="auto"`로 자동 분산.

---

## 참고 파일

| 파일 | 역할 |
|------|------|
| `autogaze/eval/run_benchmark.py` | 평가 메인 스크립트 |
| `autogaze/eval/tasks.py` | 벤치마크 태스크 레지스트리 |
| `scripts/test_nvila.py` | 단일 비디오 추론 + 지연 시간 측정 |
| `scripts/download_hlvid.sh` | HLVid 데이터 다운로드 |
| `weights/AutoGaze/` | AutoGaze 모델 가중치 |
| `weights/NVILA-8B-HD-Video/` | NVILA 모델 가중치 |
