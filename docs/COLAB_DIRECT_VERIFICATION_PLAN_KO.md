# Colab 직접 검증 및 리포트 가이드

이 문서는 Colab에서 AutoGaze on/off 실험을 검증할 때 무엇을 실제 검증 경로로 볼지, 그리고 `colab_verification.md`를 어떤 방식으로 만드는지 정리합니다.

## 핵심 결정

`scripts/run_colab_autogaze_cuda_smoke.py`는 필수 실행 경로가 아닙니다. 이 스크립트는 Colab에서 여러 명령을 한 번에 실행하기 위한 편의 wrapper입니다.

실제 검증 대상은 Linux/CUDA 머신에서 사용하던 기존 runner와 benchmark script입니다.

| 구분 | 역할 |
| --- | --- |
| authoritative path | `repro.nvila_runner`, `scripts/run_hlvid_folder_benchmark.py`, `repro.plugin_hlvid_benchmark`, `repro.vjepa_qwen_runner`, `repro.vjepa_qwen_hlvid_benchmark` |
| optional helper | `scripts/run_colab_autogaze_cuda_smoke.py` |
| report artifact | 기존 runner들이 만든 JSON/MD/visualization을 모아 `colab_verification.md` 생성 |

따라서 Colab 검증 문서는 wrapper 중심이 아니라, 기존 script를 직접 실행하고 결과 JSON을 모아 검증 report를 만드는 흐름을 기본값으로 둡니다.

## 목표

- Colab에서도 Linux CUDA/H100에서 쓰는 명령과 같은 runner를 직접 실행합니다.
- AutoGaze off/on 전후 결과를 모델군별로 비교합니다.
- text query에 대한 generated answer를 결과 표에 포함합니다.
- token, latency, memory, failure stage, visualization artifact를 한 문서에 모읍니다.
- wrapper는 빠른 smoke용 편의 기능으로만 유지합니다.

## Non-goals

- Colab wrapper를 기존 runner의 대체 경로로 만들지 않습니다.
- V-JEPA + Qwen PoC 출력으로 정확도 성능을 주장하지 않습니다. 현재 bridge는 zero-shot wiring probe입니다.
- HLVid full benchmark를 Colab 무료 GPU에서 반드시 완료한다고 가정하지 않습니다. Colab은 smoke/limit 검증, H100은 full run 기준입니다.

## 직접 실행 순서

### 1. 환경과 weight 준비

```bash
python scripts/download_vjepa_qwen_checkpoints.py \
  --output-root /content/autogaze_weights \
  --max-workers 4

python scripts/download_hlvid_example_video.py \
  --output inputs/hlvid_example/clip_av_video_5_001.mp4
```

NVILA/Qwen plugin 실험용 weight는 기존 CUDA 머신과 같은 로컬 path를 사용합니다.

### 2. 사전 엔트리포인트 검증

```bash
python scripts/verify_autogaze_entrypoints.py \
  --output-json /content/autogaze_vjepa_outputs/entrypoint_verification.json \
  --output-md /content/autogaze_vjepa_outputs/entrypoint_verification.md
```

이 단계는 모델 weight를 로드하지 않고 CLI drift, wrapper route, dry-run plan, token accounting을 확인합니다.

### 3. NVILA-HD single 직접 실행

AutoGaze on:

```bash
python -m repro.nvila_runner \
  --mode single \
  --video inputs/hlvid_example/clip_av_video_5_001.mp4 \
  --model-path /content/autogaze_weights/NVILA-8B-HD-Video \
  --autogaze-model /content/autogaze_weights/nvidia__AutoGaze \
  --gazing-mode autogaze \
  --device cuda \
  --dtype float16 \
  --num-video-frames 16 \
  --num-video-frames-thumbnail 0 \
  --video-decode-strategy seek \
  --video-resize-longest-edge 224 \
  --max-tiles-video 1 \
  --max-new-tokens 32 \
  --visualization-output-dir /content/autogaze_vjepa_outputs/nvila_visualizations \
  --output-json /content/autogaze_vjepa_outputs/nvila_autogaze_single.json \
  --summary-json /content/autogaze_vjepa_outputs/nvila_autogaze_single_summary.json
```

AutoGaze off:

```bash
python -m repro.nvila_runner \
  --mode single \
  --video inputs/hlvid_example/clip_av_video_5_001.mp4 \
  --model-path /content/autogaze_weights/NVILA-8B-HD-Video \
  --gazing-mode keep-all-single \
  --device cuda \
  --dtype float16 \
  --num-video-frames 16 \
  --num-video-frames-thumbnail 0 \
  --video-decode-strategy seek \
  --video-resize-longest-edge 224 \
  --max-tiles-video 1 \
  --max-new-tokens 32 \
  --visualization-output-dir /content/autogaze_vjepa_outputs/nvila_visualizations \
  --output-json /content/autogaze_vjepa_outputs/nvila_keep_all_single.json \
  --summary-json /content/autogaze_vjepa_outputs/nvila_keep_all_single_summary.json
```

### 4. Qwen plugin HLVid route 직접 실행

```bash
python scripts/run_hlvid_folder_benchmark.py \
  --manifest /content/HLVid/test.jsonl \
  --video-root /content/HLVid/videos \
  --output-dir /content/autogaze_vjepa_outputs/qwen_plugin_hlvid_limit3 \
  --plugin-suite qwen \
  --plugin-model qwen3-vl=/content/autogaze_weights/Qwen3-VL \
  --limit 3 \
  --continue-on-error \
  --num-video-frames 16 \
  --num-video-frames-thumbnail 0 \
  --max-tiles-video 1 \
  --max-new-tokens 32 \
  --video-decode-strategy seek \
  --video-resize-longest-edge 224 \
  --autogaze-model /content/autogaze_weights/nvidia__AutoGaze \
  --autogaze-device cuda \
  --autogaze-dtype float16 \
  --autogaze-target-scales 32+64+112+224 \
  --autogaze-target-patch-size 16 \
  --max-batch-size-autogaze 4
```

이 route는 기본적으로 다음 세 모드를 비교합니다.

- `qwen_full_vit`
- `qwen_chunked_vit`
- `qwen_chunked_vit_autogaze_sparse`

### 5. V-JEPA + Qwen dense/off 직접 실행

```bash
python -m repro.vjepa_qwen_runner \
  --video inputs/hlvid_example/clip_av_video_5_001.mp4 \
  --prompt "Describe the video in one short sentence." \
  --autogaze-mode off \
  --vjepa-model /content/autogaze_weights/facebook__vjepa2-vitl-fpc64-256 \
  --qwen-model /content/autogaze_weights/Qwen__Qwen2.5-VL-3B-Instruct \
  --require-cuda \
  --device cuda \
  --dtype float16 \
  --num-video-frames 16 \
  --frames-per-clip 16 \
  --video-decode-strategy seek \
  --video-resize-longest-edge 224 \
  --max-new-tokens 32 \
  --visualization-output-dir /content/autogaze_vjepa_outputs/visualizations \
  --output-json /content/autogaze_vjepa_outputs/vjepa_qwen_dense_off.json \
  --output-md /content/autogaze_vjepa_outputs/vjepa_qwen_dense_off.md
```

### 6. AutoGaze + V-JEPA + Qwen on 직접 실행

```bash
python -m repro.vjepa_qwen_runner \
  --video inputs/hlvid_example/clip_av_video_5_001.mp4 \
  --prompt "Describe the video in one short sentence." \
  --autogaze-mode on \
  --autogaze-model /content/autogaze_weights/nvidia__AutoGaze \
  --vjepa-model /content/autogaze_weights/facebook__vjepa2-vitl-fpc64-256 \
  --qwen-model /content/autogaze_weights/Qwen__Qwen2.5-VL-3B-Instruct \
  --require-cuda \
  --device cuda \
  --dtype float16 \
  --num-video-frames 16 \
  --frames-per-clip 16 \
  --autogaze-chunk-frames 16 \
  --max-tiles-video 1 \
  --max-batch-size-autogaze 4 \
  --autogaze-tile-size 224 \
  --autogaze-target-scales 32+64+112+224 \
  --autogaze-target-patch-size 16 \
  --video-decode-strategy seek \
  --video-resize-longest-edge 224 \
  --max-new-tokens 32 \
  --visualization-output-dir /content/autogaze_vjepa_outputs/visualizations \
  --output-json /content/autogaze_vjepa_outputs/autogaze_vjepa_qwen_on.json \
  --output-md /content/autogaze_vjepa_outputs/autogaze_vjepa_qwen_on.md
```

## `colab_verification.md` 생성 방식

리포트 생성기는 기존 runner 결과 JSON을 입력으로 받습니다. wrapper 없이 직접 실행한 결과도 같은 형식으로 보고할 수 있습니다.

기본 CLI:

```bash
python -m repro.colab_verification_report \
  --output-md /content/autogaze_vjepa_outputs/colab_verification.md \
  --title "AutoGaze Colab Verification" \
  --video inputs/hlvid_example/clip_av_video_5_001.mp4 \
  --query "Describe the video in one short sentence." \
  --entrypoint-verification-json /content/autogaze_vjepa_outputs/entrypoint_verification.json \
  --case nvila_keep_all_single=/content/autogaze_vjepa_outputs/nvila_keep_all_single.json \
  --case nvila_autogaze=/content/autogaze_vjepa_outputs/nvila_autogaze_single.json \
  --case vjepa_qwen_dense_off=/content/autogaze_vjepa_outputs/vjepa_qwen_dense_off.json \
  --case autogaze_vjepa_qwen_on=/content/autogaze_vjepa_outputs/autogaze_vjepa_qwen_on.json
```

wrapper는 내부적으로 같은 report generator를 호출하는 편의 경로로 유지합니다.

## 리포트 내용

`colab_verification.md`에는 다음 섹션을 둡니다.

| 섹션 | 내용 |
| --- | --- |
| Environment | Colab GPU, commit, repo path, weight root, output root |
| Query / Video | video path, prompt/query, frame config |
| Case Summary | case별 status, answer, total latency, peak memory, token 수 |
| Answers | AutoGaze off/on generated text를 나란히 비교 |
| Token Comparison | full/off 후보, selected patch, encoder input token, MLLM visual token |
| Latency Comparison | decode/resize, selector, ViT, bridge/projector, generate, total |
| Memory Comparison | case별 CUDA peak 및 stage peak |
| Visualizations | selected frames, AutoGaze overlay, V-JEPA token mask |
| Entrypoint Verification | verifier summary와 verified script ids |
| Failures | OOM, missing dependency, mapping failure, parse failure stage |

## 구현 구조

### 1. Report generator

- 모듈: `repro/colab_verification_report.py`
- 역할:
  - 여러 JSON 결과를 읽습니다.
  - runner별 필드 차이를 normalize합니다.
  - Markdown report를 생성합니다.
  - 이미지 artifact path가 있으면 Markdown 이미지로 렌더링합니다.
  - 없으면 `not recorded`로 표시합니다.

### 2. Wrapper

- `scripts/run_colab_autogaze_cuda_smoke.py`는 직접 Markdown 문자열을 만들지 않습니다.
- smoke 실행 후 `repro.colab_verification_report`의 renderer를 호출합니다.
- 문서에서는 optional helper로만 설명합니다.

### 3. 문서 정리

- `docs/AUTOGAZE_VJEPA_POC_KO.md`
  - wrapper-first 표현을 직접 실행-first로 수정합니다.
  - wrapper는 “한 번에 확인할 때의 선택지”로 이동합니다.
- `docs/AUTOGAZE_REPRO_RUNBOOK_KO.md`
  - Colab 검증은 기존 runner 직접 실행 흐름으로 정리합니다.
- `docs/INDEX_KO.md`
  - 이 문서를 Colab 검증 계획 문서로 연결합니다.

### 4. 테스트

필수 테스트:

```bash
.venv/bin/python -m pytest \
  tests/test_colab_verification_report.py \
  tests/test_colab_autogaze_cuda_smoke_script.py \
  tests/test_vjepa_qwen_runner.py \
  tests/test_verify_autogaze_entrypoints.py \
  -q
```

전체 회귀:

```bash
.venv/bin/python scripts/verify_autogaze_entrypoints.py --run-pytest
.venv/bin/python -m pytest -q
```

### 5. Colab 실제 확인

GPU 연결 가능 시 다음 순서로 확인합니다.

1. `verify_autogaze_entrypoints.py`
2. `repro.vjepa_qwen_runner --autogaze-mode off`
3. `repro.vjepa_qwen_runner --autogaze-mode on`
4. `repro.colab_verification_report`
5. `colab_verification.md`와 PNG artifact 화면 캡처

## Acceptance

- wrapper 없이도 직접 실행 결과 JSON으로 `colab_verification.md`를 만들 수 있습니다.
- wrapper는 같은 generator를 호출하는 optional helper입니다.
- V-JEPA + Qwen dense/off와 AutoGaze/on의 answer, latency, token, memory, visualization이 한 문서에 표시됩니다.
- NVILA/Qwen direct 결과 JSON도 같은 report에 추가할 수 있습니다.
- missing artifact나 failed case가 있어도 report 생성은 실패하지 않고 failure section에 남깁니다.
- 공식/upstream 문서는 수정하지 않습니다.
