# AutoGaze Plugin Runner 구현 계획

이 문서는 `repro.nvila_runner`를 대체하기 위한 문서가 아니라, 여러 token selector / vision encoder / MLLM 조합을 같은 실험 규격으로 붙이기 위한 확장 계획입니다. NVILA-HD 재현과 리더 설득용 결과는 안정 경로인 `repro.nvila_runner`와 `scripts/run_hlvid_folder_benchmark.py`를 우선 사용합니다.

긴 이전 구현 기록은 `docs/temp/AUTOGAZE_PLUGIN_IMPLEMENTATION_PLAN_KO_FULL_2026-05-21.md`에 보존했습니다.

## Runner 역할 분리

| 용도 | 엔트리포인트 | 상태 |
| --- | --- | --- |
| NVILA-HD single/visualization/profile | `python -m repro.nvila_runner` | 안정 경로 |
| 기본 HLVid benchmark | `python scripts/run_hlvid_folder_benchmark.py` / `python -m repro.hlvid_batch_benchmark` | 안정 경로 |
| Plugin single/probe | `python -m repro.flexible_runner` | 확장 실험 |
| Plugin HLVid benchmark | `python -m repro.plugin_hlvid_benchmark` | 확장 실험 |
| Markdown/chart report | `python -m repro.markdown_report` | 공통 |
| 폴더 단위 trend report | `python -m repro.aggregate_reports` | 공통 |

핵심 구분은 다음과 같습니다.

| 구분 | 목적 | AutoGaze 의미 |
| --- | --- | --- |
| 기본 NVILA-HD | 논문 재현/프로파일링/HLVid benchmark | NVILA-HD processor 내부 AutoGaze |
| Paper baseline | `NVILA-8B-Video` row 재현 | not applicable |
| Plugin runner | 다른 MLLM에 pre-ViT selector를 붙일 수 있는지 검증 | 외부 selector 또는 sparse plan |

## Adapter 축

확장 러너는 세 축을 분리합니다.

```text
video
  -> token_selector
       none | keep_all | autogaze | external_mask | pixelprune_reference
  -> vision_encoder
       nvila_hd_siglip | nvila_video_vision | longvila_vision | qwen_vit | llava_onevision | internvl
  -> mllm
       nvila_hd | nvila_video | longvila | qwen | llava_onevision | internvl
  -> task/scoring/report
```

각 adapter는 다음 정보를 공통으로 남겨야 합니다.

| Adapter | 필수 metadata |
| --- | --- |
| `TokenSelectorAdapter` | 입력 frame/tile/patch 수, 선택 patch 수, scale별 선택 수, 적용 불가 사유 |
| `VisionEncoderAdapter` | 입력 grid, patch size, position encoding 처리, encoder 입력 token 수, latency/memory |
| `MllmAdapter` | visual token 수, prompt token 수, prefill/generate 경계, answer text, failure stage |
| `TaskAdapter` | `video_path`, `question`, `answer`, `choices`, scoring result |

## 현재 지원 상태

| 조합 | 상태 | 목적 |
| --- | --- | --- |
| NVILA-HD + AutoGaze | 안정 | 기준 재현, HLVid, latency/token/memory profile |
| NVILA-HD keep-all | 안정 | AutoGaze off ablation, OOM 비교 |
| NVILA-8B-Video paper baseline | 준비 | 논문 table baseline 후보, AutoGaze not applicable |
| NVILA-Video plugin | probe 중심 | 별도 model family에 selector를 붙이는 실험 준비 |
| LongVILA plugin | probe 중심 | 긴 video MLLM 확장성 검토 |
| Qwen3-VL plugin | PoC | `full_vit`, `chunked_vit`, `chunked_vit_autogaze_sparse` 비교 |
| Qwen2/2.5-VL | adapter/probe | processor/video packing 차이 검토 필요 |
| LLaVA-OneVision | adapter/probe | SigLIP 계열 vision path 연결 검토 |
| InternVL3 | adapter/probe | dynamic tiling과 selector 결합 검토 |

## Integration Level

| Level | 설명 | 계산량 감소 가능성 |
| --- | --- | --- |
| `off` | selector 없이 모델 기본 processor 사용 | 없음 |
| `post_encoder_prune` | ViT 이후 visual token 일부 제거 | LLM 비용 감소, ViT 비용 감소 없음 |
| `dense_mask` | patch grid는 유지하고 mask/zero-fill 적용 | 구현 쉬움, ViT 비용 감소 제한적 |
| `pre_encoder_sparse` | 선택 patch만 ViT에 넣도록 repack | ViT와 LLM 모두 감소 가능, 모델별 position/grid 처리 필요 |
| `native_processor` | 모델 processor 내부 기능에 selector 연결 | 안정적이나 모델별 커스텀 필요 |

현재 목표는 `pre_encoder_sparse`를 Qwen 계열에서 먼저 검증하고, 이후 LongVILA/NVILA-Video로 확장하는 것입니다.

## Plugin HLVid 실행 예

```bash
.venv/bin/python -m repro.plugin_hlvid_benchmark \
  --manifest /data/HLVid/manifest.json \
  --video-root /data/HLVid/videos \
  --limit 3 \
  --modes qwen_full_vit,qwen_chunked_vit,qwen_chunked_vit_autogaze_sparse \
  --model qwen3-vl=weights/Qwen3-VL-8B-Instruct \
  --video-resize-shortest-edge 720 \
  --num-video-frames 128 \
  --num-video-frames-thumbnail 64 \
  --qwen-vit-chunk-frames 16 \
  --qwen-vit-max-spatial-chunks 4 \
  --qwen-thumbnail-mode append-video \
  --output-dir outputs/autogaze_repro/plugin_hlvid_qwen_limit3
```

이 경로는 확장성 검증용입니다. NVILA-HD 논문 재현 결과와 섞어 부르지 않습니다.

## 리포트 요구사항

Plugin runner도 기본 runner와 같은 형태의 핵심 필드를 남깁니다.

| 영역 | 필드 예 |
| --- | --- |
| identity | `model_family`, `token_selector`, `vision_encoder`, `mllm`, `integration_level` |
| video | 원본 해상도, resize 해상도, frame 수, thumbnail 수, tile 수 |
| patch/token | full/off 예상 patch, multiscale patch budget, selected patch, encoder input token, LLM visual token |
| latency | preprocess, selector, vision encoder, LLM/generate, total |
| memory | selector peak, vision peak, LLM peak, overall peak |
| benchmark | accuracy, failed, oom, parse_failed, skipped |

## 남은 구현/검증

1. Qwen `chunked_vit_autogaze_sparse`가 실제 ViT 입력 token 수를 줄이는지 CUDA smoke로 확인.
2. Qwen MRoPE/grid metadata가 sparse patch repack 이후에도 답변 품질을 망가뜨리지 않는지 확인.
3. LongVILA/NVILA-Video에서 processor의 video packing 구조를 읽어 adapter metadata를 고정.
4. HLVid 외 VideoQA task adapter를 `video_path`, `question`, `answer`, `choices` schema로 일반화.
5. Plugin benchmark 결과를 `markdown_report`와 `aggregate_reports`에서 기본 benchmark와 같은 표/그래프로 읽을 수 있게 유지.
