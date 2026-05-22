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
| `TaskAdapter` | VideoQA `video_path/question/answer/choices`, caption `references`, action `label/choices`, scoring result |

## 현재 지원 상태

| 조합 | 상태 | 목적 |
| --- | --- | --- |
| NVILA-HD + AutoGaze | 안정 | 기준 재현, HLVid, latency/token/memory profile |
| NVILA-HD keep-all | 안정 | AutoGaze off ablation, OOM 비교 |
| NVILA-8B-Video paper baseline | 준비 | 논문 table baseline 후보, AutoGaze not applicable |
| NVILA-Video plugin | off/sidecar | 별도 VILA family model에 selector를 붙이는 실험 준비. sidecar mode는 dense generation과 AutoGaze selector metric을 같이 기록 |
| LongVILA plugin | off/sidecar | 긴 video MLLM 확장성 검토. sidecar mode는 아직 LongVILA 내부 token pruning을 적용하지 않음 |
| Qwen3-VL plugin | 구현/검증 대상 | `full_vit`, `chunked_vit`, `chunked_vit_autogaze_sparse` 비교. sparse mode는 direct AutoGaze selector plan을 사용 |
| Qwen2/2.5-VL | adapter/probe | processor/video packing 차이 검토 필요 |
| LLaVA-OneVision | off/prune 실험 | AutoGaze on 요청 시 dense fallback을 막고, 선택 시 post-encoder visual token prune-generate를 시도 |
| InternVL3 | off/sidecar | dynamic tiling과 `num_patches_list` mapping probe 중심. sidecar mode는 dense generation과 selector metric을 같이 기록 |

## Integration Level

| Level | 설명 | 계산량 감소 가능성 |
| --- | --- | --- |
| `off` | selector 없이 모델 기본 processor 사용 | 없음 |
| `post_encoder_prune` | ViT 이후 visual token 일부 제거 | LLM 비용 감소, ViT 비용 감소 없음 |
| `dense_mask` | patch grid는 유지하고 mask/zero-fill 적용 | 구현 쉬움, ViT 비용 감소 제한적 |
| `pre_encoder_sparse` | 선택 patch만 ViT에 넣도록 repack | ViT와 LLM 모두 감소 가능, 모델별 position/grid 처리 필요 |
| `native_processor` | 모델 processor 내부 기능에 selector 연결 | 안정적이나 모델별 커스텀 필요 |

현재 목표는 `pre_encoder_sparse`를 Qwen 계열에서 먼저 검증하고, 이후 LongVILA/NVILA-Video/LLaVA/InternVL로 확장하는 것입니다. 상태 표기는 세 단계로 나눕니다.

| 상태 | 의미 |
| --- | --- |
| actual prune/sparse | encoder 입력 token 또는 LLM visual token을 실제로 줄인 경로 |
| sidecar generate | dense model generation은 그대로 실행하고 AutoGaze selector 결과/latency/token만 sidecar로 함께 기록한 경로 |
| probe_required | 모델별 visual packing hook이 아직 없어 pruning을 적용하지 않고 필요한 mapping 정보만 기록하는 경로 |

Qwen2.5/Qwen3의 `*_chunked_vit_autogaze_sparse`는 actual sparse 검증 대상입니다. LLaVA-OneVision `llava-onevision-autogaze-actual`는 post-encoder visual token prune-generate 실험 경로입니다. VILA-family의 `*-autogaze-actual` entry는 현재 external CLI dense generation에 AutoGaze selector sidecar metric을 붙인 단계라서 아직 모델 내부 compute gain을 주장하지 않습니다. InternVL3 sidecar mode도 selector를 무시하지 않지만 pruning은 적용하지 않습니다.

## Pre-ViT Sparse 적용 우선순위

`pre_encoder_sparse`는 AutoGaze 또는 다른 token selector가 고른 patch/tile/frame만 ViT에 넣는 경로입니다. 따라서 성공하면 ViT encoder latency/memory와 MLLM visual context가 동시에 줄어듭니다. 현재 공통 계약은 `repro.plugins.pre_vit_sparse`에 있으며, `repro.flexible_runner` inspect 결과의 `pre_vit_sparse_contract`와 `pre_vit_sparse_model_matrix`에 노출됩니다.

| 우선순위 | 모델/계열 | 상태 | 난이도 | 먼저 확인할 hook |
| --- | --- | --- | --- | --- |
| 1 | Qwen2.5-VL | `implemented_pending_cuda` | low | `pixel_values_videos`, `video_grid_thw`, `spatial_merge_size` |
| 2 | Qwen3-VL | `implemented_pending_cuda` | low | `pixel_values_videos`, `video_grid_thw`, `spatial_merge_size` |
| 3 | NVILA-Video plugin | `in_process_probe_required` | medium_high | processor output, `vision_tower.forward`, `mm_projector` |
| 4 | LongVILA | `in_process_probe_required` | medium_high | processor output, `vision_tower.forward`, `mm_projector` |
| 5 | InternVL3 | `dynamic_tile_probe_required` | medium | dynamic tile order, `num_patches_list` |
| 6 | LLaVA-OneVision | `candidate_design_required` | high | frame/tile before SigLIP pooling |

Qwen은 이미 `SparseSelectionPlan -> Qwen visual index -> chunked ViT sparse feature -> MLLM visual placeholder` 흐름을 코드에 둔 상태라 CUDA smoke가 다음 검증입니다. VILA-family는 external CLI만으로는 pre-ViT hook을 주입하기 어렵기 때문에, `repro.vila_feature_probe`의 `pre_vit_sparse_probe`가 요구 hook과 위치 정렬 위험을 먼저 기록합니다. InternVL3는 `repro.internvl_dynamic_tile_probe`가 `num_patches_list`와 dynamic tile 정책을 정리합니다. LLaVA-OneVision은 patch-level보다 frame/tile-level pre-ViT candidate를 먼저 검토합니다.

### Probe command 예시

```bash
.venv/bin/python -m repro.vila_feature_probe \
  --model-path weight/LongVILA \
  --model-family longvila \
  --video /data/HLVid/videos/example.mp4 \
  --num-video-frames 128 \
  --max-tiles-video 4 \
  --output-json outputs/autogaze_repro/longvila_pre_vit_probe.json
```

```bash
.venv/bin/python -m repro.internvl_dynamic_tile_probe \
  --model-path weight/InternVL3 \
  --model-family internvl3 \
  --video /data/HLVid/videos/example.mp4 \
  --num-video-frames 32 \
  --max-tiles-video 8 \
  --output-json outputs/autogaze_repro/internvl3_dynamic_tile_probe.json
```

```bash
.venv/bin/python -m repro.flexible_runner \
  --mode inspect \
  --model-family llava-onevision \
  --model-path weight/LLaVA-OneVision \
  --token-selector-adapter autogaze \
  --vision-encoder-adapter llava-onevision-siglip \
  --mllm-adapter llava-onevision \
  --autogaze-integration-level pre_encoder_sparse \
  --pre-encoder-prune-adapter autogaze-sparse \
  --output-json outputs/autogaze_repro/llava_pre_vit_candidate.json
```

## Plugin HLVid Mode 그룹

| 그룹 | modes | 의미 |
| --- | --- | --- |
| Qwen comparison | `qwen2.5_full_vit`, `qwen2.5_chunked_vit`, `qwen2.5_chunked_vit_autogaze_sparse`, `qwen3_full_vit`, `qwen3_chunked_vit`, `qwen3_chunked_vit_autogaze_sparse` | 같은 비디오/질문에서 full ViT, chunked ViT, AutoGaze sparse ViT를 Qwen2.5/Qwen3 각각 비교 |
| VILA-family | `nvila-video-off`, `nvila-video-autogaze-actual`, `longvila-off`, `longvila-autogaze-actual` | off는 external VILA CLI 경로, actual entry는 dense generation + AutoGaze selector metric 기록 |
| Other MLLM | `llava-onevision-off`, `llava-onevision-autogaze-actual`, `internvl3-off`, `internvl3-autogaze-sidecar-generate` | LLaVA는 post-encoder prune generate 실험, InternVL3는 sidecar generate |

`probe_required`는 AutoGaze selector 요청을 무시하지 않았다는 뜻입니다. 아직 실제 pruning 적용 성공은 아니므로 report에서 `executed`와 분리해서 봐야 합니다.
`executed_dense_with_autogaze_sidecar`는 모델 답변 생성은 수행했지만 visual pruning은 적용하지 않았다는 뜻입니다. `visual_pruning_applied=false`, `vision_encoder_latency_reduced=false`, `mllm_context_reduced=false`를 같이 확인하세요.

## Plugin HLVid 실행 예

```bash
.venv/bin/python -m repro.plugin_hlvid_benchmark \
  --manifest /data/HLVid/manifest.json \
  --video-root /data/HLVid/videos \
  --limit 3 \
  --modes qwen2.5_full_vit,qwen2.5_chunked_vit,qwen2.5_chunked_vit_autogaze_sparse,qwen3_full_vit,qwen3_chunked_vit,qwen3_chunked_vit_autogaze_sparse \
  --model qwen2.5-vl=weights/Qwen2.5-VL-7B-Instruct \
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

VILA/LLaVA/InternVL 확장 mode까지 같은 입력 row에서 같이 보려면 기본 config를 사용합니다.

```bash
.venv/bin/python -m repro.plugin_hlvid_benchmark \
  --manifest /data/HLVid/manifest.json \
  --video-root /data/HLVid/videos \
  --limit 3 \
  --modes nvila-video-off,nvila-video-autogaze-actual,longvila-off,longvila-autogaze-actual,llava-onevision-off,llava-onevision-autogaze-actual,internvl3-off,internvl3-autogaze-sidecar-generate,qwen2.5_full_vit,qwen2.5_chunked_vit,qwen2.5_chunked_vit_autogaze_sparse,qwen3_full_vit,qwen3_chunked_vit,qwen3_chunked_vit_autogaze_sparse \
  --model nvila-video=weight/NVILA-8B-Video \
  --model longvila=weight/LongVILA \
  --model llava-onevision=weight/LLaVA-OneVision \
  --model internvl3=weight/InternVL3 \
  --model qwen2.5-vl=weight/Qwen2.5-VL-7B-Instruct \
  --model qwen3-vl=weight/Qwen3-VL-8B-Instruct \
  --num-video-frames 32 \
  --num-video-frames-thumbnail 8 \
  --qwen-video-nframes 32 \
  --qwen-thumbnail-mode append-video \
  --video-resize-longest-edge 448 \
  --max-tiles-video 4 \
  --output-dir outputs/autogaze_repro/plugin_hlvid_limit3
```

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

## Caption/Action Benchmark Adapter

HLVid 이후의 task 확장은 `repro.video_task_benchmark`에서 시작합니다. 이 runner는 `flexible_runner --mode single`을 row별로 호출하고, caption/action별 schema와 scoring만 분리합니다.

| task_type | required fields | scoring |
| --- | --- | --- |
| `captioning` | `video_path`, `caption` 또는 `references` | 기본 `not_scored`; reference overlap hint만 기록 |
| `action_classification` | `video_path`, `label` 또는 `answer` | exact label match + multiple-choice letter parsing |

출력은 task별 `*_predictions.jsonl`, `*_scored.jsonl`, `*_summary.json`, `*_report.md`입니다. CUDA 검증 전에는 `configs/repro/video_task_caption_qwen_limit3.yaml`과 `configs/repro/video_task_action_qwen_limit3.yaml`을 smoke config로 사용합니다.

## 남은 구현/검증

1. Qwen `chunked_vit_autogaze_sparse`가 실제 ViT 입력 token 수를 줄이는지 CUDA smoke로 확인.
2. Qwen MRoPE/grid metadata가 sparse patch repack 이후에도 답변 품질을 망가뜨리지 않는지 확인.
3. LLaVA-OneVision post-encoder prune-generate가 실제 checkpoint별 remote/API에서 통과하는지 CUDA smoke로 확인.
4. LongVILA/NVILA-Video/InternVL에서 sidecar 이후 실제 remote-code visual pruning hook을 어디에 넣을지 확정.
5. probe/sidecar mode에서 `SparseSelectionPlan -> encoder mapping -> MLLM visual token mapping`이 실제 좌표 단위로 이어지는지 모델별 CUDA smoke로 확인.
6. Caption/action benchmark를 실제 CUDA model output으로 검증하고, 이후 새 VideoQA 데이터셋을 같은 task adapter 규격으로 추가.
