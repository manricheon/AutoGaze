# Token Selector / ViT / MLLM 연결 리포트

목표는 기존 MLLM의 vision path 앞에 AutoGaze 같은 token selector를 최대한 zero-shot으로 붙이고, ViT와 LLM 계산량이 실제로 줄어드는지 측정하는 것입니다.

긴 이전 리포트는 `docs/temp/AUTOGAZE_SELECTOR_VIT_MLLM_CONNECTION_REPORT_KO_FULL_2026-05-21.md`에 보존했습니다.

## 용어 정리

| 용어 | 의미 |
| --- | --- |
| token selector | ViT에 넣기 전 또는 후에 어떤 visual token/patch를 유지할지 고르는 모듈 |
| pre-ViT selector | patch가 ViT attention/MLP에 들어가기 전에 줄이는 방식 |
| post-encoder prune | ViT feature가 나온 뒤 MLLM에 넘길 token만 줄이는 방식 |
| dense mask | grid 크기는 유지하고 선택되지 않은 patch를 mask/zero 처리하는 방식 |
| sparse repack | 선택 patch만 모아 작은 sequence/grid로 ViT에 넣는 방식 |
| MRoPE/position metadata | Qwen류에서 visual token의 시간/공간 위치를 설명하는 position 정보 |
| visual token | MLLM이 LLM context로 받는 image/video feature token |

## 큰 연결 구조

```text
Video frames
  -> decode/sample/resize/tile
  -> Token Selector
       AutoGaze / keep-all / PixelPrune류 / none
  -> ViT Encoder
       dense grid 또는 sparse repack 입력
  -> projector / token shuffle / visual packing
  -> MLLM prefill + generate
```

AutoGaze 효과를 주장하려면 두 가지가 모두 보여야 합니다.

1. ViT에 들어가는 patch/token 수가 줄어든다.
2. LLM에 들어가는 visual token/context 길이가 줄어든다.

post-encoder prune만으로는 2번은 가능하지만 1번은 어렵습니다. pre-ViT sparse 입력이 중요한 이유가 여기에 있습니다.

## 추천 AutoGaze 출력 형태

현재 AutoGaze는 multiscale patch index를 냅니다. 여러 모델에 붙이려면 이 출력을 모델 독립적인 plan으로 정규화하는 것이 좋습니다.

```text
SparseGazePlan
  frame_index
  source_width/source_height
  resized_width/resized_height
  tile_index
  scale_id
  scale_resolution
  patch_size
  patch_row/patch_col
  normalized_box_xyxy
  keep_score(optional)
```

이 형태가 있으면 ViT별로 다음 중 하나를 선택할 수 있습니다.

| 적용 방식 | 장점 | 위험 |
| --- | --- | --- |
| selected crop/tile repack | 실제 ViT 계산 감소 가능 | position encoding 재해석 필요 |
| dense grid + attention mask | 구현 쉬움 | ViT 계산량 감소가 작음 |
| post-encoder gather | LLM token 감소 쉬움 | ViT 계산량은 그대로 |

## 모델별 연결성

| 모델군 | ViT/packing 특징 | AutoGaze zero-shot 난이도 | 우선 전략 |
| --- | --- | --- | --- |
| NVILA-HD | processor 내부 AutoGaze + SigLIP + projector | 낮음 | 안정 경로로 profile |
| NVILA-8B-Video | paper baseline, AutoGaze not applicable | 중간 | baseline/off 분리 |
| LongVILA | long-video packing/processor 확인 필요 | 중간-높음 | off/probe 후 sparse plan |
| Qwen3-VL | video grid + MRoPE metadata 중요 | 중간 | `full_vit`, `chunked_vit`, `chunked_vit_autogaze_sparse` |
| Qwen2/2.5-VL | processor와 pixel budget 중심 | 중간 | Qwen3 결과 후 이식 |
| LLaVA-OneVision | SigLIP류 vision path | 중간 | post-encoder prune 먼저 |
| InternVL3 | dynamic tiling | 높음 | tile metadata부터 정렬 |
| V-JEPA2 기반 ViT | representation learning용 ViT | 높음 | selector/encoder 연구용 |

## PixelPrune와 Qwen 관점

PixelPrune류는 Qwen 계열과 연결된 pre- 또는 mid-vision token pruning 아이디어로 참고할 가치가 큽니다. 다만 AutoGaze는 비디오의 frame/tile/scale별 patch index를 먼저 고르는 selector에 가깝고, PixelPrune는 모델 내부 feature 기준 pruning에 가까운 경우가 많습니다.

따라서 PoC 순서는 다음이 현실적입니다.

1. Qwen 기본 processor로 full video 입력이 되는지 확인.
2. chunk 단위 ViT 입력이 가능한지 확인.
3. AutoGaze가 낸 patch index를 Qwen grid/MRoPE metadata와 함께 sparse repack.
4. ViT 입력 token 수와 LLM visual token 수가 실제로 줄었는지 기록.
5. HLVid limit benchmark에서 off/chunked/sparse를 같은 질문 세트로 비교.

## 측정해야 하는 것

| 단계 | 꼭 볼 값 |
| --- | --- |
| decode/preprocess | 원본 해상도, resize 해상도, 선택 frame 수, tile 수 |
| selector | full patch, multiscale candidate patch, selected patch, reduction ratio |
| ViT | encoder input token 수, attention/MLP 추정 FLOPs, latency, peak memory |
| projector/packing | visual feature 수, token shuffle 등 token 변환 비율 |
| LLM | prefill context 길이, visual token 수, TTFT, generate latency, KV cache 추정 |
| benchmark | accuracy, failed, OOM, parse_failed, skipped |

## 남은 PoC 검증

1. Qwen sparse repack이 진짜 ViT 계산량을 줄이는지 CUDA profiler 또는 latency/token 로그로 확인.
2. sparse 위치 정보가 답변 품질에 미치는 영향을 작은 HLVid subset에서 확인.
3. LongVILA와 NVILA-Video의 processor packing metadata를 같은 `SparseGazePlan`으로 매핑.
4. V-JEPA2처럼 일반 ViT encoder에 붙일 때 어떤 downstream MLLM projector가 필요한지 별도 연구 태스크로 분리.
