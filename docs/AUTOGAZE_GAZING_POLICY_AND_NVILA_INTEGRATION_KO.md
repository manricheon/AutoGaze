# AutoGaze Gazing Policy와 NVILA Integration

이 문서는 AutoGaze가 어떤 기준으로 patch를 선택하는지, standalone AutoGaze와 NVILA-HD 내부 AutoGaze 적용이 왜 다르게 보일 수 있는지 정리합니다.

긴 이전 문서는 `docs/temp/AUTOGAZE_GAZING_POLICY_AND_NVILA_INTEGRATION_KO_FULL_2026-05-21.md`에 보존했습니다.

## AutoGaze가 하는 일

AutoGaze는 비디오 frame/tile에서 모든 patch를 그대로 ViT에 넣는 대신, task-conditioned decoder로 중요한 patch index를 autoregressive하게 선택합니다.

```text
frame/tile patches
  -> AutoGaze encoder/decoder
  -> selected patch indices by scale
  -> selected visual patches/features
  -> vision encoder / MLLM path
```

`gazing_ratio`는 선택 가능한 patch budget의 상한에 가깝습니다. 값을 낮추면 선택 patch 수는 줄어드는 경향이 있지만, latency가 완전히 비례해서 줄지는 않습니다. 모델 호출 overhead, 초기 token, stop 조건, batch 구성 비용이 남기 때문입니다.

## 첫 frame을 꼭 다 선택해야 하나

아닙니다. AutoGaze decoder는 autoregressive하게 선택을 시작하지만, 설정에 따라 첫 frame 전체를 강제로 keep-all하지 않을 수 있습니다. 그래서 첫 frame 선택 비율이 낮게 보일 수 있습니다.

중요한 것은 “첫 frame을 다 봤는가”가 아니라 다음입니다.

1. 전체 후보 patch 대비 selected patch가 얼마나 줄었는가.
2. 그 selected patch가 ViT encoder 입력 token으로 실제 반영되었는가.
3. LLM visual token/context도 줄었는가.

## Multiscale index의 의미

AutoGaze는 scale별 patch index를 낼 수 있습니다. 예를 들어 coarse scale은 넓은 영역을 빠르게 보고, fine scale은 중요한 영역을 더 자세히 선택하는 식입니다.

```text
scale 0: coarse patches
scale 1: medium patches
scale 2: fine patches
...
selected = scale별 index의 합집합 또는 processor 정책에 따른 packing
```

따라서 리포트에는 최소 세 기준을 함께 남겨야 합니다.

| 기준 | 의미 |
| --- | --- |
| 일반 full grid patch | AutoGaze 없이 단일 scale ViT에 넣는다면 예상되는 patch 수 |
| multiscale candidate patch | AutoGaze가 scale별로 고려하는 후보 patch 총량 |
| AutoGaze selected patch | 실제 선택되어 encoder/packing으로 넘어간 patch 수 |

## Standalone AutoGaze와 NVILA Integration 차이

| 항목 | Standalone Quick Start | NVILA-HD runner |
| --- | --- | --- |
| 목적 | AutoGaze 모델 자체 확인 | MLLM pipeline 안에서 실제 사용 |
| 입력 단위 | 공식 예제의 image/frame tensor | NVILA processor가 만든 frame/tile/thumbnail |
| scale/patch 설정 | Quick Start config | NVILA model/processor config |
| 출력 사용 | gaze info/selected patches 확인 | SigLIP/ViT, projector, MLLM으로 연결 |
| latency 의미 | AutoGaze-only | preprocess/AutoGaze/ViT/LLM 중 한 구간 |

그래서 “Quick Start는 3초, NVILA runner는 300ms”처럼 보이면 먼저 patch 후보 수와 scale 설정이 같은지 확인해야 합니다.

## Patch size 혼동 정리

NVILA-HD 계열에서는 config에 vision encoder patch size와 processor의 target patch size 관련 값이 따로 보일 수 있습니다. 예를 들어 SigLIP encoder 쪽은 patch size 14 계열인데, AutoGaze/preprocessor metadata에는 target patch size 16처럼 보이는 값이 있을 수 있습니다.

실험 원칙은 다음입니다.

1. vision encoder의 실제 patch embedding size는 모델 config를 따른다.
2. AutoGaze가 사용하는 target scale/patch grid는 processor config와 runtime log를 따른다.
3. benchmark에서는 임의로 ViT patch size를 바꾸기보다 AutoGaze 입력 grid를 조절한다.
4. 로그에는 원본 해상도, resize 해상도, tile 크기, patch grid, full/off patch, multiscale candidate, selected patch를 모두 남긴다.

예를 들어 입력 비디오가 224 shortest edge여도 NVILA processor가 392 tile 기준으로 처리한다면, AutoGaze도 그 processor가 만든 392 기준 tile/grid를 보게 됩니다. 이때 “224 입력이니까 224 grid”라고 해석하면 안 됩니다.

## TokenShuffle / projector / MLLM token

AutoGaze가 줄이는 것은 우선 patch 또는 vision token 후보입니다. 이후 NVILA pipeline에서는 SigLIP/ViT feature가 projector와 token shuffle류 처리를 거쳐 LLM visual token으로 변환됩니다.

```text
selected patch
  -> SigLIP/ViT feature
  -> projector
  -> token shuffle / packing
  -> LLM visual token
```

따라서 selected patch 수와 LLM visual token 수가 항상 1:1일 필요는 없습니다. 리포트에서는 둘을 분리해서 봐야 합니다.

## 확인해야 하는 로그

| 질문 | 필드 |
| --- | --- |
| AutoGaze 후보가 얼마나 컸나 | full/off patch, multiscale candidate patch |
| 얼마나 선택했나 | selected patch, scale별 selected patch |
| thumbnail은 어떻게 처리됐나 | thumbnail frame 수, thumbnail patch/token |
| ViT에 실제 몇 token이 갔나 | encoder input token |
| LLM에 실제 몇 visual token이 갔나 | LLM visual token/context length |
| 위치가 맞는지 눈으로 확인할 수 있나 | selected frame video, resized frame video, overlay video, gaze info JSON |

## Visualization 권장

시각화는 두 가지를 같이 저장하는 것이 좋습니다.

1. 선택된 frame만 모은 원본/리사이즈 기준 영상.
2. 같은 frame 위에 scale별 color mask를 올린 overlay 영상.

AutoGaze off/keep-all일 때는 overlay가 없더라도 선택 frame 영상은 저장해 benchmark 입력이 실제로 무엇이었는지 확인합니다.
