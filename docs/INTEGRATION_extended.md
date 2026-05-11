# AutoGaze 확장 통합 문서

이 문서는 원본 `INTEGRATION.md`를 수정하지 않고, 본 PoC에서 필요한 확장 통합 정책을 정리합니다. 원본 문서는 AutoGaze를 image ViT에 통합하는 기본 원리를 설명하는 1차 참조입니다.

## 1. 원본 INTEGRATION.md 요약

원본 문서의 핵심은 다음과 같습니다.

- AutoGaze는 각 video frame에서 처리할 patch 위치를 예측합니다.
- 기존 image ViT는 모든 patch를 embedding하지만, AutoGaze 경로에서는 선택된 patch만 embedding하도록 patch embedding을 수정합니다.
- 입력 shape는 image 기준 `[B, C, H, W]`에서 video 기준 `[B, T, C, H, W]`로 확장됩니다.
- 여러 frame의 token을 하나의 sequence로 처리하기 위해 attention mask가 필요합니다.
- attention mode는 block-causal, causal, bidirectional 형태를 고려합니다.
- ViT encoder layer, MLP, LayerNorm 등은 가능한 한 변경하지 않습니다.
- downstream MLLM에는 선택된 visual feature를 전달합니다.

본 문서는 위 원리를 다른 backbone과 MLLM으로 확장할 때의 정책을 정의합니다.

## 2. 확장 정책

확장 구현 원칙:

- 원본 AutoGaze 코드는 가능한 한 보존합니다.
- 직접 수정 대신 wrapper, adapter, registry, config-driven wiring을 우선합니다.
- 원본 `INTEGRATION.md`는 수정하지 않습니다.
- SigLIP, NVILA, Qwen, dataset path, frame count, resolution, token budget을 hardcode하지 않습니다.
- AutoGaze는 visual patch/token selector 또는 router로 취급합니다.
- patch index가 target backbone patch grid와 직접 일치한다고 가정하지 않습니다.
- zero-padding으로 mismatch를 조용히 숨기지 않습니다.

## 3. AutoGaze patch index와 다른 backbone mapping

AutoGaze output metadata는 가능하면 다음 정보를 보존해야 합니다.

- frame indices
- patch indices
- scale information
- original token counts
- selected token counts
- patch grid 정보

다른 backbone에 연결할 때는 `PatchGridMapper`가 다음 정보를 명시적으로 받아야 합니다.

| 항목 | 설명 |
|---|---|
| source patch grid | AutoGaze가 예측한 patch grid |
| target patch grid | target backbone의 patch grid |
| source resolution | AutoGaze 기준 resolution |
| target resolution | target backbone 기준 resolution |
| patch-size mismatch | source/target patch 크기 차이 |
| multi-scale metadata | scale별 grid와 patch index 정보 |

지원하지 않는 mapping은 명확히 에러를 발생시켜야 합니다.

## 4. modified SigLIP 통합

modified SigLIP는 canonical path의 핵심 vision encoder입니다.

필수 경로:

```text
AutoGaze OFF -> modified SigLIP ViT -> NVILA
AutoGaze ON  -> modified SigLIP ViT -> NVILA
```

정책:

- 원본 modified SigLIP multi-scale patch handling을 우선 따릅니다.
- original integration의 temporal handling을 보존합니다.
- wrapper adapter는 `ModifiedSigLIPAdapter`로 격리합니다.
- 실제 model instance가 없으면 명확한 `NotImplementedError`를 발생시킵니다.

## 5. vanilla SigLIP 통합

vanilla SigLIP는 modified SigLIP 효과를 분리하기 위한 baseline입니다.

필수 경로:

```text
AutoGaze OFF -> vanilla SigLIP ViT -> NVILA
AutoGaze ON  -> vanilla SigLIP ViT -> NVILA
```

주의:

- AutoGaze ON + vanilla SigLIP는 A3 experimental compatibility ablation입니다.
- 직접 호환된다고 주장하지 않습니다.
- Hugging Face 또는 local SigLIP loading은 adapter 뒤로 격리합니다.

가능한 통합 mode:

1. input-level crop/region reconstruction
2. post-patch-embedding token masking
3. compact token gathering
4. post-encoder pruning

각 mode는 encoder-side acceleration인지 아닌지 별도로 기록해야 합니다.

## 6. V-JEPA2 통합

V-JEPA2는 pretrained video ViT backbone 실험용 adapter입니다.

정책:

- 입력 shape `[B, T, C, H, W]`를 명시적으로 보존합니다.
- V-JEPA2 내부 수정은 피하고 adapter로 격리합니다.
- 현재 PoC의 `VJEPA2Adapter`는 dummy shape adapter이며 실제 V-JEPA2 internals는 구현하지 않았습니다.

지원 mode:

- `full`
- `crop`
- `mask`
- `compact`

지원하지 않는 mode는 명확한 에러를 발생시켜야 합니다.

## 7. generic ViT 통합

generic ViT는 backbone-agnostic integration 검증용입니다.

정책:

- arbitrary patch size와 resolution을 config로 받습니다.
- patch grid를 명시적으로 계산합니다.
- AutoGaze patch index는 target grid로 remap해야 합니다.
- dummy forward는 shape test용이며 실제 성능을 의미하지 않습니다.

## 8. NVILA 통합

NVILA는 canonical MLLM target입니다.

필수 경로:

```text
modified SigLIP -> NVILA
vanilla SigLIP -> NVILA
AutoGaze-selected visual tokens -> NVILA
```

정책:

- visual feature shape, token order, positional metadata를 명시합니다.
- `NVILAAdapter`는 실제 NVILA model instance가 없으면 `NotImplementedError`를 발생시킵니다.
- randomly initialized adapter로 zero-shot 성능을 주장하지 않습니다.

## 9. Qwen 통합

Qwen-family MLLM은 NVILA 외 MLLM 확장 경로입니다.

정책:

- Qwen 전용 pipeline으로 hardcode하지 않습니다.
- `BaseMLLMAdapter` interface를 따릅니다.
- official processor path를 우선합니다.
- 직접 visual token injection은 지원된다고 가정하지 않습니다.

단계:

| 단계 | mode | 설명 |
|---|---|---|
| Stage 0 | `official_processor` | Qwen 공식 processor baseline |
| Stage 1 | `input_region_selection` | AutoGaze-guided frame/region selection |
| Stage 2 | `post_visual_encoder_pruning` | 기술적으로 가능한 경우 post-encoder pruning |
| Stage 3 | `direct_visual_token_injection` | architecture/API 검증 후에만 허용 |

## 10. 기타 MLLM 통합

새 MLLM은 registry로 추가합니다.

공통 interface:

- `prepare_visual_inputs`
- `prepare_text_inputs`
- `forward`
- `generate`
- `count_visual_tokens`

visual input preparation과 text prompt preparation은 분리합니다.

## 11. Hugging Face MLLM 통합

Hugging Face MLLM은 기본적으로 official processor mode를 사용합니다.

정책:

- model ID와 dataset ID를 hardcode하지 않습니다.
- revision pinning을 지원합니다.
- `local_files_only`, `cache_dir`, offline mode를 지원합니다.
- access token은 환경 변수에서만 읽습니다.
- token은 log, config, output에 저장하지 않습니다.
- HF public MLLM이 AutoGaze-selected visual token injection을 지원한다고 가정하지 않습니다.

## 12. Temporal dimension handling

모든 video tensor는 다음 shape를 사용합니다.

```text
[B, T, C, H, W]
```

지원 temporal mode:

| mode | 설명 |
|---|---|
| `frame_wise` | frame별 독립 처리 |
| `mean_pool` | frame feature 평균 pooling |
| `max_pool` | frame feature max pooling |
| `concat_tokens` | temporal token concat |
| `native_autogaze` | 원본 integration 기준 처리, 현재 stub |

frame index metadata는 sampling 이후에도 보존합니다.

## 13. full / hook / native integration mode

| mode | 설명 | acceleration claim |
|---|---|---|
| `full` | full-token baseline | 없음 |
| `hook` | 기존 model 중간에 adapter/hook 연결 | 구현 위치에 따라 다름 |
| `native` | 원본 AutoGaze integration 방식 | 실제 encoder compute 감소 여부 확인 필요 |
| `crop` | input-level region/crop selection | 조건부 encoder compute 감소 가능 |
| `mask` | token mask 적용 | full encoder 이후면 encoder acceleration 아님 |
| `compact` | selected token gather | 적용 위치에 따라 다름 |
| `official_processor` | HF MLLM 공식 processor 경로 | AutoGaze token injection 아님 |

## 14. HF official processor mode

HF MLLM은 우선 official processor로 baseline을 만듭니다.

AutoGaze-guided input reduction은 별도 experimental mode로만 추가합니다.

보고 시 구분:

- reported paper/model-card result
- reproduced result
- internal PoC result
- extension result
- dummy/stub result

외부 모델이 AutoGaze를 사용했다고 명시적으로 보고하지 않은 경우, 그렇게 주장하지 않습니다.
