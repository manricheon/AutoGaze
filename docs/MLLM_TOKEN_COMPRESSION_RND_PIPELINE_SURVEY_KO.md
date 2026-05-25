# MLLM Token Compression R&D Pipeline Survey

작성일: 2026-05-23  
최종 업데이트: 2026-05-25  
목적: MLLM 토큰 처리 병목을 줄이기 위한 사내 R&D 프로젝트 파이프라인 발굴  
주요 기준: Image/CV task와 Video task를 분리하고, 각 축을 Pre-Encoder, In-Encoder, Post-Encoder/Projector 단계로 정리

관련 문서:

- [리더십/핵심 멤버용 전략 에세이](./MLLM_TOKEN_COMPRESSION_RND_STRATEGY_ESSAY_KO.md)
- [핵심 레퍼런스 논문 한국어 요약](./SHAO_TOKEN_COMPRESSION_SURVEY_SUMMARY_KO.md)

## 1. Executive Summary

MLLM의 실무 병목은 "모델이 크다"보다 "시각/청각 입력이 LLM context와 KV cache를 과도하게 점유한다"에 가깝다. 특히 고해상도 이미지, 다중 이미지, 긴 비디오, 오디오-비디오 입력은 텍스트보다 훨씬 많은 token을 만들며, prefill latency, attention FLOPs, GPU memory, KV cache 비용을 동시에 증가시킨다.

2025-2026년 문헌 기준으로 우선 검토해야 할 R&D 축은 다음과 같다. 이 표는 특정 논문이나 특정 내부 구현을 우선 채택하자는 의미가 아니라, Shao et al. 서베이의 `modality`, `compression mechanism`, `pipeline location`, `evaluation risk` 관점을 사내 프로젝트 후보로 번역한 것이다.

| 우선순위 | 프로젝트 방향 | 대상 | 파이프라인 | 실무 판단 |
|---|---|---|---|---|
| 0 | Token budget instrumentation and evaluation | 전체 | 전체 | 계측 없이 압축률만 보면 제품 병목을 잘못 판단할 수 있음 |
| 1 | Video multi-stage token budget pipeline | Video task | Pre/In/Post/KV | frame, patch, prefill, KV 병목이 동시에 발생하는 핵심 축 |
| 2 | Image/CV query-aware and OCR-safe compression | Image/CV | Post/In + fallback | 빠른 PoC가 가능하지만 OCR/small-object 보존 평가가 필수 |
| 3 | Multimodal KV cache budget manager | Image/Video/Text | Decode/KV | 서빙 GPU memory와 동시 처리량에 직접 영향 |
| 4 | In-Encoder token merge/recycling | Image/CV + Video | In-Encoder | drop보다 정보 보존 여지가 있으나 모델 내부 통합 난도 큼 |
| 5 | Pre-Encoder redundancy feasibility study | Image/CV + Video + Audio | Pre-Encoder | encoder 비용까지 줄일 수 있으나 정보 손실 리스크가 커서 조건부 검토 |
| 6 | Audio and audio-video token compression | Audio + Video | Pre/Post/Cross-modal | 옴니모달 제품 대비용 보조 축 |
| 7 | Text prompt/context compression | Text/RAG | Prompt/KV | vision 압축 후에도 RAG/history/KV 병목이 남는 문제 보완 |

핵심 레퍼런스인 Shao et al.의 TMLR 2026 서베이는 MLLM token compression을 이미지, 비디오, 오디오별 redundancy와 transformation/similarity/attention/query 기반 접근으로 분류한다. 본 문서는 그 분류를 사내 R&D 파이프라인 관점으로 재배치했다. 논문 자체의 자세한 한국어 요약은 [핵심 레퍼런스 요약 문서](./SHAO_TOKEN_COMPRESSION_SURVEY_SUMMARY_KO.md)에 별도로 정리했다.

이 문서의 주된 의사결정 대상은 vision, 특히 image/CV와 video task다. Audio와 text는 독립 제품 축이라기보다 향후 omnimodal 서비스, multimodal RAG, multi-turn serving에서 vision compression 효과를 갉아먹지 않도록 함께 관리해야 하는 보조 축으로 본다. 따라서 프로젝트 검토 축은 video/image 중심으로 잡되, 평가 지표와 runtime 설계에는 audio/text token budget을 같이 포함한다.

또한 이 문서는 AutoGaze, PixelPrune, FastVID, TokenPacker 등 특정 방법을 우선 채택하자는 제안서가 아니다. 해당 논문들은 Shao et al. taxonomy 안에서 각 stage와 mechanism을 설명하기 위한 reference sample이다. 최종 방향은 특정 논문 이름이 아니라 `어떤 modality에서`, `어느 pipeline 위치에서`, `무슨 redundancy를`, `어떤 품질 리스크를 감수하며` 줄일지로 결정해야 한다.

핵심 레퍼런스 논문을 이 문서에 반영한 방식은 다음과 같다.

| 논문 관점 | 이 문서의 반영 위치 | 의미 |
|---|---|---|
| Modality별 redundancy | 5장 Modality R&D Map | image, video, audio, text/KV를 분리해 병목을 본다. |
| Compression mechanism taxonomy | 2장, 3장, 6장 | transformation, similarity, attention, query 기반 압축을 프로젝트 성격으로 해석한다. |
| Pipeline 위치별 trade-off | 4장 Pipeline Taxonomy, 7장 Roadmap | Pre/In/Post/KV 중 어디서 줄이는지에 따라 비용 절감 범위가 다름을 명시한다. |
| Evaluation difficulty | 5.0, 7장, 8장 | compression ratio만 보지 않고 OCR, grounding, hallucination, latency, KV memory를 함께 본다. |

수치 해석 원칙은 명확히 둔다. 본 문서에 적은 speedup, FLOPs reduction, KV memory saving은 각 논문의 실험 설정에서 보고된 결과다. 사내 제품에서 같은 수치를 보장한다는 의미가 아니다. 프로젝트 제안의 근거로는 사용하되, 실제 의사결정은 동일 workload에서 `accuracy`, `TTFT`, `encoder latency`, `prefill latency`, `decode throughput`, `peak memory`, `KV cache memory`, `fallback rate`를 재측정한 뒤 내려야 한다.

### 1.1 문서 독자와 설명 레벨

이 문서는 완전한 입문 튜토리얼이 아니라, 프로젝트 세팅을 위해 내부 핵심 멤버와 의사결정권자가 같은 지도를 보게 만드는 planning document다. 따라서 설명 레벨은 세 층을 동시에 만족하는 것을 목표로 한다.

| 독자 | 이 문서에서 얻어야 하는 것 | 읽어야 할 섹션 |
|---|---|---|
| 신규 참여자 | MLLM token compression이 왜 필요한지, 기본 용어가 무엇인지 | 2장, 3장, 4장 |
| 내부 핵심 멤버 | vision 중심 R&D 흐름, modality별 병목, pipeline별 trade-off | 5장, 6장, 7장 |
| 리더십/윗선 보고 | 왜 지금 프로젝트 파이프라인을 잡아야 하는지, 어떤 후보가 우선인지 | 1장, 7장, 8장 |

핵심 멤버에게는 "기술 분류"보다 "어떤 병목을 어떤 위치에서 줄일 것인가"가 중요하다. 예를 들어 video token compression을 논의할 때는 단순히 pruning 논문을 나열하는 대신, frame/segment selection, spatial patch reduction, projector compression, KV cache management가 각각 어떤 비용을 줄이는지 분리해서 봐야 한다.

리더십 설명에서는 더 압축된 메시지가 필요하다. 권장 narrative는 다음과 같다.

```text
1. MLLM의 비용 병목은 LLM parameter보다 multimodal token 길이에서 크게 발생한다.
2. 특히 video는 frame 수와 해상도가 곱으로 늘어나므로 text보다 훨씬 빠르게 context를 채운다.
3. Token compression은 단순 최적화가 아니라 long-video/high-resolution MLLM 제품화를 위한 기반 기술이다.
4. 첫 R&D 파이프라인은 vision, 특히 video를 중심으로 잡고, image/CV는 projector와 OCR-safe compression을 병행한다.
5. Audio와 text는 후순위가 아니라 serving과 omnimodal 확장을 위한 budget 관리 축으로 함께 설계한다.
```

따라서 이 문서의 결론은 "하나의 압축 알고리즘을 고르자"가 아니다. 결론은 "Pre-Encoder, In-Encoder, Post-Encoder, Decode/KV 각 위치에 어떤 후보 프로젝트를 둘지 정하고, 서비스 적용처별로 정확도/비용 trade-off를 검증하자"이다.

### 1.2 핵심 주장: 왜 해야 하고, 무엇을 해야 하며, 어떤 효과를 기대하는가

이 R&D를 해야 하는 이유는 단순히 inference cost를 조금 낮추기 위해서가 아니다. MLLM 제품의 입력이 text-only에서 image, video, audio로 확장될수록 병목은 모델 parameter 수보다 multimodal token 수로 이동한다. 특히 video와 high-resolution image는 token 수가 서비스 요구와 거의 정면으로 충돌한다. 이 병목을 풀지 못하면 긴 영상 이해, 고해상도 문서/화면 QA, multi-turn multimodal assistant, edge/on-premise deployment가 모두 비싸거나 느리거나 불안정해진다.

정확히 해야 하는 일은 다음 세 가지다.

| 해야 할 일 | 설명 | 성공 기준 |
|---|---|---|
| 1. Token budget을 측정 가능한 자원으로 만든다 | modality별 token 수, prefill latency, encoder latency, KV memory를 계측하고 compression 위치별 비용 절감 범위를 분리한다. | 어떤 요청이 왜 느리고 비싼지 stage별로 설명 가능해야 한다. |
| 2. Vision 중심 compression pipeline을 만든다 | video는 frame/patch/KV를 함께 줄이고, image/CV는 query-aware projector와 OCR-safe compression을 병행한다. | 동일 task accuracy를 유지하면서 visual token, TTFT, peak memory를 의미 있게 줄여야 한다. |
| 3. 서비스별 fallback/evaluation policy를 만든다 | OCR, small object, temporal grounding, hallucination, WER 등 압축 취약 task를 별도 평가하고 fallback rule을 둔다. | 빠른 평균 성능뿐 아니라 실패 시나리오가 통제되어야 한다. |

기대 효과는 네 층으로 봐야 한다.

| Effect | 구체적 의미 | 왜 중요한가 |
|---|---|---|
| Cost reduction | prefill FLOPs, encoder FLOPs, KV cache memory 감소 | 같은 GPU에서 더 많은 요청을 처리하거나 더 긴 입력을 처리할 수 있다. |
| Latency reduction | TTFT와 long-video QA 응답 시간 감소 | 사용자 경험과 interactive workflow가 개선된다. |
| Product expansion | 4K 이미지, 장시간 영상, 다중 이미지/비디오, 회의/강의 영상 처리 가능 | 기존 dense 처리로는 비용상 어려운 제품 영역을 연다. |
| Control and observability | token budget, compression ratio, fallback reason을 로그로 관리 | "빠르지만 왜 틀렸는지 모르는 시스템"이 아니라 운영 가능한 MLLM이 된다. |

리더십 관점의 핵심 메시지는 다음이다.

```text
MLLM token compression은 서버비 최적화 과제가 아니라,
고해상도/장시간 멀티모달 제품을 가능하게 만드는 기반 기술이다.
우리가 해야 할 일은 단일 pruning 기법을 고르는 것이 아니라,
vision 중심 token budget pipeline을 만들고
정확도 손실을 통제 가능한 운영 지표로 바꾸는 것이다.
```

### 1.3 R&D Thesis and Evidence

| Thesis | 근거 | 프로젝트 연결 |
|---|---|---|
| Video가 1순위다 | Shao et al.은 90-minute video가 약 54M token 규모까지 커질 수 있음을 예시로 든다. 긴 영상은 frame 수와 해상도가 곱으로 늘어난다. | Project 5, 6, 7 |
| Post-Encoder만으로는 부족하다 | Post-Encoder compression은 LLM context와 KV cache를 줄이지만 vision encoder 비용은 남는다. 고해상도 입력에서는 Pre-Encoder 또는 In-Encoder 압축이 필요하다. | Project 3, 4 |
| Query-aware compression은 빠른 PoC 후보지만 fallback이 필요하다 | 사용자 질문과 관련 높은 token을 남기면 VQA 비용을 줄일 수 있으나, OCR/작은 객체/배경 근거를 잃을 수 있다. | Project 1, 2 |
| KV cache는 별도 제품화 병목이다 | visual/audio token은 prefill 이후에도 decode 단계 KV cache를 점유한다. multi-turn multimodal assistant에서는 memory 병목이 커진다. | Project 8 |
| Audio/Text는 보조 축이지만 무시하면 안 된다 | 회의/강의 영상은 audio가 중요한 segment signal이 되고, RAG/history는 text-side KV 병목을 만든다. | Project 9, 10, text/KV roadmap |

### 1.4 Shao Survey Section-Level Evidence Package

윗선 설득에는 "논문 수가 많다"보다 "대형 연구 조직들이 독립적으로 같은 병목을 시스템 설계 문제로 다루고 있다"는 근거가 더 강하다. Shao et al.의 핵심 섹션별로 바로 연결할 수 있는 evidence package는 다음과 같다.

| Shao survey 축 | 핵심 주장 | 설득에 쓸 key references | 내부 프로젝트 연결 |
|---|---|---|---|
| Background: MLLM architecture | MLLM context는 system/text token보다 visual/audio token이 지배하는 경우가 많다. 따라서 model size 축소만으로는 부족하다. | Flamingo, 2022; BLIP-2, 2023; InstructBLIP, 2023. DeepMind/Salesforce 계열 연구는 variable visual features를 fixed/query token으로 연결하는 구조를 사용했다. | Project 1, 3 |
| Image / transformation-based | 고해상도 image는 native resolution을 유지하면서도 token budget을 제어해야 한다. | Qwen2.5-VL, 2025는 dynamic resolution, window attention, absolute time encoding을 사용한다. NVLM, 2024는 tile-based dynamic high-resolution design을 OCR/reasoning 성능 근거로 제시한다. | Project 1, 4 |
| Image / similarity-based and in-encoder | visual token은 embedding/layer 내부에서도 redundancy가 남는다. drop보다 merge/recycle이 accuracy-sensitive task에 유리할 수 있다. | ToMe, 2023은 기존 ViT에 training 없이 적용 가능한 token merging으로 image/video/audio throughput 개선을 보였다. VisionZip, 2025는 attention selection과 similarity merge를 조합한다. | Project 3 |
| Image / query-based | 질문을 알고 있는 시점에는 모든 visual token을 동일하게 LLM에 넘길 필요가 없다. | BLIP-2의 Q-Former, InstructBLIP의 instruction-aware Query Transformer, LLaVA-Mini의 one visual token 계열은 query/projector stage가 압축 지점이 될 수 있음을 보여준다. | Project 1, 2 |
| Video / long-context compression | Video는 frame 수와 해상도가 곱으로 늘어나므로 image보다 더 먼저 제품 병목이 된다. | NVILA, 2025는 scale-then-compress로 high-resolution image와 long video를 처리하며 prefill/decode latency 개선을 보고했다. Qwen2.5-VL은 hours-long video와 second-level event localization을 설계 목표로 둔다. FastVID/METok은 post/pipeline 압축의 직접 baseline이다. | Project 5, 6, 7 |
| Audio / omnimodal compression | Audio는 time/spectral redundancy를 갖고, video salience를 보조하는 cross-modal signal이 될 수 있다. | Qwen2.5-Omni, 2025는 audio/video streaming을 위해 block-wise processing과 time-aligned multimodal RoPE를 사용한다. Towards Audio Token Compression, 2025는 audio encoder 이후 LLM decoder 이전 token을 최대 3x 줄이는 방향을 제시한다. | Project 9, 10 |
| Text / prompt and KV | Vision token을 줄여도 prompt, RAG context, conversation history, KV cache가 serving 병목으로 남는다. | Microsoft LLMLingua, 2023은 prompt를 최대 20x 압축하는 text-side 근거다. MEDA, 2025는 multimodal KV cache allocation으로 memory/decoding 병목을 직접 다룬다. | Project 8, text/KV roadmap |

이 표의 용도는 reference name dropping이 아니다. 각 행은 독립 프로젝트를 만들 수 있는 근거 단위다. `Qwen2.5-VL/NVILA`는 high-resolution/video 효율화가 frontier VLM 설계 주제임을 보여주고, `BLIP-2/InstructBLIP/Flamingo`는 projector/query token이 오래된 구조적 압축 지점임을 보여준다. `LLMLingua/MEDA`는 vision 압축만으로는 serving 비용 문제가 닫히지 않는다는 근거다.

## 2. Why Token Compression Matters

Shao et al. 서베이의 핵심 메시지는 단순하다. 텍스트는 길어져도 token 수가 비교적 선형적으로 증가하지만, 이미지/비디오/오디오는 해상도, 시간, 주파수 축 때문에 token 수가 훨씬 빠르게 커진다. 서베이의 예시 기준으로 10K words는 약 13K token, 4K UHD image는 약 32K token, 2-hour audio는 약 720K token, 90-minute video는 약 54M token까지 커질 수 있다. 이 차이 때문에 multimodal token compression은 "nice-to-have optimization"이 아니라 long-context MLLM의 전제 조건에 가깝다.

```text
Approximate token scale from the survey

Text, 10K words        | 13K
4K UHD image           | 32K
2-hour audio           | 720K
90-minute video        | 54M

Token count grows by:
Image  ~= resolution
Audio  ~= duration x sampling/feature rate
Video  ~= frames x spatial resolution
```

위 표의 핵심은 video token이 단순히 "이미지가 여러 장" 수준으로 늘어나는 것이 아니라는 점이다. 비디오는 frame 수, spatial resolution, tile 수, thumbnail/auxiliary stream이 곱으로 붙기 때문에 token 수가 폭발한다. 그래서 vision R&D에서도 image와 video를 분리해야 한다. Image는 공간적 중복 제거가 핵심이고, video는 공간적 중복과 시간적 중복을 동시에 줄여야 한다.

MLLM에서 token이 쌓이는 위치는 다음과 같다.

```text
Raw modality input
  Image: pixels / patches
  Video: frames x patches
  Audio: waveform or spectrogram frames
  Text : system + prompt + history
        |
        |  [A] Pre-Encoder Compression
        |      - encoder에 넣기 전에 frame/patch/audio segment를 제거
        |      - encoder FLOPs까지 줄일 수 있지만, 버린 정보는 복구하기 어려움
        v
Vision / Audio Encoder
        |
        |  [B] In-Encoder Compression
        |      - encoder layer 내부에서 token merge/prune/recycle 수행
        |      - 성능 보존 가능성은 높지만 모델 내부 수정 난도가 큼
        v
Projector / Adapter
        |
        |  [C] Post-Encoder / Projector Compression
        |      - LLM에 넣기 전 encoded visual/audio token을 선택/병합
        |      - plug-in은 쉽지만 vision/audio encoder 비용은 그대로 남음
        v
LLM Prefill Context
  [system tokens] + [visual/audio tokens] + [text prompt tokens]
        |
        |  [D] Decode / KV Cache Compression
        |      - generation 중 cache token을 유지/제거/양자화
        |      - serving memory, batch size, multi-turn latency에 직접 영향
        v
Autoregressive response
```

읽는 방법은 간단하다. Pre-Encoder에 가까울수록 계산량 절감 폭은 크지만 정보 손실 위험이 크다. Post-Encoder에 가까울수록 기존 모델에 붙이기 쉽지만 encoder 비용 절감은 제한된다. Decode/KV 단계는 model quality보다 serving economics에 직접 연결된다. 따라서 사내 R&D는 "어디서 줄였는가"를 먼저 명시해야 하며, 같은 compression ratio라도 pipeline 위치가 다르면 사업적 효과가 다르다.

서베이는 modality별 redundancy와 mechanism별 압축 방식을 함께 보라고 제안한다. 이 문서도 같은 구조를 따른다.

```text
Modality redundancy
  Image : spatial redundancy
          repeated textures, background, local patch similarity
  Video : spatio-temporal redundancy
          static background, repeated frames, slow motion
  Audio : temporal + spectral redundancy
          silence, stationary noise, sparse salient events
  Text  : prompt/history redundancy
          repeated instructions, irrelevant retrieved chunks, stale KV

Compression mechanisms
  Transformation-based : pooling, convolution, pixel/token unshuffle, stacking
  Similarity-based     : merge/group similar tokens
  Attention-based      : prune low-attention or low-importance tokens
  Query-based          : keep tokens relevant to user instruction/query
```

이 분류는 프로젝트를 읽는 기본 렌즈다. 예를 들어 video frame sampling은 Pre-Encoder이면서 temporal redundancy를 줄이는 방식이고, query-aware projector는 Post-Encoder이면서 query-based compression이다. KV cache compression은 modality token 자체를 줄이는 것은 아니지만, 이미 LLM에 들어간 token의 serving 비용을 줄이는 별도 축이다.

## 3. 용어 정의 (Glossary)

아래 용어는 문서 전체에서 반복적으로 사용된다. 영어 용어는 논문 검색과 구현 논의에 필요하고, 한글 정의는 기획/리더십 커뮤니케이션에서 같은 의미로 읽히도록 넣었다.

### 3.1 기본 개념

| English term | 한국어 용어 | 정의 |
|---|---|---|
| MLLM / Multimodal Large Language Model | 멀티모달 대규모 언어모델 | 텍스트뿐 아니라 이미지, 비디오, 오디오 등 여러 입력 모달리티를 LLM context에 연결해 추론하는 모델. |
| Modality | 모달리티 | 모델이 처리하는 입력 종류. 이 문서에서는 image, video, audio, text를 의미한다. |
| Token | 토큰 | 모델이 attention 계산에 사용하는 최소 처리 단위. 텍스트 단어 조각뿐 아니라 이미지 patch feature, 비디오 frame patch, 오디오 frame feature도 token으로 취급한다. |
| Attention | 어텐션 | token 간 관련도를 계산해 어떤 정보를 얼마나 볼지 결정하는 Transformer의 핵심 연산. token 수가 늘면 비용이 크게 증가한다. |
| Context | 컨텍스트 | LLM이 한 번의 추론에서 참고하는 전체 입력 token 묶음. system prompt, text, visual/audio token, 대화 history가 포함된다. |

### 3.2 모달리티별 토큰

| English term | 한국어 용어 | 정의 |
|---|---|---|
| Visual token | 시각 토큰 | vision encoder 또는 projector를 거쳐 LLM에 전달되는 image/video feature token. LLM 입장에서는 텍스트 token과 같은 context 공간을 점유한다. |
| Patch token | 패치 토큰 | 이미지를 일정 크기 patch로 나눈 뒤 ViT가 처리하는 token. 고해상도 이미지에서는 patch 수가 급격히 늘어난다. |
| Audio token | 오디오 토큰 | waveform, spectrogram, codec, audio encoder output에서 만들어지는 시간/주파수 단위 token. 긴 음성에서는 duration에 비례해 증가한다. |
| Prompt token | 프롬프트 토큰 | system prompt, user instruction, retrieval context, conversation history가 tokenizer를 거쳐 만들어진 token. |

### 3.3 MLLM 파이프라인

| English term | 한국어 용어 | 정의 |
|---|---|---|
| KV Cache | 키-값 캐시 | LLM decoding 때 이전 token의 key/value tensor를 저장해 재계산을 줄이는 메모리 구조. context가 길수록 GPU memory를 많이 차지한다. |
| Prefill | 프리필 | 입력 prompt와 multimodal token 전체를 LLM에 처음 넣어 attention cache를 만드는 단계. visual token 수가 많으면 TTFT가 커진다. |
| Decode | 디코드 | prefill 이후 답변 token을 하나씩 생성하는 단계. KV cache 크기와 retention policy가 latency/memory에 영향을 준다. |
| Projector / Adapter | 프로젝터 / 어댑터 | vision/audio encoder output을 LLM embedding 공간에 맞게 변환하는 모듈. Post-Encoder compression을 넣기 쉬운 위치다. |

### 3.4 압축 위치

| English term | 한국어 용어 | 정의 |
|---|---|---|
| Pre-Encoder Compression | 인코더 전 압축 | encoder에 들어가기 전에 frame, patch, audio segment를 줄이는 방식. 계산량 절감 폭이 크지만 정보 손실 위험도 크다. |
| In-Encoder Compression | 인코더 내부 압축 | ViT/audio encoder layer 중간에서 token을 merge, prune, recycle하는 방식. 성능 보존 가능성은 있지만 모델 내부 통합이 어렵다. |
| Post-Encoder Compression | 인코더 후 압축 | encoder output 이후 LLM 입력 전 token을 줄이는 방식. 기존 모델에 붙이기 쉽지만 encoder 비용은 남는다. |
| Projector Compression | 프로젝터 압축 | projector 단계에서 visual/audio token을 선택, 병합, 요약하는 방식. query-aware compression과 결합하기 좋다. |
| KV Cache Compression | KV 캐시 압축 | decoding 중 KV cache를 줄이는 방식. token 자체를 줄이기보다 serving memory와 latency를 줄인다. |

### 3.5 압축 방법

| English term | 한국어 용어 | 정의 |
|---|---|---|
| Token Pruning | 토큰 가지치기 | 중요도가 낮은 token을 제거하는 방식. 구현은 단순하지만 정보 손실 위험이 있다. |
| Token Merging | 토큰 병합 | 유사한 token을 하나 또는 소수의 대표 token으로 합치는 방식. pruning보다 정보 보존이 유리할 수 있다. |
| Token Pooling / Stacking | 토큰 풀링 / 스태킹 | 여러 token을 평균, convolution, stride, stacking으로 압축하는 transformation-based 방식. |
| Token Recycling | 토큰 재활용 | 버릴 token의 정보를 보존 token에 흡수해 완전한 삭제보다 손실을 줄이는 방식. |
| Attention-based Compression | 어텐션 기반 압축 | attention score나 attention-derived importance를 이용해 token을 선택/제거하는 방식. |
| Similarity-based Compression | 유사도 기반 압축 | token embedding 간 cosine similarity 등으로 유사 token을 병합하거나 제거하는 방식. |
| Query-aware Compression | 질의 기반 압축 | 사용자 질문이나 instruction과 관련 높은 token을 더 많이 남기는 방식. VQA, video QA에 적합하다. |
| Transformation-based Compression | 변환 기반 압축 | pooling, convolution, token unshuffle처럼 구조적 변환으로 token 수를 줄이는 방식. |

### 3.6 중복성 유형

| English term | 한국어 용어 | 정의 |
|---|---|---|
| Spatial Redundancy | 공간적 중복 | 이미지 내 배경, 반복 texture, 인접 patch 유사성처럼 공간 축에서 생기는 중복. |
| Temporal Redundancy | 시간적 중복 | 비디오/오디오에서 인접 frame 또는 segment가 거의 같은 정보를 담는 현상. |
| Spectral Redundancy | 주파수 중복 | 오디오에서 특정 주파수 대역이 반복되거나 정보량이 낮은 현상. |

### 3.7 평가 및 리스크

| English term | 한국어 용어 | 정의 |
|---|---|---|
| Compression Ratio | 압축률 | 원래 token 수 대비 얼마나 줄였는지 나타내는 비율. 예: 75% compression은 token 4개 중 1개만 남기는 의미로 쓰일 수 있어 논문별 정의 확인이 필요하다. |
| Retention Ratio | 보존율 | 원래 token 중 몇 %를 남겼는지 나타내는 비율. 25% retention은 75% pruning과 같은 의미다. |
| Grounding | 근거 정렬 | 모델 답변이 실제 이미지/비디오/audio evidence에 기반하는 정도. 압축 후 grounding이 약해지면 hallucination이 늘 수 있다. |
| Hallucination | 환각 | 입력 근거에 없는 내용을 모델이 생성하는 현상. visual/audio token pruning이 과하면 증가할 수 있다. |
| Fallback Policy | 폴백 정책 | 압축 confidence가 낮거나 OCR/작은 객체처럼 위험한 입력에서 full-token 또는 낮은 압축률 경로로 돌아가는 정책. |
| VAD / Voice Activity Detection | 음성 활동 탐지 | 오디오에서 발화 구간과 무음/비발화 구간을 구분하는 기술. audio pre-encoder compression의 기본 도구다. |
| WER / Word Error Rate | 단어 오류율 | ASR 품질 지표. audio token compression이 과하면 WER이 증가한다. |
| TTFT / Time To First Token | 첫 토큰 지연시간 | 요청 후 첫 답변 token이 나오기까지 걸리는 시간. prefill 비용과 직접 관련된다. |
| Throughput | 처리량 | 단위 시간당 처리 가능한 요청 또는 token 수. token compression의 사업적 효과를 보여주는 핵심 serving 지표다. |

## 4. Pipeline Taxonomy

| 단계 | 정의 | 대표 문제 | 장점 | 단점 |
|---|---|---|---|---|
| Pre-Encoder Compression | 원본 데이터 또는 patch/tokenization 직전에서 잉여 정보를 차단 | 정적 배경, 중복 프레임, 무음/소음 구간 | encoder FLOPs까지 줄일 수 있음 | 잘못 버리면 복구 불가 |
| In-Encoder Compression | vision/audio encoder 내부 layer에서 token merge/prune/recycle | ViT token redundancy, 위치 정보 손실 | 정보 보존과 비용 절감 균형 가능 | 모델 내부 hook, kernel 호환성 이슈 |
| Post-Encoder / Projector Compression | encoder 출력 후 LLM 입력 전 visual/audio token을 필터링/요약 | LLM prefill, context length, KV cache 증가 | 기존 MLLM에 plug-in하기 쉬움 | encoder 비용은 그대로 남음 |
| Decode/KV Cache Compression | LLM decoding 중 KV cache retention/eviction/quantization | 장문 응답, 멀티턴, long video QA | 서빙 memory와 batch capacity 개선 | generation quality 영향 검증 필요 |

## 5. Modality R&D Map

### 5.0 Balanced Reference Technology Stack

아래 항목은 특정 내부 구현을 설명하기 위한 것이 아니라, 프로젝트 기획 시 참고할 만한 외부 논문/기술 스택 후보로만 취급한다. 이전 버전처럼 일부 pre-encoder 사례만 강조하면 전체 landscape가 왜곡될 수 있으므로, modality와 pipeline stage가 골고루 보이도록 재정리했다.

| 범주 | 대표 Reference | Modality | Pipeline 위치 | Planning 관점의 의미 |
|---|---|---|---|---|
| Survey backbone | Shao et al., 2026 | Image / Video / Audio | 전체 | 본 문서의 큰 분류 기준. modality별 redundancy와 mechanism별 compression taxonomy를 제공한다. |
| Frontier VLM design signals | Flamingo, 2022; BLIP-2/InstructBLIP, 2023; NVLM, 2024; Qwen2.5-VL, 2025; NVILA, 2025 | Image & Video | Projector / Encoder / Runtime | big tech/large-lab 모델들도 visual token budget을 adapter, query token, dynamic resolution, scale-then-compress 설계로 다룬다는 근거. |
| Image pre-encoder | PixelPrune, 2026; V-PRUNE, 2025 | Image/CV | Pre-Encoder | encoder 이전 pixel/patch redundancy 제거 가능성을 보여주는 사례. high-risk 후보로 두고 다른 stage와 비교해야 한다. |
| Image post-encoder / projector | TokenPacker, 2024; LLaVA-Mini, 2025; ReDiPrune, 2026 | Image/CV | Post-Encoder / Projector | 기존 MLLM에 비교적 붙이기 쉬운 visual token reduction 계열. 빠른 PoC 후보. |
| Image/video unified pruning | VisionZip, 2025; VisionTrim, 2026; HiPrune, 2025/2026 | Image & Video | Post-Encoder / Hybrid | image와 video에 공통 적용 가능한 training-free 또는 plug-and-play compression 후보. |
| In-encoder merge/recycle | FiCoCo, 2026; PPE, 2025/2026; ADSC, 2026 | Image & Video | In-Encoder | token drop보다 정보 보존을 중시하는 계열. integration 난도는 높지만 accuracy-sensitive task에 중요. |
| Video pre/post compression | AutoGaze, 2026; FastVID, 2025; LongVU, 2024 | Video | Pre-Encoder / Post-Encoder | long-video에서 frame, patch, temporal redundancy를 줄이는 대표 사례군. 특정 방법보다 stage별 비교가 중요하다. |
| Video multi-stage runtime | METok, 2025; DyCoke, 2025; ForestPrune, 2026 | Video | Pre + Post + Decode/KV | long-video는 단일 압축 지점만으로 부족하다는 근거. runtime budget manager 설계 참고. |
| Multimodal KV cache | MEDA, 2025; AirCache, 2025; FlashCache, 2025/2026 | Image / Video / Text | Decode/KV | serving memory, batch capacity, multi-turn latency 개선을 위한 KV cache 정책 후보. |
| Audio compression | Audio Token Compression in LALMs, 2025; HeadRouter, 2026; A-ToMe, 2023 | Audio | Pre / In / Post | 긴 음성, 회의, 콜센터, audio QA 확장을 위한 secondary axis. |
| Audio-video compression | OmniZip, 2025/2026; OmniRefine, 2026; OmniSIFT, 2026 | Audio + Video | Cross-Modal | audio salience로 video token budget을 조정하는 omnimodal 확장 후보. |
| Text prompt / context | LLMLingua, 2023; LongLLMLingua, 2023; H2O, 2023; SnapKV, 2024 | Text | Prompt / Decode/KV | vision token을 줄여도 prompt/RAG/history/KV가 병목으로 남는 문제를 보완한다. |
| Evaluation benchmark | UniPruneBench, 2025; EffiVLM-BENCH, 2025; VideoMME, 2025 | Image / Video | Evaluation | compression ratio만으로 판단하지 않고 OCR, grounding, latency, hallucination을 함께 평가하기 위한 기준. |

### 5.1 Image / CV Task Axis

대상 task: Image QA, document QA, OCR, chart/table understanding, screenshot/UI QA, industrial inspection, medical image QA, robotics perception snapshot.

Image/CV에서의 압축 핵심은 "어떤 patch가 실제 답변 근거인가"를 찾는 것이다. 일반 자연 이미지는 배경과 반복 texture가 많아 aggressive pruning이 비교적 잘 맞을 수 있다. 반면 문서, 표, 차트, UI, 산업 검사 이미지는 작은 글자, 얇은 선, 미세 결함이 정답 근거가 되므로 단순 saliency나 attention score만 믿으면 위험하다. 따라서 image/CV 축은 일반 VQA와 OCR/structured image를 분리해 평가해야 한다.

실무적으로는 Post-Encoder projector compression이 가장 빠른 PoC 후보지만, 고해상도 입력이 많은 서비스에서는 Pre-Encoder pixel/patch pruning도 반드시 검토해야 한다. Post-Encoder는 LLM 비용을 줄이고, Pre-Encoder는 vision encoder 비용까지 줄인다. 두 방식은 경쟁 관계라기보다 서비스 latency budget에 따라 조합해야 하는 선택지다.

| Pipeline | Project Theme | 적용 task | 핵심 방법 | 재학습 필요성 | 실무 리스크 |
|---|---|---|---|---|---|
| Pre-Encoder | Saliency-aware patch budgeter | OCR 제외 일반 VQA, 상품 이미지 QA | 해상도/patch budget을 saliency에 따라 배분 | 선택적 | 작은 객체/문자 누락 |
| Pre-Encoder | Pixel-level redundancy pruning | 문서, GUI, OCR, screenshot QA | pixel duplicate/redundancy 기반 patch 제거 | 낮음 | OCR/작은 UI 요소 손실 가능 |
| In-Encoder | Positional-preserving token merge | 문서, UI, 산업 검사 | token merge 시 2D 위치 정보 보존 | 낮음~중간 | ViT 내부 수정 필요 |
| In-Encoder | Token recycling instead of dropping | 일반 CV QA, dense scene QA | 불필요 token을 버리지 않고 대표 token에 흡수 | 낮음 | runtime hook 복잡도 |
| Post-Encoder | Query-aware visual projector | 대부분의 이미지 QA | 질문 관련 visual token만 보존/병합 | 중간 | query bias로 배경 단서 손실 |
| Post-Encoder | Explainability-guided visual pruning | 문서/차트/일반 VQA | first-layer attention 또는 explanation proxy로 token 선택 | 낮음 | explanation proxy 안정성 |
| Decode/KV | Visual KV cache budget manager | 멀티턴 이미지 QA | modality-aware KV retention | 낮음 | serving scheduler 통합 |

### 5.2 Video Task Axis

대상 task: Long-video QA, CCTV/event detection, meeting/lecture summarization, sports highlight, industrial monitoring, egocentric/robot video, video captioning.

Video는 이 문서의 가장 중요한 축이다. Video token 병목은 image보다 더 심각하다. 같은 장면이 여러 frame에 반복되고, 배경은 거의 변하지 않으며, 질문과 무관한 시간이 대부분을 차지한다. 하지만 작은 이벤트가 몇 frame에만 나타나는 경우도 많기 때문에 단순 uniform sampling이나 무작정 frame drop은 위험하다.

Video compression은 최소 세 층으로 나눠 봐야 한다. 첫째, frame/segment 수준에서 중복 시간을 줄인다. 둘째, 선택된 frame 안에서 spatial patch를 줄인다. 셋째, LLM에 들어간 뒤 visual KV cache가 계속 비용을 만들지 않도록 decode 단계에서 관리한다. 장시간 영상 서비스에서는 이 세 층 중 하나만으로는 충분하지 않을 가능성이 높다.

| Pipeline | Project Theme | 적용 task | 핵심 방법 | 재학습 필요성 | 실무 리스크 |
|---|---|---|---|---|---|
| Pre-Encoder | Autoregressive gaze-style patch selector | HLVid, long-video QA, 고해상도/장시간 영상 | reconstruction/error budget 기반 multi-scale patch 선택 | 중간 | selector overhead와 위치 정보 보존 필요 |
| Pre-Encoder | Static background/frame redundancy filter | CCTV, 강의, 회의, 제조 라인 | scene change, optical flow, frame similarity 기반 frame/patch 제거 | 없음~낮음 | subtle event 누락 |
| Pre-Encoder | Event-aware video segment budgeter | long-video QA, 하이라이트 | event density별 token budget 차등 배정 | 낮음 | event detector 품질 의존 |
| In-Encoder | Spatiotemporal positional-preserving merge | fine-grained video QA | temporal + spatial 위치 보존 merge | 낮음~중간 | temporal ordering 손상 |
| Post-Encoder | Dynamic density video pruning | LLaVA-OneVision/Qwen2.5-VL류 | frame/token density에 따라 retention ratio 조절 | 없음~낮음 | task별 최적 ratio 편차 |
| Post-Encoder | LLM-guided keyframe prior compression | query-conditioned video QA | LLM attention으로 keyframe prior 추정 | 없음 | attention score 계산 overhead |
| Decode/KV | Long-video KV cache retention | 장문 응답, 멀티턴 video chat | visual KV 중 중요한 token만 유지 | 없음~낮음 | hallucination/grounding 악화 가능 |

### 5.3 Audio Task Axis

Audio는 이 문서의 서브 축이지만, 향후 omnimodal 제품을 고려하면 별도 갈무리가 필요하다. Shao et al.은 audio redundancy를 temporal/spectral redundancy로 본다. 즉, 긴 침묵, stationary background noise, 반복적인 음향 패턴, 제한된 주파수 대역에 정보가 몰리는 현상이 핵심이다. Audio MLLM에서는 Whisper/Conformer류 continuous encoder output, HuBERT/EnCodec류 discrete token, Mel-spectrogram 기반 2D representation이 모두 압축 대상이 될 수 있다.

Audio는 video와 결합될 때 특히 중요하다. 회의, 강의, 인터뷰 영상에서는 audio가 "어느 시점이 중요한가"를 알려주는 강한 신호가 된다. 반대로 CCTV나 제조 영상처럼 무음 visual event가 중요한 도메인에서는 audio-guided pruning이 오히려 중요한 장면을 놓칠 수 있다. 따라서 audio compression은 독립 ASR 품질뿐 아니라 video token budget을 조절하는 보조 signal로도 평가해야 한다.

| Pipeline | Project Theme | 적용 task | 핵심 방법 | 재학습 필요성 | 실무 리스크 |
|---|---|---|---|---|---|
| Pre-Encoder | Silence/noise-aware segment filter | 회의록, 콜센터, 긴 음성 QA | VAD, noise profile, spectral novelty로 무음/저정보 구간 제거 | 낮음 | 낮은 음량의 중요 발화 누락 |
| In-Encoder | Audio token merge / A-ToMe style compression | ASR, audio captioning, sound event QA | cosine similarity 기반 adjacent audio token merge | 낮음~중간 | 발화 경계/phoneme 정보 손실 |
| Post-Encoder | Audio token pooling / stacking adapter | speech translation, audio QA | token stacking, stride pooling, temporal convolution | 중간 | WER, 고유명사, 숫자 오류 증가 |
| Cross-Modal | Audio-guided video pruning | lecture/interview/video QA | salient audio segment가 video token budget을 guide | 낮음~중간 | 무음 visual event에 취약 |
| Decode/KV | Audio-context KV cache retention | long audio chat, meeting assistant | 발화 turn, speaker, topic별 cache retention | 낮음 | multi-speaker reference 손실 |

### 5.4 Text / Prompt / KV Cache Axis

Text는 Shao et al. 서베이의 주된 modality 분류에는 포함되지 않지만, MLLM 운영에서는 system prompt, user instruction, retrieval context, conversation history, KV cache가 비용의 한 축이다. Vision 중심 MLLM이라도 text side compression을 무시하면 multi-turn 또는 RAG 결합 서비스에서 병목이 남는다.

Text compression은 vision compression의 대체재가 아니다. 다만 실제 서비스에서는 visual token을 줄여도 RAG context, 긴 system prompt, 대화 history가 KV cache를 계속 차지한다. 따라서 serving 관점에서는 visual/audio token budget과 text prompt budget을 하나의 budget manager가 함께 관리하는 구조가 필요하다. 특히 document QA나 video RAG에서는 근거 문장을 줄이다가 citation 오류가 생길 수 있으므로, compression 후 answer quality뿐 아니라 evidence retention도 봐야 한다.

| Pipeline | Project Theme | 적용 task | 핵심 방법 | 재학습 필요성 | 실무 리스크 |
|---|---|---|---|---|---|
| Pre-LLM Prompt | Prompt/context compression | RAG, 문서 QA, multi-turn QA | LLMLingua류 token-level prompt compression, extractive chunk selection | 낮음~중간 | 근거 문장 누락, citation 오류 |
| Query Planner | Query-aware retrieval budget | image/doc/video RAG | query type에 따라 visual/audio/text budget 동적 배분 | 낮음 | query classifier 오류 |
| Decode/KV | Task-aware KV cache compression | long-context generation, multi-turn MLLM | token eviction, semantic chunk KV retention, KV quantization | 낮음 | long-range reference와 reasoning degradation |
| Cross-Modal | Text-guided visual/audio token selection | VQA, video QA, audio-video QA | instruction과 cross-modal similarity로 relevant token 유지 | 낮음~중간 | query bias, multi-turn 재압축 비용 |

### 5.5 Modality Takeaways

| Modality | 왜 token이 늘어나는가 | 압축이 필요한 위치 | 우선 R&D 방향 | 주의할 failure mode |
|---|---|---|---|---|
| Image / CV | 해상도 증가가 patch 수 증가로 직결됨 | Pre-Encoder, Post-Encoder, KV | 문서/GUI/OCR은 pixel-level redundancy와 query-aware projector를 병행 검토 | OCR, small object, chart/table cell 손실 |
| Video | frame 수 x 해상도 x tile 수가 곱으로 증가함 | Pre-Encoder, Post-Encoder, Decode/KV | 회사의 메인 축. frame redundancy, spatiotemporal pruning, long-video KV를 우선 검토 | subtle event, temporal ordering, grounding 손실 |
| Audio | duration과 feature frame rate가 길이를 결정하고 silence/noise가 많음 | Pre-Encoder, Post-Encoder | VAD/noise-aware filtering, token pooling, audio-guided video pruning을 후순위 확장 과제로 검토 | WER 증가, speaker/turn 경계 손실 |
| Text | system prompt, RAG context, conversation history, KV cache가 누적됨 | Pre-LLM Prompt, Decode/KV | vision/audio token 압축과 함께 prompt/KV budget manager를 설계 | 근거 문장 누락, multi-turn reference 손실 |

## 6. Project Proposals

### Research Signals from 2025-2026

최근 문헌에서 반복적으로 보이는 신호는 네 가지다.

1. Pre-Encoder compression은 encoder 비용까지 줄일 수 있지만, 위치/순서 정보 보존과 정보 손실 통제가 가장 어렵다.
2. Post-Encoder / projector compression은 기존 모델에 붙이기 쉽지만, vision encoder 비용은 그대로 남는다.
3. Video compression은 frame-level sampling만으로 부족하고, spatial-temporal structure를 같이 봐야 한다.
4. Benchmark 결과는 task-sensitive하다. UniPruneBench는 OCR이 pruning에 취약하고, pruning ratio가 성능 하락의 주요 인자임을 보고했다.

### Project 1. Image/CV Query-aware Visual Projector

**Target Pipeline & Modality**  
Post-Encoder / Projector Compression, Image/CV task.

**Problem**  
고해상도 이미지, 문서 이미지, 다중 이미지 VQA에서 비전 인코더가 만든 수백~수천 개의 visual token이 그대로 LLM에 들어간다. 질문이 "영수증 총액"인데 배경, 여백, 반복 패턴 token까지 LLM prefill과 KV cache를 점유한다.

**Why it matters**  
TokenPacker는 visual projector 단계에서 visual token 75-89% 압축을 보고했다. LLaVA-Mini는 LLaVA-v1.5의 576 vision token 대신 1 token을 사용하면서 FLOPs 77% 절감과 24GB GPU에서 10,000개 이상 video frame 처리를 보고했다. 이 계열은 기존 모델의 vision encoder를 완전히 바꾸지 않고 projector/runtime 쪽에서 시작할 수 있어 PoC 비용이 낮다.

**How to solve**  
비전 인코더 출력 후 projector 앞에 query-aware scorer를 둔다. 사용자 instruction/text embedding과 visual token 간 similarity 또는 cross-attention proxy를 계산하고, 상위 token은 보존, 중간 token은 region-level merge, 하위 token은 global context token으로 흡수한다. OCR/문서 task에서는 작은 글자 token을 보호하는 minimum retention rule을 둔다.

**Reference Papers**

- TokenPacker: Efficient Visual Projector for Multimodal LLM, 2024. Coarse-to-fine visual projector로 token 75-89% 압축.
- LLaVA-Mini: Efficient Image and Video Large Multimodal Models with One Vision Token, 2025. Modality pre-fusion으로 vision token을 1개까지 줄임.
- VisionSelector: End-to-End Learnable Visual Token Compression for Efficient Multimodal LLMs, 2025. Lightweight Top-K scorer 기반 learnable visual token selection.

**Trade-offs & Applications**  
LoRA 또는 projector fine-tuning이 필요할 가능성이 높다. 작은 객체, OCR, 의료/산업 결함처럼 local evidence가 중요한 task에서는 aggressive compression이 위험하다. 최적 적용처는 상품 이미지 QA, 일반 VQA, chart/table QA, screenshot QA, 문서 요약이다.

### Project 2. Image/CV Explainability-guided Token Pruning

**Target Pipeline & Modality**  
Post-Encoder Compression, Image/CV task.

**Problem**  
기존 visual token pruning은 "어떤 token이 답변에 중요한가"를 안정적으로 설명하기 어렵다. 서비스 적용 시 failure case 분석과 안전한 compression ratio 설정이 어렵다.

**Why it matters**  
2025년 Generic Token Compression 연구는 explanation method가 instruction에 대한 visual token 중요도를 평가할 수 있으며, Qwen2-VL, LLaVA-OneVision, VILA1.5에서 visual token 50% pruning 후 원 성능 96% 이상 유지 사례를 보고했다. 이 방향은 모델 교체보다 운영 검증과 디버깅에 유리하다.

**How to solve**  
first LLM layer attention map 또는 lightweight explanation proxy network를 사용해 token importance map을 만든다. 프로젝트 산출물은 단순 pruning이 아니라 "token keep/drop heatmap + answer confidence + fallback policy"까지 포함해야 한다. 낮은 confidence 또는 OCR-heavy 입력에서는 full-token path로 fallback한다.

**Reference Papers**

- Generic Token Compression in Multimodal Large Language Models from an Explainability Perspective, 2025. Explanation-guided visual token compression.
- VisionZip: Longer is Better but Not Necessary in Vision Language Models, CVPR 2025. Informative token selection으로 prefilling time 8x 개선을 보고.
- EffiVLM-Bench, 2025/2026. Training-free LVLM acceleration이 task/model에 민감하다는 benchmark 관찰 제공.

**Trade-offs & Applications**  
Explanation proxy가 모든 모델/도메인에 일반화된다고 가정하면 안 된다. OCR, chart, UI task는 token importance가 작은 영역에 숨어 있을 수 있다. 적용처는 내부 문서 QA, 업무용 screenshot QA, 이미지 moderation review, 시각 검색 결과 요약이다.

### Project 3. Image/CV In-Encoder Positional-Preserving Merge

**Target Pipeline & Modality**  
In-Encoder Compression, Image/CV task.

**Problem**  
token drop/merge는 spatial layout을 훼손한다. 문서, 표, UI, 산업 검사에서는 "무엇이 있는가"뿐 아니라 "어디에 있는가"가 정답을 좌우한다.

**Why it matters**  
PPE는 visual token compression 중 spatiotemporal position 손실 문제를 지적하고, MMBench, TextVQA, VideoMME 등에서 2-5% 개선을 보고했다. FiCoCo는 training-free 방식으로 MLLM context를 줄이며 최대 14.7x FLOPs reduction과 93.6% performance retention을 보고했다.

**How to solve**  
ViT 중간 layer에서 token을 단순 제거하지 않고, 유사 token을 대표 token에 merge하면서 2D position embedding을 별도 보존한다. 문서/표 task는 grid-aware merge, 일반 이미지 task는 local-window merge, 산업 검사 task는 anomaly-preserving merge를 사용한다.

**Reference Papers**

- Positional Preservation Embedding for Multimodal Large Language Models, 2025/ICLR 2026. 압축 token의 위치 구조 보존.
- Filter, Correlate, Compress: Training-Free Token Reduction for MLLM Acceleration, AAAI 2026. FiCoCo-V/FiCoCo-L로 vision encoder와 LLM decoder 양쪽 최적화.
- Vision Token Reduction via Attention-Driven Self-Compression, 2026. LLM attention 기반 progressive self-compression.

**Trade-offs & Applications**  
vision encoder 내부 hook이 필요하다. FlashAttention, torch.compile, vLLM류 serving stack과 shape/dynamic graph 충돌이 날 수 있다. 적용처는 문서 OCR QA, UI QA, 제조 결함 검출, 로봇 scene understanding이다.

### Project 4. Pre-Encoder Redundancy Feasibility Study

**Target Pipeline & Modality**  
Pre-Encoder Compression, Image/CV, Video, and Audio.

**Problem**  
Post-Encoder 또는 projector compression은 기존 MLLM에 붙이기 쉽지만, encoder 자체의 FLOPs와 memory를 줄이지 못한다. 고해상도 이미지, 긴 비디오, 장시간 오디오에서는 encoder 이전 또는 tokenization 직전에서 명백한 중복을 줄일 수 있는지 검토할 필요가 있다.

**Why it matters**  
Shao et al.의 관점에서 Pre-Encoder compression은 가장 큰 비용 절감 가능성과 가장 큰 정보 손실 리스크를 동시에 갖는 위치다. AutoGaze, PixelPrune, V-PRUNE은 이 위치에서 가능한 접근을 보여주는 사례일 뿐이며, 이 프로젝트의 목적은 특정 방법을 채택하는 것이 아니라 "encoder 이전 압축이 우리 workload에서 실제로 안전하고 유효한가"를 검증하는 것이다.

**How to solve**  
Image/CV, Video, Audio를 분리해 접근한다. Image/CV에서는 pixel/patch redundancy, document/OCR risk, GUI component preservation을 본다. Video에서는 frame redundancy, motion salience, temporal event coverage를 본다. Audio에서는 silence/noise filtering과 spectral novelty를 본다. 모든 경우에 aggressive pruning을 기본값으로 두지 않고, conservative budget, retained-evidence logging, fallback rule을 함께 설계한다.

**Reference Papers**

- Attend Before Attention: Efficient and Scalable Video Understanding via Autoregressive Gazing, CVPR 2026. pre-encoder video patch selection 사례.
- PixelPrune: Pixel-Level Adaptive Visual Token Reduction via Predictive Coding, 2026. pixel-space redundancy 기반 image/document patch pruning 사례.
- V-PRUNE: Semantic-Aware Patch Pruning Before Tokenization in Vision-Language Model Inference, 2025. tokenization 이전 patch-level pruning 사례.

**Trade-offs & Applications**  
Pre-Encoder compression은 한 번 버린 정보를 복구하기 어렵다. 따라서 이 프로젝트는 "가장 먼저 제품화할 핵심 방향"이 아니라, encoder 비용이 실제 병목인 workload에서만 조건부로 검토할 feasibility track이다. OCR, small-object detection, UI control recognition, subtle temporal event에서는 fallback 또는 conservative mode가 필요하다.

### Project 5. Video Static Background and Redundant Frame Pre-Filter

**Target Pipeline & Modality**  
Pre-Encoder Compression, Video task.

**Problem**  
긴 영상에서는 정적 배경과 중복 프레임이 대부분인데, 현재 VLLM은 이를 encoder와 LLM까지 통과시키는 경우가 많다. 회의, 강의, CCTV, 제조 라인 영상에서 비용 낭비가 크다.

**Why it matters**  
FastVID는 LLaVA-OneVision-7B에서 video token 90.3% pruning, FLOPs 8.3% 수준, prefill 7.1x 가속, 원 성능 98.0% 유지를 보고했다. LongVU는 DINOv2 feature 기반 redundant frame removal과 text-guided token reduction을 사용해 long video context 병목을 완화한다.

**How to solve**  
비디오 입력 단계에서 scene boundary, frame embedding similarity, optical flow, background subtraction을 조합해 frame/patch 후보를 줄인다. 고정 간격 샘플링 대신 event density와 motion saliency에 따라 per-segment budget을 동적으로 배정한다. subtle event 보호를 위해 periodic low-rate sentinel frame은 유지한다.

**Reference Papers**

- FastVID: Dynamic Density Pruning for Fast Video Large Language Models, 2025. Dynamic density pruning.
- LongVU: Spatiotemporal Adaptive Compression for Long Video-Language Understanding, 2024. DINOv2 기반 redundant frame removal + query-guided reduction.
- METok: Multi-Stage Event-based Token Compression for Efficient Long Video Understanding, EMNLP 2025. Event-aware long-video compression.

**Trade-offs & Applications**  
Pre-Encoder에서 버린 정보는 복구할 수 없다. 느린 변화, 작은 이상 징후, 무음 이벤트를 놓칠 수 있다. 적용처는 CCTV 요약, 제조 라인 모니터링, 강의/회의 요약, 스포츠 하이라이트 검색이다.

### Project 6. Video Multi-Stage Long-Context Compression Runtime

**Target Pipeline & Modality**  
Pre-Encoder + Post-Encoder + Decode/KV, Video task.

**Problem**  
긴 영상 병목은 한 곳에서만 생기지 않는다. frame/token redundancy, LLM prefill, decoding KV cache가 모두 비용을 만든다. 단일 pruning module만으로는 장시간 video chat이나 long-video QA를 안정적으로 처리하기 어렵다.

**Why it matters**  
METok은 LongVA-7B에 적용해 FLOPs 80.6% 감소와 KV cache memory 93.5% 절감을 보고했다. DyCoke는 training-free temporal merging과 dynamic KV pruning으로 video LLM inference speedup과 memory reduction을 보고했다.

**How to solve**  
runtime에 3단계 budget manager를 둔다. 1단계는 event/scene 기반 frame-token budget, 2단계는 instruction과 semantic alignment 기반 prefill pruning, 3단계는 decode 중 visual KV cache retention 정책이다. 서비스별로 accuracy floor를 설정하고 compression ratio를 동적으로 조절한다.

**Reference Papers**

- METok: Multi-Stage Event-based Token Compression for Efficient Long Video Understanding, EMNLP 2025. Event-aware vision encoding, prefill pruning, decoding KV optimization.
- DyCoke: Dynamic Compression of Tokens for Fast Video Large Language Models, CVPR 2025. Temporal token merging + dynamic KV pruning.
- HieraVid: Hierarchical Token Pruning for Fast Video Large Language Models, 2026. Segment/frame/layer-level hierarchical pruning.

**Trade-offs & Applications**  
runtime orchestration이 복잡하다. benchmark별 최적 retention ratio가 다르고, fine-grained grounding task에서는 성능 하락 가능성이 있다. 적용처는 장시간 영상 QA, 보안 이벤트 검색, 스포츠 분석, 온라인 교육 영상 챗봇이다.

### Project 7. Video Fine-Grained Grounding Safe Pruning

**Target Pipeline & Modality**  
Post-Encoder Compression, Video task.

**Problem**  
training-free video pruning은 MCQA처럼 coarse cue가 충분한 benchmark에서는 잘 보이지만, hallucination 평가, open-ended generation, fine-grained grounding에서는 성능 붕괴가 생길 수 있다.

**Why it matters**  
SToP는 기존 pruning이 fine-grained video understanding에서 약해질 수 있으며, sink token이 pruning 후에도 살아남아 visual evidence를 왜곡할 수 있다고 분석했다. 이는 사내 서비스에서 "빠르지만 근거가 틀린 답변"으로 이어질 수 있다.

**How to solve**  
FastVID/VisionZip류 pruning 앞뒤에 sink-token detector와 grounding consistency check를 둔다. 압축 후 답변 근거 frame을 재검증하고, evidence confidence가 낮으면 해당 segment만 재확장한다. 즉, 전체 full-token fallback이 아니라 local rehydration을 사용한다.

**Reference Papers**

- Sink-Token-Aware Pruning for Fine-Grained Video Understanding in Efficient Video LLMs, 2026. Sink token score로 existing pruning method 보정.
- FastVID, 2025. Dynamic density pruning baseline.
- VisionZip, CVPR 2025. Image/video 이해용 informative token selection baseline.

**Trade-offs & Applications**  
검증 단계가 latency를 추가한다. 하지만 안전성이 중요한 video QA에서는 필요한 비용이다. 적용처는 산업 안전, 의료/수술 영상 리뷰, 자율주행/로봇 로그 분석, 보안 관제다.

### Project 8. Multimodal KV Cache Budget Manager

**Target Pipeline & Modality**  
Decode/KV Cache Compression, Image, Video, Text.

**Problem**  
multi-image, long-video, long-dialogue 환경에서 KV cache가 GPU memory 병목이 된다. visual token은 prefill 이후에도 decoding 단계에서 cache read/write 비용을 계속 유발한다.

**Why it matters**  
MEDA는 cross-modal attention entropy로 layer-wise KV budget을 배정해 최대 72% KV memory reduction과 2.82x faster decoding을 보고했다. AirCache는 inter-modal relevancy 기반 visual KV retention을 제안했다. FlashCache는 attention score 의존 방식이 FlashAttention과 충돌할 수 있음을 지적하고 frequency-domain outlier-KV-aware compression을 제안했다.

**How to solve**  
서빙 레이어에 modality-aware KV cache manager를 구현한다. visual/audio/text token을 동일하게 eviction하지 않고, layer별 cross-modal entropy, inter-modal relevancy, outlier-KV score를 사용한다. vLLM/paged attention과 통합하려면 attention score를 매 decode step에서 직접 요구하지 않는 정책을 우선 검토한다.

**Reference Papers**

- MEDA: Dynamic KV Cache Allocation for Efficient Multimodal Long-Context Inference, 2025. Cross-modal attention entropy 기반 layer-wise KV allocation.
- AirCache: Activating Inter-modal Relevancy KV Cache Compression for Efficient Large Vision-Language Model Inference, 2025. Visual KV retention.
- Revisiting Multimodal KV Cache Compression: A Frequency-Domain-Guided Outlier-KV-Aware Approach, 2025/2026. FlashAttention 호환성 문제와 value vector 중요도 반영.

**Trade-offs & Applications**  
KV policy는 generation quality와 hallucination에 직접 영향을 준다. cache compaction kernel, batch scheduler, paged attention과의 통합이 관건이다. 적용처는 멀티턴 이미지 QA, 비디오 챗봇, RAG + 이미지/문서 QA, edge serving이다.

### Project 9. Audio-Guided Video Token Compression

**Target Pipeline & Modality**  
Pre-Encoder + Post-Encoder Compression, Audio + Video.

**Problem**  
audio-video 모델은 두 모달리티 token이 동시에 증가한다. visual-only pruning은 말소리, 음향 이벤트, 입모양, 타이밍 단서를 놓칠 수 있다.

**Why it matters**  
OmniZip은 salient audio token을 먼저 식별한 뒤 audio retention score로 video token pruning을 guide하며, 3.42x inference speedup과 1.4x memory reduction을 보고했다. OmniRefine은 audio-video chunk boundary를 정렬해 correspondence-preserving compression을 제안했다.

**How to solve**  
speech activity, audio event, spectral novelty score를 이용해 time window별 salience를 계산한다. 이 score로 video token budget을 조정하고, audio-video alignment가 깨지는 구간은 압축을 완화한다. 무음 visual event를 놓치지 않기 위해 motion score를 보조 신호로 둔다.

**Reference Papers**

- OmniZip: Audio-Guided Dynamic Token Compression for Fast Omnimodal Large Language Models, 2025/2026. Training-free audio-guided video token pruning.
- OmniRefine: Audio-Visual Cooperative Token Compression, 2026. Correspondence-preserving audio-video compression.
- A Survey of Token Compression for Efficient Multimodal Large Language Models, TMLR 2026. Audio의 temporal/spectral redundancy 분류.

**Trade-offs & Applications**  
audio가 정보량 높은 콘텐츠에는 강하지만, 무음 감시 영상이나 시각 중심 산업 영상에서는 효과가 작다. 적용처는 회의/강의/인터뷰 영상, 콜센터 녹취+화면 분석, 멀티모달 하이라이트 생성이다.

### Project 10. Audio Token Downsampling Adapter

**Target Pipeline & Modality**  
Post-Encoder / Projector Compression, Audio.

**Problem**  
Large Audio Language Model은 audio encoder가 높은 token rate를 만들며, 장문 음성에서 LLM decoder 입력 길이가 급증한다. ASR, speech translation, audio QA를 저비용 GPU나 edge에서 운영하기 어렵다.

**Why it matters**  
Towards Audio Token Compression in Large Audio Language Models는 audio encoder 출력 후 LLM decoder 입력 전 token을 줄이는 방식을 연구하고, unsupervised segmentation, uniform average pooling, LoRA realignment를 검토했다. 관련 요약 자료는 WER이 3x compression 부근까지 비교적 안정적이다가 이후 악화된다고 정리한다.

**How to solve**  
audio encoder 출력에 segmentation-aware pooling adapter를 추가한다. speech activity, phoneme boundary, spectral novelty를 기준으로 variable-length pooling을 수행하고 LoRA로 ASR/ST 성능을 회복한다. 소음 구간은 aggressive pooling, 발화 전환/숫자/고유명사 구간은 보존한다.

**Reference Papers**

- Towards Audio Token Compression in Large Audio Language Models, 2025. Audio encoder 이후 LLM decoder 이전 token compression.
- HeadRouter: Dynamic Head-Weight Routing for Task-Adaptive Audio Token Pruning in Large Audio Language Models, 2026. Task-adaptive audio token pruning.
- OmniZip, 2025/2026. Audio salience를 cross-modal compression signal로 활용.

**Trade-offs & Applications**  
ASR WER, 숫자/고유명사 오류가 핵심 리스크다. full fine-tuning 없이 LoRA로 회복 가능한지 사내 데이터에서 검증해야 한다. 적용처는 회의록, 콜센터 QA, 장시간 음성 검색, 현장 리포트 분석이다.

## 7. Recommended R&D Roadmap

### Phase 0. Benchmark and Instrumentation

먼저 사내 workload를 Image/CV task와 Video task로 분리해 token budget, prefill latency, decode latency, KV cache memory, answer quality를 계측한다.

필수 지표:

- visual/audio token count before and after compression
- text prompt token count and retained retrieved-context token count
- compression location: pre-encoder, in-encoder, post-encoder/projector, decode/KV
- selector/compressor metrics: candidate token 수, retained token 수, compression ratio, selected region/segment coverage
- encoder latency
- LLM prefill latency
- decode tokens/sec
- peak GPU memory
- KV cache memory
- task metric: VQA accuracy, OCR exact match, VideoMME-style QA accuracy, hallucination score, ASR WER, text QA citation accuracy

### Phase 1. Low-Risk Plug-in Projects

1. Video: Dynamic density pruning / redundant frame pre-filter
2. Image/CV: Query-aware visual projector and OCR-safe evaluation
3. Serving: Multimodal KV cache budget manager

이 단계의 핵심은 기존 MLLM에 비교적 붙이기 쉬운 위치에서 압축 위치별 비용 절감 범위를 분리해 측정하는 것이다. Post-Encoder는 LLM context와 KV cache 비용 중심으로 줄이고, KV cache policy는 serving memory와 decode latency를 직접 다룬다.

### Phase 2. Accuracy-Sensitive Projects

1. Image/CV: Positional-preserving in-encoder merge
2. Video: Fine-grained grounding safe pruning
3. Image/CV/Video/Audio: Pre-Encoder redundancy feasibility study
4. Video: Multi-stage long-context compression runtime

이 단계는 task별 failure mode 분석과 fallback policy가 필수다.

### Phase 3. Omnimodal Expansion

1. Audio-guided video token compression
2. Audio token downsampling adapter
3. Text/visual/audio joint KV cache policy
4. Prompt/context compression for multimodal RAG

옴니모달 제품 또는 회의/콜센터/영상 분석 서비스가 명확해질 때 착수하는 것이 타당하다.

## 8. Practical Decision Matrix

| 후보 | PoC 난도 | 제품화 난도 | 재학습 필요 | 하드웨어 가속 친화성 | 정보 손실 리스크 | 추천 |
|---|---:|---:|---|---|---|---|
| Query-aware visual projector | 중 | 중 | LoRA/adapter 가능성 | 높음 | 중 | 강함 |
| Explainability-guided pruning | 중 | 중 | 낮음 | 높음 | 중 | 강함 |
| Positional-preserving merge | 중 | 높음 | 낮음~중 | 중 | 낮음~중 | 중 |
| Pre-Encoder redundancy feasibility study | 중 | 높음 | 낮음~중 | 중 | 중~높음 | 조건부 |
| Video redundant frame pre-filter | 낮음 | 중 | 낮음 | 높음 | 중~높음 | 강함 |
| Multi-stage video runtime | 높음 | 높음 | 낮음~중 | 중 | 중 | 중 |
| Fine-grained safe pruning | 중 | 높음 | 낮음 | 중 | 낮음 | 중 |
| KV cache budget manager | 중 | 높음 | 낮음 | 중 | 중 | 강함 |
| Audio-guided video compression | 중 | 중 | 낮음 | 높음 | 중 | 조건부 |
| Audio token adapter | 중 | 중 | LoRA 필요 | 높음 | 중 | 조건부 |
| Text prompt/context compression | 낮음~중 | 중 | 낮음~중 | 높음 | 중 | 조건부 |

## 9. Source Index

아래 목록은 planning reference다. TMLR/ACL/CVF/AAAI 등 게재 논문과 arXiv preprint가 섞여 있으므로, 제품 채택 전에는 공개 코드, 라이선스, 하드웨어 재현성, 사내 benchmark 재현 여부를 별도로 확인해야 한다. 2026년 preprint 계열은 특히 방법론 후보로 취급하고, 수치는 내부 재측정 전까지 의사결정용 상한선으로만 사용한다.

- Shao et al., "A Survey of Token Compression for Efficient Multimodal Large Language Models", TMLR 2026. https://openreview.net/forum?id=G2od9JVHkE
- Alayrac et al., "Flamingo: a Visual Language Model for Few-Shot Learning", NeurIPS 2022. https://papers.nips.cc/paper_files/paper/2022/hash/960a172bc7fbf0177ccccbb411a7d800-Abstract-Conference.html
- Li et al., "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models", 2023. https://arxiv.org/abs/2301.12597
- Dai et al., "InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning", 2023. https://arxiv.org/abs/2305.06500
- Dai et al., "NVLM: Open Frontier-Class Multimodal LLMs", 2024. https://arxiv.org/abs/2409.11402
- Bai et al., "Qwen2.5-VL Technical Report", 2025. https://arxiv.org/abs/2502.13923
- Liu et al., "NVILA: Efficient Frontier Visual Language Models", CVPR 2025. https://arxiv.org/abs/2412.04468
- Xu et al., "Qwen2.5-Omni Technical Report", 2025. https://arxiv.org/abs/2503.20215
- Zhang et al., "LLaVA-Mini: Efficient Image and Video Large Multimodal Models with One Vision Token", 2025. https://arxiv.org/abs/2501.03895
- Li et al., "TokenPacker: Efficient Visual Projector for Multimodal LLM", 2024. https://arxiv.org/abs/2407.02392
- Bolya et al., "Token Merging: Your ViT but Faster", ICLR 2023. https://arxiv.org/abs/2210.09461
- Lei et al., "Generic Token Compression in Multimodal Large Language Models from an Explainability Perspective", 2025. https://arxiv.org/abs/2506.01097
- Yang et al., "VisionZip: Longer is Better but Not Necessary in Vision Language Models", CVPR 2025. https://arxiv.org/abs/2412.04467
- "VisionSelector: End-to-End Learnable Visual Token Compression for Efficient Multimodal LLMs", 2025. https://arxiv.org/abs/2510.16598
- "Attend Before Attention: Efficient and Scalable Video Understanding via Autoregressive Gazing", CVPR 2026. https://arxiv.org/abs/2603.12254
- "PixelPrune: Pixel-Level Adaptive Visual Token Reduction via Predictive Coding", 2026. https://arxiv.org/abs/2604.00886
- "V-PRUNE: Semantic-Aware Patch Pruning Before Tokenization in Vision-Language Model Inference", 2025. https://www.mdpi.com/2076-3417/15/17/9463
- "HiPrune: Hierarchical Attention for Efficient Token Pruning in Vision-Language Models", 2025/2026. https://arxiv.org/abs/2508.00553
- "A Glimpse to Compress: Dynamic Visual Token Pruning for Large Vision-Language Models", 2025. https://arxiv.org/abs/2508.01548
- "VisionTrim: Unified Vision Token Compression for Training-Free MLLM Acceleration", ICLR 2026. https://arxiv.org/abs/2601.22674
- "EffiVLM-BENCH: A Comprehensive Benchmark for Evaluating Training-Free Acceleration in Large Vision-Language Models", 2025. https://arxiv.org/abs/2506.00479
- "Can Visual Input Be Compressed? A Visual Token Compression Benchmark for Large Multimodal Models", 2025. https://arxiv.org/abs/2511.02650
- "ReDiPrune: Relevance-Diversity Pre-Projection Token Pruning for Efficient Multimodal LLMs", 2026. https://arxiv.org/abs/2603.24680
- "FastVID: Dynamic Density Pruning for Fast Video Large Language Models", 2025. https://arxiv.org/abs/2503.11187
- "LongVU: Spatiotemporal Adaptive Compression for Long Video-Language Understanding", 2024. https://arxiv.org/abs/2410.17434
- "ForestPrune: High-ratio Visual Token Compression for Video Multimodal Large Language Models via Spatial-Temporal Forest Modeling", 2026. https://arxiv.org/abs/2603.22911
- "METok: Multi-Stage Event-based Token Compression for Efficient Long Video Understanding", EMNLP 2025. https://aclanthology.org/2025.emnlp-main.954/
- "DyCoke: Dynamic Compression of Tokens for Fast Video Large Language Models", CVPR 2025. https://arxiv.org/abs/2411.15024
- "HieraVid: Hierarchical Token Pruning for Fast Video Large Language Models", 2026. https://arxiv.org/abs/2604.01881
- "Sink-Token-Aware Pruning for Fine-Grained Video Understanding in Efficient Video LLMs", 2026. https://arxiv.org/abs/2604.20937
- "MEDA: Dynamic KV Cache Allocation for Efficient Multimodal Long-Context Inference", 2025. https://arxiv.org/abs/2502.17599
- "AirCache: Activating Inter-modal Relevancy KV Cache Compression for Efficient Large Vision-Language Model Inference", 2025. https://arxiv.org/abs/2503.23956
- "Revisiting Multimodal KV Cache Compression: A Frequency-Domain-Guided Outlier-KV-Aware Approach", 2025/2026. https://arxiv.org/abs/2511.16786
- "Positional Preservation Embedding for Multimodal Large Language Models", 2025/2026. https://arxiv.org/abs/2510.22936
- "Filter, Correlate, Compress: Training-Free Token Reduction for MLLM Acceleration", AAAI 2026. https://ojs.aaai.org/index.php/AAAI/article/view/42460
- "OmniZip: Audio-Guided Dynamic Token Compression for Fast Omnimodal Large Language Models", 2025/2026. https://arxiv.org/abs/2511.14582
- "OmniRefine: Alignment-Aware Cooperative Compression for Efficient Omnimodal Large Language Models", 2026. https://arxiv.org/abs/2605.12056
- "OmniSIFT: Modality-Asymmetric Token Compression for Efficient Omni-modal Large Language Models", 2026. https://arxiv.org/abs/2602.04804
- "Towards Audio Token Compression in Large Audio Language Models", 2025. https://arxiv.org/abs/2511.20973
- "HeadRouter: Dynamic Head-Weight Routing for Task-Adaptive Audio Token Pruning in Large Audio Language Models", 2026. https://arxiv.org/abs/2604.23717
- "Accelerating Transducers through Adjacent Token Merging", 2023. https://arxiv.org/abs/2306.16009
- "LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models", EMNLP 2023. https://arxiv.org/abs/2310.05736
- Microsoft Research, "LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models", EMNLP 2023. https://www.microsoft.com/en-us/research/publication/llmlingua-compressing-prompts-for-accelerated-inference-of-large-language-models/
- "LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression", 2023. https://arxiv.org/abs/2310.06839
- "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models", NeurIPS 2023. https://arxiv.org/abs/2306.14048
- "SnapKV: LLM Knows What You are Looking for Before Generation", 2024. https://arxiv.org/abs/2404.14469
- Lin et al., "ShowUI: One Vision-Language-Action Model for GUI Visual Agent", CVPR 2025. https://arxiv.org/abs/2411.17465
