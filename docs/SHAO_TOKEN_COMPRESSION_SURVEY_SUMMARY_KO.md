# 핵심 레퍼런스 요약: A Survey of Token Compression for Efficient Multimodal Large Language Models

원문: [OpenReview](https://openreview.net/forum?id=G2od9JVHkE), [PDF](https://openreview.net/pdf/b3e544f45b798bcadca3808ced0b1c79daee8641.pdf)  
논문: Kele Shao, Keda Tao, Kejia Zhang, Sicheng Feng, Mu Cai, Yuzhang Shang, Haoxuan You, Can Qin, Yang Sui, Huan Wang  
게재: Transactions on Machine Learning Research, 2026년 1월
최종 업데이트: 2026-05-25

## 문서 목적

이 문서는 원 논문을 단순 번역한 문서가 아니다. 사내 R&D 프로젝트 세팅을 위해 논문의 핵심 주장, 분류 체계, modality별 시사점, 평가 설계 힌트를 한국어로 재정리한 planning-oriented summary다. 메인 기획 문서인 [MLLM Token Compression R&D Pipeline Survey](./MLLM_TOKEN_COMPRESSION_RND_PIPELINE_SURVEY_KO.md)와 [전략 에세이 메모](./MLLM_TOKEN_COMPRESSION_RND_STRATEGY_ESSAY_KO.md)와 함께 읽는 것을 전제로 한다.

## 한 페이지 요약

| 질문 | 요약 답변 |
|---|---|
| 이 논문이 다루는 문제는 무엇인가? | MLLM에서 이미지, 비디오, 오디오 입력이 만드는 token 폭증과 그로 인한 prefill latency, attention FLOPs, KV cache memory 병목. |
| 왜 중요한가? | 텍스트보다 multimodal token이 훨씬 빠르게 커진다. 특히 긴 비디오는 frame 수와 해상도가 곱으로 작용해 context를 압도한다. |
| 핵심 분류는 무엇인가? | Modality 기준으로 image, video, audio를 나누고, mechanism 기준으로 transformation, similarity, attention, query 기반 압축을 나눈다. |
| 회사 R&D에는 어떻게 연결되는가? | Vision, 특히 video를 메인 축으로 삼고, image/CV는 OCR-safe compression과 query-aware projector를 병행 검토하는 방향이 타당하다. |
| 한 가지 알고리즘을 고르면 되는가? | 아니다. Pre-Encoder, In-Encoder, Post-Encoder, Decode/KV cache 단계별로 병목과 trade-off가 다르므로 pipeline 형태로 프로젝트를 구성해야 한다. |
| 가장 조심해야 할 점은 무엇인가? | 압축률만 보면 안 된다. OCR, small object, temporal grounding, hallucination, WER, evidence retention 같은 failure mode를 함께 봐야 한다. |

논문을 한 문장으로 줄이면 다음과 같다.

```text
멀티모달 입력은 텍스트보다 훨씬 많은 token을 만들기 때문에,
MLLM 효율화는 모델 크기 축소가 아니라 modality별 redundancy를
pipeline 단계별로 줄이는 문제로 봐야 한다.
```

R&D planning 관점에서는 다음 네 축으로 모든 후보 기술을 태깅하는 것이 좋다.

| 태깅 축 | 예시 | 왜 필요한가 |
|---|---|---|
| Modality | Image, Video, Audio, Text/KV | 중복성이 modality별로 다르다. |
| Pipeline location | Pre-Encoder, In-Encoder, Post-Encoder, Decode/KV | 같은 압축률이라도 줄어드는 비용이 다르다. |
| Mechanism | Transformation, Similarity, Attention, Query | 구현 난도와 failure mode가 다르다. |
| Risk profile | OCR loss, grounding loss, WER increase, hallucination | 실제 서비스 적용 가능성을 판단한다. |

## 이 논문에서 얻는 실무 주장

이 논문을 내부 설득 자료로 사용할 때는 "최신 논문이 많다"보다 아래 주장 구조가 더 중요하다.

| 주장 | 논문 기반 근거 | 사내 R&D 해석 |
|---|---|---|
| Multimodal token은 제품 병목이다 | text보다 image/video/audio token scale이 훨씬 빠르게 커진다. | MLLM 제품화 비용은 모델 크기뿐 아니라 입력 token budget으로 결정된다. |
| Vision, 특히 video가 우선순위다 | video는 frame 수와 해상도가 곱으로 증가해 가장 큰 token explosion을 만든다. | long-video QA, CCTV/제조/강의/회의 영상 처리 R&D를 우선 잡는다. |
| Compression 위치가 효과를 결정한다 | Pre/In/Post/KV 단계마다 줄어드는 비용이 다르다. | 논문 후보를 나열하지 말고 pipeline stage별 프로젝트로 쪼갠다. |
| 압축은 accuracy만 보면 안 된다 | modality별 failure mode가 다르고 benchmark가 task-sensitive하다. | OCR, grounding, hallucination, WER, latency, memory를 함께 평가한다. |
| 단일 기술보다 pipeline이 필요하다 | image, video, audio의 redundancy가 다르며 mechanism도 다양하다. | 사내 공통 token budget/evaluation framework를 만들고 프로젝트 후보를 병렬 검증한다. |

따라서 이 논문이 주는 실무 결론은 다음이다.

```text
MLLM token compression은 "좋아 보이는 pruning 논문 하나를 붙이는 작업"이 아니라,
서비스 입력별 token budget을 계측하고,
압축 위치별 비용 절감과 정보 손실을 통제하는 R&D 파이프라인 구축 과제다.
```

## 0. 보고용 핵심 메시지

이 논문은 "MLLM을 더 작게 만들자"가 아니라 "멀티모달 입력이 만드는 token 폭증을 어디서 어떻게 줄일 것인가"를 다룬다. 리더십 보고에서는 다음 세 문장으로 요약할 수 있다.

```text
1. MLLM의 실서비스 병목은 고해상도 이미지, 긴 비디오, 장시간 오디오가 만드는 multimodal token 수에서 발생한다.
2. 특히 video는 frame 수와 해상도가 곱으로 증가하므로, vision 중심 token compression은 제품화 비용을 줄이는 핵심 R&D 축이다.
3. 압축은 한 지점에서 끝나지 않으며, Pre-Encoder, In-Encoder, Post-Encoder, KV cache를 나눠 프로젝트 파이프라인으로 설계해야 한다.
```

내부 핵심 멤버 관점에서는 이 논문을 top-level taxonomy로 쓰는 것이 좋다. 즉, 각 후보 기술을 `modality`, `pipeline location`, `mechanism`, `risk` 네 축으로 태깅해 프로젝트 후보군을 만들고, 실제 PoC에서는 모델/데이터/서비스 latency budget별로 검증해야 한다.

## 1. 이 논문을 왜 핵심 레퍼런스로 보는가

이 논문은 MLLM token compression을 "효율화 기법 목록"이 아니라 하나의 연구 지형으로 정리한다. 핵심은 세 가지다.

첫째, multimodal long context의 병목은 텍스트보다 이미지, 비디오, 오디오 token에서 훨씬 강하게 발생한다. 고해상도 이미지, 긴 비디오, 장시간 오디오는 LLM context window를 빠르게 채우고, attention FLOPs와 KV cache memory를 동시에 증가시킨다.

둘째, modality마다 중복성이 다르다. 이미지는 공간적 중복, 비디오는 공간적 중복과 시간적 중복, 오디오는 시간적 중복과 주파수 중복이 핵심이다. 따라서 하나의 generic pruning rule로 모든 modality를 해결하기 어렵다.

셋째, 압축 방법은 modality 기준뿐 아니라 mechanism 기준으로도 봐야 한다. 논문은 transformation-based, similarity-based, attention-based, query-based approach로 방법론을 나눈다. 이 분류는 사내 R&D 프로젝트를 "무엇을 줄이는가"와 "어떻게 줄이는가"로 동시에 판단하게 해준다.

## 2. 논문의 핵심 문제의식

MLLM은 LLM에 vision/audio encoder와 projector를 붙여 이미지, 비디오, 오디오를 텍스트 context처럼 처리한다. 이 구조는 범용성이 높지만, multimodal input이 들어오는 순간 token 수가 폭증한다.

논문은 token scale의 차이를 직관적으로 보여준다.

```text
Text, 10K words        -> 약 13K tokens
4K UHD image           -> 약 32K tokens
2-hour audio           -> 약 720K tokens
90-minute video        -> 약 54M tokens
```

중요한 점은 video다. 비디오는 frame 수와 해상도가 곱으로 늘어난다. 여기에 tile, thumbnail, multi-scale crop까지 붙으면 LLM이 감당해야 할 visual token 수가 실서비스 요구보다 훨씬 커진다. 논문은 특히 multimodal token이 많은 작업에서 전체 context의 대부분을 차지할 수 있음을 지적한다.

이 때문에 token compression은 단순 inference optimization이 아니다. long-context MLLM을 실제 제품으로 쓰기 위한 전제 조건에 가깝다.

## 3. MLLM 구조 관점의 정리

논문은 MLLM 추론 흐름을 다음처럼 본다.

```text
Image / Video / Audio
        |
Vision / Audio Encoder
        |
Projector
        |
Large Language Model
        |
Language Response
```

LLM에 들어가는 context는 대략 세 묶음이다.

```text
System tokens + Multimodal tokens + Text prompt tokens
```

여기서 multimodal tokens가 병목이다. 시스템 프롬프트나 사용자 질문은 상대적으로 짧은 경우가 많지만, 이미지/비디오/audio feature는 수천에서 수천만 token까지 커질 수 있다. 따라서 MLLM 효율화는 LLM 자체만 줄이는 방식으로는 충분하지 않다. vision/audio side에서 token을 줄이거나, projector/LLM/KV cache 단계에서 multimodal token의 생애주기를 관리해야 한다.

## 4. 논문의 두 축 분류

### 4.1 Modality 중심 분류

논문은 token compression 방법을 먼저 주된 데이터 modality로 나눈다.

| 분류 | 핵심 중복성 | 압축의 목표 |
|---|---|---|
| Image-centric compression | 공간적 중복 | 배경, 반복 texture, 인접 patch 유사성을 줄인다. |
| Video-centric compression | 공간 + 시간 중복 | 중복 frame, 정적 배경, 느린 움직임, 긴 temporal context를 줄인다. |
| Audio-centric compression | 시간 + 주파수 중복 | 무음, 배경 소음, 반복 음향, 낮은 정보량의 frequency/time segment를 줄인다. |

이 분류가 중요한 이유는 프로젝트 기획에 직접 연결되기 때문이다. 예를 들어 image compression에서 잘 되는 saliency pruning이 video에서는 temporal evidence를 놓칠 수 있다. 반대로 video keyframe sampling은 image OCR에는 의미가 없다.

### 4.2 Mechanism 중심 분류

논문은 방법론을 다시 네 가지 mechanism으로 나눈다.

| Mechanism | 한글 설명 | 대표 아이디어 | 장점 | 주의점 |
|---|---|---|---|---|
| Transformation-based | 변환 기반 | pooling, convolution, token stacking, token unshuffle | 구조가 단순하고 hardware-friendly | semantic importance를 직접 보지 않으면 중요한 token도 희석될 수 있음 |
| Similarity-based | 유사도 기반 | 가까운 embedding/token을 merge 또는 group | redundant token 제거에 직관적 | 작은 객체나 드문 이벤트가 다수 token에 묻힐 수 있음 |
| Attention-based | 어텐션 기반 | attention score로 중요 token 선택 | 모델 내부 중요도와 연결 가능 | attention이 항상 faithful explanation은 아님 |
| Query-based | 질의 기반 | 사용자 질문과 관련 높은 token 보존 | VQA/video QA처럼 task-specific 입력에 적합 | 질문에 드러나지 않은 배경 근거를 잃을 수 있음 |

사내 프로젝트를 설계할 때는 "modality"와 "mechanism"을 함께 적는 것이 좋다. 예를 들어 `Video / Post-Encoder / Query-based`와 `Image / Pre-Encoder / Similarity-based`는 둘 다 token compression이지만 구현 난도, 실패 양상, 제품 적용처가 다르다.

## 5. Image-centric compression 요약

이미지는 공간적 중복이 핵심이다. 고해상도 이미지일수록 patch token 수가 늘어나고, ViT attention 비용이 증가한다.

논문 관점에서 image compression은 크게 다음 문제를 푼다.

1. 배경과 반복 texture token이 너무 많다.
2. OCR, document QA, chart/table QA에서는 작은 영역이 답변 근거가 된다.
3. LLM에 들어가기 전 visual token 수가 많아 prefill latency와 KV cache가 커진다.

기술 방향은 다음과 같다.

| 방향 | 설명 | 실무 해석 |
|---|---|---|
| Pre-Encoder patch reduction | encoder 전 pixel/patch를 줄임 | encoder 비용까지 줄일 수 있지만 복구 불가 리스크가 큼 |
| In-Encoder pruning/merging | ViT layer 내부에서 token을 줄임 | 정보 보존 가능성이 있으나 모델 내부 수정 필요 |
| Post-Encoder visual token pruning | encoder 후 LLM 입력 전 token을 줄임 | 기존 MLLM에 붙이기 쉬운 PoC 후보 |
| Query-aware projector | 질문과 관련 높은 visual token만 보존 | VQA와 문서 QA에 유리하지만 query bias 주의 |

기획 관점에서는 image를 다시 둘로 나눠야 한다.

```text
일반 자연 이미지:
  배경/반복 texture가 많아 pruning 여지가 큼

문서/표/UI/OCR 이미지:
  작은 글자, 선, cell, 아이콘이 근거가 될 수 있어 보수적 압축 필요
```

## 6. Video-centric compression 요약

비디오는 이 논문에서 가장 token explosion이 심한 modality다. 90분 영상이 수천만 token으로 커질 수 있다는 예시는 video compression의 필요성을 직접 보여준다.

Video compression의 핵심은 세 가지다.

```text
1. 시간 축 중복:
   비슷한 frame이 계속 반복됨

2. 공간 축 중복:
   각 frame 안에서도 배경과 반복 texture가 많음

3. 질의 관련성:
   질문에 필요한 장면은 전체 영상 중 일부일 수 있음
```

논문은 video token compression 평가의 어려움도 지적한다. 일부 video QA benchmark에서는 sparse frame sampling만 잘해도 성능이 유지될 수 있다. 이 경우 실제 token compression 방법의 효과가 과소평가되거나, 반대로 쉬운 benchmark 때문에 과대평가될 수 있다. fine-grained temporal grounding, hallucination, open-ended generation을 함께 봐야 한다.

사내 기획 관점에서 video는 최소 세 단계로 나눠야 한다.

| 단계 | 질문 | 대표 접근 |
|---|---|---|
| Frame/segment level | 어느 시간 구간을 볼 것인가 | shot detection, frame similarity, event-aware sampling |
| Patch/token level | 선택된 frame 안에서 어느 영역을 볼 것인가 | spatial pruning, gaze-style selection, density pruning |
| KV/runtime level | 이미 들어간 visual context를 얼마나 유지할 것인가 | visual KV retention, dynamic cache policy |

따라서 video R&D는 단일 module보다 pipeline design으로 보는 것이 맞다.

## 7. Audio-centric compression 요약

오디오는 시간 축과 주파수 축이 모두 중요하다. 긴 음성 입력은 duration에 따라 token 수가 증가하고, 배경 소음이나 무음 구간은 정보량이 낮다. 음악/환경음/음성은 각각 redundancy 양상이 다르기 때문에 audio compression도 task-aware해야 한다.

논문 관점에서 audio compression은 다음 후보로 나뉜다.

| 방향 | 설명 | 적용처 |
|---|---|---|
| Silence/noise filtering | 무음 또는 stationary noise 구간 제거 | 회의록, 콜센터, 장시간 음성 검색 |
| Temporal downsampling | audio encoder output을 stride/pooling/stacking으로 줄임 | ASR, speech translation, audio QA |
| Similarity-based audio merge | 인접 audio token이 유사하면 병합 | sound event QA, audio captioning |
| Cross-modal audio guidance | audio salience로 video token budget 조정 | 강의, 인터뷰, 회의 영상 QA |

주의점은 ASR 품질이다. audio token을 줄이면 WER, 숫자, 고유명사, speaker boundary 오류가 늘 수 있다. 영상과 결합할 때는 audio가 중요한 장면을 알려주는 강한 signal이 될 수 있지만, 무음 visual event에는 취약하다.

## 8. 논문이 주는 평가 설계 힌트

논문은 token compression 연구에서 평가가 쉽지 않다고 본다. 단순 accuracy와 compression ratio만으로는 부족하다.

필수 평가 축은 다음과 같다.

| 평가 축 | 이유 |
|---|---|
| Accuracy / task score | 압축 후 기본 성능 유지 여부 |
| Latency / TTFT | 실제 사용자 체감과 prefill 병목 |
| FLOPs / throughput | 서버 비용과 batch capacity |
| Peak GPU memory | 긴 context와 KV cache 병목 |
| Hallucination | visual/audio evidence 손실 여부 |
| Grounding | 답변이 실제 frame/region/audio segment와 맞는지 |
| OCR / small object | compression에 취약한 fine-grained 능력 |
| Ablation by compression location | Pre/In/Post/KV 중 어디서 줄였는지 구분 |

특히 video는 benchmark를 조심해야 한다. coarse multiple-choice QA만 보면 compression이 좋아 보일 수 있다. 하지만 실제 제품은 open-ended QA, 근거 frame 제시, 세부 temporal reasoning을 요구한다.

## 9. 사내 R&D 기획으로 번역한 핵심 메시지

### 9.1 Vision을 메인 축으로 잡는 것이 타당하다

논문 구조상 image와 video는 token explosion의 핵심이다. 특히 video는 text 대비 몇 천 배 token scale로 커질 수 있어, 회사의 장기 R&D 파이프라인에서 가장 큰 비용 절감 여지가 있다.

초기 검토 축은 다음이 합리적이다. 이는 특정 알고리즘의 채택 순서가 아니라, Shao et al.의 modality별 token explosion 관점을 사내 R&D 언어로 번역한 것이다.

```text
1. Video: frame/patch/KV를 함께 줄이는 multi-stage compression
2. Image/CV: query-aware projector + OCR-safe compression
3. Serving: modality-aware KV cache manager
4. Audio/Text: omnimodal 확장과 runtime budget 보조 축
```

### 9.2 Compression location을 명시해야 한다

같은 50% token reduction이라도 위치에 따라 의미가 다르다.

| 위치 | 줄어드는 비용 | 기획 해석 |
|---|---|---|
| Pre-Encoder | encoder + LLM + KV | 효과는 크지만 위험도 큼 |
| In-Encoder | encoder 일부 + downstream 일부 | 성능 보존 가능성, 구현 난도 높음 |
| Post-Encoder | LLM prefill + KV | 빠른 PoC 후보, encoder 비용은 남음 |
| Decode/KV | serving memory + decode latency | 제품 운영 비용에 직접 연결 |

### 9.3 Modality별 fallback policy가 필요하다

압축은 실패할 수 있다. 특히 다음 입력은 보수적으로 처리해야 한다.

| 입력 유형 | 위험 |
|---|---|
| OCR/document image | 작은 글자 token 제거 |
| chart/table | cell boundary, 숫자, 축 label 손실 |
| industrial inspection | 작은 결함, 미세 texture 손실 |
| long video event detection | 짧은 이벤트 frame 누락 |
| audio ASR | 고유명사, 숫자, speaker turn 손실 |
| RAG text | 근거 문장 제거로 citation 오류 |

## 10. 이 논문에서 바로 뽑을 수 있는 프로젝트 후보

| 후보 | 논문 분류 | 우리 문서의 프로젝트 연결 |
|---|---|---|
| Query-aware visual projector | Image / Query-based / Post-Encoder | Project 1 |
| Explanation-guided pruning | Image / Attention-based / Post-Encoder | Project 2 |
| Positional-preserving merge | Image/Video / Similarity-based / In-Encoder | Project 3 |
| Pre-Encoder redundancy feasibility study | Image/Video/Audio / Pre-Encoder | Project 4 |
| Video dynamic density pruning | Video / Attention or Query-based / Post-Encoder | Project 5 |
| Multi-stage long-video compression | Video / Multi-stage | Project 6 |
| KV cache budget manager | Cross-modal / Decode | Project 8 |
| Audio-guided video compression | Audio + Video / Cross-modal | Project 9 |
| Audio token downsampling adapter | Audio / Transformation-based / Post-Encoder | Project 10 |

## 11. 핵심 섹션별 추가 레퍼런스와 설득 포인트

Shao et al. survey를 리더십 설득 자료로 쓸 때는 각 섹션을 외부 대표 연구와 연결해 읽는 것이 좋다. 아래 레퍼런스는 "우리가 특정 big tech 방법을 그대로 따라야 한다"는 의미가 아니라, token budget control이 이미 frontier multimodal architecture의 공통 설계 문제가 되었다는 근거다.

| Shao 섹션 | 같이 볼 레퍼런스 | 설득 포인트 |
|---|---|---|
| 2. Background / MLLM architecture | Flamingo, 2022; BLIP-2, 2023; InstructBLIP, 2023 | DeepMind와 Salesforce 계열 모델은 visual feature를 그대로 LLM에 넣지 않고, Perceiver Resampler/Q-Former/query transformer로 고정 또는 질의 기반 token을 만든다. projector는 단순 연결부가 아니라 압축/선택 지점이다. |
| 3. Image-centric / transformation | Qwen2.5-VL, 2025; NVLM, 2024 | Alibaba/NVIDIA 계열 VLM은 dynamic resolution, window attention, tile-based high-resolution design으로 OCR/document/GUI류 high-resolution 입력을 다룬다. 이미지 R&D는 단순 resizing이 아니라 resolution-aware token budget 문제다. |
| 3. Image-centric / similarity and in-encoder | ToMe, 2023; VisionZip, 2025 | Meta AI가 참여한 ToMe는 training-free token merging이 image/video/audio ViT 효율화에 공통으로 쓰일 수 있음을 보였다. similarity merge는 Project 3의 기본 근거다. |
| 3. Image-centric / query-based | BLIP-2, InstructBLIP, LLaVA-Mini | 질문 또는 learnable query로 visual evidence를 추출하는 방식은 이미 검증된 구조다. Image/CV query-aware projector는 빠른 PoC 후보로 타당하다. |
| 4. Video-centric | NVILA, 2025; Qwen2.5-VL, 2025; FastVID, 2025; METok, 2025 | Video는 frame/time/resolution이 곱으로 늘어난다. NVIDIA NVILA의 scale-then-compress, Qwen2.5-VL의 long-video/time encoding, FastVID/METok의 pruning/runtime 결과는 video를 1순위 R&D 축으로 둘 근거다. |
| 5. Audio-centric / cross-modal | Qwen2.5-Omni, 2025; Towards Audio Token Compression, 2025; OmniZip, 2025/2026 | Audio는 독립 token 폭증 축이면서 video salience signal이다. Streaming/omnimodal 모델은 audio-video alignment와 block-wise 처리까지 함께 설계한다. |
| 6. Insights / combining methods | VisionZip, NVILA, MEDA | 단일 pruning보다 selection + merging, scale + compress, layer-wise KV allocation처럼 stage별 조합이 강하다. 내부 프로젝트도 단일 알고리즘이 아니라 pipeline으로 설계해야 한다. |
| 7. Applications | ShowUI, 2025; medical imaging MLLM/WSI 계열; robotics/video navigation 계열 | GUI, 의료, 로봇/자율주행은 작은 visual evidence가 중요하다. 압축률보다 fallback, grounding, OCR/small-object 보존 정책이 제품화 기준이 된다. |

리더십 보고용으로는 다음 문장이 핵심이다.

```text
Token compression은 학술적 pruning 실험이 아니라,
frontier multimodal system들이 고해상도 이미지, 장시간 비디오,
스트리밍 오디오-비디오, 긴 prompt/KV cache를 처리하기 위해
공통적으로 채택하고 있는 architecture/runtime 설계 축이다.
```

## 12. 한계와 읽을 때 주의할 점

이 논문은 survey이므로 특정 방법 하나의 성능을 보증하지 않는다. 또한 2025-2026년 분야가 빠르게 바뀌고 있어, 개별 논문의 benchmark 수치는 모델, dataset, compression ratio, hardware 조건에 따라 달라질 수 있다.

따라서 이 논문은 다음처럼 써야 한다.

```text
좋은 사용법:
  R&D taxonomy, 후보 기술군 발굴, 평가 설계, 프로젝트 분류 기준

나쁜 사용법:
  특정 compression ratio나 speedup을 모든 모델에 그대로 적용한다고 가정
```

사내 프로젝트에서는 이 survey를 top-level map으로 쓰고, 실제 PoC 단계에서는 각 후보 논문의 official implementation, benchmark setting, target model compatibility를 별도로 검증해야 한다.
