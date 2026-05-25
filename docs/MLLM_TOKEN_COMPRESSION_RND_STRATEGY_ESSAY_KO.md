# MLLM Token Compression R&D Strategy Memo

작성일: 2026-05-24  
최종 업데이트: 2026-05-25  
목적: MLLM token compression R&D의 필요성, 방향성, 기대 효과를 리더십과 내부 핵심 멤버가 같은 흐름으로 이해하기 위한 전략 메모  
관련 문서: [R&D Pipeline Survey](./MLLM_TOKEN_COMPRESSION_RND_PIPELINE_SURVEY_KO.md), [Shao et al. Survey 한국어 요약](./SHAO_TOKEN_COMPRESSION_SURVEY_SUMMARY_KO.md)

이 문서는 표 중심의 기술 조사 문서가 아니라 전략 메모다. 구체 논문 목록과 수치 근거는 pipeline survey와 Shao et al. 요약 문서에 둔다. 본문에서 언급하는 효과는 논문 보고치와 R&D 기대 효과를 바탕으로 한 방향성이지, 사내 제품 성능 보장치가 아니다.

## 1. 왜 MLLM Token Compression이 지금 중요한가

MLLM의 경쟁력은 더 이상 텍스트 답변 품질만으로 결정되지 않는다. 실제 제품 요구는 이미 이미지, 문서, 화면, 영상, 음성, 대화 이력, 검색 문맥을 한 번에 처리하는 방향으로 이동하고 있다. 이 변화는 모델 구조보다 입력 구조에서 먼저 병목을 만든다. 텍스트만 다루던 LLM에서는 prompt 길이와 model size가 주된 비용 요인이었다. 그러나 MLLM에서는 이미지와 비디오, 오디오가 LLM context에 들어오는 순간 token 수가 급격히 커진다.

이 지점에서 token compression은 단순한 inference optimization이 아니다. long-video understanding, high-resolution document QA, multi-image reasoning, on-premise/edge multimodal assistant를 제품으로 만들기 위한 기반 기술이다. 우리가 이 문제를 풀지 않으면 더 큰 GPU를 쓰거나 더 짧은 입력만 허용하는 방식으로 제품 요구를 제한해야 한다. 반대로 token budget을 통제할 수 있으면 같은 모델과 같은 하드웨어에서도 더 긴 영상, 더 높은 해상도, 더 많은 동시 요청을 다룰 수 있다.

따라서 이 R&D의 핵심 질문은 "어떤 pruning 논문이 가장 좋은가"가 아니다. 질문은 더 실무적이어야 한다. 어떤 modality에서 token이 폭증하는가. 어느 pipeline 위치에서 줄여야 비용이 실제로 줄어드는가. 어느 task에서 정보 손실이 위험한가. 어떤 fallback과 평가 체계를 가져야 제품에 넣을 수 있는가. 이 질문들에 대한 답을 프로젝트 파이프라인으로 만드는 것이 이번 R&D의 목적이다.

## 2. 병목은 모델 크기가 아니라 Multimodal Token 길이에서 온다

대형 모델의 비용을 이야기할 때 흔히 parameter 수를 먼저 본다. 물론 model size는 중요하다. 하지만 MLLM에서 실서비스 병목은 parameter보다 input token length에서 더 직접적으로 발생하는 경우가 많다. 특히 prefill 단계와 KV cache memory는 입력 token 수에 민감하다. 이미지 한 장, 비디오 몇 분, 오디오 몇 시간은 텍스트 몇 문장과 같은 비용 구조가 아니다.

Shao et al.의 survey는 이 차이를 직관적으로 보여준다. 10K words는 약 13K token 수준으로 볼 수 있지만, 4K UHD image는 약 32K token, 2-hour audio는 약 720K token, 90-minute video는 약 54M token 규모까지 커질 수 있다. 수치 자체는 모델과 tokenizer, encoder 설정에 따라 달라질 수 있지만, 방향은 명확하다. multimodal input은 text보다 훨씬 빠르게 context를 채운다.

이 병목은 네 가지 비용으로 나타난다. 첫째, vision/audio encoder가 처리해야 하는 patch/frame/segment 수가 늘어난다. 둘째, projector를 거쳐 LLM에 들어가는 visual/audio token이 늘어나 prefill이 느려진다. 셋째, 긴 context는 KV cache memory를 키워 batch size와 동시 처리량을 제한한다. 넷째, multi-turn 환경에서는 이전 visual/audio context가 계속 비용으로 남는다. 여기서 중요한 점은 비용이 한 stage에만 묶이지 않는다는 것이다. encoder 비용, LLM prefill 비용, decode/KV 비용이 서로 다른 병목으로 나타나므로, 압축 위치를 분리해서 설계해야 한다.

결국 MLLM 제품의 비용 함수는 모델 크기만으로 설명되지 않는다. 입력 modality와 token budget이 제품 단가, latency, throughput, 적용 가능한 입력 길이를 결정한다. 그래서 token compression은 "있으면 좋은 기능"이 아니라 multimodal product surface를 넓히는 조건이다.

## 3. 왜 Vision, 특히 Video가 메인 축인가

이번 R&D의 중심은 vision이어야 한다. 그중에서도 video를 1순위로 두는 것이 타당하다.

이미지는 해상도에 따라 patch 수가 증가한다. 고해상도 문서, UI screenshot, 산업 검사 이미지, 의료 이미지처럼 작은 영역이 중요한 task에서는 visual token 수가 쉽게 커진다. 하지만 이미지는 기본적으로 하나의 spatial grid다. 압축 문제도 주로 공간적 중복, query 관련성, 작은 객체 보존 문제로 정리할 수 있다.

비디오는 다르다. 비디오는 spatial grid가 시간 축으로 반복된다. frame 수, 해상도, tile 수가 곱으로 붙는다. 여기에 thumbnail stream, multi-scale crop, long-context packing까지 더해지면 token 수는 매우 빠르게 증가한다. 많은 frame은 거의 같은 배경을 담고 있고, 질문과 무관한 시간 구간이 대부분일 수 있다. 동시에 중요한 이벤트는 몇 frame에만 짧게 나타날 수 있다. 이 양면성 때문에 비디오는 단순 sampling으로 해결하기 어렵다.

비디오를 줄인다는 것은 세 가지를 함께 줄인다는 뜻이다. 먼저 어느 시간 구간을 볼지 결정해야 한다. 그 다음 선택된 frame 안에서 어느 spatial patch를 볼지 결정해야 한다. 마지막으로 LLM에 들어간 visual context를 decoding 동안 얼마나 유지할지 결정해야 한다. frame-level, patch-level, KV-level이 모두 필요하다.

이 관점에서 video token compression은 단일 모듈이 아니라 pipeline 문제다. 그리고 이 pipeline을 잘 만들면 long-video QA, CCTV/제조 모니터링, 회의/강의 영상 이해, 스포츠/이벤트 분석, 로봇/자율주행 로그 분석 같은 제품 영역으로 확장할 수 있다. 이것이 video를 메인 R&D 축으로 두어야 하는 이유다.

## 4. Token Compression은 단일 알고리즘이 아니라 Pipeline 문제다

Token compression을 "어떤 token을 버릴 것인가"로만 보면 설계가 좁아진다. 더 중요한 질문은 "어디에서 줄일 것인가"다. 같은 50% token reduction이라도 encoder 전에 줄인 것과 LLM 입력 직전에 줄인 것, decoding 중 KV cache에서 줄인 것은 전혀 다른 효과를 낸다.

Pre-Encoder compression은 encoder에 들어가기 전에 frame, patch, audio segment를 줄인다. 성공하면 encoder FLOPs와 LLM prefill, KV cache까지 모두 줄일 수 있다. 효과는 크지만, 한 번 버린 정보는 복구하기 어렵다. OCR, small object, 짧은 video event처럼 작은 근거가 중요한 task에서는 위험하다.

In-Encoder compression은 vision/audio encoder 내부 layer에서 token을 병합하거나 제거한다. token drop보다 token merge나 recycling을 쓰면 정보 손실을 줄일 수 있다. 다만 모델 내부 구조와 attention mask, position encoding, serving framework와 맞물리기 때문에 통합 난도가 높다.

Post-Encoder 또는 projector compression은 encoder output 이후 LLM에 넣기 전 token을 줄인다. 기존 MLLM에 붙이기 상대적으로 쉽고 빠른 PoC에 적합하다. 특히 query-aware projector는 VQA, document QA, video QA에서 유리하다. 하지만 vision encoder 비용은 그대로 남기 때문에 고해상도/장시간 입력에서는 한계가 있다.

Decode/KV cache compression은 이미 LLM에 들어간 token의 생애주기를 관리한다. 이 단계는 model architecture보다 serving economics에 직접 연결된다. visual/audio token은 prefill 이후에도 KV cache를 차지한다. multi-turn multimodal assistant에서는 이 비용이 누적된다. 따라서 product serving 관점에서는 KV cache policy를 별도 프로젝트로 봐야 한다.

이 네 위치는 서로 대체 관계가 아니다. 좋은 전략은 이들을 조합하는 것이다. 빠른 PoC는 Post-Encoder에서 시작할 수 있다. 고해상도 입력과 긴 영상으로 갈수록 Pre-Encoder와 In-Encoder가 필요하다. multi-turn serving으로 가면 KV cache가 독립 병목이 된다.

## 5. 단계별 역할: Pre-Encoder, In-Encoder, Post-Encoder, KV Cache

Pre-Encoder 단계의 역할은 원본 입력에서 명백히 낮은 정보량의 frame, patch, segment를 초기에 차단하는 것이다. 비디오에서는 정적 배경, 중복 frame, 움직임 없는 구간이 대상이다. 이미지에서는 반복 texture, 빈 여백, pixel-level duplicate가 대상이 될 수 있다. 오디오에서는 무음, stationary noise, 낮은 spectral novelty 구간이 대상이다. 이 단계는 비용 절감 폭이 가장 크지만, domain-aware fallback이 반드시 필요하다.

In-Encoder 단계의 역할은 encoder 내부에서 redundancy를 더 정교하게 줄이는 것이다. ViT layer를 지나면서 token embedding은 더 semantic해진다. 이 지점에서는 단순 pixel 유사도보다 의미 기반 병합이 가능하다. 그러나 내부 layer를 건드리는 순간 구현과 serving 복잡도가 커진다. 따라서 이 단계는 정확도 민감 task, 예를 들어 문서 OCR, UI QA, 산업 검사, fine-grained video grounding에 적합하다.

Post-Encoder 또는 projector 단계의 역할은 LLM에 넘길 visual/audio token을 줄이는 것이다. 이 단계는 기존 모델과 결합하기 쉽다. 사용자 질문을 알고 있는 상태에서 token을 고를 수 있으므로 query-aware compression을 적용하기 좋다. "이 이미지에서 총액은 얼마인가"라는 질문에는 배경보다 숫자와 영수증 영역이 중요하다. "영상에서 사고가 언제 발생했는가"라는 질문에는 특정 event segment가 중요하다. 다만 질문에 드러나지 않은 배경 근거를 놓칠 수 있으므로 confidence와 fallback이 필요하다.

KV cache 단계의 역할은 생성 과정에서 context memory를 관리하는 것이다. visual token을 LLM에 넣는 순간 비용이 끝나는 것이 아니다. 그 token은 decoding 동안 cache로 남고, 긴 응답이나 multi-turn 대화에서 계속 영향을 준다. 따라서 visual/audio/text token을 같은 정책으로 보존하거나 제거하면 안 된다. modality-aware, layer-aware, task-aware cache budget이 필요하다.

## 6. Image/CV, Video, Audio, Text/KV의 관계

Image/CV는 video보다 작지만 제품 적용처가 명확하다. 문서 QA, screenshot QA, chart/table QA, 산업 검사, 상품 이미지 QA는 모두 고해상도 입력을 요구한다. 여기서는 query-aware projector가 빠른 후보이고, OCR-safe compression이 안전장치다. 일반 자연 이미지는 배경과 반복 texture를 줄일 여지가 크지만, 문서와 UI는 작은 글자와 얇은 선이 근거가 되므로 보수적으로 접근해야 한다.

Video는 메인 축이다. 비디오에서는 temporal redundancy와 spatial redundancy가 동시에 존재한다. 긴 영상의 대부분은 질문과 무관할 수 있지만, 중요한 이벤트는 짧고 희소하다. 따라서 video pipeline은 frame/segment filtering, spatial patch selection, projector compression, visual KV retention을 함께 설계해야 한다.

Audio는 독립 축이면서 video의 보조 신호다. 회의, 강의, 인터뷰 영상에서는 audio가 중요한 구간을 찾는 강한 signal이 된다. 말이 시작되는 구간, speaker가 바뀌는 구간, 음향 이벤트가 발생하는 구간은 video token budget을 높일 후보가 된다. 반대로 CCTV나 제조 영상처럼 무음 visual event가 중요한 도메인에서는 audio-guided pruning에 의존하면 안 된다. audio는 제품 도메인에 따라 guide가 될 수도 있고 noise가 될 수도 있다.

Text와 KV는 보조 축이지만 운영에서는 중요하다. visual token을 줄여도 system prompt, RAG context, conversation history가 길면 KV cache 병목은 남는다. 특히 document QA와 video RAG에서는 근거 문장을 줄이면 citation 오류가 생길 수 있다. 따라서 text prompt compression과 KV cache compression은 vision compression의 대체재가 아니라 serving budget manager의 일부로 봐야 한다.

정리하면, 전략의 중심은 vision이고 실행 우선순위는 video다. Image/CV는 빠른 제품화와 high-resolution QA를 위한 축이다. Audio는 omnimodal 확장과 video salience signal이다. Text/KV는 전체 serving economics를 닫는 축이다.

## 7. 우리가 해야 할 프로젝트 파이프라인

이 방향은 fringe optimization이 아니다. DeepMind의 Flamingo는 variable image/video features를 고정 크기 visual token으로 resample하는 구조를 사용했고, Salesforce의 BLIP-2/InstructBLIP는 Q-Former와 instruction-aware query transformer로 visual feature를 LLM 앞에서 압축/선택한다. Alibaba Qwen2.5-VL은 dynamic resolution과 window attention으로 native resolution 처리 비용을 줄이고, NVIDIA NVILA는 scale-then-compress를 명시적인 VLM 효율화 전략으로 둔다. Microsoft LLMLingua와 MEDA 계열 연구는 text prompt와 KV cache도 별도 compression 대상임을 보여준다. 즉, token compression은 특정 논문 하나의 아이디어가 아니라 frontier multimodal system들이 이미 채택하고 있는 설계 방향이다.

첫 번째로 해야 할 일은 token budget instrumentation이다. 어떤 입력에서 token이 얼마나 생기는지, encoder와 prefill과 decode 중 어디가 병목인지, KV cache memory가 어느 정도 쌓이는지 계측해야 한다. 계측 없이 compression 논문을 붙이면 개선처럼 보이는 숫자와 실제 제품 병목이 어긋날 수 있다. Phase 0은 모델 개선이 아니라 관측 가능성 확보가 목표다. 이 계측이 있어야 "압축률이 높다"와 "제품 비용이 줄었다"를 구분할 수 있다.

두 번째는 video dynamic token gating이다. 긴 영상에서 frame/segment redundancy를 줄이고, 선택된 frame 안에서 spatial token budget을 조정해야 한다. 이 프로젝트는 가장 직접적인 비용 절감 효과를 낼 가능성이 높다. 단, 평균 성능만 보면 안 된다. subtle event, temporal grounding, open-ended QA, hallucination을 함께 평가해야 한다.

세 번째는 image/CV query-aware projector다. 이 프로젝트는 기존 MLLM에 붙이기 쉬운 편이고, VQA/문서 QA/화면 QA에서 빠르게 PoC를 만들 수 있다. 질문과 관련 높은 visual token을 보존하고 나머지는 병합하거나 요약한다. 다만 OCR과 small object task에서는 fallback rule이 필요하다.

네 번째는 Pre-Encoder redundancy feasibility study다. Post-Encoder 방식은 LLM 비용을 줄이지만 encoder 비용은 남긴다. 고해상도 이미지, 긴 영상, 장시간 오디오에서 encoder 이전 압축이 유효할 수는 있지만, 이 축은 기본 방향이 아니라 조건부 검토 대상이다. 한 번 버린 정보는 복구하기 어렵기 때문에 domain-specific benchmark와 fallback policy가 먼저 있어야 한다.

다섯 번째는 multimodal KV cache budget manager다. 제품 serving을 생각하면 이 축은 반드시 필요하다. visual/audio/text token을 어떤 기준으로 cache에 남길지 정해야 한다. multi-turn assistant, long-video chat, RAG 결합 서비스에서는 KV cache가 비용과 latency를 지배할 수 있다.

여섯 번째는 accuracy-sensitive compression이다. Positional-preserving merge, token recycling, fine-grained grounding-safe pruning 같은 프로젝트는 빠른 PoC보다는 품질 방어에 가깝다. OCR, UI, 산업 검사, 의료/로봇/자율주행 로그처럼 작은 근거가 중요한 도메인에 필요하다.

마지막으로 audio와 text는 omnimodal expansion 단계에서 결합한다. audio-guided video pruning은 회의/강의/인터뷰 영상에 유리하다. audio token downsampling은 장시간 음성 QA와 콜센터 분석에 연결된다. prompt/context compression은 multimodal RAG에서 필요하다.

## 8. 기대 효과: 비용, Latency, 제품 확장성, 운영 가능성

기대 효과는 단일 speedup 숫자로 표현하면 안 된다. Token compression의 효과는 적용 위치에 따라 달라진다. Pre-Encoder는 encoder 비용까지 줄일 수 있고, Post-Encoder는 LLM prefill과 KV cache 비용을 줄인다. KV cache compression은 serving memory와 multi-turn latency를 줄인다. 따라서 효과를 주장할 때는 어떤 stage에서 어떤 비용이 줄었는지 분리해야 한다.

첫 번째 효과는 비용 절감이다. visual/audio token이 줄면 prefill FLOPs, encoder FLOPs, KV cache memory가 줄어든다. 같은 GPU에서 더 긴 입력을 처리하거나, 같은 입력에서 더 많은 요청을 batch 처리할 수 있다. 이는 단순 서버비 절감이 아니라 product capacity 증가다.

두 번째 효과는 latency 감소다. MLLM에서 사용자가 체감하는 지연은 종종 첫 token이 나오기 전 prefill 단계에서 발생한다. 고해상도 이미지나 긴 영상에서는 이 지연이 커진다. Token compression이 잘 작동하면 TTFT와 long-video QA 응답 시간을 줄일 수 있다.

세 번째 효과는 제품 확장성이다. 지금 dense processing으로는 비용상 어려운 입력을 열 수 있다. 4K 이미지, 장시간 영상, 다중 이미지 비교, 회의/강의 영상, CCTV/제조 라인 로그, multi-turn visual assistant가 여기에 해당한다. Token compression은 기존 제품을 조금 빠르게 하는 기술이 아니라, 처리 가능한 입력 범위를 넓히는 기술이다.

네 번째 효과는 운영 가능성이다. 좋은 compression pipeline은 단순히 token을 버리지 않는다. token budget, compression ratio, retained region, fallback reason, confidence를 로그로 남긴다. 이렇게 해야 실패했을 때 원인을 분석할 수 있다. "빠르지만 왜 틀렸는지 모르는 모델"은 제품에 넣기 어렵다. 운영 가능한 MLLM은 압축 정책도 관측 가능해야 한다.

## 9. 리스크와 평가 기준

가장 큰 리스크는 정보 손실이다. Token compression은 본질적으로 일부 정보를 덜 보겠다는 결정이다. 일반 VQA에서는 문제가 없어 보이는 압축률도 OCR, chart, table, UI, small object, industrial defect에서는 치명적일 수 있다. Video에서는 몇 frame짜리 이벤트를 놓칠 수 있고, audio에서는 고유명사나 숫자, speaker turn을 잃을 수 있다.

두 번째 리스크는 benchmark 착시다. 일부 multiple-choice video QA에서는 sparse sampling만으로도 성능이 잘 나올 수 있다. 그러면 compression 방법이 실제보다 좋아 보일 수 있다. 제품은 open-ended answer, evidence grounding, temporal localization, hallucination control을 요구한다. 따라서 benchmark는 accuracy 하나로 끝나면 안 된다.

세 번째 리스크는 hardware와 serving stack의 현실성이다. 이론적으로 token을 줄여도 dynamic shape, sparse attention, gather/scatter overhead, cache compaction 비용 때문에 실제 latency가 줄지 않을 수 있다. 그래서 PoC는 반드시 latency, peak memory, throughput, TTFT를 함께 측정해야 한다.

평가 기준은 다음처럼 잡아야 한다. 첫째, task score를 본다. VQA accuracy, OCR exact match, VideoMME-style score, ASR WER, citation accuracy가 필요하다. 둘째, 비용 지표를 본다. encoder latency, prefill latency, decode tokens/sec, peak GPU memory, KV cache memory를 측정한다. 셋째, 안전 지표를 본다. hallucination, grounding, retained evidence coverage, fallback rate를 기록한다. 넷째, compression 위치별 ablation을 한다. Pre-Encoder, In-Encoder, Post-Encoder, KV 중 어디서 줄였는지 분리하지 않으면 결과 해석이 불가능하다.

## 10. 결론: 어떤 의사결정이 필요한가

지금 필요한 결정은 특정 논문이나 특정 알고리즘을 하나 고르는 것이 아니다. 필요한 결정은 R&D 파이프라인의 형태다. Vision을 중심으로, 특히 video를 1순위로 두고, image/CV와 KV cache를 빠른 PoC 축으로 붙여야 한다. Audio와 text는 후순위로 밀어내는 것이 아니라 omnimodal expansion과 serving budget 관리의 일부로 설계해야 한다.

우리가 바로 결정해야 할 것은 네 가지다. 첫째, token budget instrumentation을 공통 기반 과제로 둘 것인가. 둘째, video dynamic token gating을 1순위 PoC로 둘 것인가. 셋째, image/CV query-aware projector와 OCR-safe evaluation을 빠른 실험 축으로 둘 것인가. 넷째, KV cache budget manager를 serving R&D 과제로 병렬 추진할 것인가.

이 결정이 내려지면 프로젝트는 비교적 명확해진다. 먼저 현재 workload에서 token과 latency와 memory를 계측한다. 다음으로 video와 image/CV에서 서로 다른 compression 위치를 검증한다. 이후 accuracy-sensitive task와 multi-turn serving으로 확장한다. 마지막으로 audio/text budget까지 포함해 omnimodal pipeline으로 넓힌다.

MLLM token compression은 단순한 최적화 과제가 아니다. 고해상도와 장시간 입력을 다루는 제품을 실제로 운영 가능한 비용과 latency로 만들기 위한 기반 기술이다. 우리가 만들어야 하는 것은 하나의 pruning 모듈이 아니라, modality별 token budget을 계측하고, pipeline 단계별로 줄이며, 정보 손실을 평가하고, 필요할 때 fallback하는 운영 가능한 R&D 시스템이다.
