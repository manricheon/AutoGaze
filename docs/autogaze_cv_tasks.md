# AutoGaze를 다양한 CV Task에 적용하기

## 1. 핵심 아이디어

AutoGaze는 원래 NVILA의 비디오 이해 파이프라인에서  
"어떤 공간 토큰을 LLM에 넘길지" 를 선택하는 **경량 gaze 선택 모듈**이다.  
그 핵심 출력물인 **`gazing_mask`(공간 attention 마스크)** 와  
**비전 인코더 특징(feature)**은 LLM 외의 디코더에도 그대로 연결할 수 있다.

```
          ┌─────────────────────────────────────────────────────────┐
          │                     AutoGaze                            │
          │                                                          │
  입력 ──► │  ShallowVideoConvNet  ──►  connector  ──►  LLaMA decoder│
  프레임   │      (B,T,196,192)         (pos embed)   (gaze 선택)    │
  (224px) │                                                          │
          │     encoder_features               gazing_mask           │
          │     (B, T, 196, 192)            (B, T, 196) boolean      │
          └──────────┬──────────────────────────┬────────────────────┘
                     │                          │
          ┌──────────▼──────────┐    ┌──────────▼──────────┐
          │   Task Decoder      │◄───│  gaze-selected       │
          │ (Recognition /      │    │  features            │
          │  Detection /        │    │  (B, T, k, 192)      │
          │  Segmentation /     │    │  k = ratio × 196     │
          │  Depth)             │    └──────────────────────┘
          └─────────────────────┘
```

### 기존 NVILA 파이프라인과 비교

| 항목 | 기존 (비디오 이해) | 새 CV Task |
|------|------------------|------------|
| 백본 | SigLIP ViT 392px (NVILA) | AutoGaze 자체 인코더 224px |
| gaze 선택 | AutoGaze → 선택된 ViT 토큰 | AutoGaze → 선택된 AG 토큰 |
| 디코더 | Qwen2 LLM | Recognition / Detection / Seg / Depth head |
| 출력 | 텍스트 | 클래스 / 박스 / 마스크 / 깊이 맵 |

---

## 2. 최소 수정 원칙

기존 `AutoGaze` 모델 코드는 **수정하지 않는다**.  
대신 `AutoGazeEncoder`라는 얇은 래퍼(thin wrapper)를 추가해  
내부 비전 인코더 특징을 노출시킨다.

```
autogaze/models/autogaze/autogaze.py        ← 변경 없음 ✅
autogaze/models/autogaze/autogaze_cv.py     ← 신규 (래퍼만 추가)
autogaze/decoders/                          ← 신규 (독립 모듈)
autogaze/tasks/{recognition,detection,...}  ← 신규 (기존 task 인터페이스 재사용)
```

---

## 3. 아키텍처 세부 설계

### 3.1 인코더 특징 추출 (`AutoGazeEncoder`)

AutoGaze 내부 `ShallowVideoConvNet` + `connector` 를 호출해  
224×224 프레임 → **(B, T, 196, 192)** 특징 맵을 반환한다.

- B: 배치, T: 프레임(이미지라면 1), N=196=14×14: 공간 토큰, C=192: 특징 차원

```python
from autogaze.models.autogaze.autogaze_cv import AutoGazeEncoder

encoder = AutoGazeEncoder(autogaze_model)
features = encoder.encode(video)  # (B, T, 196, 192)
```

### 3.2 gaze 마스크 (`AutoGaze.forward`)

기존 `AutoGaze.forward(inputs, gazing_ratio=0.5, generate_only=True)` 로  
**`gazing_mask`** (B, T, 196) 불리언 마스크를 얻는다.  
별도 compute 없이 기존 forward를 그대로 사용한다.

### 3.3 `AutoGazeCVModel` (통합 wrapper)

```python
from autogaze.models.autogaze.autogaze_cv import AutoGazeCVModel
from autogaze.decoders.recognition import RecognitionDecoder

model = AutoGazeCVModel(
    autogaze_model,
    decoder=RecognitionDecoder(num_classes=1000),
    freeze_autogaze=True,   # AutoGaze 파라미터 고정
    gazing_ratio=0.5,
)
pred = model(inputs)  # {'pred': logits, 'gaze_outputs': ...}
```

---

## 4. Task별 디코더 설계

### 4.1 Recognition (이미지/비디오 분류)

```
features (B,T,196,C)
    → 시간 풀링 → (B,196,C)
    → gaze-weighted spatial pooling → (B,C)
    → LayerNorm → Linear(C,512) → GELU → Linear(512,num_classes)
    → (B, num_classes) logits
```

- **gaze 마스크 사용법**: 공간 풀링 시 가중치로 활용 (중요 패치를 더 많이 반영)
- **손실**: Cross-entropy
- **파라미터 수**: ~250K (C=192, num_classes=1000)

### 4.2 Object Detection (DETR-lite)

```
features (B,T,196,C)
    → 시간 풀링 → (B,196,C)
    → TransformerDecoder (Q=100 learnable queries × memory=features)
    → gaze_mask → key padding mask (non-gazed region 억제)
    → class head: (B,Q,num_classes+1)
    → box head MLP: (B,Q,4)  [cx,cy,w,h] ∈ [0,1]
```

- **gaze 마스크 사용법**: 어텐션 key padding mask - gaze 선택 영역에만 쿼리가 어텐션
- **손실**: Classification CE + Box L1 (Hungarian matching은 향후 추가)
- **파라미터 수**: ~2.5M (3 decoder layers, 100 queries)

### 4.3 Semantic Segmentation (FPN-lite)

```
features (B,T,196,C)
    → 시간 풀링 → (B,196,C)
    → reshape → (B,C,14,14)
    → gaze gate (conv) × mask_map → 중요 영역 강조
    → 4×(Conv + BN + GELU + Upsample×2)
    → 14→28→56→112→224
    → Conv(16, num_classes, 1)
    → (B, num_classes, 224, 224)
```

- **gaze 마스크 사용법**: spatial gate - gaze 영역의 특징을 채널 방향으로 증폭
- **손실**: Cross-entropy (ignore_index=255)
- **파라미터 수**: ~300K

### 4.4 Depth Estimation (DPT-lite)

```
features (B,T,196,C)
    → 시간 풀링 → (B,196,C)
    → gaze-weighted scale: feat = feat + sigmoid(W·feat) × gaze_weight
    → reshape → (B,C,14,14)
    → 4×(Conv + BN + ReLU + Upsample×2)
    → Conv(16, 1, 1) + Sigmoid → × max_depth
    → (B, 1, 224, 224)
```

- **gaze 마스크 사용법**: 특징 스케일링 - gaze 영역의 특징 크기 조정
- **손실**: Scale-invariant depth loss (log-space)
- **파라미터 수**: ~280K

---

## 5. 학습 파이프라인

### 5.1 단계별 학습 전략

```
Stage 1 (freeze AutoGaze):
    AutoGaze 파라미터 고정
    Task decoder만 학습
    → 빠른 수렴, AutoGaze 표현이 task에 맞는지 검증

Stage 2 (fine-tune AutoGaze):
    AutoGaze 일부 (vision_model만) 소규모 학습률로 학습
    → end-to-end 최적화
```

### 5.2 Hydra 설정 확장

기존 `autogaze/configs/` 구조를 그대로 따른다:

```yaml
# autogaze/configs/task/recognition.yaml
_target_: autogaze.tasks.recognition.TaskRecognition
decoder:
  num_classes: 1000
  hidden_dim: 512
  dropout: 0.1

# autogaze/configs/task/detection.yaml
_target_: autogaze.tasks.detection.TaskDetection
decoder:
  num_classes: 80
  num_queries: 100
  num_decoder_layers: 3
```

### 5.3 데이터셋 요구사항

| Task | 권장 데이터셋 | 입력 | 어노테이션 |
|------|-------------|------|-----------|
| Recognition | ImageNet-1K | 224×224 이미지 | 클래스 레이블 |
| Detection | COCO 2017 | 임의 해상도 | bbox + 클래스 |
| Segmentation | ADE20K / COCO panoptic | 임의 해상도 | 픽셀 레이블 |
| Depth | NYU Depth V2 / KITTI | 640×480 / 1242×375 | 깊이 맵 |

---

## 6. 파일 구조

```
autogaze/
├── models/autogaze/
│   ├── autogaze.py                  ← 변경 없음
│   └── autogaze_cv.py               ← NEW: AutoGazeEncoder, AutoGazeCVModel
│
├── decoders/
│   ├── __init__.py                  ← NEW
│   ├── base.py                      ← NEW: TaskDecoder 추상 클래스
│   ├── recognition.py               ← NEW
│   ├── detection.py                 ← NEW
│   ├── segmentation.py              ← NEW
│   └── depth.py                     ← NEW
│
├── tasks/
│   ├── video_mae_reconstruction/    ← 기존
│   ├── recognition/
│   │   └── task_recognition.py      ← NEW
│   ├── detection/
│   │   └── task_detection.py        ← NEW
│   ├── segmentation/
│   │   └── task_segmentation.py     ← NEW
│   └── depth/
│       └── task_depth.py            ← NEW
│
└── configs/task/
    ├── recognition.yaml             ← NEW
    ├── detection.yaml               ← NEW
    ├── segmentation.yaml            ← NEW
    └── depth.yaml                   ← NEW

notebooks/
└── 08_autogaze_cv_tasks_ko.ipynb    ← NEW: 데모 노트북
```

---

## 7. 기술적 고려사항

### 7.1 특징 해상도 한계

AutoGaze 인코더는 **14×14 = 196 토큰**으로 고정된다 (224px / 16px stride).  
이는 의미 있는 공간 해상도이지만 픽셀 단위 정확도가 필요한 task에는 제한이 있다.

- **Recognition**: 문제 없음 (global feature)
- **Detection**: 14×14 → 224×224 업샘플링 시 작은 객체 검출 어려움
- **Segmentation**: 경계 부분 품질 저하 (하지만 의미론적 분할은 충분)
- **Depth**: 전반적 구조 파악은 가능, 세밀한 경계 어려움

### 향후 개선 방향
1. **더 강한 백본**: SigLIP ViT(392px, 1024-dim) + AutoGaze 선택 → task decoder 연결
2. **멀티스케일 특징**: AutoGaze scales='32+64+112+224'를 활성화하면 265 gaze 토큰 + 다중 해상도 특징 활용 가능
3. **Task-aware gaze**: task loss를 AutoGaze의 GRPO reward로 활용 → task에 특화된 gaze 패턴 학습

### 7.2 AutoGaze 파라미터 효율

| 컴포넌트 | 파라미터 수 | 역할 |
|----------|------------|------|
| ShallowVideoConvNet | ~0.5M | 특징 추출 (학습 필요) |
| Connector | ~0.1M | 위치 임베딩 (고정 가능) |
| LLaMA gaze decoder | ~4.7M | gaze 선택 (고정 추천) |
| **Task decoder** | 0.3~2.5M | **task별 예측** |

총 파라미터: ~5.6M (gaze 고정 시 task decoder만 학습 = ~0.3~2.5M)

---

## 8. 빠른 시작

```python
from autogaze.models.autogaze import AutoGaze
from autogaze.models.autogaze.autogaze_cv import AutoGazeCVModel
from autogaze.decoders.recognition import RecognitionDecoder
from autogaze.decoders.detection import DetectionDecoder
from autogaze.decoders.segmentation import SegmentationDecoder
from autogaze.decoders.depth import DepthDecoder

# 1. AutoGaze 로드
ag = AutoGaze.from_pretrained("weights/AutoGaze").eval()

# 2. Task 모델 생성
rec_model = AutoGazeCVModel(ag, RecognitionDecoder(num_classes=1000))
det_model = AutoGazeCVModel(ag, DetectionDecoder(num_classes=80))
seg_model = AutoGazeCVModel(ag, SegmentationDecoder(num_classes=150))
dep_model = AutoGazeCVModel(ag, DepthDecoder(max_depth=10.0))

# 3. 추론 (inputs = {'video': (B, T, C, H, W)})
cls_out = rec_model(inputs)['pred']      # (B, 1000)
det_out = det_model(inputs)['pred']      # {'boxes': (B,100,4), 'class_logits': ...}
seg_out = seg_model(inputs)['pred']      # (B, 150, 224, 224)
dep_out = dep_model(inputs)['pred']      # (B, 1, 224, 224)
```

---

## 9. 알려진 제약과 TODO

- [ ] DETR 스타일 Hungarian matching loss 구현
- [ ] `autogaze/datasets/image_classification.py` (ImageNet/COCO 데이터셋 래퍼)
- [ ] Hydra config YAML 완성 + train_cv.py 진입점
- [ ] Stage 2: end-to-end fine-tuning 실험
- [ ] 더 강한 백본(SigLIP 392px)으로의 확장 테스트
- [ ] Mask2Former 스타일 instance segmentation
