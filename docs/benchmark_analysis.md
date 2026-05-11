# AutoGaze PoC 벤치마크 분석 템플릿

이 문서는 실험 결과를 정리하기 위한 분석 템플릿입니다. dummy/stub 결과와 실제 benchmark 결과를 반드시 분리합니다.

## 1. Experiment Setting Summary

| Field | Value |
|---|---|
| Experiment ID | TBD |
| Result Type | dummy/stub / reproduced / extension / reported |
| AutoGaze | ON / OFF |
| Vision Encoder | TBD |
| MLLM / Decoder | TBD |
| Dataset | TBD |
| Task | Video VQA / Action Recognition |
| Frame Count | TBD |
| Resolution | TBD |
| Precision | TBD |
| Device | CUDA / MPS / CPU |
| Checkpoint | TBD |
| Git Commit | TBD |
| Config Path | TBD |

## 2. Public Claim vs Internal Measurement

| Metric | Reported by Paper/Model Card | Our Reproduction | Our Extension Result | Dummy/Stub Result | Notes |
|---|---:|---:|---:|---:|---|
| Token reduction | TBD | TBD | TBD | 0.0 for dummy full-token | scale/frame별 분리 |
| ViT latency speedup | TBD | TBD | TBD | N/A | encoder-side 여부 명시 |
| MLLM latency speedup | TBD | TBD | TBD | N/A | prefill/decode 분리 |
| VideoMME | TBD | TBD | TBD | N/A | official protocol 필요 |
| HLVid improvement | TBD | TBD | TBD | N/A | dataset availability 확인 |
| High-resolution long-video support | TBD | TBD | TBD | N/A | hardware limit 기록 |

주의:

- reported result와 reproduced result를 섞지 않습니다.
- 외부 모델이 AutoGaze를 사용했다고 명시적으로 보고하지 않은 경우 그렇게 주장하지 않습니다.

## 3. AutoGaze ON/OFF Comparison

| Pair | AutoGaze OFF | AutoGaze ON | Vision Encoder | MLLM | Dataset | Metric Delta | Latency Delta | Token Delta | Acceleration Type | Notes |
|---|---|---|---|---|---|---:|---:|---:|---|---|
| A1 vs A2 | A1 | A2 | modified SigLIP | NVILA | TBD | TBD | TBD | TBD | TBD | canonical comparison |
| A0 vs A3 | A0 | A3 | vanilla SigLIP | NVILA | TBD | TBD | TBD | TBD | compatibility-only until tested | A3 experimental |

## 4. Modified SigLIP vs Vanilla SigLIP

| Pair | AutoGaze | modified SigLIP | vanilla SigLIP | Dataset | Metric Delta | Latency Delta | Notes |
|---|---|---|---|---|---:|---:|---|
| A1 vs A0 | OFF | A1 | A0 | TBD | TBD | TBD | modified SigLIP effect |
| A2 vs A3 | ON | A2 | A3 | TBD | TBD | TBD | A3 not assumed compatible |

## 5. NVILA vs Other MLLM

| Experiment | Vision Encoder | MLLM | Integration Mode | Dataset | Metric | Latency | Token Count | Notes |
|---|---|---|---|---|---:|---:|---:|---|
| A2 | modified SigLIP | NVILA | native/original | TBD | TBD | TBD | TBD | canonical |
| Extension-Qwen | TBD | Qwen | official_processor | TBD | TBD | TBD | TBD | no direct token injection assumed |
| Extension-Generic | generic ViT | generic MLLM | dummy | dummy | N/A | N/A | TBD | smoke only |

## 6. Hugging Face Benchmark Result

| Experiment | HF Mode | HF Model ID | Model Revision | HF Dataset ID | Dataset Split | Integration Mode | Samples | Metric Source | Metric | Offline | Cache Mode | Notes |
|---|---|---|---|---|---|---|---:|---|---:|---|---|---|
| hf_model_only | hf_model_only | TBD | TBD | N/A | N/A | official_processor | 0 | internal_fallback | N/A | false | standard | smoke |
| hf_dataset_only | hf_dataset_only | N/A | N/A | TBD | validation | official_processor | TBD | internal_fallback/HF Evaluate | TBD | false | standard | smoke |
| offline_hf_cache | offline_hf_cache | TBD | TBD | TBD | TBD | official_processor | TBD | TBD | TBD | true | offline_hf_cache | revision pinned |

## 7. Token Reduction vs Accuracy

| Experiment | Dataset | Resolution | Frames | Before Tokens | After Tokens | Reduction Ratio | Accuracy / Score | Notes |
|---|---|---|---:|---:|---:|---:|---:|---|
| TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | real result required |

해석 원칙:

- token reduction이 항상 accuracy 보존을 의미하지 않습니다.
- selected token이 적어도 중요한 detail이 빠질 수 있습니다.
- scale별/프레임별 token distribution을 함께 봅니다.

## 8. Latency vs Accuracy

| Experiment | Device | Frames | Resolution | Latency ms | Throughput | Accuracy / Score | Acceleration Type | Notes |
|---|---|---:|---|---:|---:|---:|---|---|
| TBD | CUDA | TBD | TBD | TBD | TBD | TBD | TBD | warm-up 포함 여부 기록 |

구분:

1. true encoder-side acceleration
2. post-patch-embedding token masking
3. post-encoder token pruning
4. downstream token reduction only
5. compatibility-only adapter path

## 9. VRAM vs Accuracy

| Experiment | Device | Frames | Resolution | Peak VRAM MB | Accuracy / Score | Notes |
|---|---|---:|---|---:|---:|---|
| TBD | CUDA | TBD | TBD | TBD | TBD | CUDA metric |
| TBD | MPS | TBD | TBD | N/A | TBD | unavailable metric은 N/A |

## 10. Cases Where AutoGaze Helps

| Case | Evidence | Required Verification |
|---|---|---|
| high-resolution detail QA | HLVid 등에서 개선 가능성 | official protocol 재현 |
| long-video token pressure | 더 많은 frame/resolution 처리 가능성 | token/latency/VRAM 동시 기록 |
| MLLM prefill reduction | downstream token 감소 | prefill latency 분리 측정 |

## 11. Cases Where AutoGaze Hurts

| Case | Evidence | Mitigation |
|---|---|---|
| 중요한 patch 누락 | accuracy 하락 | token budget/scale 조정 |
| patch grid mismatch | 잘못된 remapping | PatchGridMapper metadata 검증 |
| unsupported MLLM visual input | generation failure | official processor mode 사용 |
| post-encoder pruning only | encoder latency 개선 없음 | acceleration type 정확히 표시 |

## 12. Cases Requiring Fine-Tuning

| Path | Why | Status |
|---|---|---|
| random VisionFeatureAdapter | feature distribution mismatch | zero-shot 성능 주장 금지 |
| non-NVILA MLLM direct token path | visual token format mismatch | API 검증 및 학습 필요 |
| vanilla SigLIP + AutoGaze | patch/grid/position mismatch 가능 | A3 experimental |

## 13. Final Analysis Checklist

- dummy/stub result와 real result를 분리했는가?
- external model result source를 명시했는가?
- AutoGaze 사용 여부를 과장하지 않았는가?
- encoder-side acceleration과 downstream token reduction을 구분했는가?
- MPS/CPU metric 한계를 `N/A`로 표시했는가?
- checkpoint, revision, cache mode, trust_remote_code를 기록했는가?
