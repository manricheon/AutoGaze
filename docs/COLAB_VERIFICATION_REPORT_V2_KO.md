# Colab/Kaggle CUDA 검증 리포트 V2

작성일: 2026-05-28  
대상 브랜치: `codex/autogaze-repro`  
기준 커밋: `d80408d Verify Qwen plugin CUDA smoke` 이후 V2 보강 작업

## 결론

V2 기준으로 확인해야 하는 축은 세 가지입니다.

| 축 | single inference | HLVid mini benchmark | AutoGaze on/off 비교 | 16프레임 시각화 |
|---|---|---|---|---|
| NVILA-HD native | CUDA smoke 확인됨 | CUDA smoke 확인됨 | `keep-all-single` vs `autogaze` | notebook 명령에 반영, 재실행 필요 |
| Qwen plugin | V2 notebook에 single 3모드 추가 | CUDA smoke 확인됨 | `qwen_full_vit`, `qwen_chunked_vit`, `qwen_chunked_vit_autogaze_sparse` | sparse plan/16프레임 로컬 예시 생성 |
| V-JEPA2 + Qwen | CUDA smoke 확인됨 | V2 notebook에 HLVid mini 추가 | `dense_off` vs `autogaze_single_grid` | 기본 16프레임 생성으로 변경 |

현재 조사 결론은 이렇습니다.

- NVILA-HD native 경로는 가장 안정적입니다. AutoGaze가 processor 내부에서 실제로 적용되고, SigLIP/Vision encoder/LLM latency와 token/memory 지표가 함께 기록됩니다.
- Qwen plugin sparse 경로는 동작합니다. 다만 AutoGaze checkpoint가 4-scale gaze decoder를 사용하므로 224 smoke에서는 `64+128+192+224`, patch size `16`, tile size `224`가 안정 조합입니다.
- V-JEPA2 + Qwen은 “동작 smoke / zero-shot bridge”로는 확인됐지만, Qwen에 맞춰 학습된 projector가 아니므로 accuracy 성능 주장은 아직 하면 안 됩니다. token/latency/memory plumbing 검증 용도로만 해석해야 합니다.
- 16프레임 시각화는 V2부터 기본값을 `16`으로 올렸습니다. 기존 Kaggle 실행의 remote artifact는 로컬에 복사되어 있지 않으므로, V2 notebook 재실행 시 생성되는 원격 artifact와 별도로 로컬 문서용 16프레임 예시 이미지를 저장했습니다.

## 16프레임 시각화

아래 이미지는 로컬 `inputs/hlvid_example/clip_av_video_5_001.mp4`에서 16프레임을 uniform sampling해 만든 문서용 확인 asset입니다. CUDA 실제 실행 artifact와 구분하기 위해 `docs/assets/colab_v2/` 아래에 저장했습니다.

### 선택 프레임 16장

![16 selected frames](assets/colab_v2/hlvid_example_16f_selected_frames.png)

### AutoGaze sparse overlay 예시

이 overlay는 로컬 Qwen sparse smoke plan인 `outputs/autogaze_repro/qwen_modes_smoke/qwen_chunked_vit_autogaze_sparse_actual_cpu_224_g002_autogaze_sparse_plan.json`을 사용했습니다.

![AutoGaze sparse overlay](assets/colab_v2/qwen_autogaze_sparse_overlay_16f.png)

생성 manifest:

```text
docs/assets/colab_v2/manifest.json
```

주의: 이 이미지는 “시각화 코드가 16프레임과 sparse patch overlay를 제대로 그리는지” 확인하는 로컬 asset입니다. Kaggle/Colab CUDA 재실행 후에는 아래 원격 artifact가 별도로 생성되어야 합니다.

## CUDA 실행 증거 요약

### NVILA-HD single

실행 조건:

```text
model: nvidia/NVILA-8B-HD-Video
AutoGaze checkpoint: /kaggle/working/autogaze_weights/nvidia__AutoGaze
device: cuda, device_map: auto, dtype: float16
video: inputs/hlvid_example/clip_av_video_5_001.mp4
frames: 16
max_tiles_video: 1
video_resize_longest_edge: 224
decode_strategy: seek
```

| mode | answer | total ms | preprocess(no AG) ms | AutoGaze ms | vision encoder ms | generate ms | LLM forward ms | encoder selected/raw | LLM visual tokens | peak memory |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `keep-all-single` | `The` | 17934.13 | 11673.03 | 9.83 | 4215.24 | 6251.27 | 1738.17 | 13328 / 13328 | 1496 | 12.44 GiB |
| `autogaze` | `The` | 10828.00 | 5083.49 | 1591.90 | 1831.84 | 4152.61 | 2075.46 | 17024 / 33920 | 1904 | 7.85 GiB |

해석:

- AutoGaze는 selector cost를 추가하지만, 이 smoke에서는 vision encoder latency와 peak memory가 감소했습니다.
- 이 수치는 `224`, `1 tile`, `16 frames` smoke이므로 논문 HLVid 성능값을 대체하지 않습니다.
- V2 notebook에는 `--visualization-output-dir OUTPUT_ROOT / 'nvila_single_visualizations'`가 추가됐습니다. 기존 Kaggle 실행은 이 플래그 적용 전이므로 NVILA 영상 overlay artifact는 재실행 후 확인해야 합니다.

### NVILA-HD HLVid mini benchmark

| mode | failed | parse_failed | accuracy_total | generated | total ms | AutoGaze ms | vision encoder ms | generate ms | LLM forward ms |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| `single-scale dense` | 0 | 0 | 0.0 | `A` | 14348.84 | 0.84 | 4825.15 | 9146.74 | 3861.17 |
| `autogaze` | 0 | 0 | 0.0 | `A` | 10914.60 | 1256.77 | 1762.68 | 4149.63 | 2137.80 |

정답은 `B`였고 두 모드 모두 `A`를 출력했습니다. 따라서 이 mini run은 정확도 주장이 아니라 benchmark wrapper가 prediction/summary/gain report를 생성하는지 확인한 smoke입니다.

## Qwen plugin 검증

### Qwen single inference

V2 notebook에 아래 세 single mode가 추가되었습니다.

| mode | selector | ViT path | MLLM path | 상태 |
|---|---|---|---|---|
| `qwen_full_vit` | off/keep-all | native full Qwen ViT | Qwen generate | V2 notebook 재실행 대상 |
| `qwen_chunked_vit` | off/keep-all | chunked Qwen ViT | Qwen generate | V2 notebook 재실행 대상 |
| `qwen_chunked_vit_autogaze_sparse` | AutoGaze on | AutoGaze-selected sparse chunked Qwen ViT | pruned Qwen visual context | Kaggle HLVid mini에서 실제 실행 확인 |

V2 single 명령은 `repro.flexible_runner --mode single`을 직접 사용합니다. Qwen2.5 weight를 `qwen3-vl` adapter override로 사용한 smoke 조합은 이전 Kaggle 검증과 동일합니다.

### Qwen HLVid mini benchmark

Kaggle T4 x2에서 `scripts/run_hlvid_folder_benchmark.py --plugin-suite qwen`으로 확인한 값입니다.

| mode | implementation | generation | answer | total ms | input build ms | Qwen ViT prepare ms | generate ms | visual tokens after/before | context tokens |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| `qwen_full_vit` | `executed` | `executed` | `A` | 41001.01 | 5024.32 | n/a | 2598.17 | n/a | 325 |
| `qwen_chunked_vit` | `executed` | `executed` | `A` | 15728.37 | 6101.83 | 283.69 | 160.55 | 256 / 256 | 325 |
| `qwen_chunked_vit_autogaze_sparse` | `executed` | `executed` | `A` | 12722.54 | 4623.53 | 183.13 | 251.05 | 140 / 256 | 209 |

조사 결과:

- 처음 실패한 `56+112+224`는 patch size `16`으로 나누어 떨어지지 않았습니다.
- `64+128+224`는 3-scale이라 AutoGaze checkpoint의 4-scale decoder와 맞지 않았습니다.
- `64+128+192+224`, patch size `16`, tile size `224`에서 sparse mode가 실행됐습니다.
- wrapper는 이제 sparse mode가 포함될 때 scale을 명시하지 않아도 resize에 맞춰 4-scale 기본값을 채웁니다.

## V-JEPA2 + Qwen 검증

### Single smoke

Kaggle actual smoke 결과:

| case | status | answer | total ms | V-JEPA selected/raw | AutoGaze selected/raw | Qwen visual tokens | peak memory |
|---|---|---|---:|---:|---:|---:|---:|
| `vjepa_qwen_dense_off` | `passed` | `Describe the video in one short sentence. The video is about` | 27263.93 | 1568 / 1568 | n/a | 1568 | 7.509 GiB |
| `autogaze_vjepa_qwen_on` | `passed` | `Describe the video in one short sentence.` | 24588.99 | 8 / 1568 | 16 / 4240 | 8 | 7.117 GiB |

해석:

- V-JEPA token은 `1568 -> 8`, Qwen visual token도 `1568 -> 8`로 감소했습니다.
- Qwen generate latency와 V-JEPA sparse encode latency는 줄었습니다.
- AutoGaze selector cost가 약 `9629 ms` 추가되어 end-to-end 속도 최적화는 별도 개선 과제입니다.
- zero-shot bridge는 projector 학습 없이 `inputs_embeds`로 연결하는 구조라서 답변 품질을 성능 주장으로 쓰면 안 됩니다.

### HLVid mini benchmark

V2 notebook에는 `repro.vjepa_qwen_hlvid_benchmark` 실행 셀이 추가되었습니다.

```text
RUN_VJEPA_QWEN_HLVID_MINI = True
--vjepa-qwen-modes dense_off,autogaze_single_grid
--visualization-max-frames 16
```

이 셀은 V2 수정 후 아직 재실행하지 않았습니다. 재실행 후 기대 artifact:

```text
/kaggle/working/autogaze_vjepa_outputs/vjepa_qwen_hlvid_mini/vjepa_qwen_hlvid_summary.json
/kaggle/working/autogaze_vjepa_outputs/vjepa_qwen_hlvid_mini/vjepa_qwen_hlvid_report.md
/kaggle/working/autogaze_vjepa_outputs/vjepa_qwen_hlvid_mini/runs/dense_off/00000.json
/kaggle/working/autogaze_vjepa_outputs/vjepa_qwen_hlvid_mini/runs/autogaze_single_grid/00000.json
/kaggle/working/autogaze_vjepa_outputs/vjepa_qwen_hlvid_mini/runs/*/visualizations/*.png
```

## V2에서 반영한 코드 변경

| 항목 | 변경 |
|---|---|
| V-JEPA2+Qwen visualization | `repro.vjepa_qwen_runner --visualization-max-frames` 기본값 `4 -> 16` |
| Colab smoke wrapper | `scripts/run_colab_autogaze_cuda_smoke.py` 기본 visualization frame 수 `16` |
| NVILA notebook | single smoke에 `--visualization-output-dir`, `--visualization-fps` 추가 |
| Qwen notebook | `repro.flexible_runner` single 3모드 추가 |
| V-JEPA benchmark notebook | `repro.vjepa_qwen_hlvid_benchmark` mini benchmark 셀 추가 |
| V2 report normalizer | NVILA nested summary, Qwen generation metrics, sparse plan artifact path를 공통 report로 normalize |
| 로컬 문서 asset | `scripts/build_colab_v2_visualization_assets.py`로 16프레임 selected/overlay PNG 생성 |

## 재실행 체크리스트

Kaggle 또는 Colab CUDA 런타임에서 [notebooks/autogaze_external_cuda_verification.ipynb](../notebooks/autogaze_external_cuda_verification.ipynb)를 위에서부터 실행합니다.

필수 확인:

- `RUN_NVILA_SINGLE = True`
- `RUN_QWEN_SINGLE = True`
- `RUN_NVILA_HLVID_MINI = True`
- `RUN_QWEN_PLUGIN_HLVID_MINI = True`
- `RUN_VJEPA_QWEN_HLVID_MINI = True`

성공 조건:

- NVILA single: `keep_all_single.json`, `autogaze.json`, `nvila_single_visualizations/*` 생성
- Qwen single: `qwen_single_smoke/qwen_full_vit.json`, `qwen_chunked_vit.json`, `qwen_chunked_vit_autogaze_sparse.json` 생성
- V-JEPA2+Qwen single: `vjepa_qwen_dense_off_cuda_smoke.json`, `autogaze_vjepa_qwen_on_cuda_smoke.json`, `visualizations/*16프레임*` 생성
- NVILA HLVid mini: `hlvid_autogaze_gain_report.json` 생성
- Qwen HLVid mini: `plugin_hlvid_summary.json`와 `runs/qwen_*/*.json` 생성
- V-JEPA2+Qwen HLVid mini: `vjepa_qwen_hlvid_summary.json` 생성

## 남은 검증 리스크

| 리스크 | 상태 | 이유 |
|---|---|---|
| full HLVid accuracy 재현 | 미완료 | 현재는 limit 1 smoke 중심 |
| V-JEPA2+Qwen accuracy 주장 | 금지 | zero-shot bridge라 semantic alignment 없음 |
| NVILA/Qwen/V-JEPA 동일 조건 속도 비교 | 부분 가능 | single smoke 조건은 맞췄지만 모델 구조가 달라 해석 범위 제한 필요 |
| 원격 visualization artifact 로컬 복사 | 미완료 | Kaggle runtime artifact를 아직 로컬 repo asset으로 복사하지 못함 |

따라서 V2의 정확한 결론은 “세 파이프라인의 single/benchmark 실행 경로와 주요 token/latency/memory 계측 경로는 준비됐고, NVILA/Qwen/V-JEPA2+Qwen의 CUDA smoke 증거가 있다. 다만 V2 notebook에서 새로 추가된 Qwen single, V-JEPA HLVid mini, NVILA 16프레임 visualization은 CUDA에서 한 번 더 재실행해야 최종 보고서 artifact로 닫힌다”입니다.
