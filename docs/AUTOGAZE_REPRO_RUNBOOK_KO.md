# AutoGaze 재현 런북

## 공식 소스

- AutoGaze 코드: https://github.com/NVlabs/AutoGaze
- AutoGaze 프로젝트 페이지: https://autogaze.github.io/
- AutoGaze 논문: https://arxiv.org/abs/2603.12254
- AutoGaze Hugging Face 컬렉션: https://huggingface.co/collections/bfshi/autogaze
- HLVid 데이터셋: https://huggingface.co/datasets/bfshi/HLVid
- NVILA-HD-Video README 경로: https://github.com/NVlabs/VILA/tree/main/vila_hd/nvila_hd_video

## 로컬 MPS 세팅

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -U pip setuptools wheel
.venv/bin/python -m pip install -r requirements-repro.txt
bash scripts/bootstrap_official_repos.sh
.venv/bin/python -m pip install -e . --no-deps --no-build-isolation
```

이 브랜치는 공식 AutoGaze 코드 계층을 레포지토리 루트에 둡니다. 그래서 upstream 레포와 NVILA-HD-Video remote code에서처럼 `autogaze` import가 그대로 동작합니다. `external/` 디렉터리는 VILA 같은 선택적 side repository 용도로만 사용합니다.

공식 AutoGaze `pyproject.toml`에는 현재 `flash_attn`이 일반 dependency로 포함되어 있습니다. Apple MPS/macOS에서는 editable install을 할 때 `--no-deps --no-build-isolation`을 사용하고, MPS 호환 런타임 dependency는 `requirements-repro.txt`를 기준으로 설치합니다.

sandbox 안에서 `torch.backends.mps.is_available()`이 false인데 sandbox 밖에서는 true라면, 실제 MPS 벤치마크는 sandbox 밖에서 실행하세요. helper test와 CLI check는 MPS가 없어도 돌릴 수 있습니다.

## MPS AutoGaze 및 SigLIP 스모크 벤치마크

패키지에 포함된 `assets/example_input.mp4`는 공식 quick start에서 사용하는 일반 MP4 비디오입니다. 재현 preset은 `configs/repro/example_input_autogaze.yaml`입니다. 소스 비디오는 448x448, 총 64프레임이고, AutoGaze/SigLIP smoke path에서는 앞 16프레임을 샘플링합니다.

```bash
.venv/bin/python -m repro.autogaze_bench \
  --device mps \
  --dtype float32 \
  --warmup 1 \
  --repeat 3 \
  --output-json outputs/autogaze_repro/mps_autogaze_siglip_bench.json \
  --output-csv outputs/autogaze_repro/mps_autogaze_siglip_bench.csv
```

이 실행은 공식 AutoGaze 모델이 로드되고, gazing metadata를 만들고, Apple MPS에서 customized SigLIP path를 구동할 수 있는지 확인합니다. 로컬 MPS 결과는 code path와 tensor contract 검증용입니다. 논문 성능과 직접 비교할 수 있는 수치는 아닙니다.

로컬 AutoGaze checkpoint로 AutoGaze/SigLIP 벤치마크를 실행하려면 checkpoint 디렉터리를 `--autogaze-model`에 넘깁니다.

```bash
.venv/bin/python -m repro.autogaze_bench \
  --autogaze-model /path/to/local/autogaze-checkpoint \
  --device cuda \
  --dtype float16
```

이 workspace에서 관찰한 로컬 smoke 결과:

- AutoGaze revision: `ba48d0f94ac2929d6fe3ee4380dc893aa6eed0ab`
- 입력: 공식 `assets/example_input.mp4`, 16프레임, 224x224
- 토큰 감소율: 약 `19.91x`
- 선택된 non-padded patch: raw patch budget `4240` 중 `213`
- MPS 평균 AutoGaze latency: 약 `1609.78 ms`
- MPS 평균 full SigLIP latency: 약 `528.80 ms`
- MPS 평균 gazed SigLIP latency: 약 `112.97 ms`
- SigLIP 단독 speedup: 약 `4.68x`
- 작은 MPS smoke input에서 AutoGaze overhead까지 포함한 SigLIP speedup: 약 `0.31x`

마지막 수치는 로컬 MPS에서 작은 smoke input으로 AutoGaze overhead를 함께 잰 값이라 약하게 나오는 것이 자연스럽습니다. 리더 설득용 속도 수치는 CUDA 측정값을 기준으로 삼아야 합니다.

## 요약 리포트

```bash
.venv/bin/python -m repro.report \
  --autogaze-json outputs/autogaze_repro/mps_autogaze_siglip_bench.json \
  --output-json outputs/autogaze_repro/mps_report_summary.json \
  --output-csv outputs/autogaze_repro/mps_report_summary.csv
```

## CUDA 단일 샘플 NVILA 확인

```bash
.venv/bin/python -m repro.nvila_runner \
  --mode single \
  --device cuda \
  --num-video-frames 128 \
  --num-video-frames-thumbnail 64 \
  --max-tiles-video 48 \
  --measure-ttft \
  --warmup-runs 1 \
  --repeat-runs 3 \
  --output-json outputs/autogaze_repro/cuda_nvila_single_128f.json
```

이 명령은 공식 NVILA-HD-Video quickstart 규모를 따라가며, 전체 HLVid 실행 전에 모델과 processor path가 정상인지 검증합니다.

NVILA runner는 output JSON에 모듈별 timing을 기록합니다. 중요한 필드는 아래와 같습니다.

- `result.video_decode_ms`: 샘플링된 프레임 decode/read 시간입니다. runner-side resize를 쓰지 않으면 NVILA remote code의 video loader를 감싼 시간입니다. `--video-resize-*`를 쓰면 runner가 전체 비디오에서 프레임을 샘플링하고 PIL frame을 리사이즈하는 시간까지 포함합니다.
- `result.video_tiling_ms`: 프레임이 준비된 뒤 NVILA processor가 비디오를 준비하는 시간입니다. dynamic spatial tiling, thumbnail 생성, SigLIP/AutoGaze 입력 tensorization이 포함됩니다. SigLIP inference 시간은 아닙니다.
- `result.autogaze_ms`: 전체 AutoGaze selection stage 시간입니다. `autogaze` 모드에서는 AutoGaze forward와 sort/pad/split bookkeeping이 들어갑니다. `keep-all` 모드에서는 AutoGaze forward를 건너뛰고 keep-all mask를 만드는 시간이 대부분입니다.
- `result.autogaze_forward_ms`: AutoGaze model forward만 잰 시간입니다. AutoGaze 모델 자체 cost를 볼 때 가장 깨끗한 필드입니다.
- `result.vision_encoder_ms`: generation 중 NVILA visual embedding 경로를 감싼 시간입니다. SigLIP feature extraction, feature cleanup/reordering, projector 준비가 포함됩니다.
- `result.siglip_vision_ms`: SigLIP vision tower forward 시간입니다. AutoGaze가 vision encoder workload를 줄였는지 볼 때 중요합니다.
- `result.mm_projector_ms`: 선택/정렬된 vision feature를 MLLM 입력 차원으로 보내는 multimodal projector forward 시간입니다.
- `result.llm_forward_ms`: `generate` 내부에서 language model forward가 누적된 시간입니다. prefill과 decode 단계의 LLM 호출이 모두 포함됩니다.
- `result.ttft_ms`: `--measure-ttft`가 켜졌을 때, 처리된 visual/text input에서 1토큰을 생성하는 데 걸린 시간입니다. 별도의 1-token generation pass로 측정되며 `total_ms`에는 포함하지 않습니다.
- `result.decode_estimated_ms`: full `generate_ms - ttft_ms`로 계산한 대략적인 generation decode 시간입니다. TTFT와 full generation이 별도 호출이므로 추정값으로 보세요.
- `result.stage_timings_ms`: `processor`, 선택적 `ttft`, full `generate`의 raw nested timing bucket입니다. top-level field가 null이거나 call count까지 봐야 할 때 확인합니다.
- `result.token_metrics`: tile, thumbnail, total 기준 encoder patch budget과 LLM visual-token budget의 AutoGaze 전후 count입니다.
- `result.compute_metrics`: token count와 모델 config로 계산한 SigLIP encoder 및 MLLM prefill 계산량/메모리 추정치입니다. 실제 wall-clock은 위 latency 필드와 함께 봐야 합니다.
- `result.processor_peak_memory_bytes`, `result.ttft_peak_memory_bytes`, `result.llm_peak_memory_bytes`, `result.peak_memory_bytes`: CUDA 실행 시 processor phase, 1-token TTFT pass, full generate pass의 CUDA peak allocation입니다.

`--measure-ttft`는 preprocessing 이후 1토큰 generation을 추가로 실행합니다. 여기서 prefill은 LLM이 prompt text와 visual token 전체를 한 번에 읽는 첫 forward입니다. 이 forward가 이후 token-by-token decode에서 재사용할 KV cache를 만듭니다. 따라서 TTFT는 순수 text decoding latency가 아니라 visual embedding, SigLIP/vision encoding, projector work, 첫 LLM prefill forward까지 포함할 수 있습니다. 세부 분리는 `result.ttft_stage_timings_ms`의 `vision_encode_total`, `siglip_vision_tower`, `mm_projector`, `llm_forward`를 확인하세요.

단일 파일 inference도 같은 output JSON에 속도/토큰/메모리 필드를 남깁니다. `--measure-ttft` 없이도 `total_ms`, `video_decode_ms`, `video_tiling_ms`, `autogaze_forward_ms`, `siglip_vision_ms`, `mm_projector_ms`, `llm_forward_ms`, `token_metrics`, `compute_metrics`, CUDA의 `processor_peak_memory_bytes`와 `llm_peak_memory_bytes`가 기록됩니다. `--measure-ttft`를 켜면 여기에 `ttft_ms`, `ttft_stage_timings_ms`, `ttft_peak_memory_bytes`, `decode_estimated_ms`가 추가됩니다. MPS에서는 CUDA peak allocation API가 없어서 memory field가 null일 수 있습니다.

raw output JSON이 너무 길면 `--print-summary --summary-json <path>`를 붙이세요. 전체 raw JSON은 `--output-json`에 그대로 저장하고, 터미널과 summary file에는 답변과 함께 `prompt`, `question`, `video_input_summary`, `autogaze_token_summary`, `key_autogaze_effect`를 별도로 정리합니다. `single` 모드에서는 `prompt`가 실행에 사용한 `--prompt` 원문이고, HLVid row 기반 실행에서는 per-row `question`이 `predictions.jsonl`과 `scored_predictions.jsonl`에 보존됩니다. `video_input_summary`에는 원본 총 프레임 수, 원본 해상도, 요청한 video/thumbnail frame 수, 실제 processor tensor 기준 frame 수, runner resize 적용 여부, resize 후 processor 입력 해상도, spatial tile/temporal chunk 수가 들어갑니다. `autogaze_token_summary`에는 사용한 프레임/타일 기준 raw patch budget과 AutoGaze가 실제 유지한 patch 수, TokenShuffle 이후 LLM visual token 수가 나뉘어 들어갑니다. `key_autogaze_effect`에는 AutoGaze 전후 차이를 가장 잘 보여주는 encoder patch 수, LLM visual token 수, reduction ratio/percent, SigLIP/MLLM 계산량 감소 추정치, 핵심 latency/memory median이 모입니다. 상세 분석용 `latency_ms`, `memory_bytes`, `tokens`, `compute` 섹션도 함께 남깁니다.

단일 실행의 비디오 입력 조건을 빠르게 확인할 때는 아래 필드를 먼저 보세요. raw JSON에는 top-level `video_input_summary`와 `result.video_input_summary`가 모두 있고, compact summary JSON에도 같은 `video_input_summary`가 들어갑니다.

| 질문 | 볼 필드 |
| --- | --- |
| 원본 비디오가 몇 프레임/몇 해상도였나? | `video_input_summary.source_frames`, `source_resolution`, `source_fps`, `source_duration_seconds` |
| 우리가 몇 프레임을 요청했나? | `video_input_summary.requested_video_frames`, `requested_thumbnail_frames` |
| 실제 processor tensor 기준 몇 프레임이 들어갔나? | `video_input_summary.actual_video_frames`, `actual_thumbnail_frames` |
| 전체 비디오 중 어디까지 샘플링했나? | `video_input_summary.sampled_frame_start`, `sampled_frame_end` |
| runner-side resize를 켰나? | `video_input_summary.runner_resize_enabled`, `runner_resize_request` |
| NVILA processor에 넘긴 비디오 frame 해상도는? | `video_input_summary.processor_input_resolution`, `processor_input_width`, `processor_input_height` |
| tile/chunk 구조는? | `video_input_summary.spatial_tiles_per_video`, `temporal_chunks_per_video` |

토큰 감소를 빠르게 볼 때는 아래 필드를 먼저 보세요.

| 질문 | 볼 필드 |
| --- | --- |
| AutoGaze 입력으로 들어간 전체 patch 수는? | `autogaze_token_summary.autogaze_selection_patch_tokens.input_patch_tokens`, raw의 `token_metrics.autogaze_input_patch_tokens` |
| 그 patch 수가 왜 큰가? | `autogaze_token_summary.autogaze_input_breakdown.expanded_formula` |
| AutoGaze가 실제 선택/유지한 patch 수는? | `autogaze_token_summary.autogaze_selection_patch_tokens.selected_patch_tokens`, raw의 `token_metrics.autogaze_selected_patch_tokens` |
| AutoGaze가 제거한 patch 수/비율은? | `autogaze_token_summary.autogaze_selection_patch_tokens.removed_patch_tokens`, `reduction_percent` |
| 사용한 프레임/타일/thumbnail 전체 encoder raw patch budget은? | `autogaze_token_summary.encoder_patch_tokens_before_siglip.raw_total_patch_tokens` |
| SigLIP 전 최종 유지 patch 수는? | `autogaze_token_summary.encoder_patch_tokens_before_siglip.selected_total_patch_tokens` |
| LLM에 들어가는 keep-all 대비 실제 visual token 수는? | `autogaze_token_summary.llm_visual_tokens_after_token_shuffle.keep_all_visual_tokens_estimated`, `actual_visual_tokens` |
| LLM visual token 감소율은? | `autogaze_token_summary.llm_visual_tokens_after_token_shuffle.reduction_ratio`, `reduction_percent` |

CUDA에서 속도 claim을 만들 때는 `single` 모드에 `--warmup-runs 1 --repeat-runs 3` 이상을 붙이는 것을 권장합니다. warmup 결과는 버리고, 측정 run은 `repeat_results`에 모두 저장되며 `repeat_summary`에 mean/median/min/max가 정리됩니다. backward compatibility를 위해 `result`는 마지막 측정 run으로 남겨둡니다. HLVid full benchmark에서는 `--warmup-runs`만 사용하세요. dataset row 자체가 여러 샘플이므로 반복 통계는 per-row 결과의 median으로 봅니다.

한 파일에서 AutoGaze 적용 이득까지 보려면 같은 video/prompt를 `--gazing-mode keep-all`과 `--gazing-mode autogaze`로 각각 한 번씩 실행해 JSON을 비교합니다. 한 번의 `single` 실행은 한 mode의 실제 end-to-end 결과만 기록합니다. `stream-profile --stream-siglip-mode both`는 한 번에 keep-all/gazed SigLIP forward를 비교할 수 있지만, projector/LLM까지 포함한 full NVILA 비교는 아닙니다.

### 시간 필드의 상하위 관계

runner의 timing은 NVILA remote-code method를 runtime에 감싸서 측정합니다. 그래서 어떤 값은 상위 구간이고, 어떤 값은 그 안에서 잡힌 하위 구간입니다. 상위와 하위 값은 병목을 설명하기 위한 관계이지, 모든 필드가 항상 더해서 정확히 같은 숫자가 되는 closed accounting은 아닙니다. 특히 `--measure-ttft`는 full generation과 별도의 1-token generation pass를 한 번 더 실행합니다.

```text
single run
|
|-- optional runner-side video_decode_sampling
|      when --video-resize-* is enabled, frames are decoded/sampled/resized
|      before calling the NVILA processor.
|
|-- processor_total
|   |-- video_tiling_and_tensorize  -> result.video_tiling_ms
|   |      dynamic tiling, thumbnail resize/crop, image tensorization
|   |
|   `-- autogaze_total             -> result.autogaze_ms
|       |  AutoGaze selection stage: model forward + gaze-info
|       |  construction + padding/splitting/bookkeeping
|       |
|       `-- autogaze_forward_batched -> result.autogaze_forward_ms
|           pure AutoGaze model forward over batched tile tensors
|
|-- model.generate full pass        -> result.generate_ms
|   |-- vision_encode_total         -> result.vision_encoder_ms
|   |   |-- siglip_vision_tower      -> result.siglip_vision_ms
|   |   `-- mm_projector.forward     -> result.mm_projector_ms
|   |
|   `-- llm.forward accumulated      -> result.llm_forward_ms
|
`-- result.total_ms = result.video_preprocess_ms + result.generate_ms

if --measure-ttft:
`-- separate 1-token model.generate pass -> result.ttft_ms
    |-- result.ttft_stage_timings_ms.siglip_vision_tower
    |-- result.ttft_stage_timings_ms.mm_projector
    `-- result.ttft_stage_timings_ms.llm_forward
```

| 상위/대표 필드 | 하위 또는 관련 필드 | 포함 관계 | 해석 |
| --- | --- | --- | --- |
| `total_ms` | `video_preprocess_ms`, `generate_ms` | `generate_one()`이 최종 산출합니다. | 단일 run의 end-to-end latency입니다. `ttft_ms`는 별도 pass라 여기에 포함하지 않습니다. |
| `video_preprocess_ms` | `processor_total`, 선택적 `video_decode_sampling` | resize 옵션이 켜지면 runner-side decode/sampling도 더합니다. | MLLM generate 전까지 입력을 만드는 전체 비용입니다. |
| `stage_timings_ms.processor.processor_total` | `video_tiling_and_tensorize`, `autogaze_total`, 내부 decode | processor call 전체를 감싼 상위 구간입니다. | tokenization, video preprocess, tiling/tensorization, AutoGaze selection을 함께 봅니다. |
| `video_decode_ms` | `video_decode_sampling` | resize 사용 시 processor 밖에서, 미사용 시 processor 안에서 측정될 수 있습니다. | 긴 4K HLVid에서 CPU decode/seek가 병목인지 확인합니다. |
| `video_tiling_ms` | `video_tiling_and_tensorize` | `processor_total`의 하위 구간입니다. | tile/thumbnail 생성과 image tensorization 비용입니다. |
| `autogaze_ms` | `autogaze_total` | `processor_total`의 하위 구간입니다. | AutoGaze stage 전체 시간입니다. 순수 모델 forward만이 아니라 gaze-info 생성, padding, split/bookkeeping이 포함됩니다. |
| `autogaze_forward_ms` | `autogaze_forward_batched` | `autogaze_total`의 하위 구간입니다. | 순수 AutoGaze 모델 forward 시간입니다. “AutoGaze 모델만 돈 시간”은 이 값을 봅니다. |
| `vision_encoder_ms` | `siglip_vision_ms`, `mm_projector_ms` | generate pass 내부 vision path 상위 구간입니다. | NVILA의 vision encoding 전체입니다. 순수 SigLIP만 보려면 `siglip_vision_ms`를 봅니다. |
| `siglip_vision_ms` | `siglip_vision_tower` | `vision_encoder_ms`의 하위 구간입니다. | AutoGaze가 patch/token을 줄여 vision tower forward가 빨라지는지 보는 핵심 latency입니다. |
| `mm_projector_ms` | `mm_projector.forward` | vision feature를 LLM hidden dimension으로 투영하는 구간입니다. | TokenShuffle 이후 visual token 수 감소가 projector 비용에 반영되는지 확인합니다. |
| `llm_forward_ms` | `llm.forward` 누적 | full generation pass에서 LLM forward hook을 누적합니다. | prefill과 decode 호출이 모두 포함됩니다. TTFT 분리는 `ttft_stage_timings_ms`를 함께 봅니다. |
| `ttft_ms` | `ttft_stage_timings_ms.*` | `--measure-ttft`일 때 별도 1-token generate pass입니다. | prefill context, KV cache, 첫 토큰 latency를 보는 값입니다. full `total_ms`에 더하지 않습니다. |

관련 코드 위치는 stage hook 정의 [ProfilePatches:L126](../repro/nvila_runner.py#L126), processor/generate 실행 [generate_one:L2427](../repro/nvila_runner.py#L2427), 결과 필드 조립 [generate_one result fields:L2492](../repro/nvila_runner.py#L2492), full/TTFT generation timer [timed_generate:L1538](../repro/nvila_runner.py#L1538), compact summary 생성 [build_single_summary:L285](../repro/nvila_runner.py#L285)입니다.

토큰 metrics는 두 단계로 나눠서 봅니다. encoder patch budget은 TokenShuffle/projector 이전의 patch 수입니다. 여기에는 실제 샘플링된 비디오 프레임, spatial tile, thumbnail, 그리고 설정된 모든 visual scale의 patch가 포함됩니다. LLM visual-token budget은 TokenShuffle/projector 이후 language model이 실제로 받는 visual placeholder token 수입니다. 헷갈릴 때는 raw `token_metrics`보다 `autogaze_token_summary`를 먼저 보세요.

encoder patch 기준:

- `token_metrics.video_sampled_frames`: tiled video processing에 들어간 전체 비디오 샘플 프레임 수
- `token_metrics.thumbnail_sampled_frames`: tiled frame과 별도로 처리된 thumbnail frame 수
- `token_metrics.spatial_tiles_per_video`, `token_metrics.temporal_chunks_per_video`, `token_metrics.tile_sequences`: spatial/temporal AutoGaze/SigLIP sequence 수
- `token_metrics.encoder_patches_per_frame_by_scale`: 예를 들어 `56`, `112`, `196`, `392` 같은 multi-scale별 patch 수
- `token_metrics.encoder_patches_per_frame_multiscale`: 한 프레임에서 multi-scale patch 수를 모두 더한 값
- `token_metrics.autogaze_input_tile_frame_instances`: sampled frame이 spatial tile로 펼쳐진 뒤의 tile-frame 개수입니다. 단일 비디오에서는 보통 `video_sampled_frames × spatial_tiles_per_video[0]`입니다.
- `token_metrics.encoder_raw_tile_patch_tokens`: AutoGaze 전 tiled-video patch budget입니다. sampled frames × spatial tiles × multi-scale patches per frame으로 계산됩니다.
- `token_metrics.autogaze_input_patch_tokens`: AutoGaze 모델이 선택하기 전에 입력으로 받은 전체 tiled-video patch 수입니다. 현재 구현에서는 `encoder_raw_tile_patch_tokens`와 같은 값입니다.
- `token_metrics.autogaze_selected_patch_tokens`: AutoGaze가 실제 유지한 non-padded tiled-video patch 수입니다. 현재 구현에서는 `encoder_autogaze_selected_tile_patch_tokens`와 같은 값입니다.
- `token_metrics.autogaze_removed_patch_tokens`, `token_metrics.autogaze_patch_reduction_ratio`: AutoGaze 입력 patch 대비 제거량과 감소 비율입니다.
- `token_metrics.encoder_raw_thumbnail_patch_tokens`: AutoGaze 전 thumbnail patch budget입니다. thumbnail frames × multi-scale patches per frame으로 계산됩니다.
- `token_metrics.encoder_raw_patch_tokens`: tile과 thumbnail을 합친 raw patch budget
- `token_metrics.encoder_autogaze_selected_tile_patch_tokens`: AutoGaze 이후 실제 유지된 non-padded tile patch 수입니다. `keep-all`에서는 raw tile patch budget과 같아야 합니다.
- `token_metrics.encoder_autogaze_selected_thumbnail_patch_tokens`: 유지된 thumbnail patch 수입니다. 현재 runner 설정에서는 thumbnail이 keep-all이라 raw thumbnail patch budget과 같아야 합니다.
- `token_metrics.encoder_autogaze_selected_patch_tokens`: tile과 thumbnail을 합친 AutoGaze 이후 유지 patch 수
- `token_metrics.encoder_tile_token_reduction_ratio`, `token_metrics.encoder_thumbnail_token_reduction_ratio`, `token_metrics.encoder_token_reduction_ratio`: tile, thumbnail, total 기준 raw patch 수를 유지 patch 수로 나눈 값

여기서 `autogaze_input_patch_tokens`와 `autogaze_selected_patch_tokens`가 AutoGaze 자체의 입력 대비 선택 결과를 가장 직접적으로 보여줍니다. `encoder_autogaze_selected_*`는 LLM token이 아니라 **SigLIP에 들어가기 전 AutoGaze가 유지하기로 선택한 non-padded encoder patch 위치 수**입니다. 현재 runner에서는 thumbnail은 keep-all이라 tile 쪽 선택량이 AutoGaze 효과를 가장 직접적으로 보여줍니다.

예를 들어 720p로 resize된 128프레임 비디오라도 `max_tiles_video=8`이면 단일 720p frame 하나가 아니라 최대 8개 spatial tile로 나뉩니다. 기본 scale `56+112+196+392`, patch size 14에서는 한 tile-frame당 patch 위치가 `16+64+196+784=1060`개입니다. 따라서 AutoGaze 입력 patch 수는 `128 frames × 8 tiles/frame × 1060 patches = 1,085,440`처럼 백만 단위가 될 수 있습니다. 이 값은 LLM visual token 수가 아니라 SigLIP/TokenShuffle 이전의 encoder patch-position budget입니다. 실제 LLM 쪽 비교는 `llm_keep_all_visual_tokens_estimated`와 `llm_actual_visual_tokens`를 봐야 합니다.

LLM visual-token 기준:

- `token_metrics.llm_keep_all_visual_tokens_estimated`: 모든 tile/thumbnail patch를 유지했을 때 TokenShuffle 이후 예상 visual token 수
- `token_metrics.llm_actual_visual_tokens`: AutoGaze/keep-all padding strategy가 반영된 processor output의 실제 visual placeholder token 수
- `token_metrics.llm_visual_token_reduction_ratio`: keep-all 예상 LLM visual token 수를 실제 visual token 수로 나눈 값

HLVid `--mode hlvid` summary에는 `question_count`, `question_samples`, `benchmark_samples`, `token_budget_summary`가 추가됩니다. `question_samples`는 질문 원문 확인용이고, `benchmark_samples`는 대상 비디오, 질문, 모델 답변, parsed 답변, 정답, 정오 여부를 같이 보여주는 읽기 쉬운 샘플 표입니다. summary가 너무 커지지 않도록 앞쪽 샘플만 담고, 전체 row의 질문/정답/모델 출력은 `predictions.jsonl`과 `scored_predictions.jsonl`에 남깁니다. `token_budget_summary`는 성공한 row들의 `token_metrics`에서 median/mean을 모은 것이고, `failed` row는 token metric이 없으므로 집계에서 빠집니다. `scripts/run_hlvid_folder_benchmark.py`의 최종 gain report에서도 top-level `benchmark_samples.keep_all`, `benchmark_samples.autogaze`, `autogaze.tokens`와 `gains.autogaze_token_reduction_median`에 raw/selected patch 수와 LLM visual token 수가 함께 들어갑니다.

계산량/메모리 추정치는 `compute_metrics`에 있습니다. 이 값은 profiler가 직접 센 FLOPs가 아니라, token 수와 hidden size/layer 수를 이용한 analytical MAC estimate입니다.

- `compute_metrics.siglip_encoder.keep_all` / `actual`: SigLIP vision tower에 들어가는 sequence 수, token 수, dense attention pair 수, attention projection MACs, attention N^2 MACs, MLP MACs, hidden/attention/MLP activation byte 추정치입니다.
- `compute_metrics.siglip_encoder.keep_all_to_actual_attention_macs_ratio`: AutoGaze 전후 SigLIP attention 계산량 감소 비율입니다. patch 수 감소보다 attention pair 감소가 더 크게 보일 수 있습니다.
- `compute_metrics.siglip_encoder.keep_all_to_actual_mlp_macs_ratio`: AutoGaze 전후 SigLIP MLP 계산량 감소 비율입니다. MLP는 token 수에 거의 선형으로 줄어듭니다.
- `compute_metrics.mllm.actual_prefill_context_tokens`: LLM prefill에 실제 들어간 전체 context 길이입니다. text token과 visual token이 모두 포함됩니다.
- `compute_metrics.mllm.keep_all_prefill_context_tokens_estimated`: AutoGaze 없이 keep-all visual token을 넣었을 때의 prefill context 길이 추정치입니다.
- `compute_metrics.mllm.prefill_context_reduction_ratio`: keep-all 대비 실제 prefill context 감소 비율입니다.
- `compute_metrics.mllm.actual_kv_cache_bytes_after_prefill_estimated`: 실제 prefill 이후 KV cache 크기 추정치입니다. Qwen 계열 GQA 설정의 `num_key_value_heads`를 반영합니다.
- `compute_metrics.mllm.kv_cache_reduction_ratio`, `prefill_attention_pair_reduction_ratio`, `prefill_total_macs_reduction_ratio`: LLM KV cache, causal attention pair, 전체 prefill MAC 감소 비율입니다.

### 측정 커버리지 재점검

현재 runner가 리포트에 남기는 항목과 중요도는 아래와 같습니다.

| 영역 | 필드 | 설명 | 코드 위치 | 중요도 |
| --- | --- | --- | --- | --- |
| 입력/샘플링 | `video_input_summary.*`, `input_token_count`, `input_shapes`, `token_metrics.video_sampled_frames`, `token_metrics.thumbnail_sampled_frames` | 원본 프레임 수/해상도, 요청 frame 수, 실제 processor tensor 기준 frame 수, resize 후 입력 해상도, text+visual context 길이입니다. 설정이 의도대로 반영됐는지 확인합니다. | [build_video_input_summary:L1626](../repro/nvila_runner.py#L1626), [generate_one:L2528](../repro/nvila_runner.py#L2528), [compute_visual_token_metrics:L1383](../repro/nvila_runner.py#L1383), [build_stream_profile_token_metrics:L515](../repro/nvila_runner.py#L515) | 높음 |
| 비디오 decode | `video_decode_ms`, `stage_timings_ms.*.video_decode_sampling`, stream의 `timing_ms.video_decode_scan/seek` | CPU decode와 seek sampling 비용입니다. 긴 4K HLVid에서는 병목 여부를 먼저 봅니다. | [ProfilePatches:L81](../repro/nvila_runner.py#L81), [run_stream_profile:L1822](../repro/nvila_runner.py#L1822) | 높음 |
| resize/tiling | `video_tiling_ms`, stream의 `video_frame_resize`, `spatial_tile_build` | NVILA dynamic tiling과 runner resize cost입니다. AutoGaze 효과와 무관한 전처리 overhead를 분리합니다. | [ProfilePatches:L81](../repro/nvila_runner.py#L81), [run_stream_profile:L1822](../repro/nvila_runner.py#L1822) | 높음 |
| AutoGaze | `autogaze_ms`, `autogaze_forward_ms`, stream의 `tile_autogaze_tensorize`, `tile_autogaze_forward` | AutoGaze를 넣어서 추가된 비용입니다. token 감소 이득이 이 비용을 이기는지 판단합니다. | [ProfilePatches:L81](../repro/nvila_runner.py#L81), [run_autogaze_on_stream_tile_sequences:L1576](../repro/nvila_runner.py#L1576) | 높음 |
| SigLIP latency | `siglip_vision_ms`, stream의 `siglip_gazed_forward`, `siglip_keep_all_forward` | vision encoder의 실제 forward 시간입니다. AutoGaze의 1차 효과가 드러나는 지점입니다. | [ProfilePatches:L81](../repro/nvila_runner.py#L81), [run_siglip_on_stream_batch:L1527](../repro/nvila_runner.py#L1527) | 높음 |
| SigLIP 계산량 | `compute_metrics.siglip_encoder.*` | attention/MLP MACs와 activation byte 추정치입니다. latency가 noisy할 때도 계산량 감소를 설명할 수 있습니다. | [build_autogaze_effect_metrics:L1043](../repro/nvila_runner.py#L1043), [build_stream_profile_compute_metrics:L362](../repro/nvila_runner.py#L362) | 높음 |
| projector | `mm_projector_ms` | SigLIP hidden state를 LLM hidden dimension으로 보내는 TokenShuffle+MLP 시간입니다. visual token 수가 줄면 같이 줄 수 있습니다. | [mm_projector.forward hook:L95-L97](../repro/nvila_runner.py#L95-L97), [output field:L2313](../repro/nvila_runner.py#L2313) | 중간 |
| LLM latency | `llm_forward_ms`, `ttft_ms`, `decode_estimated_ms` | prefill과 generation decode 비용입니다. AutoGaze의 MLLM context 감소 효과를 봅니다. | [llm.forward hook:L99-L100](../repro/nvila_runner.py#L99-L100), [timed_generate:L1337](../repro/nvila_runner.py#L1337), [generate_one:L2220](../repro/nvila_runner.py#L2220) | 높음 |
| LLM context/KV | `compute_metrics.mllm.actual_prefill_context_tokens`, `kv_cache_reduction_ratio`, `prefill_total_macs_reduction_ratio` | LLM이 실제로 받은 context 길이와 KV cache/attention 계산 감소 추정치입니다. | [build_autogaze_effect_metrics:L1043](../repro/nvila_runner.py#L1043) | 높음 |
| CUDA memory | `processor_peak_memory_bytes`, `ttft_peak_memory_bytes`, `llm_peak_memory_bytes`, `peak_memory_bytes` | CUDA peak allocation입니다. OOM 리스크와 배치/프레임 설정 선택에 중요합니다. | [timed_generate peak:L1339-L1348](../repro/nvila_runner.py#L1339-L1348), [generate_one memory fields:L2227-L2302](../repro/nvila_runner.py#L2227-L2302), [stream CUDA peak:L1883-L2211](../repro/nvila_runner.py#L1883-L2211) | 높음 |
| token/patch | `token_metrics.encoder_*`, `token_metrics.llm_*` | encoder patch 감소와 LLM visual token 감소를 분리해서 보여줍니다. 리더 설득용 핵심 근거입니다. | [compute_visual_token_metrics:L1181](../repro/nvila_runner.py#L1181), [build_stream_profile_token_metrics:L313](../repro/nvila_runner.py#L313) | 높음 |
| 정확도 | HLVid `accuracy_scored`, `accuracy_total`, `failed`, `skipped`, `parse_failed` | AutoGaze 속도 이득이 성능 손실을 만들었는지 확인합니다. | [score_predictions:L96](../repro/hlvid.py#L96), [summarize_run:L194](../repro/hlvid_batch_benchmark.py#L194), [build_gain_report:L205](../repro/hlvid_batch_benchmark.py#L205) | 높음 |

아직 직접 측정하지 않는 항목도 있습니다. `compute_metrics`의 MACs는 실제 GPU hardware counter가 아니라 config 기반 추정치입니다. CUDA의 정확한 SM utilization, DRAM bandwidth, per-layer peak activation은 Nsight/PyTorch profiler가 필요합니다. MPS는 CUDA처럼 reliable한 peak allocation을 제공하지 않으므로 memory 평가는 CUDA에서 최종 확인하세요.

실제 동작 순서대로 보면 아래 필드를 따라가면 됩니다.

1. Dataset row 로드: HLVid `manifest`, `question`, `answer`, `video_path`
2. 비디오 resolve: `video_resolved`, `video_input_info`, `video_resize`
3. frame sampling/decode: `video_decode_ms`, stream의 `decode_strategy`, `decode_frames_read`
4. PIL 변환/resize: stream의 `video_frame_to_pil`, `video_frame_resize`
5. spatial tiling/thumbnail: `video_tiling_ms`, stream의 `spatial_tile_build`, `thumbnail_resize`
6. AutoGaze 입력 tensorization: stream의 `tile_autogaze_tensorize`
7. AutoGaze selection 전체: `autogaze_ms`
8. AutoGaze model forward-only: `autogaze_forward_ms`, stream의 `tile_autogaze_forward`
9. SigLIP vision tower: `siglip_vision_ms` 또는 stream의 `siglip_gazed_forward/keep_all_forward`
10. visual token 정렬/TokenShuffle/projector: `mm_projector_ms`, `token_metrics.llm_actual_visual_tokens`
11. LLM prefill/TTFT: `ttft_ms`, `compute_metrics.mllm.actual_prefill_context_tokens`, `actual_kv_cache_bytes_after_prefill_estimated`
12. LLM decode: `llm_forward_ms`, `decode_estimated_ms`, `generated_tokens`
13. scoring/report: HLVid `accuracy_scored`, batch report의 `gains.*`

AutoGaze 적용 전후 실제 gain은 반드시 같은 설정의 `keep-all`과 `autogaze`를 나란히 비교합니다.

| 확인 질문 | 볼 필드 |
| --- | --- |
| 단일 파일 summary에서 먼저 볼 핵심은? | `summary.key_autogaze_effect.encoder_patch_*`, `summary.key_autogaze_effect.llm_visual_*`, `summary.key_autogaze_effect.siglip_total_macs_reduction_ratio`, `summary.key_autogaze_effect.mllm_*`, `summary.key_autogaze_effect.total_ms_median` |
| 정확도가 유지됐나? | `autogaze.accuracy.accuracy_scored`, `keep_all.accuracy.accuracy_scored`, `gains.accuracy_scored_delta` |
| 전체 latency가 줄었나? | `gains.latency_speedup_median.total_ms` |
| SigLIP가 빨라졌나? | `gains.latency_speedup_median.siglip_vision_ms`, stream의 `siglip_keep_all_forward / siglip_gazed_forward` |
| LLM prefill 부담이 줄었나? | `compute_metrics.mllm.prefill_context_reduction_ratio`, `kv_cache_reduction_ratio` |
| visual token이 줄었나? | `token_metrics.llm_visual_token_reduction_ratio`, batch report의 `gains.autogaze_token_reduction_median` |
| 메모리 이득이 있나? | `gains.memory_reduction_ratio_median.llm_peak_memory_bytes`, `ttft_peak_memory_bytes` |
| AutoGaze overhead가 이득을 잡아먹나? | `autogaze_ms`, `autogaze_forward_ms`와 `siglip_vision_ms/llm_forward_ms` 감소량 비교 |

AutoGaze와 full-token baseline을 비교하려면 같은 input을 두 번 실행하고 `--gazing-mode`만 바꿉니다. `autogaze`는 NVILA quickstart의 tile selection ratio를 사용합니다. `keep-all`은 `gazing_ratio_tile=1`, `task_loss_requirement_tile=None`으로 설정하여 public NVILA processor가 AutoGaze를 호출하지 않고 keep-all mask를 만들게 합니다.

```bash
.venv/bin/python -m repro.nvila_runner \
  --mode single \
  --device cuda \
  --gazing-mode autogaze \
  --num-video-frames 128 \
  --num-video-frames-thumbnail 64 \
  --max-tiles-video 48 \
  --measure-ttft \
  --warmup-runs 1 \
  --repeat-runs 3 \
  --output-json outputs/autogaze_repro/cuda_nvila_single_128f_autogaze.json

.venv/bin/python -m repro.nvila_runner \
  --mode single \
  --device cuda \
  --gazing-mode keep-all \
  --num-video-frames 128 \
  --num-video-frames-thumbnail 64 \
  --max-tiles-video 48 \
  --measure-ttft \
  --warmup-runs 1 \
  --repeat-runs 3 \
  --output-json outputs/autogaze_repro/cuda_nvila_single_128f_keep_all.json
```

속도 관점에서는 두 JSON 파일의 `repeat_summary.total_ms`, `repeat_summary.video_decode_ms`, `repeat_summary.video_tiling_ms`, `repeat_summary.autogaze_forward_ms`, `repeat_summary.siglip_vision_ms`, `repeat_summary.vision_encoder_ms`, `repeat_summary.llm_forward_ms` median을 비교합니다. 단일 실행만 했다면 같은 필드를 `result.*`에서 보면 됩니다. 토큰 관점에서는 tile, thumbnail, total patch budget을 같이 비교하세요. 핵심 필드는 `token_metrics.encoder_raw_tile_patch_tokens`, `token_metrics.encoder_autogaze_selected_tile_patch_tokens`, `token_metrics.encoder_raw_thumbnail_patch_tokens`, `token_metrics.encoder_autogaze_selected_thumbnail_patch_tokens`, `token_metrics.encoder_raw_patch_tokens`, `token_metrics.encoder_autogaze_selected_patch_tokens`, `token_metrics.encoder_token_reduction_ratio`, `token_metrics.llm_keep_all_visual_tokens_estimated`, `token_metrics.llm_actual_visual_tokens`, `token_metrics.llm_visual_token_reduction_ratio`입니다.

feasibility test를 위해 `nvila_runner`는 public NVILA processor가 tiling하기 전에 sampled video frame을 downscale할 수 있습니다. 이 기능은 runner-side preprocessing입니다. runner가 전체 비디오에서 `--num-video-frames`만큼 샘플링하고, 그 프레임을 리사이즈한 뒤 PIL frame list로 NVILA processor에 넘깁니다.

```bash
.venv/bin/python -m repro.nvila_runner \
  --mode single \
  --device cuda \
  --video inputs/hlvid_example/clip_av_video_5_001.mp4 \
  --gazing-mode keep-all \
  --num-video-frames 128 \
  --num-video-frames-thumbnail 64 \
  --max-tiles-video 48 \
  --video-resize-shortest-edge 720 \
  --measure-ttft \
  --output-json outputs/autogaze_repro/hlvid_example_nvila_single_128f_keep_all_resize720.json
```

exact-size test는 `--video-resize-width`와 `--video-resize-height`를 함께 사용하세요. max-side constraint가 필요하면 `--video-resize-longest-edge`를 씁니다. AutoGaze/keep-all patch scale도 processor init 경로로 바꿀 수 있습니다.

```bash
.venv/bin/python -m repro.nvila_runner \
  --mode single \
  --device cuda \
  --video inputs/hlvid_example/clip_av_video_5_001.mp4 \
  --gazing-mode autogaze \
  --num-video-frames 128 \
  --num-video-frames-thumbnail 64 \
  --max-tiles-video 48 \
  --video-resize-shortest-edge 720 \
  --autogaze-resize-scales 56+112+196+392 \
  --autogaze-target-patch-size 14 \
  --measure-ttft \
  --output-json outputs/autogaze_repro/hlvid_example_nvila_single_128f_autogaze_resize720.json
```

같은 resize flag로 preflight를 먼저 실행하세요. 출력에는 `source_metadata`, `effective_video`, `video_resize`가 포함되므로 테스트가 원본 4K 기준인지 downscaled frame 기준인지 확인할 수 있습니다.

로컬 AutoGaze checkpoint로 NVILA-HD-Video를 실행하려면 `--autogaze-model`에 checkpoint 디렉터리를 넘깁니다. runner는 이 값을 NVILA processor의 `autogaze_model_id`로 전달합니다. 이는 모델 remote code에서 사용하는 인자입니다.

```bash
.venv/bin/python -m repro.nvila_runner \
  --mode single \
  --device cuda \
  --autogaze-model /path/to/local/autogaze-checkpoint \
  --num-video-frames 128 \
  --num-video-frames-thumbnail 64 \
  --max-tiles-video 48 \
  --measure-ttft \
  --output-json outputs/autogaze_repro/cuda_nvila_single_local_autogaze.json
```

NVILA-HD-Video와 AutoGaze를 모두 로컬 checkpoint 디렉터리에서 실행하려면 `--nvila-model`과 `--autogaze-model`을 함께 사용합니다. `--nvila-model`은 `--model-path`의 alias입니다.

```bash
.venv/bin/python -m repro.nvila_runner \
  --mode single \
  --device cuda \
  --nvila-model /path/to/local/nvila-checkpoint \
  --autogaze-model /path/to/local/autogaze-checkpoint \
  --num-video-frames 128 \
  --num-video-frames-thumbnail 64 \
  --max-tiles-video 48 \
  --measure-ttft \
  --output-json outputs/autogaze_repro/cuda_nvila_single_local_models.json
```

## HF Space 예시 비디오

`https://huggingface.co/spaces/bfshi/AutoGaze`의 AutoGaze Space는 세 개의 예시 비디오 `doorbell.mp4`, `tomjerry.mp4`, `security.mp4`를 사용합니다. 아래 명령으로 로컬에 다운로드합니다.

```bash
.venv/bin/python scripts/download_hf_space_examples.py
```

비디오는 의도적으로 gitignore된 `inputs/hf_space_autogaze/` 아래에 저장됩니다. 대응 preset은 `configs/repro/hf_space_autogaze_examples.yaml`입니다. 이 preset에는 Space 설정이 기록되어 있습니다. UI gazing ratio는 `0.75`, model gazing ratio는 `0.75 * 196 / 265`, task loss requirement는 `0.7`, temporal chunk는 16프레임, spatial chunk는 224x224, spatial batch size는 `2`입니다.

기본 Space 예시 비디오인 `doorbell.mp4`에 fixed total-frame sampling으로 NVILA를 실행하려면:

```bash
.venv/bin/python -m repro.nvila_runner \
  --preset-config configs/repro/hf_space_autogaze_examples.yaml
```

같은 NVILA/AutoGaze 설정을 유지하면서 다른 Space 예시를 사용하려면:

```bash
.venv/bin/python -m repro.nvila_runner \
  --preset-config configs/repro/hf_space_autogaze_examples.yaml \
  --video inputs/hf_space_autogaze/security.mp4 \
  --output-json outputs/autogaze_repro/hf_space_security_nvila_single.json
```

Space 예시 비디오에서 HLVid-like stress check를 하려면 전체 샘플링 프레임 수를 override합니다.

```bash
.venv/bin/python -m repro.nvila_runner \
  --preset-config configs/repro/hf_space_autogaze_examples.yaml \
  --video inputs/hf_space_autogaze/security.mp4 \
  --num-video-frames 1024 \
  --output-json outputs/autogaze_repro/hf_space_security_nvila_1024f.json
```

## NVILA 메모리 preflight

긴 비디오나 고해상도 비디오를 NVILA에 넣기 전에 preflight mode를 먼저 실행하세요. 이 모드는 8B 모델을 로드하지 않습니다. 로컬 비디오 metadata를 읽고, NVILA dynamic tiling estimate를 따라 tile sequence count, keep-all visual token 수, 현재 public processor path 기준 CPU preprocessing memory lower bound를 보고합니다.

로컬 비디오에 대해:

```bash
.venv/bin/python -m repro.nvila_runner \
  --mode preflight \
  --video inputs/hf_space_autogaze/security.mp4 \
  --num-video-frames 1024 \
  --num-video-frames-thumbnail 64 \
  --max-tiles-video 48 \
  --preflight-json outputs/autogaze_repro/preflight_space_security_1024.json
```

특정 비디오를 다운로드하기 전에 HLVid-like 4K/5분 estimate를 보려면:

```bash
.venv/bin/python -m repro.nvila_runner \
  --mode preflight \
  --video hlvid_4k_virtual.mp4 \
  --preflight-width 3840 \
  --preflight-height 2160 \
  --preflight-source-frames 9000 \
  --num-video-frames 1024 \
  --num-video-frames-thumbnail 64 \
  --max-tiles-video 48 \
  --preflight-json outputs/autogaze_repro/preflight_4k_1024_virtual.json
```

4K/1024프레임 estimate에서는 현재 public processor path 기준으로 약 `45`개 spatial tile, `2880`개 tile sequence, 약 `5.44M` keep-all visual token, Python/PIL overhead를 제외한 CPU preprocessing memory lower bound 약 `202 GiB`가 보고됩니다. 이 결과가 나오면 full generation을 시도하기 전에 `--num-video-frames`나 `--max-tiles-video`를 줄이거나, chunked preprocessing 및 vision encoding을 구현해야 한다는 신호로 보세요.

## NVILA 청크 스트리밍 pre-LLM profile

긴 HLVid 비디오에서 전체 sampled frame, spatial tile, AutoGaze tensor를 한 번에 만들면 CPU/GPU memory가 먼저 터질 수 있습니다. `stream-profile` 모드는 NVILA-8B LLM을 로드하지 않고, 비디오 decode부터 AutoGaze selection 직전/직후까지를 temporal chunk 단위로 처리합니다. `--stream-run-siglip`을 켜면 AutoGaze 이후 custom SigLIP vision tower forward까지 같은 chunk 단위로 이어서 측정합니다. raw frame과 tile image는 `--stream-chunk-frames`만큼 처리한 뒤 버립니다.

해상도별 추천 조합과 로컬 실측 결과는 `docs/STREAMING_PIPELINE_CONFIG_RECOMMENDATIONS_KO.md`에 정리했습니다. 재사용 가능한 preset은 `configs/repro/streaming_pipeline_profiles.yaml`입니다.

keep-all baseline부터 memory와 decode/tiling 시간을 확인하려면:

```bash
.venv/bin/python -m repro.nvila_runner \
  --mode stream-profile \
  --device cpu \
  --stream-decode-strategy seek \
  --video inputs/hlvid_example/clip_av_video_5_001.mp4 \
  --gazing-mode keep-all \
  --num-video-frames 128 \
  --num-video-frames-thumbnail 64 \
  --max-tiles-video 48 \
  --stream-chunk-frames 16 \
  --stream-profile-json outputs/autogaze_repro/stream_profile_hlvid_keep_all_128f.json
```

AutoGaze forward까지 실제로 재려면 `--gazing-mode autogaze`를 사용합니다. MPS에서는 `float32`가 안전하고, 4K/45-tile 설정이 무거우면 먼저 `--video-resize-shortest-edge 720`이나 `--max-tiles-video 1`로 path 확인을 하세요.
stream-profile의 AutoGaze transform은 `--autogaze-resize-scales`의 최대 scale에 맞춰 resize/crop size를 설정합니다. 예를 들어 기본 `56+112+196+392`에서는 AutoGaze 입력 tile이 392x392가 되어야 합니다.

```bash
.venv/bin/python -m repro.nvila_runner \
  --mode stream-profile \
  --device mps \
  --stream-dtype float32 \
  --stream-decode-strategy seek \
  --video inputs/hlvid_example/clip_av_video_5_001.mp4 \
  --gazing-mode autogaze \
  --num-video-frames 128 \
  --num-video-frames-thumbnail 64 \
  --max-tiles-video 48 \
  --stream-chunk-frames 16 \
  --max-batch-size-autogaze 4 \
  --video-resize-shortest-edge 720 \
  --stream-profile-json outputs/autogaze_repro/stream_profile_hlvid_autogaze_128f_resize720_mps.json
```

SigLIP vision tower까지 MPS에서 확인하려면 patch16 SigLIP와 AutoGaze target scale을 맞춥니다. 아래 command는 `google/siglip2-base-patch16-224`를 사용하므로 `32+64+112+224 / patch16`을 명시합니다.

```bash
.venv/bin/python -m repro.nvila_runner \
  --mode stream-profile \
  --device mps \
  --stream-dtype float32 \
  --video inputs/hf_space_autogaze/security.mp4 \
  --gazing-mode autogaze \
  --num-video-frames 64 \
  --num-video-frames-thumbnail 32 \
  --max-tiles-video 1 \
  --stream-chunk-frames 16 \
  --max-batch-size-autogaze 1 \
  --autogaze-target-scales 32+64+112+224 \
  --autogaze-target-patch-size 16 \
  --stream-run-siglip \
  --stream-siglip-mode both \
  --stream-siglip-model google/siglip2-base-patch16-224 \
  --stream-siglip-max-embed-batch-size 1 \
  --stream-profile-json outputs/autogaze_repro/security_64f_1tile_siglip_google_both_mps.json
```

`--stream-siglip-mode gazed`는 AutoGaze가 선택한 patch만 SigLIP에 넣습니다. `both`는 같은 chunk에서 keep-all SigLIP도 한 번 더 돌려 비교값을 남깁니다. keep-all은 sequence length가 커서 긴 4K/다중 tile 조건에서는 먼저 `gazed`로 확인하세요.

이 모드에서 개별 timing은 모두 분리됩니다.

- `timing_ms.video_decode_scan`: 마지막 sampled frame index까지 비디오 프레임을 순차 decode한 시간입니다. 모든 frame을 메모리에 쌓지는 않지만, 마지막 샘플이 비디오 끝에 있으면 decode scan 자체는 길어집니다.
- `timing_ms.video_keyframe_index_scan`: `--stream-decode-strategy seek`에서 packet metadata만 읽어 keyframe index를 만드는 시간입니다. 4K HLVid example에서는 전체 8992 packet scan이 약 0.15초였습니다.
- `timing_ms.video_seek`, `timing_ms.video_decode_seek`: `seek` 전략에서 target frame이 속한 keyframe group으로 이동하고, target frame까지 필요한 GOP 구간만 decode한 시간입니다.
- `timing_ms.video_frame_to_pil`: sampled frame만 PIL RGB image로 변환한 시간입니다.
- `timing_ms.video_frame_resize`: `--video-resize-*`를 켰을 때 sampled frame resize 시간입니다.
- `timing_ms.spatial_tile_build`: 현재 chunk의 sampled frames를 NVILA dynamic tile grid로 리사이즈/crop하는 시간입니다.
- `timing_ms.tile_autogaze_tensorize`: tile image들을 AutoGaze input tensor로 바꾸는 시간입니다.
- `timing_ms.tile_autogaze_forward`: AutoGaze 모델 forward 시간입니다. `keep-all` 모드에서는 null입니다.
- `timing_ms.siglip_gazed_forward`: `--stream-run-siglip` 사용 시 AutoGaze가 선택한 patch sequence만 custom SigLIP vision tower에 넣은 forward 시간입니다.
- `timing_ms.siglip_keep_all_forward`: `--stream-siglip-mode keep-all` 또는 `both` 사용 시 같은 chunk의 전체 patch sequence를 SigLIP에 넣은 baseline forward 시간입니다.
- `timing_ms.keep_all_mask_build`: `keep-all` 모드에서 AutoGaze 없이 raw patch를 전부 유지하는 mask/count를 만드는 시간입니다.
- `timing_ms.thumbnail_resize`, `timing_ms.thumbnail_tensorize`: thumbnail frame resize와 tensorization 시간입니다.
- `timing_ms.eof_sample_padding`: container metadata frame count와 실제 decode 가능한 frame count가 어긋날 때 마지막 decoded sampled frame을 반복해 requested sample count를 맞추는 시간입니다.
- `timing_ms.pre_llm_stream_total_measured`: 위 stage들의 합입니다.

토큰/패치 필드는 `single` 모드의 `token_metrics`와 같은 이름을 최대한 유지합니다.

- `token_metrics.encoder_raw_tile_patch_tokens`: sampled video frames × spatial tiles × multi-scale patches per frame
- `token_metrics.autogaze_input_patch_tokens`, `token_metrics.autogaze_selected_patch_tokens`, `token_metrics.autogaze_removed_patch_tokens`: AutoGaze 입력 tiled-video patch 수, 선택 patch 수, 제거 patch 수입니다. thumbnail은 포함하지 않습니다.
- `token_metrics.encoder_autogaze_selected_tile_patch_tokens`: AutoGaze 이후 유지된 tile patch 수. `keep-all`에서는 raw tile patch와 같습니다.
- `token_metrics.encoder_raw_thumbnail_patch_tokens`, `token_metrics.encoder_autogaze_selected_thumbnail_patch_tokens`: thumbnail patch budget과 유지 patch 수입니다. 현재 thumbnail은 keep-all입니다.
- `token_metrics.encoder_raw_patch_tokens`, `token_metrics.encoder_autogaze_selected_patch_tokens`, `token_metrics.encoder_token_reduction_ratio`: tile+thumbnail total 기준 patch 감소량입니다.
- `token_metrics.llm_keep_all_visual_tokens_estimated`: TokenShuffle 이후 keep-all visual token 추정치입니다.
- `token_metrics.llm_autogaze_visual_tokens_lower_bound_estimated`: selected patch를 TokenShuffle로 묶었을 때의 lower-bound 추정치입니다. 실제 LLM token 수는 public NVILA generate path에서 visual token sequence를 모은 뒤에만 확정됩니다.

stream-profile payload에도 `autogaze_token_summary`가 같이 들어갑니다. full NVILA `single/hlvid`에서는 `llm_actual_visual_tokens`가 실제 processor output 기준으로 확정되고, stream-profile에서는 LLM을 실행하지 않으므로 `llm_autogaze_visual_tokens_lower_bound_estimated`를 참고값으로 봅니다.

`stream-profile`도 `compute_metrics`를 남깁니다.

- `compute_metrics.siglip_encoder.keep_all` / `actual`: stream chunk에서 나온 tile gaze slot과 thumbnail keep-all slot을 기준으로 SigLIP attention/MLP MACs를 추정합니다.
- `compute_metrics.siglip_encoder.keep_all_to_actual_dense_attention_pair_ratio`: keep-all 대비 AutoGaze 적용 후 dense attention pair 감소 비율입니다.
- `compute_metrics.siglip_encoder.keep_all_to_actual_total_macs_ratio`: attention projection, attention N^2, MLP를 합친 SigLIP encoder 계산량 감소 추정치입니다.
- `compute_metrics.mllm.full_llm_not_run_in_stream_profile`: `true`입니다. stream-profile은 projector/LLM을 실행하지 않으므로 `single`/`hlvid --measure-ttft` 결과의 `compute_metrics.mllm`과 `ttft_ms`를 같이 봐야 합니다.

메모리 필드는 stream 처리 경계 확인용입니다.

- `memory_bytes.raw_frame_buffer_peak`: 현재 구현이 동시에 보유한 sampled raw frame buffer peak입니다.
- `memory_bytes.tile_pil_buffer_peak`: 현재 chunk에서 만든 PIL tile buffer peak입니다.
- `stream_plan.memory.streaming_autogaze_tile_tensor_bytes_per_batch`: `--max-batch-size-autogaze`만큼 tile sequence를 tensorize할 때의 예상 AutoGaze tensor peak입니다.
- `stream_plan.memory.streaming_autogaze_tile_tensor_bytes_full_chunk`: 한 temporal chunk의 모든 spatial tile을 한 번에 tensorize한다고 가정했을 때의 비교용 크기입니다.
- `memory_bytes.autogaze_tile_tensor_peak_per_temporal_chunk`: 실제 AutoGaze tensor peak입니다. 현재 구현은 `--max-batch-size-autogaze` 단위로 tensorization/forward를 나눠서 full chunk tensor를 만들지 않습니다. `keep-all`에서는 0입니다.
- `memory_bytes.siglip_gazed_hidden_peak`, `memory_bytes.siglip_keep_all_hidden_peak`: SigLIP output hidden state의 peak 크기입니다. attention 내부 activation 전체 peak는 CUDA에서 `cuda_peak_memory_bytes`와 함께 봐야 합니다.
- `memory_bytes.thumbnail_tensor`: thumbnail tensor 크기입니다.
- `memory_bytes.cuda_peak_memory_bytes`: CUDA에서 실행했을 때 PyTorch peak allocation입니다.

stream-profile의 측정값을 코드에서 따라가려면 아래 위치를 보면 됩니다.

| 측정 그룹 | 코드 위치 |
| --- | --- |
| stage timer 누적 | [StageProfiler:L45](../repro/nvila_runner.py#L45) |
| decode/resize/tiling stream loop | [run_stream_profile:L1822](../repro/nvila_runner.py#L1822) |
| AutoGaze tensorize/forward | [run_autogaze_on_stream_tile_sequences:L1576](../repro/nvila_runner.py#L1576) |
| gazed/keep-all SigLIP forward | [run_siglip_on_stream_batch:L1527](../repro/nvila_runner.py#L1527) |
| stream token/patch count | [build_stream_profile_token_metrics:L313](../repro/nvila_runner.py#L313) |
| stream SigLIP/MLLM compute estimate | [build_stream_profile_compute_metrics:L362](../repro/nvila_runner.py#L362) |
| full NVILA hook 기반 timing | [ProfilePatches:L81](../repro/nvila_runner.py#L81), [forward hooks:L95-L100](../repro/nvila_runner.py#L95-L100) |

중요한 경계가 하나 있습니다. 이 모드는 decode, resize, tiling, AutoGaze와 선택적 custom SigLIP forward까지는 chunk streaming으로 측정합니다. 하지만 public NVILA generation path는 최종 visual token sequence를 모아서 LLM prefill/generation에 넣습니다. 그래서 projector와 LLM forward 시간은 기존 `single`/`hlvid` 모드의 `result.mm_projector_ms`, `result.llm_forward_ms`, `result.ttft_ms`로 비교하세요. full NVILA path에서의 vision encoder hook은 여전히 `result.siglip_vision_ms`에도 기록됩니다.

이 workspace에서 확인한 긴 비디오 smoke:

```bash
.venv/bin/python -m repro.nvila_runner \
  --mode stream-profile \
  --device cpu \
  --stream-decode-strategy seek \
  --video inputs/hlvid_example/clip_av_video_5_001.mp4 \
  --gazing-mode keep-all \
  --num-video-frames 16 \
  --num-video-frames-thumbnail 8 \
  --max-tiles-video 1 \
  --stream-chunk-frames 16 \
  --stream-profile-json outputs/autogaze_repro/stream_profile_hlvid_example_keep_all_16f_1tile.json
```

`seek` 결과는 4K/약 5분/8992프레임 MP4에서 `decoded_selected_frames=16`, `decode_frames_read=124`, `raw_frame_buffer_peak=398131200` bytes, `video_keyframe_index_scan≈150 ms`, `video_decode_seek≈792 ms`, `spatial_tile_build≈696 ms`였습니다. 기존 scan 방식은 같은 16프레임에서도 끝까지 8992프레임을 decode해서 `video_decode_scan≈53660-68639 ms`가 걸렸습니다.

128프레임 keep-all smoke에서는 `decode_frames_read=812`, `video_keyframe_index_scan≈150 ms`, `video_decode_seek≈5584 ms`였습니다. 즉, 5분 4K 비디오를 끝까지 full decode하지 않고, keyframe group별로 필요한 GOP 구간만 decode하는 경로가 동작합니다.

이 workspace에서 확인한 SigLIP 포함 smoke:

- `outputs/autogaze_repro/security_64f_1tile_siglip_google_both_mps.json`: 64프레임을 16프레임 chunk 4개로 처리했고, `siglip_gazed_forward≈1054 ms`, `siglip_keep_all_forward≈15209 ms`, tile patch 감소율 `≈41.8x`였습니다.
- `outputs/autogaze_repro/hlvid_4k_16f_1tile_siglip_google_gazed_mps.json`: HLVid 4K/약 5분 비디오에서 16프레임, 1 tile, gazed SigLIP까지 통과했습니다. 이 파일은 기존 scan 결과라 `video_decode_scan≈68639 ms`가 포함되어 있고, 새 실행에서는 `--stream-decode-strategy seek`를 붙여야 합니다.
- `outputs/autogaze_repro/hlvid_4k_16f_1tile_siglip_google_gazed_seek_mps.json`: 같은 HLVid 4K/16프레임 조건에서 seek decode를 적용한 결과입니다. `decode_frames_read=124`, `video_keyframe_index_scan≈235 ms`, `video_decode_seek≈820 ms`, `tile_autogaze_forward≈3584 ms`, `siglip_gazed_forward≈291 ms`, `pre_llm_stream_total_measured≈6417 ms`였습니다.
- `outputs/autogaze_repro/hlvid_4k_128f_1tile_siglip_google_both_seek_mps.json`: 같은 HLVid 4K/128프레임/64 thumbnail/1 tile 조건에서 `both`를 돌렸습니다. `tile_autogaze_forward≈10839 ms`, `siglip_gazed_forward≈1007 ms`, `siglip_keep_all_forward≈14856 ms`, tile patch 감소 `≈16.1x`, SigLIP attention MAC 감소 `≈29.1x`, total SigLIP MAC 감소 `≈4.10x`였습니다.
- `outputs/autogaze_repro/hlvid_720p_128f_1tile_siglip_google_both_seek_mps.json`: 같은 128프레임을 runner-side `--video-resize-shortest-edge 720`으로 돌렸습니다. `tile_autogaze_forward≈10033 ms`, `siglip_gazed_forward≈745 ms`, `siglip_keep_all_forward≈14829 ms`였습니다.
- `outputs/autogaze_repro/hlvid_128f_siglip_autogaze_tradeoff_report.json`: 위 두 결과에서 branch별 비교값만 뽑은 요약입니다.

128프레임 `both` 결과를 해석할 때는 `pre_llm_stream_total_measured`를 그대로 비교하면 안 됩니다. `both`는 AutoGaze+gazed SigLIP와 keep-all SigLIP을 한 번에 모두 실행하므로, 총합에는 두 branch가 동시에 들어 있습니다. 순수 모델 forward 기준으로는 아래처럼 봅니다.

| 조건 | keep-all SigLIP only | AutoGaze forward + gazed SigLIP | 순수 모델 forward speedup | 추정 stream speedup |
| --- | ---: | ---: | ---: | ---: |
| HLVid 4K 128f | 14.86s | 11.85s | 1.25x | 1.10x |
| HLVid 720p 128f | 14.83s | 10.78s | 1.38x | 1.14x |

따라서 이 MPS smoke만 놓고 보면 “SigLIP 자체”는 크게 빨라집니다. 4K는 `14.86s -> 1.01s`, 720p는 `14.83s -> 0.74s`입니다. 하지만 AutoGaze forward가 `10-11s`라서 AutoGaze까지 포함한 vision-only 순이득은 아직 작습니다. 현재 가장 강한 장점은 latency보다 token/compute/memory 근거입니다. tile patch는 `16.1x`, LLM visual token lower-bound는 `5760 -> 2120`으로 `2.72x`, SigLIP attention MACs는 `29.1x`, SigLIP hidden peak는 `13.0 MB -> 0.9 MB` 수준으로 줄었습니다. full NVILA MLLM에서는 줄어든 visual token이 prefill context와 KV cache를 줄이므로, CUDA에서 `single/hlvid --measure-ttft`로 `ttft_ms`, `llm_forward_ms`, `compute_metrics.mllm.*`, `llm_peak_memory_bytes`를 확인해야 최종 latency claim을 만들 수 있습니다.

## HLVid example AutoGaze-only smoke

기본 NVILA runner example 비디오는 `bfshi/HLVid`의 `example/clip_av_video_5_001.mp4`입니다. 4K, 약 5분짜리 MP4이고 파일 크기는 약 1.7 GB라서 로컬 다운로드에는 충분한 디스크 여유 공간이 필요합니다.

resume 가능한 다운로드:

```bash
.venv/bin/python scripts/download_hlvid_example_video.py
```

파일은 `inputs/hlvid_example/clip_av_video_5_001.mp4`에 저장됩니다. 중간에 끊기면 같은 명령을 다시 실행하세요. 이미 받은 byte 이후부터 이어받도록 `Range` 요청을 보냅니다.

디스크 공간이 부족하면 다운로드 없이 remote URL을 직접 streaming해서 AutoGaze-only smoke를 돌릴 수 있습니다. 이 실행은 NVILA나 SigLIP를 로드하지 않습니다. 대신 NVILA video sampling shape를 흉내 내서 전체 비디오에서 `128`프레임을 uniform sampling하고, 16프레임 AutoGaze chunk로 나누며, 4K dynamic spatial tiling을 `max_tiles_video=48` 기준으로 적용하고, `64`개 thumbnail frame도 NVILA thumbnail subsampling policy로 처리합니다.

```bash
.venv/bin/python -m repro.hlvid_example_autogaze \
  --device mps \
  --num-video-frames 128 \
  --num-video-frames-thumbnail 64 \
  --max-tiles-video 48 \
  --max-batch-size-autogaze 16 \
  --output-json outputs/autogaze_repro/hlvid_example_autogaze_only_128f.json
```

다운로드한 파일을 사용하려면:

```bash
.venv/bin/python -m repro.hlvid_example_autogaze \
  --video inputs/hlvid_example/clip_av_video_5_001.mp4 \
  --device cuda \
  --dtype float16 \
  --num-video-frames 128 \
  --num-video-frames-thumbnail 64 \
  --max-tiles-video 48 \
  --output-json outputs/autogaze_repro/hlvid_example_autogaze_only_128f_cuda.json
```

4K HLVid example에서 `max_tiles_video=48`은 `9x5=45` spatial tile로 결정됩니다. 128 sampled frame과 16-frame temporal chunk를 쓰면 `8 * 45 = 360`개의 AutoGaze tile sequence가 생깁니다. 이 smoke는 NVILA-8B를 로드하기 전에 AutoGaze sampling, chunking, tiling, thumbnail 처리, token reduction이 기대대로 동작하는지 확인하는 용도입니다.

## HLVid manifest

```bash
.venv/bin/python -m repro.hlvid manifest \
  --config default \
  --split test \
  --output data/hlvid/manifest_test.json
```

manifest 명령은 `datasets.load_dataset` 대신 Hugging Face Dataset Viewer API를 사용합니다. 그래서 전체 비디오 payload를 다운로드하지 않고 metadata를 수집할 수 있습니다. Hugging Face dataset card 기준으로 현재 `test` split은 268개 row와 약 152 GB 파일을 노출합니다.

작은 metadata check:

```bash
.venv/bin/python -m repro.hlvid manifest \
  --config default \
  --split test \
  --limit 5 \
  --output data/hlvid/manifest_test_5.json
```

## HLVid dry run

```bash
.venv/bin/python -m repro.nvila_runner \
  --mode hlvid \
  --device cuda \
  --config default \
  --split test \
  --limit 1 \
  --num-video-frames 128 \
  --num-video-frames-thumbnail 64 \
  --max-tiles-video 48 \
  --measure-ttft \
  --warmup-runs 1 \
  --resume \
  --predictions outputs/autogaze_repro/hlvid_dry_run_predictions.jsonl \
  --summary outputs/autogaze_repro/hlvid_dry_run_summary.json \
  --scored-predictions outputs/autogaze_repro/hlvid_dry_run_scored.jsonl
```

## HLVid 논문 대응 실행

fixed total-frame sampling setup용 preset은 `configs/repro/hlvid_like_nvila_1024.yaml`입니다.

```bash
.venv/bin/python -m repro.nvila_runner \
  --preset-config configs/repro/hlvid_like_nvila_1024.yaml
```

동일한 expanded command:

```bash
.venv/bin/python -m repro.nvila_runner \
  --mode hlvid \
  --device cuda \
  --gazing-mode autogaze \
  --config default \
  --split test \
  --limit 268 \
  --num-video-frames 1024 \
  --num-video-frames-thumbnail 64 \
  --max-tiles-video 48 \
  --measure-ttft \
  --warmup-runs 1 \
  --resume \
  --predictions outputs/autogaze_repro/hlvid_full_predictions.jsonl \
  --summary outputs/autogaze_repro/hlvid_full_summary.json \
  --scored-predictions outputs/autogaze_repro/hlvid_full_scored.jsonl
```

matching keep-all baseline은 모든 설정을 동일하게 유지하고 gaze mode와 output path만 바꿉니다.

```bash
.venv/bin/python -m repro.nvila_runner \
  --mode hlvid \
  --device cuda \
  --gazing-mode keep-all \
  --config default \
  --split test \
  --limit 268 \
  --num-video-frames 1024 \
  --num-video-frames-thumbnail 64 \
  --max-tiles-video 48 \
  --measure-ttft \
  --warmup-runs 1 \
  --resume \
  --predictions outputs/autogaze_repro/hlvid_full_keep_all_predictions.jsonl \
  --summary outputs/autogaze_repro/hlvid_full_keep_all_summary.json \
  --scored-predictions outputs/autogaze_repro/hlvid_full_keep_all_scored.jsonl
```

`accuracy_scored`를 NVILA-8B-HD-Video의 project-page HLVid target인 `52.6`과 비교하세요. skipped, failed, parse-failed sample은 별도로 보고해야 합니다. AutoGaze-vs-keep-all claim을 만들 때는 accuracy뿐 아니라 median 또는 mean 기준 `total_ms`, `video_decode_ms`, `video_tiling_ms`, `autogaze_forward_ms`, `vision_encoder_ms`, `llm_forward_ms`, `token_metrics.encoder_token_reduction_ratio`, `token_metrics.llm_visual_token_reduction_ratio`를 함께 비교하세요. 논문 대응 setup은 target GPU가 허용하는 범위에서 NVILA-8B-HD-Video, 최대 1024프레임, 최대 해상도 3584를 기준으로 합니다.

### HLVid 폴더 기반 일괄 benchmark

데이터셋을 로컬 폴더로 받은 경우에는 `scripts/run_hlvid_folder_benchmark.py`를 쓰면 됩니다. 이 스크립트는 폴더에서 manifest를 찾고, `keep-all`과 `autogaze`를 같은 설정으로 각각 실행한 뒤, accuracy/속도/메모리/token/compute gain report를 만듭니다.

지원하는 폴더 형태:

```text
/path/to/HLVid/
  manifest_test.json        # 또는 manifest*.jsonl, metadata.jsonl, csv, parquet
  videos/
    clip_av_video_5_001.mp4
    ...
```

Hugging Face snapshot 그대로 받은 경우도 지원합니다.

```text
/path/to/HLVid/
  data/
    test-00000-of-00001.parquet
  example/
    clip_av_video_5_001.mp4
  videos_part_0001.tar
  ...
  videos_part_0016.tar
```

이 형태에서는 `data/test-*.parquet`를 manifest로 자동 인식합니다. 다만 full benchmark는 `video_path`가 가리키는 mp4 파일이 실제 filesystem에 있어야 하므로, `videos_part_*.tar`를 별도 위치에 풀고 `--video-root /path/to/extracted/videos`를 넘기세요. tar만 있는 상태는 준비 report에서는 감지되지만 full NVILA 실행은 할 수 없습니다.

실행 전에 다운로드/추출 상태만 확인하려면:

```bash
.venv/bin/python scripts/run_hlvid_folder_benchmark.py \
  --dataset-dir /path/to/HLVid \
  --video-root /path/to/extracted/videos \
  --prepare-only \
  --layout-report outputs/autogaze_repro/hlvid_dataset_layout_report.json
```

report에서 `ready_for_full_benchmark=true`, `missing_videos=0`이면 full run 준비가 된 상태입니다. `video_archive_count=16`인데 `missing_videos>0`이면 HF tar archive는 있지만 아직 mp4가 추출되지 않은 상태입니다.

manifest는 `question_id`, `category`, `video_path`, `question`, `answer` 컬럼을 가져야 합니다. 파일명이 다르면 `--manifest`로 직접 지정하세요. `video_path`는 manifest 기준 문자열이고, runner는 자동 발견된 video root 또는 `--video-root`로 전달된 폴더 아래에서 찾습니다.
full benchmark 실행 전에도 같은 video-file preflight를 돌립니다. 누락된 mp4가 있으면 기본적으로 실행을 멈추므로, 일부 샘플만 의도적으로 실패 처리하려는 경우에만 `--allow-missing-videos --continue-on-error`를 같이 쓰세요.

### HLVid를 일반 inference 입력으로 쓰기

HLVid는 benchmark용 manifest지만, 각 row는 `video_path + question` 형태라 일반 inference 입력으로도 그대로 쓸 수 있습니다. 한 파일을 직접 지정할 때는 `single` 모드를 씁니다.

```bash
.venv/bin/python -m repro.nvila_runner \
  --mode single \
  --video /path/to/HLVid/videos/clip_av_video_5_001.mp4 \
  --prompt "What does the white text on the green road sign say? A. Hampden St B. Hampden Ave C. Hampden Blvd D. Hampden Rd Please answer directly with the letter of the correct answer." \
  --device cuda \
  --gazing-mode autogaze \
  --num-video-frames 128 \
  --num-video-frames-thumbnail 64 \
  --max-tiles-video 8 \
  --measure-ttft \
  --warmup-runs 1 \
  --repeat-runs 3 \
  --print-summary \
  --summary-json outputs/autogaze_repro/single_hlvid_infer_autogaze_summary.json \
  --output-json outputs/autogaze_repro/single_hlvid_infer_autogaze.json
```

같은 파일의 keep-all baseline도 필요하면 `--gazing-mode keep-all`과 다른 `--output-json`으로 한 번 더 실행합니다. 이렇게 나온 두 JSON의 `total_ms`, `siglip_vision_ms`, `mm_projector_ms`, `llm_forward_ms`, `ttft_ms`, `token_metrics.*`, `compute_metrics.*`, `*_peak_memory_bytes`를 비교하면 단일 파일 기준 속도/토큰/메모리 gain을 볼 수 있습니다.

manifest 여러 row를 순회하되 benchmark wrapper 없이 prediction만 보고 싶으면 `hlvid` 모드를 직접 씁니다.

```bash
.venv/bin/python -m repro.nvila_runner \
  --mode hlvid \
  --manifest /path/to/HLVid/data/test-00000-of-00001.parquet \
  --hlvid-video-root /path/to/HLVid/videos \
  --device cuda \
  --gazing-mode autogaze \
  --limit 10 \
  --num-video-frames 128 \
  --num-video-frames-thumbnail 64 \
  --max-tiles-video 8 \
  --measure-ttft \
  --warmup-runs 1 \
  --predictions outputs/autogaze_repro/hlvid_infer_predictions.jsonl \
  --summary outputs/autogaze_repro/hlvid_infer_summary.json \
  --scored-predictions outputs/autogaze_repro/hlvid_infer_scored.jsonl \
  --continue-on-error
```

이 경우 `predictions.jsonl`의 각 줄이 일반 inference 결과이며 각 줄에 `question`이 그대로 들어갑니다. `summary`에는 전체 개수와 앞쪽 `question_samples`, 그리고 `benchmark_samples`만 들어가므로, 특정 샘플의 질문/출력/토큰/메모리를 같이 볼 때는 predictions 파일을 우선 보면 됩니다.

CUDA full benchmark 예시:

```bash
.venv/bin/python scripts/run_hlvid_folder_benchmark.py \
  --dataset-dir /path/to/HLVid \
  --output-dir outputs/autogaze_repro/hlvid_folder_1024 \
  --device cuda \
  --num-video-frames 1024 \
  --num-video-frames-thumbnail 64 \
  --max-tiles-video 48 \
  --max-batch-size-autogaze 16 \
  --max-batch-size-siglip 32 \
  --measure-ttft \
  --warmup-runs 1 \
  --continue-on-error
```

처음 CUDA 검증은 더 작은 설정으로 시작하세요.

```bash
.venv/bin/python scripts/run_hlvid_folder_benchmark.py \
  --dataset-dir /path/to/HLVid \
  --output-dir outputs/autogaze_repro/hlvid_folder_smoke \
  --device cuda \
  --limit 3 \
  --num-video-frames 128 \
  --num-video-frames-thumbnail 64 \
  --max-tiles-video 8 \
  --video-resize-shortest-edge 720 \
  --measure-ttft \
  --warmup-runs 1 \
  --continue-on-error
```

생성되는 주요 파일:

- `hlvid_keep_all_predictions.jsonl`: AutoGaze 미적용 baseline per-sample 결과
- `hlvid_autogaze_predictions.jsonl`: AutoGaze 적용 per-sample 결과
- `hlvid_keep_all_summary.json`, `hlvid_autogaze_summary.json`: 각 run의 HLVid scoring summary
- `hlvid_autogaze_gain_report.json`: keep-all 대비 AutoGaze gain report
- `hlvid_autogaze_gain_report.csv`: 리더 보고용 single-row 요약

리더 리뷰용으로 빠르게 샘플을 확인할 때는 `hlvid_autogaze_gain_report.json`의 `benchmark_samples.autogaze`를 먼저 보세요. 각 항목에는 `target_video`, `question`, `model_answer`, `parsed_model_answer`, `correct_answer`, `ground_truth_answer`, `correct`, `status`가 들어갑니다. keep-all과 AutoGaze의 같은 샘플을 비교하려면 `benchmark_samples.keep_all`도 같이 봅니다.

`--limit 3`으로 실행했을 때 `readable_summary.run_counts.autogaze_rows=3`이면 AutoGaze 모드가 HLVid row 3개를 처리했다는 뜻입니다. wrapper에서 keep-all과 AutoGaze를 둘 다 켠 기본 상태라면 `keep_all_rows=3`, `autogaze_rows=3`이 각각 생깁니다. `--warmup-runs`로 실행된 warmup은 predictions/scoring row에 포함하지 않습니다.

`--skip-keep-all`로 AutoGaze만 돌린 경우에도 report에는 keep-all 섹션과 `readable_summary.mode_status.keep_all`이 남습니다. 이때 `keep_all_rows=0`, keep-all metric은 0 또는 빈 값으로 보이고, 실제 baseline이 없으므로 cross-mode speedup/reduction ratio는 `null`로 표시됩니다. AutoGaze 자체의 before/after token 감소율은 AutoGaze row 안의 keep-all estimate와 actual 값으로 계속 계산됩니다.

`hlvid_autogaze_gain_report.json`에서 우선 볼 항목:

- `readable_summary.latency_ms_median`: `total_ms`, `video_decode_ms`, `video_tiling_ms`, `autogaze_forward_ms`, `siglip_vision_ms`, `mm_projector_ms`, `llm_forward_ms`, `ttft_ms`를 keep-all/autogaze median과 함께 보여줍니다.
- `readable_summary.memory_bytes_median`: CUDA peak memory를 keep-all/autogaze median으로 비교합니다.
- `readable_summary.tokens_median`: encoder patch, AutoGaze input tile patch, LLM visual token을 before/after 형태로 보여줍니다.
- `gains.accuracy_scored_delta`: AutoGaze와 keep-all의 HLVid accuracy 차이
- `gains.latency_speedup_median.total_ms`: 전체 end-to-end median speedup
- `gains.latency_speedup_median.siglip_vision_ms`: SigLIP vision tower median speedup
- `gains.latency_speedup_median.llm_forward_ms`, `gains.latency_speedup_median.ttft_ms`: MLLM 쪽 latency speedup
- `gains.memory_reduction_ratio_median.llm_peak_memory_bytes`: LLM generate CUDA peak memory 감소 비율
- `gains.autogaze_token_reduction_median.llm_visual_token_reduction_ratio`: LLM visual token 감소 비율
- `gains.compute_reduction_median.siglip_total_macs`, `gains.compute_reduction_median.mllm_kv_cache`: 계산량/KV cache 감소 추정
- `gains.reduction_percent_median`: `ratio` 대신 `(before - after) / before * 100`으로 계산한 감소율입니다. 여기서 분모는 keep-all 또는 AutoGaze 적용 전 값입니다.

ratio와 percent는 의도가 다릅니다. `*_ratio_*`는 `before_or_keep_all / after_or_autogaze`라 2.0이면 “2배 작아짐”입니다. `reduction_percent_median`은 원래 값을 분모로 둔 감소율이라 50이면 “원래 대비 50% 감소”입니다.

이미 prediction JSONL이 있는 상태에서 report만 다시 만들려면 같은 `--output-dir`에 대해 `--report-only`를 붙입니다.

## 검증

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/python -m repro.autogaze_bench --help
.venv/bin/python -m repro.hlvid --help
.venv/bin/python scripts/run_hlvid_folder_benchmark.py --help
.venv/bin/python -m repro.nvila_runner --help
.venv/bin/python -m repro.report --help
```
