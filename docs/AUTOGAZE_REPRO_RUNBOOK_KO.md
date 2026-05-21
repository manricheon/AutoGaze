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

- `result.total_ms`: end-to-end latency입니다. additive하게 다시 계산하려면 `result.latency_accounting.additive_total_ms`를 보세요. 현재 primary 공식은 `total_ms = video_preprocess_without_autogaze_ms + autogaze_total_ms + generate_ms`입니다. 기존 비교용 inclusive 공식 `total_ms = video_preprocess_ms + generate_ms`는 `legacy_inclusive_total_ms`에 남깁니다.
- `result.video_preprocess_without_autogaze_ms`: AutoGaze stage를 뺀 video preprocess 시간입니다. `video_decode_ms`, `video_tiling_ms`, processor/tokenization overhead를 포함합니다.
- `result.autogaze_total_ms`: AutoGaze stage 전체 시간입니다. `gazing_info_total_ms`/`autogaze_ms`와 같은 stage를 primary formula에서 별도 항으로 보여주기 위한 alias입니다.
- `result.video_decode_ms`: 샘플링된 프레임 decode/read 시간입니다. runner-side resize를 쓰지 않으면 NVILA remote code의 video loader를 감싼 시간입니다. `--video-resize-*`를 쓰면 runner가 전체 비디오에서 프레임을 샘플링하고 PIL frame을 리사이즈하는 시간까지 포함합니다. 이때 `--video-decode-strategy auto|seek|scan`으로 샘플링 decode 방식을 고를 수 있고, 기본값 `auto`는 keyframe seek를 먼저 시도한 뒤 실패하면 scan으로 fallback합니다.
- `result.video_tiling_ms`: 프레임이 준비된 뒤 NVILA processor가 비디오를 준비하는 시간입니다. dynamic spatial tiling, thumbnail 생성, SigLIP/AutoGaze 입력 tensorization이 포함됩니다. SigLIP inference 시간은 아닙니다.
- `result.autogaze_ms`: 전체 AutoGaze selection stage 시간입니다. `autogaze` 모드에서는 AutoGaze forward와 sort/pad/split bookkeeping이 들어갑니다. `keep-all` 모드에서는 AutoGaze forward를 건너뛰고 keep-all mask를 만드는 시간이 대부분입니다.
- `result.gazing_info_total_ms`: `autogaze_ms`와 같은 의미의 명시적 alias입니다. “AutoGaze 모델만”이 아니라 gazing info stage 전체입니다.
- `result.autogaze_forward_ms`: AutoGaze model forward만 잰 시간입니다. AutoGaze 모델 자체 cost를 볼 때 가장 깨끗한 필드입니다.
- `result.autogaze_model_forward_ms`: `autogaze_forward_ms`와 같은 의미의 명시적 alias입니다.
- `result.vision_encoder_ms`: generation 중 NVILA visual embedding 경로를 감싼 시간입니다. SigLIP feature extraction, feature cleanup/reordering, projector 준비가 포함됩니다.
- `result.siglip_vision_ms`: SigLIP vision tower forward 시간입니다. AutoGaze가 vision encoder workload를 줄였는지 볼 때 중요합니다.
- `result.mm_projector_ms`: 선택/정렬된 vision feature를 MLLM 입력 차원으로 보내는 multimodal projector forward 시간입니다.
- `result.llm_forward_ms`: `generate` 내부에서 language model forward가 누적된 시간입니다. prefill과 decode 단계의 LLM 호출이 모두 포함됩니다.
- `result.ttft_ms`: `--measure-ttft`가 켜졌을 때, 처리된 visual/text input에서 1토큰을 생성하는 데 걸린 시간입니다. 별도의 1-token generation pass로 측정되며 `total_ms`에는 포함하지 않습니다.
- `result.generation_decode_after_ttft_estimated_ms`: full `generate_ms - ttft_ms`로 계산한 대략적인 generation decode 시간입니다. TTFT와 full generation이 별도 호출이므로 추정값으로 보세요. legacy alias인 `decode_estimated_ms`와 같은 값이며, 비디오 decode 시간이 아닙니다.
- `result.latency_accounting`: `total_ms`에 더해도 되는 additive field와, 이미 상위 시간에 포함된 nested breakdown을 분리한 설명 블록입니다. primary additive field는 `video_preprocess_without_autogaze_ms`, `autogaze_total_ms`, `generate_ms`입니다. `video_preprocess_ms`는 AutoGaze를 포함한 legacy inclusive field라 primary total에 다시 더하지 않습니다. `video_decode_ms`, `video_tiling_ms`, `autogaze_ms`, `siglip_vision_ms`, `llm_forward_ms`, `ttft_ms` 같은 값은 병목 분석용 하위 값입니다.
- `result.stage_timings_ms`: `processor`, 선택적 `ttft`, full `generate`의 raw nested timing bucket입니다. top-level field가 null이거나 call count까지 봐야 할 때 확인합니다.
- `result.token_metrics`: tile, thumbnail, total 기준 encoder patch budget과 LLM visual-token budget의 AutoGaze 전후 count입니다.
- `result.compute_metrics`: token count와 모델 config로 계산한 SigLIP encoder 및 MLLM prefill 계산량/메모리 추정치입니다. 실제 wall-clock은 위 latency 필드와 함께 봐야 합니다.
- `result.processor_peak_memory_bytes`, `result.ttft_peak_memory_bytes`, `result.llm_peak_memory_bytes`, `result.peak_memory_bytes`: CUDA 실행 시 processor phase, 1-token TTFT pass, full generate pass의 CUDA peak allocation입니다.

`--measure-ttft`는 preprocessing 이후 1토큰 generation을 추가로 실행합니다. 여기서 prefill은 LLM이 prompt text와 visual token 전체를 한 번에 읽는 첫 forward입니다. 이 forward가 이후 token-by-token decode에서 재사용할 KV cache를 만듭니다. 따라서 TTFT는 순수 text decoding latency가 아니라 visual embedding, SigLIP/vision encoding, projector work, 첫 LLM prefill forward까지 포함할 수 있습니다. 세부 분리는 `result.ttft_stage_timings_ms`의 `vision_encode_total`, `siglip_vision_tower`, `mm_projector`, `llm_forward`를 확인하세요.

단일 파일 inference도 같은 output JSON에 속도/토큰/메모리 필드를 남깁니다. `--measure-ttft` 없이도 `total_ms`, `generate_ms`, `video_preprocess_without_autogaze_ms`, `autogaze_total_ms`, `video_decode_ms`, `video_tiling_ms`, `gazing_info_total_ms`, `autogaze_model_forward_ms`, `siglip_vision_ms`, `mm_projector_ms`, `llm_forward_ms`, `latency_accounting`, `token_metrics`, `compute_metrics`, CUDA의 `processor_peak_memory_bytes`와 `llm_peak_memory_bytes`가 기록됩니다. `--measure-ttft`를 켜면 여기에 `ttft_ms`, `ttft_stage_timings_ms`, `ttft_peak_memory_bytes`, `generation_decode_after_ttft_estimated_ms`가 추가됩니다. MPS에서는 CUDA peak allocation API가 없어서 memory field가 null일 수 있습니다.

raw output JSON이 너무 길면 `--print-summary --summary-json <path>`를 붙이세요. 전체 raw JSON은 `--output-json`에 그대로 저장하고, 터미널과 summary file에는 답변과 함께 `prompt`, `question`, `video_input_summary`, `autogaze_token_summary`, `key_autogaze_effect`, `latency_accounting`를 별도로 정리합니다. `single` 모드에서는 `prompt`가 실행에 사용한 `--prompt` 원문이고, HLVid row 기반 실행에서는 per-row `question`이 `predictions.jsonl`과 `scored_predictions.jsonl`에 보존됩니다. `video_input_summary`에는 원본 총 프레임 수, 원본 해상도, 요청한 video/thumbnail frame 수, 실제 processor tensor 기준 frame 수, runner resize 적용 여부, resize 후 processor 입력 해상도, decode 전략/읽은 frame 수, spatial tile/temporal chunk 수가 들어갑니다. `autogaze_token_summary`에는 사용한 프레임/타일 기준 raw patch budget과 AutoGaze가 실제 유지한 patch 수, TokenShuffle 이후 LLM visual token 수가 나뉘어 들어갑니다. `key_autogaze_effect`에는 AutoGaze 전후 차이를 가장 잘 보여주는 encoder patch 수, LLM visual token 수, reduction ratio/percent, SigLIP/MLLM 계산량 감소 추정치, 핵심 latency/memory median이 모입니다. 상세 분석용 `latency_ms`, `memory_bytes`, `tokens`, `compute` 섹션도 함께 남깁니다.

결과를 공유용 Markdown으로 바꾸려면 `repro.markdown_report`를 사용하세요. single inference JSON, HLVid summary/gain report JSON, stream-profile JSON을 입력으로 받을 수 있고, 모델 pipeline, video/input 정보, frame/patch/tokenization, step-by-step module metrics, 핵심 latency/token/memory, benchmark score를 한 파일에 정리합니다. HLVid gain report의 `Module Detail Metrics`는 benchmark 비교형 latency detail이면 `keep_all`, `AutoGaze`, speedup, reduction을 같은 표에 보여주며, `--skip-autogaze`나 `--skip-keep-all`처럼 한쪽 mode가 없으면 해당 컬럼은 `-`로 남깁니다. keep-all과 AutoGaze를 둘 다 실행한 gain report에는 `Benchmark Correctness Comparison` 섹션도 추가되어 `both_correct`, `keep_all_only_correct`, `autogaze_only_correct`, `both_wrong`, missing bucket과 샘플 row를 같이 보여줍니다.

```bash
.venv/bin/python -m repro.markdown_report \
  --input-json outputs/autogaze_repro/hlvid_autogaze_gain_report.json \
  --output-md outputs/autogaze_repro/hlvid_autogaze_gain_report.md
```

선택된 프레임과 AutoGaze patch mask를 눈으로 확인하려면 `--visualization-output-dir`를 붙이세요. 실행이 끝난 뒤 사람이 보기 좋은 selected-frame 기준 `<label>_selected_frames.mp4`, selected-frame 위에 AutoGaze tile patch를 scale별 색상 mask로 표시한 `<label>_autogaze_overlay.mp4`, 실제 NVILA processor 입력 해상도 기준 `<label>_processor_frames.mp4`, processor-frame 위에 같은 AutoGaze mask를 표시한 `<label>_processor_autogaze_overlay.mp4`, 실제 `gazing_info`와 frame별 overlay box를 담은 `<label>_gazing_info.json`을 저장합니다. 기본 selected overlay video는 selected-frame video와 같은 해상도로 저장되고, processor overlay video는 `video_input_summary.processor_input_width/height` 기준으로 저장됩니다. patch 좌표는 processor가 실제로 쓰는 `cols * 392 x rows * 392` tile canvas에서 계산한 뒤 selected-frame 크기와 processor-frame 크기로 각각 매핑합니다. 오버레이는 외곽선을 굵게 그리지 않고 alpha mask만 적용합니다. 현재 runner에서는 thumbnail patch는 keep-all이므로 JSON에는 남기되 overlay video에는 tile AutoGaze 선택만 그립니다.

```bash
.venv/bin/python -m repro.nvila_runner \
  --mode single \
  --video inputs/hlvid_example/clip_av_video_5_001.mp4 \
  --video-resize-shortest-edge 720 \
  --num-video-frames 128 \
  --num-video-frames-thumbnail 64 \
  --max-tiles-video 8 \
  --gazing-mode autogaze \
  --visualization-output-dir outputs/autogaze_repro/visualizations \
  --visualization-fps 4 \
  --visualization-selected-max-long-side 1280 \
  --output-json outputs/autogaze_repro/nvila_single_with_viz.json
```

`--gazing-mode keep-all`에서도 같은 옵션을 사용할 수 있습니다. 이 경우 selected/processor frame 비디오는 저장되고 overlay는 `skipped_keep_all`로 기록됩니다. HLVid benchmark에서는 `repro.nvila_runner --mode hlvid`에 직접 같은 옵션을 붙이거나, batch wrapper에 `--visualization-output-dir`를 전달하면 keep-all/autogaze 양쪽 runner로 전달됩니다. 시각화 생성 시간은 `total_ms` 같은 benchmark latency에 포함하지 않고, 결과 JSON의 `result.visualization` 또는 per-row `visualization`에서 저장 경로와 decode 정보를 확인합니다. 관련 구현은 [gaze_visualization.py](../repro/gaze_visualization.py), runner 연결은 [nvila_runner.py](../repro/nvila_runner.py), batch forwarding은 [hlvid_batch_benchmark.py](../repro/hlvid_batch_benchmark.py)를 보세요.

단일 실행의 비디오 입력 조건을 빠르게 확인할 때는 아래 필드를 먼저 보세요. raw JSON에는 top-level `video_input_summary`와 `result.video_input_summary`가 모두 있고, compact summary JSON에도 같은 `video_input_summary`가 들어갑니다.

| 질문 | 볼 필드 |
| --- | --- |
| 원본 비디오가 몇 프레임/몇 해상도였나? | `video_input_summary.source_frames`, `source_resolution`, `source_fps`, `source_duration_seconds` |
| 우리가 몇 프레임을 요청했나? | `video_input_summary.requested_video_frames`, `requested_thumbnail_frames` |
| 실제 processor tensor 기준 몇 프레임이 들어갔나? | `video_input_summary.actual_video_frames`, `actual_thumbnail_frames` |
| 전체 비디오 중 어디까지 샘플링했나? | `video_input_summary.sampled_frame_start`, `sampled_frame_end` |
| runner-side resize를 켰나? | `video_input_summary.runner_resize_enabled`, `runner_resize_request` |
| NVILA processor에 넘긴 비디오 frame 해상도는? | `video_input_summary.processor_input_resolution`, `processor_input_width`, `processor_input_height` |
| 빠른 seek decode가 실제로 적용됐나? | `video_input_summary.video_decode_requested_strategy`, `video_decode_strategy`, `video_decode_strategy_fallback_error` |
| decode가 얼마나 덜 읽었나? | `video_input_summary.video_decode_frames_read`, `video_decode_seek_groups`, `video_decode_keyframes_indexed` |
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

가장 먼저 볼 한 줄 답은 아래입니다.

- `total_ms = video_preprocess_without_autogaze_ms + autogaze_total_ms + generate_ms`가 primary additive 공식입니다.
- 기존 `video_preprocess_ms`는 AutoGaze를 포함한 inclusive preprocess field입니다. 과거 로그와 비교하기 위해 `legacy_inclusive_total_ms`에 `total_ms = video_preprocess_ms + generate_ms`도 남깁니다.
- `generate_ms`는 preprocessing이 끝난 뒤 NVILA `model.generate()`를 잰 값입니다. 여기에는 `vision_encoder_ms`, `siglip_vision_ms`, `mm_projector_ms`, `llm_forward_ms`가 포함되지만, AutoGaze와 video decode는 포함되지 않습니다.
- AutoGaze stage 전체 시간은 `autogaze_total_ms`로 별도 top-level 항처럼 표시합니다. `gazing_info_total_ms`/`autogaze_ms`는 같은 stage의 alias이고, 순수 AutoGaze 모델 forward만 보려면 `autogaze_model_forward_ms`/`autogaze_forward_ms`를 봅니다.
- `video_decode_ms`는 `video_preprocess_without_autogaze_ms` 안에 포함됩니다. `generation_decode_after_ttft_estimated_ms`는 이름이 비슷해도 비디오 decode가 아니라 generation-side decode 추정값입니다.
- `ttft_ms`는 별도 1-token measurement pass라 `total_ms`에 포함하지 않고 더하지도 않습니다.

JSON/Markdown 로그에는 같은 내용을 기계가 읽기 쉬운 형태로도 남깁니다. `latency_accounting.hierarchy.ascii_tree`는 아래 트리를 그대로 담고, `latency_accounting.hierarchy.quick_answers`는 “AutoGaze가 generate에 포함되는가”, “video decode는 preprocess에 포함되는가” 같은 즉답을 담습니다. `latency_accounting.hierarchy.nodes`는 각 노드의 `included_in`, `add_to_total_ms`, alias, 측정값을 함께 기록합니다.

```text
total_ms = video_preprocess_without_autogaze_ms + autogaze_total_ms + generate_ms
|-- video_preprocess_without_autogaze_ms
|   |-- video_decode_ms (included; not an extra total term)
|   |-- video_tiling_ms (included; not an extra total term)
|   `-- other processor/tokenization overhead
|-- autogaze_total_ms
|   `-- autogaze_model_forward_ms / autogaze_forward_ms (model forward only)
`-- generate_ms
    |-- vision_encoder_ms
    |   |-- siglip_vision_ms
    |   `-- mm_projector_ms
    |-- llm_forward_ms
    `-- generation_decode_after_ttft_estimated_ms (generation-side estimate, not video decode)
ttft_ms: separate 1-token measurement pass, excluded from total_ms
```

| 상위/대표 필드 | 하위 또는 관련 필드 | 포함 관계 | 해석 |
| --- | --- | --- | --- |
| `total_ms` | `video_preprocess_without_autogaze_ms`, `autogaze_total_ms`, `generate_ms` | `generate_one()`이 최종 산출합니다. | 단일 run의 end-to-end latency입니다. `ttft_ms`는 별도 pass라 여기에 포함하지 않습니다. |
| `video_preprocess_without_autogaze_ms` | `video_decode_ms`, `video_tiling_ms`, 기타 processor/tokenization overhead | `video_preprocess_ms - autogaze_total_ms`로 계산합니다. | AutoGaze를 제외한 입력 준비 비용입니다. |
| `autogaze_total_ms` | `gazing_info_total_ms`, `autogaze_ms`, `autogaze_model_forward_ms` | primary formula에서 별도 항으로 표시합니다. | AutoGaze 도입 비용을 전처리-only와 분리해서 봅니다. |
| `video_preprocess_ms` | `video_preprocess_without_autogaze_ms`, `autogaze_total_ms` | backward compatibility용 inclusive field입니다. | 과거 로그와 비교할 때 쓰며, primary total에 다시 더하지 않습니다. |
| `stage_timings_ms.processor.processor_total` | `video_tiling_and_tensorize`, `autogaze_total`, 내부 decode | processor call 전체를 감싼 상위 구간입니다. | tokenization, video preprocess, tiling/tensorization, AutoGaze selection을 함께 봅니다. |
| `video_decode_ms` | `video_decode_sampling` | resize 사용 시 processor 밖에서, 미사용 시 processor 안에서 측정될 수 있습니다. | 긴 4K HLVid에서 CPU decode/seek가 병목인지 확인합니다. |
| `video_tiling_ms` | `video_tiling_and_tensorize` | `processor_total`의 하위 구간입니다. | tile/thumbnail 생성과 image tensorization 비용입니다. |
| `autogaze_ms` | `autogaze_total` | `processor_total`의 하위 구간입니다. | AutoGaze stage 전체 시간입니다. 순수 모델 forward만이 아니라 gaze-info 생성, padding, split/bookkeeping이 포함됩니다. |
| `gazing_info_total_ms` | `autogaze_total` | `autogaze_ms`와 같은 값의 명시적 alias입니다. | summary에서 “AutoGaze 모델 forward-only”와 헷갈리지 않게 stage 전체 시간을 표시합니다. |
| `autogaze_forward_ms` | `autogaze_forward_batched` | `autogaze_total`의 하위 구간입니다. | 순수 AutoGaze 모델 forward 시간입니다. “AutoGaze 모델만 돈 시간”은 이 값을 봅니다. |
| `autogaze_model_forward_ms` | `autogaze_forward_batched` | `autogaze_forward_ms`와 같은 값의 명시적 alias입니다. | 리포트에서 stage 전체와 forward-only를 나눠 읽기 쉽게 합니다. |
| `vision_encoder_ms` | `siglip_vision_ms`, `mm_projector_ms` | generate pass 내부 vision path 상위 구간입니다. | NVILA의 vision encoding 전체입니다. 순수 SigLIP만 보려면 `siglip_vision_ms`를 봅니다. |
| `siglip_vision_ms` | `siglip_vision_tower` | `vision_encoder_ms`의 하위 구간입니다. | AutoGaze가 patch/token을 줄여 vision tower forward가 빨라지는지 보는 핵심 latency입니다. |
| `mm_projector_ms` | `mm_projector.forward` | vision feature를 LLM hidden dimension으로 투영하는 구간입니다. | TokenShuffle 이후 visual token 수 감소가 projector 비용에 반영되는지 확인합니다. |
| `llm_forward_ms` | `llm.forward` 누적 | full generation pass에서 LLM forward hook을 누적합니다. | prefill과 decode 호출이 모두 포함됩니다. TTFT 분리는 `ttft_stage_timings_ms`를 함께 봅니다. |
| `ttft_ms` | `ttft_stage_timings_ms.*` | `--measure-ttft`일 때 별도 1-token generate pass입니다. | prefill context, KV cache, 첫 토큰 latency를 보는 값입니다. full `total_ms`에 더하지 않습니다. |
| `generation_decode_after_ttft_estimated_ms` | `generate_ms - ttft_ms` | legacy `decode_estimated_ms`와 같은 generation-side 추정값입니다. | 비디오 decode가 아니며, 별도 TTFT pass 기반 추정이라 closed accounting용으로 쓰지 않습니다. |
| `latency_accounting` | `additive_total_ms`, `legacy_inclusive_total_ms`, `nested_preprocess_breakdown_ms`, `nested_generate_breakdown_ms` | primary 3-part total과 legacy inclusive total, nested breakdown을 기계적으로 분리한 설명 블록입니다. | “전처리-only + AutoGaze + generate”와 “inclusive preprocess + generate”를 혼동하지 않게 합니다. |

관련 코드 위치는 stage hook 정의 [ProfilePatches:L164](../repro/nvila_runner.py#L164), latency hierarchy 공통 정의 [latency_hierarchy_summary:L286](../repro/hlvid.py#L286), latency accounting 공통 summary [latency_accounting_summary:L419](../repro/hlvid.py#L419), runner latency accounting 생성 [build_latency_accounting:L342](../repro/nvila_runner.py#L342), processor/generate 실행과 결과 필드 조립 [generate_one:L3359](../repro/nvila_runner.py#L3359), full/TTFT generation timer [timed_generate:L2196](../repro/nvila_runner.py#L2196), compact summary 생성 [build_single_summary:L686](../repro/nvila_runner.py#L686), HLVid readable summary [build_readable_summary:L209](../repro/hlvid_batch_benchmark.py#L209), Markdown accounting 렌더링 [render_latency_accounting_section:L300](../repro/markdown_report.py#L300)입니다.

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

예를 들어 720p로 resize된 128프레임 비디오라도 `max_tiles_video=8`이면 단일 720p frame 하나가 아니라 최대 8개 spatial tile로 나뉩니다. 여기서 주의할 점은 NVILA-HD weight의 `preprocessor_config.json`에는 `target_patch_size=16`이 있고, 모델 `config.json`의 `vision_config.patch_size`는 14라는 점입니다. AutoGaze Quick Start 기준으로는 실제로 붙일 vision encoder의 patch size에 `target_patch_size`를 맞춰야 하므로, runner 기본값은 sparse SigLIP gather가 정렬되도록 `56+112+196+392`, patch size 14를 명시 주입합니다. 이때 한 tile-frame당 AutoGaze/SigLIP embedding 위치는 `16+64+196+784=1060`개입니다. patch16 release metadata 경로는 호환성/ablation으로만 따로 비교해야 합니다. 실제 LLM 쪽 비교는 `llm_keep_all_visual_tokens_estimated`와 `llm_actual_visual_tokens`를 봐야 합니다.

LLM visual-token 기준:

- `token_metrics.llm_keep_all_visual_tokens_estimated`: 모든 tile/thumbnail patch를 유지했을 때 TokenShuffle 이후 예상 visual token 수
- `token_metrics.llm_actual_visual_tokens`: AutoGaze/keep-all padding strategy가 반영된 processor output의 실제 visual placeholder token 수
- `token_metrics.llm_visual_token_reduction_ratio`: keep-all 예상 LLM visual token 수를 실제 visual token 수로 나눈 값

HLVid `--mode hlvid` summary에는 `question_count`, `question_samples`, `benchmark_samples`, `latency_ms`, `memory_bytes`, `tokens`, `compute`, `readable_performance_summary`, `token_budget_summary`가 추가됩니다. `question_samples`는 질문 원문 확인용이고, `benchmark_samples`는 대상 비디오, 질문, 모델 답변, parsed 답변, 정답, 정오 여부를 같이 보여주는 읽기 쉬운 샘플 표입니다. `latency_ms`/`memory_bytes`/`tokens`/`compute`는 각 metric의 count/mean/median/min/max이고, `readable_performance_summary.key_metrics_median`에는 중요한 latency/token/memory median만 모읍니다. latency는 `total_ms`, `preprocess_without_autogaze_ms`, `preprocess_total_ms`, `autogaze_total_ms`, `vit_encoder_ms`, `llm_ms`, token은 sampled frame 수, encoder patch before/after, AutoGaze input/selected patch, LLM visual token before/after와 reduction ratio, memory는 `processor_peak`, `ttft_peak`, `llm_peak`, `overall_peak`입니다. `preprocess_total_ms`는 legacy inclusive preprocess라 AutoGaze를 포함합니다. primary additive 공식은 `preprocess_without_autogaze_ms + autogaze_total_ms + generate_ms`이고, 이 공식은 `readable_performance_summary.latency_accounting` 또는 batch gain report의 `readable_summary.latency_accounting`에서 확인합니다. 세부 decode/tile/projector/TTFT median은 `readable_performance_summary.latency_ms_detail_median` 또는 top-level `latency_ms`에서 확인합니다. summary가 너무 커지지 않도록 앞쪽 샘플만 담고, 전체 row의 질문/정답/모델 출력은 `predictions.jsonl`과 `scored_predictions.jsonl`에 남깁니다. `token_budget_summary`는 성공한 row들의 `token_metrics`에서 median/mean을 모은 것이고, `failed` row는 token metric이 없으므로 집계에서 빠집니다. `scripts/run_hlvid_folder_benchmark.py`의 최종 gain report에서도 top-level `benchmark_samples.keep_all`, `benchmark_samples.autogaze`, `benchmark_samples.correctness_comparison`, `correctness_comparison`, `autogaze.tokens`와 `gains.autogaze_token_reduction_median`에 raw/selected patch 수와 LLM visual token 수가 함께 들어갑니다. `correctness_comparison`은 `question_id`가 있으면 그것으로, 없으면 `video_path + question`으로 keep-all/AutoGaze row를 pair합니다.

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
| 입력/샘플링 | `video_input_summary.*`, `input_token_count`, `input_shapes`, `token_metrics.video_sampled_frames`, `token_metrics.thumbnail_sampled_frames` | 원본 프레임 수/해상도, 요청 frame 수, 실제 processor tensor 기준 frame 수, resize 후 입력 해상도, text+visual context 길이입니다. 설정이 의도대로 반영됐는지 확인합니다. | [build_video_input_summary:L2199](../repro/nvila_runner.py#L2199), [generate_one:L3273](../repro/nvila_runner.py#L3273), [compute_visual_token_metrics:L1950](../repro/nvila_runner.py#L1950), [build_stream_profile_token_metrics:L938](../repro/nvila_runner.py#L938) | 높음 |
| 비디오 decode | `video_decode_ms`, `stage_timings_ms.*.video_decode_sampling`, `video_input_summary.video_decode_*`, stream의 `timing_ms.video_decode_scan/seek` | CPU decode와 seek sampling 비용입니다. `single`/`hlvid`에서 `--video-resize-*`를 켠 경우 `--video-decode-strategy auto|seek|scan`이 적용되고, 실제 전략과 읽은 frame 수가 summary에 남습니다. 긴 4K HLVid에서는 병목 여부를 먼저 봅니다. | [ProfilePatches:L156](../repro/nvila_runner.py#L156), [load_sampled_video_frames:L1252](../repro/nvila_runner.py#L1252), [build_video_input_summary:L2199](../repro/nvila_runner.py#L2199), [run_stream_profile:L2874](../repro/nvila_runner.py#L2874) | 높음 |
| resize/tiling | `video_tiling_ms`, stream의 `video_frame_resize`, `spatial_tile_build` | NVILA dynamic tiling과 runner resize cost입니다. AutoGaze 효과와 무관한 전처리 overhead를 분리합니다. | [ProfilePatches:L156](../repro/nvila_runner.py#L156), [run_stream_profile:L2874](../repro/nvila_runner.py#L2874) | 높음 |
| AutoGaze | `gazing_info_total_ms`, `autogaze_ms`, `autogaze_model_forward_ms`, `autogaze_forward_ms`, stream의 `tile_autogaze_tensorize`, `tile_autogaze_forward` | AutoGaze를 넣어서 추가된 비용입니다. `gazing_info_total_ms`/`autogaze_ms`는 stage 전체, `autogaze_model_forward_ms`/`autogaze_forward_ms`는 모델 forward-only입니다. token 감소 이득이 이 비용을 이기는지 판단합니다. | [ProfilePatches:L156](../repro/nvila_runner.py#L156), [run_autogaze_on_stream_tile_sequences:L2628](../repro/nvila_runner.py#L2628) | 높음 |
| SigLIP latency | `siglip_vision_ms`, stream의 `siglip_gazed_forward`, `siglip_keep_all_forward` | vision encoder의 실제 forward 시간입니다. AutoGaze의 1차 효과가 드러나는 지점입니다. | [ProfilePatches:L156](../repro/nvila_runner.py#L156), [run_siglip_on_stream_batch:L2579](../repro/nvila_runner.py#L2579) | 높음 |
| SigLIP 계산량 | `compute_metrics.siglip_encoder.*` | attention/MLP MACs와 activation byte 추정치입니다. latency가 noisy할 때도 계산량 감소를 설명할 수 있습니다. | [build_autogaze_effect_metrics:L1812](../repro/nvila_runner.py#L1812), [build_stream_profile_compute_metrics:L994](../repro/nvila_runner.py#L994) | 높음 |
| projector | `mm_projector_ms` | SigLIP hidden state를 LLM hidden dimension으로 보내는 TokenShuffle+MLP 시간입니다. visual token 수가 줄면 같이 줄 수 있습니다. | [mm_projector.forward hook:L164](../repro/nvila_runner.py#L164), [generate_one:L3273](../repro/nvila_runner.py#L3273) | 중간 |
| LLM latency | `llm_forward_ms`, `ttft_ms`, `generation_decode_after_ttft_estimated_ms`, legacy `decode_estimated_ms` | prefill과 generation decode 비용입니다. AutoGaze의 MLLM context 감소 효과를 봅니다. generation decode estimate는 비디오 decode가 아닙니다. | [llm.forward hook:L167](../repro/nvila_runner.py#L167), [timed_generate:L2112](../repro/nvila_runner.py#L2112), [generate_one:L3273](../repro/nvila_runner.py#L3273) | 높음 |
| latency accounting | `latency_accounting.*`, batch `readable_summary.latency_accounting`, Markdown `Latency Accounting` section | `total_ms = video_preprocess_without_autogaze_ms + autogaze_total_ms + generate_ms`가 primary additive 공식이고, `video_preprocess_ms`는 legacy inclusive field라는 점을 명시합니다. | [build_latency_accounting:L342](../repro/nvila_runner.py#L342), [build_readable_summary:L209](../repro/hlvid_batch_benchmark.py#L209), [render_latency_accounting_section:L300](../repro/markdown_report.py#L300) | 높음 |
| LLM context/KV | `compute_metrics.mllm.actual_prefill_context_tokens`, `kv_cache_reduction_ratio`, `prefill_total_macs_reduction_ratio` | LLM이 실제로 받은 context 길이와 KV cache/attention 계산 감소 추정치입니다. | [build_autogaze_effect_metrics:L1812](../repro/nvila_runner.py#L1812) | 높음 |
| CUDA memory | `processor_peak_memory_bytes`, `ttft_peak_memory_bytes`, `llm_peak_memory_bytes`, `peak_memory_bytes` | CUDA peak allocation입니다. OOM 리스크와 배치/프레임 설정 선택에 중요합니다. | [timed_generate:L2112](../repro/nvila_runner.py#L2112), [generate_one:L3273](../repro/nvila_runner.py#L3273), [run_stream_profile:L2874](../repro/nvila_runner.py#L2874) | 높음 |
| token/patch | `token_metrics.encoder_*`, `token_metrics.llm_*` | encoder patch 감소와 LLM visual token 감소를 분리해서 보여줍니다. 리더 설득용 핵심 근거입니다. | [compute_visual_token_metrics:L1950](../repro/nvila_runner.py#L1950), [build_stream_profile_token_metrics:L938](../repro/nvila_runner.py#L938) | 높음 |
| 정확도 | HLVid `accuracy_scored`, `accuracy_total`, `failed`, `skipped`, `parse_failed` | AutoGaze 속도 이득이 성능 손실을 만들었는지 확인합니다. | [score_predictions:L333](../repro/hlvid.py#L333), [summarize_run:L584](../repro/hlvid_batch_benchmark.py#L584), [build_gain_report:L595](../repro/hlvid_batch_benchmark.py#L595) | 높음 |

아직 직접 측정하지 않는 항목도 있습니다. `compute_metrics`의 MACs는 실제 GPU hardware counter가 아니라 config 기반 추정치입니다. CUDA의 정확한 SM utilization, DRAM bandwidth, per-layer peak activation은 Nsight/PyTorch profiler가 필요합니다. MPS는 CUDA처럼 reliable한 peak allocation을 제공하지 않으므로 memory 평가는 CUDA에서 최종 확인하세요.

실제 동작 순서대로 보면 아래 필드를 따라가면 됩니다.

1. Dataset row 로드: HLVid `manifest`, `question`, `answer`, `video_path`
2. 비디오 resolve: `video_resolved`, `video_input_info`, `video_resize`
3. frame sampling/decode: `video_decode_ms`, `video_input_summary.video_decode_strategy`, `video_input_summary.video_decode_frames_read`, stream의 `decode_strategy`, `decode_frames_read`
4. PIL 변환/resize: stream의 `video_frame_to_pil`, `video_frame_resize`
5. spatial tiling/thumbnail: `video_tiling_ms`, stream의 `spatial_tile_build`, `thumbnail_resize`
6. AutoGaze 입력 tensorization: stream의 `tile_autogaze_tensorize`
7. AutoGaze selection 전체: `gazing_info_total_ms` 또는 legacy `autogaze_ms`
8. AutoGaze model forward-only: `autogaze_model_forward_ms` 또는 legacy `autogaze_forward_ms`, stream의 `tile_autogaze_forward`
9. SigLIP vision tower: `siglip_vision_ms` 또는 stream의 `siglip_gazed_forward/keep_all_forward`
10. visual token 정렬/TokenShuffle/projector: `mm_projector_ms`, `token_metrics.llm_actual_visual_tokens`
11. LLM prefill/TTFT: `ttft_ms`, `compute_metrics.mllm.actual_prefill_context_tokens`, `actual_kv_cache_bytes_after_prefill_estimated`
12. LLM decode: `llm_forward_ms`, `generation_decode_after_ttft_estimated_ms`, legacy `decode_estimated_ms`, `generated_tokens`
13. latency accounting 확인: `latency_accounting.additive_total_ms`, batch `readable_summary.latency_accounting`
14. scoring/report: HLVid `accuracy_scored`, batch report의 `gains.*`

AutoGaze 적용 전후 실제 gain은 반드시 같은 설정의 `keep-all`과 `autogaze`를 나란히 비교합니다.

| 확인 질문 | 볼 필드 |
| --- | --- |
| 단일 파일 summary에서 먼저 볼 핵심은? | `summary.key_autogaze_effect.encoder_patch_*`, `summary.key_autogaze_effect.llm_visual_*`, `summary.key_autogaze_effect.siglip_total_macs_reduction_ratio`, `summary.key_autogaze_effect.mllm_*`, `summary.key_autogaze_effect.total_ms_median` |
| 정확도가 유지됐나? | `autogaze.accuracy.accuracy_scored`, `keep_all.accuracy.accuracy_scored`, `gains.accuracy_scored_delta`, `correctness_comparison.counts.*` |
| 전체 latency가 줄었나? | `gains.latency_speedup_median.total_ms` |
| SigLIP가 빨라졌나? | `gains.latency_speedup_median.siglip_vision_ms`, stream의 `siglip_keep_all_forward / siglip_gazed_forward` |
| LLM prefill 부담이 줄었나? | `compute_metrics.mllm.prefill_context_reduction_ratio`, `kv_cache_reduction_ratio` |
| visual token이 줄었나? | `token_metrics.llm_visual_token_reduction_ratio`, batch report의 `gains.autogaze_token_reduction_median` |
| 메모리 이득이 있나? | `gains.memory_reduction_ratio_median.llm_peak_memory_bytes`, `ttft_peak_memory_bytes` |
| AutoGaze overhead가 이득을 잡아먹나? | `gazing_info_total_ms`/`autogaze_ms`, `autogaze_model_forward_ms`/`autogaze_forward_ms`와 `siglip_vision_ms/llm_forward_ms` 감소량 비교 |

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

속도 관점에서는 두 JSON 파일의 `repeat_summary.total_ms`, `repeat_summary.generate_ms`, `repeat_summary.video_preprocess_without_autogaze_ms`, `repeat_summary.autogaze_total_ms`, `repeat_summary.video_decode_ms`, `repeat_summary.video_tiling_ms`, `repeat_summary.gazing_info_total_ms`, `repeat_summary.autogaze_model_forward_ms`, `repeat_summary.siglip_vision_ms`, `repeat_summary.vision_encoder_ms`, `repeat_summary.llm_forward_ms` median을 비교합니다. 단일 실행만 했다면 같은 필드를 `result.*`에서 보면 됩니다. `latency_accounting.additive_total_ms`는 primary 3-part total 재계산용으로 보고, `latency_accounting.legacy_inclusive_total_ms`는 과거 `video_preprocess_ms + generate_ms` 관점으로 읽으세요. 나머지 decode/tile/AutoGaze/SigLIP/LLM field는 nested breakdown입니다. 토큰 관점에서는 tile, thumbnail, total patch budget을 같이 비교하세요. 핵심 필드는 `token_metrics.encoder_raw_tile_patch_tokens`, `token_metrics.encoder_autogaze_selected_tile_patch_tokens`, `token_metrics.encoder_raw_thumbnail_patch_tokens`, `token_metrics.encoder_autogaze_selected_thumbnail_patch_tokens`, `token_metrics.encoder_raw_patch_tokens`, `token_metrics.encoder_autogaze_selected_patch_tokens`, `token_metrics.encoder_token_reduction_ratio`, `token_metrics.llm_keep_all_visual_tokens_estimated`, `token_metrics.llm_actual_visual_tokens`, `token_metrics.llm_visual_token_reduction_ratio`입니다.

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

`nvila_runner`는 NVILA-HD와 논문 baseline 재현을 안정적으로 유지하는 러너입니다. 이 러너도 컴포넌트 단위 식별자를 받지만, 범위는 NVILA-HD / NVILA-8B-Video paper baseline에 한정합니다. preset은 검증된 조합을 빠르게 고르는 용도이고, 컴포넌트 CLI는 각 부분의 adapter/name/path를 명시해서 report에 남기는 용도입니다.

- token selector: `--token-selector-adapter auto|none|keep-all|autogaze`, `--token-selector-name`, `--token-selector-path`
- vision encoder: `--vision-encoder-adapter auto|nvila-hd-siglip|nvila-video-vision`, `--vision-encoder-name`, `--vision-encoder-path`
- MLLM: `--mllm-adapter auto|nvila-hd|nvila-video`, `--mllm-name`, `--mllm-path`
- model family: `--model-family auto|nvila-hd-video-autogaze|nvila-video-baseline`

`--model-path`/`--nvila-model`은 기존 호환용으로 유지됩니다. `--mllm-path`를 주면 실제 model load path로 승격되고, 결과 JSON에는 `run_identity.components.token_selector`, `run_identity.components.vision_encoder`, `run_identity.components.mllm`로 기록됩니다. `--pipeline-preset`은 현재 `--paper-preset`의 alias라서 같은 preset 기본값을 적용합니다.

중요한 구분은 아래와 같습니다. `nvila-video-baseline`은 AutoGaze 논문 table의 `NVILA-8B-Video` baseline 재현 후보입니다. AutoGaze는 `not_applicable`이고, processor kwargs에도 AutoGaze 관련 필드를 넣지 않습니다. 반대로 튜닝되지 않은 NVILA-Video, LongVILA, Qwen2-VL류에 AutoGaze on/off를 붙이는 실험은 `nvila_runner`에 넣지 않고 새 확장 러너인 `repro.flexible_runner`에서 다룹니다.

예를 들어 로컬 `NVILA-8B-Video` baseline을 컴포넌트 형태로 명시하면:

```bash
.venv/bin/python -m repro.nvila_runner \
  --mode hlvid \
  --device cuda \
  --pipeline-preset autogaze-hlvid-baseline \
  --mllm-path weight/NVILA-8B-Video \
  --mllm-adapter nvila-video \
  --mllm-name local-nvila-8b-video \
  --token-selector-adapter none \
  --token-selector-name not_applicable \
  --vision-encoder-adapter nvila-video-vision \
  --vision-encoder-name nvila-8b-video-vision \
  --manifest /path/to/HLVid/data/test-00000-of-00001.parquet \
  --hlvid-video-root /path/to/HLVid/videos \
  --measure-ttft \
  --continue-on-error
```

HD AutoGaze 쪽을 같은 형태로 명시하면:

```bash
.venv/bin/python -m repro.nvila_runner \
  --mode single \
  --device cuda \
  --mllm-path /path/to/local/nvila-hd-video \
  --mllm-adapter nvila-hd \
  --mllm-name local-nvila-hd-video \
  --token-selector-adapter autogaze \
  --token-selector-path /path/to/local/autogaze-checkpoint \
  --token-selector-name local-autogaze \
  --vision-encoder-adapter nvila-hd-siglip \
  --vision-encoder-name nvila-hd-siglip \
  --video inputs/hlvid_example/clip_av_video_5_001.mp4 \
  --num-video-frames 128 \
  --num-video-frames-thumbnail 64 \
  --max-tiles-video 48 \
  --measure-ttft \
  --output-json outputs/autogaze_repro/cuda_nvila_single_component_style.json
```

paper baseline용 `NVILA-8B-Video`를 로컬 weight에서 직접 지정하려면 preset 없이 아래처럼 씁니다. 이 경우 `--model-family nvila-video-baseline`을 함께 주면 runner가 baseline path로 인식해서 AutoGaze processor kwargs를 넣지 않습니다.

```bash
.venv/bin/python -m repro.nvila_runner \
  --mode hlvid \
  --device cuda \
  --model-path weight/NVILA-8B-Video \
  --model-family nvila-video-baseline \
  --manifest /path/to/HLVid/data/test-00000-of-00001.parquet \
  --hlvid-video-root /path/to/HLVid/videos \
  --num-video-frames 256 \
  --num-video-frames-thumbnail 0 \
  --max-tiles-video 1 \
  --video-resize-longest-edge 448 \
  --gazing-mode keep-all \
  --measure-ttft \
  --warmup-runs 1 \
  --predictions outputs/autogaze_repro/hlvid_local_nvila_8b_video_predictions.jsonl \
  --summary outputs/autogaze_repro/hlvid_local_nvila_8b_video_summary.json \
  --scored-predictions outputs/autogaze_repro/hlvid_local_nvila_8b_video_scored.jsonl \
  --continue-on-error
```

같은 `NVILA-8B-Video` weight를 paper baseline이 아니라 AutoGaze on/off 실험 대상으로 쓰려면 `repro.flexible_runner`를 사용합니다. `NVILA-8B-Video` root config의 `model_type=llava_llama`는 일반 Transformers `AutoModel`만으로는 현재 로드되지 않으므로, native/off 실행은 공식 VILA 코드의 `vila-infer` CLI를 감싸서 수행합니다. 공식 VILA README도 video inference 예시를 `vila-infer --model-path ... --conv-mode auto --text ... --media ...` 형태로 안내합니다.

첫 단계는 모델을 로드하지 않는 `inspect` 모드로 어떤 plugin 조합을 실행할지 JSON으로 고정하는 것입니다. 그 다음 `--mode single`을 쓰면 `nvila-video` native/off는 `vila-infer`를 통해 실제 실행을 시도하고, 아직 실제 generate를 하지 않는 planned 계열 adapter는 `probe_required` JSON을 만듭니다. 이 probe는 모델별 processor 입력, vision feature shape, MLLM visual token packing boundary를 CUDA 머신에서 어디서 찍어야 하는지 명확히 남기는 용도입니다.

```bash
.venv/bin/python -m repro.flexible_runner \
  --mode inspect \
  --model-path weight/NVILA-8B-Video \
  --model-family nvila-video-plugin \
  --token-selector-adapter keep-all \
  --vision-encoder-adapter nvila-video-vision \
  --mllm-adapter nvila-video \
  --autogaze-integration-level none \
  --num-video-frames 128 \
  --output-json outputs/autogaze_repro/flexible_nvila_video_plugin_off_inspect.json
```

CUDA 머신에 VILA 환경이 있고 `vila-infer`가 PATH에 잡혀 있으면 native/off single 실행은 아래처럼 갑니다. PATH가 다르면 `--external-mllm-command /path/to/vila-infer`로 지정합니다.

```bash
.venv/bin/python -m repro.flexible_runner \
  --mode single \
  --model-path weight/NVILA-8B-Video \
  --model-family nvila-video-plugin \
  --token-selector-adapter keep-all \
  --vision-encoder-adapter nvila-video-vision \
  --mllm-adapter nvila-video \
  --autogaze-integration-level none \
  --external-mllm-command vila-infer \
  --video /path/to/video.mp4 \
  --num-video-frames 256 \
  --max-tiles-video 8 \
  --output-json outputs/autogaze_repro/flexible_nvila_video_plugin_off_single.json
```

이 adapter는 공식 CLI가 지원하는 `--num_video_frames`, `--video_max_tiles`를 넘기고, CLI stdout의 마지막 non-empty line을 `generation.text`로 기록합니다. `max_new_tokens`는 현재 공식 `vila-infer` 인자에 직접 노출되지 않으므로 `external_cli.max_new_tokens_supported=false`로 기록합니다.

AutoGaze on 실험은 아래처럼 token selector checkpoint를 별도로 지정합니다. `planned_plugin`이나 `post_encoder_token_prune`으로 기록하며, 실제 pre-encoder sparse gather 또는 post-encoder token prune을 해당 모델 내부 feature packing에 삽입하는 구현은 다음 단계입니다.

```bash
.venv/bin/python -m repro.flexible_runner \
  --mode inspect \
  --model-path weight/NVILA-8B-Video \
  --model-family nvila-video-plugin \
  --token-selector-adapter autogaze \
  --token-selector-path weight/AutoGaze \
  --vision-encoder-adapter nvila-video-vision \
  --mllm-adapter nvila-video \
  --autogaze-integration-level planned_plugin \
  --num-video-frames 128 \
  --output-json outputs/autogaze_repro/flexible_nvila_video_plugin_autogaze_requested_inspect.json
```

planned adapter의 `single` probe 결과를 만들려면 아래처럼 실행합니다. 이 명령은 모델 weight를 올려서 generate하지 않고, `mllm_runtime.feature_packing_probe`와 `generation.metrics.feature_packing_probe`에 다음 구현에서 확인해야 할 지점을 남깁니다.

```bash
.venv/bin/python -m repro.flexible_runner \
  --mode single \
  --model-path weight/NVILA-8B-Video \
  --model-family nvila-video-plugin \
  --token-selector-adapter autogaze \
  --token-selector-path weight/AutoGaze \
  --vision-encoder-adapter nvila-video-vision \
  --mllm-adapter nvila-video \
  --autogaze-integration-level post_encoder_token_prune \
  --video /path/to/video.mp4 \
  --num-video-frames 128 \
  --output-json outputs/autogaze_repro/flexible_nvila_video_plugin_probe.json
```

결과 JSON에서 우선 볼 부분은 아래입니다.

```text
implementation_status: probe_required
mllm_runtime.feature_packing_probe.required_inputs
mllm_runtime.feature_packing_probe.post_encoder_hook
generation.metrics.feature_packing_probe.token_accounting_targets
```

LongVILA, Qwen2-VL, Qwen3-VL도 같은 확장 러너 identity 축을 사용합니다. Qwen3-VL은 공식 Transformers 경로에서 `pixel_values_videos`, `video_grid_thw`, `mm_token_type_ids`를 사용하고 `get_video_features` entrypoint가 있으므로, 우선은 vision encoder를 그대로 둔 `post_encoder_token_prune` 방식으로 MLLM context 감소를 확인하는 순서가 맞습니다.

LongVILA native/off도 VILA 계열이므로 `vila-infer` CLI adapter로 먼저 실행합니다. AutoGaze on을 요청하면 아직 실제 generate를 하지 않고 `probe_required`로 남깁니다.

```bash
.venv/bin/python -m repro.flexible_runner \
  --mode single \
  --model-path weight/LongVILA \
  --model-family longvila \
  --token-selector-adapter keep-all \
  --vision-encoder-adapter longvila-siglip \
  --mllm-adapter longvila \
  --autogaze-integration-level none \
  --external-mllm-command vila-infer \
  --video /path/to/video.mp4 \
  --num-video-frames 256 \
  --max-tiles-video 8 \
  --output-json outputs/autogaze_repro/flexible_longvila_plugin_off_single.json
```

```bash
.venv/bin/python -m repro.flexible_runner \
  --mode inspect \
  --model-path weight/longvila \
  --model-family longvila \
  --token-selector-adapter autogaze \
  --token-selector-path weight/AutoGaze \
  --vision-encoder-adapter longvila-siglip \
  --mllm-adapter longvila \
  --autogaze-integration-level planned_plugin \
  --num-video-frames 128 \
  --output-json outputs/autogaze_repro/flexible_longvila_autogaze_requested_inspect.json
```

Qwen3-VL post-prune 후보는 아래처럼 inspect합니다.

```bash
.venv/bin/python -m repro.flexible_runner \
  --mode inspect \
  --model-path weight/Qwen3-VL-8B-Instruct \
  --model-family qwen3-vl \
  --token-selector-adapter autogaze \
  --token-selector-path weight/AutoGaze \
  --vision-encoder-adapter qwen3-vl-vision \
  --mllm-adapter qwen3-vl \
  --autogaze-integration-level post_encoder_token_prune \
  --gazing-ratio 0.1 \
  --num-video-frames 128 \
  --output-json outputs/autogaze_repro/flexible_qwen3_vl_autogaze_post_prune_poc.json
```

이 명령은 현재 **AutoGaze + Qwen post-encoder attachment PoC**입니다. 모델을 로드하지 않고 `SparseSelectionPlan`, Qwen `get_video_features` 이후 hook 위치, AutoGaze 적용 전후 visual token estimate를 JSON에 남깁니다. 아직 Qwen visual embedding을 실제로 잘라서 scored generation까지 수행하지는 않습니다.

PixelPrune처럼 Qwen3-VL model load 전에 pre-ViT pruning hook을 적용하는 경로는 실제 실행 경로로 열어두었습니다. CUDA 머신에 `pixelprune` 패키지가 설치되어 있으면 runner가 모델 로드 전에 hook을 적용하고, 그 다음 Qwen native generation을 그대로 수행합니다.

```bash
.venv/bin/python -m repro.flexible_runner \
  --mode single \
  --model-path weight/Qwen3-VL-8B-Instruct \
  --model-family qwen3-vl \
  --token-selector-adapter keep-all \
  --vision-encoder-adapter qwen3-vl-vision \
  --mllm-adapter qwen3-vl \
  --autogaze-integration-level pre_encoder_sparse \
  --pre-encoder-prune-adapter pixelprune \
  --pixelprune-threshold 0.0 \
  --video /path/to/video.mp4 \
  --output-json outputs/autogaze_repro/flexible_qwen3_vl_pixelprune_pre_vit.json
```

중요: PixelPrune이 요청되었는데 패키지 import/hook 적용에 실패하면 runner는 dense Qwen 실행으로 조용히 넘어가지 않고 `failed_missing_dependency`로 중단합니다. 이 경로는 HLVid plugin benchmark의 `qwen3-vl-pixelprune-pre-vit` 모드에서도 사용할 수 있습니다.

다른 MLLM 후보 현황은 아래처럼 잡았습니다.

| family | 1차 목표 | pre 쪽 가능성 | 현재 runner 상태 |
|---|---|---|---|
| `qwen2-vl` | post-encoder token prune | grid probe 필요 | single dry-run + inspect config |
| `qwen2.5-vl` | post-encoder token prune | grid probe 필요 | single dry-run + inspect config |
| `qwen3-vl` | AutoGaze post-encoder attachment PoC + PixelPrune pre-ViT execution | PixelPrune reference 있음 | AutoGaze PoC + PixelPrune single 실행 |
| `qwen3-vl-moe` | post-encoder token prune + PixelPrune pre-ViT reference | PixelPrune reference 있음 | inspect |
| `llava-onevision` | post-pool token prune | hard, 196 tokens/frame pooling 이후가 현실적 | single dry-run + inspect config |
| `nvila-video-plugin` | post-encoder token prune | patch/position alignment probe 필요 | single probe + inspect config |
| `internvl3` | post-encoder token prune | dynamic tiling probe 필요 | single probe + inspect config |
| `longvila` | post-encoder token prune | VILA feature packing probe 필요 | native/off single + AutoGaze-on probe |

## 일반화 전 4개 우선 검증 트랙

범용 selector/plugin 일반화 전에 아래 네 트랙을 먼저 닫습니다. 설정 파일은 `configs/repro/autogaze_priority_validation.yaml`입니다.

```text
1. NVILA-HD + AutoGaze native profiling
   runner: repro.hlvid_batch_benchmark
   modes : hd_keep_all_optional, hd_autogaze
   goal  : latency / memory / token / HLVid score 완전 검증

2. NVILA-8B-Video baseline + AutoGaze on/off probe
   runner: repro.plugin_hlvid_benchmark
   modes : nvila-video-off, nvila-video-autogaze-probe
   goal  : paper baseline과 plugin on/off 실험을 분리하고 feature packing boundary 확인

3. LongVILA + AutoGaze PoC
   runner: repro.plugin_hlvid_benchmark
   modes : longvila-off, longvila-autogaze-probe
   goal  : long video feature packing boundary 확인

4. Qwen3-VL + AutoGaze PoC
   runner: repro.plugin_hlvid_benchmark
   modes : qwen3-vl-off, qwen3-vl-autogaze-poc
   goal  : SparseSelectionPlan과 post-encoder token estimate 생성, 이후 실제 feature prune/generate로 확장
```

현재 local smoke 기준 상태:

| track | local smoke status | 의미 |
|---|---|---|
| NVILA-HD + AutoGaze | 기존 `nvila_runner`/`hlvid_batch_benchmark` 경로 사용 | CUDA에서 실제 모델/HLVid로 재측정 필요 |
| NVILA-8B-Video + AutoGaze | `probe_required` | VILA feature packing probe가 다음 구현 |
| LongVILA + AutoGaze | `probe_required` | LongVILA feature packing probe가 다음 구현 |
| Qwen3-VL + AutoGaze | `poc_ready` | SparseSelectionPlan/token estimate 있음, 실제 visual embedding prune은 다음 구현 |

`plugin_hlvid_benchmark` report는 각 mode별 `status_counts`와 `next_action`을 같이 출력합니다. 로컬 smoke 기준 다음 액션은 아래처럼 해석합니다.

| mode | expected status | next_action | 의미 |
|---|---|---|---|
| `nvila-video-autogaze-probe` | `probe_required` | `run_vila_feature_packing_probe` | NVILA-Video remote code 내부에서 processor, vision output, projector output, visual token insertion boundary를 찍어야 함 |
| `longvila-autogaze-probe` | `probe_required` | `run_vila_feature_packing_probe` | LongVILA도 VILA 계열 packing boundary probe가 먼저 필요 |
| `qwen3-vl-autogaze-poc` | `poc_ready` | `implement_qwen_visual_feature_prune_generate` | Qwen `get_video_features` 이후 feature를 실제로 줄이고 LLM context에 넣는 구현이 다음 단계 |
| `qwen3-vl-pixelprune-pre-vit` | `failed_missing_dependency` when PixelPrune missing | `install_pixelprune_and_rerun` | PixelPrune이 없으면 dense fallback 없이 실패시키는 것이 정상 |

다운로드는 다음 형태로 받으면 됩니다. 현재 로컬에는 `weight/NVILA-8B-Video` 경로로 받아두었습니다.

```bash
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Efficient-Large-Model/NVILA-8B-Video', repo_type='model', local_dir='weight/NVILA-8B-Video')"
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

### H100 preflight sweep

H100 80GB에서 바로 full run을 던지기 전에 `h100-preflight-sweep`으로 config risk를 먼저 볼 수 있습니다. 기본 budget은 실제 가용 여유를 남기기 위해 `70 GiB`이고, risk band는 `green <55GiB`, `yellow 55-70GiB`, `red >=70GiB`, context 초과는 `context_red`입니다.

paper baseline 후보:

```bash
.venv/bin/python -m repro.nvila_runner \
  --mode h100-preflight-sweep \
  --paper-preset autogaze-hlvid-baseline \
  --video /path/to/HLVid/videos/clip_av_video_5_001.mp4 \
  --h100-budget-gib 70 \
  --h100-sweep-json outputs/autogaze_repro/h100_paper_baseline_sweep.json
```

HD AutoGaze 확장 후보:

```bash
.venv/bin/python -m repro.nvila_runner \
  --mode h100-preflight-sweep \
  --paper-preset autogaze-hlvid-hd \
  --video /path/to/HLVid/videos/clip_av_video_5_001.mp4 \
  --h100-budget-gib 80 \
  --h100-reduction-ratios 128,200,256,300,400 \
  --stream-chunk-frames 16 \
  --max-batch-size-autogaze 16 \
  --max-batch-size-siglip 32 \
  --autogaze-residency-policy resident \
  --autogaze-model-resident-gib 0 \
  --h100-sweep-json outputs/autogaze_repro/h100_hd_autogaze_sweep.json
```

sweep grid는 frames `[1024,512,256,128,64,32]`, thumbnail frames `[512,256,128,64,32,16]`, max tiles `[48,32,16,8,4,1]`, resize shortest edge `[None,1080,720,512,448,384]`, token reduction ratio는 CLI의 `--h100-reduction-ratios`입니다. 콘솔에는 전체 JSON 대신 `summary`만 출력됩니다. 먼저 `requested_config_table`을 보세요. 이 표가 현재 CLI/preset 설정에 대한 해상도, 프레임 수, tile 수, LLM visual/context token, VRAM, 병목 stage를 보여줍니다. `llm_context_fits`, `llm_context_margin_tokens`, `llm_context_utilization_percent`, `min_tile_reduction_ratio_for_context`, `max_tile_sequence_tokens_for_context`는 고정된 LLM context limit 안에 들어가는지 판단하는 1차 gate입니다. 그 다음 `sweep_decision_table`에서 가능한 대안 조합을 봅니다. 전체 상세 row는 `--h100-sweep-json` 파일의 `sweep.rows`에 저장됩니다. 이 estimator는 scheduling용 보수 추정치이고, 실제 CUDA run에서는 `processor_peak_memory_bytes`, `ttft_peak_memory_bytes`, `llm_peak_memory_bytes`, `peak_memory_bytes`를 최종 근거로 삼아야 합니다.

주의: public `NVILA-8B-HD-Video`의 full `single`/`hlvid` generate path는 thumbnail frame을 최소 1개 필요로 합니다. `--num-video-frames-thumbnail 0`은 `stream-profile`이나 AutoGaze-only sweep에서는 가능하지만, HD full generate에서는 processor/model 내부 가정 때문에 실패할 수 있습니다. paper baseline `NVILA-8B-Video`는 별도 family라서 thumbnail 0을 계속 사용할 수 있습니다.

OOM preflight는 AutoGaze tensor residency를 두 방식으로 구분합니다. `--stream-chunk-frames 0` 또는 미지정 API 호출은 sampled tile tensor를 전체 비디오 단위로 잡는 현재 public processor 위험을 보여주고, `--stream-chunk-frames 16 --max-batch-size-autogaze 16`은 decode/tile/AutoGaze가 temporal chunk 단위로 흘러간다는 가정의 working set을 보여줍니다. 리포트에서는 `memory.autogaze_working_mode`, `autogaze_tile_tensor_full_video_bytes_estimated`, `autogaze_tensor_residency_bytes_estimated`, `autogaze_forward_batch_tensor_bytes_estimated`를 비교하세요. 단, LLM prefill은 여전히 누적된 visual token sequence를 한 번에 받는 것으로 추정하므로 `tokens.actual_context_tokens_estimated`와 `risk.context_red`를 같이 봐야 합니다.

AutoGaze 모델 weight가 GPU에 계속 남는 경우는 `--autogaze-residency-policy resident --autogaze-model-resident-gib <GiB>`로 별도 반영합니다. 이 값은 H100에서 실제 AutoGaze 처리 직후 `torch.cuda.max_memory_allocated`나 `nvidia-smi`로 보정하세요. AutoGaze를 명시적으로 내리고 generate를 돌리는 실험을 할 때만 `--autogaze-residency-policy unload-before-generate`로 비교합니다.

AutoGaze가 streaming된다는 가정에서는 `bottlenecks` 섹션을 먼저 보세요. `bottlenecks.stage_memory_gib_estimated.autogaze`는 AutoGaze chunk tensor residency, `vision_encoder`는 SigLIP dense attention/MLP batch peak 추정, `mllm_prefill`은 LLM prefill attention score와 KV cache 추정입니다. `vision_encoder.actual.tile_sequence_tokens`, `vision_encoder.actual.max_sequence_tokens_per_batch`, `mllm.actual.context_tokens`, `mllm.actual.kv_cache_bytes_after_prefill_estimated`를 같이 보면 병목이 ViT인지 LLM인지 분리해서 볼 수 있습니다. synthetic `--h100-reduction-ratios`는 tile patch slot 감소율로 해석하고, thumbnail patch는 keep-all로 둔 뒤 token shuffle 이후 LLM visual token을 다시 계산합니다.

HLVid 폴더 전체의 mp4 metadata를 기준으로 가장 보수적인 sweep을 만들려면 batch wrapper의 preflight 모드를 씁니다. 이 모드는 mp4를 전부 디코드하지 않고 stream metadata만 읽어서 가장 큰 해상도와 가장 긴 frame count를 기준으로 paper baseline / HD AutoGaze 추천 config를 나눠 냅니다.

```bash
.venv/bin/python scripts/run_hlvid_folder_benchmark.py \
  --dataset-dir /path/to/HLVid \
  --video-root /path/to/extracted/videos \
  --h100-preflight \
  --h100-budget-gib 70 \
  --h100-reduction-ratios 1,2,3,4 \
  --h100-preflight-output outputs/autogaze_repro/hlvid_h100_preflight_report.json
```

report에서는 `dataset_video_summary`, `recommendations.paper_baseline_reproduction_configs`, `recommendations.hd_autogaze_scaling_configs`, `sweeps.*.risk_band_counts`를 먼저 봅니다. 로컬에 일부 mp4만 있는 상태에서 의도적으로 가능한 파일만 보고 싶으면 `--allow-missing-videos`를 붙이세요.

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
| stage timer 누적 | [StageProfiler:L115](../repro/nvila_runner.py#L115) |
| decode/resize/tiling stream loop | [run_stream_profile:L2874](../repro/nvila_runner.py#L2874) |
| AutoGaze tensorize/forward | [run_autogaze_on_stream_tile_sequences:L2628](../repro/nvila_runner.py#L2628) |
| gazed/keep-all SigLIP forward | [run_siglip_on_stream_batch:L2579](../repro/nvila_runner.py#L2579) |
| stream token/patch count | [build_stream_profile_token_metrics:L938](../repro/nvila_runner.py#L938) |
| stream SigLIP/MLLM compute estimate | [build_stream_profile_compute_metrics:L994](../repro/nvila_runner.py#L994) |
| full NVILA hook 기반 timing | [ProfilePatches:L156](../repro/nvila_runner.py#L156), [forward hooks:L164-L167](../repro/nvila_runner.py#L164) |

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

논문 table의 baseline은 `NVILA-8B-HD-Video --gazing-mode keep-all`이 아니라 별도 checkpoint family인 `NVILA-8B-Video`로 봅니다. 따라서 report에서는 세 가지를 분리합니다.

- `paper_baseline_nvila_8b_video`: `Efficient-Large-Model/NVILA-8B-Video`, 256 frames, max resolution target 448, AutoGaze not applicable, HLVid reference `42.5`
- `hd_autogaze`: `nvidia/NVILA-8B-HD-Video`, 1024 frames, max resolution target 3584, AutoGaze enabled, HLVid reference `52.6`
- `hd_keep_all_optional`: NVILA-HD에서 AutoGaze selection만 끈 ablation입니다. OOM/비교용으로 유용하지만 paper baseline이라고 부르지 않습니다.

runner 단일 모드/HLVid 모드에서는 `--paper-preset`을 사용할 수 있습니다. preset은 기본값을 채우지만 CLI에서 직접 지정한 값은 유지합니다.

```bash
.venv/bin/python -m repro.nvila_runner \
  --mode hlvid \
  --paper-preset autogaze-hlvid-baseline \
  --manifest /path/to/HLVid/data/test-00000-of-00001.parquet \
  --hlvid-video-root /path/to/HLVid/videos \
  --device cuda \
  --measure-ttft \
  --warmup-runs 1 \
  --predictions outputs/autogaze_repro/hlvid_paper_baseline_predictions.jsonl \
  --summary outputs/autogaze_repro/hlvid_paper_baseline_summary.json \
  --scored-predictions outputs/autogaze_repro/hlvid_paper_baseline_scored.jsonl \
  --continue-on-error
```

HD AutoGaze paper target은 다음처럼 실행합니다.

```bash
.venv/bin/python -m repro.nvila_runner \
  --mode hlvid \
  --paper-preset autogaze-hlvid-hd \
  --manifest /path/to/HLVid/data/test-00000-of-00001.parquet \
  --hlvid-video-root /path/to/HLVid/videos \
  --device cuda \
  --measure-ttft \
  --warmup-runs 1 \
  --predictions outputs/autogaze_repro/hlvid_paper_hd_autogaze_predictions.jsonl \
  --summary outputs/autogaze_repro/hlvid_paper_hd_autogaze_summary.json \
  --scored-predictions outputs/autogaze_repro/hlvid_paper_hd_autogaze_scored.jsonl \
  --continue-on-error
```

각 row와 summary에는 `run_identity.model_family`, `run_identity.paper_preset`, `run_identity.paper_reference_score`, `run_identity.is_paper_baseline_candidate`, `run_identity.autogaze_applicability`가 남습니다. 추가로 `run_identity.components` 아래에 token selector / vision encoder / MLLM의 adapter, name, path가 각각 남습니다. baseline preset에서는 token selector가 `adapter=none`, `applicability=not_applicable`로 기록되고, NVILA processor에 AutoGaze-specific kwargs를 넣지 않습니다. AutoGaze 관련 reduction metric은 `not_applicable`로 해석하세요.

기존 fixed total-frame sampling setup용 preset은 `configs/repro/hlvid_like_nvila_1024.yaml`입니다. 이것은 HD AutoGaze 재현/ablation용이며, 위의 `NVILA-8B-Video` paper baseline과는 다릅니다.

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

`accuracy_scored`를 NVILA-8B-HD-Video의 project-page HLVid target인 `52.6`과 비교하세요. skipped, failed, parse-failed sample은 별도로 보고해야 합니다. AutoGaze-vs-keep-all claim을 만들 때는 accuracy뿐 아니라 median 또는 mean 기준 `total_ms`, `generate_ms`, `video_decode_ms`, `video_tiling_ms`, `gazing_info_total_ms`, `autogaze_model_forward_ms`, `vision_encoder_ms`, `llm_forward_ms`, `token_metrics.encoder_token_reduction_ratio`, `token_metrics.llm_visual_token_reduction_ratio`를 함께 비교하세요. 논문 대응 setup은 target GPU가 허용하는 범위에서 NVILA-8B-HD-Video, 최대 1024프레임, 최대 해상도 3584를 기준으로 합니다.

### HLVid 폴더 기반 일괄 benchmark

데이터셋을 로컬 폴더로 받은 경우에는 `scripts/run_hlvid_folder_benchmark.py`를 쓰면 됩니다. 이 스크립트는 폴더에서 manifest를 찾고, 기본 모드에서는 `keep-all`과 `autogaze`를 같은 설정으로 각각 실행한 뒤 accuracy/속도/메모리/token/compute gain report를 만듭니다. `--paper-baseline --paper-hd-autogaze --paper-comparison-report`를 쓰면 논문 baseline 비교 모드가 켜지고, `paper_baseline_nvila_8b_video`와 `hd_autogaze`를 별도 column으로 비교합니다. `--paper-hd-keep-all-optional`은 OOM/ablation 확인용입니다. `--video-resize-*`를 켠 benchmark에서는 기본적으로 `--video-decode-strategy auto`가 하위 `repro.nvila_runner`에 전달되어 keyframe seek sampling을 먼저 사용합니다.

AutoGaze policy를 명시하려면 batch wrapper에도 `--gazing-ratio-tile`과 `--task-loss-requirement-tile`을 같이 줍니다. `--gazing-ratio-tile`을 생략하면 하위 `nvila_runner`의 NVILA 기본 정책 `[0.2] + [0.06] * 15`가 유지됩니다. Quick Start와 같은 0.75 정책으로 비교하려면 `--gazing-ratio-tile 0.75 --task-loss-requirement-tile 0.7`을 붙이세요.

AutoGaze selector의 inference-only 시간을 보고 싶으면 `--autogaze-generate-only`를 추가합니다. 이 옵션은 하위 `repro.nvila_runner`에 그대로 전달되며, AutoGaze가 patch index를 생성한 뒤 logits/task-loss 계산용 추가 forward를 생략합니다. 논문/Quick Start 기본 forward와 비교할 때는 켰는지 껐는지를 report의 `autogaze_runtime_config.generate_only`에서 반드시 확인하세요.

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

후속 VideoQA 계열 benchmark는 같은 schema로 맞추면 됩니다. 공통 required field는 `video_path`, `question`, `answer`이고, optional field는 `question_id`, `choices`, `category`, `duration`, `source`입니다. helper는 [videoqa_task_schema.py](../repro/videoqa_task_schema.py)에 있으며, multiple-choice scoring은 우선 `A/B/C/D` 답변 안정화에 맞춰두었습니다.

paper comparison wrapper 예시:

```bash
.venv/bin/python scripts/run_hlvid_folder_benchmark.py \
  --dataset-dir /path/to/HLVid \
  --video-root /path/to/extracted/videos \
  --output-dir outputs/autogaze_repro/hlvid_paper_comparison \
  --paper-baseline \
  --paper-hd-autogaze \
  --paper-comparison-report \
  --gazing-ratio-tile 0.75 \
  --task-loss-requirement-tile 0.7 \
  --autogaze-generate-only \
  --measure-ttft \
  --warmup-runs 1 \
  --continue-on-error
```

출력 핵심 파일은 `hlvid_paper_comparison_report.json`과 `hlvid_paper_comparison_report.csv`입니다. report의 `modes.paper_baseline_nvila_8b_video.paper_reference_accuracy=42.5`, `modes.hd_autogaze.paper_reference_accuracy=52.6`, `measured_accuracy`, `delta_from_reference`, `failed`, `oom`, `parse_failed`, `skipped`를 먼저 보세요. `measured_accuracy`는 reference와 같은 percent 단위이고, 원래 fraction은 `measured_accuracy_fraction`에 같이 남습니다.

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
  --video-decode-strategy auto \
  --measure-ttft \
  --warmup-runs 1 \
  --continue-on-error
```

`--video-decode-strategy`는 `auto`, `seek`, `scan` 중 하나입니다. HLVid benchmark에서 decode 속도 개선을 확인하려면 `--video-resize-shortest-edge` 또는 다른 `--video-resize-*` 옵션을 함께 켜야 합니다. resize를 켜지 않으면 runner가 비디오 path를 그대로 NVILA processor에 넘기므로 이 옵션은 NVILA 내부 loader를 바꾸지 않습니다. `auto`는 안전한 기본값이고, seek fallback을 숨기고 싶지 않은 검증 run에서는 `--video-decode-strategy seek`를 사용하세요. 각 sample의 실제 적용 여부는 predictions JSONL 또는 summary의 `video_input_summary.video_decode_strategy`, `video_decode_frames_read`, `video_decode_strategy_fallback_error`에서 확인합니다.

생성되는 주요 파일:

- `hlvid_keep_all_predictions.jsonl`: AutoGaze 미적용 baseline per-sample 결과
- `hlvid_autogaze_predictions.jsonl`: AutoGaze 적용 per-sample 결과
- `hlvid_keep_all_summary.json`, `hlvid_autogaze_summary.json`: 각 run의 HLVid scoring summary와 per-mode latency/memory/token/compute summary
- `hlvid_autogaze_gain_report.json`: keep-all 대비 AutoGaze gain report
- `hlvid_autogaze_gain_report.csv`: 리더 보고용 single-row 요약

리더 리뷰용으로 빠르게 샘플을 확인할 때는 `hlvid_autogaze_gain_report.json`의 `benchmark_samples.autogaze`를 먼저 보세요. 각 항목에는 `target_video`, `question`, `model_answer`, `parsed_model_answer`, `correct_answer`, `ground_truth_answer`, `correct`, `status`가 들어갑니다. keep-all과 AutoGaze의 같은 샘플을 비교하려면 `correctness_comparison.samples` 또는 `benchmark_samples.correctness_comparison`을 보세요. 각 row는 `bucket`, `keep_all_answer`, `keep_all_correct`, `autogaze_answer`, `autogaze_correct`를 같이 담습니다.

`--limit 3`으로 실행했을 때 `readable_summary.run_counts.autogaze_rows=3`이면 AutoGaze 모드가 HLVid row 3개를 처리했다는 뜻입니다. wrapper에서 keep-all과 AutoGaze를 둘 다 켠 기본 상태라면 `keep_all_rows=3`, `autogaze_rows=3`이 각각 생깁니다. `--warmup-runs`로 실행된 warmup은 predictions/scoring row에 포함하지 않습니다.

`--skip-keep-all`로 AutoGaze만 돌린 경우에도 report에는 keep-all 섹션과 `readable_summary.mode_status.keep_all`이 남습니다. 이때 `keep_all_rows=0`, keep-all metric은 0 또는 빈 값으로 보이고, 실제 baseline이 없으므로 cross-mode speedup/reduction ratio는 `null`로 표시됩니다. AutoGaze 자체의 before/after token 감소율은 AutoGaze row 안의 keep-all estimate와 actual 값으로 계속 계산됩니다.

`hlvid_autogaze_gain_report.json`에서 우선 볼 항목:

- `readable_summary.key_metrics_median`: 중요한 latency/token/memory를 한 번에 보는 첫 번째 섹션입니다.
- `readable_summary.latency_ms_median`: `total_ms`, `preprocess_without_autogaze_ms`, `preprocess_total_ms`, `autogaze_total_ms`, `vit_encoder_ms`, `llm_ms`를 keep-all/autogaze median과 함께 보여줍니다.
- `readable_summary.latency_ms_detail_median`: decode, tiling, AutoGaze forward-only, SigLIP, projector, TTFT 같은 세부 latency median입니다.
- `readable_summary.latency_accounting`: `total_ms = video_preprocess_without_autogaze_ms + autogaze_total_ms + generate_ms` primary additive 공식, `legacy_inclusive_total_ms`, total에 다시 더하면 안 되는 nested latency field 목록입니다.
- `readable_summary.memory_bytes_median`: CUDA peak memory를 keep-all/autogaze median으로 비교합니다.
- `readable_summary.tokens_median`: encoder patch, AutoGaze input tile patch, LLM visual token을 before/after 형태로 보여줍니다.
- `gains.accuracy_scored_delta`: AutoGaze와 keep-all의 HLVid accuracy 차이
- `correctness_comparison.counts`: `both_correct`, `keep_all_only_correct`, `autogaze_only_correct`, `both_wrong`, `keep_all_missing`, `autogaze_missing` 샘플 수
- `correctness_comparison.samples`: pair별 정답/오답이 갈린 샘플 확인용 표 데이터
- `gains.latency_speedup_median.total_ms`: 전체 end-to-end median speedup
- `gains.latency_speedup_median.siglip_vision_ms`: SigLIP vision tower median speedup
- `gains.latency_speedup_median.llm_forward_ms`, `gains.latency_speedup_median.ttft_ms`: MLLM 쪽 latency speedup
- `gains.memory_reduction_ratio_median.llm_peak_memory_bytes`: LLM generate CUDA peak memory 감소 비율
- `gains.autogaze_token_reduction_median.llm_visual_token_reduction_ratio`: LLM visual token 감소 비율
- `gains.compute_reduction_median.siglip_total_macs`, `gains.compute_reduction_median.mllm_kv_cache`: 계산량/KV cache 감소 추정
- `gains.reduction_percent_median`: `ratio` 대신 `(before - after) / before * 100`으로 계산한 감소율입니다. 여기서 분모는 keep-all 또는 AutoGaze 적용 전 값입니다.

ratio와 percent는 의도가 다릅니다. `*_ratio_*`는 `before_or_keep_all / after_or_autogaze`라 2.0이면 “2배 작아짐”입니다. `reduction_percent_median`은 원래 값을 분모로 둔 감소율이라 50이면 “원래 대비 50% 감소”입니다.

이미 prediction JSONL이 있는 상태에서 report만 다시 만들려면 같은 `--output-dir`에 대해 `--report-only`를 붙입니다.

## Plugin Runner 8단계 완료 기준

확장성 검증용 8단계는 stable NVILA-HD runner가 아니라 `repro.flexible_runner`와 `repro.plugin_hlvid_benchmark` 기준으로 진행합니다.

현재 repo에서 닫은 범위:

| 단계 | 상태 | 실행/산출물 |
|---|---|---|
| 1. NVILA-Video off smoke | 준비 완료 | `nvila-video` adapter가 공식 `vila-infer` CLI를 호출 |
| 2. LongVILA off smoke | 준비 완료 | `longvila` adapter도 같은 VILA CLI 경로 사용 |
| 3. stdout parsing | 완료 | `Assistant: ...`, JSON `{"answer": ...}`, 마지막 non-empty line 순서로 추출 |
| 4. feature packing probe | 완료 | `feature_packing_probe`에 required input, hook, token accounting target 기록. VILA 계열은 `config.json` 기반 static probe를 `probe_collected`로 추가 기록 |
| 5. post-encoder prune 준비 | 진행 중 | AutoGaze on은 기본적으로 PoC/probe로 막고, Qwen은 명시 플래그에서 post-encoder prune-generate 경로를 실험 |
| 6. InternVL3 off adapter | 준비 완료 | `repro.internvl3_off_infer` helper를 external command로 호출 |
| 7. Qwen probe | 완료 + 실험 경로 추가 | Qwen2/2.5/3-VL AutoGaze on은 `get_video_features` 이후 probe로 기록. `--enable-qwen-prune-generate`를 켜면 visual placeholder를 줄인 `inputs_embeds` generate를 시도 |
| 8. HLVid limit3 report | 준비 완료 | `repro.plugin_hlvid_benchmark`가 predictions/summary/Markdown 생성 |

CUDA 머신에서 NVILA-Video와 LongVILA off smoke는 아래처럼 실행합니다.

```bash
.venv/bin/python -m repro.flexible_runner \
  --mode single \
  --model-path weight/NVILA-8B-Video \
  --model-family nvila-video-plugin \
  --token-selector-adapter keep-all \
  --vision-encoder-adapter nvila-video-vision \
  --mllm-adapter nvila-video \
  --autogaze-integration-level none \
  --external-mllm-command vila-infer \
  --video /path/to/video.mp4 \
  --num-video-frames 256 \
  --max-tiles-video 8 \
  --output-json outputs/autogaze_repro/flexible_nvila_video_plugin_off_single.json
```

```bash
.venv/bin/python -m repro.flexible_runner \
  --mode single \
  --model-path weight/LongVILA \
  --model-family longvila \
  --token-selector-adapter keep-all \
  --vision-encoder-adapter longvila-siglip \
  --mllm-adapter longvila \
  --autogaze-integration-level none \
  --external-mllm-command vila-infer \
  --video /path/to/video.mp4 \
  --num-video-frames 256 \
  --max-tiles-video 8 \
  --output-json outputs/autogaze_repro/flexible_longvila_plugin_off_single.json
```

InternVL3 off smoke는 helper를 사용합니다. 공식 InternVL3 모델 카드의 Transformers 경로는 `AutoModel`, `AutoTokenizer`, `pixel_values`, `num_patches_list`, `model.chat(...)` 흐름이며, repo helper는 이 흐름을 CLI로 감싼 것입니다.

```bash
.venv/bin/python -m repro.flexible_runner \
  --mode single \
  --model-path weight/InternVL3 \
  --model-family internvl3 \
  --token-selector-adapter keep-all \
  --vision-encoder-adapter internvl-dynamic-vision \
  --mllm-adapter internvl3 \
  --autogaze-integration-level none \
  --external-mllm-command ".venv/bin/python -m repro.internvl3_off_infer" \
  --video /path/to/video.mp4 \
  --num-video-frames 8 \
  --max-tiles-video 1 \
  --output-json outputs/autogaze_repro/flexible_internvl3_off_single.json
```

VILA 계열 AutoGaze-on probe는 로컬 모델 폴더에 `config.json`이 있으면 static feature packing probe를 수집합니다. 이 단계는 모델을 실제로 hook하지는 않고, vision tower/projector/video token 관련 config key와 다음 runtime instrumentation target을 기록합니다.

```bash
.venv/bin/python -m repro.vila_feature_probe \
  --model-path weight/NVILA-8B-Video \
  --model-family nvila-video-plugin \
  --video /path/to/video.mp4 \
  --num-video-frames 256 \
  --max-tiles-video 8 \
  --output-json outputs/autogaze_repro/vila_feature_probe_nvila_video.json
```

Qwen3-VL 플러그인 실험의 기준 모델은 Hugging Face의 `Qwen/Qwen3-VL-8B-Instruct`입니다. 로컬 경로는 runner 설정과 맞춰 `weight/Qwen3-VL-8B-Instruct`를 사용합니다. 다운로드는 아래 스크립트로 받습니다.

Qwen 비디오 입력 경로는 별도 helper 패키지인 `qwen-vl-utils`가 필요합니다. 기본 설치는 아래처럼 repro requirements를 다시 설치하면 됩니다.

```bash
.venv/bin/python -m pip install -r requirements-repro.txt
```

이미 대부분 설치되어 있고 해당 패키지만 빠졌다면:

```bash
.venv/bin/python -m pip install qwen-vl-utils
```

```bash
.venv/bin/python scripts/download_qwen_model.py \
  --repo-id Qwen/Qwen3-VL-8B-Instruct \
  --output-dir weight/Qwen3-VL-8B-Instruct
```

먼저 어떤 경로로 받을지 확인만 하려면:

```bash
.venv/bin/python scripts/download_qwen_model.py --dry-run
```

Qwen3-VL AutoGaze post-encoder probe는 아래처럼 실행합니다. 이 명령은 모델을 로드하지 않고 `poc_ready`를 기록합니다.

```bash
.venv/bin/python -m repro.flexible_runner \
  --mode single \
  --model-path weight/Qwen3-VL-8B-Instruct \
  --model-family qwen3-vl \
  --token-selector-adapter autogaze \
  --token-selector-path weight/AutoGaze \
  --vision-encoder-adapter qwen3-vl-vision \
  --mllm-adapter qwen3-vl \
  --autogaze-integration-level post_encoder_token_prune \
  --video /path/to/video.mp4 \
  --output-json outputs/autogaze_repro/flexible_qwen3_vl_autogaze_probe.json
```

Qwen3-VL에서 실제 prune-generate 경로를 실험하려면 아래처럼 명시 플래그를 켭니다. 이 모드는 `get_video_features` 이후 visual feature를 줄이고, 줄어든 visual placeholder 기준으로 `inputs_embeds`를 만들어 `generate`에 넣습니다. `--sparse-selection-plan-json`을 주면 AutoGaze sparse plan의 `selected_patches`를 Qwen `video_grid_thw` 기준 visual feature index로 매핑합니다. plan이 없으면 `gazing_ratio` 기반 placeholder selection으로 fallback되므로, 이 경우는 “Qwen post-encoder bridge 검증”으로만 해석하고 최종 AutoGaze 정확도 비교로 보지는 않습니다.

```bash
.venv/bin/python -m repro.flexible_runner \
  --mode single \
  --model-path weight/Qwen3-VL-8B-Instruct \
  --model-family qwen3-vl \
  --token-selector-adapter autogaze \
  --token-selector-path weight/AutoGaze \
  --vision-encoder-adapter qwen3-vl-vision \
  --mllm-adapter qwen3-vl \
  --autogaze-integration-level post_encoder_token_prune \
  --enable-qwen-prune-generate \
  --sparse-selection-plan-json outputs/autogaze_repro/example_sparse_plan.json \
  --gazing-ratio 0.1 \
  --video /path/to/video.mp4 \
  --output-json outputs/autogaze_repro/flexible_qwen3_vl_autogaze_prune_generate.json
```

Qwen에서 AutoGaze 모델을 실제로 먼저 돌린 뒤 바로 Qwen ViT/MLLM prune-generate에 붙이려면 `--run-autogaze-selector`를 추가합니다. 이 모드는 다음 순서로 실행됩니다.

```text
video
  -> direct AutoGaze selector
       - sampled frames: --num-video-frames
       - chunk: --autogaze-chunk-frames
       - scales/patch: --autogaze-target-scales, --autogaze-target-patch-size
       - output: SparseSelectionPlan JSON
  -> Qwen processor / Qwen ViT get_video_features
       - qwen_vl_utils nframes는 기본적으로 --num-video-frames와 맞춤
  -> selected_patches를 Qwen video_grid_thw visual indices로 매핑
  -> Qwen visual placeholder를 줄인 inputs_embeds
  -> Qwen MLLM generate
```

```bash
.venv/bin/python -m repro.flexible_runner \
  --mode single \
  --model-path weight/Qwen3-VL-8B-Instruct \
  --model-family qwen3-vl \
  --token-selector-adapter autogaze \
  --token-selector-path /path/to/weights/AutoGaze \
  --vision-encoder-adapter qwen3-vl-vision \
  --mllm-adapter qwen3-vl \
  --autogaze-integration-level post_encoder_token_prune \
  --enable-qwen-prune-generate \
  --run-autogaze-selector \
  --autogaze-generate-only \
  --autogaze-repo external/AutoGaze \
  --autogaze-target-scales 32+64+112+224 \
  --autogaze-target-patch-size 16 \
  --autogaze-encoder-patch-size 16 \
  --autogaze-chunk-frames 16 \
  --max-batch-size-autogaze 1 \
  --num-video-frames 16 \
  --gazing-ratio 0.75 \
  --task-loss-requirement 0.7 \
  --video external/AutoGaze/assets/example_input.mp4 \
  --output-json outputs/autogaze_repro/flexible_qwen3_vl_direct_autogaze_prune_generate.json
```

출력에서 먼저 확인할 위치:

- `direct_autogaze_selector.sparse_selection_plan_json`: 실제 AutoGaze가 만든 sparse plan 파일
- `direct_autogaze_selector.tokens.raw_patch_tokens`: AutoGaze 입력 후보 패치 수
- `direct_autogaze_selector.tokens.selected_patch_tokens`: AutoGaze가 선택한 non-padded 패치 수
- `generation.metrics.qwen_prune_generate.selection_source`: `sparse_selection_plan`이면 실제 AutoGaze plan이 Qwen prune에 사용된 것
- `generation.metrics.tokens.visual_tokens_before_prune / visual_tokens_after_prune`: Qwen MLLM에 들어가는 visual placeholder 감소량

주의: `post_encoder_token_prune` 경로는 Qwen ViT는 full video feature를 계산하고, AutoGaze 이득을 MLLM context/prefill/KV cache 쪽에서 먼저 확인하는 경로입니다. ViT 연산량 감소까지 보려면 아래의 `qwen_chunked_vit_autogaze_sparse` 또는 기존 `pre_encoder_sparse + autogaze-sparse` 경로를 사용합니다.

Qwen ViT 연산량까지 줄이는 실험 경로는 `pre_encoder_sparse + autogaze-sparse`로 켭니다. 이 경로는 AutoGaze plan을 Qwen `video_grid_thw`의 merged visual token index로 매핑한 뒤, Qwen visual transformer에 들어가는 merged-token group만 통과시키고, 같은 index의 visual placeholder만 MLLM 입력에 남깁니다. 아직 모델별 실험 hook이므로 CUDA smoke에서 `qwen_pre_vit_sparse.status`와 `metric_status`를 먼저 확인해야 합니다.

```bash
.venv/bin/python -m repro.flexible_runner \
  --mode single \
  --model-path weight/Qwen3-VL-8B-Instruct \
  --model-family qwen3-vl \
  --token-selector-adapter autogaze \
  --token-selector-path /path/to/weights/AutoGaze \
  --vision-encoder-adapter qwen3-vl-vision \
  --mllm-adapter qwen3-vl \
  --autogaze-integration-level pre_encoder_sparse \
  --pre-encoder-prune-adapter autogaze-sparse \
  --enable-qwen-prune-generate \
  --run-autogaze-selector \
  --autogaze-generate-only \
  --autogaze-repo external/AutoGaze \
  --autogaze-target-scales 32+64+112+224 \
  --autogaze-target-patch-size 16 \
  --autogaze-encoder-patch-size 16 \
  --autogaze-chunk-frames 16 \
  --max-batch-size-autogaze 1 \
  --num-video-frames 16 \
  --qwen-video-max-pixels 200704 \
  --gazing-ratio 0.75 \
  --task-loss-requirement 0.7 \
  --video external/AutoGaze/assets/example_input.mp4 \
  --output-json outputs/autogaze_repro/flexible_qwen3_vl_direct_autogaze_pre_vit_sparse.json
```

Qwen ViT 비교는 아래 세 모드를 같은 입력으로 나란히 돌립니다.

```text
qwen_full_vit
  video -> qwen_vl_utils/processer -> native Qwen get_video_features(full) -> MLLM

qwen_chunked_vit
  video -> qwen_vl_utils/processor -> pixel_values_videos 생성
        -> Qwen processor patch grid를 temporal chunk + spatial tile로 분할
        -> Qwen ViT chunk/tile forward(keep-all) -> concat features -> MLLM

qwen_chunked_vit_autogaze_sparse
  video -> direct AutoGaze selector -> SparseSelectionPlan
        -> qwen_vl_utils/processor -> pixel_values_videos 생성
        -> Qwen processor patch grid를 temporal chunk + spatial tile로 분할
        -> AutoGaze selected merged-token만 Qwen ViT chunk/tile forward
        -> selected visual placeholder만 MLLM context에 packing
```

여기서 “spatial tile”은 NVILA처럼 원본 프레임을 먼저 crop해서 Qwen processor에 여러 비디오로 넣는 완전한 전처리 tile은 아닙니다. 현재 1차 구현은 Qwen processor가 resize/patchify한 뒤 나온 `video_grid_thw=[T,H,W]`의 `H/W` patch grid를 NVILA식 max-tile ratio로 나눕니다. 그래서 ViT block residency와 sparse token 계산량 비교에는 쓸 수 있지만, Qwen processor의 decode/resize peak 자체를 줄이려면 후속으로 processor 이전 spatial crop path가 필요합니다.

단일 비디오에서 비교:

```bash
# 1) native full Qwen ViT
.venv/bin/python -m repro.flexible_runner \
  --mode single \
  --model-path weight/Qwen3-VL-8B-Instruct \
  --model-family qwen3-vl \
  --token-selector-adapter keep-all \
  --vision-encoder-adapter qwen3-vl-vision \
  --mllm-adapter qwen3-vl \
  --autogaze-integration-level none \
  --qwen-vit-mode qwen_full_vit \
  --num-video-frames 32 \
  --qwen-video-nframes 32 \
  --qwen-video-max-pixels 200704 \
  --video /path/to/video.mp4 \
  --output-json outputs/autogaze_repro/qwen_full_vit.json

# 2) chunked Qwen ViT, AutoGaze off
.venv/bin/python -m repro.flexible_runner \
  --mode single \
  --model-path weight/Qwen3-VL-8B-Instruct \
  --model-family qwen3-vl \
  --token-selector-adapter keep-all \
  --vision-encoder-adapter qwen3-vl-vision \
  --mllm-adapter qwen3-vl \
  --autogaze-integration-level none \
  --qwen-vit-mode qwen_chunked_vit \
  --qwen-vit-chunk-frames 16 \
  --qwen-vit-max-spatial-chunks 4 \
  --num-video-frames 32 \
  --qwen-video-nframes 32 \
  --qwen-video-max-pixels 200704 \
  --video /path/to/video.mp4 \
  --output-json outputs/autogaze_repro/qwen_chunked_vit.json

# 3) chunked Qwen ViT + direct AutoGaze sparse
.venv/bin/python -m repro.flexible_runner \
  --mode single \
  --model-path weight/Qwen3-VL-8B-Instruct \
  --model-family qwen3-vl \
  --token-selector-adapter autogaze \
  --token-selector-path /path/to/weights/AutoGaze \
  --vision-encoder-adapter qwen3-vl-vision \
  --mllm-adapter qwen3-vl \
  --autogaze-integration-level pre_encoder_sparse \
  --pre-encoder-prune-adapter autogaze-sparse \
  --enable-qwen-prune-generate \
  --run-autogaze-selector \
  --autogaze-generate-only \
  --autogaze-repo external/AutoGaze \
  --autogaze-target-scales 32+64+112+224 \
  --autogaze-target-patch-size 16 \
  --autogaze-encoder-patch-size 16 \
  --autogaze-chunk-frames 16 \
  --max-batch-size-autogaze 1 \
  --qwen-vit-mode qwen_chunked_vit_autogaze_sparse \
  --qwen-vit-chunk-frames 16 \
  --qwen-vit-max-spatial-chunks 4 \
  --num-video-frames 32 \
  --qwen-video-nframes 32 \
  --qwen-video-max-pixels 200704 \
  --gazing-ratio 0.75 \
  --task-loss-requirement 0.7 \
  --video /path/to/video.mp4 \
  --output-json outputs/autogaze_repro/qwen_chunked_vit_autogaze_sparse.json
```

확인할 핵심 필드:

- `generation.metrics.qwen_vit.mode`: 세 비교 모드 중 어떤 경로인지
- `generation.metrics.qwen_vit.processor_chunking`: 현재는 Qwen processor가 만든 `pixel_values_videos` 이후 chunking입니다. 즉 decode/resize 메모리는 아직 Qwen processor 정책의 영향을 받습니다.
- `generation.metrics.qwen_vit.spatial_chunking.tile_grid`: Qwen patch grid를 몇 개 spatial tile로 나눴는지입니다. `--qwen-vit-max-spatial-chunks`를 지정하지 않으면 Qwen chunked 모드는 `--max-tiles-video` 값을 기본으로 씁니다.
- `generation.metrics.qwen_vit.raw_patch_tokens_before_vit`: Qwen ViT patch embedding 입력 토큰 수
- `generation.metrics.qwen_vit.visual_tokens_before_prune / visual_tokens_after_prune`: Qwen merged visual token 기준 AutoGaze 전후 수
- `generation.metrics.tokens.visual_token_reduction_ratio`: 분모는 AutoGaze 전 Qwen visual placeholder 수, 분자는 AutoGaze 후 placeholder 수입니다.
- `generation.metrics.latency_ms.qwen_vit_prepare`: chunked/sparse path에서 ViT feature 생성과 MLLM input packing에 걸린 시간입니다.

중요한 제한:

- `qwen_chunked_vit`는 peak ViT residency를 줄이기 위한 temporal+spatial chunked feature extraction 실험입니다. AutoGaze off라서 최종 visual token 수는 줄지 않습니다.
- `qwen_chunked_vit_autogaze_sparse`는 AutoGaze 선택 token만 Qwen visual transformer block과 MLLM context에 통과시킵니다.
- 아직 “진짜 decode streaming”은 아닙니다. 긴 4K 영상에서 decode/resize 자체 OOM이 나면 `--qwen-video-nframes`, `--qwen-video-max-pixels`를 먼저 줄여야 합니다.

Qwen off 또는 AutoGaze off로 큰 비디오를 넣을 때는 NVILA runner와 다르게 Qwen의 `qwen_vl_utils`가 비디오 디코드/샘플링/resize를 담당합니다. runner는 Qwen 계열 비디오 실행에서 `--qwen-video-nframes`를 기본적으로 `--num-video-frames`와 맞추지만, 4K/긴 영상은 아래처럼 해상도 cap도 같이 주는 편이 안전합니다.

```bash
.venv/bin/python -m repro.flexible_runner \
  --mode single \
  --model-path weight/Qwen3-VL-8B-Instruct \
  --model-family qwen3-vl \
  --token-selector-adapter keep-all \
  --vision-encoder-adapter qwen3-vl-vision \
  --mllm-adapter qwen3-vl \
  --autogaze-integration-level none \
  --num-video-frames 32 \
  --qwen-video-nframes 32 \
  --qwen-video-max-pixels 200704 \
  --video /path/to/large_4k_video.mp4 \
  --output-json outputs/autogaze_repro/flexible_qwen3_vl_off_large_video.json
```

`qwen_vl_utils` 단계에서 죽으면 JSON의 `generation.metrics.metric_status.reason` 또는 터미널 에러에 `video`, `nframes`, `fps`, `max_pixels`가 같이 표시됩니다. 여러 비디오 종류를 테스트하는 것은 좋지만, 우선은 같은 비디오에 대해 `nframes`와 `max_pixels`를 고정한 뒤 비교해야 NVILA runner와 같은 조건이 됩니다.

HLVid limit3 plugin 비교는 `configs/repro/plugin_hlvid_limit3.yaml`에 대응합니다. CUDA 머신의 HLVid 폴더가 mp4 flat 구조여도 `video_root / basename(video_path)` fallback으로 찾습니다.

```bash
.venv/bin/python -m repro.plugin_hlvid_benchmark \
  --manifest /path/to/HLVid/data/test-00000-of-00001.parquet \
  --video-root /path/to/HLVid/videos \
  --output-dir outputs/autogaze_repro/plugin_hlvid_limit3 \
  --limit 3 \
  --modes nvila-video-off,longvila-off,internvl3-off,qwen_full_vit,qwen_chunked_vit,qwen_chunked_vit_autogaze_sparse \
  --external-mllm-command vila-infer \
  --model nvila-video=weight/NVILA-8B-Video \
  --model longvila=weight/LongVILA \
  --model internvl3=weight/InternVL3 \
  --model qwen3-vl=weight/Qwen3-VL-8B-Instruct \
  --num-video-frames 256 \
  --qwen-video-nframes 256 \
  --qwen-video-max-pixels 200704 \
  --qwen-vit-chunk-frames 16 \
  --qwen-vit-max-spatial-chunks 8 \
  --max-tiles-video 8
```

결과 파일:

- `plugin_hlvid_predictions.jsonl`: mode별 per-row output, question, answer, status, metric_status
- `plugin_hlvid_summary.json`: mode별 score와 failed/parse_failed 분리
- `plugin_hlvid_report.md`: 리더 리뷰용 간단 표

## 검증

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/python -m repro.plugin_hlvid_benchmark --help
.venv/bin/python -m repro.autogaze_bench --help
.venv/bin/python -m repro.hlvid --help
.venv/bin/python scripts/run_hlvid_folder_benchmark.py --help
.venv/bin/python -m repro.nvila_runner --help
.venv/bin/python -m repro.report --help
.venv/bin/python -m repro.markdown_report --help
```
