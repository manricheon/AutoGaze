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
  --output-json outputs/autogaze_repro/cuda_nvila_single_128f.json
```

이 명령은 공식 NVILA-HD-Video quickstart 규모를 따라가며, 전체 HLVid 실행 전에 모델과 processor path가 정상인지 검증합니다.

NVILA runner는 output JSON에 모듈별 timing을 기록합니다. 중요한 필드는 아래와 같습니다.

- `result.video_decode_ms`: 샘플링된 프레임 decode/read 시간
- `result.video_tiling_ms`: dynamic tiling 및 image tensorization 시간
- `result.autogaze_ms`: padding/splitting bookkeeping까지 포함한 전체 AutoGaze selection stage 시간
- `result.autogaze_forward_ms`: AutoGaze가 실제 호출된 경우 AutoGaze model forward 시간
- `result.vision_encoder_ms`: SigLIP feature와 projector 준비를 포함한 NVILA vision encoding stage 시간
- `result.siglip_vision_ms`: SigLIP vision tower forward 시간
- `result.mm_projector_ms`: multimodal projector forward 시간
- `result.llm_forward_ms`: generation 동안 누적된 LLM forward 시간
- `result.ttft_ms`: `--measure-ttft`가 켜졌을 때, 처리된 visual/text input에서 1토큰을 생성하는 데 걸린 시간
- `result.decode_estimated_ms`: full `generate_ms - ttft_ms`로 계산한 대략적인 generation decode 시간
- `result.stage_timings_ms`: `processor`, 선택적 `ttft`, full `generate`의 raw nested timing bucket
- `result.token_metrics`: encoder patch budget 및 LLM visual-token budget 기준 AutoGaze 전후 visual token count
- `result.processor_peak_memory_bytes`, `result.peak_memory_bytes`: CUDA 실행 시 processor phase와 full generate phase의 CUDA peak allocation

`--measure-ttft`는 preprocessing 이후 1토큰 generation을 추가로 실행합니다. 이 파이프라인에서 TTFT는 순수 text decoding latency만이 아닙니다. `generate`에서 필요한 visual embedding, SigLIP/vision encoding, projector work, 첫 LLM forward까지 포함될 수 있습니다. 세부 분리는 `result.ttft_stage_timings_ms`의 `vision_encode_total`, `siglip_vision_tower`, `mm_projector`, `llm_forward`를 확인하세요.

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
  --output-json outputs/autogaze_repro/cuda_nvila_single_128f_autogaze.json

.venv/bin/python -m repro.nvila_runner \
  --mode single \
  --device cuda \
  --gazing-mode keep-all \
  --num-video-frames 128 \
  --num-video-frames-thumbnail 64 \
  --max-tiles-video 48 \
  --measure-ttft \
  --output-json outputs/autogaze_repro/cuda_nvila_single_128f_keep_all.json
```

속도 관점에서는 두 JSON 파일의 `total_ms`, `video_decode_ms`, `video_tiling_ms`, `autogaze_forward_ms`, `siglip_vision_ms`, `vision_encoder_ms`, `llm_forward_ms`를 비교합니다. 토큰 관점에서는 `token_metrics.encoder_raw_patch_tokens`, `token_metrics.encoder_autogaze_selected_patch_tokens`, `token_metrics.encoder_token_reduction_ratio`, `token_metrics.llm_keep_all_visual_tokens_estimated`, `token_metrics.llm_actual_visual_tokens`, `token_metrics.llm_visual_token_reduction_ratio`를 비교합니다.

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
  --resume \
  --predictions outputs/autogaze_repro/hlvid_full_keep_all_predictions.jsonl \
  --summary outputs/autogaze_repro/hlvid_full_keep_all_summary.json \
  --scored-predictions outputs/autogaze_repro/hlvid_full_keep_all_scored.jsonl
```

`accuracy_scored`를 NVILA-8B-HD-Video의 project-page HLVid target인 `52.6`과 비교하세요. skipped, failed, parse-failed sample은 별도로 보고해야 합니다. AutoGaze-vs-keep-all claim을 만들 때는 accuracy뿐 아니라 median 또는 mean 기준 `total_ms`, `video_decode_ms`, `video_tiling_ms`, `autogaze_forward_ms`, `vision_encoder_ms`, `llm_forward_ms`, `token_metrics.encoder_token_reduction_ratio`, `token_metrics.llm_visual_token_reduction_ratio`를 함께 비교하세요. 논문 대응 setup은 target GPU가 허용하는 범위에서 NVILA-8B-HD-Video, 최대 1024프레임, 최대 해상도 3584를 기준으로 합니다.

## 검증

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/python -m repro.autogaze_bench --help
.venv/bin/python -m repro.hlvid --help
.venv/bin/python -m repro.nvila_runner --help
.venv/bin/python -m repro.report --help
```
