# HLVid NVILA-HD 입력/출력 가이드

이 문서는 HLVid를 `scripts/evaluate_hlvid_nvila.py`로 돌릴 때 AutoGaze, SigLIP, NVILA가 각각 어떤 입력을 받고 어떤 출력을 만드는지 정리합니다.

중요한 기준은 하나입니다.

```text
HLVid 재현/평가 경로 = scripts/evaluate_hlvid_nvila.py + official NVILA-HD processor
단일 비디오 실험/시각화 경로 = scripts/infer_full.py + PoC resize/chop/resize_then_chop
```

HLVid 성능 확인에는 `resize_then_chop`를 기준으로 해석하면 안 됩니다. `resize_then_chop`는 PoC 시각화/latency 실험용입니다.

## 실행 명령

모델을 로드하지 않는 dry-run:

```bash
python scripts/evaluate_hlvid_nvila.py \
  --config configs/poc_inference/hlvid_nvila_hd_smoke.yaml \
  --dataset-path /path/to/hlvid_sample.jsonl \
  --video-root /path/to/videos \
  --output-dir outputs/hlvid_nvila_dry_run \
  --dry-run
```

로컬 weight로 20개 subset 실행:

```bash
python scripts/evaluate_hlvid_nvila.py \
  --config configs/poc_inference/hlvid_nvila_hd_subset.yaml \
  --dataset-name bfshi/HLVid \
  --model-path weights/NVILA-8B-HD-Video \
  --processor-path weights/NVILA-8B-HD-Video \
  --allow-real-model-loading \
  --local-files-only \
  --max-samples 20 \
  --output-dir outputs/hlvid_nvila_real
```

AutoGaze ON/OFF를 같은 데이터로 비교:

```bash
python scripts/evaluate_hlvid_nvila.py \
  --config configs/poc_inference/hlvid_nvila_hd_subset.yaml \
  --dataset-path /data/HLVid/annotations.jsonl \
  --video-root /data/HLVid/videos \
  --model-path weights/NVILA-8B-HD-Video \
  --processor-path weights/NVILA-8B-HD-Video \
  --allow-real-model-loading \
  --local-files-only \
  --max-samples 20 \
  --compare-autogaze-on-off \
  --output-dir outputs/hlvid_nvila_on_off
```

이 명령은 아래 두 하위 디렉터리를 만듭니다.

```text
outputs/hlvid_nvila_on_off/autogaze_on_off_comparison/autogaze_off/
outputs/hlvid_nvila_on_off/autogaze_on_off_comparison/autogaze_on/
```

그리고 비교 요약을 저장합니다.

```text
outputs/hlvid_nvila_on_off/autogaze_on_off_comparison/logs/autogaze_comparison.json
outputs/hlvid_nvila_on_off/autogaze_on_off_comparison/logs/autogaze_comparison.md
```

단일 모드만 강제로 바꾸고 싶으면:

```bash
--autogaze-mode on
--autogaze-mode off
--autogaze-mode config
```

`off`는 tile/thumbnail gazing 설정을 `null`로 만들어 pruning 없이 모든 patch를 유지하는 비교 모드입니다.

## 로컬 dataset-path 의미

`--dataset-path`는 비디오 폴더가 아니라 **annotation 파일 또는 annotation이 들어 있는 dataset 디렉터리**입니다.

파일로 주는 경우:

```text
/data/HLVid/annotations.jsonl
```

디렉터리로 주는 경우:

```text
/data/HLVid/
```

디렉터리 모드에서는 script가 그 안에서 `.jsonl`, `.json`, `.csv` annotation 파일을 찾습니다. `annotations.jsonl`, `metadata.json`, `hlvid.jsonl` 같은 이름을 우선적으로 고릅니다.

`--video-root`는 annotation 안의 상대 `video_path` 앞에 붙는 비디오 root입니다.

예를 들어 annotation이 이렇게 되어 있으면:

```json
{"video_path": "clip_av_video_5_001.mp4", "question": "...", "A": "...", "B": "...", "C": "...", "D": "...", "answer": "B"}
```

아래처럼 실행합니다.

```bash
python scripts/evaluate_hlvid_nvila.py \
  --config configs/poc_inference/hlvid_nvila_hd_smoke.yaml \
  --dataset-path /data/HLVid/annotations.jsonl \
  --video-root /data/HLVid/videos \
  --dry-run
```

실제 비디오 경로는 다음처럼 해석됩니다.

```text
/data/HLVid/videos/clip_av_video_5_001.mp4
```

`--dataset-path`를 디렉터리로 주고 `/data/HLVid/videos` 폴더가 있으면 `--video-root`를 생략해도 기본값으로 그 폴더를 사용합니다.

## HLVid Record 입력

한 샘플은 보통 다음 필드를 가집니다.

```text
video_path
question
options: A/B/C/D
answer
```

Evaluator는 이것을 다음 프롬프트로 바꿉니다.

```text
<video>

Question: ...
A. ...
B. ...
C. ...
D. ...
Please answer directly with the letter of the correct answer.
```

## NVILA-HD Processor 입력/출력

입력:

```text
source video path
text prompt with <video> token
```

`hlvid_nvila_hd_subset.yaml` 기본값:

```yaml
num_video_frames: 128
num_video_frames_thumbnail: 64
max_tiles_video: 48
tile_size: 392x392
autogaze_chunk_size: 16 frames
gazing_ratio_tile: [0.2] + [0.06] * 15
task_loss_requirement_tile: 0.6
gazing_ratio_thumbnail: 1
task_loss_requirement_thumbnail: null
max_batch_size_autogaze: 16
max_batch_size_siglip: 32
```

실제 처리:

```text
원본 비디오
-> num_video_frames 개를 균일 샘플링
-> 각 샘플 프레임을 원본 종횡비에 맞는 tile grid로 resize
-> 392x392 tile들로 crop
-> 별도 thumbnail frame 생성
```

예시: 원본이 `1920x1080`, `max_tiles_video=48`이면 대략 `9 x 5 = 45` spatial tiles가 선택될 수 있습니다.

```text
1920x1080 frame
-> 3528x1960 resize
-> 392x392 tile 45개
```

128 sampled frames이면:

```text
128 frames / 16 AutoGaze frames = 8 temporal chunks
45 spatial tiles * 8 temporal chunks = 360 tile clips
```

Processor 출력:

```text
pixel_values_videos_tiles
pixel_values_videos_thumbnails
num_spatial_tiles_each_video
gazing_info
expanded input_ids with enough video placeholders
```

## AutoGaze 입력/출력

AutoGaze는 원본 고해상도 프레임을 직접 보지 않습니다.

입력:

```text
tile clips: [num_tile_clips, 16, 3, 392, 392]
thumbnail clips: [num_thumbnail_frames, 1, 3, 392, 392]
```

출력:

```text
gazing_pos
if_padded_gazing
num_gazing_each_frame
```

의미:

```text
gazing_pos: 선택된 multi-scale patch index
if_padded_gazing: dummy/padded 선택 여부
num_gazing_each_frame: frame별 선택 patch 수
```

Tile 쪽은 AutoGaze가 patch를 줄이는 핵심 경로입니다. Thumbnail 쪽은 기본 설정에서 `gazing_ratio_thumbnail=1`, `task_loss_requirement_thumbnail=null`이라 global context 보존에 가깝습니다.

## SigLIP 입력/출력

SigLIP 입력:

```text
pixel_values_videos_tiles
pixel_values_videos_thumbnails
gazing_info
```

SigLIP는 official NVILA 모델 안의 modified SigLIP 경로로 실행됩니다. `gazing_info`가 들어가면 선택된 patch 중심으로 feature를 만들고, padded gaze feature는 이후 제거됩니다.

SigLIP 출력:

```text
selected visual features per tile/frame
selected visual features per thumbnail
```

## NVILA 입력/출력

NVILA 입력:

```text
text prompt tokens
video placeholder tokens
AutoGaze-selected SigLIP visual features
```

NVILA 내부 처리:

```text
selected SigLIP features
-> frame-first order로 재정렬
-> frame별 token 수를 TokenShuffle(9)에 맞게 padding
-> mm_projector
-> LLM visual embeddings
-> text prompt와 함께 generate
```

최종 출력:

```text
prediction_text
prediction_choice: A/B/C/D
correct: true/false/null
```

## 실행마다 저장되는 리포트

`scripts/evaluate_hlvid_nvila.py`는 매번 아래 파일을 저장합니다.

```text
outputs/.../predictions/hlvid_predictions.json
outputs/.../predictions/hlvid_predictions.jsonl
outputs/.../predictions/hlvid_predictions.csv
outputs/.../logs/poc_summary.json
outputs/.../logs/metrics.json
outputs/.../logs/metrics.csv
outputs/.../logs/run_report.json
outputs/.../logs/hlvid_report.md
```

`hlvid_predictions.csv`에는 sample별 주요 필드가 들어갑니다.

```text
latency_ms
processor_latency_ms
model_generate_latency_ms
input_token_count
video_placeholder_token_count
output_new_token_count
input_tensor_mib
process_peak_rss_mib
cuda_max_memory_allocated_mib
```

`run_report.json`과 `hlvid_report.md`에는 전체 요약이 들어갑니다.

```text
latency_report
memory_report
token_consumption_report
official_high_resolution_processing
```

## 리포트 해석

Latency:

```text
processor_latency_ms:
  NVILA processor가 비디오 path를 받아 frame sampling, tile/thumbnail 생성,
  AutoGaze gazing_info 구성, tokenizer 입력 생성까지 수행하는 시간

model_generate_latency_ms:
  NVILA model.generate 자체 시간

decode_latency_ms:
  output token을 text로 decode하는 시간

latency_ms:
  sample 하나의 전체 generation wrapper 시간
```

Memory:

```text
process_peak_rss_mib:
  프로세스 기준 peak RSS

cuda_max_memory_allocated_mib:
  CUDA 사용 시 torch peak allocated memory

input_tensor_mib:
  processor가 만든 tensor 입력들의 대략적인 합계 크기
```

Token consumption:

```text
input_token_count:
  text + expanded video placeholders를 포함한 input_ids 길이

video_placeholder_token_count:
  NVILA가 visual feature를 꽂기 위해 확장한 video token 수

output_new_token_count:
  새로 생성된 answer token 수
```

주의: official NVILA processor 내부의 tile 수와 AutoGaze selected patch 수는 processor/model 내부에서 결정됩니다. 이 evaluator의 token report는 LLM 입력 token과 generated token 중심입니다. 세부 AutoGaze patch 수를 시각화하려면 `infer_full.py` 또는 `infer_autogaze.py`의 PoC probe를 따로 사용해야 합니다.

## 어떤 모드를 써야 하나

HLVid 성능/재현 목적:

```text
scripts/evaluate_hlvid_nvila.py
configs/poc_inference/hlvid_nvila_hd_subset.yaml
```

단일 비디오 latency/시각화/debug 목적:

```text
scripts/infer_full.py
configs/poc_inference/hlvid_infer_full_resize_safe.yaml
configs/poc_inference/hlvid_infer_full_resize_then_chop_safe.yaml
```

공식 고해상도 처리 해석:

```text
source video path
-> official NVILA-HD processor
-> 128 sampled frames
-> 392x392 dynamic tiles + thumbnails
-> AutoGaze gazing_info
-> modified SigLIP
-> NVILA
```
