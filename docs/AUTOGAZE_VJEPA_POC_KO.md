# AutoGaze + V-JEPA + Qwen PoC

이 문서는 AutoGaze가 선택한 multiscale patch를 V-JEPA token index로 옮기는 PoC를 다룹니다. 목표는 아직 성능 주장이 아니라, `AutoGaze patch index -> V-JEPA tubelet/grid index -> sparse encoder hook -> Qwen bridge 후보`가 구조적으로 가능한지 확인하는 것입니다.

## 지원 모드

| 모드 | 목적 | 정확도 주장 |
|---|---|---|
| single-grid mapping | AutoGaze bbox를 하나의 V-JEPA crop/grid에 overlap 매핑 | 안 함 |
| scale-aware mapping | AutoGaze scale별로 V-JEPA pass를 분리한 token packing 후보 검증 | 안 함 |
| tiny sparse encoder smoke | random-weight tiny V-JEPA encoder에 selected hidden states만 통과 | 안 함 |
| Qwen bridge smoke | selected V-JEPA feature를 Qwen visual placeholder embedding으로 삽입 | 안 함 |
| actual single runner off | 실제 video -> dense V-JEPA keep-all -> Qwen generate | 안 함 |
| actual single runner on | 실제 video -> AutoGaze -> V-JEPA sparse encoder -> Qwen generate | 안 함 |
| HLVid wrapper | HLVid row를 순회하며 dense/off와 AutoGaze/on runner 실행/스코어링 | 참고용 |

## 핵심 정책

- `tubelet_size=2`이면 같은 tubelet에 들어가는 두 프레임의 선택 patch를 union합니다.
- 저해상도 AutoGaze patch는 V-JEPA grid에서 bbox overlap되는 모든 spatial cell로 확장합니다.
- scale-aware 모드에서는 scale마다 별도 V-JEPA grid를 두므로 coarse patch가 고해상도 grid로 과도하게 펼쳐지는지 비교할 수 있습니다.
- 기본 `nvidia/AutoGaze` checkpoint는 V-JEPA/Qwen PoC에서 `--autogaze-tile-size 224`와 `--autogaze-target-scales 32+64+112+224`를 우선 사용합니다. 즉 largest scale은 V-JEPA crop 224에 맞추고, 낮은 해상도 scale도 checkpoint의 scale 개수에 맞춰 함께 둡니다. `32+64+112+224`는 NVILA-HD에서 쓰던 `56+112+196+392` pyramid를 `224 / 392` 비율로 줄인 값입니다.
- V-JEPA 기본 crop이 224이고 patch size가 16이면 dense single-grid는 한 프레임당 `14 x 14 = 196` spatial cell을 봅니다. AutoGaze의 coarse scale patch는 이 224 기준 grid에서 overlap되는 cell들의 union으로 옮깁니다. 그래서 coarse patch 하나가 V-JEPA cell 여러 개로 펼쳐질 수 있습니다.
- `56+112+196+392` 같은 NVILA-HD multiscale 설정은 해당 scale 개수를 지원하는 AutoGaze checkpoint와 392 기준 VLM pipeline에서만 사용하세요. V-JEPA/Qwen PoC에서는 224 기준 설정을 기본 비교점으로 둡니다.
- `--gazing-ratio`를 생략하면 AutoGaze checkpoint의 inference 기본 정책을 사용합니다. token 감소율을 명시적으로 sweep하려면 `--gazing-ratio 0.1`, `--gazing-ratio 0.25`처럼 지정합니다.
- Qwen 연결은 학습된 projector 전까지 `zero_shot_wiring_probe`로만 봅니다.

```text
single-grid

video frames
  -> AutoGaze multiscale selected patches
  -> bbox overlap on one V-JEPA grid
  -> frame-pair union by tubelet
  -> selected V-JEPA token indices
  -> dense patch embedding
  -> sparse encoder hook
  -> Qwen inputs_embeds bridge
```

```text
scale-aware

video frames
  -> AutoGaze scale  32 selected patches -> V-JEPA  32 grid selected tokens
  -> AutoGaze scale  64 selected patches -> V-JEPA  64 grid selected tokens
  -> AutoGaze scale 112 selected patches -> V-JEPA 112 grid selected tokens
  -> AutoGaze scale 224 selected patches -> V-JEPA 224 grid selected tokens
  -> concat scale-aware features with scale/frame/row/col metadata
  -> Qwen bridge candidate
```

## 실제 비디오 전체 파이프라인

`repro.vjepa_qwen_runner`는 synthetic plan을 쓰지 않고 실제 비디오에서 end-to-end로 실행합니다. `--autogaze-mode off`에서는 V-JEPA token을 전부 keep-all로 통과시키고, `--autogaze-mode on`에서는 AutoGaze가 고른 patch bbox를 V-JEPA token index로 매핑한 뒤 selected token embedding만 V-JEPA encoder block에 통과시킵니다. 두 경우 모두 마지막에는 V-JEPA feature를 Qwen `inputs_embeds`에 삽입합니다.

```text
video
  -> sampled frames / optional resize / spatial tile canvas
  -> [off] dense V-JEPA token selection
  -> [on]  AutoGaze actual selector
       output: SparseSelectionPlan(selected multiscale patch bbox)
  -> V-JEPA frame sampler
       output: [B, T, C, H, W]
  -> V-JEPA patch embedding
  -> [off] all V-JEPA token indices
  -> [on]  selected token gather by AutoGaze bbox -> V-JEPA grid/tubelet index
  -> dense/sparse V-JEPA encoder
  -> deterministic V-JEPA-to-Qwen dim bridge
  -> Qwen generate(inputs_embeds)
```

중요: 마지막 bridge는 학습된 projector가 아니라 repeat/truncate 기반 wiring probe입니다. 즉 CUDA에서 end-to-end 동작과 token/latency/memory 계측은 가능하지만, 이 출력으로 모델 성능을 주장하면 안 됩니다.

실행 예시:

```bash
python -m repro.vjepa_qwen_runner \
  --video inputs/hlvid_example/clip_av_video_5_001.mp4 \
  --prompt "Describe the video in one short sentence." \
  --autogaze-mode on \
  --autogaze-model weight/AutoGaze \
  --vjepa-model weight/vjepa2-vitl-fpc64-256 \
  --qwen-model weight/Qwen2.5-VL-3B-Instruct \
  --device cuda \
  --dtype float16 \
  --num-video-frames 16 \
  --frames-per-clip 16 \
  --autogaze-chunk-frames 16 \
  --max-tiles-video 1 \
  --autogaze-tile-size 224 \
  --autogaze-target-scales 32+64+112+224 \
  --autogaze-target-patch-size 16 \
  --vjepa-selection-policy single_scale_union \
  --video-decode-strategy seek \
  --video-resize-longest-edge 448 \
  --max-new-tokens 32 \
  --output-json outputs/autogaze_vjepa/vjepa_qwen_actual.json \
  --output-md outputs/autogaze_vjepa/vjepa_qwen_actual.md
```

AutoGaze off dense baseline은 같은 bridge와 Qwen generate를 유지하되 AutoGaze selector를 건너뜁니다. 이 모드는 AutoGaze checkpoint 없이도 V-JEPA/Qwen 모델만 있으면 실행됩니다.

```bash
python -m repro.vjepa_qwen_runner \
  --video inputs/hlvid_example/clip_av_video_5_001.mp4 \
  --prompt "Describe the video in one short sentence." \
  --autogaze-mode off \
  --vjepa-model weight/vjepa2-vitl-fpc64-256 \
  --qwen-model weight/Qwen2.5-VL-3B-Instruct \
  --device cuda \
  --dtype float16 \
  --frames-per-clip 16 \
  --video-decode-strategy seek \
  --video-resize-longest-edge 448 \
  --max-new-tokens 32 \
  --output-json outputs/autogaze_vjepa/vjepa_qwen_dense_off.json \
  --output-md outputs/autogaze_vjepa/vjepa_qwen_dense_off.md
```

멀티스케일을 더 직접적으로 보고 싶으면 아래처럼 scale별 V-JEPA sparse pass를 수행합니다. 이 모드는 더 무겁지만 AutoGaze scale별 index가 V-JEPA grid에서 얼마나 줄어드는지 확인하기 좋습니다.
단, `--autogaze-target-scales`의 scale 개수는 AutoGaze checkpoint의 `config.scales`와 맞아야 합니다.

```bash
python -m repro.vjepa_qwen_runner \
  --video inputs/hlvid_example/clip_av_video_5_001.mp4 \
  --autogaze-mode on \
  --autogaze-model weight/AutoGaze \
  --vjepa-model weight/vjepa2-vitl-fpc64-256 \
  --qwen-model weight/Qwen2.5-VL-3B-Instruct \
  --device cuda \
  --num-video-frames 16 \
  --frames-per-clip 16 \
  --autogaze-tile-size 224 \
  --autogaze-target-scales 32+64+112+224 \
  --vjepa-selection-policy scale_aware_multi_pass \
  --output-json outputs/autogaze_vjepa/vjepa_qwen_scale_aware_actual.json
```

결과에서 우선 볼 값:

- `tokens.autogaze_raw_patch_tokens`: AutoGaze가 후보로 본 multiscale patch 수
- `tokens.autogaze_selected_patch_tokens`: AutoGaze가 실제 선택한 patch 수
- `tokens.vjepa_raw_tokens`: V-JEPA가 dense로 처리했을 token 수
- `tokens.vjepa_selected_tokens`: AutoGaze 선택을 V-JEPA grid/tubelet으로 옮긴 뒤 실제 sparse encoder에 들어간 token 수
- `tokens.qwen_visual_tokens_inserted`: Qwen context에 삽입된 visual token 수
- `latency_ms.autogaze_selector_total`, `latency_ms.vjepa_sparse_encode`, `latency_ms.qwen_generate`

## HLVid Wrapper

HLVid manifest와 mp4 root가 있으면 같은 runner를 여러 row에 대해 반복 실행할 수 있습니다.

```bash
python -m repro.vjepa_qwen_hlvid_benchmark \
  --manifest /data/HLVid/test.jsonl \
  --video-root /data/HLVid/videos \
  --output-dir outputs/autogaze_vjepa/hlvid_limit3 \
  --limit 3 \
  --continue-on-error \
  --autogaze-model weight/AutoGaze \
  --vjepa-model weight/vjepa2-vitl-fpc64-256 \
  --qwen-model weight/Qwen2.5-VL-3B-Instruct \
  --device cuda \
  --dtype float16 \
  --num-video-frames 16 \
  --frames-per-clip 16 \
  --autogaze-chunk-frames 16 \
  --max-tiles-video 1 \
  --autogaze-tile-size 224 \
  --autogaze-target-scales 32+64+112+224 \
  --video-decode-strategy seek \
  --video-resize-longest-edge 448 \
  --vjepa-qwen-modes dense_off,autogaze_single_grid,autogaze_scale_aware
```

생성 파일:

- `vjepa_qwen_hlvid_predictions.jsonl`: row별 raw output, failure stage, token/latency
- `vjepa_qwen_hlvid_scored.jsonl`: HLVid answer parsing 결과
- `vjepa_qwen_hlvid_summary.json`: aggregate summary
- `vjepa_qwen_hlvid_report.md`: policy별 요약 markdown

모드 의미:

- `dense_off`: AutoGaze를 건너뛰고 V-JEPA token 전체를 Qwen bridge로 보냅니다.
- `autogaze_single_grid`: AutoGaze multiscale bbox를 하나의 V-JEPA crop grid로 union 매핑합니다.
- `autogaze_scale_aware`: AutoGaze scale별로 V-JEPA sparse pass를 분리해서 실행합니다.

## 로컬 실행

```bash
.venv/bin/python -m repro.vjepa_poc \
  --synthetic \
  --scale-aware \
  --tiny-encoder-smoke \
  --qwen-bridge-smoke \
  --output-json outputs/autogaze_vjepa/local_smoke.json \
  --output-md outputs/autogaze_vjepa/local_smoke.md
```

예상 요약:

```text
single-grid raw_token_count      = 392
single-grid selected_token_count = 6
scale-aware raw_token_count      = 490
scale-aware selected_token_count = 3
tiny sparse encoder status       = passed
Qwen bridge smoke status         = passed
```

## Colab CUDA 검증 셀

Colab에서는 먼저 GPU 런타임을 켠 뒤 브랜치를 clone합니다. 이 경로는 실제 V-JEPA checkpoint와 Qwen checkpoint를 로드하고, selected V-JEPA sparse feature를 Qwen `inputs_embeds`에 삽입해서 `generate`까지 호출합니다.

첫 셀은 CUDA 확인입니다. 실패하면 Colab 메뉴에서 `Runtime > Change runtime type > GPU`로 바꾼 뒤 다시 실행합니다.

```python
import torch

assert torch.cuda.is_available(), "CUDA is required. Enable a Colab GPU runtime first."
print(torch.cuda.get_device_name(0))
```

```python
import os, subprocess, textwrap, json, pathlib

repo_dir = pathlib.Path("/content/AutoGaze")
if not repo_dir.exists():
    subprocess.check_call([
        "git", "clone",
        "--branch", "codex/autogaze-vjepa",
        "https://github.com/manricheon/AutoGaze.git",
        str(repo_dir),
    ])
os.chdir(repo_dir)
print("cwd:", os.getcwd())
```

```python
import subprocess

subprocess.check_call([
    "python", "-m", "pip", "install", "-q",
    "-r", "requirements-repro.txt",
    "transformers>=4.57.0",
])
```

```python
import subprocess

subprocess.check_call([
    "python", "-m", "pytest",
    "tests/test_vjepa_mapping.py",
    "tests/test_vjepa_sparse_runtime.py",
    "tests/test_vjepa_poc.py",
    "tests/test_vjepa_qwen_bridge.py",
    "tests/test_vjepa_qwen_colab_smoke.py",
    "tests/test_vjepa_qwen_runner.py",
    "tests/test_vjepa_qwen_hlvid_benchmark.py",
    "-q",
])
```

```python
import json, pathlib, subprocess

out_dir = pathlib.Path("/content/autogaze_vjepa_outputs")
out_dir.mkdir(parents=True, exist_ok=True)

subprocess.check_call([
    "python", "-m", "repro.vjepa_poc",
    "--synthetic",
    "--scale-aware",
    "--tiny-encoder-smoke",
    "--qwen-bridge-smoke",
    "--output-json", str(out_dir / "vjepa_mapping_qwen_bridge_probe.json"),
    "--output-md", str(out_dir / "vjepa_mapping_qwen_bridge_probe.md"),
])

payload = json.loads((out_dir / "vjepa_mapping_qwen_bridge_probe.json").read_text())
print(json.dumps({
    "implementation_status": payload["implementation_status"],
    "single_grid": {
        "raw": payload["vjepa"]["raw_token_count"],
        "selected": payload["vjepa"]["selected_token_count"],
        "reduction": payload["vjepa"]["reduction_ratio"],
    },
    "scale_aware": {
        "raw": payload["scale_aware_vjepa"]["raw_token_count"],
        "selected": payload["scale_aware_vjepa"]["selected_token_count"],
        "reduction": payload["scale_aware_vjepa"]["reduction_ratio"],
    },
    "tiny_encoder": payload["vjepa_sparse_encoder_smoke"],
    "qwen_bridge": payload["vjepa_qwen_bridge_smoke"]["bridge_metadata"],
}, indent=2))
```

체크포인트를 명시적으로 다운로드합니다. 기본 모델은 `nvidia/AutoGaze`, `facebook/vjepa2-vitl-fpc64-256`, `Qwen/Qwen2.5-VL-3B-Instruct`입니다.

```python
import json, pathlib, subprocess

weights = pathlib.Path("/content/autogaze_weights")
subprocess.check_call([
    "python", "scripts/download_vjepa_qwen_checkpoints.py",
    "--output-root", str(weights),
    "--autogaze-model", "nvidia/AutoGaze",
    "--vjepa-model", "facebook/vjepa2-vitl-fpc64-256",
    "--qwen-model", "Qwen/Qwen2.5-VL-3B-Instruct",
])

print("downloaded:", weights)
```

실제 CUDA generate smoke입니다. 이 smoke는 정확도 벤치마크가 아니라, V-JEPA sparse encoder output이 Qwen `generate`까지 연결되는지 확인합니다.

```python
import json, pathlib, subprocess

weights = pathlib.Path("/content/autogaze_weights")
out_dir = pathlib.Path("/content/autogaze_vjepa_outputs")
out_dir.mkdir(parents=True, exist_ok=True)

subprocess.check_call([
    "python", "-m", "repro.vjepa_qwen_colab_smoke",
    "--require-cuda",
    "--device", "cuda",
    "--dtype", "float16",
    "--frames-per-clip", "4",
    "--crop-size", "224",
    "--vjepa-model", str(weights / "facebook__vjepa2-vitl-fpc64-256"),
    "--qwen-model", str(weights / "Qwen__Qwen2.5-VL-3B-Instruct"),
    "--output-json", str(out_dir / "colab_vjepa_qwen_cuda_smoke.json"),
])

payload = json.loads((out_dir / "colab_vjepa_qwen_cuda_smoke.json").read_text())
print(json.dumps({
    "status": payload["status"],
    "integration_level": payload["integration_level"],
    "device": payload["device"],
    "tokens": payload["tokens"],
    "generated_text": payload["generated_text"],
}, indent=2))
```

실제 AutoGaze selector까지 포함한 end-to-end smoke는 아래 셀을 사용합니다.

```python
import json, pathlib, subprocess

weights = pathlib.Path("/content/autogaze_weights")
out_dir = pathlib.Path("/content/autogaze_vjepa_outputs")
video = pathlib.Path("/content/AutoGaze/inputs/hlvid_example/clip_av_video_5_001.mp4")

subprocess.check_call([
    "python", "-m", "repro.vjepa_qwen_runner",
    "--video", str(video),
    "--prompt", "Describe the video in one short sentence.",
    "--autogaze-mode", "on",
    "--autogaze-model", str(weights / "nvidia__AutoGaze"),
    "--vjepa-model", str(weights / "facebook__vjepa2-vitl-fpc64-256"),
    "--qwen-model", str(weights / "Qwen__Qwen2.5-VL-3B-Instruct"),
    "--require-cuda",
    "--device", "cuda",
    "--dtype", "float16",
    "--num-video-frames", "16",
    "--frames-per-clip", "16",
    "--autogaze-chunk-frames", "16",
    "--max-tiles-video", "1",
    "--autogaze-tile-size", "224",
    "--autogaze-target-scales", "32+64+112+224",
    "--video-decode-strategy", "seek",
    "--video-resize-longest-edge", "448",
    "--output-json", str(out_dir / "actual_autogaze_vjepa_qwen.json"),
    "--output-md", str(out_dir / "actual_autogaze_vjepa_qwen.md"),
])

payload = json.loads((out_dir / "actual_autogaze_vjepa_qwen.json").read_text())
print(json.dumps({
    "status": payload["status"],
    "autogaze_mode": payload["autogaze_mode"],
    "tokens": payload["tokens"],
    "latency_ms": payload["latency_ms"],
    "generated_text": payload["generated_text"],
}, indent=2))
```

동일 bridge에서 AutoGaze off dense baseline도 바로 비교할 수 있습니다.

```python
subprocess.check_call([
    "python", "-m", "repro.vjepa_qwen_runner",
    "--video", str(video),
    "--prompt", "Describe the video in one short sentence.",
    "--autogaze-mode", "off",
    "--vjepa-model", str(weights / "facebook__vjepa2-vitl-fpc64-256"),
    "--qwen-model", str(weights / "Qwen__Qwen2.5-VL-3B-Instruct"),
    "--require-cuda",
    "--device", "cuda",
    "--dtype", "float16",
    "--frames-per-clip", "16",
    "--video-decode-strategy", "seek",
    "--video-resize-longest-edge", "448",
    "--output-json", str(out_dir / "actual_vjepa_qwen_dense_off.json"),
    "--output-md", str(out_dir / "actual_vjepa_qwen_dense_off.md"),
])
```

HLVid wrapper까지 확인하려면 최소 manifest 하나를 만들고 같은 비디오/질문으로 `dense_off`와 AutoGaze on 모드를 함께 실행합니다. 실제 HLVid manifest도 같은 schema를 사용하므로, CUDA 머신에서는 `manifest`와 `video-root`만 실제 경로로 바꾸면 됩니다.

```python
import json, pathlib, subprocess

weights = pathlib.Path("/content/autogaze_weights")
out_dir = pathlib.Path("/content/autogaze_vjepa_outputs")
video = pathlib.Path("/content/AutoGaze/inputs/hlvid_example/clip_av_video_5_001.mp4")
mini = out_dir / "mini_hlvid_vjepa_qwen.jsonl"
mini.write_text(json.dumps({
    "question_id": "mini-001",
    "category": "smoke",
    "video_path": video.name,
    "question": "Describe the video in one short sentence.",
    "answer": "unknown",
}) + "\n")

subprocess.check_call([
    "python", "-m", "repro.vjepa_qwen_hlvid_benchmark",
    "--manifest", str(mini),
    "--video-root", str(video.parent),
    "--output-dir", str(out_dir / "mini_hlvid_vjepa_qwen"),
    "--limit", "1",
    "--continue-on-error",
    "--autogaze-model", str(weights / "nvidia__AutoGaze"),
    "--vjepa-model", str(weights / "facebook__vjepa2-vitl-fpc64-256"),
    "--qwen-model", str(weights / "Qwen__Qwen2.5-VL-3B-Instruct"),
    "--device", "cuda",
    "--dtype", "float16",
    "--num-video-frames", "16",
    "--frames-per-clip", "16",
    "--autogaze-chunk-frames", "16",
    "--max-tiles-video", "1",
    "--autogaze-tile-size", "224",
    "--autogaze-target-scales", "32+64+112+224",
    "--video-decode-strategy", "seek",
    "--video-resize-longest-edge", "448",
    "--vjepa-qwen-modes", "dense_off,autogaze_single_grid",
])

summary = json.loads((out_dir / "mini_hlvid_vjepa_qwen" / "vjepa_qwen_hlvid_summary.json").read_text())
print(json.dumps(summary, indent=2))
```

## Colab에서 NVILA/Qwen 계열 스크립트 같이 확인

V-JEPA PoC와 별개로, 기존 NVILA-HD native AutoGaze on/off와 Qwen plugin route도 같은 Colab 런타임에서 CLI가 깨지지 않는지 확인합니다. 아래 명령은 실제 모델 weight가 있는 경우 바로 실행하고, weight가 아직 없으면 `--help` 검증부터 먼저 합니다.

NVILA-HD single on/off:

```bash
python -m repro.nvila_runner --mode single \
  --video /content/AutoGaze/inputs/hlvid_example/clip_av_video_5_001.mp4 \
  --model-path /content/autogaze_weights/nvidia__NVILA-8B-HD-Video \
  --autogaze-model /content/autogaze_weights/nvidia__AutoGaze \
  --device cuda \
  --dtype float16 \
  --gazing-mode autogaze \
  --num-video-frames 16 \
  --num-video-frames-thumbnail 8 \
  --max-tiles-video 1 \
  --video-decode-strategy seek \
  --video-resize-longest-edge 720 \
  --max-batch-size-autogaze 4 \
  --max-batch-size-siglip 4 \
  --max-new-tokens 8 \
  --output-json /content/autogaze_vjepa_outputs/nvila_single_autogaze.json \
  --summary-json /content/autogaze_vjepa_outputs/nvila_single_autogaze_summary.json \
  --print-summary
```

```bash
python -m repro.nvila_runner --mode single \
  --video /content/AutoGaze/inputs/hlvid_example/clip_av_video_5_001.mp4 \
  --model-path /content/autogaze_weights/nvidia__NVILA-8B-HD-Video \
  --device cuda \
  --dtype float16 \
  --gazing-mode keep-all-single \
  --num-video-frames 16 \
  --num-video-frames-thumbnail 8 \
  --max-tiles-video 1 \
  --video-decode-strategy seek \
  --video-resize-longest-edge 720 \
  --max-batch-size-siglip 4 \
  --max-new-tokens 8 \
  --output-json /content/autogaze_vjepa_outputs/nvila_single_keep_all_single.json \
  --summary-json /content/autogaze_vjepa_outputs/nvila_single_keep_all_single_summary.json \
  --print-summary
```

Qwen plugin HLVid route:

```bash
python scripts/run_hlvid_folder_benchmark.py \
  --plugin-suite qwen \
  --manifest /data/HLVid/test.jsonl \
  --video-root /data/HLVid/videos \
  --output-dir /content/autogaze_vjepa_outputs/plugin_qwen_hlvid_limit3 \
  --plugin-model qwen3-vl=/content/autogaze_weights/Qwen__Qwen2.5-VL-3B-Instruct \
  --autogaze-model /content/autogaze_weights/nvidia__AutoGaze \
  --limit 3 \
  --continue-on-error \
  --device cuda \
  --dtype float16 \
  --num-video-frames 16 \
  --num-video-frames-thumbnail 0 \
  --max-tiles-video 1 \
  --video-decode-strategy seek \
  --video-resize-longest-edge 448 \
  --autogaze-tile-size 224 \
  --autogaze-target-scales 32+64+112+224 \
  --autogaze-target-patch-size 16 \
  --max-batch-size-autogaze 4 \
  --qwen-vit-chunk-frames 16 \
  --max-new-tokens 8
```

현재 plugin suite의 model override key는 `qwen3-vl`입니다. Qwen2.5 checkpoint를 테스트할 때도 우선 이 key에 로컬 path를 넣습니다. adapter 내부에서 Qwen 계열 processor/grid path를 공유하기 위한 임시 이름이고, 결과 리포트에서는 실제 `model_path`를 함께 확인하세요.

검증 포인트:

- NVILA on/off 결과는 `summary_json`의 key comparison에서 `keep-all-single`과 `autogaze`가 분리되어야 합니다.
- Qwen plugin route는 `qwen_full_vit`, `qwen_chunked_vit`, `qwen_chunked_vit_autogaze_sparse` row를 만들어야 합니다.
- V-JEPA/Qwen PoC는 `tokens.vjepa_raw_tokens > tokens.vjepa_selected_tokens`와 `tokens.qwen_visual_tokens_inserted == tokens.vjepa_selected_tokens`를 우선 확인합니다.
- OOM이나 dependency error는 benchmark row의 `failure.kind`와 `failure.stage`에 남아야 합니다.

## 실제 AutoGaze output 연결

AutoGaze selector가 `SparseSelectionPlan` JSON을 만든 뒤에는 synthetic 대신 아래처럼 실행합니다.

```bash
python -m repro.vjepa_poc \
  --sparse-selection-plan-json /content/autogaze_selector_plan.json \
  --frames-per-clip 16 \
  --tubelet-size 2 \
  --crop-size 224 \
  --patch-size 16 \
  --scale-aware \
  --tiny-encoder-smoke \
  --qwen-bridge-smoke \
  --output-json /content/autogaze_vjepa_outputs/real_plan_vjepa_probe.json \
  --output-md /content/autogaze_vjepa_outputs/real_plan_vjepa_probe.md
```

## 해석 기준

- `vjepa.raw_token_count`: single-grid V-JEPA가 원래 처리할 token 수입니다.
- `vjepa.selected_token_count`: AutoGaze 선택을 V-JEPA grid로 옮긴 뒤 남은 token 수입니다.
- `scale_aware_vjepa.selected_token_count`: scale별 V-JEPA pass 후보에서 남은 token 수입니다.
- `vjepa_sparse_encoder_smoke.status=passed`: selected hidden states와 원래 position index를 사용해 V-JEPA encoder layer 호출이 가능한 상태입니다.
- `vjepa_qwen_bridge_smoke.status=passed`: Qwen visual placeholder에 V-JEPA feature를 삽입하는 wiring이 동작한 상태입니다.
- `repro.vjepa_qwen_colab_smoke`의 `status=passed`: 실제 V-JEPA checkpoint와 실제 Qwen checkpoint가 CUDA에서 `generate`까지 호출된 상태입니다.
- `repro.vjepa_qwen_runner`의 `status=passed`: `autogaze_mode=off`는 dense V-JEPA keep-all baseline, `autogaze_mode=on`은 실제 AutoGaze selector output이 V-JEPA sparse encoder와 Qwen generate까지 이어진 상태입니다.
- 이 PoC는 Qwen 정확도를 주장하지 않습니다. 현재 bridge는 deterministic repeat/truncate projection이므로, 정확도 주장을 하려면 V-JEPA feature를 Qwen visual embedding space로 맞추는 학습 projector가 필요합니다.
