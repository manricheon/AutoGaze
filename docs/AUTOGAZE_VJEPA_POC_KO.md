# AutoGaze + V-JEPA + Qwen PoC

이 문서는 AutoGaze가 선택한 multiscale patch를 V-JEPA token index로 옮기는 PoC를 다룹니다. 목표는 아직 성능 주장이 아니라, `AutoGaze patch index -> V-JEPA tubelet/grid index -> sparse encoder hook -> Qwen bridge 후보`가 구조적으로 가능한지 확인하는 것입니다.

## 지원 모드

| 모드 | 목적 | 정확도 주장 |
|---|---|---|
| single-grid mapping | AutoGaze bbox를 하나의 V-JEPA crop/grid에 overlap 매핑 | 안 함 |
| scale-aware mapping | AutoGaze scale별로 V-JEPA pass를 분리한 token packing 후보 검증 | 안 함 |
| tiny sparse encoder smoke | random-weight tiny V-JEPA encoder에 selected hidden states만 통과 | 안 함 |
| Qwen bridge smoke | selected V-JEPA feature를 Qwen visual placeholder embedding으로 삽입 | 안 함 |

## 핵심 정책

- `tubelet_size=2`이면 같은 tubelet에 들어가는 두 프레임의 선택 patch를 union합니다.
- 저해상도 AutoGaze patch는 V-JEPA grid에서 bbox overlap되는 모든 spatial cell로 확장합니다.
- scale-aware 모드에서는 scale마다 별도 V-JEPA grid를 두므로 coarse patch가 고해상도 grid로 과도하게 펼쳐지는지 비교할 수 있습니다.
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
  -> AutoGaze scale 112 selected patches -> V-JEPA 112 grid selected tokens
  -> AutoGaze scale 224 selected patches -> V-JEPA 224 grid selected tokens
  -> concat scale-aware features with scale/frame/row/col metadata
  -> Qwen bridge candidate
```

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
    "transformers>=4.57.0",
    "accelerate",
    "huggingface_hub",
    "qwen-vl-utils",
    "av",
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

체크포인트를 명시적으로 다운로드합니다. 기본 모델은 `facebook/vjepa2-vitl-fpc64-256`과 `Qwen/Qwen2.5-VL-3B-Instruct`입니다.

```python
import json, pathlib, subprocess

weights = pathlib.Path("/content/autogaze_weights")
subprocess.check_call([
    "python", "scripts/download_vjepa_qwen_checkpoints.py",
    "--output-root", str(weights),
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
- 이 PoC는 Qwen 정확도를 주장하지 않습니다. 현재 bridge는 deterministic repeat/truncate projection이므로, 정확도 주장을 하려면 V-JEPA feature를 Qwen visual embedding space로 맞추는 학습 projector가 필요합니다.
