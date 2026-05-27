# 외부 CUDA 플랫폼 검증 리포트

작성일: 2026-05-28  
대상 브랜치: `codex/autogaze-repro`, `codex/autogaze-vjepa`  
검증 커밋: `cfea645 Fix VJEPA embedding axis handling for Colab`

## 결론

Colab, Kaggle, Hugging Face Jobs를 CUDA 검증 플랫폼 후보로 직접 확인했습니다. 세 플랫폼 모두 페이지/API 접근은 확인했지만, 현재 계정/런타임 상태 때문에 실제 CUDA generate 재실행은 완료하지 못했습니다.

대신 Colab에 남아 있던 CUDA 실패 로그를 확인해 실제 코드 버그를 수정했고, 로컬에서 가능한 entrypoint/dry-run/unit/full test 검증은 모두 통과했습니다. CUDA 머신, Colab quota 회복 런타임, phone-verified Kaggle 런타임, 또는 credit이 있는 HF Jobs에서는 아래 재실행 명령으로 같은 검증을 이어가면 됩니다.

## 외부 플랫폼 접근 상태

| 플랫폼 | 접근 | CUDA 실행 상태 | 막힌 이유 |
| --- | --- |
| Colab | 가능 | 실패 | Colab 사용량 제한으로 GPU 백엔드 연결 불가 |
| Kaggle `manricheon02/autogaze` | 가능 | 실패 | 계정 phone verification 전이라 GPU/Internet 기능 잠김 |
| Kaggle `manricheon/autogaze` | 실패 | 불가 | 현재 Chrome 로그인 세션에서 해당 notebook을 찾을 수 없음 |
| Hugging Face Jobs | API 접근 가능 | 실패 | 계정 pre-paid credit 부족으로 GPU job 생성 402 |

Colab 증거 스크린샷:

![Colab GPU backend blocked](assets/colab_gpu_backend_blocked_2026-05-28.png)

Kaggle 증거 스크린샷:

![Kaggle GPU locked](assets/kaggle_gpu_locked_2026-05-28.png)

Hugging Face Jobs GPU smoke 요청 결과:

```text
Error executing run: API request failed: 402 Payment Required
Server response:
{
  "error": "Pre-paid credit balance is insufficient - add more credits to your account to use Jobs."
}
```

## Colab에서 확인한 기존 실패

열려 있던 Colab 출력에는 `repro.vjepa_qwen_colab_smoke` 실행 결과가 남아 있었고, 상태는 `failed`였습니다.

핵심 에러:

```text
RuntimeError: Given groups=1, weight of size [1024, 3, 2, 16, 16],
expected input[1, 4, 3, 224, 224] to have 3 channels, but got 4 channels instead
```

해석:

- V-JEPA Conv3D patch embedding은 일부 런타임/remote-code 경로에서 `[B, C, T, H, W]` 입력을 직접 기대합니다.
- 기존 smoke path는 `[B, T, C, H, W]`를 그대로 넘길 수 있어 Colab에서 `C=4`로 오해되었습니다.
- 반면 일반 Transformers V-JEPA wrapper는 `[B, T, C, H, W]`를 받고 내부에서 `[B, C, T, H, W]`로 바꾸는 경로가 있습니다.
- 따라서 두 형태를 모두 지원하도록 embedding module 타입에 따라 축 순서를 정규화해야 합니다.

## 반영한 수정

수정 커밋: `cfea645`

| 파일 | 변경 |
| --- | --- |
| `repro/vjepa_qwen_colab_smoke.py` | V-JEPA embedding module이 wrapper인지 direct patch embedder인지 감지해 입력 축 순서를 자동 정규화 |
| `repro/vjepa_qwen_runner.py` | 실제 비디오 runner에서도 같은 축 정규화 경로 사용 |
| `tests/test_vjepa_qwen_colab_smoke.py` | direct patch embedding layout과 wrapper embedding layout을 모두 테스트 |
| `tests/test_vjepa_qwen_runner.py` | runner patch embedding boundary가 두 입력 경로를 모두 처리하는지 테스트 |

브랜치 반영:

| 브랜치 | 상태 |
| --- | --- |
| `codex/autogaze-repro` | push 완료 |
| `codex/autogaze-vjepa` | push 완료 |

## 로컬 검증 결과

CUDA 모델 로드는 로컬 MPS/CPU 환경에서 검증 대상이 아니므로 제외했습니다. 대신 Colab 실패 원인에 해당하는 V-JEPA 입력 경계와 전체 entrypoint route는 검증했습니다.

| 검증 | 결과 |
| --- | --- |
| `pytest tests/test_vjepa_qwen_colab_smoke.py tests/test_vjepa_qwen_runner.py -q` | `16 passed` |
| `pytest tests/test_vjepa_sparse_runtime.py tests/test_vjepa_poc.py tests/test_vjepa_qwen_bridge.py -q` | `11 passed` |
| `scripts/verify_autogaze_entrypoints.py --run-pytest` | `passed=true`, `check_count=26`, `command_count=20` |
| full pytest | `404 passed` |
| `git diff --check` | 통과 |
| 공식/upstream 문서 diff | 없음 |

## CUDA 재검증 명령

GPU 백엔드가 연결되는 Kaggle/Colab 또는 CUDA 머신에서 아래 순서로 다시 실행합니다.

### 1. 최신 코드 받기

```bash
cd /kaggle/working  # Kaggle
# cd /content       # Colab
test -d AutoGaze || git clone --branch codex/autogaze-vjepa https://github.com/manricheon/AutoGaze.git AutoGaze
cd AutoGaze
git fetch origin codex/autogaze-vjepa codex/autogaze-repro
git checkout codex/autogaze-vjepa
git pull --ff-only origin codex/autogaze-vjepa
git log --oneline -1
```

기대 커밋:

```text
cfea645 Fix VJEPA embedding axis handling for Colab
```

### 2. 사전 검증

```bash
python scripts/verify_autogaze_entrypoints.py \
  --output-json /kaggle/working/autogaze_vjepa_outputs/entrypoint_verification.json \
  --output-md /kaggle/working/autogaze_vjepa_outputs/entrypoint_verification.md
```

기대:

```text
passed=true
failed_command_count=0
failed_check_count=0
```

### 3. Colab CUDA smoke wrapper

```bash
python scripts/run_colab_autogaze_cuda_smoke.py \
  --weights-root /kaggle/working/autogaze_weights \
  --output-root /kaggle/working/autogaze_vjepa_outputs \
  --video inputs/hlvid_example/clip_av_video_5_001.mp4 \
  --num-video-frames 16 \
  --frames-per-clip 16 \
  --video-resize-longest-edge 224 \
  --max-new-tokens 4
```

기대:

- `results.vjepa_qwen_dense_off.status=passed`
- `results.autogaze_vjepa_qwen_on.status=passed`
- `tokens.vjepa_selected_tokens < tokens.vjepa_raw_tokens`
- `tokens.qwen_visual_tokens_inserted == tokens.vjepa_selected_tokens`
- `/kaggle/working/autogaze_vjepa_outputs/colab_verification.md` 생성
- `/kaggle/working/autogaze_vjepa_outputs/visualizations/` 아래 selected frame, V-JEPA mask, AutoGaze overlay 생성

### 4. 직접 실행 결과를 묶어 리포트 생성

wrapper 대신 기존 runner를 직접 실행했다면 마지막에 아래 명령으로 `colab_verification.md`를 생성합니다.

```bash
python -m repro.colab_verification_report \
  --output-md /kaggle/working/autogaze_vjepa_outputs/colab_verification.md \
  --title "AutoGaze Kaggle/Colab Verification" \
  --video inputs/hlvid_example/clip_av_video_5_001.mp4 \
  --query "Describe the video in one short sentence." \
  --entrypoint-verification-json /kaggle/working/autogaze_vjepa_outputs/entrypoint_verification.json \
  --case vjepa_qwen_dense_off=/kaggle/working/autogaze_vjepa_outputs/vjepa_qwen_dense_off_cuda_smoke.json \
  --case autogaze_vjepa_qwen_on=/kaggle/working/autogaze_vjepa_outputs/autogaze_vjepa_qwen_on_cuda_smoke.json
```

### 5. Kaggle notebook cell

Kaggle notebook에는 아래 셀 하나를 넣고 Run All을 누르면 됩니다. 단, 우측 `Session options`에서 phone verification 이후 GPU와 Internet이 활성화되어 있어야 합니다.

```python
import os, pathlib, subprocess, sys, json

def run(cmd):
    print("\n$", " ".join(map(str, cmd)))
    subprocess.check_call(list(map(str, cmd)))

root = pathlib.Path("/kaggle/working")
repo = root / "AutoGaze"
out = root / "autogaze_vjepa_outputs"
weights = root / "autogaze_weights"
out.mkdir(parents=True, exist_ok=True)

if not repo.exists():
    run(["git", "clone", "--branch", "codex/autogaze-vjepa", "https://github.com/manricheon/AutoGaze.git", repo])
os.chdir(repo)
run(["git", "fetch", "origin", "codex/autogaze-vjepa"])
run(["git", "checkout", "codex/autogaze-vjepa"])
run(["git", "pull", "--ff-only", "origin", "codex/autogaze-vjepa"])
run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-repro.txt", "transformers>=4.57.0", "qwen-vl-utils", "av", "pytest"])
run([sys.executable, "scripts/verify_autogaze_entrypoints.py", "--output-json", out / "entrypoint_verification.json", "--output-md", out / "entrypoint_verification.md"])
run([
    sys.executable, "scripts/run_colab_autogaze_cuda_smoke.py",
    "--weights-root", weights,
    "--output-root", out,
    "--video", "inputs/hlvid_example/clip_av_video_5_001.mp4",
    "--num-video-frames", "16",
    "--frames-per-clip", "16",
    "--video-resize-longest-edge", "224",
    "--max-new-tokens", "4",
])

summary = json.loads((out / "colab_autogaze_cuda_smoke_summary.json").read_text())
print(json.dumps(summary["summary"], indent=2))
print("verification:", out / "colab_verification.md")
```

## 완료/미완료 판단

| 요구사항 | 현재 상태 |
| --- | --- |
| Colab 접근 확인 | 완료 |
| Colab에서 기존 실패 원인 확인 | 완료 |
| Kaggle notebook 접근 확인 | 완료: `manricheon02/autogaze` |
| Kaggle GPU 실행 | 미완료: phone verification 필요 |
| HF Jobs GPU 실행 | 미완료: prepaid credit 부족 |
| V-JEPA + Qwen Colab smoke 코드 수정 | 완료 |
| 로컬 entrypoint/unit/full test 검증 | 완료 |
| 외부 CUDA generate 재실행 | 미완료: 접근 가능한 GPU 플랫폼 필요 |
| `colab_verification.md` 실제 CUDA 결과 생성 | 미완료: GPU 연결 필요 |

현재 상태는 “CUDA 실행 직전까지 코드와 문서, report generator는 준비 완료”입니다. GPU 백엔드가 연결되는 환경에서는 위 명령을 그대로 실행해 실제 CUDA 결과와 visualization artifact를 생성하면 됩니다.
