#!/usr/bin/env python3
"""
NVILA-8B-HD-Video 추론 테스트 스크립트
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Usage:
  python scripts/test_nvila.py [VIDEO_PATH] [--question "질문"] [--frames N]

Arguments:
  VIDEO_PATH        비디오 파일 경로 (기본: assets/example_input.mp4)
  --question        단일 질문 (기본: 3가지 예제 질문)
  --frames N        샘플링 프레임 수, AutoGaze max_num_frames의 배수 (기본: 16)
  --model-path PATH NVILA 가중치 경로 (기본: weights/NVILA-8B-HD-Video)
  --autogaze-path P AutoGaze 가중치 경로 (기본: weights/AutoGaze)

사전 요건:
  bash scripts/download_models.sh weights nvila     # NVILA 다운로드 (~16 GB)
  bash scripts/download_models.sh weights autogaze  # AutoGaze 다운로드 (~50 MB)
  pip install opencv-python-headless einops accelerate

메모리 요건:
  CUDA  : ≥ 20 GB VRAM (bfloat16 기준) — A100/H100 권장
  MPS   : ≥ 24 GB Unified Memory (M1 Max/Ultra, M2 Ultra 등)
  CPU   : ≥ 32 GB RAM (매우 느림)

참고 — processing_nvila.py 패치:
  weights/NVILA-8B-HD-Video/processing_nvila.py 에 아래 두 가지 패치가 이미 적용됨:
  1. AutoGaze device: 하드코딩된 "cuda" → cuda/mps/cpu 자동 감지
  2. num_video_frames: 8 → 16 (AutoGaze max_num_frames 배수 요건 충족)
  HuggingFace 모듈 캐시(~/.cache/huggingface/modules/...)에도 동일하게 적용 필요.
"""

import argparse
import time
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore", category=FutureWarning)

# ── 경로 기본값 ───────────────────────────────────────────────────
REPO_ROOT     = Path(__file__).resolve().parent.parent
DEFAULT_MODEL  = REPO_ROOT / "weights" / "NVILA-8B-HD-Video"
DEFAULT_AG     = REPO_ROOT / "weights" / "AutoGaze"
DEFAULT_VIDEO  = REPO_ROOT / "assets" / "example_input.mp4"

# ── 인수 파싱 ─────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="NVILA-8B-HD-Video 비디오 질의응답 테스트")
parser.add_argument("video",          nargs="?", default=str(DEFAULT_VIDEO))
parser.add_argument("--question",     default=None,             help="단일 질문 (없으면 예제 3가지 실행)")
parser.add_argument("--frames",       type=int, default=16,     help="샘플링 프레임 수 (16의 배수)")
parser.add_argument("--model-path",   default=str(DEFAULT_MODEL))
parser.add_argument("--autogaze-path",default=str(DEFAULT_AG))
parser.add_argument("--max-new-tokens", type=int, default=256)
args = parser.parse_args()

video_path   = args.video
model_path   = args.model_path
ag_path      = args.autogaze_path
n_frames     = args.frames
max_new_tok  = args.max_new_tokens

assert Path(video_path).exists(),   f"비디오 없음: {video_path}"
assert Path(model_path).exists(),   f"NVILA 가중치 없음: {model_path}\n  → bash scripts/download_models.sh weights nvila"
assert Path(ag_path).exists(),      f"AutoGaze 가중치 없음: {ag_path}\n  → bash scripts/download_models.sh weights autogaze"
assert n_frames % 16 == 0 and n_frames >= 16, "--frames 는 16의 배수여야 합니다 (예: 16, 32, 64)"

questions = (
    [args.question] if args.question
    else [
        "이 비디오에서 무엇이 일어나고 있나요? 구체적으로 설명해 주세요.",
        "비디오에서 사람이 어떤 동작을 하고 있나요?",
        "어떤 환경(장소, 조명 등)에서 촬영된 영상인가요?",
    ]
)

# ── 디바이스 ──────────────────────────────────────────────────────
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
dtype = torch.bfloat16

print("=" * 60)
print("NVILA-8B-HD-Video 추론 테스트")
print("=" * 60)
print(f"디바이스     : {device}  ({dtype})")
print(f"비디오       : {video_path}")
print(f"NVILA 경로   : {model_path}")
print(f"AutoGaze 경로: {ag_path}")
print(f"프레임 수    : {n_frames}")
print()

# ── 재귀 디바이스 이동 헬퍼 ─────────────────────────────────────
def _to_device(v):
    """텐서·리스트·딕셔너리를 재귀적으로 device로 이동."""
    if isinstance(v, torch.Tensor):
        return v.to(device)
    if isinstance(v, list):
        return [_to_device(x) for x in v]
    if isinstance(v, dict):
        return {k: _to_device(vv) for k, vv in v.items()}
    return v


# ── 1. 프로세서 로드 ──────────────────────────────────────────────
print("[1/3] 프로세서 로드 중 ...")
t0 = time.perf_counter()

from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained(
    model_path,
    trust_remote_code=True,
    autogaze_model_id=ag_path,   # 로컬 AutoGaze 가중치 사용
)

# num_video_frames를 실행 시점에 덮어쓰기 (패치 없이도 동작)
processor.num_video_frames = n_frames

t_proc = time.perf_counter() - t0
video_token = processor.tokenizer.video_token   # '<vila/video>'
print(f"  완료 ({t_proc:.1f}s)")
print(f"  프레임: {processor.num_video_frames} tile + {processor.num_video_frames_thumbnail} thumbnail")
print(f"  gazing_ratio: tile={processor.gazing_ratio_tile}, thumb={processor.gazing_ratio_thumbnail}")
print(f"  비디오 토큰: {repr(video_token)}")
print()

# ── 2. 모델 로드 ──────────────────────────────────────────────────
print("[2/3] NVILA 모델 로드 중 (~16 GB) ...")
t0 = time.perf_counter()

from transformers import AutoModel
model = AutoModel.from_pretrained(
    model_path,
    trust_remote_code=True,
    dtype=dtype,
    device_map="auto",          # GPU VRAM / Unified Memory에 자동 분산
)
model.eval()

t_model = time.perf_counter() - t0
n_params = sum(p.numel() for p in model.parameters()) / 1e9
print(f"  완료 ({t_model:.1f}s)  파라미터: {n_params:.2f}B")
print()

# ── 3. 질의응답 루프 ──────────────────────────────────────────────
print("[3/3] 비디오 질의응답")
print("=" * 60)

for qi, question in enumerate(questions):
    print(f"\n[Q{qi + 1}] {question}")
    print("-" * 55)

    # NVILA 텍스트 형식: 비디오 토큰 플레이스홀더 + 질문
    # processor 내부에서 <vila/video>를 실제 vision token 수만큼 확장함
    prompt = f"{video_token}\n{question}"

    t0 = time.perf_counter()
    inputs = processor(
        text=prompt,
        videos=video_path,      # 문자열 경로 → processor 내에서 프레임 추출 + AutoGaze 실행
    )
    t_prep = time.perf_counter() - t0

    # 모든 값을 device로 이동 (list → tensor 변환 포함)
    inputs_dev = _to_device(dict(inputs))
    for key in ("input_ids", "attention_mask"):
        if key in inputs_dev and isinstance(inputs_dev[key], list):
            inputs_dev[key] = torch.tensor(inputs_dev[key], device=device)

    input_ids   = inputs_dev.pop("input_ids")
    extra_kwargs = inputs_dev

    t0 = time.perf_counter()
    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tok,
            do_sample=False,
            temperature=None,
            top_p=None,
            **extra_kwargs,
        )
    t_gen = time.perf_counter() - t0

    new_ids = generated_ids[:, input_ids.shape[1]:]
    answer  = processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
    n_tok   = new_ids.shape[1]

    print(f"[A{qi + 1}] {answer}")
    print()
    print(f"  전처리: {t_prep:.1f}s  |  생성: {t_gen:.1f}s  |  "
          f"{n_tok}토큰  |  {n_tok / max(t_gen, 1e-3):.1f} tok/s")

print()
print("=" * 60)
print("완료 ✓")
