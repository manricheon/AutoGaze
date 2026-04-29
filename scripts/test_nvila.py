#!/usr/bin/env python3
"""
NVILA-8B-HD-Video 추론 테스트 스크립트
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Usage:
  python scripts/test_nvila.py [VIDEO_PATH] [--question "질문"] [--frames N]
  python scripts/test_nvila.py [VIDEO_PATH] [--question "질문"] [--stride N]

프레임 샘플링 방식 (둘 중 하나 선택):
  --frames N   전체 영상에서 N개를 균일 샘플링 (linspace). 16의 배수.
  --stride N   매 N번째 프레임을 순서대로 추출. 추출된 수를 16 배수로 truncate.
               예: 영상 300프레임, stride=10 → 30개 추출 → 32보다 작으므로 16개 사용

두 방식 모두 동시에 지정하면 비교 모드로 실행합니다.

Arguments:
  VIDEO_PATH        비디오 파일 경로 (기본: assets/example_input.mp4)
  --question        단일 질문 (없으면 예제 3가지 실행)
  --frames N        균일 샘플링 프레임 수 (16의 배수)
  --stride N        매 N번째 프레임 추출 (stride 샘플링)
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

import cv2
import numpy as np
import torch
from PIL import Image

warnings.filterwarnings("ignore", category=FutureWarning)

# ── 경로 기본값 ───────────────────────────────────────────────────
REPO_ROOT     = Path(__file__).resolve().parent.parent
DEFAULT_MODEL  = REPO_ROOT / "weights" / "NVILA-8B-HD-Video"
DEFAULT_AG     = REPO_ROOT / "weights" / "AutoGaze"
DEFAULT_VIDEO  = REPO_ROOT / "assets" / "example_input.mp4"

# ── 인수 파싱 ─────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="NVILA-8B-HD-Video 비디오 질의응답 테스트")
parser.add_argument("video",            nargs="?", default=str(DEFAULT_VIDEO))
parser.add_argument("--question",       default=None,             help="단일 질문 (없으면 예제 3가지 실행)")
parser.add_argument("--frames",         type=int, default=None,   help="균일 샘플링 프레임 수 (16의 배수). --stride와 동시 지정 시 비교 모드")
parser.add_argument("--stride",         type=int, default=None,   help="매 N번째 프레임 추출 (stride 샘플링). --frames와 동시 지정 시 비교 모드")
parser.add_argument("--model-path",     default=str(DEFAULT_MODEL))
parser.add_argument("--autogaze-path",  default=str(DEFAULT_AG))
parser.add_argument("--max-new-tokens", type=int, default=256)
args = parser.parse_args()

video_path   = args.video
model_path   = args.model_path
ag_path      = args.autogaze_path
max_new_tok  = args.max_new_tokens

# --frames / --stride 둘 다 없으면 기본값 16
if args.frames is None and args.stride is None:
    args.frames = 16

assert Path(video_path).exists(),   f"비디오 없음: {video_path}"
assert Path(model_path).exists(),   f"NVILA 가중치 없음: {model_path}\n  → bash scripts/download_models.sh weights nvila"
assert Path(ag_path).exists(),      f"AutoGaze 가중치 없음: {ag_path}\n  → bash scripts/download_models.sh weights autogaze"
if args.frames is not None:
    assert args.frames % 16 == 0 and args.frames >= 16, "--frames 는 16의 배수여야 합니다 (예: 16, 32, 64)"

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


# ── stride 샘플링 헬퍼 ────────────────────────────────────────────
def load_frames_stride(video_path: str, stride: int) -> tuple[list[Image.Image], int]:
    """
    매 stride번째 프레임을 순서대로 추출.
    추출된 프레임 수를 16 배수로 truncate (AutoGaze 요건).
    반환: (PIL 이미지 리스트, 실제 사용 프레임 수)
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = list(range(0, total, stride))

    # 16 배수로 truncate
    n = (len(indices) // 16) * 16
    if n == 0:
        n = 16  # 최소 16개 보장 (부족하면 마지막 프레임 반복)
    indices = indices[:n]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        elif frames:
            frames.append(frames[-1])  # 읽기 실패 시 이전 프레임 복사
    cap.release()

    # 부족한 경우 마지막 프레임 패딩
    while len(frames) < n:
        frames.append(frames[-1])

    return frames, len(frames)


def get_video_info(video_path: str) -> tuple[int, float]:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return total, fps


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


# ── 타이밍 누산기 ─────────────────────────────────────────────────
_t: dict = {}

def _reset_timing(n_text_tok: int = 0) -> None:
    _t.clear()
    _t.update({
        'autogaze':     0.0,   # AutoGaze forward 누적
        'vit':          0.0,   # _run_vision_tower_batched 누적
        'proj':         0.0,   # _embed 전체 - vit (projection + token 조립)
        'llm_prefill':  0.0,   # llm.forward 첫 번째 호출 (prefill)
        'llm_decode':   0.0,   # llm.forward 이후 호출 누적 (decode)
        'llm_calls':    0,
        'llm_seq_len':  0,     # prefill 시 LLM 입력 시퀀스 길이
        'embed_seq_len':0,     # _embed 출력 시퀀스 길이 (multimodal)
        'n_text_tok':   n_text_tok,
    })


def _install_timing_hooks(processor, model) -> None:
    """processor/model 에 타이밍 래퍼를 설치한다 (1회 호출)."""

    # 1) AutoGaze ─────────────────────────────────────────────────
    if getattr(processor, '_autogaze_model', None) is not None:
        _orig_ag = processor._autogaze_model.forward
        def _ag_forward(*args, **kwargs):
            t0 = time.perf_counter()
            result = _orig_ag(*args, **kwargs)
            _t['autogaze'] += time.perf_counter() - t0
            return result
        processor._autogaze_model.forward = _ag_forward

    # 2) ViT (_run_vision_tower_batched) ──────────────────────────
    _orig_vit = model._run_vision_tower_batched
    def _vit_batched(*args, **kwargs):
        t0 = time.perf_counter()
        result = _orig_vit(*args, **kwargs)
        _t['vit'] += time.perf_counter() - t0
        return result
    model._run_vision_tower_batched = _vit_batched

    # 3) _embed (ViT + projection 전체, 시각 토큰 수 캡처) ─────────
    _orig_embed = model._embed
    def _embed(*args, **kwargs):
        vit_before = _t['vit']
        t0 = time.perf_counter()
        result = _orig_embed(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        _t['proj'] += max(0.0, elapsed - (_t['vit'] - vit_before))
        _t['embed_seq_len'] = result.shape[1]
        return result
    model._embed = _embed

    # 4) LLM (prefill / decode 분리) ──────────────────────────────
    _orig_llm = model.llm.forward
    def _llm_forward(*args, **kwargs):
        t0 = time.perf_counter()
        result = _orig_llm(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        _t['llm_calls'] += 1
        if _t['llm_calls'] == 1:
            _t['llm_prefill'] = elapsed
            # prefill 시 inputs_embeds 로 시퀀스 길이 기록
            ie = kwargs.get('inputs_embeds')
            if ie is not None:
                _t['llm_seq_len'] = ie.shape[1]
            elif args:
                a0 = args[0]
                if isinstance(a0, torch.Tensor):
                    _t['llm_seq_len'] = a0.shape[1]
        else:
            _t['llm_decode'] += elapsed
        return result
    model.llm.forward = _llm_forward


def _print_timing(t_prep: float, t_gen: float, n_tok: int, n_frames: int) -> None:
    n_vis   = max(0, _t['embed_seq_len'] - _t['n_text_tok'])
    tok_s   = n_tok / max(_t['llm_decode'], 1e-4)
    other_p = max(0.0, t_prep - _t['autogaze'])
    other_g = max(0.0, t_gen  - _t['vit'] - _t['llm_prefill'] - _t['llm_decode'])

    lines = [
        ("전처리 (합계)",       f"{t_prep:.2f}s",        ""),
        ("  AutoGaze",          f"{_t['autogaze']:.2f}s", f"{n_frames} 프레임 입력"),
        ("  기타 전처리",        f"{other_p:.2f}s",       ""),
        ("생성 (합계)",          f"{t_gen:.2f}s",         ""),
        ("  ViT 인코딩",         f"{_t['vit']:.2f}s",     f"→ {n_vis} 시각 토큰"),
        ("  LLM 프리필",         f"{_t['llm_prefill']:.2f}s", f"{_t['llm_seq_len']} 입력 토큰"),
        ("  LLM 디코드",         f"{_t['llm_decode']:.2f}s",  f"{n_tok} 토큰  /  {tok_s:.1f} tok/s"),
        ("  기타 생성",          f"{other_g:.2f}s",       ""),
    ]
    w1 = max(len(l[0]) for l in lines)
    w2 = max(len(l[1]) for l in lines)
    width = w1 + w2 + max(len(l[2]) for l in lines) + 6
    sep = "  " + "─" * (width + 4)

    print()
    print(sep)
    for name, t, detail in lines:
        print(f"  {name:<{w1}}  {t:>{w2}}   {detail}")
    print(sep)
    print(f"  {'전체':<{w1}}  {t_prep + t_gen:>{w2}.2f}s")
    print(sep)


# ── 비디오 메타데이터 출력 ────────────────────────────────────────
total_frames, fps = get_video_info(video_path)
duration = total_frames / fps if fps else 0

print("=" * 60)
print("NVILA-8B-HD-Video 추론 테스트")
print("=" * 60)
print(f"디바이스     : {device}  ({dtype})")
print(f"비디오       : {video_path}")
print(f"  총 프레임  : {total_frames}  ({fps:.1f} fps, {duration:.1f}초)")
print(f"NVILA 경로   : {model_path}")
print(f"AutoGaze 경로: {ag_path}")
print()


# ── 샘플링 시나리오 구성 ──────────────────────────────────────────
# 각 시나리오: (이름, videos_arg, n_frames_label, processor_frames_override)
#   videos_arg: 문자열 경로(균일 샘플링) 또는 PIL 리스트(stride 샘플링)
#   processor_frames_override: None이면 덮어쓰지 않음

scenarios = []

if args.frames is not None:
    scenarios.append({
        "name":      f"균일 샘플링 {args.frames}프레임 (linspace)",
        "videos":    video_path,          # 문자열 → processor가 linspace 샘플링
        "n_frames":  args.frames,
        "proc_override": args.frames,     # processor.num_video_frames 덮어쓰기
    })

if args.stride is not None:
    pil_frames, n_actual = load_frames_stride(video_path, args.stride)
    coverage = n_actual * args.stride / total_frames * 100 if total_frames else 0
    scenarios.append({
        "name":      f"stride={args.stride} 샘플링 ({n_actual}프레임, 전체의 {coverage:.0f}% 커버)",
        "videos":    pil_frames,          # PIL 리스트 → processor가 그대로 사용
        "n_frames":  n_actual,
        "proc_override": n_actual,
    })

compare_mode = len(scenarios) > 1


# ── 1. 프로세서 로드 ──────────────────────────────────────────────
print("[1/3] 프로세서 로드 중 ...")
t0 = time.perf_counter()

from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained(
    model_path,
    trust_remote_code=True,
    autogaze_model_id=ag_path,
)

t_proc = time.perf_counter() - t0
video_token = processor.tokenizer.video_token
print(f"  완료 ({t_proc:.1f}s)")
print(f"  비디오 토큰: {repr(video_token)}")
print(f"  gazing_ratio: tile={processor.gazing_ratio_tile}, thumb={processor.gazing_ratio_thumbnail}")
print()


# ── 2. 모델 로드 ──────────────────────────────────────────────────
print("[2/3] NVILA 모델 로드 중 (~16 GB) ...")
t0 = time.perf_counter()

from transformers import AutoModel
model = AutoModel.from_pretrained(
    model_path,
    trust_remote_code=True,
    dtype=dtype,
    device_map="auto",
)
model.eval()

t_model = time.perf_counter() - t0
n_params = sum(p.numel() for p in model.parameters()) / 1e9
print(f"  완료 ({t_model:.1f}s)  파라미터: {n_params:.2f}B")
print()

# 타이밍 훅 설치 (1회)
_install_timing_hooks(processor, model)


# ── 3. 추론 함수 ──────────────────────────────────────────────────
def run_inference(scenario: dict, question: str) -> tuple[str, float, float, int]:
    """
    단일 시나리오 추론.
    반환: (답변, 전처리 시간, 생성 시간, 생성 토큰 수)
    """
    processor.num_video_frames = scenario["proc_override"]

    prompt = f"{video_token}\n{question}"
    n_text_tok = len(processor.tokenizer.encode(prompt)) - 1  # <video> 토큰 제외

    _reset_timing(n_text_tok)

    t0 = time.perf_counter()
    inputs = processor(text=prompt, videos=scenario["videos"])
    t_prep = time.perf_counter() - t0

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

    return answer, t_prep, t_gen, n_tok


# ── 4. 질의응답 ───────────────────────────────────────────────────
print("[3/3] 비디오 질의응답")

for qi, question in enumerate(questions):
    print("=" * 60)
    print(f"[Q{qi + 1}] {question}")

    if compare_mode:
        # 시나리오별 답변을 나란히 출력
        for sc in scenarios:
            print(f"\n  ▶ {sc['name']}  ({sc['n_frames']}프레임)")
            print("  " + "-" * 53)
            answer, t_prep, t_gen, n_tok = run_inference(sc, question)
            for line in answer.splitlines():
                print(f"  {line}")
            _print_timing(t_prep, t_gen, n_tok, sc['n_frames'])
    else:
        sc = scenarios[0]
        print(f"샘플링: {sc['name']}")
        print("-" * 55)
        answer, t_prep, t_gen, n_tok = run_inference(sc, question)
        print(f"[A{qi + 1}] {answer}")
        _print_timing(t_prep, t_gen, n_tok, sc['n_frames'])

print()
print("=" * 60)
print("완료 ✓")
