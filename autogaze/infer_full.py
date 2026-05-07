#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Full pipeline video QA inference: AutoGaze + ViT + MLLM.

두 파일의 역할 분리
-------------------
  autogaze/infer.py       AutoGaze만 실행 — gaze map 추출/시각화/JSON 저장
  autogaze/infer_full.py  이 파일 — AutoGaze + ViT + MLLM 전체 파이프라인

지원 MLLM (--mllm)
-------------------
  nvila           NVILA-8B-HD-Video  (SigLIP ViT, native AutoGaze 통합)
  qwen25vl        Qwen2.5-VL-7B      (zero-shot hook 방식)
  qwen25vl_full   Qwen2.5-VL-7B      (full ViT 통합, 시간별 gaze map)
  vjepa2          V-JEPA2 인코더     (zero-shot hook, 특징 추출 전용)
  vjepa2_full     V-JEPA2 인코더     (full 통합, 특징 추출 전용)
  vjepa2_llm      V-JEPA2 ViT + projector + LLM  (MCQ video QA 가능)
  siglip          순수 HF SigLIP     (zero-shot hook, 특징 추출 전용)
                  → NVILA 수정 버전과의 비교용

ViT / AutoGaze 통합 구조
-------------------------
  MLLM          ViT 백본         AutoGaze 통합 방식                  MCQ
  ─────────     ───────────────  ──────────────────────────────────  ────
  nvila         SigLIP (수정)    NVILAProcessor 내장 (mask_with_gazing)  ✓
  qwen25vl      Qwen2.5-VL ViT   zero-shot forward hook              ✓
  qwen25vl_full Qwen2.5-VL ViT   class monkey-patch                  ✓
  vjepa2        V-JEPA2 ViT-L    zero-shot forward hook              특징만
  vjepa2_full   V-JEPA2 ViT-L    class monkey-patch                  특징만
  siglip        SigLIP (원본HF)  zero-shot forward hook (per-frame)  특징만

사용 예시
---------
  # NVILA 기본 실행 (AutoGaze ON)
  python autogaze/infer_full.py assets/example_input.mp4 \\
      --mllm nvila --model-path weights/NVILA-8B-HD-Video \\
      --autogaze-path weights/AutoGaze

  # AutoGaze ON/OFF 비교
  python autogaze/infer_full.py assets/example_input.mp4 \\
      --mllm nvila --compare-autogaze

  # Qwen2.5-VL, full ViT 통합
  python autogaze/infer_full.py assets/example_input.mp4 \\
      --mllm qwen25vl_full \\
      --model-path Qwen/Qwen2.5-VL-7B-Instruct \\
      --autogaze-path weights/AutoGaze

  # Gazing ratio sweep (0.1 → 1.0 단계별 비교)
  python autogaze/infer_full.py assets/example_input.mp4 \\
      --mllm nvila --sweep-ratio --ratio-step 0.25

  # V-JEPA2 ViT + LLM (projector 필요)
  python autogaze/infer_full.py assets/example_input.mp4 \\
      --mllm vjepa2_llm \\
      --model-path facebook/vjepa2-vitl-fpc64-256 \\
      --lm-path Qwen/Qwen2.5-7B-Instruct \\
      --projector-path weights/vjepa2_projector \\
      --autogaze-path weights/AutoGaze

  # 순수 HF SigLIP — 특징 추출 (AutoGaze 없음, NVILA 수정 버전 비교용)
  python autogaze/infer_full.py assets/example_input.mp4 \\
      --mllm siglip \\
      --model-path google/siglip-so400m-patch14-224 \\
      --no-autogaze

  # SigLIP + AutoGaze hook (zero-shot, 수정 없는 vanilla SigLIP에 gaze masking 적용)
  python autogaze/infer_full.py assets/example_input.mp4 \\
      --mllm siglip \\
      --model-path google/siglip-so400m-patch14-224 \\
      --autogaze-path weights/AutoGaze

  # AutoGaze OFF (기준선)
  python autogaze/infer_full.py assets/example_input.mp4 \\
      --mllm qwen25vl --model-path Qwen/Qwen2.5-VL-7B-Instruct \\
      --no-autogaze

  # 질문 직접 지정 + stride 샘플링
  python autogaze/infer_full.py my_video.mp4 \\
      --question "What is happening in this video?" \\
      --stride 10

  # Gaze map 시각화 저장
  python autogaze/infer_full.py assets/example_input.mp4 \\
      --save-gaze --output-dir results/gaze_viz/
"""

import argparse
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
from PIL import Image

warnings.filterwarnings("ignore", category=FutureWarning)

REPO_ROOT     = Path(__file__).resolve().parent.parent
DEFAULT_MODEL  = REPO_ROOT / "weights" / "NVILA-8B-HD-Video"
DEFAULT_AG     = REPO_ROOT / "weights" / "AutoGaze"
DEFAULT_VIDEO  = REPO_ROOT / "assets" / "example_input.mp4"

DEFAULT_QUESTIONS = [
    "이 비디오에서 무엇이 일어나고 있나요? 구체적으로 설명해 주세요.",
    "비디오에서 주요 피사체나 활동은 무엇인가요?",
    "어떤 환경(장소, 조명 등)에서 촬영된 영상인가요?",
]


# ─────────────────────────────────────────────────────────────────────────────
# Video loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_frames_uniform(video_path: str, num_frames: int) -> List[Image.Image]:
    """Uniformly sample num_frames from video (linspace indices)."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = num_frames

    indices = np.linspace(0, total - 1, num=num_frames, dtype=int).tolist()
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        elif frames:
            frames.append(frames[-1])
    cap.release()

    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else Image.new("RGB", (224, 224)))
    return frames[:num_frames]


def load_frames_stride(video_path: str, stride: int) -> List[Image.Image]:
    """Extract every stride-th frame, truncated to nearest multiple of 16."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = list(range(0, total, stride))

    n = (len(indices) // 16) * 16
    if n == 0:
        n = 16
    indices = indices[:n]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        elif frames:
            frames.append(frames[-1])
    cap.release()

    while len(frames) < n:
        frames.append(frames[-1] if frames else Image.new("RGB", (224, 224)))
    return frames[:n]


def get_video_info(video_path: str):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return total, fps


# ─────────────────────────────────────────────────────────────────────────────
# Runner loading with timing instrumentation
# ─────────────────────────────────────────────────────────────────────────────

def build_runner(
    mllm: str,
    model_path: str,
    autogaze_path: Optional[str],
    gazing_ratio: float,
    **kwargs,
):
    """Load runner via registry and optionally install timing hooks."""
    from autogaze.eval.models import load_runner
    return load_runner(
        mllm=mllm,
        model_path=model_path,
        autogaze_path=autogaze_path,
        gazing_ratio=gazing_ratio,
        **kwargs,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Timing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()


def _timed_run(runner, frames: List[Image.Image], prompt: str,
               max_new_tokens: int) -> tuple:
    """Run inference and return (answer, elapsed_s, gaze_s).

    gaze_s is the AutoGaze-only time; 0.0 when AutoGaze is disabled.
    Instruments runner._run_autogaze() if present.
    """
    gaze_s = 0.0

    if hasattr(runner, '_run_autogaze') and runner.selector is not None:
        orig_ag = runner._run_autogaze
        _gaze_time = [0.0]

        def _timed_ag(frm):
            _sync()
            t0 = time.perf_counter()
            result = orig_ag(frm)
            _sync()
            _gaze_time[0] += time.perf_counter() - t0
            return result

        runner._run_autogaze = _timed_ag
        _sync()
        t0 = time.perf_counter()
        answer = runner.run(frames, prompt, max_new_tokens=max_new_tokens)
        _sync()
        elapsed = time.perf_counter() - t0
        runner._run_autogaze = orig_ag   # restore
        gaze_s = _gaze_time[0]
    else:
        _sync()
        t0 = time.perf_counter()
        answer = runner.run(frames, prompt, max_new_tokens=max_new_tokens)
        _sync()
        elapsed = time.perf_counter() - t0

    return answer, elapsed, gaze_s


# ─────────────────────────────────────────────────────────────────────────────
# Gaze visualisation
# ─────────────────────────────────────────────────────────────────────────────

def save_gaze_viz(runner, frames: List[Image.Image], out_dir: Path,
                  video_stem: str) -> Optional[Path]:
    """Save gaze overlay grid PNG if the runner supports it."""
    if not hasattr(runner, '_run_autogaze') or runner.selector is None:
        print("  [viz] AutoGaze 비활성화 — gaze 시각화 건너뜀")
        return None

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import torch.nn.functional as F
    except ImportError:
        print("  [viz] matplotlib 없음 — pip install matplotlib")
        return None

    print("  [viz] Gaze map 계산 중...")
    with torch.no_grad():
        gaze_map = runner._run_autogaze(frames)    # (1, T, 14, 14)
    gaze_map = gaze_map[0].cpu().float().numpy()   # (T, 14, 14)

    T = min(len(frames), gaze_map.shape[0], 16)
    fig, axes = plt.subplots(2, T, figsize=(T * 2.2, 5), squeeze=False)

    for t in range(T):
        # 원본 프레임
        axes[0, t].imshow(frames[t])
        axes[0, t].axis("off")
        axes[0, t].set_title(f"F{t}", fontsize=7)

        # Gaze overlay
        frame_resized = np.array(frames[t].resize((224, 224)))
        gm = gaze_map[t]                         # (14, 14)
        mask_up = F.interpolate(
            torch.tensor(gm).unsqueeze(0).unsqueeze(0).float(),
            size=(224, 224), mode="nearest",
        ).squeeze().numpy()

        overlay = frame_resized.copy().astype(float)
        overlay[mask_up == 0] *= 0.25
        overlay = overlay.clip(0, 255).astype(np.uint8)

        axes[1, t].imshow(overlay)
        axes[1, t].imshow(gm, cmap="cool", alpha=0.35,
                          extent=[0, 224, 224, 0], aspect="auto")
        n_sel = int(gm.sum())
        axes[1, t].set_title(f"Gaze({n_sel})", fontsize=7)
        axes[1, t].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=8)
    axes[1, 0].set_ylabel(f"Gaze (r={runner.gazing_ratio:.2f})", fontsize=8)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_stem}_gaze_viz.png"
    plt.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  [viz] 저장 → {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Output printers
# ─────────────────────────────────────────────────────────────────────────────

def _print_header(label: str, w: int = 56):
    print()
    print("═" * w)
    print(f"  {label}")
    print("═" * w)


def _print_timing_row(name: str, elapsed: float, gaze_s: float,
                      ag_enabled: bool, ratio: float):
    mllm_s = elapsed - gaze_s
    ag_str = f"{gaze_s:.2f}s" if ag_enabled and gaze_s > 0 else "—"
    print()
    print(f"  {'항목':<18}{'시간':>8}   {'비고'}")
    print("  " + "─" * 46)
    if ag_enabled:
        print(f"  {'AutoGaze':18}{gaze_s:>7.2f}s   ratio={ratio:.2f}")
    print(f"  {'ViT + LLM':18}{mllm_s:>7.2f}s")
    print(f"  {'전체 (합계)':18}{elapsed:>7.2f}s   AG {'ON' if ag_enabled else 'OFF'}")
    print("  " + "─" * 46)


def _print_comparison(res_on: tuple, res_off: tuple, ratio: float):
    """res_on / res_off: (answer, elapsed, gaze_s)"""
    ans_on,  t_on,  g_on  = res_on
    ans_off, t_off, g_off = res_off

    print()
    print(f"  {'항목':<18} {'AutoGaze ON':>14} {'AutoGaze OFF':>14}")
    print("  " + "─" * 50)
    print(f"  {'AutoGaze':<18} {g_on:>13.2f}s {'—':>14}")
    print(f"  {'ViT + LLM':<18} {t_on-g_on:>13.2f}s {t_off:>13.2f}s")
    print(f"  {'전체':<18} {t_on:>13.2f}s {t_off:>13.2f}s")
    if t_off > 0:
        saving = 100 * (t_off - t_on) / t_off
        print(f"  {'절감률':<18} {saving:>12.1f}%")
    print("  " + "─" * 50)
    print()
    print(f"  [ON ]  {ans_on}")
    print(f"  [OFF]  {ans_off}")


def _print_sweep_table(sweep: list, ratio_w: int = 6):
    """sweep: list of (ratio, answer, elapsed, gaze_s)"""
    hdrs  = ["ratio", "AutoGaze", "ViT+LLM", "전체", "답변 (앞 60자)"]
    rows  = []
    for r, ans, t, g in sweep:
        rows.append([
            f"{r:.2f}",
            f"{g:.2f}s" if g > 0 else "—",
            f"{t-g:.2f}s",
            f"{t:.2f}s",
            ans.replace("\n", " ")[:60] + ("…" if len(ans) > 60 else ""),
        ])
    col_w = [max(len(hdrs[i]), max(len(row[i]) for row in rows)) for i in range(len(hdrs))]
    sep   = "  " + " ─┼─ ".join("─" * w for w in col_w)
    hdr   = "  " + " │ ".join(f"{h:^{w}}" for h, w in zip(hdrs, col_w))
    print()
    print(hdr)
    print(sep)
    for row in rows:
        print("  " + " │ ".join(f"{v:>{w}}" for v, w in zip(row, col_w)))
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="AutoGaze + ViT + MLLM 전체 파이프라인 추론",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("video", nargs="?", default=str(DEFAULT_VIDEO),
                   help="비디오 파일 경로 (기본: assets/example_input.mp4)")
    p.add_argument("--question", default=None,
                   help="단일 질문 (없으면 예제 3가지 실행)")

    # ── MLLM 선택 ──────────────────────────────────────────────────
    p.add_argument("--mllm", default="nvila",
                   choices=["nvila", "qwen25vl", "qwen25vl_full",
                            "vjepa2", "vjepa2_full", "vjepa2_llm",
                            "siglip"],
                   help="MLLM 백엔드 (기본: nvila)")
    p.add_argument("--model-path", default=str(DEFAULT_MODEL),
                   help="MLLM 가중치 경로 또는 HuggingFace ID")
    p.add_argument("--autogaze-path", default=str(DEFAULT_AG),
                   help="AutoGaze 가중치 경로 (기본: weights/AutoGaze)")

    # ── AutoGaze 제어 ──────────────────────────────────────────────
    p.add_argument("--no-autogaze", action="store_true",
                   help="AutoGaze 비활성화 (모든 패치 사용)")
    p.add_argument("--gazing-ratio", type=float, default=0.75,
                   help="Gazing ratio 0~1 (기본: 0.75). 낮을수록 선택 패치 감소.")

    # ── 비교 / 스윕 모드 ───────────────────────────────────────────
    p.add_argument("--compare-autogaze", action="store_true",
                   help="AutoGaze ON/OFF 결과를 나란히 비교")
    p.add_argument("--sweep-ratio", action="store_true",
                   help="ratio 0.1 → 1.0 단계별 비교 (--ratio-step 으로 간격 조정)")
    p.add_argument("--ratio-step", type=float, default=0.25,
                   help="--sweep-ratio 단계 간격 (기본: 0.25)")

    # ── 프레임 샘플링 ──────────────────────────────────────────────
    p.add_argument("--frames", type=int, default=16,
                   help="균일 샘플링 프레임 수, 16의 배수 (기본: 16)")
    p.add_argument("--stride", type=int, default=None,
                   help="매 N번째 프레임 추출 (stride 샘플링). 지정 시 --frames 무시.")

    # ── vjepa2_llm 전용 ────────────────────────────────────────────
    p.add_argument("--lm-path", default=None,
                   help="[vjepa2_llm 전용] LLM 경로 또는 HF ID "
                        "(예: Qwen/Qwen2.5-7B-Instruct)")
    p.add_argument("--projector-path", default=None,
                   help="[vjepa2_llm 전용] VJEPA2Projector 체크포인트 경로 "
                        "(없으면 랜덤 초기화 — 학습 전 실행용)")

    # ── 생성 / 출력 ────────────────────────────────────────────────
    p.add_argument("--max-new-tokens", type=int, default=256,
                   help="최대 생성 토큰 수 (기본: 256)")
    p.add_argument("--save-gaze", action="store_true",
                   help="Gaze map 시각화 PNG 저장")
    p.add_argument("--output-dir", default="results/infer_full",
                   help="출력 디렉토리 (기본: results/infer_full)")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── 검증 ──────────────────────────────────────────────────────
    video_path = args.video
    assert Path(video_path).exists(), f"비디오 파일 없음: {video_path}"

    if not args.no_autogaze:
        ag_path = args.autogaze_path
        if not Path(ag_path).exists():
            print(f"[WARN] AutoGaze 가중치 없음: {ag_path}")
            print("  → bash scripts/download_models.sh weights autogaze")
            print("  → AutoGaze OFF 모드로 계속합니다.")
            args.no_autogaze = True

    if not Path(args.model_path).exists():
        # HuggingFace ID 형식이면 통과 (e.g. "Qwen/Qwen2.5-VL-7B-Instruct")
        if "/" not in args.model_path:
            print(f"[WARN] 모델 가중치 없음: {args.model_path}")
            print("  → bash scripts/download_models.sh weights nvila")

    # ── 비디오 정보 ────────────────────────────────────────────────
    total_frames, fps = get_video_info(video_path)
    duration = total_frames / fps

    _print_header("AutoGaze Full Pipeline 추론")
    print(f"  디바이스     : {'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'}")
    print(f"  비디오       : {video_path}")
    print(f"    총 프레임  : {total_frames}  ({fps:.1f} fps, {duration:.1f}초)")
    print(f"  MLLM         : {args.mllm}")
    print(f"  모델 경로    : {args.model_path}")
    print(f"  AutoGaze     : {'OFF' if args.no_autogaze else f'ON  (ratio={args.gazing_ratio})'}")
    print()

    # ── 프레임 로드 ────────────────────────────────────────────────
    if args.stride is not None:
        frames = load_frames_stride(video_path, args.stride)
        print(f"  [샘플링] stride={args.stride} → {len(frames)}프레임")
    else:
        frames = load_frames_uniform(video_path, args.frames)
        print(f"  [샘플링] 균일 {len(frames)}프레임")

    questions = [args.question] if args.question else DEFAULT_QUESTIONS

    out_dir    = Path(args.output_dir)
    video_stem = Path(video_path).stem

    # ── 모델 로드 ──────────────────────────────────────────────────
    autogaze_path = None if args.no_autogaze else args.autogaze_path

    # vjepa2_llm 전용 인수 검증
    extra_kwargs: dict = {}
    if args.mllm == "vjepa2_llm":
        if not args.lm_path:
            print("[ERROR] --mllm vjepa2_llm 에는 --lm-path 가 필요합니다.")
            print("  예: --lm-path Qwen/Qwen2.5-7B-Instruct")
            raise SystemExit(1)
        extra_kwargs["lm_path"]        = args.lm_path
        extra_kwargs["projector_path"] = args.projector_path  # None 허용

    print()
    print(f"[1/2] {args.mllm.upper()} 로드 중...")
    t0 = time.perf_counter()
    runner = build_runner(
        mllm         = args.mllm,
        model_path   = args.model_path,
        autogaze_path= autogaze_path,
        gazing_ratio = args.gazing_ratio,
        **extra_kwargs,
    )
    t_load = time.perf_counter() - t0
    print(f"  완료 ({t_load:.1f}s)\n")

    # ── Gaze 시각화 저장 (선택) ────────────────────────────────────
    if args.save_gaze:
        save_gaze_viz(runner, frames, out_dir, video_stem)

    # ─────────────────────────────────────────────────────────────
    # 추론 실행
    # ─────────────────────────────────────────────────────────────
    print("[2/2] 추론")

    # 특징 추출 전용 러너 (MCQ 불가)
    if not runner.supports_mcq:
        print(f"\n[INFO] {args.mllm} 는 특징 추출 전용 러너입니다 (LLM 없음).")
        print("       encode_video() 로 특징 추출을 실행합니다.\n")
        t0 = time.perf_counter()
        feats = runner.encode_video(frames)
        t_enc = time.perf_counter() - t0
        print(f"  특징 텐서 shape : {feats.shape}")
        print(f"  인코딩 시간      : {t_enc*1000:.1f} ms")
        if args.mllm == "siglip":
            T = len(frames)
            N = feats.shape[1] // T
            print(f"  (T={T} frames × N={N} patches/frame × C={feats.shape[2]})")
            print()
            print("  MCQ 추론을 하려면 projector + LLM 과 연결 후 학습이 필요합니다.")
            print("  NVILA 는 이 SigLIP 을 수정한 버전 + NVILAProcessor 로 MCQ 가능합니다.")
        else:
            print()
            print("  MCQ 추론은 vjepa2_llm (--lm-path 필요) 을 사용하세요.")
        print()
        print("=" * 56)
        print("완료.")
        return

    if args.sweep_ratio:
        # ── Ratio sweep 모드 ──────────────────────────────────────
        step   = args.ratio_step
        ratios = [round(i * step, 10) for i in range(1, int(1.0 / step) + 1)]
        if ratios[-1] < 1.0:
            ratios.append(1.0)

        for qi, question in enumerate(questions):
            _print_header(f"[Q{qi+1}] {question[:70]}")
            sweep_results = []

            for r in ratios:
                print(f"  ratio={r:.2f} 추론 중...", end="\r", flush=True)
                # Adjust runner's gazing_ratio in-place
                if runner.selector is not None:
                    runner.selector.gazing_ratio = r
                    runner.gazing_ratio           = r
                _ag_path = None if r >= 1.0 else autogaze_path
                if r >= 1.0 and runner.selector is not None:
                    # run without AutoGaze for the r=1.0 baseline
                    _orig_sel = runner.selector
                    runner.selector = None
                    ans, t_total, g_s = _timed_run(runner, frames, question, args.max_new_tokens)
                    runner.selector = _orig_sel
                else:
                    ans, t_total, g_s = _timed_run(runner, frames, question, args.max_new_tokens)
                sweep_results.append((r, ans, t_total, g_s))

            print(" " * 40, end="\r")
            _print_sweep_table(sweep_results)

        # Restore ratio
        if runner.selector is not None:
            runner.selector.gazing_ratio = args.gazing_ratio
            runner.gazing_ratio           = args.gazing_ratio

    elif args.compare_autogaze:
        # ── AutoGaze ON/OFF 비교 모드 ─────────────────────────────
        # Build baseline runner (same mllm, no AutoGaze)
        print("  AutoGaze OFF 러너 로드 중...")
        t0 = time.perf_counter()
        runner_base = build_runner(
            mllm         = args.mllm,
            model_path   = args.model_path,
            autogaze_path= None,           # OFF
            gazing_ratio = 1.0,
        )
        print(f"  완료 ({time.perf_counter()-t0:.1f}s)\n")

        for qi, question in enumerate(questions):
            _print_header(f"[Q{qi+1}] {question[:70]}")

            # AutoGaze ON
            print("  AutoGaze ON  추론 중...")
            res_on = _timed_run(runner, frames, question, args.max_new_tokens)

            # AutoGaze OFF
            print("  AutoGaze OFF 추론 중...")
            res_off = _timed_run(runner_base, frames, question, args.max_new_tokens)

            _print_comparison(res_on, res_off, args.gazing_ratio)

    else:
        # ── 단일 모드 ─────────────────────────────────────────────
        for qi, question in enumerate(questions):
            _print_header(f"[Q{qi+1}] {question[:70]}")
            print()
            ans, t_total, g_s = _timed_run(
                runner, frames, question, args.max_new_tokens
            )
            print(f"  {ans}")
            _print_timing_row(
                "inference", t_total, g_s,
                ag_enabled=runner.selector is not None,
                ratio=args.gazing_ratio,
            )

    print()
    print("=" * 56)
    print("완료.")


if __name__ == "__main__":
    main()
