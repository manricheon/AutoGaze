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
  siglip_qwen25   Qwen2.5-VL-7B      (SigLIP/Qwen visual path, hook 기본)
  vjepa2_nvila    V-JEPA2 + NVILA    (full 기본)
  vjepa2_qwen25   V-JEPA2 + Qwen2.5  (full 기본)
  generic_mllm    임의 HF MLLM       (configurable hook; PoC용)
  vjepa2          V-JEPA2 인코더     (zero-shot hook, 특징 추출 전용)
  siglip          순수 HF SigLIP     (zero-shot hook, 특징 추출 전용)
                  → NVILA 수정 버전과의 비교용

ViT / AutoGaze 통합 구조
-------------------------
  MLLM          ViT 백본         AutoGaze 통합 방식                  MCQ
  ─────────     ───────────────  ──────────────────────────────────  ────
  nvila         SigLIP (수정)    NVILAProcessor 내장 (mask_with_gazing)  ✓
  siglip_qwen25 Qwen2.5-VL ViT   hook 또는 full (--integration)       ✓
  vjepa2_nvila  V-JEPA2 ViT-L    hook 또는 full (--integration)       ✓
  vjepa2_qwen25 V-JEPA2 ViT-L    hook 또는 full (--integration)       ✓
  generic_mllm  사용자 지정 ViT    configurable hook only             ✓
  vjepa2        V-JEPA2 ViT-L    zero-shot forward hook              특징만
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
      --mllm siglip_qwen25 --integration full \\
      --model-path Qwen/Qwen2.5-VL-7B-Instruct \\
      --autogaze-path weights/AutoGaze

  # Gazing ratio sweep (0.1 → 1.0 단계별 비교)
  python autogaze/infer_full.py assets/example_input.mp4 \\
      --mllm nvila --sweep-ratio --ratio-step 0.25

  # V-JEPA2 ViT + LLM (projector 필요)
  python autogaze/infer_full.py assets/example_input.mp4 \\
      --mllm vjepa2_qwen25 \\
      --vjepa2-path facebook/vjepa2-vitl-fpc64-256 \\
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
      --mllm siglip_qwen25 --model-path Qwen/Qwen2.5-VL-7B-Instruct \\
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
import platform
import resource
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

PRIMARY_RUNNERS = {
    "nvila",
    "siglip_qwen25",
    "vjepa2_nvila",
    "vjepa2_qwen25",
    "generic_mllm",
    "vjepa2",
    "siglip",
}

DEPRECATED_RUNNERS = {
    "qwen25vl",
    "qwen25vl_full",
    "vjepa2_full",
    "vjepa2_llm",
    "nvila_vjepa2",
}


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


def _runner_model_path_and_kwargs(args, *, baseline: bool = False) -> tuple[str, Optional[str], float, dict]:
    """Resolve CLI args into load_runner arguments.

    V-JEPA2-based runners use the V-JEPA2 encoder as ``model_path`` except
    ``vjepa2_nvila``, which needs NVILA as ``model_path`` and receives
    ``vjepa2_path`` separately.  AutoGaze OFF/baseline never forwards an
    AutoGaze path.  For NVILA default/native mode, OFF is resolved to hook mode
    so the processor can run as an all-patch baseline without initializing
    AutoGaze.
    """
    model_path = args.model_path
    ratio = 1.0 if baseline else args.gazing_ratio
    autogaze_path = None if (baseline or args.no_autogaze) else args.autogaze_path

    integration = args.integration
    if integration is None:
        if args.mllm == "qwen25vl_full":
            integration = "full"
        elif args.mllm == "vjepa2_full":
            integration = "full"

    kwargs = {}
    if integration is not None:
        kwargs["integration"] = integration

    if args.mllm in {"vjepa2_nvila", "nvila_vjepa2"}:
        kwargs["vjepa2_path"] = args.vjepa2_path
    elif args.vjepa2_path and args.mllm in {"vjepa2", "vjepa2_qwen25", "vjepa2_full", "vjepa2_llm"}:
        model_path = args.vjepa2_path

    if args.lm_path:
        kwargs["lm_path"] = args.lm_path
    if args.projector_path:
        kwargs["projector_path"] = args.projector_path
    if args.mllm == "generic_mllm":
        kwargs.update({
            "processor_path": args.generic_processor_path,
            "vision_hook": args.generic_vision_hook,
            "patch_grid": args.generic_patch_grid,
            "has_cls_token": args.generic_has_cls_token,
            "media_key": args.generic_media_key,
            "prompt_template": args.generic_prompt_template,
        })

    if args.mllm == "nvila" and (baseline or args.no_autogaze):
        autogaze_path = None
        ratio = 1.0
        if kwargs.get("integration") in (None, "native") or "integration" not in kwargs:
            kwargs["integration"] = "hook"

    return model_path, autogaze_path, ratio, kwargs


def _runner_has_autogaze(runner) -> bool:
    if getattr(runner, "integration", None) == "native" and getattr(runner, "gazing_ratio", 1.0) < 1.0:
        return True
    return getattr(runner, "selector", None) is not None


# ─────────────────────────────────────────────────────────────────────────────
# Timing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()


def _process_rss_mb() -> float:
    """Return current process RSS when psutil is available; otherwise max RSS."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 ** 2)
    except Exception:
        max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() == "Darwin":
            return max_rss / (1024 ** 2)
        return max_rss / 1024


def _device_memory_mb() -> dict[str, float]:
    if torch.cuda.is_available():
        return {
            "cuda_allocated": torch.cuda.memory_allocated() / (1024 ** 2),
            "cuda_reserved": torch.cuda.memory_reserved() / (1024 ** 2),
            "cuda_peak_allocated": torch.cuda.max_memory_allocated() / (1024 ** 2),
        }
    if torch.backends.mps.is_available():
        stats = {}
        if hasattr(torch.mps, "current_allocated_memory"):
            stats["mps_allocated"] = torch.mps.current_allocated_memory() / (1024 ** 2)
        if hasattr(torch.mps, "driver_allocated_memory"):
            stats["mps_driver"] = torch.mps.driver_allocated_memory() / (1024 ** 2)
        return stats
    return {}


def _memory_snapshot() -> dict:
    return {
        "rss_mb": _process_rss_mb(),
        "device": _device_memory_mb(),
    }


def _reset_device_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _timed_run(runner, frames: List[Image.Image], prompt: str,
               max_new_tokens: int) -> tuple:
    """Run inference and return (answer, elapsed_s, gaze_s, mem_before, mem_after).

    gaze_s is the AutoGaze-only time; 0.0 when AutoGaze is disabled.
    Instruments runner._run_autogaze() if present.
    """
    gaze_s = 0.0
    _reset_device_peak()
    mem_before = _memory_snapshot()

    if hasattr(runner, '_run_autogaze') and _runner_has_autogaze(runner):
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

    mem_after = _memory_snapshot()
    return answer, elapsed, gaze_s, mem_before, mem_after


# ─────────────────────────────────────────────────────────────────────────────
# Gaze visualisation
# ─────────────────────────────────────────────────────────────────────────────

def save_gaze_viz(runner, frames: List[Image.Image], out_dir: Path,
                  video_stem: str) -> Optional[Path]:
    """Save gaze overlay grid PNG if the runner supports it."""
    if not hasattr(runner, '_run_autogaze') or not _runner_has_autogaze(runner):
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
    axes[1, 0].set_ylabel(f"Gaze (r={getattr(runner, 'gazing_ratio', 1.0):.2f})", fontsize=8)

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


def _format_memory_line(before: dict, after: dict) -> list[str]:
    rss_delta = after["rss_mb"] - before["rss_mb"]
    lines = [f"RSS={after['rss_mb']:.1f} MB (delta {rss_delta:+.1f} MB)"]
    device = after.get("device") or {}
    if "cuda_peak_allocated" in device:
        lines.append(
            "CUDA "
            f"allocated={device['cuda_allocated']:.1f} MB, "
            f"reserved={device['cuda_reserved']:.1f} MB, "
            f"peak={device['cuda_peak_allocated']:.1f} MB"
        )
    elif "mps_allocated" in device or "mps_driver" in device:
        parts = []
        if "mps_allocated" in device:
            parts.append(f"allocated={device['mps_allocated']:.1f} MB")
        if "mps_driver" in device:
            parts.append(f"driver={device['mps_driver']:.1f} MB")
        lines.append(f"MPS {', '.join(parts)}")
    return lines


def _print_resource_lines(
    *,
    runner,
    n_frames: int,
    max_new_tokens: int,
    mem_before: dict,
    mem_after: dict,
) -> None:
    visual_tokens = None
    if hasattr(runner, "n_visual_tokens"):
        visual_tokens = runner.n_visual_tokens(n_frames)
    token_note = f"visual={visual_tokens}" if visual_tokens is not None else "visual=unknown"
    print(f"  {'Tokens':18}{token_note}, max_new={max_new_tokens}")
    for line in _format_memory_line(mem_before, mem_after):
        print(f"  {'Memory':18}{line}")


def _print_timing_row(name: str, elapsed: float, gaze_s: float,
                      ag_enabled: bool, ratio: float, *, runner=None,
                      n_frames: int = 0, max_new_tokens: int = 0,
                      mem_before: Optional[dict] = None,
                      mem_after: Optional[dict] = None):
    mllm_s = elapsed - gaze_s
    ag_str = f"{gaze_s:.2f}s" if ag_enabled and gaze_s > 0 else "—"
    print()
    print(f"  {'항목':<18}{'시간':>8}   {'비고'}")
    print("  " + "─" * 46)
    if ag_enabled:
        print(f"  {'AutoGaze':18}{gaze_s:>7.2f}s   ratio={ratio:.2f}")
    print(f"  {'ViT + LLM':18}{mllm_s:>7.2f}s")
    print(f"  {'전체 (합계)':18}{elapsed:>7.2f}s   AG {'ON' if ag_enabled else 'OFF'}")
    if runner is not None and mem_before is not None and mem_after is not None:
        _print_resource_lines(
            runner=runner,
            n_frames=n_frames,
            max_new_tokens=max_new_tokens,
            mem_before=mem_before,
            mem_after=mem_after,
        )
    print("  " + "─" * 46)


def _print_comparison(res_on: tuple, res_off: tuple, ratio: float, *,
                      runner_on=None, runner_off=None, n_frames: int = 0,
                      max_new_tokens: int = 0):
    """res_on / res_off: (answer, elapsed, gaze_s, mem_before, mem_after)"""
    ans_on,  t_on,  g_on,  mem_on_before,  mem_on_after  = res_on
    ans_off, t_off, g_off, mem_off_before, mem_off_after = res_off

    print()
    print(f"  {'항목':<18} {'AutoGaze ON':>14} {'AutoGaze OFF':>14}")
    print("  " + "─" * 50)
    print(f"  {'AutoGaze':<18} {g_on:>13.2f}s {'—':>14}")
    print(f"  {'ViT + LLM':<18} {t_on-g_on:>13.2f}s {t_off:>13.2f}s")
    print(f"  {'전체':<18} {t_on:>13.2f}s {t_off:>13.2f}s")
    if t_off > 0:
        saving = 100 * (t_off - t_on) / t_off
        print(f"  {'절감률':<18} {saving:>12.1f}%")
    on_tokens = runner_on.n_visual_tokens(n_frames) if runner_on and hasattr(runner_on, "n_visual_tokens") else None
    off_tokens = runner_off.n_visual_tokens(n_frames) if runner_off and hasattr(runner_off, "n_visual_tokens") else None
    print(f"  {'Visual tokens':<18} {str(on_tokens or 'unknown'):>14} {str(off_tokens or 'unknown'):>14}")
    print(f"  {'Max new tokens':<18} {max_new_tokens:>14} {max_new_tokens:>14}")
    print(f"  {'RSS':<18} {mem_on_after['rss_mb']:>10.1f} MB {mem_off_after['rss_mb']:>10.1f} MB")
    for label, key in (("CUDA peak", "cuda_peak_allocated"), ("MPS allocated", "mps_allocated")):
        on_dev = (mem_on_after.get("device") or {}).get(key)
        off_dev = (mem_off_after.get("device") or {}).get(key)
        if on_dev is not None or off_dev is not None:
            print(f"  {label:<18} {on_dev or 0:>10.1f} MB {off_dev or 0:>10.1f} MB")
    print("  " + "─" * 50)
    print()
    print(f"  [ON ]  {ans_on}")
    print(f"  [OFF]  {ans_off}")


def _print_sweep_table(sweep: list, ratio_w: int = 6):
    """sweep: list of (ratio, answer, elapsed, gaze_s, mem_before, mem_after, visual_tokens)."""
    hdrs  = ["ratio", "tokens", "RSS", "AutoGaze", "ViT+LLM", "전체", "답변 (앞 60자)"]
    rows  = []
    for r, ans, t, g, _mem_before, mem_after, visual_tokens in sweep:
        rows.append([
            f"{r:.2f}",
            str(visual_tokens) if visual_tokens is not None else "unknown",
            f"{mem_after['rss_mb']:.0f} MB",
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
                   choices=sorted(PRIMARY_RUNNERS | DEPRECATED_RUNNERS),
                   help=(
                       "MLLM 백엔드. 권장 키: nvila, siglip_qwen25, "
                       "vjepa2_nvila, vjepa2_qwen25, generic_mllm, vjepa2, siglip. "
                       "기존 qwen25vl/qwen25vl_full/vjepa2_llm 등은 호환용 alias."
                   ))
    p.add_argument("--integration", default=None, choices=["native", "hook", "full"],
                   help="AutoGaze 통합 방식 override: native, hook, full")
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

    # ── V-JEPA2 / LLM 조합 ─────────────────────────────────────────
    p.add_argument("--vjepa2-path", default=None,
                   help="[V-JEPA2 러너] V-JEPA2 encoder 경로 또는 HF ID")
    p.add_argument("--lm-path", default=None,
                   help="[vjepa2_qwen25/vjepa2_llm 전용] LLM 경로 또는 HF ID "
                        "(예: Qwen/Qwen2.5-7B-Instruct)")
    p.add_argument("--projector-path", default=None,
                   help="[V-JEPA2+LLM 전용] VJEPA2Projector 체크포인트 경로 "
                        "(없으면 랜덤 초기화 — 학습 전 실행용)")
    p.add_argument("--generic-processor-path", default=None,
                   help="[generic_mllm] Processor 경로/HF ID (기본: --model-path)")
    p.add_argument("--generic-vision-hook", default=None,
                   help="[generic_mllm] AutoGaze mask를 적용할 dotted module path")
    p.add_argument("--generic-patch-grid", type=int, default=14,
                   help="[generic_mllm] 한 프레임/타일의 patch grid side length")
    p.add_argument("--generic-has-cls-token", action="store_true",
                   help="[generic_mllm] 첫 CLS 토큰은 보존")
    p.add_argument("--generic-media-key", default="images", choices=["images", "videos"],
                   help="[generic_mllm] processor media input key")
    p.add_argument("--generic-prompt-template", default="{prompt}",
                   help="[generic_mllm] prompt template; {prompt} 포함")

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

    if args.mllm in DEPRECATED_RUNNERS:
        print(f"[WARN] Deprecated --mllm alias 사용 중: {args.mllm}")
        print("  권장 키는 docs/eval_guide.md 의 {vit}_{lm} 형식입니다.")

    if not args.no_autogaze:
        ag_path = args.autogaze_path
        if not Path(ag_path).exists():
            print(f"[WARN] AutoGaze 가중치 없음: {ag_path}")
            print("  → bash scripts/download_models.sh weights autogaze")
            print("  → AutoGaze OFF 모드로 계속합니다.")
            args.no_autogaze = True

    model_path_for_check = args.vjepa2_path if args.mllm in {"vjepa2", "vjepa2_qwen25", "vjepa2_full", "vjepa2_llm"} and args.vjepa2_path else args.model_path
    if not Path(model_path_for_check).exists():
        # HuggingFace ID 형식이면 통과 (e.g. "Qwen/Qwen2.5-VL-7B-Instruct")
        if "/" not in model_path_for_check:
            print(f"[WARN] 모델 가중치 없음: {model_path_for_check}")
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

    # V-JEPA2 조합 인수 검증
    if args.mllm in {"vjepa2_nvila", "nvila_vjepa2", "vjepa2_qwen25", "vjepa2", "vjepa2_full", "vjepa2_llm"} and not args.vjepa2_path:
        print("[ERROR] V-JEPA2 기반 runner에는 --vjepa2-path 가 필요합니다.")
        raise SystemExit(1)
    if args.mllm in {"vjepa2_qwen25", "vjepa2_llm"}:
        if not args.lm_path:
            print(f"[ERROR] --mllm {args.mllm} 에는 --lm-path 가 필요합니다.")
            print("  예: --lm-path Qwen/Qwen2.5-7B-Instruct")
            raise SystemExit(1)
    if args.mllm == "generic_mllm" and not args.generic_vision_hook and not args.no_autogaze:
        print("[ERROR] AutoGaze ON 상태의 --mllm generic_mllm 에는 --generic-vision-hook 이 필요합니다.")
        print("  예: --generic-vision-hook model.visual.patch_embed")
        raise SystemExit(1)

    print()
    print(f"[1/2] {args.mllm.upper()} 로드 중...")
    t0 = time.perf_counter()
    model_path, runner_ag_path, runner_ratio, extra_kwargs = _runner_model_path_and_kwargs(args)
    runner = build_runner(
        mllm         = args.mllm,
        model_path   = model_path,
        autogaze_path= runner_ag_path,
        gazing_ratio = runner_ratio,
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
        _reset_device_peak()
        mem_before = _memory_snapshot()
        _sync()
        t0 = time.perf_counter()
        feats = runner.encode_video(frames)
        _sync()
        t_enc = time.perf_counter() - t0
        mem_after = _memory_snapshot()
        print(f"  특징 텐서 shape : {feats.shape}")
        print(f"  인코딩 시간      : {t_enc*1000:.1f} ms")
        _print_resource_lines(
            runner=runner,
            n_frames=len(frames),
            max_new_tokens=0,
            mem_before=mem_before,
            mem_after=mem_after,
        )
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
                if hasattr(runner, "set_gazing_ratio"):
                    runner.set_gazing_ratio(r)
                elif getattr(runner, "selector", None) is not None:
                    runner.selector.gazing_ratio = r
                    runner.gazing_ratio = r
                else:
                    runner.gazing_ratio = r
                _ag_path = None if r >= 1.0 else autogaze_path
                if r >= 1.0 and _runner_has_autogaze(runner):
                    # run without AutoGaze for the r=1.0 baseline
                    _orig_sel = getattr(runner, "selector", None)
                    if hasattr(runner, "selector"):
                        runner.selector = None
                    ans, t_total, g_s, mem_before, mem_after = _timed_run(
                        runner, frames, question, args.max_new_tokens
                    )
                    if hasattr(runner, "selector"):
                        runner.selector = _orig_sel
                else:
                    ans, t_total, g_s, mem_before, mem_after = _timed_run(
                        runner, frames, question, args.max_new_tokens
                    )
                visual_tokens = runner.n_visual_tokens(len(frames)) if hasattr(runner, "n_visual_tokens") else None
                sweep_results.append((r, ans, t_total, g_s, mem_before, mem_after, visual_tokens))

            print(" " * 40, end="\r")
            _print_sweep_table(sweep_results)

        # Restore ratio
        if hasattr(runner, "set_gazing_ratio"):
            runner.set_gazing_ratio(args.gazing_ratio)
        elif getattr(runner, "selector", None) is not None:
            runner.selector.gazing_ratio = args.gazing_ratio
            runner.gazing_ratio = args.gazing_ratio

    elif args.compare_autogaze:
        # ── AutoGaze ON/OFF 비교 모드 ─────────────────────────────
        # Build baseline runner (same mllm, no AutoGaze)
        print("  AutoGaze OFF 러너 로드 중...")
        t0 = time.perf_counter()
        base_model_path, base_ag_path, base_ratio, base_kwargs = _runner_model_path_and_kwargs(args, baseline=True)
        runner_base = build_runner(
            mllm         = args.mllm,
            model_path   = base_model_path,
            autogaze_path= base_ag_path,
            gazing_ratio = base_ratio,
            **base_kwargs,
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

            _print_comparison(
                res_on,
                res_off,
                args.gazing_ratio,
                runner_on=runner,
                runner_off=runner_base,
                n_frames=len(frames),
                max_new_tokens=args.max_new_tokens,
            )

    else:
        # ── 단일 모드 ─────────────────────────────────────────────
        for qi, question in enumerate(questions):
            _print_header(f"[Q{qi+1}] {question[:70]}")
            print()
            ans, t_total, g_s, mem_before, mem_after = _timed_run(
                runner, frames, question, args.max_new_tokens
            )
            print(f"  {ans}")
            _print_timing_row(
                "inference", t_total, g_s,
                ag_enabled=_runner_has_autogaze(runner),
                ratio=args.gazing_ratio,
                runner=runner,
                n_frames=len(frames),
                max_new_tokens=args.max_new_tokens,
                mem_before=mem_before,
                mem_after=mem_after,
            )

    print()
    print("=" * 56)
    print("완료.")


if __name__ == "__main__":
    main()
