#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Evaluate AutoGaze on standard video QA benchmarks.

Supported MLLMs (--mllm)
------------------------
  nvila        NVILA-8B with native AutoGaze processor  (default)
  qwen25vl     Qwen2.5-VL-7B with AutoGaze zero-shot hook

Video loading modes
-------------------
Most benchmarks store video bytes directly in the HuggingFace dataset,
so --video-dir is NOT required for them:

  Streams from HuggingFace (no download needed):
    videomme, videomme_w_sub, mvbench, nextqa, egoschema, mlvu, longvideobench

  Requires local download (--video-dir):
    hlvid  →  bash scripts/download_hlvid.sh data/HLVid

Supported tasks
---------------
  videomme          VideoMME without subtitles
  videomme_w_sub    VideoMME with subtitles
  mvbench           MVBench
  nextqa            NExT-QA (multiple-choice validation)
  egoschema         EgoSchema
  mlvu              MLVU (MCQ split)
  longvideobench    LongVideoBench
  hlvid             HLVid  (needs --video-dir)

Usage
-----
  # VideoMME with NVILA (default)
  python -m autogaze.eval.run_benchmark \\
      --task videomme \\
      --output results/videomme_ag075.json

  # VideoMME with Qwen2.5-VL-7B
  python -m autogaze.eval.run_benchmark \\
      --task videomme \\
      --mllm qwen25vl \\
      --model-path Qwen/Qwen2.5-VL-7B-Instruct \\
      --output results/videomme_qwen25vl_ag075.json

  # HLVid — needs local video files
  python -m autogaze.eval.run_benchmark \\
      --task hlvid \\
      --video-dir data/HLVid/videos \\
      --output results/hlvid_ag075.json

  # AutoGaze OFF baseline
  python -m autogaze.eval.run_benchmark \\
      --task videomme \\
      --no-autogaze \\
      --output results/videomme_baseline.json

  # Smoke test (first 50 samples)
  python -m autogaze.eval.run_benchmark \\
      --task videomme \\
      --max-samples 50 \\
      --output results/videomme_smoke.json
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import av
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from autogaze.eval.tasks import TASKS, TaskConfig

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Video loading — two modes
# ─────────────────────────────────────────────────────────────────────────────

def _decode_frames(container: av.container.InputContainer, num_frames: int) -> List[Image.Image]:
    """Shared decoder: uniformly sample *num_frames* from an open av container."""
    stream = container.streams.video[0]
    total  = stream.frames or 0

    frames_raw: List[np.ndarray] = []

    if total == 0:
        # Decode all first, then subsample
        frames_raw = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
        total = len(frames_raw)
        if total == 0:
            raise ValueError("Video has no frames")
        indices = set(np.linspace(0, total - 1, num_frames, dtype=int).tolist())
        frames_raw = [frames_raw[i] for i in sorted(indices)]
    else:
        indices = set(np.linspace(0, total - 1, num_frames, dtype=int).tolist())
        container.seek(0)
        for i, f in enumerate(container.decode(video=0)):
            if i in indices:
                frames_raw.append(f.to_ndarray(format="rgb24"))
            if len(frames_raw) == num_frames:
                break

    # Pad to exactly num_frames
    while len(frames_raw) < num_frames:
        frames_raw.append(frames_raw[-1])
    return [Image.fromarray(f) for f in frames_raw[:num_frames]]


def load_video_frames(video_path: Path, num_frames: int = 16) -> List[Image.Image]:
    """Load frames from a local video file."""
    container = av.open(str(video_path))
    try:
        return _decode_frames(container, num_frames)
    finally:
        container.close()


def load_video_from_bytes(raw: Any, num_frames: int = 16) -> List[Image.Image]:
    """Load frames from HuggingFace bytes column.

    *raw* can be:
      - bytes / bytearray
      - dict with a "bytes" key  (HuggingFace datasets Audio/Video format)
      - dict with a "path" key   (absolute local path fallback)
    """
    if isinstance(raw, dict):
        if raw.get("bytes"):
            data = raw["bytes"]
        elif raw.get("path"):
            # HF sometimes stores a relative archive path here, not a real
            # local path.  Only use it when the file actually exists.
            p = Path(raw["path"])
            if p.exists():
                return load_video_frames(p, num_frames)
            raise ValueError(
                f"HF video dict has no bytes and path is not a local file: {raw['path']!r}"
            )
        else:
            raise ValueError(
                f"HF video dict has neither bytes nor path "
                f"(keys={list(raw.keys())}, values={list(raw.values())!r:.120})"
            )
    else:
        data = raw

    container = av.open(io.BytesIO(bytes(data)))
    try:
        return _decode_frames(container, num_frames)
    finally:
        container.close()


def _resolve_video_path(
    video_id: str, video_dir: Optional[Path], video_ext: str
) -> Optional[Path]:
    """Try several path conventions to locate a local video file.

    Returns None immediately if *video_dir* is None (HF-bytes tasks that
    don't require a local video directory).
    """
    if video_dir is None:
        return None
    candidates = [
        video_dir / video_id,
        video_dir / (video_id + video_ext),
        video_dir / Path(video_id).with_suffix(video_ext),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Model loading  (delegates to the runner registry in models.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_runner(
    mllm: str,
    model_path: str,
    autogaze_path: Optional[str],
    gazing_ratio: float,
    dtype: torch.dtype = torch.bfloat16,
):
    """Load and return the appropriate MLLM runner.

    ``autogaze_path=None`` produces a full-patch (AutoGaze OFF) baseline
    regardless of the chosen MLLM.
    """
    from autogaze.eval.models import load_runner as _load
    return _load(
        mllm=mllm,
        model_path=model_path,
        autogaze_path=autogaze_path,
        gazing_ratio=gazing_ratio,
        dtype=dtype,
    )


# kept for backward-compat / notebook imports
def load_nvila(model_path, autogaze_path, gazing_ratio, dtype=torch.bfloat16):
    runner = load_runner("nvila", model_path, autogaze_path, gazing_ratio, dtype)
    return runner.processor, runner.model


def run_nvila(processor, model, frames, prompt, max_new_tokens=16):
    from autogaze.eval.models import NVILARunner
    runner = NVILARunner.__new__(NVILARunner)
    runner.processor = processor
    runner.model = model
    return runner.run(frames, prompt, max_new_tokens)


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    task_name: str,
    video_dir: Optional[Path],
    model_path: str,
    autogaze_path: Optional[str],
    gazing_ratio: float,
    num_frames: int,
    max_new_tokens: int,
    output_path: Path,
    max_samples: Optional[int],
    resume: bool,
    mllm: str = "nvila",
    hf_data_dir: Optional[Path] = None,
    runner_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run full evaluation and return result dict."""

    task: TaskConfig = TASKS[task_name]
    use_subtitle     = task_name.endswith("_w_sub")
    use_hf_bytes     = task.video_bytes_col is not None

    if not use_hf_bytes and video_dir is None:
        raise ValueError(
            f"Task '{task_name}' has no embedded HuggingFace video bytes.\n"
            f"Please provide --video-dir pointing to local video files.\n"
            f"See docs/eval_guide.md for download instructions."
        )

    # ── load dataset ─────────────────────────────────────────────────────── #
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    hf_source = str(hf_data_dir) if hf_data_dir else task.hf_repo
    log.info("Loading dataset %s / %s", hf_source, task.hf_split)
    if hf_data_dir:
        log.info("  Source: local directory %s", hf_data_dir)
    if use_hf_bytes:
        log.info("  Video mode: HuggingFace bytes (no --video-dir needed)")
    else:
        log.info("  Video mode: local files from %s", video_dir)

    ds = load_dataset(hf_source, split=task.hf_split, **task.hf_kwargs)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    log.info("  %d samples", len(ds))

    # ── resume: load already-written results ─────────────────────────────── #
    done: Dict[str, Any] = {}
    if resume and output_path.exists():
        with open(output_path) as f:
            prev = json.load(f)
        done = {r["sample_id"]: r for r in prev.get("per_sample", [])}
        log.info("Resuming: %d / %d already done", len(done), len(ds))

    # ── load model ───────────────────────────────────────────────────────── #
    runner = load_runner(
        mllm=mllm,
        model_path=model_path,
        autogaze_path=autogaze_path,
        gazing_ratio=gazing_ratio,
        **(runner_kwargs or {}),
    )

    # ── inference loop ───────────────────────────────────────────────────── #
    per_sample: List[Dict[str, Any]] = list(done.values())
    n_correct   = sum(1 for r in per_sample if r.get("correct"))
    n_skip      = 0

    for idx, sample in enumerate(ds):
        sample_id = str(sample.get("question_id") or sample.get("q_uid") or
                        sample.get("qid") or idx)

        if sample_id in done:
            continue

        # Load frames — prefer HF bytes, fall back to local file
        video_id = task.get_video_id(sample)
        try:
            if use_hf_bytes and sample.get(task.video_bytes_col) is not None:
                frames = load_video_from_bytes(sample[task.video_bytes_col], num_frames)
            else:
                # Bytes column was None (missing for this sample).  Try local file.
                if use_hf_bytes and video_dir is None:
                    log.warning(
                        "[%d] Video bytes missing for %s and no --video-dir provided"
                        " — skipping", idx, video_id,
                    )
                    n_skip += 1
                    continue
                video_path = _resolve_video_path(video_id, video_dir, task.video_ext)
                if video_path is None:
                    log.warning("[%d] Video not found: %s — skipping", idx, video_id)
                    n_skip += 1
                    continue
                frames = load_video_frames(video_path, num_frames)
        except Exception as e:
            log.warning("[%d] Frame loading error: %s — skipping", idx, e)
            n_skip += 1
            continue

        # Build prompt and run inference — capture latency and VRAM
        prompt    = task.build_prompt(sample, use_subtitle=use_subtitle)
        _cuda_available = torch.cuda.is_available()
        if _cuda_available:
            torch.cuda.reset_peak_memory_stats()
        t0        = time.perf_counter()
        generated = runner.run(frames, prompt, max_new_tokens)
        elapsed   = time.perf_counter() - t0
        peak_vram_mb = (
            torch.cuda.max_memory_allocated() / 1024 / 1024
            if _cuda_available else None
        )

        predicted = task.parse_prediction(generated)
        gt        = task.get_ground_truth(sample)
        correct   = predicted == gt

        result: Dict[str, Any] = {
            "sample_id"       : sample_id,
            "video_id"        : video_id,
            "question"        : str(sample[task.question_col]),
            "ground_truth"    : gt,
            "predicted"       : predicted,
            "generated"       : generated,
            "correct"         : correct,
            "latency_ms"      : round(elapsed * 1000, 1),
            "n_tokens_visual" : runner.n_visual_tokens(len(frames)),
        }
        if peak_vram_mb is not None:
            result["peak_vram_mb"] = round(peak_vram_mb, 1)
        if task.category_col and task.category_col in sample:
            result["category"] = str(sample[task.category_col])
        if task.duration_col and task.duration_col in sample:
            result["duration"] = str(sample[task.duration_col])

        per_sample.append(result)
        n_correct += int(correct)

        # Progress log
        n_done    = len(per_sample)
        n_total   = len(ds)
        acc_so_far = n_correct / n_done * 100
        log.info(
            "[%d/%d] %s → pred=%s gt=%s %s  (acc=%.1f%%  %.0fms)",
            n_done, n_total,
            sample_id, predicted, gt,
            "✓" if correct else "✗",
            acc_so_far, elapsed * 1000,
        )

        # Incremental save every 50 samples
        if n_done % 50 == 0:
            _save_results(output_path, task_name, per_sample, autogaze_path, gazing_ratio, num_frames, mllm=mllm)

    # ── compute metrics ───────────────────────────────────────────────────── #
    metrics = _compute_metrics(per_sample, task)

    # ── final save ────────────────────────────────────────────────────────── #
    _save_results(output_path, task_name, per_sample, autogaze_path, gazing_ratio, num_frames,
                  metrics=metrics, n_skip=n_skip, mllm=mllm)

    _print_summary(task_name, metrics, autogaze_path, gazing_ratio, n_skip, mllm=mllm)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _compute_metrics(
    per_sample: List[Dict[str, Any]],
    task: TaskConfig,
) -> Dict[str, Any]:
    if not per_sample:
        return {"overall_accuracy": 0.0}

    n_correct = sum(1 for r in per_sample if r.get("correct"))
    overall   = n_correct / len(per_sample) * 100

    metrics: Dict[str, Any] = {
        "overall_accuracy": round(overall, 2),
        "n_total"  : len(per_sample),
        "n_correct": n_correct,
    }

    # Efficiency metrics (latency, VRAM, visual token count)
    latencies = [r["latency_ms"] for r in per_sample if "latency_ms" in r]
    if latencies:
        metrics["avg_latency_ms"] = round(sum(latencies) / len(latencies), 1)

    vrams = [r["peak_vram_mb"] for r in per_sample if r.get("peak_vram_mb") is not None]
    if vrams:
        metrics["avg_peak_vram_mb"] = round(sum(vrams) / len(vrams), 1)

    tokens = [r["n_tokens_visual"] for r in per_sample if r.get("n_tokens_visual") is not None]
    if tokens:
        metrics["avg_n_tokens_visual"] = round(sum(tokens) / len(tokens))

    # Per-category breakdown
    for breakdown_col in ("category", "duration"):
        groups: Dict[str, list] = defaultdict(list)
        for r in per_sample:
            if breakdown_col in r:
                groups[r[breakdown_col]].append(r["correct"])
        if groups:
            per_group: Dict[str, float] = {}
            for g, vals in sorted(groups.items()):
                per_group[g] = round(sum(vals) / len(vals) * 100, 2)
            metrics[f"accuracy_by_{breakdown_col}"] = per_group

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_results(
    path: Path,
    task_name: str,
    per_sample: List[Dict[str, Any]],
    autogaze_path: Optional[str],
    gazing_ratio: float,
    num_frames: int,
    metrics: Optional[Dict] = None,
    n_skip: int = 0,
    mllm: str = "nvila",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task"         : task_name,
        "mllm"         : mllm,
        "autogaze"     : autogaze_path is not None,
        "autogaze_path": autogaze_path,
        "gazing_ratio" : gazing_ratio,
        "num_frames"   : num_frames,
        "n_skipped"    : n_skip,
        "metrics"      : metrics or {},
        "per_sample"   : per_sample,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _print_summary(
    task_name: str,
    metrics: Dict[str, Any],
    autogaze_path: Optional[str],
    gazing_ratio: float,
    n_skip: int,
    mllm: str = "nvila",
) -> None:
    ag_tag = f"AutoGaze ON (ratio={gazing_ratio})" if autogaze_path else "AutoGaze OFF"
    print("\n" + "=" * 60)
    print(f"  Task    : {task_name}")
    print(f"  MLLM    : {mllm}")
    print(f"  Mode    : {ag_tag}")
    print(f"  Total   : {metrics.get('n_total', 0)}  (skipped: {n_skip})")
    print(f"  Accuracy: {metrics.get('overall_accuracy', 0):.2f}%")

    if "avg_latency_ms" in metrics:
        print(f"  Latency : {metrics['avg_latency_ms']:.1f} ms/sample (avg)")
    if "avg_peak_vram_mb" in metrics:
        print(f"  VRAM    : {metrics['avg_peak_vram_mb']:.1f} MB peak (avg)")
    if "avg_n_tokens_visual" in metrics:
        print(f"  Tokens  : {metrics['avg_n_tokens_visual']} visual tokens/sample (avg)")

    for key in ("accuracy_by_duration", "accuracy_by_category"):
        breakdown = metrics.get(key)
        if breakdown:
            label = key.replace("accuracy_by_", "").capitalize()
            print(f"\n  Per-{label}:")
            for k, v in breakdown.items():
                print(f"    {k:<30} {v:.2f}%")

    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    from autogaze.eval.models import RUNNERS
    p = argparse.ArgumentParser(
        description="Evaluate AutoGaze on video QA benchmarks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--task", required=True, choices=list(TASKS.keys()),
        help="Benchmark task name",
    )
    p.add_argument(
        "--mllm", default="nvila", choices=sorted(RUNNERS.keys()),
        help=(
            "Runner key ({vit}_{lm} convention). "
            "Primary: nvila, vjepa2_nvila, siglip_qwen25, vjepa2_qwen25, vjepa2, siglip. "
            "Deprecated aliases: nvila_vjepa2, qwen25vl, qwen25vl_full, vjepa2_llm, vjepa2_full."
        ),
    )
    p.add_argument(
        "--integration", default=None,
        choices=["native", "hook", "full"],
        help=(
            "AutoGaze integration mode override. "
            "native: processor-level (NVILA only). "
            "hook: zero-shot forward hook (accuracy validation, no latency gain). "
            "full: tokens physically removed (latency/VRAM benchmark). "
            "If omitted, each runner uses its own default."
        ),
    )
    p.add_argument(
        "--video-dir", default=None, type=Path,
        help=(
            "Root directory containing benchmark videos. "
            "Not needed for tasks that embed video bytes in HuggingFace "
            "(videomme, mvbench, nextqa, egoschema, mlvu, longvideobench). "
            "Required for: hlvid."
        ),
    )
    p.add_argument(
        "--hf-data-dir", default=None, type=Path,
        help=(
            "Local directory of a pre-downloaded HF dataset repo. "
            "Overrides the remote HF repo ID so the benchmark runs fully offline. "
            "Download once with: "
            "huggingface-cli download <repo> --repo-type dataset --local-dir <dir>"
        ),
    )
    p.add_argument(
        "--model-path", default="weights/NVILA-8B-HD-Video",
        help="Path to MLLM weights (NVILA default; set to HF hub ID for Qwen2.5-VL)",
    )
    p.add_argument(
        "--autogaze-path", default="weights/AutoGaze",
        help="Path to AutoGaze weights (ignored when --no-autogaze is set)",
    )
    p.add_argument(
        "--no-autogaze", action="store_true",
        help="Disable AutoGaze (full-patch baseline)",
    )
    p.add_argument(
        "--gazing-ratio", type=float, default=0.75,
        help="AutoGaze gazing ratio (0–1)",
    )
    p.add_argument(
        "--num-frames", type=int, default=16,
        help="Number of frames to sample per video",
    )
    p.add_argument(
        "--max-new-tokens", type=int, default=16,
        help="Max tokens for model generation (MCQ needs very few)",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="Output JSON path (default: results/{task}_{tag}.json)",
    )
    p.add_argument(
        "--max-samples", type=int, default=None,
        help="Cap dataset size (useful for smoke tests)",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Skip samples already in --output file",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    autogaze_path = None if args.no_autogaze else args.autogaze_path

    if args.output is None:
        ag_tag  = f"ag{int(args.gazing_ratio*100):03d}" if autogaze_path else "baseline"
        args.output = Path("results") / f"{args.task}_{args.mllm}_{ag_tag}.json"

    extra: Dict[str, Any] = {}
    if args.integration is not None:
        extra["integration"] = args.integration

    evaluate(
        task_name     = args.task,
        video_dir     = args.video_dir,
        model_path    = args.model_path,
        autogaze_path = autogaze_path,
        gazing_ratio  = args.gazing_ratio,
        num_frames    = args.num_frames,
        max_new_tokens= args.max_new_tokens,
        output_path   = args.output,
        max_samples   = args.max_samples,
        resume        = args.resume,
        mllm          = args.mllm,
        hf_data_dir   = args.hf_data_dir,
        runner_kwargs = extra,
    )


if __name__ == "__main__":
    main()
