#!/usr/bin/env python3
"""
Extract video files from a HuggingFace dataset to a local directory.

Use this when the dataset's video bytes column returns {"bytes": None, ...}
(i.e. the HF cache is incomplete or the bytes haven't been downloaded yet).

After extraction, pass --video-dir to run_benchmark.py as a fallback source.

Usage
-----
# Extract all videos from a task's dataset
python scripts/extract_hf_videos.py --task videomme --out data/videomme_videos

# Load from a pre-downloaded local dataset repo instead of HF hub
python scripts/extract_hf_videos.py --task mvbench \
    --hf-data-dir data/MVBench \
    --out data/mvbench_videos

# Smoke test: only first 20 samples
python scripts/extract_hf_videos.py --task nextqa --out data/nextqa_videos --max 20

Download commands (run once before this script)
-----------------------------------------------
huggingface-cli download lmms-lab/Video-MME        --repo-type dataset --local-dir data/Video-MME
huggingface-cli download OpenGVLab/MVBench         --repo-type dataset --local-dir data/MVBench
huggingface-cli download lmms-lab/NExTQA           --repo-type dataset --local-dir data/NExTQA
huggingface-cli download lmms-lab/EgoSchema        --repo-type dataset --local-dir data/EgoSchema
huggingface-cli download MLVU/MLVU                 --repo-type dataset --local-dir data/MLVU
huggingface-cli download longvideobench/LongVideoBench --repo-type dataset --local-dir data/LongVideoBench

Then run this script with --hf-data-dir pointing to the downloaded folder.

After extraction, run the benchmark with --video-dir:
python -m autogaze.eval.run_benchmark \
    --task videomme \
    --video-dir data/videomme_videos \
    --mllm nvila ...
"""

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract HF dataset video bytes to local mp4 files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task",        required=True, help="Task name, e.g. videomme")
    parser.add_argument("--out",         required=True, type=Path, help="Output directory for video files")
    parser.add_argument("--hf-data-dir", default=None,  type=Path,
                        help="Pre-downloaded local dataset repo dir (skips HF hub download)")
    parser.add_argument("--max",         default=None,  type=int,  help="Cap number of samples (for testing)")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("pip install datasets")

    try:
        from autogaze.eval.tasks import TASKS
    except ImportError:
        sys.exit("Run from the AutoGaze repo root: python scripts/extract_hf_videos.py ...")

    if args.task not in TASKS:
        sys.exit(f"Unknown task '{args.task}'. Available: {sorted(TASKS.keys())}")

    task = TASKS[args.task]
    if task.video_bytes_col is None:
        sys.exit(
            f"Task '{args.task}' has no video_bytes_col — videos must be downloaded "
            "manually (e.g. via scripts/download_hlvid.sh for hlvid)."
        )

    hf_source = str(args.hf_data_dir) if args.hf_data_dir else task.hf_repo
    print(f"Loading {hf_source}  split={task.hf_split}")
    ds = load_dataset(hf_source, split=task.hf_split, **task.hf_kwargs)
    if args.max:
        ds = ds.select(range(min(args.max, len(ds))))
    print(f"  {len(ds)} samples")

    args.out.mkdir(parents=True, exist_ok=True)

    n_saved = n_skip = n_exist = 0
    for i, sample in enumerate(ds):
        raw = sample.get(task.video_bytes_col)

        # Derive a clean filename from the video column or the path field
        video_id = str(sample[task.video_col])
        if isinstance(raw, dict) and raw.get("path"):
            # Use the leaf filename from the HF path (e.g. "video/Action/1.mp4" → "1.mp4")
            # Keep the relative sub-path if it contains directory structure, so that
            # _resolve_video_path can find it via the full relative path candidate.
            hf_path = Path(raw["path"])
            # Store as: out_dir / relative_path (preserves sub-folder structure)
            out_path = args.out / hf_path
        else:
            # Flat layout: out_dir / video_id.ext
            safe_id = Path(video_id).stem if Path(video_id).suffix else video_id
            out_path = args.out / (safe_id + task.video_ext)

        if out_path.exists():
            n_exist += 1
            if i % 500 == 0:
                print(f"  [{i}/{len(ds)}] already exists: {out_path.name}")
            continue

        # Extract bytes
        data: bytes | None = None
        if isinstance(raw, dict):
            data = raw.get("bytes")
        elif isinstance(raw, (bytes, bytearray)):
            data = bytes(raw)

        if not data:
            print(f"  [{i}/{len(ds)}] SKIP — no bytes: video_id={video_id!r}")
            n_skip += 1
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data)
        n_saved += 1

        if i % 100 == 0:
            print(f"  [{i}/{len(ds)}] saved {out_path}")

    print()
    print(f"Done.  saved={n_saved}  skipped={n_skip}  already_existed={n_exist}")
    print(f"Output directory: {args.out.resolve()}")
    if n_skip > 0:
        print()
        print(f"WARNING: {n_skip} samples had no bytes.")
        print("  → Re-run with --hf-data-dir pointing to the fully downloaded repo, or")
        print("    download videos for those IDs separately and place them in the output dir.")
    print()
    print("Run benchmark with:")
    print(f"  python -m autogaze.eval.run_benchmark \\")
    print(f"      --task {args.task} \\")
    print(f"      --video-dir {args.out} \\")
    print(f"      --mllm nvila ...")


if __name__ == "__main__":
    main()
