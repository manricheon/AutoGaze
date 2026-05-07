"""
One-time data preparation: download bfshi/AutoGaze-Training-Data and
pre-tokenise frames into (T, 3, 224, 224) tensors + multi-hot masks.

Saves sharded .pt files under --out_dir for fast DataLoader access.

Usage:
    python -m mamba_gaze.scripts.prepare_data \
        --out_dir /data/autogaze_preprocessed \
        --split   train \
        --num_frames 16 \
        --num_shards 100 \
        --num_workers 16
"""

import argparse
import io
import os
import multiprocessing as mp
from pathlib import Path
from typing import Optional

import torch
import torchvision.transforms.v2 as T2

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


def _process_sample(args):
    idx, sample, num_frames, img_size, field_map = args
    from mamba_gaze.data.mask_converter import batch_seq_to_multihot, N_TOKENS

    transform = T2.Compose([
        T2.Resize((img_size, img_size)),
        T2.ToImage(),
        T2.ToDtype(torch.float32, scale=True),
        T2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    fm = field_map

    # Decode video
    from PIL import Image
    video_field = sample[fm["video"]]
    frames = []
    if isinstance(video_field, list):
        for f in video_field:
            if isinstance(f, bytes):
                frames.append(Image.open(io.BytesIO(f)).convert("RGB"))
            elif isinstance(f, Image.Image):
                frames.append(f.convert("RGB"))
            else:
                frames.append(f)
    elif isinstance(video_field, bytes):
        import torchvision.io as tio
        vf, _, _ = tio.read_video(io.BytesIO(video_field), output_format="TCHW")
        frames = [T2.functional.to_pil_image(vf[i]) for i in range(len(vf))]

    # Sample/pad frames
    n = len(frames)
    if n == 0:
        return None
    if n >= num_frames:
        step = n / num_frames
        frames = [frames[min(int(i * step), n - 1)] for i in range(num_frames)]
    else:
        frames = frames + [frames[-1]] * (num_frames - n)

    video = torch.stack([transform(f) for f in frames])   # (T, 3, H, W)

    # Parse gazing
    gazing_seq = sample.get(fm["gazing_seq"], [])
    if gazing_seq and isinstance(gazing_seq[0], (list, tuple)):
        per_frame = (list(gazing_seq) + [[]] * num_frames)[:num_frames]
        idx_list  = [torch.tensor(f, dtype=torch.long) for f in per_frame]
    else:
        flat  = torch.tensor(list(gazing_seq), dtype=torch.long) if gazing_seq else torch.empty(0, dtype=torch.long)
        chunk = max(1, len(flat) // num_frames)
        idx_list = [flat[i * chunk: (i + 1) * chunk] for i in range(num_frames)]

    gazing_mh = batch_seq_to_multihot(idx_list, N_TOKENS)   # (T, 265)

    # Recon loss
    recon_loss = sample.get(fm["recon_loss"], [0.0] * num_frames)
    recon_t = torch.tensor(list(recon_loss), dtype=torch.float32)
    if recon_t.shape[0] < num_frames:
        recon_t = torch.cat([recon_t, torch.zeros(num_frames - recon_t.shape[0])])
    else:
        recon_t = recon_t[:num_frames]

    return {"video": video, "gazing_multihot": gazing_mh, "recon_loss": recon_t}


def prepare(
    out_dir: str,
    split: str = "train",
    num_frames: int = 16,
    img_size: int = 224,
    num_shards: int = 100,
    num_workers: int = 8,
    cache_dir: Optional[str] = None,
    field_map: Optional[dict] = None,
):
    if load_dataset is None:
        raise ImportError("pip install datasets huggingface_hub")

    out_path = Path(out_dir) / split
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading bfshi/AutoGaze-Training-Data ({split}) ...")
    ds = load_dataset("bfshi/AutoGaze-Training-Data", split=split, cache_dir=cache_dir)
    N  = len(ds)
    print(f"  {N} samples found.")

    fm = {
        "video":      "video",
        "gazing_seq": "gazing_seq",
        "recon_loss": "recon_loss",
        **(field_map or {}),
    }

    shard_size = (N + num_shards - 1) // num_shards
    shard_idx  = 0
    buf: list  = []

    args_list = [
        (i, ds[i], num_frames, img_size, fm)
        for i in range(N)
    ]

    ctx = mp.get_context("fork")
    with ctx.Pool(num_workers) as pool:
        for i, result in enumerate(pool.imap(_process_sample, args_list, chunksize=32)):
            if result is None:
                continue
            buf.append(result)
            if len(buf) >= shard_size or i == N - 1:
                shard_path = out_path / f"shard_{shard_idx:04d}.pt"
                torch.save(buf, shard_path)
                print(f"  Saved shard {shard_idx:04d} ({len(buf)} samples) → {shard_path}")
                shard_idx += 1
                buf = []

    print(f"Done. {shard_idx} shards written to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir",     required=True)
    parser.add_argument("--split",       default="train")
    parser.add_argument("--num_frames",  type=int, default=16)
    parser.add_argument("--img_size",    type=int, default=224)
    parser.add_argument("--num_shards",  type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--cache_dir",   default=None)
    args = parser.parse_args()

    prepare(
        out_dir    = args.out_dir,
        split      = args.split,
        num_frames = args.num_frames,
        img_size   = args.img_size,
        num_shards = args.num_shards,
        num_workers= args.num_workers,
        cache_dir  = args.cache_dir,
    )


if __name__ == "__main__":
    main()
