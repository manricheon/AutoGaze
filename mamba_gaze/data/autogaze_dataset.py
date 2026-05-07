"""
Dataset loader for bfshi/AutoGaze-Training-Data (HuggingFace).

Expected HF dataset fields (configurable via field_map):
  video       : bytes | List[PIL.Image] | List[bytes]
  gazing_seq  : List[int] | List[List[int]]  — selected global token indices
  recon_loss  : List[float]                  — per-frame AutoGaze recon loss
"""

import io
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T2
from PIL import Image

from .mask_converter import batch_seq_to_multihot, N_TOKENS


class AutoGazeDataset(Dataset):
    """
    Wraps bfshi/AutoGaze-Training-Data for MambaGaze phase-1/2 training.

    Each sample returns::

        {
          "video"           : FloatTensor(T, 3, H, W)   — normalized to [-1, 1]
          "gazing_multihot" : FloatTensor(T, 265)        — multi-hot across all scales
          "recon_loss"      : FloatTensor(T,)            — teacher recon loss per frame
        }
    """

    DEFAULT_FIELDS: Dict[str, str] = {
        "video":      "video",
        "gazing_seq": "gazing_seq",
        "recon_loss": "recon_loss",
    }

    def __init__(
        self,
        split: str = "train",
        num_frames: int = 16,
        img_size: int = 224,
        normalize: bool = True,
        field_map: Optional[Dict[str, str]] = None,
        max_samples: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ):
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError("pip install datasets huggingface_hub") from e

        self.num_frames = num_frames
        self.n_tokens   = N_TOKENS
        self.fm = {**self.DEFAULT_FIELDS, **(field_map or {})}

        raw = load_dataset("bfshi/AutoGaze-Training-Data", split=split, cache_dir=cache_dir)
        self.data = raw.select(range(max_samples)) if max_samples else raw

        tfms = [T2.Resize((img_size, img_size)), T2.ToImage(),
                T2.ToDtype(torch.float32, scale=True)]
        if normalize:
            tfms.append(T2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        self.transform = T2.Compose(tfms)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.data[idx]

        frames = self._load_video(sample[self.fm["video"]])
        frames = self._sample_frames(frames)                    # exactly num_frames PIL images
        video  = torch.stack([self.transform(f) for f in frames])   # (T, 3, H, W)

        gazing_seq  = sample.get(self.fm["gazing_seq"],  [])
        recon_loss  = sample.get(self.fm["recon_loss"],  [0.0] * self.num_frames)

        gazing_mh = self._parse_gazing(gazing_seq)              # (T, 265)
        recon_t   = self._pad_or_trim(
            torch.tensor(list(recon_loss), dtype=torch.float32))

        return {"video": video, "gazing_multihot": gazing_mh, "recon_loss": recon_t}

    # ── internals ─────────────────────────────────────────────────────────────

    def _load_video(self, field) -> List[Image.Image]:
        if isinstance(field, list):
            out = []
            for item in field:
                if isinstance(item, bytes):
                    out.append(Image.open(io.BytesIO(item)).convert("RGB"))
                elif isinstance(item, Image.Image):
                    out.append(item.convert("RGB"))
                else:
                    out.append(item)
            return out
        if isinstance(field, bytes):
            try:
                import torchvision.io as tio
                vf, _, _ = tio.read_video(io.BytesIO(field), output_format="TCHW")
                return [T2.functional.to_pil_image(vf[i]) for i in range(len(vf))]
            except Exception:
                return [Image.open(io.BytesIO(field)).convert("RGB")]
        raise TypeError(f"Cannot decode video field of type {type(field)}")

    def _sample_frames(self, frames: List) -> List:
        n = len(frames)
        if n == 0:
            blank = Image.new("RGB", (224, 224))
            return [blank] * self.num_frames
        if n >= self.num_frames:
            step = n / self.num_frames
            return [frames[min(int(i * step), n - 1)] for i in range(self.num_frames)]
        return frames + [frames[-1]] * (self.num_frames - n)

    def _parse_gazing(self, gazing_seq) -> torch.Tensor:
        if not gazing_seq:
            return torch.zeros(self.num_frames, self.n_tokens)

        seq = list(gazing_seq)
        if seq and isinstance(seq[0], (list, tuple)):
            # List of T lists of indices
            per_frame = (seq + [[]] * self.num_frames)[: self.num_frames]
            idx_list  = [torch.tensor(f, dtype=torch.long) for f in per_frame]
        else:
            # Flat: evenly split among frames
            flat  = torch.tensor(seq, dtype=torch.long)
            chunk = max(1, len(flat) // self.num_frames)
            idx_list = [
                flat[i * chunk: (i + 1) * chunk] for i in range(self.num_frames)
            ]

        rows = batch_seq_to_multihot(idx_list, self.n_tokens)  # (T, 265)
        return rows

    def _pad_or_trim(self, t: torch.Tensor) -> torch.Tensor:
        if t.shape[0] >= self.num_frames:
            return t[: self.num_frames]
        pad = torch.zeros(self.num_frames - t.shape[0], dtype=t.dtype)
        return torch.cat([t, pad])


def build_dataloader(
    split: str,
    batch_size: int,
    num_workers: int = 8,
    pin_memory: bool = True,
    distributed: bool = False,
    **dataset_kwargs,
) -> DataLoader:
    ds = AutoGazeDataset(split=split, **dataset_kwargs)

    sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(ds, shuffle=(split == "train"))

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train" and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(split == "train"),
    )
