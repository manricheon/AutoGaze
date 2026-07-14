"""V-JEPA 2.1 teacher adapter, loaded from the official torch.hub repo
(facebookresearch/vjepa2). Training-only, like vjepa2_sparse.py.

Implements the same interface as vjepa2_sparse.VJEPA2Teacher
(`dense_features` / `sparse_features` / `predict`) so the trainer is
teacher-agnostic. Differences vs the HF wrapper, verified empirically on
`vjepa2_1_vit_base_384` (encoder 86.8M/768-d, predictor 22.9M):

- The original encoder supports sparse natively: `encoder(x, masks=[keep])`
  gathers the kept tokens AND drives RoPE with their original t-major flat
  indices — the same canonical `idx = t*N + n` order Borissal emits.
- Input layout is channels-first video (B, C, T, H, W); the adapter permutes
  from our (B, T, C, H, W).
- The straight-through gate is applied to the encoder's OUTPUT features
  (the original encoder has no clean input-embedding injection point,
  unlike our HF sparse forward). Forward values are unchanged (gate==1.0);
  it is purely a gradient conduit to the selector — equally valid.
- IMPORTANT: the 2.1 predictor projects to the DISTILLATION-TEACHER space
  (`predictor_proj`: -> 1664-d, the ViT-gigantic teacher it was distilled
  against; `return_all_tokens=True` so forward returns (x_pred, x_context)).
  Its output is NOT in the 2.1 encoder's own 768-d space, so the
  predictor-coverage loss needs 1664-d targets (a gigantic teacher's dense
  features) rather than this encoder's own dense features. See
  docs/borissal/training.md §5 for the wiring options.

torch.hub note: the repo's current main hardcodes a localhost checkpoint
base URL (dev leftover; the real one is commented out). Workaround: fetch
the checkpoint from https://dl.fbaipublicfiles.com/vjepa2/<file>.pt into
~/.cache/torch/hub/checkpoints/ first; torch.hub then uses the cached file.
"""

from typing import Optional

import torch
import torch.nn as nn

HUB_REPO = "facebookresearch/vjepa2"


class VJEPA21HubTeacher(nn.Module):
    """Frozen V-JEPA 2.1 (torch.hub) wrapped as an SSL teacher."""

    def __init__(self, encoder: nn.Module, predictor: nn.Module):
        super().__init__()
        if encoder.patch_size != 16 or encoder.tubelet_size != 2:
            raise ValueError(
                f"teacher grid mismatch: expected patch_size=16/tubelet_size=2, "
                f"got {encoder.patch_size}/{encoder.tubelet_size}"
            )
        encoder.eval()
        encoder.requires_grad_(False)
        predictor.eval()
        predictor.requires_grad_(False)
        self.encoder = encoder
        self.predictor = predictor

    @classmethod
    def from_hub(cls, entrypoint: str = "vjepa2_1_vit_base_384",
                 repo_dir: Optional[str] = None, **kwargs) -> "VJEPA21HubTeacher":
        """repo_dir: path to a LOCAL clone of facebookresearch/vjepa2 (a dir
        containing hubconf.py) -- required on offline machines, where the
        default source="github" cannot fetch the repo code. Checkpoints are
        still resolved from ~/.cache/torch/hub/checkpoints/ (pre-download
        them there; see the module docstring)."""
        if repo_dir:
            encoder, predictor = torch.hub.load(repo_dir, entrypoint, source="local",
                                                pretrained=True, **kwargs)
        else:
            encoder, predictor = torch.hub.load(HUB_REPO, entrypoint, pretrained=True, **kwargs)
        return cls(encoder, predictor)

    @property
    def hidden_size(self) -> int:
        return self.encoder.embed_dim

    @property
    def predictor_output_size(self) -> int:
        return self.predictor.predictor_proj.out_features

    @staticmethod
    def _to_bcthw(video: torch.Tensor) -> torch.Tensor:
        # ours: (B, T, C, H, W) -> original repo: (B, C, T, H, W)
        return video.permute(0, 2, 1, 3, 4).contiguous()

    @torch.no_grad()
    def dense_features(self, video: torch.Tensor) -> torch.Tensor:
        """video (B, T, C, H, W) -> (B, L, D) full-token encoder features."""
        return self.encoder(self._to_bcthw(video))

    def sparse_features(
        self, video: torch.Tensor, keep_index: torch.Tensor, gate: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """(B, K, D) features of the kept subset via the encoder's native mask
        path. Differentiable w.r.t. gate (output-feature gate, see module doc)."""
        if (keep_index < 0).any():
            raise ValueError("keep_index must be unpadded (no -1); use uniform allocation for training")
        feats = self.encoder(self._to_bcthw(video), masks=[keep_index])
        if gate is not None:
            feats = feats * gate.unsqueeze(-1)
        return feats

    def predict(
        self,
        sparse_feats: torch.Tensor,   # (B, K, D_enc) student features at context_idx
        context_idx: torch.Tensor,    # (B, K) long
        target_idx: torch.Tensor,     # (B, K_t) long
        num_tokens: int,              # unused (kept for interface parity with VJEPA2Teacher)
    ) -> torch.Tensor:
        """Predict target-token features from the sparse context.

        Returns (B, K_t, D_out) where D_out is the DISTILLATION-teacher space
        (1664 for the released 2.1 models), NOT the encoder's own dim -- see
        module docstring.
        """
        x_pred, _ = self.predictor(sparse_feats, masks_x=[context_idx], masks_y=[target_idx])
        return x_pred
