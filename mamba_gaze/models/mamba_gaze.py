"""
MambaGaze: unified Mamba-based feedforward video token selector.

Replaces AutoGaze's autoregressive LLaMA decoder with:
  PatchEmbedder → MambaBackbone → MultiScaleSelectionHead
                                 → ReconPredictor (frame budget)

Output is AutoGaze-compatible: same gazing_mask / gazing_pos / etc. keys.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .patch_embedder  import PatchEmbedder
from .mamba_backbone  import MambaBackbone
from .selection_head  import MultiScaleSelectionHead
from .recon_predictor import ReconPredictor
from ..data.mask_converter import SCALE_PATCHES, SCALE_OFFSETS, N_TOKENS


def _get_gazing_pos_from_mask(mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    (B, N) binary mask → (B, K) gazing_pos, (B, K) if_padded.
    K = max selected per row; rows with fewer entries are padded with -1.
    """
    B, N = mask.shape
    mask = mask.long()
    idx  = torch.arange(N, device=mask.device).expand(B, N)
    key  = (1 - mask) * N + idx
    order = key.argsort(dim=1, stable=True)
    sorted_idx = idx.gather(1, order)

    counts = mask.sum(dim=1)                           # (B,)
    K = int(counts.max().item())
    if K == 0:
        empty = sorted_idx.new_empty(B, 0)
        return empty, empty.bool()

    topk = sorted_idx[:, :K]
    pos  = torch.arange(K, device=mask.device).expand(B, K)
    valid = pos < counts.unsqueeze(1)
    gazing_pos   = topk.masked_fill(~valid, -1)
    if_padded    = gazing_pos == -1
    return gazing_pos, if_padded


class MambaGaze(nn.Module):
    """
    Feedforward multi-scale video gaze token selector.

    Args:
        embed_dim:      Patch embedding / hidden dimension.
        img_size:       Input frame resolution (square).
        patch_size:     Patch stride (default 16 → 14×14 grid for 224px input).
        num_frames:     Temporal input length T.
        backbone_depth: Number of MambaLayer stacks (4–6).
        mamba_d_state:  SSM state dimension.
        mamba_d_conv:   Depthwise conv kernel in SSM.
        mamba_expand:   Inner-dim expansion factor.
        gazing_ratio:   Fraction of tokens to keep per scale.
        gumbel_temp:    Initial Gumbel temperature (annealed during training).
        frame_budget_eps: Skip frames with pred_recon < eps (0 = disabled).
    """

    SCALES = [32, 64, 112, 224]

    def __init__(
        self,
        embed_dim: int = 192,
        img_size: int = 224,
        patch_size: int = 16,
        num_frames: int = 16,
        backbone_depth: int = 4,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        gazing_ratio: float = 0.5,
        gumbel_temp: float = 1.0,
        frame_budget_eps: float = 0.0,
    ):
        super().__init__()
        h = w = img_size // patch_size
        self.num_frames        = num_frames
        self.img_size          = img_size
        self.patch_size        = patch_size
        self.frame_budget_eps  = frame_budget_eps

        # AutoGaze-compatible bookkeeping
        self.scales = self.SCALES
        self.num_vision_tokens_each_frame = N_TOKENS            # 265
        self.num_vision_tokens_each_scale_each_frame = SCALE_PATCHES  # [4,16,49,196]
        self.frame_sampling_rate = 1

        self.patch_embedder  = PatchEmbedder(embed_dim, img_size, patch_size)
        self.backbone        = MambaBackbone(embed_dim, backbone_depth,
                                             mamba_d_state, mamba_d_conv,
                                             mamba_expand, h, w)
        self.selection_head  = MultiScaleSelectionHead(embed_dim, gazing_ratio, gumbel_temp)
        self.recon_predictor = ReconPredictor(embed_dim)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        inputs: Dict,
        gazing_ratio: Optional[float] = None,
        temperature: Optional[float] = None,
        budget_eps: Optional[float] = None,
        teacher_mask: Optional[torch.Tensor] = None,   # (B, T, N_TOKENS) for distillation
        **kwargs,
    ) -> Dict:
        """
        Args:
            inputs:       dict with key ``'video'``: (B, T, C, H, W)
            gazing_ratio: fraction of tokens per scale to keep
            temperature:  Gumbel temperature (overrides model default)
            budget_eps:   frames with pred_recon < eps are skipped (None = disabled)
            teacher_mask: (B, T, 265) binary mask for optional BCE distillation loss

        Returns:
            AutoGaze-compatible dict; additional keys:
              ``per_scale_logits``, ``pred_recon_loss``, ``distill_loss``
        """
        video = inputs["video"]                                  # (B, T, C, H, W)
        B, T, C, H, W = video.shape

        # ── 1. Resize if needed ────────────────────────────────────────────────
        if H != self.img_size or W != self.img_size:
            video = F.interpolate(
                video.reshape(B * T, C, H, W),
                size=(self.img_size,) * 2, mode="bicubic", align_corners=False,
            ).reshape(B, T, C, self.img_size, self.img_size)

        # ── 2. Patch embedding ────────────────────────────────────────────────
        F_feat = self.patch_embedder(video)                      # (B, T, h, w, d)

        # ── 3. Mamba backbone ─────────────────────────────────────────────────
        H_feat = self.backbone(F_feat)                           # (B, T, h, w, d)

        # ── 4. Reconstruction loss prediction ─────────────────────────────────
        pred_recon = self.recon_predictor(H_feat)                # (B, T)

        # ── 5. Frame budget mask ──────────────────────────────────────────────
        eps = budget_eps if budget_eps is not None else self.frame_budget_eps
        frame_budget_mask = None
        if eps > 0:
            frame_budget_mask = (pred_recon > eps).float()       # (B, T)

        # ── 6. Multi-scale selection ──────────────────────────────────────────
        per_scale_logits, per_scale_masks = self.selection_head(
            H_feat, gazing_ratio, temperature, frame_budget_mask
        )
        # per_scale_masks[s]: (B, T, N_s)

        # ── 7. AutoGaze-compatible gazing_mask (list of per-scale tensors) ────
        gazing_mask: List[torch.Tensor] = per_scale_masks

        # ── 8. gazing_pos (from hard mask, no gradient) ───────────────────────
        with torch.no_grad():
            hard_scales = [(m.detach() > 0.5).float() for m in per_scale_masks]
            all_hard = torch.cat(hard_scales, dim=-1)            # (B, T, 265)
            # Flatten over T for per-frame pos extraction
            gazing_pos_list, if_padded_list = [], []
            num_gazing_each_frame = []
            for t in range(T):
                gp, ip = _get_gazing_pos_from_mask(all_hard[:, t])  # (B, K_t)
                gazing_pos_list.append(gp)
                if_padded_list.append(ip)
                K_t = (~ip).sum(dim=-1).max().item()
                num_gazing_each_frame.append(int(K_t))

            # Pad all frames to same K for stacking
            K_max = max(g.shape[1] for g in gazing_pos_list) if gazing_pos_list else 0
            if K_max > 0:
                def _pad(t_: torch.Tensor, K: int):
                    if t_.shape[1] < K:
                        pad = t_.new_full((B, K - t_.shape[1]), -1)
                        t_ = torch.cat([t_, pad], dim=1)
                    return t_
                gazing_pos = torch.cat([_pad(g, K_max) for g in gazing_pos_list], dim=1)
                if_padded  = gazing_pos == -1
            else:
                gazing_pos = all_hard.new_empty(B, 0, dtype=torch.long)
                if_padded  = gazing_pos.bool()
            num_gazing = torch.tensor(num_gazing_each_frame, device=video.device)

        # ── 9. Optional distillation loss ─────────────────────────────────────
        distill_loss = None
        if teacher_mask is not None:
            # teacher_mask: (B, T, 265) — split per scale for BCE
            combined_logits = torch.cat(per_scale_logits, dim=-1)  # (B, T, 265)
            distill_loss = F.binary_cross_entropy_with_logits(
                combined_logits, teacher_mask.float().to(combined_logits)
            )

        return {
            # AutoGaze-compatible
            "gazing_mask":                  gazing_mask,
            "gazing_pos":                   gazing_pos,
            "if_padded_gazing":             if_padded,
            "num_gazing_each_frame":        num_gazing,
            "scales":                       self.scales,
            "frame_sampling_rate":          self.frame_sampling_rate,
            "num_vision_tokens_each_frame": self.num_vision_tokens_each_frame,
            # MambaGaze-specific
            "per_scale_logits":   per_scale_logits,
            "pred_recon_loss":    pred_recon,
            "distill_loss":       distill_loss,
            # Stubs for full AutoGaze API
            "log_action_probs":            None,
            "task_loss_prediction":        None,
            "has_task_loss_requirement":   False,
            "task_loss_requirement":       None,
            "past_key_values":             None,
            "past_input_embeds":           None,
            "past_attention_mask":         None,
            "past_conv_values":            None,
        }

    # ── utilities ─────────────────────────────────────────────────────────────

    def set_gumbel_temperature(self, temp: float) -> None:
        self.selection_head.set_temperature(temp)

    def count_parameters(self) -> Dict[str, int]:
        def _n(m):
            return sum(p.numel() for p in m.parameters())
        return {
            "patch_embedder":   _n(self.patch_embedder),
            "backbone":         _n(self.backbone),
            "selection_head":   _n(self.selection_head),
            "recon_predictor":  _n(self.recon_predictor),
            "total":            _n(self),
        }

    @classmethod
    def from_config(cls, cfg: dict) -> "MambaGaze":
        m = cfg.get("model", {})
        s = cfg.get("selection", {})
        return cls(
            embed_dim       = m.get("embed_dim", 192),
            img_size        = m.get("img_size", 224),
            patch_size      = m.get("patch_size", 16),
            num_frames      = m.get("num_frames", 16),
            backbone_depth  = m.get("backbone_depth", 4),
            mamba_d_state   = m.get("mamba_d_state", 16),
            mamba_d_conv    = m.get("mamba_d_conv", 4),
            mamba_expand    = m.get("mamba_expand", 2),
            gazing_ratio    = s.get("default_gazing_ratio", 0.5),
            gumbel_temp     = s.get("gumbel_temp_init", 1.0),
            frame_budget_eps= s.get("frame_budget_eps", 0.0),
        )
