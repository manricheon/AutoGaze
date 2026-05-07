"""
Downstream evaluation: VideoMME / MLVU via lmms-eval.

MambaGaze produces a gazing_mask.  We wrap any ViT-based LMM so that
its patch embeddings are zeroed for non-selected tokens (same technique
as autogaze/models/autogaze/autogaze_cv.py).

Usage:
    python -m mamba_gaze.eval.downstream \
        --config mamba_gaze/configs/default.yaml \
        --ckpt   checkpoints/phase2_ste/phase2ste_epoch0030.pt \
        --model  lmms_lab/LLaVA-Video-7B-Qwen2 \
        --task   videomme \
        --gazing_ratio 0.5
"""

import argparse
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

try:
    import yaml
except ImportError:
    yaml = None


# ── token masking hook (zero-out non-selected patches) ───────────────────────

@contextmanager
def token_mask_context(
    embed_module: nn.Module,
    mask: torch.Tensor,               # (B, N) bool — True = keep
    has_cls_token: bool = True,
):
    """
    Registers a forward hook that zeroes non-selected tokens on the output of
    ``embed_module`` (the ViT patch embedding layer).
    """
    def _hook(module, input, output):
        # output: (B, N+1, d) if has_cls or (B, N, d)
        if isinstance(output, (tuple, list)):
            h, *rest = output
        else:
            h, rest = output, None

        if has_cls_token:
            cls, patches = h[:, :1], h[:, 1:]
        else:
            patches = h

        B, N, d = patches.shape
        if mask.shape[-1] != N:
            # Interpolate mask if spatial resolutions differ
            from ..data.mask_converter import SCALE_HW
            # Use the 224-scale (14×14 = 196) if N < 265; else full 265
            m = mask.float().unsqueeze(1)
            tgt = int(N ** 0.5)
            m = torch.nn.functional.interpolate(
                m.reshape(B, 1, -1).unsqueeze(-1),
                size=(N, 1), mode="nearest"
            ).reshape(B, N)
        else:
            m = mask.float()

        patches = patches * m.unsqueeze(-1)

        if has_cls_token:
            out = torch.cat([cls, patches], dim=1)
        else:
            out = patches

        if rest is not None:
            return (out, *rest)
        return out

    handle = embed_module.register_forward_hook(_hook)
    try:
        yield
    finally:
        handle.remove()


# ── lmms-eval integration ─────────────────────────────────────────────────────

def run_lmms_eval(
    lmm_model_name: str,
    task: str,
    gaze_model: nn.Module,
    gazing_ratio: float,
    device: torch.device,
    output_path: str = "eval_results",
    num_frames: int = 16,
):
    """
    Launch lmms-eval with MambaGaze token selection applied via hook.
    Requires: pip install lmms-eval
    """
    try:
        import lmms_eval
    except ImportError:
        print("lmms-eval not installed. Run: pip install lmms-eval")
        return {}

    try:
        from lmms_eval import simple_evaluate
        from lmms_eval.models import get_model
    except ImportError as e:
        print(f"lmms-eval import error: {e}")
        return {}

    lmm = get_model(lmm_model_name)(device=str(device))

    # Find the patch embedding submodule
    embed_mod = _find_embed_module(lmm.model)
    if embed_mod is None:
        print("Warning: could not find patch embedding module; running without masking.")
        return simple_evaluate(lmm, tasks=[task], output_path=output_path)

    gaze_model.eval()

    # Monkey-patch the LMM's video encoding to inject gaze mask
    _original_encode = lmm.encode_video

    def _gaze_encode_video(video_tensor, *args, **kwargs):
        with torch.no_grad():
            out  = gaze_model({"video": video_tensor.to(device)}, gazing_ratio=gazing_ratio)
            # Combine per-scale masks → (B, 265) for the last frame (or mean over T)
            per_scale = out["gazing_mask"]
            mask_265  = torch.cat([m[:, -1] for m in per_scale], dim=-1).bool()  # (B, 265)

        with token_mask_context(embed_mod, mask_265[:, -196:]):   # use 224-scale portion
            return _original_encode(video_tensor, *args, **kwargs)

    lmm.encode_video = _gaze_encode_video

    results = simple_evaluate(lmm, tasks=[task], output_path=output_path)
    return results


def _find_embed_module(model: nn.Module) -> Optional[nn.Module]:
    """Heuristic: find the ViT patch embedding submodule."""
    candidates = ["embeddings", "patch_embed", "vision_model.embeddings",
                  "model.vision_tower.vision_model.embeddings"]
    for name in candidates:
        parts = name.split(".")
        m = model
        try:
            for p in parts:
                m = getattr(m, p)
            return m
        except AttributeError:
            continue
    return None


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",       default="mamba_gaze/configs/default.yaml")
    parser.add_argument("--ckpt",         required=True)
    parser.add_argument("--model",        default="lmms_lab/LLaVA-Video-7B-Qwen2",
                        help="HF model name for lmms-eval")
    parser.add_argument("--task",         default="videomme",
                        help="lmms-eval task name (videomme, mlvu, etc.)")
    parser.add_argument("--gazing_ratio", type=float, default=0.5)
    parser.add_argument("--output",       default="eval_results")
    args = parser.parse_args()

    cfg = {}
    if yaml and Path(args.config).exists():
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from ..models.mamba_gaze import MambaGaze
    gaze_model = MambaGaze.from_config(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    gaze_model.load_state_dict(ckpt.get("model", ckpt))
    print(f"Loaded {args.ckpt}")

    results = run_lmms_eval(
        lmm_model_name=args.model,
        task=args.task,
        gaze_model=gaze_model,
        gazing_ratio=args.gazing_ratio,
        device=device,
        output_path=args.output,
    )

    if results:
        print("\n── Downstream Eval Results ──────────────────────")
        for k, v in results.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
