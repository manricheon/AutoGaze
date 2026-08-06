#!/usr/bin/env python
"""Diagnose SDPA vs FlashAttention-2 divergence on token-dropped inputs (CUDA).

Answers the question raised on 2026-07-28: downstream scores trend differently
by attention backend. Three possible layers:
  (1) normal numeric noise (accumulation order, bf16) -- expected ~1e-3..1e-2;
  (2) pruning amplifies numerics (OOD, fewer tokens, per-clip tiling);
  (3) a REAL varlen/mask bug: after the drop we rebuild cu_seqlens
      (attach_qwen3vl), and FA2 consumes it via flash_attn_varlen_func while
      SDPA assembles a block-diagonal mask from it -- if one backend
      misinterprets non-standard slice lengths, the two compute DIFFERENT
      attention patterns, not just different roundoff.

Protocol (per clip x ratio): run the vision tower under both backends on
  a) dense input      -> baseline divergence = layer (1)
  b) pruned input     -> if divergence >> dense divergence, layer (3): fix
                         the attach path before trusting ANY backend result.
fp32 SDPA is the reference; report max|diff| and cosine per (backend, input).

Usage (CUDA host):
  uv run python scripts/diag_attn_backend.py --videos-dir videos/internvid_eval16 \
      --limit 4 --ratios 0.25,0.5
Verdict rule of thumb (bf16): pruned/dense divergence ratio < 3x = numerics
(fix = pin one backend per experiment); > 10x = suspect the varlen path.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from autogaze.models.borissal import Borissal, BorissalConfig             # noqa: E402
from autogaze.models.borissal.adapters import to_qwen3vl_video_tokens     # noqa: E402
from autogaze.models.borissal.attach_qwen3vl import build_pruned_inputs   # noqa: E402
from autogaze.models.borissal.video_io import load_video, unnormalize     # noqa: E402


def _vision_tower(model_id, attn_impl, dtype, device):
    from transformers import AutoConfig
    from eval_mllm_attach import _model_class
    cfg = AutoConfig.from_pretrained(model_id)
    model = _model_class(cfg).from_pretrained(
        model_id, dtype=dtype, attn_implementation=attn_impl).to(device).eval()
    return model


@torch.no_grad()
def _tower_out(model, pixel_values, grid_thw, keep=None, patch_idx=None):
    """Vision-tower output for dense (keep=None) or pruned input, via the same
    build_pruned_inputs machinery the eval uses (leak-free encoder-stage drop)."""
    if keep is None:
        return model.model.visual(pixel_values, grid_thw=grid_thw)
    pruned = build_pruned_inputs(model, pixel_values, grid_thw, keep, patch_idx)
    return pruned.vision_embeds


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos-dir", default=str(REPO_ROOT / "videos" / "internvid_eval16"))
    p.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument("--ratios", default="0.25,0.5")
    p.add_argument("--limit", type=int, default=4)
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--scale", type=int, default=384)
    p.add_argument("--out", default=str(REPO_ROOT / "outputs" / "borissal" / "diag_attn.json"))
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required (FA2 has no CPU/MPS kernel)")

    from transformers import AutoProcessor
    device = torch.device("cuda")
    proc = AutoProcessor.from_pretrained(args.model)
    merge = proc.image_processor.merge_size

    towers = {
        "sdpa_fp32": _vision_tower(args.model, "sdpa", torch.float32, device),
        "sdpa_bf16": _vision_tower(args.model, "sdpa", torch.bfloat16, device),
        "fa2_bf16": _vision_tower(args.model, "flash_attention_2", torch.bfloat16, device),
    }
    ratios = [float(r) for r in args.ratios.split(",")]
    clips = sorted(Path(args.videos_dir).glob("*.mp4"))[: args.limit]
    report = []
    for path in clips:
        video = load_video(str(path), num_frames=args.num_frames, size=args.scale)
        rgb = unnormalize(video)[0].permute(0, 2, 3, 1).clamp(0, 1).float().cpu().numpy()
        rgb = (rgb * 255.0).round().clip(0, 255).astype("uint8")  # uint8: avoid double-rescale
        vin = proc(text=["x"], videos=[rgb], return_tensors="pt",
                   do_sample_frames=False, do_resize=False)
        assert float(vin["pixel_values_videos"].std()) > 0.05, "input scaling bug"
        pv = vin["pixel_values_videos"].to(device)
        grid = vin["video_grid_thw"].to(device)

        cases = {"dense": (None, None)}
        for ratio in ratios:
            sel = Borissal(BorissalConfig.v0_7(scale=args.scale)).select(video, gazing_ratio=ratio)
            tok = to_qwen3vl_video_tokens(sel, merge, "strict")
            cases[f"pruned@{ratio}"] = (tok["keep_token_index"].to(device),
                                        tok["qwen_patch_index"].to(device))
        row = {"clip": path.name}
        for case, (keep, pidx) in cases.items():
            outs = {}
            for name, tower in towers.items():
                pv_c = pv.to(torch.float32 if name.endswith("fp32") else torch.bfloat16)
                outs[name] = _tower_out(tower, pv_c, grid, keep, pidx).float()
            ref = outs["sdpa_fp32"]
            for name in ("sdpa_bf16", "fa2_bf16"):
                d = (outs[name] - ref).abs().max().item()
                cos = torch.nn.functional.cosine_similarity(
                    outs[name].flatten(), ref.flatten(), dim=0).item()
                row[f"{case}/{name}"] = {"max_abs_vs_fp32": round(d, 5),
                                         "cos_vs_fp32": round(cos, 6)}
            d_ab = (outs["sdpa_bf16"] - outs["fa2_bf16"]).abs().max().item()
            row[f"{case}/sdpa_vs_fa2"] = round(d_ab, 5)
            print(f"[{path.name}] {case}: sdpa-vs-fa2 max|diff|={d_ab:.4f}  "
                  f"(bf16 vs fp32: sdpa {row[f'{case}/sdpa_bf16']['max_abs_vs_fp32']:.4f}, "
                  f"fa2 {row[f'{case}/fa2_bf16']['max_abs_vs_fp32']:.4f})", flush=True)
        report.append(row)

    dense_d = [r["dense/sdpa_vs_fa2"] for r in report]
    pruned_d = [v for r in report for k, v in r.items()
                if k.startswith("pruned") and k.endswith("sdpa_vs_fa2")]
    ratio = (sum(pruned_d) / len(pruned_d)) / max(1e-9, sum(dense_d) / len(dense_d))
    verdict = ("VARLEN-PATH SUSPECT (fix attach before trusting either backend)"
               if ratio > 10 else
               "numerics (pin one backend per experiment; pre-registered: sdpa)")
    print(f"\npruned/dense divergence ratio = {ratio:.1f}x -> {verdict}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"clips": report, "divergence_ratio": round(ratio, 2), "verdict": verdict}, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
