#!/usr/bin/env python
"""Mobile-export pre-check for Borissal selectors (trace + ONNX).

Attempts, for v0 (v0.2 preset) and v1 (current defaults: cosine head +
global context), with a FIXED input shape and uniform allocation (the
export-safe configuration per the design.md mobile review):

  1. torch.jit.trace of a tensor-only select() wrapper, + output equality
     re-check on a second input (catches silently-baked-in constants)
  2. torch.onnx.export (opset 17) + onnx.checker

CoreML conversion (coremltools) is intentionally out of scope here — run
onnx -> coreml on the CI/Linux side; this script's job is catching
trace/op problems early on the dev box.

Usage:
    uv run python scripts/export_borissal_check.py            # both models
    uv run python scripts/export_borissal_check.py --scale 256
Outputs a PASS/FAIL table; artifacts land in outputs/borissal/export/ (gitignored).
"""

import argparse
import traceback
from pathlib import Path

import torch

from autogaze.models.borissal import Borissal, BorissalConfig, BorissalV1, BorissalV1Config

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "outputs" / "borissal" / "export"


class _SelectWrapper(torch.nn.Module):
    """Tensor-only facade over select() for trace/onnx (fixed ratio/alloc/spread)."""

    def __init__(self, model, ratio: float, alloc: str = "uniform", spread: float = 0.0):
        super().__init__()
        self.model = model
        self.ratio = ratio
        self.alloc = alloc
        self.spread = spread

    def forward(self, video):
        sel = self.model.select(video, gazing_ratio=self.ratio,
                                per_frame_allocation=self.alloc, spread_fraction=self.spread)
        return sel.scores, sel.keep_mask, sel.keep_index


def check_trace(wrapper, video, video2):
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, video, check_trace=False)
        # re-run on DIFFERENT content: catches constants silently baked in
        eager = wrapper(video2)
        tr = traced(video2)
        for i, (a, b) in enumerate(zip(eager, tr)):
            if not torch.equal(a if a.dtype == torch.bool else a, b):
                if a.dtype.is_floating_point and torch.allclose(a, b, atol=1e-5):
                    continue
                return False, f"output {i} mismatch on second input (baked-in constant?)"
    return True, "ok"


def check_onnx(wrapper, video, path):
    import onnx
    with torch.no_grad():
        torch.onnx.export(
            wrapper, (video,), str(path), opset_version=17,
            input_names=["video"], output_names=["scores", "keep_mask", "keep_index"],
            dynamo=False,
        )
    onnx.checker.check_model(onnx.load(str(path)))
    return True, f"ok ({path.stat().st_size / 1e6:.1f} MB)"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scale", type=int, default=384)
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--ratio", type=float, default=0.25)
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(0)
    video = torch.rand(1, args.num_frames, 3, args.scale, args.scale, generator=g)
    video2 = torch.rand(1, args.num_frames, 3, args.scale, args.scale, generator=g)

    v1 = BorissalV1(BorissalV1Config(scale=args.scale)).eval()
    cases = {
        "v0.2": (Borissal(BorissalConfig.v0_2(scale=args.scale)).eval(), "uniform", 0.0),
        "v1": (v1, "uniform", 0.0),
        # hybrid focus+spread over global allocation (K_total fixed -> static shapes)
        "v1-hyb": (v1, "global", 0.25),
        # the adopted v0.3 preset (sweep winners: peak fusion + coherence ds=4
        # + DoG blob -- verdicts in docs/borissal/design.md)
        "v0.3": (Borissal(BorissalConfig.v0_3(scale=args.scale)).eval(),
                 "uniform", 0.0),
    }

    rows = []
    for name, (model, alloc, spread) in cases.items():
        wrapper = _SelectWrapper(model, args.ratio, alloc=alloc, spread=spread)
        for check, fn in [("jit.trace", lambda w: check_trace(w, video, video2)),
                          ("onnx", lambda w: check_onnx(w, video, OUT_DIR / f"borissal_{name.replace('.', '')}.onnx"))]:
            try:
                ok, msg = fn(wrapper)
            except Exception as e:
                ok, msg = False, f"{type(e).__name__}: {e}"
                (OUT_DIR / f"{name}_{check}_error.txt").write_text(traceback.format_exc())
            rows.append((name, check, "PASS" if ok else "FAIL", msg))

    print(f"\n{'model':6} {'check':10} {'result':7} detail")
    print("-" * 70)
    for r in rows:
        print(f"{r[0]:6} {r[1]:10} {r[2]:7} {r[3]}")


if __name__ == "__main__":
    main()
