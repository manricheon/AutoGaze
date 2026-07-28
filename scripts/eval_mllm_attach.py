#!/usr/bin/env python
"""Downstream A/B: attach Borissal to a real MLLM by DROPPING vision tokens.

The judge every v0.x verdict has been waiting for. Instead of a SigLIP proxy
(which mis-ranked v0.4 and disagrees with v0.6 across the board), this measures
the description task itself: how much worse does the captioner get when it only
receives the selected tokens?

PRIMARY METRIC -- `nll`: teacher-forced negative log-likelihood of the DENSE
caption under each pruned input. One forward pass, no sampling, no judge model,
and no ground-truth captions needed (the dense run supplies the reference), so
it is deterministic and far lower-variance than generate-then-score. This is the
"Frame-Voyager-style caption-loss ranking" that design.md deferred. Lower is
better; `dense` is the floor by construction.

Report `nll_delta = nll(config) - nll(dense)`: the description-relevant
information the selection threw away, in nats/token.

PRUNE STAGE (see attach_qwen3vl):
  --prune-stage encoder  DEFAULT and the actual method: only selected patches
                         enter the ViT, exactly as AutoGaze prunes before NVILA's
                         SigLIP. Leak-free. Needs whole 2x2 blocks
                         (score_coarsen=2: v0.5/v0.6).
  --prune-stage llm      diagnostic only: full ViT, drop before the LLM. Surviving
                         tokens already attended to the dropped ones, so a good
                         score proves nothing about the discarded pixels. Useful
                         only to quantify how much the ViT smuggles through.

Usage (CUDA is the intended host; --limit 1 --smoke is the Mac plumbing check):
  uv run python scripts/eval_mllm_attach.py --videos-dir videos/internvid_eval16 \
      --configs v0.3,v0.5,v0.6,v0.6-static,random --ratios 0.25,0.5 \
      --generate
Outputs: outputs/borissal/mllm_attach/{results.json, nll_vs_ratio.png, captions.json}
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from autogaze.models.borissal import Borissal, BorissalConfig            # noqa: E402
from autogaze.models.borissal.adapters import to_qwen3vl_video_tokens    # noqa: E402
from autogaze.models.borissal.attach_qwen3vl import (                    # noqa: E402
    build_pruned_inputs, pruned_forward,
)
from autogaze.models.borissal.device import resolve_device               # noqa: E402
from autogaze.models.borissal.video_io import load_video, unnormalize    # noqa: E402

OUT_DIR = REPO_ROOT / "outputs" / "borissal" / "mllm_attach"
PROMPT = "Describe this video in detail: the setting, the objects, and what happens."

# Selector configs. `dense` = no pruning (the reference); `random` = same token
# budget, uniformly random merged tokens (the control that tells you whether the
# selector is doing anything at all).
SELECTORS = {
    "v0.3": lambda s: BorissalConfig.v0_3(scale=s),
    # v0.4 (frame-rate-aware motion) LOST against V-JEPA+Qwen, but that downstream's
    # encoder is a temporal model, where motion is redundant. On a per-frame or
    # weakly-temporal encoder the selected patches are the ONLY route for motion
    # information, so v0.4 is expected to do better -- the pre-registered hypothesis
    # in docs/borissal/downstream-stacks.md. Keep it in every sweep.
    "v0.4": lambda s: BorissalConfig.v0_4(scale=s),
    "v0.5": lambda s: BorissalConfig.v0_5(scale=s),
    "v0.6": lambda s: BorissalConfig.v0_6(scale=s),                      # all-on default (global alloc)
    # proxy-best variant: static_guard alone (the only knob that won the SigLIP
    # screen), everything else back to v0.5 behaviour
    "v0.6-static": lambda s: BorissalConfig.v0_6(
        scale=s, static_guard=True, laplacian_gate=False, center_bias=0.0,
        keyframe_prior=False, per_frame_allocation="uniform", luma_mode="mean"),
    "v0.6-uniform": lambda s: BorissalConfig.v0_6(scale=s, per_frame_allocation="uniform"),
    # v0.7 "Datdol" anchor-novelty: motion = when to update, appearance = what
    # to represent. Whole-cube selection -> partial_blocks strict-safe.
    "v0.7": lambda s: BorissalConfig.v0_7(scale=s),
    # E-B(review): full-site coverage even at low budgets -- K_a = min(K_cubes, Sc)
    "v0.7-cov": lambda s: BorissalConfig.v0_7(scale=s, anchor_fraction=1.0),
    # signal_grid comparison: "fine" = original 24x24 signals (v0.7 default is now "cube")
    "v0.7-fine": lambda s: BorissalConfig.v0_7(scale=s, signal_grid="fine"),
}


def _parse_spec(name):
    """'base,k=v,k=v' -> (base, overrides dict). Values cast like
    eval_borissal_semantic.build_selection: float if '.', else int, else str."""
    base, _, ov = name.partition(",")
    overrides = {}
    for kv in filter(None, ov.split(",")):
        k, val = kv.split("=", 1)
        try:
            overrides[k.strip()] = float(val) if "." in val else int(val)
        except ValueError:
            overrides[k.strip()] = val.strip()
    return base, overrides


def _selector_config(name, scale):
    base, overrides = _parse_spec(name)
    cfg = SELECTORS[base](scale)
    if overrides:
        cfg = type(cfg)(**{**cfg.__dict__, **overrides})
    return cfg


def _token_selection(name, video, scale, ratio, merge, partial_blocks, generator):
    """-> (keep_token_index (1,K), qwen_patch_index (1,K*m^2) or None, n_partial)."""
    if name == "random":
        # control: same budget, random merged tokens. Built directly in token
        # space so it is exactly comparable to a whole-block selection.
        T, H, W = video.shape[1], video.shape[3], video.shape[4]
        Tg, Hm, Wm = T // 2, H // 16 // merge, W // 16 // merge
        n_tok = Tg * Hm * Wm
        k = max(1, round(ratio * n_tok))
        idx = torch.randperm(n_tok, generator=generator)[:k].sort().values.unsqueeze(0)
        return idx, (idx[0][:, None] * merge * merge + torch.arange(merge * merge)).reshape(1, -1), 0
    if name.startswith("coarse:"):
        # 12x12-native signals+selection, expanded 2x back to the patch-16 grid
        import sys as _sys
        _sys.path.insert(0, str(REPO_ROOT / "scripts"))
        from eval_borissal_semantic import expand_selection_2x
        cfg = _selector_config(name[len("coarse:"):], scale)
        cfg = type(cfg)(**{**cfg.__dict__, "patch_size": 32})
        sel = expand_selection_2x(Borissal(cfg).select(video, gazing_ratio=ratio))
    else:
        cfg = _selector_config(name, scale)
        sel = Borissal(cfg).select(video, gazing_ratio=ratio)
    out = to_qwen3vl_video_tokens(sel, merge, partial_blocks)
    return out["keep_token_index"], out["qwen_patch_index"], out["n_partial_blocks"]


def _processor_inputs(proc, rgb_frames, text):
    return proc(text=[text], videos=[rgb_frames], return_tensors="pt",
                do_sample_frames=False, do_resize=False)


@torch.no_grad()
def _dense_caption(model, proc, inputs, max_new_tokens):
    gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new = gen[0, inputs["input_ids"].shape[1]:]
    return proc.tokenizer.decode(new, skip_special_tokens=True).strip()


@torch.no_grad()
def _caption_nll(model, inputs_with_caption, n_caption_tokens, keep, patch_idx, stage):
    """Teacher-forced NLL (nats/token) of the caption tokens only."""
    pruned = build_pruned_inputs(
        model, inputs_with_caption, keep, prune_stage=stage,
        qwen_patch_index=patch_idx if stage == "encoder" else None)
    labels = torch.full_like(pruned.input_ids, -100)
    labels[:, -n_caption_tokens:] = pruned.input_ids[:, -n_caption_tokens:]
    _, loss = pruned_forward(model, pruned, labels=labels)
    return float(loss), pruned.n_vision_tokens, pruned.n_vision_tokens_dense


@torch.no_grad()
def _greedy_generate_pruned(model, inputs, keep, patch_idx, stage, max_new_tokens, tokenizer):
    """Greedy decode on a pruned prompt (no KV cache: the pruned path assembles
    inputs_embeds/position_ids itself, and re-running a short prompt is cheap
    enough for an offline A/B). Text tokens get the next mrope position, shared
    across all rope dims -- which is exactly how the dense path positions text."""
    pruned = build_pruned_inputs(model, inputs, keep, prune_stage=stage,
                                qwen_patch_index=patch_idx if stage == "encoder" else None)
    embed = (model.model if hasattr(model, "model") else model).get_input_embeddings()
    out_ids = []
    for _ in range(max_new_tokens):
        logits, _ = pruned_forward(model, pruned)
        nxt = int(logits[0, -1].argmax())
        if nxt in (tokenizer.eos_token_id, getattr(model.config, "eos_token_id", None)):
            break
        out_ids.append(nxt)
        tok = torch.tensor([[nxt]], device=pruned.input_ids.device)
        pruned.input_ids = torch.cat([pruned.input_ids, tok], dim=1)
        pruned.inputs_embeds = torch.cat([pruned.inputs_embeds, embed(tok)], dim=1)
        pruned.attention_mask = torch.cat(
            [pruned.attention_mask, torch.ones_like(tok)], dim=1)
        if pruned.position_ids is not None:
            nxt_pos = pruned.position_ids[..., -1:] + 1
            pruned.position_ids = torch.cat([pruned.position_ids, nxt_pos], dim=-1)
        pruned.visual_pos_masks = torch.cat(
            [pruned.visual_pos_masks, torch.zeros_like(tok, dtype=torch.bool)], dim=1)
    return tokenizer.decode(out_ids, skip_special_tokens=True).strip()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos-dir", default=str(REPO_ROOT / "videos" / "internvid_eval16"))
    p.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument("--configs", default="v0.3,v0.4,v0.5,v0.6,random")
    p.add_argument("--ratios", default="0.25")
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--scale", type=int, default=384)
    p.add_argument("--prune-stage", choices=["encoder", "llm"], default="encoder",
                   help="'encoder' IS the method (select before the vision tower, as AutoGaze "
                        "does before NVILA's SigLIP); 'llm' is a diagnostic upper bound only")
    p.add_argument("--partial-blocks", choices=["strict", "any", "full"], default="any",
                   help="'any' by default so v0.3 (score_coarsen=1) is runnable; the realised "
                        "token count is always reported, so read n_tokens, not the ratio")
    p.add_argument("--limit", type=int, default=0, help="0 = all clips")
    p.add_argument("--clips-file", default=None,
                   help="text file with one clip filename per line (e.g. a dev split "
                        "extracted from docs/borissal/evalset_manifest.json); restricts "
                        "--videos-dir to exactly those clips, in file order")
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--generate", action="store_true", help="also greedy-decode each pruned config")
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default=None, choices=[None, "float32", "bfloat16", "float16"])
    p.add_argument("--smoke", action="store_true", help="plumbing only: 1 clip, 16 new tokens")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=str(OUT_DIR))
    args = p.parse_args()

    from transformers import AutoConfig, AutoProcessor

    if args.smoke:
        args.limit, args.max_new_tokens = 1, 16
    device = resolve_device(args.device or "auto")
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
             "float16": torch.float16}.get(args.dtype) or (
        torch.float32 if device.type == "cpu" else torch.bfloat16)
    ratios = [float(r) for r in args.ratios.split(",") if r]
    # ';' separates specs so a spec can carry ',k=v' overrides
    # (e.g. "v0.7;v0.7,anchor_fraction=0.75;random"); a plain comma list
    # without '=' keeps working as before.
    sep = ";" if (";" in args.configs or "=" in args.configs) else ","
    configs = [c.strip() for c in args.configs.split(sep) if c.strip()]

    def _known(c):
        if c == "random":
            return True
        if c.startswith("coarse:"):
            c = c[len("coarse:"):]
        return _parse_spec(c)[0] in SELECTORS

    unknown = [c for c in configs if not _known(c)]
    if unknown:
        raise SystemExit(f"unknown configs {unknown}; known: {sorted(SELECTORS) + ['random']} "
                         f"(+ ',k=v' overrides, ';'-separated)")

    cfg = AutoConfig.from_pretrained(args.model)
    merge = cfg.vision_config.spatial_merge_size
    patch = cfg.vision_config.patch_size
    if args.scale % (patch * merge):
        raise SystemExit(
            f"--scale {args.scale} must be a multiple of patch*merge = {patch * merge} so the "
            f"selector grid and the processor grid coincide")
    proc = AutoProcessor.from_pretrained(args.model)
    print(f"loading {args.model} ({dtype}) on {device} ...", flush=True)
    model_cls = _model_class(cfg)
    model = model_cls.from_pretrained(args.model, dtype=dtype).to(device).eval()

    clips = sorted(Path(args.videos_dir).glob("*.mp4"))
    if args.clips_file:
        wanted = [n.strip() for n in Path(args.clips_file).read_text().splitlines() if n.strip()]
        by_name = {c.name: c for c in clips}
        missing = [n for n in wanted if n not in by_name]
        if missing:
            raise SystemExit(f"--clips-file names not in --videos-dir: {missing[:5]}")
        clips = [by_name[n] for n in wanted]
    if args.limit:
        clips = clips[: args.limit]
    if not clips:
        raise SystemExit(f"no .mp4 found under {args.videos_dir}")
    print(f"{len(clips)} clips x {len(configs)} configs x {len(ratios)} ratios, "
          f"stage={args.prune_stage}", flush=True)

    gen = torch.Generator().manual_seed(args.seed)
    per_clip, captions = [], []
    t_start = time.time()
    for ci, path in enumerate(clips):
        video = load_video(str(path), num_frames=args.num_frames, size=args.scale)
        rgb = unnormalize(video)[0].permute(0, 2, 3, 1).clamp(0, 1).float().cpu().numpy()

        prompt = proc.apply_chat_template(
            [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": PROMPT}]}],
            tokenize=False, add_generation_prompt=True)
        inputs = _processor_inputs(proc, rgb, prompt).to(device)
        ref_caption = _dense_caption(model, proc, inputs, args.max_new_tokens)
        if not ref_caption:
            print(f"  [{path.name}] empty dense caption -- skipped")
            continue

        with_cap = _processor_inputs(proc, rgb, prompt + ref_caption).to(device)
        n_cap = int(with_cap["input_ids"].shape[1] - inputs["input_ids"].shape[1])
        if n_cap <= 0:
            print(f"  [{path.name}] caption tokenized to 0 tokens -- skipped")
            continue

        row = {"clip": path.name, "n_caption_tokens": n_cap, "dense_caption": ref_caption,
               "results": {}}
        nll_dense, _, n_dense_tok = _caption_nll(model, with_cap, n_cap, None, None, "llm")
        row["results"]["dense"] = {"nll": nll_dense, "n_tokens": n_dense_tok,
                                  "n_tokens_dense": n_dense_tok, "ratio": 1.0}
        for name in configs:
            for ratio in ratios:
                keep, patch_idx, n_partial = _token_selection(
                    name, video, args.scale, ratio, merge, args.partial_blocks, gen)
                keep, patch_idx = keep.to(device), patch_idx.to(device)
                t0 = time.time()
                nll, n_tok, n_all = _caption_nll(
                    model, with_cap, n_cap, keep, patch_idx, args.prune_stage)
                entry = {"nll": nll, "nll_delta": nll - nll_dense, "n_tokens": n_tok,
                         "n_tokens_dense": n_all, "realised_ratio": n_tok / max(1, n_all),
                         "ratio": ratio, "n_partial_blocks": n_partial,
                         "sec": round(time.time() - t0, 3)}
                if args.generate:
                    entry["caption"] = _greedy_generate_pruned(
                        model, inputs.to(device), keep, patch_idx, args.prune_stage,
                        args.max_new_tokens, proc.tokenizer)
                row["results"][f"{name}@{ratio}"] = entry
        per_clip.append(row)
        captions.append({"clip": path.name, "dense": ref_caption,
                         **{k: v.get("caption") for k, v in row["results"].items() if "caption" in v}})
        best = min(((k, v["nll"]) for k, v in row["results"].items() if k != "dense"),
                   key=lambda kv: kv[1], default=("-", float("nan")))
        print(f"  [{ci + 1}/{len(clips)}] {path.name}: dense nll {nll_dense:.4f}, "
              f"best {best[0]} {best[1]:.4f}", flush=True)

    if not per_clip:
        raise SystemExit("no clip produced a usable measurement")

    keys = [k for k in per_clip[0]["results"]]
    summary = {}
    for k in keys:
        vals = [c["results"][k]["nll"] for c in per_clip if k in c["results"]]
        deltas = [c["results"][k].get("nll_delta", 0.0) for c in per_clip if k in c["results"]]
        toks = [c["results"][k]["n_tokens"] for c in per_clip if k in c["results"]]
        summary[k] = {
            "nll_mean": sum(vals) / len(vals),
            "nll_delta_mean": sum(deltas) / len(deltas),
            "n_tokens_mean": sum(toks) / len(toks),
            "n_clips": len(vals),
            # paired win/loss vs dense is meaningless (dense is the floor); the
            # useful pairing is config-vs-config, done in the report step
        }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"args": vars(args), "summary": summary, "per_clip": per_clip}
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2))
    (out_dir / "captions.json").write_text(json.dumps(captions, indent=2))

    print(f"\n{'config':22s} {'nll':>8s} {'d(nll)':>9s} {'tokens':>8s}  n")
    print("-" * 60)
    for k, v in sorted(summary.items(), key=lambda kv: kv[1]["nll_mean"]):
        print(f"{k:22s} {v['nll_mean']:8.4f} {v['nll_delta_mean']:+9.4f} "
              f"{v['n_tokens_mean']:8.1f}  {v['n_clips']}")
    _plot(summary, out_dir / "nll_vs_ratio.png", args)
    print(f"\nwrote {out_dir}/results.json  ({time.time() - t_start:.0f}s)")
    print("Lower nll = better. Read n_tokens, not the requested ratio "
          "(partial-block policy changes the realised budget).")


def _model_class(cfg):
    """Qwen3-VL and Qwen3.5 expose different conditional-generation classes."""
    import transformers
    for name in (f"{cfg.architectures[0]}" if getattr(cfg, "architectures", None) else "",
                 "Qwen3VLForConditionalGeneration"):
        if name and hasattr(transformers, name):
            return getattr(transformers, name)
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM


def _plot(summary, path, args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    labels = [k for k in summary if k != "dense"]
    if not labels:
        return
    labels.sort(key=lambda k: summary[k]["nll_mean"])
    vals = [summary[k]["nll_delta_mean"] for k in labels]
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(labels) + 3), 4.2))
    ax.bar(range(len(labels)), vals, color="#5B6B7B")
    ax.axhline(0, color="#C0392B", lw=1, label="dense (no pruning)")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("caption NLL increase vs dense (nats/token)")
    ax.set_title(f"Borissal -> {args.model.split('/')[-1]}, prune_stage={args.prune_stage}\n"
                 f"lower = less description information lost")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
