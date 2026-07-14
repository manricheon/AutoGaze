#!/usr/bin/env python
"""Self-supervised training for Borissal v1 against a frozen V-JEPA2 teacher.

Single-device by default (Mac CPU/MPS smoke); becomes multi-GPU DDP
automatically under torchrun (torchrun --nproc_per_node=N scripts/train_borissal_v1.py ...).
See docs/borissal/training.md for the objective rationale, loss-combination
matrix, and full run recipes.

Examples:
    # Mac smoke: tiny random teacher, duplicated example clip, ~30 steps
    uv run python scripts/train_borissal_v1.py --smoke

    # Real teacher (patch16/tubelet2 asserted), local folder of mp4s
    uv run python scripts/train_borissal_v1.py \
        --data-root <folder> --teacher <hf_id_or_local_path> \
        --scale 384 --steps 200 --batch-size 2
"""

import argparse
import json
import os
import resource
import shutil
import tempfile
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from autogaze.models.borissal import BorissalV1, BorissalV1Config, MODEL_TAG_V1, resolve_device
from autogaze.models.borissal.data import VideoFolderDataset
from autogaze.models.borissal.losses import (
    LossWeights,
    combine_losses,
    dense_sparse_match_loss,
    predictor_coverage_loss,
    score_entropy_loss,
    uniqueness_reward_loss,
    v0_distill_loss,
)
from autogaze.models.borissal.modeling_borissal import _pack_gazing_mask
from autogaze.models.borissal.vjepa2_sparse import VJEPA2Teacher

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=None, help="folder of .mp4s (recursive)")
    p.add_argument("--teacher", default=None, help="HF id or local path; omit for tiny random teacher")
    p.add_argument("--smoke", action="store_true",
                   help="tiny random teacher + duplicated example clip + few steps")
    p.add_argument("--scale", type=int, default=384)
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--input-mode", choices=["maps", "pixels", "both"], default="both")
    p.add_argument("--residual-scoring", action="store_true")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=2, help="per-process batch size")
    p.add_argument("--num-workers", type=int, default=4,
                   help="DataLoader workers (PyAV decode is the IO bottleneck on real data)")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ratio-sampling", choices=["fixed", "uniform"], default="uniform")
    p.add_argument("--ratio", type=float, default=0.5, help="fixed ratio (ratio-sampling=fixed)")
    p.add_argument("--ratio-min", type=float, default=0.15)
    p.add_argument("--ratio-max", type=float, default=0.75)
    p.add_argument("--w-pred", type=float, default=1.0)
    p.add_argument("--w-match", type=float, default=0.0)
    # default 0.01: both real-teacher smoke runs showed grad_norm decaying to
    # ~0 within 30 steps (score-head saturation kills the ST soft path) --
    # a small entropy term counteracts that. Set 0 to disable.
    p.add_argument("--w-entropy", type=float, default=0.01)
    p.add_argument("--w-v0-distill", type=float, default=0.0)
    p.add_argument("--v0-distill-warmup-steps", type=int, default=0)
    # anti-scatter reward (design.md "Borissal v0.2" Finding 1); costs one
    # extra predictor pass per step when nonzero.
    p.add_argument("--w-uniqueness", type=float, default=0.0)
    p.add_argument("--input-v0-preset", choices=["v0.1", "v0.2"], default="v0.2",
                   help="which non-learned signal preset feeds the learned scorer")
    p.add_argument("--out-dir", default=str(REPO_ROOT / "weights" / "borissal_v1"))
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=0, help="0 = save only at the end")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def setup_distributed():
    """Returns (rank, world_size, local_rank). Plain single-process unless
    launched via torchrun (env RANK/WORLD_SIZE present)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return rank, world, local_rank
    return 0, 1, 0


def sample_ratio(args, step: int, device, distributed: bool) -> float:
    if args.ratio_sampling == "fixed":
        return args.ratio
    r = torch.empty(1, device=device).uniform_(args.ratio_min, args.ratio_max)
    if distributed:
        dist.broadcast(r, src=0)  # same budget on every rank
    return r.item()


def peak_memory_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1e6
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / 1e6
    # ru_maxrss units: bytes on macOS, kilobytes on Linux
    import sys
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 1e6 if sys.platform == "darwin" else raw / 1e3


def complement_indices(hard_keep: torch.Tensor) -> torch.Tensor:
    """hard_keep: (B, T_grid, N_pf) bool with exact-k rows -> (B, L-K) ascending."""
    B = hard_keep.shape[0]
    flat = hard_keep.reshape(B, -1)
    comp, pad = _pack_gazing_mask(~flat)
    assert not pad.any(), "complement should be unpadded under uniform allocation"
    return comp


def main():
    args = parse_args()
    rank, world, local_rank = setup_distributed()
    distributed = world > 1
    is_main = rank == 0

    torch.manual_seed(args.seed + rank)
    device = torch.device(f"cuda:{local_rank}") if distributed and torch.cuda.is_available() \
        else resolve_device(args.device)

    # ---------------------------------------------------------------- data
    tmp_dir = None
    if args.smoke and args.data_root is None:
        # duplicate the example clip into a temp folder as a minimal dataset
        tmp_dir = tempfile.mkdtemp(prefix="borissal_smoke_")
        src = REPO_ROOT / "assets" / "example_input.mp4"
        for i in range(4):
            shutil.copy(src, os.path.join(tmp_dir, f"clip{i}.mp4"))
        args.data_root = tmp_dir
    if args.smoke:
        args.num_workers = 0  # tiny duplicated-clip dataset; avoid worker spawn overhead
    if args.smoke and args.teacher is None:
        args.scale = 128
        args.steps = min(args.steps, 30)

    dataset = VideoFolderDataset(args.data_root, num_frames=args.num_frames, size=args.scale)
    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler,
        shuffle=(sampler is None), num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"), drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )

    # ------------------------------------------------------------- teacher
    # "hub:<entrypoint>" -> torch.hub V-JEPA 2.1 adapter; anything else -> HF.
    hub_teacher = False
    if args.teacher and args.teacher.startswith("hub:"):
        from autogaze.models.borissal.vjepa21_hub import VJEPA21HubTeacher
        teacher = VJEPA21HubTeacher.from_hub(args.teacher[len("hub:"):])
        hub_teacher = True
    elif args.teacher:
        teacher = VJEPA2Teacher.from_pretrained(args.teacher)
    else:
        teacher = VJEPA2Teacher.tiny_random(crop_size=args.scale, frames_per_clip=args.num_frames)
    teacher = teacher.to(device)

    # ------------------------------------------------------------- selector
    config = BorissalV1Config(
        scale=args.scale,
        input_mode=args.input_mode,
        residual_scoring=args.residual_scoring,
        v0_preset=args.input_v0_preset,
    )
    model = BorissalV1(config).to(device)
    model.train()
    wrapped = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None) if distributed else model

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    weights = LossWeights(
        predictor_coverage=args.w_pred,
        dense_sparse_match=args.w_match,
        score_entropy=args.w_entropy,
        v0_distill=args.w_v0_distill,
        v0_distill_warmup_steps=args.v0_distill_warmup_steps,
        uniqueness_reward=args.w_uniqueness,
    )

    # Collapse probe: a FIXED clip + fixed ratio, re-selected in eval mode at
    # every log point. If probe_overlap_prev pins to 1.0 while grad_norm dies,
    # the selector has frozen onto a constant pattern (the failure mode the
    # user flagged). Random-selection IoU at ratio 0.3 is ~0.3 for reference.
    probe_clip = dataset[0].unsqueeze(0).to(device)
    probe_prev_mask = None

    out_dir = Path(args.out_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"

    # ------------------------------------------------------------ train loop
    step = 0
    data_iter = iter(loader)
    t_start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    while step < args.steps:
        optimizer.zero_grad(set_to_none=True)
        accum_logs = None
        for _ in range(args.grad_accum):
            try:
                video = next(data_iter)
            except StopIteration:
                if sampler is not None:
                    sampler.set_epoch(step)
                data_iter = iter(loader)
                video = next(data_iter)
            video = video.to(device)

            ratio = sample_ratio(args, step, device, distributed)
            out = (wrapped.module if distributed else wrapped).forward_train(video, gazing_ratio=ratio)

            B = video.shape[0]
            hard = out["hard_keep"]                    # (B, T_grid, N_pf)
            keep_idx = out["keep_index"]               # (B, K) ascending, unpadded
            tgt_idx = complement_indices(hard)         # (B, L-K)
            L = hard.shape[1] * hard.shape[2]

            # ST gate values at the kept positions, in keep_idx order
            gate_flat = out["st_gate"].reshape(B, L)
            gate = torch.gather(gate_flat, 1, keep_idx)

            teacher_dense = teacher.dense_features(video)                     # no_grad
            sparse = teacher.sparse_features(video, keep_idx, gate=gate)      # grads -> gate
            predicted = teacher.predict(sparse, keep_idx, tgt_idx, num_tokens=L)
            if hub_teacher:
                # V-JEPA 2.1's predictor projects into its distillation-teacher
                # space (1664-d), not the encoder's own space -- so the coverage
                # target is an ORACLE REFERENCE pass through the same predictor
                # head: context = ALL tokens (dense feats). The selector learns
                # to make the sparse-context prediction match the
                # full-information prediction. (Our adaptation to 2.1's head
                # structure -- see training.md §5.)
                with torch.no_grad():
                    all_idx = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
                    teacher_targets = teacher.predict(teacher_dense, all_idx, tgt_idx, num_tokens=L)
            else:
                teacher_targets = torch.gather(
                    teacher_dense, 1, tgt_idx.unsqueeze(-1).expand(-1, -1, teacher_dense.size(-1))
                )

            v0_scores_flat = None
            if weights.v0_distill > 0:
                with torch.no_grad():
                    _, v0_inter = model._v0.select_with_intermediates(video)
                v0_scores_flat = v0_inter["score"].reshape(B, hard.shape[1], hard.shape[2])

            def uniqueness_term():
                # Gradient conduit on the REST side: inverse ST gate
                # (forward value 1 at unselected positions, backward -d soft).
                rest_gate = torch.gather(1.0 - gate_flat, 1, tgt_idx)
                rest_sparse = teacher.sparse_features(video, tgt_idx, gate=rest_gate)
                pred_sel = teacher.predict(rest_sparse, tgt_idx, keep_idx, num_tokens=L)
                if hub_teacher:
                    with torch.no_grad():
                        all_idx = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
                        sel_targets = teacher.predict(teacher_dense, all_idx, keep_idx, num_tokens=L)
                else:
                    sel_targets = torch.gather(
                        teacher_dense, 1, keep_idx.unsqueeze(-1).expand(-1, -1, teacher_dense.size(-1))
                    )
                return uniqueness_reward_loss(pred_sel, sel_targets)

            total, logs = combine_losses(weights, step, {
                "predictor_coverage": lambda: predictor_coverage_loss(predicted, teacher_targets),
                "dense_sparse_match": lambda: dense_sparse_match_loss(sparse, teacher_dense, keep_idx),
                "score_entropy": lambda: score_entropy_loss(out["probs"]),
                "v0_distill": lambda: v0_distill_loss(
                    out["scores"].reshape(B, hard.shape[1], hard.shape[2]), v0_scores_flat
                ),
                "uniqueness_reward": uniqueness_term,
            })
            (total / args.grad_accum).backward()
            accum_logs = logs  # keep the last micro-batch's numbers

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

        # Gradient-reach diagnostics: does the loss actually move the scores
        # of UNSELECTED and low-probability patches? (Channels: softmax
        # coupling ~p_j, Gumbel exploration, entropy term, uniqueness inverse
        # gate -- all vanish as p_j -> 0 except exploration, so this is the
        # early-warning meter for score lock-in.)
        grad_reach = {}
        lgrad = out["logits"].grad
        if lgrad is not None:
            with torch.no_grad():
                g = lgrad.abs()
                sel = out["hard_keep"]
                grad_reach["lgrad_sel_mean"] = g[sel].mean().item() if sel.any() else 0.0
                grad_reach["lgrad_unsel_mean"] = g[~sel].mean().item() if (~sel).any() else 0.0
                # lowest-probability decile: the patches most at risk of freezing
                p = out["probs"]
                n_low = max(1, p.shape[-1] // 10)
                low_idx = p.topk(n_low, dim=-1, largest=False).indices
                grad_reach["lgrad_low_decile_mean"] = g.gather(-1, low_idx).mean().item()
                grad_reach["lgrad_unsel_zero_frac"] = (g[~sel] == 0).float().mean().item() if (~sel).any() else 0.0

        optimizer.step()
        step += 1

        if is_main and (step % args.log_every == 0 or step == 1 or step == args.steps):
            with torch.no_grad():
                # selection overlap with v0 as a cheap quality proxy
                v0_sel = model._v0.select(video, gazing_ratio=ratio)
                v0_mask = v0_sel.keep_mask
                v1_mask = hard.reshape(B, L)
                overlap = (v0_mask & v1_mask).sum().item() / max(1, v1_mask.sum().item())
                # collapse probe: fixed clip + fixed ratio, eval-mode selection.
                # probe_overlap_prev pinned at 1.0 + dying grad_norm = frozen
                # selector (content-independent constant pattern).
                model.eval()
                probe_mask = model.select(probe_clip, gazing_ratio=0.3).keep_mask
                model.train()
                if probe_prev_mask is not None:
                    inter_ = (probe_mask & probe_prev_mask).sum().item()
                    union_ = (probe_mask | probe_prev_mask).sum().item()
                    probe_iou = inter_ / max(1, union_)
                else:
                    probe_iou = None
                probe_prev_mask = probe_mask
                probs_ = out["probs"]
                entropy_mean = (-(probs_.clamp_min(1e-9).log() * probs_).sum(-1)).mean().item()
            record = {
                "step": step,
                "ratio": round(ratio, 4),
                "grad_norm": round(grad_norm.item(), 5),
                "v0_overlap": round(overlap, 4),
                "probe_overlap_prev": None if probe_iou is None else round(probe_iou, 4),
                "score_entropy_mean": round(entropy_mean, 4),
                "sec_per_step": round((time.perf_counter() - t_start) / step, 3),
                "peak_mem_mb": round(peak_memory_mb(device), 1),
                **{k: round(v, 9) for k, v in grad_reach.items()},
                **{f"loss/{k}": round(v, 6) for k, v in accum_logs.items()},
            }
            print(json.dumps(record))
            with open(log_path, "a") as f:
                f.write(json.dumps(record) + "\n")

        if is_main and args.save_every and step % args.save_every == 0:
            torch.save({"model_tag": MODEL_TAG_V1, "config": vars(config) | {"input_mode": config.input_mode},
                        "state_dict": model.state_dict(), "step": step},
                       out_dir / f"checkpoint_step{step}.pt")

    if is_main:
        ckpt_path = out_dir / "checkpoint_final.pt"
        torch.save({"model_tag": MODEL_TAG_V1, "config": config.__dict__,
                    "state_dict": model.state_dict(), "step": step}, ckpt_path)
        print(f"saved {ckpt_path}")

    if distributed:
        dist.destroy_process_group()
    if tmp_dir:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
