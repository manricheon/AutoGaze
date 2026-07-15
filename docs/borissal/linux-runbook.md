# Borissal v1 — Linux scale-run runbook

Step-by-step execution guide for the first real (scale) training run.
This consolidates training.md §6 (commands) and §7 (checklist / judgment
ladder) into one runnable sequence, adds the **offline asset manifest**,
and gives launch variants for 4- and 8-GPU machines.

Assumptions (confirmed 2026-07-15): the Linux machine has 4 or 8 CUDA
GPUs and only restricted egress — **plan as if offline** and carry the
assets in. The plain online commands are recorded alongside as the
baseline install method.

Judgment calls (what the numbers must show) live in training.md §7.7;
this file is only the *how to run* companion.

## Step 0 — Prepare the transfer bundle (on a machine with internet)

| Asset | How to get it | Size (approx) | Where it goes on the Linux box |
|---|---|---|---|
| Repo, branch `feat/borissal` | `git bundle create autogaze.bundle feat/borissal` (or a tarball of the checkout) | small | anywhere; `git clone autogaze.bundle -b feat/borissal AutoGaze` |
| Training data | `hf download bfshi/AutoGaze-Training-Data --repo-type dataset --local-dir AutoGaze-Training-Data` | hundreds of GB | `<D>/` (any path; passed via `--data-root`) |
| V-JEPA 2.1-L teacher | `curl -L -o vjepa2_1_vitl_dist_vitG_384.pt https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt` | ~1.7 GB | `~/.cache/torch/hub/checkpoints/` |
| vjepa2 repo (offline torch.hub) | `git clone https://github.com/facebookresearch/vjepa2` | small | anywhere; passed via `--hub-repo-dir` |
| SigLIP2 (semantic gate, eval-only) | `hf download google/siglip2-base-patch16-384` then copy the HF cache dir | ~1.5 GB | `~/.cache/huggingface/hub/` |
| VideoMAE gate ckpt | already on the Mac: `weights/VideoMAE_AutoGaze/videomae.pt` (HF `bfshi/VideoMAE_AutoGaze`) | 2.0 GB | `weights/VideoMAE_AutoGaze/videomae.pt` |
| Python wheelhouse (only if PyPI is unreachable) | see below | several GB (CUDA torch) | `wheelhouse/` |

Wheelhouse build (skip if the machine can reach PyPI — the "restricted"
machine may allow it; check first):

```bash
uv pip compile pyproject.toml --extra dev -o requirements-lock.txt
python3 -m pip download -r requirements-lock.txt -d wheelhouse/ \
    --platform manylinux2014_x86_64 --python-version 311 --only-binary=:all:
```

(`uv pip download` does not exist; use pip for the download step. If a
package has no manylinux wheel, drop `--only-binary` for that one and
build on the target.)

## Step 1 — Environment

Online baseline (training.md §7.1, record for reference even on the
offline path):

```bash
git clone git@github.com:manricheon/AutoGaze.git && cd AutoGaze
git checkout feat/borissal
uv venv --python 3.11
uv pip install -e .            # CUDA torch wheel resolves automatically on Linux
uv pip install -e '.[dev]'
# Do NOT install .[cuda] (flash_attn) — legacy stack only, unused by Borissal.
```

Offline variant (repo from the bundle, deps from the wheelhouse):

```bash
git clone autogaze.bundle -b feat/borissal AutoGaze && cd AutoGaze
uv venv --python 3.11
uv pip install --no-index --find-links ../wheelhouse -e '.[dev]'
```

## Step 2 — Place assets and set offline env

```bash
mkdir -p ~/.cache/torch/hub/checkpoints
cp <bundle>/vjepa2_1_vitl_dist_vitG_384.pt ~/.cache/torch/hub/checkpoints/
cp -r <bundle>/huggingface-cache/* ~/.cache/huggingface/   # SigLIP2 seed
export HF_HUB_OFFLINE=1     # needed only for eval_borissal_semantic.py
```

Verify the five dataset train dirs exist under `<D>`:
`InternVid_res448_250K/train`, `100DoH_res448_250K/train`,
`Ego4D_res448_250K/train`, `scanning_SAM_res448_50K/train`,
`scanning_idl_res448_50K/train`.

Define once for the commands below:

```bash
DATA_ROOT="<D>/InternVid_res448_250K/train,<D>/100DoH_res448_250K/train,<D>/Ego4D_res448_250K/train,<D>/scanning_SAM_res448_50K/train,<D>/scanning_idl_res448_50K/train"
HUB=<path to the vjepa2 clone>
```

## Step 3 — Smoke gate (mandatory before the real run)

```bash
uv run pytest tests/ -q      # expect 49 passed

# 1-GPU tiny run: first CUDA validation of the hub 2.1-L teacher forward
# (only the B variant was validated locally on the Mac)
uv run python scripts/train_borissal_v1.py \
    --data-root "$DATA_ROOT" --max-files 8 \
    --teacher hub:vjepa2_1_vit_large_384 --scale 384 --hub-repo-dir "$HUB" \
    --steps 20 --save-every 10 --out-dir outputs/borissal/smoke

# resume smoke: re-run the same command with --resume auto and confirm
# z_loss continues its trajectory (no cold-optimizer spike) and tb/ +
# train_log.jsonl are written.
```

## Step 4 — Measure the coverage floor (§7 item 0)

```bash
uv run python scripts/eval_borissal_coverage.py \
    --video <a few dataset clips> --ratios 0.25 \
    --teacher hub:vjepa2_1_vit_large_384 --scale 384 \
    --hub-repo-dir "$HUB" --configs random
```

Use the reported random coverage as `<FLOOR>` below. Expect the hub 2.1
oracle-reference scale (~0.05 measured locally), NOT the HF vitl-256
scale (~8.2) — the floor is teacher-family-specific.

## Step 5 — Launch the real run (keep global batch = 128)

```bash
# 8 GPUs: N=8, A=2       (8 x 8 x 2 = 128)
# 4 GPUs: N=4, A=4       (4 x 8 x 4 = 128; if OOM: --batch-size 4 --grad-accum 8)
torchrun --nproc_per_node=<N> scripts/train_borissal_v1.py \
    --data-root "$DATA_ROOT" \
    --teacher hub:vjepa2_1_vit_large_384 --scale 384 --hub-repo-dir "$HUB" \
    --batch-size 8 --grad-accum <A> --steps 20000 \
    --w-v0-distill 1.0 --v0-distill-warmup-steps 500 \
    --w-uniqueness 1.0 --coverage-floor <FLOOR> \
    --ratio-sampling uniform --ratio-min 0.15 --ratio-max 0.75 \
    --save-every 1000 --num-workers 8 --resume auto \
    --out-dir outputs/borissal/scale_run_1
```

lr defaults to 1e-4; entropy annealing stays off; RL stays off (SSL-only
stage-1 decision, training.md §8). `--resume auto` is safe to pass from
the start — it no-ops until a `checkpoint_last.pt` exists, and makes
restarts after interruption a re-run of the same command.

## Step 6 — Watch the first few hundred steps (§7 items 1–2)

tensorboard events under `outputs/borissal/scale_run_1/tb/`;
`train_log.jsonl` is the source of truth (`plot_borissal_training.py`).

- `loss/uniqueness_reward` trending DOWN — the key early bar; it was flat
  in the 60-step local test and must actually move at scale.
- coverage-floor overflow bounded (< ~1.0).
- `score_entropy_mean` > ~3.5 and not in freefall; `grad_norm` alive;
  `probe_overlap_prev` never pinned at 1.0;
  `lgrad_low_decile_mean` within ~2 orders of `lgrad_sel_mean`.
- `peak_mem_mb` for batch-size headroom.

## Step 7 — Per-checkpoint diagnostics (§7 item 3)

```bash
uv run python scripts/borissal_model_diagnostics.py \
    --checkpoint outputs/borissal/scale_run_1/checkpoint_<step>.pt
```

Look for: `perturb_near_over_far` DECREASING from the untrained ~40–55x;
`v0_spearman` not converging to 1.0; `brightness_spearman` falling from
the ~0.83 init bias.

## Step 8 — Adoption gate (§7 item 4; semantic recall is PRIMARY)

Run all three axes with `random`, `v0.2`, and `v1:<checkpoint>`, each at
spread s ∈ {0, 0.25, 0.5} (`--spread`):

```bash
uv run python scripts/eval_borissal_semantic.py ...        # PRIMARY
uv run python scripts/eval_borissal_coverage.py ...        # training consistency
uv run python scripts/eval_borissal_videomae_recon.py \
    --videomae-ckpt weights/VideoMAE_AutoGaze/videomae.pt  # reference axis only
```

Bars (local single-clip references at ratio 0.25):
- semantic recall: v1 > random (0.267) is necessary; approaching/beating
  v0.2+s0.25 (0.309) is the adoption bar.
- coverage eval: v1 uniqueness > random — the bar the local run missed
  (7.85 < 7.93); coverage near the measured floor.
- VideoMAE recon is scatter-biased — report, don't gate on it.

## Step 9 — Visual dumps and the verdict

```bash
uv run python scripts/borissal_dump_outputs.py --video <clip> \
    --model v1 --checkpoint <ckpt>
uv run python scripts/borissal_benchmark.py --model both
```

No edge bands, content-following, frame-consistent. Remember the local
caution: visual appeal and the uniqueness metric can disagree — report
both.

Verdict branches:
- **Gates pass** → adopt; promote `spread_fraction=0.25` from deploy
  override to preset default (reference.md §3 caveat resolves), plan the
  mobile/deploy path.
- **Entropy collapses / grads die** despite WP-A + global context →
  RL trigger per training.md §8 (`--rl-after-step N --rl-samples 4
  --rl-cov-weight 1.0`).
- **Gates missed without collapse** → diagnose via the deferred §7.8
  items (REAL-X predictor calibration, EVAL-X audit, Frame-Voyager
  caption-loss ranking) before touching the recipe.
