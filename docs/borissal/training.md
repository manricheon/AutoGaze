# Borissal v1 — Self-Supervised Training Guide

How the learned selector (Borissal v1) is trained without ground-truth
gazing labels, why the objective is designed this way, and how to run it.
The SSL-objective/loss-combination design here is a core contribution of
this line of work — treat this document as the canonical description.

See `reference.md` for the selector itself, `design.md` for engineering
decisions, `progress.md` for the session log.

## 1. The objective: predictor information-coverage

**Setup.** A frozen V-JEPA2 model is the teacher. For each clip:

1. Teacher runs **dense** (all `L` tokens) → target features, no gradients.
2. The selector picks `K = round(ratio · N_pf)` patches per tubelet
   (differentiable path, §2).
3. Teacher's encoder runs **sparse** — only the selected tokens, with their
   original t-major flat indices driving 3D-RoPE, so each surviving token
   keeps its true (t, h, w) position. (The stock HF encoder is dense-only;
   `vjepa2_sparse.sparse_encoder_forward` re-wires its own primitives —
   `apply_masks`-style gather + per-layer `position_mask` — to run the
   subset. Verified identical API on the pinned transformers==5.5.0.)
4. V-JEPA2's **predictor** infills the features of every *unselected*
   position from the sparse context.
5. Loss: `L_pred = MSE(predictor(selected), teacher_dense[unselected])`.

**Why this objective.** A selection is good exactly when what it kept
suffices to reconstruct what it dropped: minimizing `L_pred` maximizes the
information coverage of the kept set. This is the natural selector-flavored
inversion of V-JEPA's own pretraining task (predict masked features from
context) — we reuse its pretrained predictor as a fixed measuring
instrument for "how much information did this selection preserve."

**Why not plain dense-vs-sparse matching** (`MSE(sparse_feats,
dense_feats[selected])`): it rewards selecting tokens whose features change
least when the rest of the clip is removed — i.e. context-independent,
uniform background patches. That is precisely the *wrong* preference for a
selector. It's available as an optional small-weight auxiliary
(`--w-match`), never as the main loss.

## 2. The differentiable selection path

Inference is hard top-k (mobile-safe, exact budget). Training needs
gradients through "which tokens were kept":

- **Gumbel-perturbed top-k**: per-tubelet scores get Gumbel(0,1)·τ noise
  (`gumbel_tau`, default 1.0) before the hard top-k, so different
  selections get explored across steps. Eval mode is noise-free and
  deterministic (tested).
- **Straight-through gate**: the kept tokens' embeddings are multiplied by
  `gate = hard + softmax(scores)·N_pf − (softmax(scores)·N_pf).detach()` —
  forward value exactly 1.0 (selection unchanged), backward routes
  `∂L/∂gate` into the score head. The `·N_pf` rescale keeps gradient
  magnitude O(1) regardless of grid size (raw softmax probs average
  `1/N_pf` ≈ 0.0017 at the 24×24 grid, which would starve the selector of
  gradient — found empirically during smoke).
- Training always uses **uniform per-tubelet allocation** (exact k,
  data-independent K) — matching the mobile-export-safe inference default.

`gazing_ratio` is **sampled per batch** during training
(`--ratio-sampling uniform --ratio-min 0.15 --ratio-max 0.75`, synced
across DDP ranks) so one selector generalizes across budgets instead of
overfitting to a single ratio.

## 3. Loss combinations

`L = w_pred·L_pred + w_match·L_match + w_ent·L_entropy + w_v0·L_v0_distill`
(`losses.py`; zero-weight terms are skipped entirely).

| Term | Flag | Role | When to use |
|---|---|---|---|
| Predictor coverage | `--w-pred` (default 1.0) | Main objective (§1) | Always |
| Dense-sparse match | `--w-match` (default 0) | Auxiliary consistency | Small weight only, if selection oscillates; watch for background bias |
| Score entropy | `--w-entropy` (default 0) | Anti-collapse regularizer (penalizes peaked score distributions) | If selection collapses to a few patches early; anneal to 0 |
| v0 distillation | `--w-v0-distill` + `--v0-distill-warmup-steps` | Warmup: KL(v1 scores ‖ v0 saliency) linearly decayed to 0 | Start from the proven saliency prior instead of random selection; also the fastest sanity check of the training plumbing (loss should drop ~99% in tens of steps) |

Suggested first real experiment matrix (one knob at a time):
1. `w_pred=1` alone (baseline).
2. `w_pred=1, w_v0=1, warmup=500` (prior-guided start).
3. `w_pred=1, w_ent=0.01` (if 1 collapses).
4. Ablate `--input-mode maps|pixels|both` on the best of the above.

## 4. v0 leverage options (all config-switchable, default off unless noted)

- `--input-mode both` (default): v0's motion/spatial patch maps are input
  channels — the network learns a contextual weighting policy over the
  proven saliency signal (superset of v0's fixed `motion_weight`) plus
  complementary pixel cues.
- `--residual-scoring`: `score = v0_score + f_θ(·)` — v1 learns only a
  residual on top of v0's ranking.
- `--w-v0-distill`: warmup imitation, §3.
- v0 remains the deploy fallback: same `Selection` contract, zero weights.

## 5. Teacher notes

- Wrapper (`vjepa2_sparse.VJEPA2Teacher`) is teacher-agnostic: anything with
  `dense_features / sparse_features / predict` in the same shapes plugs in.
  Grid must be patch16/tubelet2 (asserted); resolution follows the clip.
- **Works out of the box (pinned transformers==5.5.0):** official
  `facebook/vjepa2-vitl-fpc64-256` (ViT-L, 326M) — used for the local
  real-teacher validation run.
- **V-JEPA 2.1 (the team's preferred teacher) is supported via torch.hub**:
  `--teacher hub:vjepa2_1_vit_base_384` (B, used for local validation) or
  `hub:vjepa2_1_vit_large_384` (L, for scale training) — official
  `facebookresearch/vjepa2` entrypoints, each returning `(encoder,
  predictor)`. Adapter: `vjepa21_hub.VJEPA21HubTeacher`. The original
  encoder supports sparse natively (`masks=[keep_index]`, same t-major
  canonical order); the ST gate is applied to the encoder *output* features
  there (no clean input-embedding injection point in the original code —
  an equally valid gradient conduit; the HF wrapper gates at input
  embeddings).
  - **2.1-specific loss wiring — oracle-reference coverage.** 2.1's
    `predictor_proj` outputs the *distillation-teacher* space (1664-d,
    ViT-gigantic), not the encoder's own space, so predictions can't be
    compared to the 2.1 encoder's dense features directly. Instead the
    trainer runs the SAME predictor head twice: a no-grad **reference pass**
    with context = ALL tokens (full-information prediction of the target
    positions) and the **student pass** with context = selected tokens only;
    the loss is MSE between the two, in the same 1664-d space, same head,
    same positions. The selector thus learns selections whose
    sparse-context prediction matches the full-information prediction —
    the same coverage intuition, made well-defined for 2.1's head
    structure. NOTE: this is our adaptation, not a published recipe —
    validate empirically at scale.
  - HF-side note: HF-hosted 2.1 checkpoints do NOT load into the native
    `vjepa2` class (dual patch embeds / modality embeds / distillation
    norms / 1664 proj — confirmed empirically); torch.hub is the supported
    2.1 path.
  - torch.hub caveat: the repo's current main hardcodes a localhost
    checkpoint URL (dev leftover). Pre-download
    `https://dl.fbaipublicfiles.com/vjepa2/<file>.pt` into
    `~/.cache/torch/hub/checkpoints/` and torch.hub uses the cache.
- `--smoke` with no `--teacher` uses a tiny randomly-initialized V-JEPA2
  config: validates the full graph/gradients in seconds, but its random
  features give the coverage loss no meaningful preference between
  selections (observed: near-flat loss) — use it for plumbing checks only,
  never to judge selection learning.

## 6. Running

Mac smoke (plumbing check, ~30s):
```bash
uv run python scripts/train_borissal_v1.py --smoke --device cpu
# strongest plumbing signal: v0-distill only, loss should collapse toward 0
uv run python scripts/train_borissal_v1.py --smoke --device cpu \
    --w-pred 0 --w-v0-distill 1 --ratio-sampling fixed --ratio 0.3
```

Mac real-teacher check (V-JEPA2 ViT-L via HF, MPS, minutes):
```bash
uv run python scripts/train_borissal_v1.py --smoke \
    --teacher facebook/vjepa2-vitl-fpc64-256 --scale 256 \
    --batch-size 1 --device mps --steps 30 --out-dir weights/borissal_v1_real
```

Mac real-teacher check (V-JEPA **2.1**-B via torch.hub, oracle-reference loss):
```bash
uv run python scripts/train_borissal_v1.py --smoke \
    --teacher hub:vjepa2_1_vit_base_384 --scale 384 \
    --batch-size 1 --device mps --steps 30 --out-dir weights/borissal_v1_vjepa21b
```

Linux multi-GPU (AutoGaze-Training-Data, V-JEPA 2.1-L teacher):
```bash
torchrun --nproc_per_node=8 scripts/train_borissal_v1.py \
    --data-root /path/to/AutoGaze-Training-Data \
    --teacher hub:vjepa2_1_vit_large_384 --scale 384 \
    --batch-size 8 --grad-accum 2 --steps 20000 \
    --ratio-sampling uniform --ratio-min 0.15 --ratio-max 0.75
```
(HF `facebook/vjepa2-vitl-fpc64-256` at `--scale 256` remains a supported
alternative teacher with the plain dense-target coverage loss.)
(DDP activates automatically under torchrun; single-process otherwise.
gazing_labels.json is not needed — this is fully self-supervised.)

Inspect results:
```bash
# overlay/score dumps for a checkpoint
uv run python scripts/borissal_dump_outputs.py --video <clip> \
    --model v1 --checkpoint weights/borissal_v1_real/checkpoint_final.pt
# latency + peak memory, v0 vs v1
uv run python scripts/borissal_benchmark.py --model both
```

Logged per `--log-every` steps (stdout + `train_log.jsonl` in the out dir):
per-term losses, grad norm, sampled ratio, v0-selection overlap (cheap
quality proxy), sec/step, peak memory (MB).
