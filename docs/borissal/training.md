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

- **Gumbel-perturbed top-k**: per-tubelet scores get Gumbel(0,1) noise
  before the hard top-k, so different selections get explored across
  steps. Eval mode is noise-free and deterministic (tested).
- **Straight-through gate** (canonical ST-Gumbel-softmax since the
  2026-07 theory upgrade, §8): the kept tokens' embeddings are multiplied
  by `gate = hard + soft·N_pf − (soft·N_pf).detach()` with
  `soft = softmax((scores + gumbel)/τ)` — the soft path comes from the
  SAME noised logits that produced the hard selection, at temperature
  `gumbel_tau` (default **2/3**, the Concrete/Gumbel-softmax canonical
  value; τ < 0.5 is the documented gradient-variance blowup zone, so no
  annealing). Forward value exactly 1.0 (selection unchanged), backward
  routes `∂L/∂gate` into the score head. The `·N_pf` rescale keeps
  gradient magnitude O(1) regardless of grid size (raw softmax probs
  average `1/N_pf` ≈ 0.0017 at the 24×24 grid, which would starve the
  selector of gradient — found empirically during smoke). The returned
  `probs` stays the CLEAN `softmax(scores)` so the entropy loss and
  saturation monitoring are not polluted by per-sample Gumbel sharpness.
- **Cosine score head** (`cosine_scores`, default on since §8): logits =
  normalized-feature · normalized-weight × learnable temperature (X-MoE,
  arXiv:2204.09179) — |logit| ≤ temp by construction, so the score head
  cannot run the logit-norm arms race behind saturation. The trainer also
  puts the head (+ its temperature) on a lower lr (`--head-lr-scale`,
  default 0.1 — router-style treatment).
- **Block-structured training selection** (`train_block_size` /
  `--train-block-size`, default 1 = per-token): b > 1 selects b×b spatial
  BLOCKS in `forward_train` (block-mean logits → block Gumbel-top-k →
  gate expanded to tokens; budget snapped to whole blocks). Why: scattered
  per-token selection is both the provable optimum of coverage-style
  objectives and deep off-distribution for the multi-block-trained V-JEPA
  predictor (I-JEPA ablation: scattered 17.6 vs multi-block 54.2 IN-1k 1%
  linear) — blocks remove the scatter shortcut by construction and move
  the frozen predictor back onto its training distribution. Inference
  `select()` is unaffected (v0.2's coarse-to-fine gate covers that side).
- Training always uses **uniform per-tubelet allocation** (exact k,
  data-independent K) — matching the mobile-export-safe inference default.

**Gradient reach to unselected / low-score patches** (a frequently-raised
worry: if a patch is never selected, can its score ever recover?). Four
channels carry gradient to unselected logits: (1) the softmax Jacobian
couples every logit — magnitude ∝ p_j, so it VANISHES as a patch's
probability → 0; (2) Gumbel exploration occasionally hard-selects
low-score patches, giving them direct gate gradient — the primary
*revival* channel, alive only while logits stay unsaturated (which the
entropy default protects); (3) the entropy term itself (all logits,
~p·log p scale); (4) the uniqueness inverse gate when enabled. Because
every differentiable channel decays with p_j, this is measured LIVE: the
trainer logs `lgrad_sel_mean` / `lgrad_unsel_mean` /
`lgrad_low_decile_mean` (mean |d loss/d logit| over selected, unselected,
and the lowest-probability decile) and `lgrad_unsel_zero_frac`. Healthy
smoke values: zero-frac 0.0 and unselected ≈ 0.5–0.6× the selected mean.
If `lgrad_low_decile_mean` collapses orders of magnitude below
`lgrad_sel_mean` at scale, scores are locking in — raise `--w-entropy`
and/or `gumbel_tau`. Locked by
`test_gradient_reaches_unselected_and_low_prob_patches`.

`gazing_ratio` is **sampled per batch** during training
(`--ratio-sampling uniform --ratio-min 0.15 --ratio-max 0.75`, synced
across DDP ranks) so one selector generalizes across budgets instead of
overfitting to a single ratio.

## 3. Loss combinations

`L = w_pred·L_pred + w_match·L_match + w_ent·L_entropy + w_z·L_zloss +
w_v0·L_v0_distill + w_uniq·L_uniq + w_hard·L_hardness`
(`losses.py`; zero-weight terms are skipped entirely).

| Term | Flag | Role | When to use |
|---|---|---|---|
| Predictor coverage | `--w-pred` (default 1.0) | Main objective (§1) | Always |
| Dense-sparse match | `--w-match` (default 0) | Auxiliary consistency | Small weight only, if selection oscillates; watch for background bias |
| Score entropy | `--w-entropy` (default **0.01**) | Anti-collapse regularizer (penalizes peaked score distributions) | On by default (see rationale below); raise if grad_norm still dies, set 0 to disable |
| v0 distillation | `--w-v0-distill` + `--v0-distill-warmup-steps` | Warmup: KL(v1 scores ‖ v0 saliency) linearly decayed to 0 | Start from the proven saliency prior instead of random selection; also the fastest sanity check of the training plumbing (loss should drop ~99% in tens of steps) |
| Uniqueness reward | `--w-uniqueness` (default 0) | Anti-scatter: rewards selections the REST cannot reconstruct (capped negative MSE, inverse-ST-gate gradient conduit) | Counters the measured scatter bias of pure coverage (design.md "Borissal v0.2" Finding 1: random beat saliency on coverage alone). Costs one extra predictor pass. **§8 recipe promotes this to the PRIMARY objective** with coverage as a floor |
| Router z-loss | `--w-zloss` (default **1e-3**) | Penalizes logit magnitude at the source (`mean(logsumexp²)`, ST-MoE arXiv:2202.08906) | On by default — the published coefficient stabilized 3/3 unstable runs WITH a quality gain; directly attacks the measured saturation (P2) |
| Coverage floor | `--coverage-floor` (default 0) | Turns coverage into a CONSTRAINT: `relu(mse − floor)`, zero pressure below the floor | Set to the matched-ratio random baseline from `eval_borissal_coverage.py` when uniqueness is primary (§8); 0 keeps pure minimization |
| Hardness ranking | `--w-hardness` (default 0) | HPM-style (arXiv:2304.05919): score head ranks REST tokens by per-token predictor error (pairwise BCE; reuses the coverage pass — free) | Experimental direct score-supervision channel that bypasses the ST gate. HPM caveat: ALL-hard selection underperforms random — keep a v0-distill warmup as the easy-to-hard curriculum |

**Defaults with rationale:** `--w-entropy` defaults to **0.01** (not 0) —
both real-teacher smoke runs showed grad_norm decaying to ~0 within 30
steps (score-head saturation kills the ST soft path); the entropy term
demonstrably keeps `score_entropy_mean` rising instead of collapsing.
`--entropy-anneal-steps N` linearly decays it to 0 over N steps (ST-MoE
prescription for long runs; default 0 = constant).

**Collapse monitoring (built into the trainer log):** every `--log-every`
steps the trainer re-selects a FIXED probe clip at a fixed ratio in eval
mode and logs `probe_overlap_prev` (IoU with the previous probe selection)
plus `score_entropy_mean`. A `probe_overlap_prev` pinned at 1.0 together
with dying `grad_norm` = the selector has frozen onto a constant,
content-independent pattern — the exact failure mode to watch for. (For
reference, random selection at ratio 0.3 gives IoU ≈ 0.3.)

Suggested first real experiment matrix (one knob at a time):
1. `w_pred=1, w_ent=0.01` (baseline — entropy default on, see above).
2. `w_pred=1, w_ent=0.01, w_v0=1, warmup=500` (prior-guided start).
3. `w_pred=1, w_ent=0.01, w_uniqueness=0.1..0.5` (anti-scatter; watch that
   coverage doesn't degrade too far — the two terms pull oppositely by
   design).
4. Ablate `--input-mode maps|pixels|both` and `--input-v0-preset v0.2|v0.1`
   on the best of the above.

## 4. v0 leverage options (all config-switchable, default off unless noted)

- `--input-v0-preset v0.2` (default): the non-learned signal generation
  feeding the learned scorer uses the gate-validated v0.2 SIGNAL knobs
  (frame-diff motion, noise floor, score blend); selection-stage knobs
  (global allocation, block gate) are irrelevant to the maps and internally
  overridden to the cheap defaults. `v0.1` gives the plain baseline signals
  for ablation.
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

## 7. Linux/CUDA scale-run preparation checklist

Everything below was audited on 2026-07-14; the codebase itself needs no
changes to run on Linux (all ops are standard aten with CUDA kernels; DDP
activates automatically under torchrun).

**1. Environment**
```bash
git clone git@github.com:manricheon/AutoGaze.git && cd AutoGaze
git checkout feat/borissal
uv venv --python 3.11
uv pip install -e .          # torch (CUDA wheel on Linux), transformers==5.5.0, av, timm, einops
uv pip install -e '.[dev]'   # pytest (optional but recommended: run the suite once)
# Do NOT install .[cuda] (flash_attn) -- legacy-stack only, not used by Borissal.
uv run pytest tests/ -q      # expect 43 passed
```
torch.hub V-JEPA2 repo deps (`torch`, `timm`, `einops`) are already in the
base dependencies.

**2. Teacher checkpoint (pick one)**
- V-JEPA 2.1-L via torch.hub (`--teacher hub:vjepa2_1_vit_large_384`,
  `--scale 384`): pre-download the weights first (upstream main hardcodes a
  localhost URL -- see §5):
  ```bash
  mkdir -p ~/.cache/torch/hub/checkpoints
  curl -L -o ~/.cache/torch/hub/checkpoints/vjepa2_1_vitl_dist_vitG_384.pt \
      https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt
  ```
  (B variant: `vjepa2_1_vitb_dist_vitG_384.pt`, already validated locally.)
- HF V-JEPA2-L (`--teacher facebook/vjepa2-vitl-fpc64-256`, `--scale 256`):
  downloads automatically on first run; plain dense-target coverage loss.

**3. Data**
```bash
hf download bfshi/AutoGaze-Training-Data --repo-type dataset --local-dir AutoGaze-Training-Data
```
Any folder of .mp4s works (recursive glob); gazing_labels.json is NOT
needed (fully self-supervised). Point `--data-root` at it.

**4. Launch (see §6 for the full command)** -- key flags for the first
real run, per the §3 matrix baseline: `--w-entropy 0.01` (default),
`--ratio-sampling uniform`, `--num-workers 8` (PyAV decode is the IO
bottleneck; scale with CPU cores), `--save-every 1000`.

**5. Watch during training** (`train_log.jsonl`):
- `loss/predictor_coverage` trending down (never observed on the
  degenerate Mac smoke sets; this is the first real learning-curve test),
- `grad_norm` NOT decaying to ~0 (score saturation -- raise `--w-entropy`
  or lower `--lr` if it does),
- `probe_overlap_prev` NOT pinned at 1.0 (frozen-selection collapse),
- `score_entropy_mean` stable or rising,
- `peak_mem_mb` (CUDA-accurate) for batch-size headroom.

**6. Known gaps accepted for the first run** (revisit if they bite):
- No optimizer-state resume (checkpoints are model-only; `--save-every`
  gives restart points but training restarts cold from a state_dict).
- `video_io.load_video` decodes the full clip before sampling frames --
  fine for the pre-trimmed AutoGaze-Training-Data clips, wasteful for long
  videos; `--num-workers` is the mitigation.
- wandb is installed but the trainer logs to stdout+jsonl only.

**7. Scale-run options from the theory survey deliberately deferred to
Linux compute** (§8 has the rationale):
- REAL-X-style calibration: fine-tune a light adapter on the predictor
  with RANDOM masks spanning the 0.15–0.75 keep range before trusting it
  as a reward model (the frozen 2.1 predictor was trained on ~90%-masked
  multi-block geometry — our keep ratios are off-distribution for it).
  Never train it jointly with the selector (L2X "selection as
  communication" degeneracy).
- EVAL-X-style audit: report reward under our evaluator vs under an
  independent random-mask-calibrated evaluator; a large gap = the selector
  is exploiting the frozen predictor's inductive bias, not finding
  information.
- Frame-Voyager-style model selection: offline, rank candidate selections
  per validation clip by a frozen captioner's description loss — the only
  metric guaranteed aligned with the downstream description task.

## 8. Theory-driven upgrade (2026-07-14): why the objective had to change

Two parallel literature surveys (token selection / differentiable top-k;
SSL informative masking / selector training mechanics) were run against
our three MEASURED pathologies. Full findings + citations in design.md
("Theory notes"); what landed in code:

**The core finding — P1/P3 are the objective's optimum, not bugs.**
Coverage minimization (predict the rest from the selection) is a soft
facility-location / D-optimal-design objective: its provable optimum is
uniformly scattered anchors plus boundary points — exactly the measured
"random beats saliency" (P1) and edge/band drift (P3). I-JEPA's ablation
quantifies the same effect from the SSL side (scattered context 17.6 vs
multi-block 54.2, IN-1k 1% linear: scatter = interpolation shortcut), and
REAL-X explains the gradient mechanics: a selector trained against a
frozen evaluator optimizes that evaluator's off-distribution inductive
bias (the 2.1 predictor was trained at ~90% multi-block masking; our
15–75%-keep scattered contexts are far off-distribution). Conclusion: no
regularizer fixes P1/P3 while dense-feature coverage remains the
maximized objective — hence the WP-B inversion.

**Phased recipe (StableMoE-style: learn routing → distill → commit):**
1. **Warmup** — `--w-v0-distill 1 --v0-distill-warmup-steps 500`: start
   from the proven v0 saliency prior (doubles as HPM's easy-to-hard
   curriculum; HPM measured that ALL-hard selection underperforms random).
2. **ST phase** — uniqueness-primary + coverage floor:
   `--w-uniqueness 1.0 --coverage-floor <random baseline from
   eval_borissal_coverage.py> --w-zloss 1e-3 --w-entropy 0.01
   [--train-block-size 2] [--w-hardness 0.1]`. τ fixed at 2/3, no
   annealing.
3. **RL phase (optional)** — `--rl-after-step N --rl-samples 4
   --rl-cov-weight 1.0`: REINFORCE with the Kool leave-one-out baseline
   (AdaMAE precedent: same frozen-teacher + tiny-selector setup, REINFORCE
   made foreground saliency emerge). Reward = uniqueness −
   rl-cov-weight·relu(coverage − floor), computed entirely under no_grad —
   NO teacher backward graph, so peak memory is LOWER than the ST phase;
   cost is 2·rl-samples predictor forward passes per step. Unbiased
   gradients are immune to softmax saturation by construction. Caveats:
   per-token sampling only (`train_block_size` shapes just the ST phase);
   `--rl-samples ≥ 2` required (LOO needs peers).

**Watch (in addition to §7.5):** `loss/rl_adv_std` should stay > 0 (zero =
all samples get identical reward = no learning signal); `score_entropy_mean`
falling fast in the ST phase → raise `--w-zloss` before `--w-entropy`
(z-loss attacks the cause, entropy the symptom).

**Deliberately NOT adopted** (full reasons in design.md): DPP/diversity
regularizers (our objective is already over-spread), SIMPLE/perturbed
top-k estimators (the z-loss+cosine bundle plus the RL phase cover the
same failure modes more cheaply at our n=4608), GradNorm/uncertainty loss
balancing (fixed weights win head-to-head, arXiv:2201.04122), SemMAE/
AutoMAE-style part/GAN machinery (block sampling achieves the contiguity
prior directly).
