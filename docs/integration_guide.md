# AutoGaze Integration Guide

This document covers two topics:

1. **Integration modes** — how AutoGaze attaches to an existing ViT/MLLM.
2. **Adding a new backend** — step-by-step guide for plugging in a new ViT, a new LLM, or a new ViT+LLM combination.

---

## 1. System Architecture

AutoGaze sits between raw video frames and the vision encoder.  It selects the most informative patch positions and discards the rest before (or inside) the ViT.

```text
Video frames
    │
    ▼
AutoGaze model  →  gaze mask  (14 × 14 binary per frame)
    │
    ▼
Vision Encoder  (processes only selected patches)
    │
    ▼
LLM / task head  →  answer / features
```

---

## 2. Integration Modes

### Quick reference

| Mode | Mechanism | Sequence length | Latency gain | Use case |
|:--|:--|:--|:--:|:--|
| **native** | AutoGaze baked into the model processor | Reduced internally | ✅ yes | NVILA only; most transparent |
| **hook** | Forward hook zeroes non-selected token embeddings | **Unchanged** | ❌ no | Zero-shot validation — no model code changes |
| **full** | ViT `forward()` modified to skip non-selected tokens | **Reduced** | ✅ yes | Latency and VRAM benchmarks |

---

### Where each mode intercepts the pipeline

```text
  frames ──▶ AutoGaze ──▶ gaze mask (T × 196)
                               │
         ┌─────────────────────┼─────────────────────┐
         ▼                     ▼                     ▼

      NATIVE                 HOOK                  FULL
      ─────────────          ─────────────          ─────────────
      Processor              ViT runs               ViT skips
      pre-selects            full N-token           non-sel. toks
      patches before         seq; hook              inside
      ViT runs               zeros dead             forward()
                             embeddings

      seq: k tokens          seq: N tokens          seq: k tokens
      speed: faster          speed: same            speed: faster
      NVILA only             any model              code required
         │                     │                      │
         └─────────────────────┴──────────────────────┘
                                     │
                               LLM / task head
```

---

### Token sequence: what the LLM actually sees

Suppose `gazing_ratio = 0.75` and a single frame has `196` patches (14 × 14 grid).  
`▪` = active patch, `·` = zeroed (hook) or removed (full).  Grids are schematic — each cell ≈ 4 real patches.

```text
             Baseline              Hook mode             Full mode
            ┌────────────┐        ┌────────────┐        ┌─────────┐
            │▪▪▪▪▪▪▪▪▪▪▪▪│        │▪▪▪·▪·▪▪▪···│        │▪▪▪▪▪▪▪▪▪│
            │▪▪▪▪▪▪▪▪▪▪▪▪│        │·▪▪▪▪·▪·▪·▪▪│        │▪▪▪▪▪▪▪▪▪│
            │▪▪▪▪▪▪▪▪▪▪▪▪│        │▪▪·▪▪▪▪·▪▪·▪│        │▪▪▪▪▪▪▪▪▪│
            │▪▪▪▪▪▪▪▪▪▪▪▪│        │▪·▪▪·▪▪▪·▪▪·│        │▪▪▪▪▪▪▪▪▪│
            ├────────────┤        ├────────────┤        ├─────────┤
            │ 196 tokens │        │ 196 slots  │        │~147 tok.│
            │ (all kept) │        │(~147 ▪, ·=0│        │(removed)│
            └────────────┘        └────────────┘        └─────────┘
                  ▼                     ▼                    ▼
            seq_len = 196          seq_len = 196        seq_len ≈ 147
            attn: O(196²)          attn: O(196²)        attn: O(147²)
            baseline speed         no speed gain        ~1.8× faster
```

---

### Mode-by-mode explanation

#### native (NVILA only)

AutoGaze is compiled into the NVILA image processor.  The processor converts `gazing_ratio` into a tiling strategy before the ViT even runs — so no post-hoc patching is needed.

```text
  NVILA Processor  (native mode)
  ┌──────────────────────────────────────────────┐
  │  frame ──▶ AutoGaze ──▶ gaze_ratio_tile=0.75 │
  │                │                             │
  │                ▼                             │
  │          select_patches() ──▶ k-token list   │
  │                                    │         │
  │                               passed to ViT  │
  └──────────────────────────────────────────────┘
```

Requires `--autogaze-path` even for baseline (the processor reads the config on `__init__`).  Use `--no-autogaze` to set `gazing_ratio=1.0` which passes all patches (equivalent to off).

#### hook

A PyTorch forward hook is registered on the **last encoder block**.  After the block produces its output tensor `(B, N, C)`, the hook multiplies by the gaze mask — zeroing out embeddings for non-selected patches.  No ViT code changes needed.

```text
  ViT encoder (all N blocks run normally)
       │
       ▼  last block output: (B, 196, C)
       │
  hook applied ──▶  output × gaze_mask(196,)
       │                (non-selected rows → 0)
       ▼
  masked output: (B, 196, C)   ← seq_len unchanged
```

**Advantage**: works on any ViT out of the box.  **Limitation**: the full sequence still flows through all attention layers, so no latency reduction.

#### full

The ViT's `forward()` method is modified (or subclassed) to physically remove non-selected tokens before the attention computation.  Only selected tokens enter the transformer layers.

```text
  patch embedding: (B, 196, C)
          │
  gaze mask ──▶ gather (keep k=~147 rows)
          │
          ▼  (B, ~147, C)    ← seq_len reduced before attention
          │
  transformer layers
  (KV cache size ∝ k, not N)
          │
          ▼  (B, ~147, C) out
```

**Advantage**: $O(k^2)$ attention instead of $O(N^2)$ — real latency and VRAM savings.  **Limitation**: requires per-ViT code modification (see compatibility matrix below).

---

### Which mode should I use?

```text
  New model? ──▶ Start with HOOK
                     │
                     ▼
              Accuracy OK? ──No──▶ Check AutoGaze weights / gazing_ratio
                     │
                    Yes
                     │
                     ▼
              Need speed? ──No──▶ Stay on HOOK
                     │
                    Yes
                     │
                     ▼
               Use FULL mode
               (see compatibility matrix)
```

Hook mode accuracy ≈ full mode accuracy.  Speed improvement only visible in full mode.

---

## 3. Compatibility Matrix

| ViT type | Hook | Full | Required fix for Full |
| :--- | :---: | :---: | :--- |
| Image ViT (SigLIP / CLIP) | ✅ | ✅ | Block-causal attention mask |
| Video ViT — absolute PE (V-JEPA2) | ✅ | ✅ | None (native temporal awareness) |
| Video ViT — RoPE (e.g. future models) | ✅ | ✅ | RoPE position re-indexing after masking |
| Hierarchical / window attention (Qwen2.5) | ✅ | ✅ | Pre-reorder masking (before cu_seqlens) |

> Hook always works out of the box for any ViT.  The "Required fix" column applies only when upgrading to full mode.

---

### 3.1 Image ViT — Block-Causal Attention Mask

**The problem**: standard image ViTs use full bidirectional attention across all patch positions.  When full mode removes tokens, the spatial relationship between remaining tokens breaks — token `T3` (originally at column 3) now appears adjacent to `T1` (column 1), causing position confusion.

```text
  Original 8-patch sequence (row of a 14×14 grid):
  ┌────┬────┬────┬────┬────┬────┬────┬────┐
  │ T1 │ T2 │ T3 │ T4 │ T5 │ T6 │ T7 │ T8 │   positions 0..7
  └────┴────┴────┴────┴────┴────┴────┴────┘
   ← full bidirectional attention between all pairs →

  After full mode removes T2, T4, T6 (ratio ≈ 62.5%):
  ┌────┬────┬────┬────┬────┐
  │ T1 │ T3 │ T5 │ T7 │ T8 │   new positions 0..4
  └────┴────┴────┴────┴────┘
   ⚠ T3 now "thinks" it's at position 1, not 2
   ⚠ T1 and T3 appear adjacent, but T2 existed between them
```

**The fix — block-causal attention mask**: restrict each retained token to attend only to tokens that appear *lexicographically before it* in the original grid.  This preserves spatial ordering without needing absolute position recovery.

```text
  Retained tokens: T1  T3  T5  T7  T8
  Causal attention mask (✓ = can attend):

            T1  T3  T5  T7  T8
       T1 [  ✓   ·   ·   ·   · ]
       T3 [  ✓   ✓   ·   ·   · ]   T3 attends to T1 and itself
       T5 [  ✓   ✓   ✓   ·   · ]
       T7 [  ✓   ✓   ✓   ✓   · ]
       T8 [  ✓   ✓   ✓   ✓   ✓ ]

  Result: each token's context is its original left-spatial neighbors — no
  position confusion, no absolute index needed.
```

Reference: `autogaze/vision_encoders/siglip/` (NVILA native integration).

---

### 3.2 Video ViT — Absolute Positional Embeddings (V-JEPA2)

**No fix required for full mode.** V-JEPA2 adds absolute positional embeddings to each token *before* any attention layer.  When tokens are removed, the remaining tokens retain their correct embeddings — the model already knows where each token came from.

```text
  Before masking:
  token_emb + pos_emb(t=0, x=2, y=3)  → T_023
  token_emb + pos_emb(t=0, x=2, y=5)  → T_025
  token_emb + pos_emb(t=1, x=2, y=3)  → T_123

  After masking (remove T_025):
  T_023 and T_123 still carry their original (t,x,y) embeddings.
  The attention layers receive correct positional signals automatically.
```

**Temporal chunking note**: V-JEPA2 uses `tubelet_size=2`, so the gaze mask (one per frame) must be averaged over pairs of frames before application.  This is already handled in `autogaze/vision_encoders/vjepa2/`.

---

### 3.3 Video ViT — RoPE Positional Encoding

**The problem**: RoPE (Rotary Position Embedding) encodes position inside the attention operation itself — each query and key vector is *rotated* by an angle proportional to its position index.  After removing tokens, the remaining tokens get new sequential indices (0, 1, 2, …), but their rotations should use their **original** indices.

```text
  Original sequence (5 tokens):
  T0 T1 T2 T3 T4
  RoPE rotation angles: θ·0  θ·1  θ·2  θ·3  θ·4

  Full mode removes T1, T3:
  Remaining: T0  T2  T4
  ❌ Wrong:  new indices 0, 1, 2 → rotations θ·0  θ·1  θ·2
             T2 gets angle θ·1 but should have θ·2
             T4 gets angle θ·2 but should have θ·4

  ✅ Fix: keep original indices
             T0(idx=0) T2(idx=2) T4(idx=4) → rotations θ·0  θ·2  θ·4
```

**Implementation**:

```python
# After masking, retain original position ids — do NOT renumber
kept_positions = torch.where(gaze_mask)[0]   # original indices of kept tokens
# Pass kept_positions into the RoPE layer instead of torch.arange(N_kept)
rotary_emb = model.rotary_embedding(seq_len=kept_positions.max() + 1)
cos, sin = rotary_emb[kept_positions]        # select angles by original index
```

---

### 3.4 Hierarchical / Window Attention (Qwen2.5-VL)

**The problem**: Qwen2.5-VL groups patches into fixed-size spatial windows before the first attention layer.  The window boundaries (`cu_seqlens`) are computed from the total patch count — if masking happens *after* windowing, the window structure becomes inconsistent.

```text
  Original: 8 patches grouped into 2 windows of size 4
  ┌──────────────────┐  ┌──────────────────┐
  │ T1  T2  T3  T4  │  │ T5  T6  T7  T8  │
  │   window 1      │  │   window 2      │
  └──────────────────┘  └──────────────────┘
  cu_seqlens = [0, 4, 8]

  ❌ Wrong: mask after windowing
  After masking T2 and T6 inside already-formed windows:
  ┌──────────────────┐  ┌──────────────────┐
  │ T1  ·   T3  T4  │  │ T5  ·   T7  T8  │
  └──────────────────┘  └──────────────────┘
  cu_seqlens still [0, 4, 8] → length mismatch, CUDA error

  ✅ Fix: apply mask BEFORE windowing
  Masked tokens: T1  T3  T4  T5  T7  T8   (6 tokens)
  Regroup into windows of size 3:
  ┌──────────────┐  ┌──────────────┐
  │ T1  T3  T4  │  │ T5  T7  T8  │
  └──────────────┘  └──────────────┘
  cu_seqlens = [0, 3, 6]   ← recomputed from masked set
```

Reference: `autogaze/vision_encoders/qwen25vl/modeling_qwen25vl_ag.py`.

---

### 3.5 Summary: what you need to change per ViT

| ViT type | Attention style | Hook | Full — what to change |
| :--- | :--- | :---: | :--- |
| SigLIP / CLIP | Bidirectional | ✅ | Add block-causal attention mask |
| V-JEPA2 | Absolute PE | ✅ | Nothing — works out of the box |
| Future RoPE ViT | Rotary PE | ✅ | Pass original indices to RoPE |
| Qwen2.5-VL | Window attention | ✅ | Mask before `cu_seqlens` computation |
| Any new ViT | Unknown | ✅ | Start with hook, identify PE type, then apply the matching fix above |

---

## 4. Runner Naming Convention

Runner keys follow **`{vit}_{lm}`** — ViT name first, LLM name second.  Integration mode is a separate `--integration` flag, not part of the key.

| Runner key | ViT | LLM | Default integration |
| :--- | :--- | :--- | :--- |
| `nvila` | SigLIP (custom) | NVILA | native |
| `vjepa2_nvila` | V-JEPA2 | NVILA | full |
| `siglip_qwen25` | SigLIP (Qwen internal) | Qwen2.5-VL | hook |
| `vjepa2_qwen25` | V-JEPA2 | Qwen2.5-7B | full |
| `vjepa2` | V-JEPA2 | — (features only) | hook |
| `siglip` | SigLIP (HF vanilla) | — (features only) | hook |

---

## 5. Adding a New Backend

### 5.1 Adding a New ViT (hook mode — fast path)

Hook mode needs no model code changes.  The only requirement is that the ViT produces patch-level token embeddings you can zero out.

**Step 1 — create the encoder module** (optional, only if you need custom loading):

```text
autogaze/vision_encoders/<your_vit>/
    __init__.py
    modeling_<your_vit>_ag.py   ← AutoGaze-aware forward wrapper
```

**Step 2 — implement the hook**:

```python
import torch
from autogaze.models.autogaze.autogaze import AutoGazeModel
from autogaze.models.autogaze.image_processing_autogaze import AutoGazeImageProcessor

def apply_autogaze_hook(vit_model, ag_model, ag_processor, frames, gazing_ratio):
    """Zero out non-selected token embeddings via a forward hook."""
    # 1. Run AutoGaze to get the gaze mask
    inputs = ag_processor(images=frames, return_tensors="pt")
    with torch.no_grad():
        out = ag_model(inputs, gazing_ratio=gazing_ratio)
    gaze_mask = out["gazing_mask"][-1]   # (T, H_ag*W_ag)  — 14×14 = 196 positions

    # 2. Interpolate to the ViT's patch grid
    H_vit = W_vit = int(vit_model.config.image_size / vit_model.config.patch_size)
    mask_2d = gaze_mask.reshape(-1, 14, 14).float()
    mask_vit = torch.nn.functional.interpolate(
        mask_2d.unsqueeze(0), size=(H_vit, W_vit), mode="bilinear"
    ).squeeze(0).reshape(-1, H_vit * W_vit)   # (T, N_patches)

    # 3. Register the hook
    def _hook(module, input, output):
        # output: (B, N, C)  where N = H_vit * W_vit (+ optional CLS)
        n_patch = H_vit * W_vit
        output[:, -n_patch:] *= mask_vit.unsqueeze(-1).to(output.device)
        return output

    handle = vit_model.encoder.layers[-1].register_forward_hook(_hook)
    return handle   # call handle.remove() after the forward pass
```

**Step 3 — create a runner class**:

```python
from autogaze.eval.models import BaseMLLMRunner, _local
import torch

class MyViTRunner(BaseMLLMRunner):
    name = "myvit"
    supports_mcq = False   # feature extraction only (no LLM)

    def __init__(self, model_path, autogaze_path, gazing_ratio,
                 dtype=torch.bfloat16, integration="hook"):
        from transformers import AutoModel, AutoProcessor
        self.model = AutoModel.from_pretrained(
            model_path, local_files_only=_local(model_path), torch_dtype=dtype,
        ).eval()
        self.integration = integration
        self.gazing_ratio = gazing_ratio
        # load AutoGaze if requested
        if autogaze_path:
            from autogaze.models.autogaze.autogaze import AutoGazeModel
            from autogaze.models.autogaze.image_processing_autogaze import AutoGazeImageProcessor
            self.ag_model = AutoGazeModel.from_pretrained(autogaze_path).eval()
            self.ag_proc  = AutoGazeImageProcessor.from_pretrained(autogaze_path)
        else:
            self.ag_model = self.ag_proc = None

    def run(self, frames, prompt, max_new_tokens=16):
        raise NotImplementedError("MyViTRunner is feature-extraction only.")
```

**Step 4 — register in `RUNNERS`**:

```python
# autogaze/eval/models.py  (bottom of file, primary keys section)
from autogaze.vision_encoders.myvit.modeling_myvit_ag import MyViTRunner

RUNNERS["myvit"] = MyViTRunner
_RUNNER_DEFAULTS["myvit"] = {"integration": "hook"}
```

**Step 5 — test with the benchmark CLI**:

```bash
python -m autogaze.eval.run_benchmark \
    --task videomme \
    --mllm myvit \
    --model-path weights/MyViT \
    --autogaze-path weights/AutoGaze \
    --integration hook \
    --max-samples 50
```

---

### 5.2 Upgrading to Full Mode

Full mode physically removes non-selected tokens inside the ViT's `forward()`.  This requires modifying (or wrapping) the ViT's transformer backbone.

**Pattern**:

```python
# autogaze/vision_encoders/<your_vit>/modeling_<your_vit>_ag.py

class AutoGazeMyViTEncoder(nn.Module):
    """Drop-in replacement for <YourViT>.encoder with AutoGaze masking."""

    def __init__(self, original_encoder, gaze_mask):
        super().__init__()
        self.encoder = original_encoder
        # gaze_mask: (N_patches,) bool — True = keep
        self.register_buffer("gaze_mask", gaze_mask)

    def forward(self, hidden_states, **kwargs):
        # Select only gazed tokens
        hidden_states = hidden_states[:, self.gaze_mask]    # (B, k, C)
        return self.encoder(hidden_states, **kwargs)
```

**Key considerations by ViT type**:

| ViT type | What to fix |
| :--- | :--- |
| **Image ViT** (SigLIP) | Token removal changes spatial positions → add block-causal attention mask so remaining tokens attend only to spatially preceding ones |
| **Video ViT, absolute PE** (V-JEPA2) | No position correction needed; absolute embeddings are set before masking |
| **Video ViT, RoPE** | After removal, remaining token indices must be remapped to their original positions before applying RoPE frequencies |
| **Hierarchical / window attn** (Qwen2.5) | Apply mask before the window-reordering step (`cu_seqlens` computation) |

See `autogaze/vision_encoders/vjepa2/modeling_vjepa2_ag.py` (V-JEPA2 full mode) and `autogaze/vision_encoders/qwen25vl/modeling_qwen25vl_ag.py` (Qwen2.5 full mode) as reference implementations.

---

### 5.3 Adding a New LLM (paired with an existing ViT)

**Step 1 — subclass an existing ViT runner**:

```python
class MyViTMyLLMRunner(MyViTRunner):
    name = "myvit_myllm"
    supports_mcq = True

    def __init__(self, model_path, autogaze_path, gazing_ratio,
                 lm_path, projector_path=None, dtype=torch.bfloat16, **kw):
        # Load the ViT part via super().__init__
        super().__init__(model_path, autogaze_path, gazing_ratio, dtype=dtype, **kw)

        # Load the LLM
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.lm = AutoModelForCausalLM.from_pretrained(
            lm_path, local_files_only=_local(lm_path), torch_dtype=dtype, device_map="auto",
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(lm_path, local_files_only=_local(lm_path))

        # Load or create a projector (maps ViT dim → LLM dim)
        from autogaze.vision_encoders.vjepa2.projector import VJEPA2Projector
        vit_hidden = self.model.config.hidden_size
        lm_hidden  = self.lm.config.hidden_size
        if projector_path:
            self.projector = VJEPA2Projector.from_pretrained(projector_path)
        else:
            self.projector = VJEPA2Projector(vit_hidden, lm_hidden)

    def run(self, frames, prompt, max_new_tokens=16):
        # 1. Encode video with AutoGaze
        with torch.no_grad():
            video_feats = self.encode_video(frames)   # (1, T, C_vit)
        video_tokens = self.projector(video_feats)    # (1, T, C_lm)

        # 2. Build input: [video tokens] + [text tokens]
        text_ids  = self.tokenizer(prompt, return_tensors="pt").input_ids
        text_emb  = self.lm.get_input_embeddings()(text_ids.to(video_tokens.device))
        inputs_emb = torch.cat([video_tokens, text_emb], dim=1)

        # 3. Generate
        with torch.inference_mode():
            out = self.lm.generate(inputs_embeds=inputs_emb, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()
```

**Step 2 — register**:

```python
RUNNERS["myvit_myllm"] = MyViTMyLLMRunner
_RUNNER_DEFAULTS["myvit_myllm"] = {"integration": "full"}
```

**Step 3 — run**:

```bash
python -m autogaze.eval.run_benchmark \
    --task videomme \
    --mllm myvit_myllm \
    --model-path weights/MyViT \
    --autogaze-path weights/AutoGaze \
    --gazing-ratio 0.75 \
    -- lm_path=weights/MyLLM
```

> **Note**: extra constructor kwargs (`lm_path`, `projector_path`, `vjepa2_path`) can be passed as `**runner_kwargs` via `load_runner()` or from a Python call.  The CLI currently exposes `--vjepa2-path` only for `vjepa2_nvila`; add a similar argument for new runners that need it.

---

### 5.4 Minimal Checklist for a New Backend

```text
□  Encoder module (autogaze/vision_encoders/<vit>/) or inline in models.py
□  Runner class subclasses BaseMLLMRunner
□  Runner implements load (in __init__) + run()
□  name = "<vit>_<lm>" set on the class
□  RUNNERS["<vit>_<lm>"] = MyRunner registered
□  _RUNNER_DEFAULTS["<vit>_<lm>"] = {"integration": "hook"|"full"} set
□  Hook mode validated first (--integration hook, check accuracy delta)
□  Full mode implemented if latency/VRAM benchmarks needed
□  Entry added to runner table in docs/eval_guide.md
□  GEMINI.md updated (local guide)
```

---

## 6. Integration Mode Deep Dives

### 6.1 Block-Causal Masking (Image ViT → Full Mode)

Standard image ViTs use bidirectional attention.  When tokens are removed, remaining tokens lose their spatial context.  The fix is to re-introduce causal structure so each token only attends to the *lexicographically preceding* retained tokens.

```text
  Full 14×14 grid (196 patches), rasterised row-major:

   col →   0   1   2   3   4 … 13
  row
   0      T0  T1  T2  T3  T4 … T13
   1     T14 T15 T16 T17 T18 … T27
   …
  13    T182 …                T195

  Gaze mask keeps ~147 of these.  After removal, remaining tokens are
  re-indexed 0..146.  Block-causal mask ensures token at new index i
  can only attend to tokens at new indices ≤ i — preserving the
  original left-to-right, top-to-bottom spatial ordering.
```

Reference: `autogaze/vision_encoders/siglip/` (NVILA native integration).

---

### 6.2 RoPE Position Correction (RoPE ViT → Full Mode)

RoPE encodes position via query/key rotation.  After token removal, the rotation frequencies must be recomputed from each token's **original** index, not its new sequential position.

```text
  Without fix:                     With fix:
  T0  T2  T4  T6                   T0  T2  T4  T6
  ↓   ↓   ↓   ↓                    ↓   ↓   ↓   ↓
  θ·0 θ·1 θ·2 θ·3  ← wrong         θ·0 θ·2 θ·4 θ·6  ← correct
      (gaps ignored)                    (original indices preserved)
```

```python
# After masking, retain original position ids
kept_positions = torch.where(gaze_mask)[0]   # original indices of kept tokens
# Pass kept_positions to the RoPE layer instead of arange(N_kept)
```

---

### 6.3 Temporal Chunking (Video ViTs)

AutoGaze produces one gaze map per input frame (T maps total).  Video ViTs typically reduce temporal resolution via tubelet embedding before their attention layers.

```text
  Input (6 frames, one 196-score gaze map each):
  F0  F1     F2  F3     F4  F5

  Tubelet grouping (tubelet_size = 2):
  ┌────────┐  ┌────────┐  ┌────────┐
  │ F0, F1 │  │ F2, F3 │  │ F4, F5 │   ← 3 temporal chunks
  └───┬────┘  └───┬────┘  └───┬────┘
      ▼            ▼            ▼
  avg(F0,F1)  avg(F2,F3)  avg(F4,F5)   ← merge gaze scores
      ▼            ▼            ▼
  ┌────────┐  ┌────────┐  ┌────────┐
  │ mask_0 │  │ mask_1 │  │ mask_2 │   ← 196-binary mask per chunk
  └───┬────┘  └───┬────┘  └───┬────┘
      ▼            ▼            ▼
  chunk_0      chunk_1      chunk_2    ← applied to patch embeddings
```

This is already implemented in `autogaze/vision_encoders/vjepa2/` and `autogaze/vision_encoders/qwen25vl/`.

---

## 7. Performance Reference

| Configuration | Tokens | Latency (ms) | VRAM (GB) |
| :--- | :---: | :---: | :---: |
| SigLIP / NVILA — baseline | 3,136 | ~320 | ~18.5 |
| SigLIP / NVILA — AutoGaze full (75%) | 784 | ~145 | ~16.8 |
| V-JEPA2 — baseline | 1,568 | ~55 (ViT) | ~2.1 (ViT) |
| V-JEPA2 — AutoGaze full (75%) | 392 | ~22 (ViT) | ~1.1 (ViT) |

*16-frame 224px clips, A100 GPU.*
