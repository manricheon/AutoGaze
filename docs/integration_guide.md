# AutoGaze Integration Guide

This document covers two topics:

1. **Integration modes** — how AutoGaze attaches to an existing ViT/MLLM.
2. **Adding a new backend** — step-by-step guide for plugging in a new ViT, a new LLM, or a new ViT+LLM combination.

---

## 1. System Architecture

AutoGaze sits between raw video frames and the vision encoder.  It selects the most informative patch positions and discards the rest before (or inside) the ViT.

```
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

| Mode | Mechanism | Sequence length | Use case |
|:--|:--|:--|:--|
| **native** | AutoGaze baked into the model processor | Reduced internally | NVILA only; most transparent |
| **hook** | Forward hook zeroes non-selected token embeddings | **Unchanged** ($N_{all}$) | Zero-shot accuracy validation for new models |
| **full** | ViT `forward()` modified to skip non-selected tokens | **Reduced** ($N_{gazed}$) | Latency and VRAM benchmarks |

### Which mode should I use?

- **Start with hook** when adding a new model.  It requires no model code changes and lets you confirm that AutoGaze helps accuracy before investing in full integration.
- **Switch to full** once hook accuracy is validated.  Only full mode gives the $O(k^2)$ attention speedup and VRAM reduction.

Hook mode accuracy ≈ full mode accuracy.  Speed improvement only visible in full mode.

---

## 3. Compatibility Matrix

| ViT type | Hook | Full | Required fix for Full |
|:--|:--:|:--:|:--|
| Image ViT (SigLIP / CLIP) | ✅ | ✅ | Block-causal attention mask |
| Video ViT — absolute PE (V-JEPA2) | ✅ | ✅ | None (native temporal awareness) |
| Video ViT — RoPE (e.g. future models) | ✅ | ✅ | RoPE position re-indexing after masking |
| Hierarchical / window attention (Qwen2.5) | ✅ | ✅ | Pre-reorder masking (before cu_seqlens) |

---

## 4. Runner Naming Convention

Runner keys follow **`{vit}_{lm}`** — ViT name first, LLM name second.  Integration mode is a separate `--integration` flag, not part of the key.

| Runner key | ViT | LLM | Default integration |
|:--|:--|:--|:--|
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

```
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
|:--|:--|
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

```
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

Standard image ViTs use bidirectional attention.  When tokens are removed, remaining tokens lose their spatial context.  The fix is to re-introduce causal structure so each token only attends to the *lexicographically preceding* retained tokens (preserving relative spatial order without needing absolute position recovery).

Reference: `autogaze/vision_encoders/siglip/` (NVILA native integration).

### 6.2 RoPE Position Correction (RoPE ViT → Full Mode)

RoPE encodes position via query/key rotation.  After token removal, the rotation frequencies must be recomputed from each token's **original** index, not its new position in the shorter sequence.  Otherwise the model sees incorrect relative distances.

Pattern:
```python
# After masking, retain original position ids
kept_positions = torch.where(gaze_mask)[0]   # original indices of kept tokens
# Pass kept_positions to the RoPE layer instead of arange(N_kept)
```

### 6.3 Temporal Chunking (Video ViTs)

AutoGaze operates at frame level (T gaze maps for T frames).  Most video ViTs reduce temporal resolution via tubelet embedding (`tubelet_size=2` → $T_p = T / 2$ patch groups).  When applying the gaze mask:

1. Average gaze scores within each tubelet group → one mask per temporal chunk.
2. Apply the per-chunk mask to the corresponding patch embeddings.

This is already implemented in `autogaze/vision_encoders/vjepa2/` and `autogaze/vision_encoders/qwen25vl/`.

---

## 7. Performance Reference

| Configuration | Tokens | Latency (ms) | VRAM (GB) |
|:--|:--:|:--:|:--:|
| SigLIP / NVILA — baseline | 3,136 | ~320 | ~18.5 |
| SigLIP / NVILA — AutoGaze full (75%) | 784 | ~145 | ~16.8 |
| V-JEPA2 — baseline | 1,568 | ~55 (ViT) | ~2.1 (ViT) |
| V-JEPA2 — AutoGaze full (75%) | 392 | ~22 (ViT) | ~1.1 (ViT) |

*16-frame 224px clips, A100 GPU.*
