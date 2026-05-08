# AutoGaze Model Guide

AutoGaze is an autoregressive gaze model that predicts **which patches of a video frame carry the most information**, enabling downstream vision encoders and MLLMs to skip the rest.  This guide covers the model's purpose, architecture, training, and inference in detail.

---

## 1. Introduction

### 1.1 The Problem

Standard video ViTs process every patch of every frame with equal compute.  For a 16-frame video at 224×224 with patch size 16, that is **16 × 196 = 3,136 tokens** — most of which carry background or redundant content.

```text
  All frames, all patches          After AutoGaze
  ────────────────────────         ─────────────────────
  ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪        ▪ · ▪ ▪ · · · ▪ ▪ ·
  ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪        · ▪ · ▪ ▪ · ▪ · · ▪
  ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪        ▪ ▪ · · · ▪ ▪ ▪ · ·
  ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪        · · ▪ · ▪ · · ▪ ▪ ▪
  3,136 tokens / 16 frames         ~784 tokens (75% kept)
  O(3136²) attention               O(784²)  ~1.8× faster
```

### 1.2 The Solution

AutoGaze is a **lightweight autoregressive model** (< 50 MB) that runs before the heavy ViT.  It outputs a binary **gaze mask** — one 196-bit vector per frame — that marks which 16×16 patches to keep.

```text
  Video frames
       │
       ▼
  ┌───────────┐
  │ AutoGaze  │  ← small, fast (~50 MB)
  └─────┬─────┘
        │   gaze mask  (T × 196 binary)
        ▼
  ┌───────────────────────────────┐
  │  Downstream ViT / MLLM        │  ← sees only selected patches
  └───────────────────────────────┘
```

### 1.3 Key Properties

| Property | Value |
| :--- | :--- |
| Model size | ~50 MB |
| Default patch vocabulary | 196 IDs (14×14 grid) + 1 EOS = 197 tokens |
| Typical gazing ratio | 0.25–0.75 (25–75% patches kept) |
| Accuracy drop (VideoMME) | < 0.1 pp at ratio 0.75 |
| ViT speedup (ratio 0.75) | ~1.8× attention; 2–4× end-to-end with full integration |

---

## 2. Model Architecture

### 2.1 Overview

AutoGaze has three trainable components in sequence:

```text
  ┌─────────────────────────────────────────────────────────────────┐
  │                       AutoGaze Model                            │
  │                                                                 │
  │  ┌──────────────────┐   ┌───────────┐   ┌─────────────────┐   │
  │  │ ShallowVideo     │   │ Connector │   │  Gaze Decoder   │   │
  │  │ ConvNet          │──▶│           │──▶│  (LLaMA-based)  │   │
  │  │ (vision encoder) │   │ pos embed │   │  AR generation  │   │
  │  └──────────────────┘   └───────────┘   └─────────────────┘   │
  │                                                │                │
  │  Input: (B, T, 3, 224, 224)         Output: gaze_pos (B, T, k) │
  └─────────────────────────────────────────────────────────────────┘
```

---

### 2.2 Vision Encoder — ShallowVideoConvNet

A deliberately **shallow** (not a full ViT) 3D convolutional network.  It only needs to extract enough spatial and temporal context to guide patch selection — not to fully understand the scene.

```text
  Input video: (B, T, 3, H, W)  e.g. (1, 16, 3, 224, 224)
         │
         ▼
  ┌──────────────────────────────────────────────────────────┐
  │  temporal_conv: Conv3d(3 → hidden_dim,                   │
  │    kernel=(temporal_patch_size, 16, 16),                 │
  │    stride=(temporal_patch_size, 16, 16))                 │
  │                                                          │
  │  → patchifies frames spatially (16px stride → 14×14)    │
  │  → groups adjacent frames temporally (tubelet)          │
  └──────────────────────┬───────────────────────────────────┘
         │  (B, hidden_dim, T', 14, 14)
         ▼
  LayerNorm (per spatial position)
         │
         ▼
  ┌──────────────────────────────────────────────────────────┐
  │  N × Conv3dBlockForStreaming                             │
  │  (Conv3d + ReLU, causal padding on time axis)           │
  │  supports streaming: reuses previous frame context       │
  └──────────────────────┬───────────────────────────────────┘
         │  (B, hidden_dim, T', 14, 14)
         ▼
  out_proj: Conv3d(hidden_dim → out_dim, kernel=1)
         │
         ▼
  Output: (B, T', 196, out_dim)   e.g. (1, 16, 196, 192)
```

**Why shallow?**  A full ViT would be too slow — AutoGaze must be faster than what it saves.  A few 3D conv layers capture enough local spatio-temporal texture to rank patches reliably.

**Streaming support:**  Each `Conv3dBlockForStreaming` accepts `past_conv_values` (the last `temporal_kernel_size − 1` temporal frames).  This allows frame-by-frame streaming without re-processing past frames.

---

### 2.3 Connector

A minimal learned projection.  It adds a **positional embedding** to each patch position so the gaze decoder can distinguish spatial locations.

```text
  Input:  (B, T', 196, 192)  ← raw vision features
                │
  pos_embed:  (196, 192)     ← learned, one vector per patch position
                │
  Output: x + pos_embed      ← shape unchanged: (B, T', 196, 192)
```

The 196 positions correspond to the 14×14 patch grid (row-major order):

```text
  ┌────┬────┬────┬─────────────────────┬────┐
  │  0 │  1 │  2 │  …                  │ 13 │  ← row 0
  ├────┼────┼────┼─────────────────────┼────┤
  │ 14 │ 15 │ 16 │  …                  │ 27 │  ← row 1
  ├────┼────┼────┼─────────────────────┼────┤
  │ …  │    │    │                     │    │
  ├────┼────┼────┼─────────────────────┼────┤
  │182 │183 │184 │  …                  │195 │  ← row 13
  └────┴────┴────┴─────────────────────┴────┘
  patch_id = row × 14 + col        (0 to 195)
  EOS token = 196
```

---

### 2.4 Gaze Decoder — LLaMA-based AR Model

The gaze decoder is a small **LLaMA transformer** repurposed as an autoregressive sequence generator over patch indices.

**Vocabulary**: instead of natural language words, the vocabulary is patch indices:

```text
  Token ID  │  Meaning
  ──────────┼──────────────────────────────
  0 – 195   │  patch at position (id // 14, id % 14)
  196       │  EOS — "stop selecting for this frame"
```

**Sequence structure** (decoder input during generation):

```text
  ┌─────────────────────┬──────────────┬─────────────────────┬──────────────┬────┐
  │   frame 0 vision    │  frame 0     │   frame 1 vision    │  frame 1     │    │
  │   196 patch embeds  │  gaze tokens │   196 patch embeds  │  gaze tokens │ …  │
  │   (from connector)  │  k₀ IDs      │   (from connector)  │  k₁ IDs      │    │
  └─────────────────────┴──────────────┴─────────────────────┴──────────────┴────┘
  ◀─────────────────── grows left-to-right as generation proceeds ──────────────▶
```

Vision tokens are **continuous embeddings** (from `ShallowVideoConvNet + Connector`).  Gaze tokens are **discrete integer embeddings** from the decoder's own `embed_tokens` table.

**Logits processors** (active during `generate()`):

| Processor | Effect |
| :--- | :--- |
| `NoRepeatTokensLogitsProcessor` | Sets score of any already-generated patch to −∞ (no re-selection) |
| `NoEosTokenLogitsProcessor` | Sets EOS score to −∞ (prevents premature stop; generation ends via `max_new_tokens`) |

**Multi-token prediction** (`num_multi_token_pred = K`):

```text
  Standard (K=1):                Multi-token pred (K=2):
  ─────────────────              ─────────────────────────
  step 0: predict p₀             step 0: predict p₀, p₁  (2 tokens simultaneously)
  step 1: predict p₁             step 1: predict p₂, p₃
  step 2: predict p₂             …
  …
  N steps for N patches          ⌈N/2⌉ steps for N patches
```

The decoder head outputs `K × vocab_size` logits per position.  At inference, each of the K predictions is applied to K consecutive gaze positions in the context, reducing forward passes by K×.

---

## 3. Training

### 3.1 Training Stages

AutoGaze is trained in two stages:

```text
  Stage 1 — NTP Pre-training (Supervised Fine-tuning)
  ────────────────────────────────────────────────────
  Objective:  next-token prediction on pseudo-labels
  Labels:     ground-truth patch selection sequences
              (computed offline by VideoMAE reconstruction quality)
  Loss:       cross-entropy over generated patch IDs

  Stage 2 — GRPO (Reinforcement Learning)
  ────────────────────────────────────────
  Policy:     AutoGaze generates patch selection sequences
  Reward:     VideoMAE reconstruction loss on selected patches
              (lower reconstruction loss = better selection)
  Algorithm:  GRPO (Group Relative Policy Optimization)
              samples K candidate sequences, scores each by reward,
              updates policy toward higher-reward selections
```

### 3.2 Reward Signal

VideoMAE acts as a **reconstruction critic**:

```text
  Candidate gaze sequence
        │   (selected patches for one frame)
        ▼
  ┌─────────────┐
  │  VideoMAE   │  masked autoencoder — reconstructs masked regions
  │  encoder    │  from only the selected (unmasked) patches
  └──────┬──────┘
         │  reconstruction quality  (0 = perfect, 1 = poor)
         ▼
  reward = 1 − reconstruction_loss   (higher = better selection)
```

The model learns: **select patches that maximally preserve scene information**.

### 3.3 Task Loss Requirement

An optional conditioning scalar `r ∈ [0, 1]` tells the model to **stop generating once reconstruction quality r is achieved**.  This gives inference-time control over the speed/quality tradeoff without re-training:

```text
  task_loss_requirement = 0.7  →  stop when 70% reconstruction quality reached
                                   (fewer patches selected, faster ViT)

  task_loss_requirement = 0.9  →  stop only when 90% quality reached
                                   (more patches, higher accuracy)
```

---

## 4. Inference Walkthrough

### 4.1 End-to-End Flow

```text
  ① Input video frames
     (B, T, 3, H, W)  e.g. (1, 16, 3, 480, 854)
           │
           ▼
  ② Resize to model input size
     F.interpolate → (B, T, 3, 224, 224)
           │
           ▼
  ③ ShallowVideoConvNet
     3D conv patchification + trunk blocks
     → (B, T', 196, 192)     T' = T / temporal_patch_size
           │
           ▼
  ④ Connector
     + learned pos_embed (196, 192)
     → (B, T', 196, 192)      shape unchanged
           │
           ▼
  ⑤ AR Gaze Decoding  (frame by frame — see §4.2)
     for each frame t ∈ 0..T'-1:
       append vision features to decoder context
       decoder generates patch index sequence
     → gaze_pos (B, T', k)   k ≤ max_gaze_tokens
           │
           ▼
  ⑥ Build binary gaze mask
     shape: (B, T', 196)
     mask[b, t, gaze_pos[b, t, :]] = 1
           │
           ▼
  ⑦ Apply mask to downstream ViT / MLLM
     (hook / full / native integration — see integration_guide.md)
```

### 4.2 Step-by-Step: Single Frame Encoding (Step ③)

```text
  Frame t: raw RGB  (3, 224, 224)
           │
  temporal_conv ────▶ reshape  ──▶ (196, 192)  ← 196 patch embeddings
     stride 16×16                               ← each covers 16×16 px
           │
  LayerNorm + trunk blocks
           │
  out_proj (1×1 conv)
           │
  Output: 196 feature vectors, one per 16×16 patch location
```

Visual mapping of patch IDs to spatial regions:

```text
  224 × 224 frame, 14 × 14 patch grid (16 px/patch):

  cols: 0    16   32   48  …  208  224
        │    │    │    │       │    │
  rows: ┌────┬────┬────┬──── … ─────┐
   0    │  0 │  1 │  2 │  …   │ 13 │
   16   ├────┼────┼────┼──── … ─────┤
        │ 14 │ 15 │ 16 │  …   │ 27 │
   32   ├────┼────┼────┼──── … ─────┤
        │ …                          │
  208   ├────┬────┬────┬──── … ─────┤
        │182 │183 │184 │  …   │195 │
  224   └────┴────┴────┴──── … ─────┘
```

---

## 5. Autoregressive Decoding Process

This section shows exactly how the gaze decoder generates patch selections for a video, one frame at a time.

### 5.1 Context Layout

The decoder maintains a growing sequence of **interleaved vision and gaze tokens**:

```text
  Time ──────────────────────────────────────────────────────────────▶

  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
  │ frame 0       │ │ frame 0 gaze  │ │ frame 1       │ │ frame 1 gaze  │  …
  │ v₀  v₁ … v₁₉₅│ │ p₄₂ p₁₃₇ p₈ │ │ v₀  v₁ … v₁₉₅│ │ p₁₂ p₉₁ …   │
  │ (196 vectors) │ │ (k₀ indices) │ │ (196 vectors) │ │ (k₁ indices) │
  └───────────────┘ └───────────────┘ └───────────────┘ └───────────────┘
        ▲                   ▲                 ▲                  ▲
    continuous           discrete          continuous         discrete
    features          patch ID embs         features         patch ID embs
```

KV cache from past frames is **reused** — the decoder knows prior frame selections when deciding the current frame.

### 5.2 Single-Frame Generation (Step-by-Step)

Suppose `max_gaze_tokens = 3` and we are selecting patches for frame 1.

```text
  Context before frame 1:
  [v₀(0) … v₀(195) | p₄₂ p₁₃₇ p₈ | v₁(0) … v₁(195)]
                                                        ▲ cursor here

  ── Step 1 ──────────────────────────────────────────────────────
  Decoder forward → logits over {0..196}
  NoRepeat: no tokens already in history can be selected
  NoEOS:    score[196] = −∞  (can't stop yet)
  argmax  → patch 91  ✓
  Context: [… v₁(0) … v₁(195) | 91]

  ── Step 2 ──────────────────────────────────────────────────────
  Decoder forward → logits over {0..196}
  NoRepeat: score[91] = −∞  (just selected)
  NoEOS:    score[196] = −∞
  argmax  → patch 12  ✓
  Context: [… v₁(0) … v₁(195) | 91  12]

  ── Step 3 ──────────────────────────────────────────────────────
  argmax  → patch 174  ✓
  Context: [… v₁(0) … v₁(195) | 91  12  174]

  ── max_gaze_tokens reached → stop ─────────────────────────────
  gaze_pos[1] = [91, 12, 174]
```

**Result for frame 1**: only 3 of 196 patches selected.  The gaze mask for frame 1:

```text
  mask[1, :] = all zeros except positions 91, 12, 174:

  ┌─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┐
  │·│·│·│·│·│·│·│·│·│·│·│·│▪│·│  row 0  (patch 12 = row 0, col 12)
  ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
  │·│·│·│·│·│·│·│·│·│·│·│·│·│·│
  ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
  ·  ·  ·  ·  ·  ·  ▪  ·  ·  ·      row 6  (patch 91 = row 6, col 7)
  ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·
  ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
  ·  ·  ·  ·  ·  ·  ·  ·  ▪  ·      row 12 (patch 174 = row 12, col 6)
  └─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┘
```

### 5.3 Multi-Token Prediction in Detail

With `num_multi_token_pred = K = 2`, the decoder predicts two tokens per step:

```text
  ── Step 1 (K=2) ────────────────────────────────────────────────
  Decoder produces 2 × vocab_size logits:
    head₀ logits → patch 91
    head₁ logits → patch 12   (predicted as "2nd after 91")
  Both appended atomically: context grows by 2

  ── Step 2 (K=2) ────────────────────────────────────────────────
    head₀ logits → patch 174
    head₁ logits → patch 55
  Context grows by 2 more

  Total: 4 patches in 2 forward passes   (vs 4 passes for K=1)
```

The K head predictions are trained with **shifted supervision**: head₀ predicts `t+1`, head₁ predicts `t+2`, etc.  At inference the K tokens are accepted in order.

### 5.4 KV Cache and Streaming

The decoder uses standard KV cache so each step only recomputes the new token's Q·K·V, not the whole history.  Additionally, the `ShallowVideoConvNet` supports **streaming mode** via `past_conv_values`:

```text
  Frame-by-frame streaming:

  Frame 0 ──▶ conv trunk ──▶ features₀ + past₀  ──▶ gaze₀
  Frame 1 ──▶ conv trunk (reuse past₀) ──▶ features₁ + past₁  ──▶ gaze₁
  Frame 2 ──▶ conv trunk (reuse past₁) ──▶ features₂ + past₂  ──▶ gaze₂
  …

  past_conv_values = last (temporal_kernel_size − 1) temporal slices
  No re-computation of past frames needed.
```

---

## 6. Configuration Reference

Key configuration parameters (from `AutoGazeConfig` / `GazeModelConfig`):

| Parameter | Default | Description |
| :--- | :---: | :--- |
| `input_img_size` | 224 | Frame resolution fed to vision encoder |
| `num_vision_tokens_each_frame` | 196 | Patches per frame (14×14) |
| `vision_model_config.kernel_size` | 16 | Spatial stride = patch size in pixels |
| `vision_model_config.temporal_patch_size` | 1 | Frames grouped into one token chunk |
| `vision_model_config.hidden_dim` | 192 | Vision encoder channel width |
| `vision_model_config.depth` | 1 | Number of Conv3dBlock trunk layers |
| `gaze_decoder_config.num_multi_token_pred` | 1 | Parallel token prediction heads |
| `gaze_decoder_config.hidden_size` | 4096 | LLaMA hidden dimension |
| `gaze_decoder_config.num_hidden_layers` | 32 | LLaMA transformer depth |
| `gazing_ratio_config.fixed.gazing_ratio` | 0.5 | Default fraction of patches to keep |

---

## 7. Key Source Files

| Purpose | Path |
| :--- | :--- |
| Top-level model class (`AutoGaze`) | `autogaze/models/autogaze/autogaze.py` |
| Core model (`AutoGazeModel`, encoders) | `autogaze/models/autogaze/modeling_autogaze.py` |
| Configuration classes | `autogaze/models/autogaze/configuration_autogaze.py` |
| LLaMA multi-token prediction decoder | `autogaze/models/autogaze/modeling_llama_multi_token_pred.py` |
| Image processor | `autogaze/models/autogaze/processing_autogaze.py` |
| Integration modes guide | `docs/integration_guide.md` |
| Eval / benchmark guide | `docs/eval_guide.md` |
| Korean comprehensive guide | `docs/guide_ko.md` |
