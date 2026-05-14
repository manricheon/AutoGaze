# Generic ViT AutoGaze Compatibility

This document analyzes whether AutoGaze outputs can be applied to non-SigLIP ViT plus MLLM systems without training or trainable adapters. It uses the original `INTEGRATION.md` and `QUICK_START.md` behavior as the compatibility baseline.

Short answer: AutoGaze outputs are not automatically portable to arbitrary ViT/MLLM stacks. They are portable only when selected patch IDs can be mapped to the target patch grid, the target model can receive deterministic spatial/temporal positions for sparse patches, and the downstream projector plus visual placeholders can accept the resulting token count. When those conditions are not true or not known, use AutoGaze only for input-level frame/window/chop selection or clearly labeled post-compute token reduction.

## AutoGaze Integration Mode Taxonomy

| Mode | Meaning | Encoder compute reduction possible? | MLLM token reduction possible? | Positional remapping required? | Training required? | Safe for zero-shot probing? |
|---|---|---:|---:|---:|---:|---:|
| `native_sparse_patch` | Physically process only AutoGaze-selected patches/tokens through a natively compatible vision path. | Yes | Yes, if projector/placeholders accept variable tokens | Yes | No | Only after full architecture verification |
| `light_modified_sparse` | Small deterministic code changes let an existing encoder receive sparse selected patches. | Yes | Yes, if connector and placeholders are dynamic | Yes | No | Medium risk; requires source-level checks |
| `autogaze_zero_mask` | Keep dense image/video grid, zero-mask or mean-mask unselected patch regions before encoding. | No | No, unless downstream also prunes | No sparse remap; dense positions preserved | No | Yes, useful as robustness/performance probe |
| `post_encoder_zero_mask` | Run the full encoder, then zero out unselected visual tokens. | No | Maybe, if masked/pruned before MLLM | Usually no sparse remap, but token alignment still matters | No | Yes, but not encoder acceleration |
| `input_selection_only` | AutoGaze selects frames, windows, crops, or chops; external model uses official processor. | Sometimes, by reducing input count/area | Model-dependent | No sparse token remap | No | Yes |
| `unsupported_for_now` | Position, dense-grid, connector, or placeholder behavior is unknown or incompatible. | No claim | No claim | Unknown or blocked | Not allowed as hidden adapter work | No |

`native_sparse_patch` and `light_modified_sparse` are the only modes in this taxonomy that can claim selected-token ViT encoder acceleration. `autogaze_zero_mask` and `post_encoder_zero_mask` are useful ablations because they preserve the model's dense positional layout, but they must not be reported as encoder-side acceleration. `input_selection_only` can still reduce total runtime when fewer frames or crops are processed, but it is not the original AutoGaze sparse patch integration.

## What AutoGaze Provides

The core `gazing_info` passed to the AutoGaze-compatible SigLIP implementation contains:

| Output | Meaning | Compatibility impact |
|---|---|---|
| `gazing_pos` | Flattened selected patch positions across all frames, shape `(B, N)` | The target encoder must be able to map these positions to its own patch grid. |
| `if_padded_gazing` | Boolean mask for padded/dummy gazes, shape `(B, N)` | Padded positions must not be treated as real visual evidence. Attention masks must hide them. |
| `num_gazing_each_frame` | Selected positions per frame, including padding | Needed to build temporal attention masks and per-frame token accounting. |
| selected token count | Non-padded selected visual tokens | Downstream projector and MLLM placeholder count must match this count. |
| selected patches per frame | Distribution of selected patches over frames | Required for frame-local attention and temporal position assignment. |
| multi-scale patch records | Selected patches from multiple scales such as `32+64+112+224` | The target model must know which scale each token came from and how that scale maps to target resolution. |
| `target_scales` / `target_patch_size` | Optional retargeting for another ViT patch size/resolution | Useful only if the target ViT can accept the same deterministic grid mapping. |

AutoGaze patch IDs are not local to each frame in static video mode. They are flattened across the full video token sequence. `QUICK_START.md` also documents that AutoGaze can retarget to other patch sizes/resolutions by passing `target_scales` and `target_patch_size`, but that still assumes the target encoder can consume the corresponding sparse patch/position sequence.

## Application Levels

| Level | What AutoGaze controls | Encoder-side acceleration? | Token reduction after computation? | Compatibility-only? | Requirements |
|---|---|---:|---:|---:|---|
| input-level frame selection | Which frames/windows enter the external model | Sometimes, if fewer frames are decoded/encoded | No | No | Safe official processor path for selected frames. |
| input-level crop/chop selection | Which crops/regions enter the external model | Sometimes, if fewer/lower-cost crops are encoded | No | No | Deterministic crop/chop mapping; official image/video path. |
| sparse patch embedding before ViT | Which patches are patch-embedded and encoded | Yes | No | No | Patch grid, positions, attention masks, projector, placeholders all verified. |
| post-patch-embedding token masking | Tokens are embedded, then masked before/deep in ViT | Usually no for patch embedding; maybe partial for later layers | Yes | Sometimes | Must not claim full encoder-side acceleration. |
| post-encoder token pruning | Full vision encoder runs, then tokens are pruned before MLLM | No | Yes | No | Projector/LLM can consume pruned variable tokens. |
| direct visual token injection into MLLM | External visual embeddings are inserted into LLM stream | No encoder acceleration unless paired with sparse encoder | Maybe | High | Projector output dimension, placeholders, masks, and position IDs exactly match. |

Only sparse patch embedding before the ViT is true AutoGaze encoder-side acceleration in the original sense. Frame/chop selection can also reduce total work, but it is input selection, not selected-token ViT acceleration. Post-encoder pruning and direct injection are downstream token-budget strategies unless the upstream encoder computation is actually reduced.

## Positional Encoding Compatibility

A sparse sequence with holes is valid only when the model can receive explicit spatial/temporal positions or can deterministically gather the same position embeddings that dense tokens would have used. Dense-grid operations break this assumption unless they can be reconstructed exactly.

| Positional / token scheme | Sparse selected patches without training? | Missing patches / holes valid? | Deterministic position IDs? | Dense grid required? | Window/fixed-count risk | Projector / placeholder risk | Verdict |
|---|---:|---:|---:|---:|---|---|---|
| absolute learned 2D position embedding | Possible | Yes, if gather uses original patch indices | Yes, gather/interpolate learned positions | Usually no | Low unless downstream pooling assumes grid | Variable token support still required | Candidate when patch grid/projector/placeholders are open. |
| interpolated absolute position embedding | Possible | Yes, if interpolation happens before gather | Yes | Usually no | Low to medium | Same as above | Candidate if interpolation math is deterministic. |
| 2D RoPE | Possible only with explicit x/y IDs | Yes, if x/y IDs for holes are passed | Yes if model API accepts positions | Usually no | Window attention may reintroduce dense needs | Projector/placeholders still required | `rope_sparse_candidate` only after code inspection. |
| 3D / temporal RoPE | Possible only with explicit t/x/y IDs | Yes, if time IDs and spatial IDs are explicit | Yes if API accepts positions | Usually no | Temporal chunking/windowing can block | Projector/placeholders still required | Candidate only if t/x/y IDs are externally controllable. |
| M-RoPE / multi-resolution RoPE | High risk | Unknown | Only if official grid metadata can be regenerated for sparse holes | Often yes through grid metadata | High: dynamic resolution/window attention | High: visual merger and placeholders are processor-owned | Blocked until position, window, merger, and placeholders are solved. |
| relative position bias | Usually blocked | Holes alter relative distances unless handled | Maybe | Often yes | Medium to high | Projector/placeholders still required | Needs code inspection; sparse may change attention semantics. |
| window attention | Usually blocked | Holes break local windows unless windows are reconstructed | Maybe | Yes | High: windows expect dense local neighborhoods | Projector/placeholders still required | Blocked unless sparse windows are implemented exactly. |
| dense-grid patch merging | Blocked | Holes invalidate merge neighborhoods | No, unless dense grid is filled/masked | Yes | High | Token count after merge changes | Not a direct sparse candidate. |
| visual token merger / pixel shuffle | Blocked | Holes alter structured merge/pixel shuffle | Usually no | Yes | High | Very high | Use input selection only. |
| Perceiver / Q-Former / resampler | Usually not encoder-side sparse | Input holes may be tolerated after full encoding | Positions depend on upstream encoder | Often consumes dense encoder output | Fixed latent count | High: fixed connector output | Post-encoder or input-selection only unless connector accepts sparse positions natively. |

## Direct Sparse Integration Criteria

All conditions must be true before marking a model as a direct sparse AutoGaze candidate:

1. Patch grid is accessible.
2. AutoGaze selected multi-scale patch indices can be mapped to the model patch grid.
3. Patch embedding can run on selected patches or multi-scale patches.
4. Positional embeddings or position IDs can be adapted deterministically.
5. Attention does not require dense-grid tokens, or dense requirements can be reconstructed exactly.
6. Projector / visual connector accepts variable token counts.
7. LLM visual placeholder count can be adjusted to match visual token count.
8. Attention masks and position IDs can be generated correctly.
9. No training or trainable adapter is required.

If any condition is false or unknown, mark direct sparse integration as blocked or `needs_code_inspection`.

## Non-Training Rule

Allowed:

- deterministic patch grid mapping;
- deterministic crop/chop mapping;
- positional embedding interpolation;
- explicit position ID generation;
- official processor path;
- frozen projector path;
- variable visual token count if the original model already supports it.

Not allowed:

- randomly initialized Linear projection to match dimensions;
- newly trainable projector;
- trainable positional adapter;
- hidden fine-tuning requirement;
- silently changing feature dimensions.

If dimensions do not match, the path is blocked unless the original model already provides a compatible frozen projector or connector.

## Model-Level Implications

| Model | Direct sparse status | Why |
|---|---|---|
| LongVILA-R1-7B | `needs_code_inspection` / `native_candidate` | VILA/SigLIP lineage is promising, but TSP/video pooling, projector, and placeholders must be verified. |
| LLaVA-OV | `needs_code_inspection` / `siglip_candidate` | SigLIP candidate, but anyres packing and video pooling can require dense/packed grids. |
| VideoLLaMA3-7B | `needs_code_inspection` / `siglip_candidate` | SigLIP-NaViT candidate, but NaViT packing, spatial merge, and compression must be inspected. |
| LongVA-7B | `input_selection_only` for checked checkpoint | Checked config uses CLIP ViT, not SigLIP; direct sparse AutoGaze mapping is not established. |
| Apollo-7B | `post_encoder_only` / input selection | Hybrid SigLIP plus InternVideo2 and Perceiver connector hide simple sparse patch mapping. |
| Qwen2.5-VL-7B | `input_selection_only` | M-RoPE, `grid_thw`, visual merger, and window attention make sparse holes unsafe by default. |
| InternVL3.5-8B | `input_selection_only` | Dynamic tiling and pixel shuffle depend on dense tile layout. |
| VideoChat-Flash | `input_selection_only` | UMT hierarchy and strong token compression do not expose a simple AutoGaze patch grid. |

## V-JEPA2 Implications

The local `weights/vjepa2-vitl-fpc64-256` metadata describes a video encoder with `crop_size=256`, `patch_size=16`, `frames_per_clip=64`, `tubelet_size=2`, and `hidden_size=1024`. That creates a nominal dense tubelet grid of `32 x 16 x 16` before any model-specific packing. AutoGaze selected 2D patch indices can be mapped to dense image-space masks or frame/chop decisions, but sparse tubelet execution is not verified from the local config alone.

Recommended V-JEPA2 modes:

| Mode | Compatibility status | Compute claim |
|---|---|---|
| `vjepa2_official_dense` | feasible baseline with official dense input | Dense baseline only |
| `autogaze_frame_selection_vjepa2` | feasible input-level selection | May reduce work by reducing selected clips/frames |
| `autogaze_chop_selection_vjepa2` | feasible input-level crop/chop selection | May reduce work by reducing selected crops/chops |
| `autogaze_zero_mask_vjepa2` | feasible zero-shot probing with dense positions preserved | No encoder-side sparse acceleration |
| `vjepa2_context_mask_probe` | `needs_source_inspection` | No compute-reduction claim until context-mask semantics are verified |
| `vjepa2_sparse_tubelet` | blocked/experimental | No claim until selected tubelets can carry correct 3D-RoPE positions through encoder attention |
| `vjepa2_to_mllm_projector` | blocked without a compatible frozen projector | Training would be required for arbitrary MLLM projector replacement |

V-JEPA2 is a strong candidate for AutoGaze-guided frame/chop selection and zero-mask probing. It is not a drop-in replacement for SigLIP in NVILA, LongVA, LLaVA-OV, Qwen2.5-VL, InternVL, or VideoChat projectors unless a compatible frozen connector is already provided and verified.

## Practical Recommendation

Use three tiers:

1. Safe now: `official_processor`, `autogaze_frame_selection`, `autogaze_chop_selection`.
2. Research/code-inspection only: `siglip_sparse_patch`, `rope_sparse_patch`, `post_encoder_pruning`.
3. Disabled by default: `direct_visual_token_injection`.

The next direct sparse implementation target should be LongVILA-R1 or LLaVA-OneVision because both are closest to SigLIP/VILA-style pipelines. Qwen2.5-VL, InternVL3.5, and VideoChat-Flash should remain input-selection-only until their positional/window/merger/placeholder contracts are explicitly solved.

## Evidence Levels for Asset Verification

External model readiness is tracked independently from compatibility theory:

| Evidence level | Meaning | What it does not prove |
|---|---|---|
| Theoretical compatibility | The architecture appears compatible or incompatible from docs/model-family knowledge. | It does not prove checkpoint availability or runnable code. |
| Config-inspected compatibility | A local `config.json` or processor config was parsed without loading weights. | It does not prove projector, placeholder, or processor runtime behavior unless the config explicitly exposes those contracts. |
| Local asset verified | Required config, processor/tokenizer, and weight files exist locally and are not obviously partial. | It does not prove real model loading or inference. |
| Real model loaded | The requested adapter loaded the explicit local/remote checkpoint. | It does not prove generation quality or AutoGaze sparse compatibility. |
| Real inference passed | A minimal smoke produced an answer or feature summary. | It is not a benchmark or paper reproduction. |

`configs/poc_inference/model_asset_manifest.yaml` and the asset scripts keep these statuses separate. A downloaded or locally verified model can still be `input_selection_only` or blocked for direct sparse integration if its positional encoding, dense-grid operations, projector, or visual placeholders are not compatible with AutoGaze selected patch/tubelet indices.

## RoPE Sparse Integration Checklist

RoPE-based ViTs are not automatically blocked, but `rope_sparse_patch` is allowed only after all required controls are verified:

| Requirement | Why it matters | Failure status |
|---|---|---|
| Patch/tubelet grid accessible | AutoGaze indices must map to the target model's real visual grid. | `needs_source_inspection` if unknown |
| AutoGaze index -> `(t,h,w,scale)` mapper | Sparse RoPE needs explicit temporal and spatial coordinates, not just token order. | `blocked_architecture` if no coordinate path exists |
| Position ID builder | 2D/3D/M-RoPE position IDs must be generated deterministically for holes. | `needs_source_inspection` if hidden in processor |
| Attention accepts sparse order | Arbitrary sparse token order must be accepted, or dense windows must be reconstructed exactly. | `blocked_architecture` if dense windows are mandatory |
| No dense reshape/merge before injection | Pixel shuffle, visual merger, patch merging, and window packing can require dense grids. | `blocked_architecture` |
| Variable projector token count | Sparse encoder output must match frozen connector assumptions. | `blocked_requires_training` if a new connector is needed |
| Placeholder alignment | LLM visual placeholders must equal selected visual token count. | `blocked_architecture` if processor owns fixed placeholders |
| Attention mask builder | Visual and LLM masks must match sparse position IDs and token count. | `needs_source_inspection` if not exposed |

The deterministic token order for a RoPE sparse attempt is frame/time, then scale, then row, then column. When `--integration-mode rope_sparse_patch` is explicitly requested, the PoC writes `outputs/<run>/autogaze/rope_sparse_mapping_metadata.json` with generated coordinates, position IDs, token order, attention-mask shape, visual token count, placeholder count, dense-grid/window flags, and support status. This metadata is a validation artifact; it does not imply support unless the status is `implemented`.

Current RoPE-like conclusions:

| Model | RoPE / position type | `rope_sparse_patch` status | Fallback modes |
|---|---|---|---|
| Qwen2.5-VL | M-RoPE with `grid_thw`, temporal patching, window attention, visual merger | `blocked_architecture` | official processor, `autogaze_frame_selection`, `autogaze_chop_selection`, `autogaze_zero_mask`, post-encoder probes |
| VideoLLaMA3 | SigLIP-NaViT plus Qwen2.5 RoPE after visual compression | `needs_source_inspection` | official processor, frame/chop selection, zero-mask |
| V-JEPA2 | 3D-RoPE over dense tubelet grid | `needs_source_inspection`; sparse tubelet API not verified | dense feature extraction, frame selection, zero-mask |
| LongVILA / LLaVA variants | LLM RoPE plus SigLIP-style vision positions | not a vision RoPE sparse path by default | official processor, zero-mask, input selection |

## Zero-Mask Fallback Requirements

`autogaze_zero_mask` preserves the dense positional layout and masks unselected visual content. It is a compatibility and robustness probe, not encoder-side acceleration.

| Stage | Behavior | Status |
|---|---|---|
| `pixel` | Builds a dense image-space mask from AutoGaze selected patch boxes and applies zero/mean fill before the official processor or dense path. | implemented for processed tensors |
| `patch_embedding` | Mask dense tokens after patch embedding while keeping token count fixed. | prepared as metadata-only until model hooks are verified |
| `post_encoder` | Mask dense visual tokens after full encoder. | prepared as metadata-only until encoder outputs are mapped |

Required metrics are `integration_mode=autogaze_zero_mask`, `zero_mask_stage`, `zero_mask_value`, dense token count processed, selected token count, selected/masked ratios, `zero_mask_encoder_compute_reduction=false`, and `zero_mask_expected_speedup=none`.
