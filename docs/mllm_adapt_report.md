# AutoGaze External MLLM Adaptation Review

This review classifies external MLLMs for no-training AutoGaze integration. The compatibility standard is the original AutoGaze `INTEGRATION.md` and `QUICK_START.md` path: AutoGaze predicts multi-scale patch indices, a compatible vision encoder maps those indices onto its patch grid, positional handling remains deterministic, and the frozen projector/LLM placeholder sequence accepts the resulting visual token count.

No row below permits randomly initialized Linear adapters, newly trained projectors, trainable positional adapters, or hidden fine-tuning. If a dimension, positional, projector, or placeholder mismatch exists, direct selected-token injection is blocked until the original model already exposes a compatible path.

## Source References

Local references:

- `INTEGRATION.md`
- `QUICK_START.md`
- `docs/nvila-hd-video-readme.md`
- `docs/inference_guide_for_poc.md`
- `scripts/infer_full.py`
- `scripts/poc_model_adapters.py`
- `scripts/poc_model_registry.py`

External model references checked:

- LLaVA-OneVision Transformers docs and HF config: https://huggingface.co/docs/transformers/main/model_doc/llava_onevision and https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-hf
- LongVA-7B HF config: https://huggingface.co/lmms-lab/LongVA-7B
- LongVILA-R1-7B HF model card/config: https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B
- Apollo-7B HF config: https://huggingface.co/GoodiesHere/Apollo-LMMs-Apollo-7B-t32
- VideoLLaMA3-7B HF model card and VL3-SigLIP-NaViT config: https://huggingface.co/DAMO-NLP-SG/VideoLLaMA3-7B and https://huggingface.co/DAMO-NLP-SG/VL3-SigLIP-NaViT
- VideoChat-Flash HF model card/config: https://huggingface.co/OpenGVLab/VideoChat-Flash-Qwen2-7B_res448
- InternVL3.5-8B HF model card/config: https://huggingface.co/OpenGVLab/InternVL3_5-8B
- Qwen2.5-VL-7B HF model card/config and Transformers docs: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct and https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/qwen2_5_vl.md

Reference map:

| Model | Official repo / model card / paper reference recorded |
|---|---|
| LLaVA-OV / LLaVA-OneVision 8B | HF Transformers docs and `llava-hf/llava-onevision-qwen2-7b-ov-hf`; paper linked from HF as arXiv 2408.03326 |
| LongVA-7B | `lmms-lab/LongVA-7B`; paper linked from HF as arXiv 2406.16852 |
| LongVILA-R1-7B | `Efficient-Large-Model/LongVILA-R1-7B`; code `NVLabs/Long-RL`; paper linked from HF as arXiv 2507.07966 |
| Apollo-7B | `GoodiesHere/Apollo-LMMs-Apollo-7B-t32`; paper linked from HF as arXiv 2412.10360 |
| VideoLLaMA3-7B | `DAMO-NLP-SG/VideoLLaMA3-7B` and `DAMO-NLP-SG/VL3-SigLIP-NaViT`; paper linked from HF as arXiv 2501.13106 |
| VideoChat-Flash | `OpenGVLab/VideoChat-Flash-Qwen2-7B_res448`; paper linked from HF as arXiv 2501.00574 |
| InternVL3.5-8B | `OpenGVLab/InternVL3_5-8B`; InternVL3.5 paper/docs linked from HF |
| Qwen2.5-VL-7B | `Qwen/Qwen2.5-VL-7B-Instruct`, Qwen blog, and HF Transformers Qwen2.5-VL docs; technical report linked as arXiv 2502.13923 |

## Section A: AutoGaze Integration Feasibility for Table-Model MLLMs

This section covers the video MLLMs from the AutoGaze paper comparison table and classifies how AutoGaze can be used without training or trainable adapters.

### Integration Mode Taxonomy

| Mode | Definition | Encoder compute reduction possible? | MLLM token reduction possible? | Positional remapping needed? | Training required? | Zero-shot probing safety |
|---|---|---:|---:|---:|---:|---|
| `native_sparse_patch` | The model natively processes only AutoGaze-selected patches/tokens. | Yes | Yes, if connector/placeholders accept variable tokens | Yes | No | Safe only after source verification |
| `light_modified_sparse` | Deterministic model-code changes gather selected patches and positions before the encoder. | Yes | Yes, if downstream supports it | Yes | No | Medium risk; source-level checks required |
| `autogaze_zero_mask` | Dense input is preserved and unselected image/patch regions are zero- or mean-masked. | No | No by itself | No sparse remap; dense positions remain | No | Safe for probing |
| `post_encoder_zero_mask` | Full encoder runs, then unselected visual tokens are zeroed or pruned. | No | Maybe | Token alignment still required | No | Safe if labeled non-acceleration |
| `input_selection_only` | AutoGaze selects frames, windows, crops, or chops before the official processor. | Sometimes, by reducing inputs | Model-dependent | No sparse remap | No | Safest |
| `unsupported_for_now` | Position, dense-grid, connector, or placeholder requirements are incompatible or unknown. | No claim | No claim | Unknown/blocked | Not allowed as hidden adapter work | Not safe |

Only `native_sparse_patch` and `light_modified_sparse` can claim selected-token encoder acceleration. `autogaze_zero_mask` and `post_encoder_zero_mask` are ablation/probing modes. `input_selection_only` can reduce practical runtime through fewer frames or regions, but it is not the original AutoGaze sparse patch path.

### Integration Criteria

A model is a direct/sparse AutoGaze candidate only when all items are true:

- Vision encoder is SigLIP or SigLIP-compatible.
- Patch grid is accessible.
- AutoGaze selected multi-scale patch indices can be mapped to that model patch grid.
- Positional embeddings can be adapted without training by interpolation or deterministic remapping.
- Patch embedding can run on selected or multi-scale patches.
- Frozen projector accepts variable visual token counts, or the correct dynamic count can be passed.
- LLM visual placeholder count can be adjusted to exactly match visual token count.
- Attention masks and position IDs can be generated correctly.
- No random or trainable adapter is required.

If any item is false or unknown, direct integration is blocked or marked `needs_code_inspection`.

### Positional Encoding Implication From `INTEGRATION.md`

The original AutoGaze integration changes the patch embedding and attention-mask path, not the trained transformer blocks. The selected `gazing_pos` indices are used to gather patches and their matching position embeddings before the vision encoder, then an inter-frame attention mask is constructed from `num_gazing_each_frame` and `if_padded_gazing`. For external MLLMs this means sparse/direct integration is safe only when the external model exposes the same three control points:

- deterministic mapping from AutoGaze multi-scale patch IDs to the external patch grid;
- deterministic position embedding interpolation or remapping for the selected patches;
- correct downstream visual token count, attention mask, position ID, and placeholder alignment.

Models with M-RoPE, dynamic tiling, NaViT packing, video token pooling, hierarchical compression, pixel shuffle, or fixed-token resamplers are therefore blocked for direct selected-token injection until those model-specific positional contracts are reproduced without training.

### Model-By-Model Summary

| Model | Status | Risk | Encoder type | LLM type | Vision-token pipeline | Positional / ID behavior | INTEGRATION.md-style method | Direct selected-token injection | Safer input-level selection | Main blockers | Required code changes | Training / trainable adapter needed | Final recommendation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LLaVA-OV / LLaVA-OneVision 8B | `siglip_candidate` | medium | SigLIP vision encoder, patch14, anyres image path and pooled video path | Qwen2 | Official processor creates image/video tokens; video is pooled to a fixed per-frame token sequence in the Transformers path | SigLIP 2D image grid plus LLM RoPE; anyres/newline and video pooling must stay aligned | Possible in principle for SigLIP sparse patch path, but not verified | Blocked until pooled video token count, placeholder expansion, and position IDs are verified | Yes | Anyres packing, video pooling to 196 tokens/frame, image/video token placeholder counts | Official processor adapter first; then inspect `LlavaOnevisionForConditionalGeneration` pack/pool/projector path | No for input selection; direct path blocked, not solved by trainable adapter | Use AutoGaze frame/chop selection now; inspect sparse SigLIP path later |
| LongVA-7B | `input_selection_only` | medium | CLIP ViT-L/14-336 anyres/unires, not SigLIP in the checked HF config | Qwen2 | LLaVA-style anyres/unires visual tokens through CLIP tower and MLP projector | CLIP absolute positional behavior and LongVA long-context Qwen2 RoPE | No for current 7B checkpoint because the checked tower is CLIP, not SigLIP | Blocked | Yes | Not SigLIP; unires pooling/projector contract is model-specific | Official/chat adapter plus AutoGaze-selected frame/window/chop inputs | No for input selection; direct sparse path would require a different compatible tower or unsupported adapter work | Keep as input-selection-only unless a SigLIP LongVA variant is inspected |
| LongVILA-R1-7B | `native_candidate` | medium | VILA SigLIP SO400M patch14-448 plus TSP video encoder | Qwen2 | VILA processor embeds media, then projector/LLM consumes prompt embeddings; supports long video frame counts | SigLIP position interpolation plus TSP video pooling and Qwen2 RoPE | Closest external candidate because it is VILA/SigLIP family | Unknown; blocked until `_embed`, projector, TSP pooling, and placeholders are verified | Yes | Need to map AutoGaze multi-scale patches to VILA SigLIP grid while preserving TSP pooling and prompt embeds | Official processor smoke; code-inspect `siglip_encoder.py`, `media_encoder.py`, `base_projector.py`, `_embed` | No if native sparse path can be reused; otherwise blocked rather than train a new adapter | Highest-priority direct/sparse investigation after a real official-processor smoke test |
| Apollo-7B | `post_encoder_only` | high | Hybrid vision tower: SigLIP SO400M plus InternVideo2 with Perceiver connector | Qwen2 | Hybrid tower produces tokens that a Perceiver-style connector resamples to fixed output tokens | Hybrid spatial/temporal position handling plus connector latent positions | Partial only if the SigLIP branch can be isolated without breaking hybrid fusion | Blocked | Yes | Dual encoder, InternVideo2 temporal tower, Perceiver connector fixed tokens | Official processor adapter; optional post-encoder pruning label only after code inspection | No for input selection; direct sparse path blocked by hybrid connector | Use input-level chop/region selection; do not claim encoder-side acceleration |
| VideoLLaMA3-7B | `siglip_candidate` | high | VL3-SigLIP-NaViT patch14 tuned vision encoder | Qwen2.5 | Official processor returns video tensors for a custom VideoLLaMA3 Qwen2 path; configs expose spatial merge/compression controls | NaViT layout, spatial merge, token compression, and Qwen2.5 RoPE must align | Possible only after code inspection of NaViT positional handling and compression | Unknown; blocked | Yes | NaViT variable layout, `spatial_merge_size`, token compression, placeholder count | Official processor adapter first; inspect visual encoder, projector, and compression modules | No for input selection; direct path blocked until native variable-token support is proven | Candidate for later sparse work; start with input selection |
| VideoChat-Flash | `input_selection_only` | high | UMT-L or UMT-HD-L video tower with hierarchical compression | Qwen2 or Qwen2.5 depending checkpoint | Model compresses video to a small fixed token count per frame and supports very long frame sequences | UMT hierarchy plus Qwen/Yarn long-context position handling | No; not SigLIP and no AutoGaze patch-grid compatibility | Blocked | Yes | UMT token hierarchy, hierarchical compression, fixed per-frame token interface | Official chat adapter using selected frames/windows only | No for input selection; direct sparse path unsupported | Use AutoGaze to select frames/windows/chops before official chat |
| InternVL3.5-8B | `input_selection_only` | high | InternViT with dynamic tiling and pixel-shuffle compression | Qwen3-8B | Dynamic high-resolution tiling creates patch groups; pixel shuffle compresses visual tokens before LLM | Dynamic tile layout and Qwen3 RoPE; Flash variants add Visual Resolution Router | No; not SigLIP and token layout is tile/pixel-shuffle specific | Blocked | Yes | Dynamic tile counts, thumbnail handling, pixel shuffle, image placeholder expansion | Official chat adapter; AutoGaze can guide tile/chop choice only | No for input selection; direct path blocked | Use AutoGaze as high-res crop/tile selector only |
| Qwen2.5-VL-7B | `input_selection_only` | high | Native dynamic-resolution Qwen2.5-VL ViT with window attention and visual merger | Qwen2.5 | Official processor emits `pixel_values_videos`, `video_grid_thw`, placeholders, and M-RoPE-aware position inputs | M-RoPE with dynamic spatial/temporal IDs; window attention and temporal patching | No for direct sparse AutoGaze in current PoC | Blocked by default | Yes | M-RoPE, `video_grid_thw`, visual merger, attention masks, placeholder counts | Keep official processor path; optional post-patch masking must be labeled non-encoder-side acceleration | No for input selection; direct selected-token injection blocked | Keep Qwen as input-selection/post-patch-mask experiment, not direct token injection |

Status labels used in this report: `native_candidate`, `siglip_candidate`, `rope_sparse_candidate`, `input_selection_only`, `post_encoder_only`, `unsupported_for_now`, and `needs_code_inspection`. No target model is currently marked as a functional `rope_sparse_candidate`; that label is reserved for a model that exposes explicit spatial/temporal position IDs, avoids or exactly reconstructs dense-grid/window operations, and can align projector plus placeholder counts without training.

### Revised Sparse Feasibility Table

| Model | Status | Positional encoding behavior | Dense-grid dependency | Pooling / merger / resampler | Placeholder behavior | Direct sparse feasibility | Recommended integration mode | Risk |
|---|---|---|---|---|---|---|---|---|
| LLaVA-OV / LLaVA-OneVision 8B | `siglip_candidate` | SigLIP absolute 2D positions plus Qwen2 RoPE; anyres/newline positions must remain aligned | yes/packed-grid likely | video pooling and anyres packing | dynamic behavior needs inspection | `needs_code_inspection` | `autogaze_frame_selection` first | medium |
| LongVA-7B | `input_selection_only` | CLIP absolute 2D positions plus Qwen2 RoPE in checked config | yes/anyres likely | LLaVA-style projector/unires path | placeholder expansion needs inspection | blocked for checked CLIP tower | `autogaze_frame_selection` | medium |
| LongVILA-R1-7B | `native_candidate` | SigLIP absolute/interpolated positions plus TSP/video pooling and Qwen2 RoPE | unknown | TSP/video pooling and VILA projector | VILA `_embed` placeholders need inspection | `needs_code_inspection` | official/input selection first, sparse later | medium |
| Apollo-7B | `post_encoder_only` | hybrid SigLIP positions plus InternVideo temporal positions and Perceiver latents | yes | Perceiver connector fixed/resampled tokens | connector output placeholders need inspection | blocked | `autogaze_chop_selection` | high |
| VideoLLaMA3-7B | `siglip_candidate` | SigLIP-NaViT positioning plus Qwen2.5 RoPE | unknown | spatial merge/token compression | placeholder count after compression needs inspection | `needs_code_inspection` | `autogaze_frame_selection` | high |
| VideoChat-Flash | `input_selection_only` | UMT hierarchical spatial/temporal positions plus Qwen/Yarn positions | yes | hierarchical compression to very low tokens/frame | compressed token placeholders fixed/opaque | blocked | `autogaze_frame_selection` | high |
| InternVL3.5-8B | `input_selection_only` | InternViT dynamic tile positions plus Qwen3 RoPE | yes | pixel shuffle / visual resolution routing | dynamic tile placeholders are processor-owned | blocked | `autogaze_chop_selection` | high |
| Qwen2.5-VL-7B | `input_selection_only` | M-RoPE from official `grid_thw`, temporal patching, window attention | yes | visual merger | processor-generated dynamic placeholders | blocked | official processor / input selection | high |

Direct sparse feasibility means `siglip_sparse_patch` or `rope_sparse_patch` can run before the vision encoder without training. `needs_code_inspection` does not mean supported; it means the architecture is plausible enough to inspect, but the PoC must still block sparse execution.

### Model Notes

### LLaVA-OV / LLaVA-OneVision 8B

The checked Transformers docs describe a SigLIP vision encoder and Qwen2 backbone. They also state that videos are pooled to a fixed total sequence per frame, so sparse AutoGaze tokens cannot simply replace dense grid tokens without preserving the pooling and placeholder contract. Recommendation: input selection first; later inspect whether the video pooling stage can consume sparse SigLIP outputs with deterministic position remapping. Risk: medium.

### LongVA-7B

The checked `lmms-lab/LongVA-7B` config uses `openai/clip-vit-large-patch14-336` as the vision tower, not SigLIP. That changes the initial hypothesis: this checkpoint is not an AutoGaze SigLIP sparse candidate. Recommendation: input-level frame/window/chop selection only unless a separate SigLIP LongVA variant is located and inspected. Risk: medium.

### LongVILA-R1-7B

The checked config uses VILA custom code, Qwen2 LLM, and `paligemma-siglip-so400m-patch14-448` as the vision tower. This is the closest external candidate to the NVILA-family path, but direct token injection still needs code inspection of VILA media embedding, TSP video pooling, projector behavior, and visual placeholder construction. Recommendation: highest-priority native sparse investigation after official-processor smoke. Risk: medium.

### Apollo-7B

The checked Apollo config uses a hybrid vision tower with SigLIP and InternVideo2, plus a Perceiver connector that outputs a configured visual token count. This makes direct sparse SigLIP patch injection unsafe unless the SigLIP branch can be isolated without breaking the hybrid connector. Recommendation: input-level chop selection and, later, clearly labeled post-encoder pruning only. Risk: high.

### VideoLLaMA3-7B

The model card identifies Qwen2.5 as the base language model and a separately released VL3-SigLIP-NaViT vision encoder with patch size 14. This is a SigLIP-family candidate, but NaViT layout, spatial merge, and token compression make direct injection unverified. Recommendation: official processor/input selection first; sparse work only after model-code inspection. Risk: high.

### VideoChat-Flash

The model card states the 7B model is built on UMT-L and Qwen2, using only 16 tokens per frame with hierarchical compression. Because the visual encoder is not SigLIP and compression is central to the model, AutoGaze should only select frames/windows/chops before official inference. Recommendation: input-selection-only. Risk: high.

### InternVL3.5-8B

The model card/config describe InternViT, dynamic high-resolution tiling, Qwen3, and pixel shuffle compression; Flash variants add Visual Resolution Router. This is incompatible with direct AutoGaze patch replacement until tile layout, compression, and placeholder expansion are solved. Recommendation: use AutoGaze as crop/tile selector only. Risk: high.

### Qwen2.5-VL-7B

The model card/config describe a native dynamic-resolution ViT with window attention, `video_grid_thw`, temporal patching, visual merger, and M-RoPE. Direct selected-token injection remains disabled because selected AutoGaze patches would need exact M-RoPE IDs, attention masks, merger outputs, and placeholder counts. Recommendation: official processor path plus input-level selection or clearly labeled patch masking only. Risk: high.

### Adapter Plan

Implemented lightweight registry/stub targets:

- `llava_ov`
- `longva`
- `longvila_r1`
- `apollo`
- `videollama3`
- `videochat_flash`
- `internvl3_5`
- `qwen2_5_vl`

Each adapter must report:

- requested model
- actual model loaded
- integration mode
- real checkpoint loaded
- generation ran
- unsupported reason
- positional encoding compatibility
- token count compatibility
- training/trainable adapter requirement status

Implemented PoC surface in this branch:

- `scripts/poc_model_registry.py` registers canonical `nvila`, legacy `qwen`, external MLLMs, `generic_mllm`, `generic_vit`, and an explicit `external` vision encoder key.
- `scripts/poc_model_adapters.py` keeps canonical NVILA/Qwen behavior and adds lazy external adapter stubs. External adapters return `stub-only` when real loading is disabled. With real loading enabled, they require an explicit `model_id` or checkpoint path and attempt standard Transformers model/processor loading lazily.
- `scripts/infer_full.py` accepts `--integration-mode` and routes external model requests to the requested adapter. Unsupported modes are blocked with clear reasons; no external model is redirected to NVILA and no external vision encoder is redirected to modified SigLIP.
- `configs/poc_inference/external/*.yaml` are planning/smoke configs only. None are marked runnable unless tested.

Supported integration mode names:

- `official_processor`
- `autogaze_frame_selection`
- `autogaze_chop_selection`
- `siglip_sparse_patch`
- `rope_sparse_patch`
- `post_encoder_pruning`
- `direct_visual_token_injection`

Mode rules:

- `official_processor`: model-specific official processor/chat path; no AutoGaze token injection.
- `autogaze_frame_selection`: AutoGaze selects frames/windows; the MLLM still receives official inputs.
- `autogaze_chop_selection`: AutoGaze selects or guides regions/chops; the MLLM still receives official inputs.
- `siglip_sparse_patch`: only for verified SigLIP-compatible models with deterministic grid/position mapping.
- `rope_sparse_patch`: only for models exposing explicit spatial/temporal position IDs and no unresolved dense-grid/window/merger assumptions.
- `post_encoder_pruning`: may reduce downstream tokens, but must not be described as encoder-side acceleration.
- `direct_visual_token_injection`: disabled by default; allowed only after position IDs, attention masks, projector compatibility, dynamic token count, and placeholders are verified.

### Next Implementation Target

Recommended immediate real smoke test:

1. LLaVA-OneVision official processor input-selection smoke if the goal is fastest Hugging Face coverage.
2. LongVILA-R1 official processor smoke if the goal is closest NVILA/VILA-family sparse integration.

Recommended direct/sparse investigation order:

1. LongVILA-R1-7B
2. LLaVA-OneVision
3. VideoLLaMA3

Keep LongVA, Apollo, VideoChat-Flash, InternVL3.5, and Qwen2.5-VL in input-selection or post-encoder-only mode until the blocked token-layout issues are explicitly solved.

## Section B: V-JEPA2 as an Alternative Video Encoder and Decoder / MLLM Attachment Plan

This section covers V-JEPA2 as a non-SigLIP video encoder for AutoGaze-guided PoC experiments. The local checkpoint metadata inspected in `weights/vjepa2-vitl-fpc64-256/config.json` reports `model_type=vjepa2`, `crop_size=256`, `image_size=256`, `frames_per_clip=64`, `patch_size=16`, `tubelet_size=2`, and `hidden_size=1024`. The local video processor config reports resize to shortest edge 292, center crop to 256, channels-first tensors, ImageNet normalization, and `VJEPA2VideoProcessor`.

### V-JEPA2 + AutoGaze + Decoder / MLLM Plan

| Question | Conclusion | Status |
|---|---|---|
| Can AutoGaze output be applied to V-JEPA2 instead of SigLIP ViT? | Yes for frame/window/chop selection and dense zero-mask probing. Direct sparse tubelet execution is not verified. | `feasible_input_only` / `feasible_zero_mask_only` |
| Can AutoGaze-selected patches/tubelets be represented using V-JEPA2 positional encoding? | Possibly in theory if selected `(t,h,w)` tubelet IDs can be passed into the encoder's 3D position path, but the local processor/config does not expose a sparse tubelet API. | `needs_source_inspection` |
| Can V-JEPA2 outputs connect to reviewed MLLMs without training? | Not as a drop-in replacement. Existing MLLM projectors are tied to SigLIP, InternViT, Qwen native ViT, UMT, or hybrid towers. | `blocked_without_training` |
| Which decoder heads are suitable for a V-JEPA2 PoC? | Official classification head/checkpoints, frozen feature extraction, simple probe/retrieval, and dense/zero-mask AutoGaze comparisons. | `feasible` |

V-JEPA2 input is treated as dense video tensor data, conceptually `[B,T,C,H,W]`, before processor/model-specific layout handling. The nominal dense token grid for the local checkpoint is `32` temporal tubelets by `16 x 16` spatial patches. AutoGaze can map selected patch boxes into this dense grid for zero-mask probing, but physically passing only selected tubelets requires source-level confirmation that patchify, 3D-RoPE position assignment, attention masks, and any predictor masks remain correct with holes.

Selected real smoke evidence: `configs/poc_inference/external/selected_vjepa2_smoke.yaml` was run on `assets/example_input.mp4` with 4 sampled frames and `skip_predictor=True`. The local V-JEPA2 model loaded from `weights/vjepa2-vitl-fpc64-256` and produced feature shape `[1, 512, 1024]` with mean-pooled shape `[1, 1024]`. This validates dense local feature extraction only. It does not validate sparse tubelet execution, context-mask compute reduction, or MLLM projector compatibility.

Important compute statement: `context_mask` / `target_mask` may be useful for a JEPA-style probe only after source inspection. This branch does not claim that either mask reduces encoder compute. Until verified, `context_mask` is documented as a probe/masking path, not selected-token acceleration.

### V-JEPA2 Integration Modes

| Mode | Behavior | AutoGaze use | Compute claim | Status |
|---|---|---|---|---|
| `vjepa2_official_dense` | Use V-JEPA2 official processor/model on full dense clips. | None | Dense baseline only | `feasible` |
| `autogaze_frame_selection_vjepa2` | AutoGaze selects frames/windows, V-JEPA2 receives selected dense clips. | Input-level frame/window selection | May reduce work by reducing clips, not sparse tubelets | `feasible_input_only` |
| `autogaze_chop_selection_vjepa2` | AutoGaze guides selected crops/chops, V-JEPA2 receives dense crops/clips. | Input-level region/chop selection | May reduce work by reducing crops/chops | `feasible_input_only` |
| `autogaze_zero_mask_vjepa2` | Dense frames remain; unselected AutoGaze regions are zero/mean masked. | Dense image-space mask | No encoder acceleration | `feasible_zero_mask_only` |
| `vjepa2_context_mask_probe` | Map AutoGaze patch/tubelet IDs to V-JEPA2 `context_mask` / `target_mask` if source supports it. | Predictor/context mask probe | Unknown; no claim until source inspection | `needs_source_inspection` |
| `vjepa2_sparse_tubelet` | Physically pass only selected tubelets into the V-JEPA2 encoder. | Native sparse tubelet selection | Potentially yes, but not implemented | `unsupported_for_now` |
| `vjepa2_to_mllm_projector` | Use V-JEPA2 features as MLLM visual tokens. | External visual encoder replacement | No claim; connector mismatch dominates | `blocked_without_training` |

### V-JEPA2 Decoder Recommendations

| Decoder / head | Task type | Training needed | Zero-shot PoC | VQA | Action recognition | Query text | Frame selection | Chop selection | Zero-mask | MLLM without training | Risk | Recommended status |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| VJEPA2ForVideoClassification / action recognition head | classification | No if using a trained classification checkpoint; yes for new labels | Yes with trained head | No | Yes | No | Yes | Yes | Yes | No | low | highest priority |
| frozen V-JEPA2 + linear classifier | classification/probe | Yes for supervised labels | No for new labels | No | Yes | No | Yes | Yes | Yes | No | medium | useful after labels |
| frozen V-JEPA2 + MLP classifier | classification/probe | Yes | No for new labels | No | Yes | No | Yes | Yes | Yes | No | medium | useful after labels |
| frozen V-JEPA2 + temporal pooling + kNN/retrieval classifier | retrieval/probe | No for pure retrieval; labels/index needed | Yes for retrieval-style PoC | Limited | Yes, if indexed | Maybe through text embedding outside V-JEPA2 | Yes | Yes | Yes | No | medium | medium priority |
| frozen V-JEPA2 + temporal transformer decoder | temporal reasoning | Yes | No | Limited | Yes | No | Yes | Yes | Yes | No | high | future work |
| frozen V-JEPA2 + Q-Former / Perceiver connector + LLM | VQA/MLLM | Yes | No | Yes after training | Maybe | Yes after training | Yes | Yes | Yes | No | high | blocked without training |
| frozen V-JEPA2 + existing MLLM projector | VQA/MLLM | No only if a compatible frozen projector already exists; otherwise yes | No by default | Yes only if projector verified | Maybe | Yes if projector verified | Yes | Yes | Yes | Usually no | high | blocked without verified projector |
| V-JEPA2-AC / action-conditioned head | action-conditioned video | Depends on available checkpoint | Maybe with official checkpoint | No | Yes | Maybe action prompts only | Yes | Yes | Yes | No | medium | needs source/checkpoint inspection |

Highest-priority V-JEPA2 PoC path: official dense feature/classification smoke, then AutoGaze frame selection and zero-mask comparisons. Low-priority/blocked paths: V-JEPA2 to NVILA/LongVA/LLaVA/Qwen/InternVL/VideoChat projectors, direct visual token injection, and new Q-Former/Perceiver/MLP connectors unless training is explicitly allowed in a separate task.

### V-JEPA2 as Vision Encoder for Candidate MLLMs

| MLLM target | Expected native vision encoder | Expected visual feature dimension | Projector / connector type | Accept V-JEPA2 dim without training | Variable visual tokens | Dynamic placeholders | Positional / temporal compatibility | Recommended V-JEPA2 mode | Training required | Status | Risk |
|---|---|---|---|---|---|---|---|---|---|---|---|
| NVILA | SigLIP/NVILA vision tower | SigLIP-family projector dimension | NVILA processor/projector | No | Processor-managed | Processor-managed | SigLIP grid, not V-JEPA2 tubelets | input/zero-mask outside NVILA or separate feature extraction | Yes unless compatible projector exists | `blocked_without_training` | high |
| LongVILA-R1 | VILA SigLIP SO400M + TSP | SigLIP/VILA-specific | VILA projector + media embedding | Unknown, likely no | Unknown | Unknown | VILA SigLIP/TSP positions, not V-JEPA2 3D tubelets | native LongVILA official processor; V-JEPA2 separate probe | Yes/unknown | `needs_source_inspection` | high |
| LongVA | CLIP ViT-L/14-336 in checked config | CLIP/LLaVA projector dimension | LLaVA-style MLP projector | No | Unknown | Unknown | CLIP anyres/unires, not V-JEPA2 | V-JEPA2 separate feature extraction | Yes | `blocked_without_training` | high |
| LLaVA-OV / OneVision | SigLIP patch14 anyres/video | SigLIP projector dimension | multimodal projector with packed/pool tokens | No | Unknown | Unknown | anyres/video pooling, not V-JEPA2 tubelets | V-JEPA2 separate feature extraction | Yes | `blocked_without_training` | high |
| VideoLLaMA3 | VL3-SigLIP-NaViT | SigLIP-NaViT output | custom projector/compressor | No/unknown | Unknown | Unknown | NaViT/compression, not V-JEPA2 tubelets | V-JEPA2 separate feature extraction | Yes/unknown | `blocked_without_training` | high |
| Apollo | SigLIP + InternVideo2 hybrid | hybrid tower dimensions | Perceiver connector | Unknown | Likely fixed/resampled | Unknown | Hybrid temporal positions may make V-JEPA2 plausible only if code has branch | source inspection before any claim | Unknown | `needs_source_inspection` | high |
| Qwen2.5-VL | native Qwen ViT | Qwen visual merger dimension | native visual merger | No | Processor dynamic only | Processor-owned | M-RoPE/grid_thw/window attention incompatible with V-JEPA2 features | official Qwen input selection; V-JEPA2 separate probe | Yes | `blocked_without_training` | high |
| InternVL3.5 | InternViT dynamic tiling | InternVL projector dimension | pixel shuffle / MLP connector | No | Unknown | Dynamic tile placeholders | InternViT tile layout, not V-JEPA2 | InternVL input selection; V-JEPA2 separate probe | Yes | `blocked_without_training` | high |
| VideoChat-Flash | UMT-L / UMT-HD-L | UMT compressed feature dimension | hierarchical compressor | No | Likely compressed/fixed | Unknown | UMT hierarchy, not V-JEPA2 | input selection only; V-JEPA2 separate probe | Yes | `blocked_without_training` | high |
| generic_mllm | user-defined | user-defined | user-provided frozen connector | Only if explicitly verified | Unknown | Unknown | Depends on provided connector | `vjepa2_to_mllm_projector` only with verified frozen connector | Unknown/yes by default | `stub-only` | high |

### V-JEPA2 Registry / Adapter Status

The registry now treats `vjepa2` as a `video_encoder` entry with `positional_encoding_type=3d_rope`, `patch_structure=tubelet`, `patch_size=16`, `crop_size=256`, `frames_per_clip=64`, `tubelet_size=2`, `supports_official_dense=true`, `supports_autogaze_frame_selection=true`, `supports_autogaze_chop_selection=true`, `supports_autogaze_zero_mask=true`, `supports_context_mask_probe=unknown`, `supports_sparse_tubelet=unknown`, and `supports_direct_mllm_projection=false`.

The adapter exposes:

- `run_official_dense()`
- `run_autogaze_frame_selection()`
- `run_autogaze_chop_selection()`
- `run_autogaze_zero_mask()`
- `run_context_mask_probe()`
- `run_sparse_tubelet()`
- `run_video_classification()`
- `run_feature_extraction()`
- `get_patch_grid()`
- `get_tubelet_grid()`
- `get_position_encoding_status()`
- `get_output_dim()`
- `supports_mllm_projection()`
- `recommend_decoder()`
- `status_report()`

`run_context_mask_probe()` and `run_sparse_tubelet()` intentionally raise clear `NotImplementedError` until V-JEPA2 source-level mask semantics and sparse tubelet position handling are verified.

## Asset Acquisition and Evidence Status

Asset acquisition is now staged through `configs/poc_inference/model_asset_manifest.yaml` and the scripts in `scripts/prepare_external_model_assets.py`, `scripts/verify_external_model_assets.py`, and `scripts/inspect_external_model_configs.py`. These scripts do not load full model weights during dry-run, verification, or config inspection. Model download is disabled unless `--download` is explicitly supplied, and authentication tokens are read only from an environment variable such as `HF_TOKEN`.

| Model | Theoretical compatibility | Config-inspected compatibility | Local asset verified | Real model loaded | Real inference passed | Current blocker |
|---|---|---|---|---|---|---|
| LLaVA-OV / OneVision | `siglip_candidate` | pending local config | no | no | no | checkpoint missing; anyres/video pooling and placeholder alignment need inspection |
| LongVA | `input_selection_only` for checked hypothesis | pending local config | no | no | no | checkpoint missing; CLIP/unires dense-token contract blocks direct sparse path |
| LongVILA-R1 | `native_candidate` / `needs_code_inspection` | pending local config | no | no | no | checkpoint missing; VILA SigLIP/TSP projector and placeholder handling need inspection |
| Apollo | `post_encoder_only` / input selection | pending local config | no | no | no | checkpoint missing; hybrid tower and Perceiver connector block direct sparse path |
| VideoLLaMA3 | `siglip_candidate` / `needs_code_inspection` | pending local config | no | no | no | checkpoint missing; NaViT packing and compression need inspection |
| VideoChat-Flash | `input_selection_only` | pending local config | no | no | no | checkpoint missing; UMT hierarchy/compression block direct sparse path |
| InternVL3.5 | `input_selection_only` | pending local config | no | no | no | checkpoint missing; dynamic tiling and pixel shuffle block direct sparse path |
| Qwen2.5-VL | `input_selection_only` | inspected locally: `model_type=qwen2_5_vl`, `patch_size=14`, M-RoPE indicators visible | verified local in `weights/Qwen2.5-VL-7B-Instruct` | no | no | direct sparse blocked by M-RoPE, `grid_thw`, window attention, visual merger, and placeholder contract |
| V-JEPA2 | `feasible_input_only` / `feasible_zero_mask_only` | inspected locally: `model_type=vjepa2`, `crop_size=256`, `frames_per_clip=64`, `patch_size=16`, `tubelet_size=2`, `hidden_size=1024` | verified local in `weights/vjepa2-vitl-fpc64-256` | no | no | sparse tubelet and MLLM projector paths need source/projector verification |

Evidence levels must remain separate:

- Theoretical compatibility means the architecture looks plausible from docs or known model families.
- Config-inspected compatibility means a local config was parsed without loading weights.
- Local asset verified means expected config, processor/tokenizer, and weight files exist.
- Real model loaded means the explicit adapter loaded the requested checkpoint.
- Real inference passed means a minimal smoke run produced an answer or feature summary.

No row above is upgraded to direct sparse/token-injection support by local asset availability alone.

## Per-Model Mode Matrix

| Model | direct_visual_token_injection | autogaze_frame_selection | autogaze_chop_selection | autogaze_zero_mask | post_encoder_zero_mask | sparse patch feasibility | RoPE sparse feasibility | Recommended first mode | Recommended next mode |
|---|---|---|---|---|---|---|---|---|---|
| LongVILA-R1 | `needs_source_inspection`; disabled by default because projector/TSP/placeholders are unverified | `blocked_missing_assets`, implementable with official processor after assets | `blocked_missing_assets`, feasible as dense crop input | `blocked_missing_assets`, first safe probe after assets | `needs_source_inspection` | `needs_source_inspection` for `native_sparse_patch` / `light_modified_sparse` | not primary vision path; LLM RoPE does not solve SigLIP/TSP sparse mapping | `autogaze_zero_mask` | official processor, then sparse source inspection |
| LongVA | disabled; CLIP/unires projector and placeholders unverified | `blocked_missing_assets`, feasible with official path | `blocked_missing_assets`, feasible | `blocked_missing_assets`, safe dense probe | `needs_source_inspection` | `blocked_architecture` for checked CLIP tower | not applicable to checked CLIP vision tower | `autogaze_zero_mask` or frame selection | official processor |
| LLaVA-OV / OneVision | disabled; pooled video tokens/placeholders unverified | `blocked_missing_assets`, feasible | `blocked_missing_assets`, feasible | `blocked_missing_assets`, safe dense probe | `needs_source_inspection` | `needs_source_inspection` for SigLIP/anyres path | LLM RoPE only; vision sparse path is SigLIP/anyres | `autogaze_zero_mask` | inspect anyres/video pooling |
| VideoLLaMA3 | disabled; compressor/projector/placeholders unverified | `blocked_missing_assets`, feasible | `blocked_missing_assets`, feasible | `blocked_missing_assets`, safe dense probe | `needs_source_inspection` | `needs_source_inspection` for SigLIP-NaViT | `needs_source_inspection` if NaViT/position IDs are exposed before compression | `autogaze_zero_mask` | inspect NaViT/compressor |
| Apollo | disabled; hybrid Perceiver connector fixed/resampled | `blocked_missing_assets`, feasible | `blocked_missing_assets`, preferred input-level mode | `blocked_missing_assets`, safe dense probe | `needs_source_inspection` | `blocked_architecture` unless SigLIP branch can be isolated | not verified | `autogaze_chop_selection` / zero-mask | inspect SigLIP branch only |
| Qwen2.5-VL | `blocked_architecture`; disabled by M-RoPE/grid/merger/placeholders | runnable if local assets and real loading allowed | runnable if local assets and real loading allowed | implementable as image-space probe; no acceleration claim | `needs_source_inspection` | blocked: not SigLIP | `blocked_architecture` for M-RoPE/window/merger path | official processor / frame selection | zero-mask probe |
| InternVL3.5 | `blocked_architecture`; dynamic tiling/pixel shuffle/placeholders | `blocked_missing_assets`, feasible | `blocked_missing_assets`, preferred input-level mode | `blocked_missing_assets`, safe image-space probe | `needs_source_inspection` | blocked | not primary route | `autogaze_chop_selection` | zero-mask |
| VideoChat-Flash | `blocked_architecture`; hierarchical compression/fixed tokens | `blocked_missing_assets`, feasible | `blocked_missing_assets`, feasible | `blocked_missing_assets`, safe dense probe | `needs_source_inspection` | blocked | not primary route | `autogaze_frame_selection` | zero-mask |

## RoPE Sparse Patch Status

`rope_sparse_patch` is not enabled by default for any target. The PoC now prepares validation metadata when the mode is explicitly selected, but the support status remains blocked or `needs_source_inspection` unless all positional, attention, dense-grid, projector, and placeholder conditions are verified.

| Model | positional_encoding_type | rope_sparse_patch_status | position_id_control_available | dense_grid_dependency | window_attention_dependency | projector_variable_token_support | placeholder_alignment_support | Fallback if blocked |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-VL | M-RoPE / `grid_thw` / window attention | `blocked_architecture` | processor-owned, not verified for holes | true | true | processor dynamic only | processor-owned | official processor, frame/chop selection, zero-mask |
| VideoLLaMA3 | SigLIP-NaViT plus Qwen2.5 RoPE | `needs_source_inspection` | unknown | unknown | unknown | unknown | unknown | zero-mask, frame selection |
| V-JEPA2 | 3D-RoPE tubelets | `needs_source_inspection` | not exposed by local processor/config | dense official path | unknown | not an MLLM projector path | not applicable | feature extraction, frame selection, zero-mask |
| LongVILA-R1 | SigLIP absolute/interpolated vision positions plus Qwen2 RoPE | not primary vision RoPE path | unknown for VILA media path | unknown | unknown | unknown | unknown | zero-mask, official processor |

## V-JEPA2 Decoder Recommendation

Recommended V-JEPA2 decoder order:

1. `VJEPA2ForVideoClassification` / action-recognition head if a trained head checkpoint is provided.
2. Frozen V-JEPA2 feature extraction plus temporal pooling.
3. Frozen V-JEPA2 feature extraction plus kNN/retrieval probe.
4. Frozen V-JEPA2 plus simple classifier/probe.
5. V-JEPA2 `context_mask` probe after source inspection.
6. V-JEPA2 sparse tubelet experimental path after source inspection.
7. V-JEPA2 plus MLLM projector only if a compatible frozen projector exists.
8. V-JEPA2 plus trainable Q-Former/Perceiver/MLP connector is out of scope for this no-training branch.

The selected V-JEPA2 config is `configs/poc_inference/external/selected_vjepa2_smoke.yaml`: dense feature extraction plus temporal-pooling probe. It records feature-shape fields in smoke outputs when real feature extraction is run; dry-run records them as unavailable.
