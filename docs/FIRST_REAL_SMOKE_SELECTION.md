# First Real Smoke Selection

`docs/FIRST_REAL_SMOKE_SELECTION.md` was reconstructed from `docs/EXTERNAL_MODEL_PRIORITIZED_PLAN.md` because it was missing at the start of this run.

## Selected Tier 1 External MLLM

| Field | Value |
|---|---|
| Selected model | LongVILA-R1 / `longvila_r1` |
| Priority | Tier 1, rank 1 |
| Integration mode | `autogaze_zero_mask` |
| Config | `configs/poc_inference/external/selected_tier1_smoke.yaml` |
| Required checkpoint path | `weights/longvila_r1` |
| Required processor/tokenizer path | `weights/longvila_r1` |
| Output directory | `outputs/poc_inference/external/selected_tier1_real_smoke` for real smoke; `outputs/poc_inference/external/selected_tier1_dummy_smoke` for explicit dummy-weight plumbing smoke |
| Real smoke status | `BLOCKED`: local checkpoint exists but is incomplete |
| Dummy smoke status | `PARTIAL`: explicit dummy-weight smoke is allowed for routing/output validation only |

LongVILA-R1 remains the first selected Tier 1 model because it is the closest VILA/SigLIP-family candidate to the canonical NVILA-style path. Direct token injection and sparse patch support remain disabled until VILA media embedding, TSP pooling, projector behavior, and placeholder alignment are verified.

## Selected V-JEPA2 Target

| Field | Value |
|---|---|
| Selected model | V-JEPA2 / `vjepa2` |
| Priority | Tier 1-B |
| Selected mode | `vjepa2_feature_extraction` |
| Config | `configs/poc_inference/external/selected_vjepa2_smoke.yaml` |
| Required checkpoint path | `weights/vjepa2-vitl-fpc64-256` |
| Output directory | `outputs/poc_inference/external/selected_vjepa2_real_smoke` |
| Real smoke status | `PASS`: local V-JEPA2 model loaded and produced dense features |

V-JEPA2 is treated as an encoder-only feature extraction target. It is not connected to NVILA, Qwen, LongVILA, or other MLLM projectors because no compatible frozen projector has been verified.
