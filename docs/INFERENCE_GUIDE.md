# Inference Guide

The maintained PoC inference guide for this cleaned branch is:

```text
docs/inference_guide_for_poc.md
```

This file is kept because project instructions require `docs/INFERENCE_GUIDE.md` to exist. The detailed guide is intentionally separate and focused on the remaining inference-only branch surface.

External model checkpoint preparation and smoke testing are documented in:

```text
docs/MODEL_ASSET_MANIFEST.md
docs/EXTERNAL_MODEL_SMOKE_PLAN.md
```

Use dry-run first:

```bash
python scripts/prepare_external_model_assets.py \
  --manifest configs/poc_inference/model_asset_manifest.yaml \
  --model all \
  --dry-run \
  --write-report docs/MODEL_ASSET_DOWNLOAD_REPORT.md
```

For incomplete external checkpoints such as a partially downloaded LongVILA-R1 folder, use explicit dummy-weight smoke to test routing and output files without loading real shards:

```bash
python scripts/run_external_model_smoke.py \
  --config configs/poc_inference/external/selected_tier1_smoke.yaml \
  --video-path dummy \
  --query-text "Describe the main action in the video." \
  --allow-dummy-weights \
  --local-files-only
```

Dummy smoke is not a model-quality test. Its outputs are deliberately labeled as dummy and do not count as real inference.

Direct sparse token injection is disabled by default for external models. Use official processors, AutoGaze input-level frame/chop selection, or zero-mask probes until positional IDs, dense-grid behavior, projector compatibility, and visual placeholders are verified.

The existing A0-A3 AutoGaze ON/OFF + NVILA inference configs remain the canonical path and keep their previous `official_processor` behavior.
