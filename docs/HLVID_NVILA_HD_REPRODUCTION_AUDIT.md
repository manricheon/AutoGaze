# HLVid NVILA-HD Reproduction-Ready Audit

This branch is prepared for HLVid-style NVILA-HD-Video + AutoGaze smoke and subset evaluation. It is not a completed paper reproduction and no full HLVid benchmark has been run.

## Concrete Deliverables

| Requirement | Artifact / Evidence | Status |
|---|---|---|
| Read project and canonical references | `docs/PROJECT_REQUEST.md`, `docs/nvila-hd-video-readme.md`, `QUICK_START.md` | done |
| Preserve original AutoGaze files | Work is in `scripts/`, `configs/poc_inference/`, and `docs/`; no original `INTEGRATION.md` or `QUICK_START.md` edits are required | done |
| HLVid dataset loading | `scripts/evaluate_hlvid_nvila.py` supports local JSON/JSONL/CSV and Hugging Face `bfshi/HLVid` via `datasets` | done |
| Multiple-choice prompt and answer parsing | `build_hlvid_prompt`, `extract_choice`, `normalize_choice` in `scripts/evaluate_hlvid_nvila.py` | done |
| NVILA-HD README processor settings | `configs/poc_inference/hlvid_nvila_hd_eval.yaml` uses 128 frames, 64 thumbnails, 48 tiles, tile ratio `[0.2]+[0.06]*15`, tile loss `0.6`, thumbnail ratio `1`, AutoGaze batch 16, SigLIP batch 32 | done |
| Smoke/subset configs | `configs/poc_inference/hlvid_nvila_hd_smoke.yaml`, `configs/poc_inference/hlvid_nvila_hd_subset.yaml` | done |
| Metrics/reporting | `evaluate_hlvid_nvila.py` writes JSON, JSONL, CSV, metrics CSV, and Markdown report | done |
| Tile-safe single-video PoC configs | `configs/poc_inference/hlvid_infer_full_resize_safe.yaml`, `configs/poc_inference/hlvid_infer_full_resize_then_chop_safe.yaml` | done |
| A0/A1/A2/A3 behavior preserved | Existing protected configs remain separate and are not required by HLVid evaluator | done |
| Full HLVid evaluation avoided | `allow_real_model_loading` defaults false and configs cap smoke/subset size | done |

## Actual High-Resolution Video Path

The HLVid reproduction path should use the official NVILA-HD processor, not the PoC `resize`, `chop`, or `resize_then_chop` scaler.

The official path is:

```text
source video path
-> NVILAProcessor._load_video_frames(..., num_video_frames)
-> uniform frame sampling
-> dynamic aspect-ratio spatial tile grid bounded by max_tiles_video
-> 392x392 tile tensors plus whole-frame 392x392 thumbnails
-> AutoGaze on tile clips and thumbnails
-> gazing_info into modified SigLIP / NVILA model
-> MC answer generation
```

Important consequences:

- `num_video_frames=128` means the processor samples 128 frames from the whole video. It is not the same as PoC `num_frames=16 --frame-interval 16`.
- `max_tiles_video=48` bounds spatial tile count per sampled frame. For a 1920x1080 frame, the processor chooses the closest aspect-ratio grid under that budget and resizes the frame to `cols*392` by `rows*392` before cropping tiles.
- The thumbnail path is separate from tile processing and is controlled by `num_video_frames_thumbnail`, `gazing_ratio_thumbnail`, and `task_loss_requirement_thumbnail`.
- The PoC `hlvid_infer_full_resize_*` configs are only single-video visualization/OOM probes. They are not the canonical HLVid reproduction path.

## Commands

Dry-run with a local HLVid-style file:

```bash
python scripts/evaluate_hlvid_nvila.py \
  --config configs/poc_inference/hlvid_nvila_hd_smoke.yaml \
  --dataset-path /path/to/hlvid_sample.jsonl \
  --video-root /path/to/videos \
  --output-dir outputs/hlvid_nvila_hd_smoke_dry_run \
  --dry-run
```

Subset run with local NVILA-HD weights:

```bash
python scripts/evaluate_hlvid_nvila.py \
  --config configs/poc_inference/hlvid_nvila_hd_subset.yaml \
  --dataset-name bfshi/HLVid \
  --model-path weights/NVILA-8B-HD-Video \
  --processor-path weights/NVILA-8B-HD-Video \
  --allow-real-model-loading \
  --local-files-only \
  --max-samples 20 \
  --output-dir outputs/hlvid_nvila_hd_subset_real
```

Single-video PoC visualization, not reproduction:

```bash
python scripts/infer_full.py \
  --config configs/poc_inference/hlvid_infer_full_resize_then_chop_safe.yaml \
  --video-path /path/to/hlvid_video.mp4 \
  --query-text "Question: ... A. ... B. ... C. ... D. ... Please answer directly with the letter of the correct answer." \
  --allow-real-model-loading \
  --local-files-only \
  --output-dir outputs/hlvid_single_video_probe
```

## Output Contract

`scripts/evaluate_hlvid_nvila.py` writes:

```text
predictions/hlvid_predictions.json
predictions/hlvid_predictions.jsonl
predictions/hlvid_predictions.csv
logs/poc_summary.json
logs/metrics.json
logs/metrics.csv
logs/hlvid_report.md
```

The report records the official high-resolution path so results are not confused with PoC resize/chop preprocessing.

## Remaining Limits

- Full HLVid evaluation has not been run.
- Exact per-sample tile counts are owned by the NVILA processor and require real processor execution.
- Processor-internal AutoGaze time is included in the official processor/model path; standalone PoC AutoGaze timing is only for `infer_full.py` / `infer_autogaze.py` probe runs.
- This path does not use direct visual-token injection outside the official NVILA processor/model implementation.
