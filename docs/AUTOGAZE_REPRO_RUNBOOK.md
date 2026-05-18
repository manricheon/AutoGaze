# AutoGaze Reproduction Runbook

## Official Sources

- AutoGaze code: https://github.com/NVlabs/AutoGaze
- AutoGaze project page: https://autogaze.github.io/
- AutoGaze paper: https://arxiv.org/abs/2603.12254
- AutoGaze collection: https://huggingface.co/collections/bfshi/autogaze
- HLVid dataset: https://huggingface.co/datasets/bfshi/HLVid
- NVILA-HD-Video README path: https://github.com/NVlabs/VILA/tree/main/vila_hd/nvila_hd_video

## Local MPS Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -U pip setuptools wheel
.venv/bin/python -m pip install -r requirements-repro.txt
bash scripts/bootstrap_official_repos.sh
.venv/bin/python -m pip install -e . --no-deps --no-build-isolation
```

This branch now keeps the official AutoGaze code hierarchy at the repository root, so `autogaze` imports work the same way as they do in the upstream repository and in the NVILA-HD-Video remote code. The `external/` directory is only used for optional side repositories such as VILA.

The official AutoGaze `pyproject.toml` currently includes `flash_attn` as a normal dependency. On Apple MPS/macOS, use `--no-deps --no-build-isolation` for the editable install and rely on `requirements-repro.txt` for the MPS-compatible runtime dependencies.

If `torch.backends.mps.is_available()` is false inside a sandbox but true outside it, run the actual MPS benchmark outside the sandbox. The helper tests and CLI checks do not require MPS.

## MPS AutoGaze And SigLIP Smoke Benchmark

The packaged `assets/example_input.mp4` is a regular MP4 video used by the official quick start. Its reproduction preset is `configs/repro/example_input_autogaze.yaml`: 448x448 source video, 64 total frames, first 16 frames sampled for the AutoGaze/SigLIP smoke path.

```bash
.venv/bin/python -m repro.autogaze_bench \
  --device mps \
  --dtype float32 \
  --warmup 1 \
  --repeat 3 \
  --output-json outputs/autogaze_repro/mps_autogaze_siglip_bench.json \
  --output-csv outputs/autogaze_repro/mps_autogaze_siglip_bench.csv
```

This confirms that the official AutoGaze model loads, emits gazing metadata, and drives the customized SigLIP path on Apple MPS. Local MPS results are a code-path and tensor-contract validation, not a paper-comparable performance claim.

To run the AutoGaze/SigLIP benchmark with a local AutoGaze checkpoint, pass the checkpoint directory to `--autogaze-model`:

```bash
.venv/bin/python -m repro.autogaze_bench \
  --autogaze-model /path/to/local/autogaze-checkpoint \
  --device cuda \
  --dtype float16
```

Observed local smoke result on this workspace:

- AutoGaze revision: `ba48d0f94ac2929d6fe3ee4380dc893aa6eed0ab`
- Input: official `assets/example_input.mp4`, 16 frames, 224x224
- Token reduction ratio: about `19.91x`
- Selected non-padded patches: `213` out of raw patch budget `4240`
- Mean AutoGaze latency on MPS: about `1609.78 ms`
- Mean full SigLIP latency on MPS: about `528.80 ms`
- Mean gazed SigLIP latency on MPS: about `112.97 ms`
- SigLIP-only speedup: about `4.68x`
- SigLIP speedup including AutoGaze overhead on this small MPS smoke input: about `0.31x`

The last number is expected to be weak on local MPS because AutoGaze overhead is measured on a small smoke input. CUDA measurements should be used for leader-facing speed claims.

## Summary Report

```bash
.venv/bin/python -m repro.report \
  --autogaze-json outputs/autogaze_repro/mps_autogaze_siglip_bench.json \
  --output-json outputs/autogaze_repro/mps_report_summary.json \
  --output-csv outputs/autogaze_repro/mps_report_summary.csv
```

## CUDA Single-Sample NVILA Check

```bash
.venv/bin/python -m repro.nvila_runner \
  --mode single \
  --device cuda \
  --num-video-frames 128 \
  --num-video-frames-thumbnail 64 \
  --max-tiles-video 48 \
  --measure-ttft \
  --output-json outputs/autogaze_repro/cuda_nvila_single_128f.json
```

This mirrors the official NVILA-HD-Video quickstart scale and validates the model and processor path before the full HLVid run.

The NVILA runner records module-level timings in the output JSON. The most important fields are:

- `result.video_decode_ms`: sampled frame decode/read time. Without runner-side resize this wraps NVILA remote code's video loader. With `--video-resize-*`, this also includes runner-side full-video sampling and PIL frame resize before frames are handed to the processor.
- `result.video_tiling_ms`: NVILA processor video preparation after frames are available. This covers dynamic spatial tiling, thumbnail construction, and pixel tensorization for SigLIP/AutoGaze inputs. It does not mean SigLIP inference.
- `result.autogaze_ms`: full AutoGaze selection stage. In `autogaze` mode this includes AutoGaze forward plus sort/pad/split bookkeeping. In `keep-all` mode it mostly measures keep-all mask construction because AutoGaze forward is skipped.
- `result.autogaze_forward_ms`: AutoGaze model forward time only, when AutoGaze is actually invoked. This is the cleanest field for AutoGaze model cost.
- `result.vision_encoder_ms`: NVILA visual embedding path during generation. It wraps the vision encoding method, including SigLIP feature extraction, feature cleanup/reordering, and projection preparation.
- `result.siglip_vision_ms`: SigLIP vision tower forward time. Use this to see whether AutoGaze reduced the vision encoder workload.
- `result.mm_projector_ms`: multimodal projector forward time after vision features are selected/stacked.
- `result.llm_forward_ms`: accumulated LLM forward time inside `generate`. It includes prefill and decoding calls made by the language model.
- `result.ttft_ms`: time to generate one token from the processed visual/text input when `--measure-ttft` is enabled. This is measured by an extra one-token generation pass and is not included in `total_ms`.
- `result.decode_estimated_ms`: approximate generation decode time, computed as full `generate_ms - ttft_ms`. Treat this as an estimate because TTFT and full generation are separate calls.
- `result.stage_timings_ms`: raw nested timing buckets for `processor`, optional `ttft`, and full `generate`. Use this when the top-level field is null or when you need per-call counts.
- `result.token_metrics`: visual token and patch counts before/after AutoGaze for tiles, thumbnails, and total encoder/LLM budgets.
- `result.processor_peak_memory_bytes` and `result.peak_memory_bytes`: CUDA peak allocation for processor and full generate phases, when running on CUDA.

`--measure-ttft` runs an additional one-token generation after preprocessing. In this pipeline, TTFT is not just text decoding latency: it includes visual embedding, SigLIP/vision encoding when needed by `generate`, projector work, and the first LLM forward. Use `result.ttft_stage_timings_ms` to split that TTFT bucket into `vision_encode_total`, `siglip_vision_tower`, `mm_projector`, and `llm_forward` when those hooks are available.

Token metrics are split into two levels. The encoder patch budget is counted before TokenShuffle/projector, and includes every selected frame, every spatial tile, thumbnails, and every configured visual scale. The LLM visual-token budget is counted after TokenShuffle/projector and corresponds to the number of visual placeholder tokens consumed by the language model.

For encoder patch accounting:

- `token_metrics.video_sampled_frames`: number of full-video frames sampled for tiled video processing.
- `token_metrics.thumbnail_sampled_frames`: number of thumbnail frames processed alongside the tiled frames.
- `token_metrics.spatial_tiles_per_video`, `token_metrics.temporal_chunks_per_video`, `token_metrics.tile_sequences`: how many spatial/temporal AutoGaze/SigLIP sequences were produced.
- `token_metrics.encoder_patches_per_frame_by_scale`: multi-scale patch breakdown, for example scale `56`, `112`, `196`, `392`.
- `token_metrics.encoder_patches_per_frame_multiscale`: sum of the multi-scale patch counts for one frame.
- `token_metrics.encoder_raw_tile_patch_tokens`: total tiled-video patch budget before AutoGaze, computed from sampled frames × spatial tiles × multi-scale patches per frame.
- `token_metrics.encoder_raw_thumbnail_patch_tokens`: total thumbnail patch budget before AutoGaze, computed from thumbnail frames × multi-scale patches per frame.
- `token_metrics.encoder_raw_patch_tokens`: tile plus thumbnail raw patch budget.
- `token_metrics.encoder_autogaze_selected_tile_patch_tokens`: non-padded tile patches actually kept after AutoGaze. In `keep-all`, this should match the raw tile patch budget.
- `token_metrics.encoder_autogaze_selected_thumbnail_patch_tokens`: non-padded thumbnail patches kept. With the current runner settings thumbnails are keep-all, so this should match the raw thumbnail patch budget.
- `token_metrics.encoder_autogaze_selected_patch_tokens`: tile plus thumbnail kept patches after AutoGaze.
- `token_metrics.encoder_tile_token_reduction_ratio`, `token_metrics.encoder_thumbnail_token_reduction_ratio`, `token_metrics.encoder_token_reduction_ratio`: raw divided by kept patches for tile, thumbnail, and total budgets.

For LLM visual-token accounting:

- `token_metrics.llm_keep_all_visual_tokens_estimated`: estimated visual tokens if every tile and thumbnail patch were kept, after TokenShuffle.
- `token_metrics.llm_actual_visual_tokens`: actual visual token placeholders in the processor output after AutoGaze/keep-all padding strategy.
- `token_metrics.llm_visual_token_reduction_ratio`: estimated keep-all LLM visual tokens divided by actual visual tokens.

To compare AutoGaze against a full-token baseline, run the same input twice with only `--gazing-mode` changed. `autogaze` uses the NVILA quickstart tile selection ratios. `keep-all` sets `gazing_ratio_tile=1` and `task_loss_requirement_tile=None`, which makes the public NVILA processor construct keep-all masks without invoking AutoGaze.

```bash
.venv/bin/python -m repro.nvila_runner \
  --mode single \
  --device cuda \
  --gazing-mode autogaze \
  --num-video-frames 128 \
  --num-video-frames-thumbnail 64 \
  --max-tiles-video 48 \
  --measure-ttft \
  --output-json outputs/autogaze_repro/cuda_nvila_single_128f_autogaze.json

.venv/bin/python -m repro.nvila_runner \
  --mode single \
  --device cuda \
  --gazing-mode keep-all \
  --num-video-frames 128 \
  --num-video-frames-thumbnail 64 \
  --max-tiles-video 48 \
  --measure-ttft \
  --output-json outputs/autogaze_repro/cuda_nvila_single_128f_keep_all.json
```

For the speed story, compare `total_ms`, `video_decode_ms`, `video_tiling_ms`, `autogaze_forward_ms`, `siglip_vision_ms`, `vision_encoder_ms`, and `llm_forward_ms` between the two JSON files. For the token story, compare tile, thumbnail, and total patch budgets: `token_metrics.encoder_raw_tile_patch_tokens`, `token_metrics.encoder_autogaze_selected_tile_patch_tokens`, `token_metrics.encoder_raw_thumbnail_patch_tokens`, `token_metrics.encoder_autogaze_selected_thumbnail_patch_tokens`, `token_metrics.encoder_raw_patch_tokens`, `token_metrics.encoder_autogaze_selected_patch_tokens`, `token_metrics.encoder_token_reduction_ratio`, `token_metrics.llm_keep_all_visual_tokens_estimated`, `token_metrics.llm_actual_visual_tokens`, and `token_metrics.llm_visual_token_reduction_ratio`.

For feasibility tests, `nvila_runner` can downscale sampled video frames before the public NVILA processor tiles them. This is runner-side preprocessing: the runner samples `--num-video-frames` across the full video, resizes those frames, then passes the resized PIL frames to the NVILA processor.

```bash
.venv/bin/python -m repro.nvila_runner \
  --mode single \
  --device cuda \
  --video inputs/hlvid_example/clip_av_video_5_001.mp4 \
  --gazing-mode keep-all \
  --num-video-frames 128 \
  --num-video-frames-thumbnail 64 \
  --max-tiles-video 48 \
  --video-resize-shortest-edge 720 \
  --measure-ttft \
  --output-json outputs/autogaze_repro/hlvid_example_nvila_single_128f_keep_all_resize720.json
```

Use `--video-resize-width` and `--video-resize-height` together for exact-size tests, or `--video-resize-longest-edge` for a max-side constraint. AutoGaze/keep-all patch scale can also be changed through the processor init path:

```bash
.venv/bin/python -m repro.nvila_runner \
  --mode single \
  --device cuda \
  --video inputs/hlvid_example/clip_av_video_5_001.mp4 \
  --gazing-mode autogaze \
  --num-video-frames 128 \
  --num-video-frames-thumbnail 64 \
  --max-tiles-video 48 \
  --video-resize-shortest-edge 720 \
  --autogaze-resize-scales 56+112+196+392 \
  --autogaze-target-patch-size 14 \
  --measure-ttft \
  --output-json outputs/autogaze_repro/hlvid_example_nvila_single_128f_autogaze_resize720.json
```

Run preflight with the same resize flags first. The output includes `source_metadata`, `effective_video`, and `video_resize`, so you can see whether the test is using original 4K dimensions or the downscaled frame dimensions.

To run NVILA-HD-Video with a local AutoGaze checkpoint, pass the same checkpoint directory through `--autogaze-model`. The runner forwards it to the NVILA processor as `autogaze_model_id`, which is the argument used by the model's remote code:

```bash
.venv/bin/python -m repro.nvila_runner \
  --mode single \
  --device cuda \
  --autogaze-model /path/to/local/autogaze-checkpoint \
  --num-video-frames 128 \
  --num-video-frames-thumbnail 64 \
  --max-tiles-video 48 \
  --measure-ttft \
  --output-json outputs/autogaze_repro/cuda_nvila_single_local_autogaze.json
```

To run both NVILA-HD-Video and AutoGaze from local checkpoint directories, use `--nvila-model` and `--autogaze-model` together. `--nvila-model` is an alias for `--model-path`.

```bash
.venv/bin/python -m repro.nvila_runner \
  --mode single \
  --device cuda \
  --nvila-model /path/to/local/nvila-checkpoint \
  --autogaze-model /path/to/local/autogaze-checkpoint \
  --num-video-frames 128 \
  --num-video-frames-thumbnail 64 \
  --max-tiles-video 48 \
  --measure-ttft \
  --output-json outputs/autogaze_repro/cuda_nvila_single_local_models.json
```

## HF Space Example Videos

The AutoGaze Space at `https://huggingface.co/spaces/bfshi/AutoGaze` uses three example videos: `doorbell.mp4`, `tomjerry.mp4`, and `security.mp4`. Download them locally with:

```bash
.venv/bin/python scripts/download_hf_space_examples.py
```

They are saved under `inputs/hf_space_autogaze/`, which is intentionally gitignored. The matching preset is `configs/repro/hf_space_autogaze_examples.yaml`. It records the Space settings: UI gazing ratio `0.75`, model gazing ratio `0.75 * 196 / 265`, task loss requirement `0.7`, 16-frame temporal chunks, 224x224 spatial chunks, and spatial batch size `2`.

To run NVILA on the default Space example video (`doorbell.mp4`) with fixed total-frame sampling:

```bash
.venv/bin/python -m repro.nvila_runner \
  --preset-config configs/repro/hf_space_autogaze_examples.yaml
```

To use another Space example while keeping the same NVILA/AutoGaze settings:

```bash
.venv/bin/python -m repro.nvila_runner \
  --preset-config configs/repro/hf_space_autogaze_examples.yaml \
  --video inputs/hf_space_autogaze/security.mp4 \
  --output-json outputs/autogaze_repro/hf_space_security_nvila_single.json
```

For an HLVid-like stress check on a Space example, override the total sampled frame count:

```bash
.venv/bin/python -m repro.nvila_runner \
  --preset-config configs/repro/hf_space_autogaze_examples.yaml \
  --video inputs/hf_space_autogaze/security.mp4 \
  --num-video-frames 1024 \
  --output-json outputs/autogaze_repro/hf_space_security_nvila_1024f.json
```

## NVILA Memory Preflight

Before running a long-form or high-resolution video through NVILA, run preflight mode. It does not load the 8B model. It reads local video metadata, mirrors NVILA's dynamic tiling estimate, and reports tile sequence counts, keep-all visual tokens, and a lower-bound CPU preprocessing memory estimate for the current public processor path.

For a local video:

```bash
.venv/bin/python -m repro.nvila_runner \
  --mode preflight \
  --video inputs/hf_space_autogaze/security.mp4 \
  --num-video-frames 1024 \
  --num-video-frames-thumbnail 64 \
  --max-tiles-video 48 \
  --preflight-json outputs/autogaze_repro/preflight_space_security_1024.json
```

For an HLVid-like 4K/5-minute estimate before downloading a specific video:

```bash
.venv/bin/python -m repro.nvila_runner \
  --mode preflight \
  --video hlvid_4k_virtual.mp4 \
  --preflight-width 3840 \
  --preflight-height 2160 \
  --preflight-source-frames 9000 \
  --num-video-frames 1024 \
  --num-video-frames-thumbnail 64 \
  --max-tiles-video 48 \
  --preflight-json outputs/autogaze_repro/preflight_4k_1024_virtual.json
```

On the 4K/1024-frame estimate, the current public processor path reports about `45` spatial tiles, `2880` tile sequences, about `5.44M` keep-all visual tokens, and about `202 GiB` lower-bound CPU preprocessing memory before Python/PIL overhead. Treat that as a signal to reduce `--num-video-frames`/`--max-tiles-video` or implement chunked preprocessing and vision encoding before attempting full generation.

## HLVid Example AutoGaze-Only Smoke

The default NVILA runner example video is `example/clip_av_video_5_001.mp4` from `bfshi/HLVid`. It is a 4K, about 5-minute MP4 and is about 1.7 GB, so local download needs enough free disk space.

To download it with resume support:

```bash
.venv/bin/python scripts/download_hlvid_example_video.py
```

The file is saved to `inputs/hlvid_example/clip_av_video_5_001.mp4`. If the download is interrupted, rerun the same command and it will request the remaining byte range.

When disk space is tight, the AutoGaze-only smoke can stream the remote URL directly. This does not run NVILA or SigLIP. It mirrors the NVILA video sampling shape by sampling `128` frames uniformly across the full video, splitting them into 16-frame AutoGaze chunks, applying 4K dynamic spatial tiling with `max_tiles_video=48`, and processing `64` thumbnail frames using NVILA's thumbnail subsampling policy.

```bash
.venv/bin/python -m repro.hlvid_example_autogaze \
  --device mps \
  --num-video-frames 128 \
  --num-video-frames-thumbnail 64 \
  --max-tiles-video 48 \
  --max-batch-size-autogaze 16 \
  --output-json outputs/autogaze_repro/hlvid_example_autogaze_only_128f.json
```

To use the downloaded file instead of remote streaming:

```bash
.venv/bin/python -m repro.hlvid_example_autogaze \
  --video inputs/hlvid_example/clip_av_video_5_001.mp4 \
  --device cuda \
  --dtype float16 \
  --num-video-frames 128 \
  --num-video-frames-thumbnail 64 \
  --max-tiles-video 48 \
  --output-json outputs/autogaze_repro/hlvid_example_autogaze_only_128f_cuda.json
```

For the 4K HLVid example, `max_tiles_video=48` resolves to `9x5=45` spatial tiles. With 128 sampled frames and 16-frame temporal chunks, this means `8 * 45 = 360` AutoGaze tile sequences. This smoke is useful for validating AutoGaze sampling, chunking, tiling, thumbnail handling, and token reduction before loading NVILA-8B.

## HLVid Manifest

```bash
.venv/bin/python -m repro.hlvid manifest \
  --config default \
  --split test \
  --output data/hlvid/manifest_test.json
```

The manifest command uses the Hugging Face Dataset Viewer API rather than `datasets.load_dataset`, so it can collect metadata without downloading the full video payload. The Hugging Face dataset card currently exposes the `test` split with 268 rows and about 152 GB of files.

Small metadata check:

```bash
.venv/bin/python -m repro.hlvid manifest \
  --config default \
  --split test \
  --limit 5 \
  --output data/hlvid/manifest_test_5.json
```

## HLVid Dry Run

```bash
.venv/bin/python -m repro.nvila_runner \
  --mode hlvid \
  --device cuda \
  --config default \
  --split test \
  --limit 1 \
  --num-video-frames 128 \
  --num-video-frames-thumbnail 64 \
  --max-tiles-video 48 \
  --measure-ttft \
  --resume \
  --predictions outputs/autogaze_repro/hlvid_dry_run_predictions.jsonl \
  --summary outputs/autogaze_repro/hlvid_dry_run_summary.json \
  --scored-predictions outputs/autogaze_repro/hlvid_dry_run_scored.jsonl
```

## HLVid Paper-Facing Run

The preset for this fixed total-frame sampling setup is `configs/repro/hlvid_like_nvila_1024.yaml`.

```bash
.venv/bin/python -m repro.nvila_runner \
  --preset-config configs/repro/hlvid_like_nvila_1024.yaml
```

Equivalent expanded command:

```bash
.venv/bin/python -m repro.nvila_runner \
  --mode hlvid \
  --device cuda \
  --gazing-mode autogaze \
  --config default \
  --split test \
  --limit 268 \
  --num-video-frames 1024 \
  --num-video-frames-thumbnail 64 \
  --max-tiles-video 48 \
  --measure-ttft \
  --resume \
  --predictions outputs/autogaze_repro/hlvid_full_predictions.jsonl \
  --summary outputs/autogaze_repro/hlvid_full_summary.json \
  --scored-predictions outputs/autogaze_repro/hlvid_full_scored.jsonl
```

For the matching keep-all baseline, keep every setting identical and change only the gaze mode and output paths:

```bash
.venv/bin/python -m repro.nvila_runner \
  --mode hlvid \
  --device cuda \
  --gazing-mode keep-all \
  --config default \
  --split test \
  --limit 268 \
  --num-video-frames 1024 \
  --num-video-frames-thumbnail 64 \
  --max-tiles-video 48 \
  --measure-ttft \
  --resume \
  --predictions outputs/autogaze_repro/hlvid_full_keep_all_predictions.jsonl \
  --summary outputs/autogaze_repro/hlvid_full_keep_all_summary.json \
  --scored-predictions outputs/autogaze_repro/hlvid_full_keep_all_scored.jsonl
```

Compare `accuracy_scored` with the project-page HLVid target of `52.6` for NVILA-8B-HD-Video. Report skipped, failed, and parse-failed samples separately. For the AutoGaze-vs-keep-all claim, compare accuracy together with median or mean `total_ms`, `video_decode_ms`, `video_tiling_ms`, `autogaze_forward_ms`, `vision_encoder_ms`, `llm_forward_ms`, `token_metrics.encoder_token_reduction_ratio`, and `token_metrics.llm_visual_token_reduction_ratio`. The paper-facing setup is NVILA-8B-HD-Video with up to 1024 frames and maximum resolution 3584 where the target GPU permits it.

## Verification

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/python -m repro.autogaze_bench --help
.venv/bin/python -m repro.hlvid --help
.venv/bin/python -m repro.nvila_runner --help
.venv/bin/python -m repro.report --help
```
