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

Compare `accuracy_scored` with the project-page HLVid target of `52.6` for NVILA-8B-HD-Video. Report skipped, failed, and parse-failed samples separately. The paper-facing setup is NVILA-8B-HD-Video with up to 1024 frames and maximum resolution 3584 where the target GPU permits it.

## Verification

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/python -m repro.autogaze_bench --help
.venv/bin/python -m repro.hlvid --help
.venv/bin/python -m repro.nvila_runner --help
.venv/bin/python -m repro.report --help
```
