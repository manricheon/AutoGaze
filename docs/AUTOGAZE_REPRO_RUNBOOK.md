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
.venv/bin/python -m pip install -e external/AutoGaze --no-deps --no-build-isolation
```

The official AutoGaze `pyproject.toml` currently includes `flash_attn` as a normal dependency. On Apple MPS/macOS, use `--no-deps --no-build-isolation` for the editable install and rely on `requirements-repro.txt` for the MPS-compatible runtime dependencies.

If `torch.backends.mps.is_available()` is false inside a sandbox but true outside it, run the actual MPS benchmark outside the sandbox. The helper tests and CLI checks do not require MPS.

## MPS AutoGaze And SigLIP Smoke Benchmark

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

Observed local smoke result on this workspace:

- AutoGaze revision: `ba48d0f94ac2929d6fe3ee4380dc893aa6eed0ab`
- Input: official `external/AutoGaze/assets/example_input.mp4`, 16 frames, 224x224
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
