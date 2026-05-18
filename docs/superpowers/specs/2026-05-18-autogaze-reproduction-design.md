# AutoGaze Reproduction Design

## Goal

Rebuild the AutoGaze reproduction workspace from a clean official baseline, verify that the AutoGaze efficiency path runs on the local Apple MPS environment, and prepare the same code path for CUDA validation with NVILA-8B-HD-Video on a GPU machine.

The first deliverable is a reproducible benchmark harness that measures token reduction and latency for AutoGaze and the AutoGaze-compatible SigLIP vision encoder. The second deliverable is a CUDA-ready NVILA integration path that can be tested on a GPU machine without redesigning the local workflow.

## Official Baseline

The work uses the official public artifacts as the source of truth:

- Code: `https://github.com/NVlabs/AutoGaze`
- Project page: `https://autogaze.github.io/`
- Paper: arXiv `2603.12254`, "Attend Before Attention: Efficient and Scalable Video Understanding via Autoregressive Gazing"
- Model and benchmark collection: `https://huggingface.co/collections/bfshi/autogaze`
- AutoGaze model: `nvidia/AutoGaze`
- NVILA HD model: `nvidia/NVILA-8B-HD-Video`

The local workspace starts cleanly under `/Users/mrc/Documents/New project`. Existing untracked content such as `outputs/` is treated as user-owned and is not modified by this reproduction setup.

## Scope

Phase 1 validates the efficiency mechanism locally on MPS:

- Clone the official AutoGaze repository into a fresh directory.
- Install dependencies in an isolated environment without relying on prior local experiments.
- Run an AutoGaze smoke test on synthetic or sample video input.
- Measure AutoGaze latency, selected patch count, padded patch count, effective visual token count, and token reduction ratio.
- Measure compatible SigLIP baseline latency against AutoGaze-gazed SigLIP latency using the same video input and preprocessing assumptions.
- Save machine-readable results as JSON and CSV so the numbers can be compared across MPS and CUDA.

Phase 2 prepares NVILA integration for CUDA:

- Inspect and document the official NVILA-HD-Video entry path from AutoGaze/VILA instructions.
- Add a runner or wrapper that records the same benchmark metadata where local hardware permits import/config validation.
- Keep end-to-end NVILA latency and quality claims gated behind CUDA execution because MPS is not a reliable platform for 8B video MLLM performance conclusions.

## Non-Goals

This effort does not retrain AutoGaze. It does not modify AutoGaze architecture or claim paper-level speedups from MPS measurements. It does not make local MPS results stand in for CUDA latency claims. It also does not attempt full HLVid or VideoMME score reproduction before the efficiency and NVILA execution path is stable.

## Architecture

The reproduction workspace will keep official code and local harness code separate:

- `external/AutoGaze/`: clean clone of `NVlabs/AutoGaze`.
- `repro/`: local scripts, benchmark runners, report helpers, and environment checks.
- `outputs/autogaze_repro/`: generated benchmark outputs, logs, CSV files, JSON files, and optional visualizations.
- `docs/`: design, implementation plan, reproduction notes, and CUDA handoff guidance.

The harness will use a small device abstraction so the same script can run on `mps`, `cuda`, or `cpu`. The MPS path is treated as smoke and correctness validation. The CUDA path is treated as the performance validation path.

## Benchmark Design

The primary benchmark compares three stages:

1. AutoGaze-only inference on 16-frame chunks.
2. SigLIP baseline vision encoding over all visual patches.
3. SigLIP vision encoding with `gazing_info` from AutoGaze.

Each run records:

- Device, dtype, Python version, PyTorch version, platform, git commit of the official AutoGaze clone, and model identifiers.
- Input video source, frame count, chunk count, input resolution, target scales, target patch size, `gazing_ratio`, and `task_loss_requirement`.
- Warmup count, measured repeat count, per-repeat latency, mean latency, median latency, and peak memory if available on the device.
- Raw patch budget, selected non-padded patches, padded gazing positions, effective token reduction ratio, and the output hidden-state shape for SigLIP.

Default parameters follow the official quick start where possible:

- `gazing_ratio=0.75`
- `task_loss_requirement=0.7`
- 16-frame chunks
- 224x224 AutoGaze input for the initial smoke test
- SigLIP model `google/siglip2-base-patch16-224` for the first local comparison

After the initial smoke test, the same harness can add higher-resolution target scales such as the official 392 input example for SigLIP2 SO400M, but that is secondary to establishing the baseline reproduction path.

## Validation Strategy

Local MPS validation must prove that:

- AutoGaze imports from the fresh official clone.
- The pretrained AutoGaze model and image processor load from Hugging Face.
- A 16-frame input produces `gazing_pos`, `if_padded_gazing`, and `num_gazing_each_frame`.
- The harness computes non-padded selected patches and token reduction ratio from model outputs.
- The customized SigLIP path accepts `gazing_info` and produces a hidden state with padded positions included.
- Baseline and gazed SigLIP timings are collected with identical warmup and repeat rules.

CUDA validation on the GPU machine must prove that:

- The same benchmark command runs with `--device cuda`.
- CUDA synchronization is used around measured regions.
- CUDA memory statistics are recorded.
- NVILA-8B-HD-Video can run with AutoGaze-enabled preprocessing on the target GPU machine.
- Any reported speedup clearly states hardware, input size, model, precision, warmup, repeat count, and whether AutoGaze overhead is included.

## Reporting

The final local report will separate evidence into three buckets:

- Confirmed locally on MPS: runnable code path, tensor contracts, token reduction, and smoke-test latency.
- Prepared for CUDA: scripts, command lines, environment checks, and expected output schema.
- Pending CUDA execution: paper-comparable latency speedups and NVILA end-to-end throughput or quality metrics.

This distinction is central to leader-facing communication: MPS proves the method and harness are real; CUDA proves the performance claim.

## Risks

The main technical risk is dependency drift in a newly released official repository. The mitigation is to pin the official commit hash and record package versions in every benchmark output. Another risk is that MPS may lack kernel support for a dependency or model path. The mitigation is to keep MPS validation focused on AutoGaze and compatible SigLIP, while treating NVILA end-to-end as CUDA-only for performance claims. A third risk is model or dataset download size. The mitigation is to start with AutoGaze and SigLIP only, then add NVILA once the smaller harness is stable.

## Acceptance Criteria

The design is complete when the implementation can produce:

- A clean official AutoGaze checkout.
- A local environment setup path that does not depend on older local experiments.
- A successful AutoGaze smoke run on MPS.
- A JSON and CSV benchmark result containing token reduction and latency fields.
- A SigLIP baseline versus gazed SigLIP comparison on MPS.
- CUDA-ready benchmark commands documented with the same output schema.
- NVILA integration notes and runner entrypoint sufficient for execution on a CUDA machine.
