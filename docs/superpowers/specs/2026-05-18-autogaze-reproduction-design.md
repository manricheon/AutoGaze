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
- HLVid benchmark: `bfshi/HLVid`

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

Phase 3 adds HLVid benchmark reproduction:

- Download or stream `bfshi/HLVid` metadata and videos using the Hugging Face dataset layout.
- Preserve the dataset split, `question_id`, `category`, `video_path`, `question`, and `answer` fields in every prediction artifact.
- Run a small local dry run that validates dataset loading, prompt formatting, answer parsing, and scoring without downloading the full 152 GB dataset.
- Run the CUDA benchmark with `nvidia/NVILA-8B-HD-Video` using the paper-facing setup: up to 1024 frames and maximum resolution 3584 where the target GPU permits it.
- Compare against the paper-reported HLVid result for NVILA-8B-HD-Video, 52.6 on the test set, and the reported +10.1 improvement over NVILA-8B-Video.
- Record both accuracy and efficiency: exact-match multiple-choice score, per-sample latency, AutoGaze token reduction, vision encoder time, MLLM prefill time, decode time, total time, and failure reasons.

## Non-Goals

This effort does not retrain AutoGaze. It does not modify AutoGaze architecture or claim paper-level speedups from MPS measurements. It does not make local MPS results stand in for CUDA latency claims. It also does not attempt VideoMME or other full benchmark score reproduction before the efficiency, NVILA, and HLVid execution paths are stable.

## Architecture

The reproduction workspace will keep official code and local harness code separate:

- repository root: official `NVlabs/AutoGaze` hierarchy, including `autogaze/`, `assets/`, and upstream docs, so official examples and NVILA remote code can resolve `import autogaze` without custom path shims.
- `external/VILA/`: clean clone of `NVlabs/VILA` if the NVILA runner requires VILA evaluation or inference entrypoints.
- `repro/`: local scripts, benchmark runners, report helpers, and environment checks.
- `data/hlvid/`: HLVid metadata, manifest files, and user-managed video cache pointers.
- `outputs/autogaze_repro/`: generated benchmark outputs, logs, CSV files, JSON files, prediction files, score summaries, and optional visualizations.
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

## HLVid Benchmark Design

HLVid is treated as a first-class benchmark, not an optional appendix, because the paper highlights it as the high-resolution long-form setting where AutoGaze changes the reachable operating point. The official project page describes HLVid as high-resolution, long-form video QA with 5-minute 4K videos. The Hugging Face dataset card exposes a `test` split with 268 rows and about 152 GB of files. The project page reports NVILA-8B-HD-Video at 1024 frames, maximum resolution 3584, and HLVid score 52.6, which is +10.1 over NVILA-8B-Video.

The benchmark runner will have three modes:

1. `--mode manifest`: load the Hugging Face dataset metadata, produce a pinned manifest, and verify required columns.
2. `--mode dry-run`: run one to five samples using local or already-cached videos to validate prompt formatting, answer extraction, scoring, logging, and resumability.
3. `--mode full`: run the selected split on CUDA with NVILA-8B-HD-Video and AutoGaze-enabled preprocessing, writing resumable per-sample predictions.

Each HLVid prediction row records:

- `question_id`, `category`, `video_path`, `question`, `answer`, raw model output, parsed answer, correctness, and parse status.
- Model id, model revision, AutoGaze revision, VILA revision, dataset revision, frame count, resolution cap, dtype, device, and seed.
- AutoGaze selected patches, padded patches, token reduction ratio, vision encoder time, MLLM prefill time, decode time, total sample time, generated token count, and peak CUDA memory where available.

The scoring rule defaults to direct multiple-choice exact match because the dataset questions request the answer letter directly. The implementation will also preserve raw outputs so scoring can be audited if a model emits extra text around the answer letter.

## Validation Strategy

Local MPS validation must prove that:

- AutoGaze imports from the repository-root official hierarchy.
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

HLVid validation must prove that:

- The dataset manifest can be generated from `bfshi/HLVid` without losing sample ids or labels.
- Dry-run scoring produces deterministic records for cached samples.
- The full CUDA run is resumable and can restart from completed prediction files without overwriting prior results.
- Accuracy is reported as total correct divided by total scored samples, with skipped, failed, and parse-failed samples separated from the denominator unless explicitly included.
- The final HLVid report compares local CUDA results against the paper-facing targets: NVILA-8B-HD-Video with 1024 frames, maximum resolution 3584, HLVid score 52.6, and +10.1 over NVILA-8B-Video.

## Reporting

The final local report will separate evidence into three buckets:

- Confirmed locally on MPS: runnable code path, tensor contracts, token reduction, and smoke-test latency.
- Prepared for CUDA: scripts, command lines, environment checks, and expected output schema.
- Prepared for HLVid: dataset manifest, dry-run predictions, scoring script, full-run command, and output schema.
- Pending CUDA execution: paper-comparable latency speedups, NVILA end-to-end throughput, and HLVid quality metrics.

This distinction is central to leader-facing communication: MPS proves the method and harness are real; CUDA proves the performance claim.

## Risks

The main technical risk is dependency drift in a newly released official repository. The mitigation is to pin the official commit hash and record package versions in every benchmark output. Another risk is that MPS may lack kernel support for a dependency or model path. The mitigation is to keep MPS validation focused on AutoGaze and compatible SigLIP, while treating NVILA end-to-end as CUDA-only for performance claims. A third risk is model or dataset download size. The mitigation is to start with AutoGaze and SigLIP only, then add NVILA once the smaller harness is stable. HLVid adds a larger data risk because the Hugging Face dataset card lists about 152 GB of files; the mitigation is to separate metadata manifest generation and dry-run scoring from full video download and CUDA execution.

## Acceptance Criteria

The design is complete when the implementation can produce:

- A clean official AutoGaze hierarchy at repository root.
- A local environment setup path that does not depend on older local experiments.
- A successful AutoGaze smoke run on MPS.
- A JSON and CSV benchmark result containing token reduction and latency fields.
- A SigLIP baseline versus gazed SigLIP comparison on MPS.
- CUDA-ready benchmark commands documented with the same output schema.
- NVILA integration notes and runner entrypoint sufficient for execution on a CUDA machine.
- An HLVid manifest and dry-run path that verifies dataset loading, prompt formatting, answer parsing, and scoring.
- A full HLVid CUDA benchmark command that records score, latency, memory, token reduction, and per-sample failures.
