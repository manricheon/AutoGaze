# Generic AutoGaze MLLM Integration Guide

This guide covers the first-mile PoC path for applying AutoGaze to an arbitrary Hugging Face MLLM that contains a ViT-like visual encoder.

## Why This Exists

The project needs a reproducible way to test AutoGaze ON/OFF before writing a custom integration for every model. The new `generic_mllm` runner provides that path through hook mode:

```text
video frames -> AutoGaze -> 14x14 gaze map
                         -> resize/tile to target patch grid
target MLLM vision module -> forward hook zeros non-gazed patch tokens
                         -> normal MLLM generation
```

This does not reduce ViT sequence length. It is for compatibility and quality checks. Use model-specific `full` or `native` runners for latency/VRAM claims.

## Integration Combinations

| Path | Works for arbitrary MLLM? | Code changes | Speed benefit | Purpose |
| :--- | :---: | :---: | :---: | :--- |
| AutoGaze OFF | yes | none | baseline | Full-patch quality reference |
| Generic hook | often | no model code; needs module path | no | First-mile ON/OFF PoC |
| Model-specific hook | yes, after runner work | runner code | no | Stable repeated benchmarking |
| Model-specific full | no, per ViT | ViT forward modification | yes | Real efficiency benchmark |
| Native | no, model-specific | processor/model integration | yes | Production-style path |

## Required Options

`generic_mllm` needs the user to identify where patch tokens appear in the model:

- `--generic-vision-hook`: dotted module path to hook.
- `--generic-patch-grid`: patch grid side length for one frame or tile.
- `--generic-media-key`: whether the processor expects `images=` or `videos=`.

Optional:

- `--generic-has-cls-token`: preserve a leading CLS token.
- `--generic-processor-path`: use a different processor than `--model-path`.
- `--generic-prompt-template`: wrap benchmark prompts, e.g. `"<image>\n{prompt}"`.

## Benchmark Example

```bash
python -m autogaze.eval.run_benchmark \
  --task videomme \
  --mllm generic_mllm \
  --model-path <hf-or-local-mllm> \
  --autogaze-path weights/AutoGaze \
  --gazing-ratio 0.75 \
  --generic-vision-hook vision_model.embeddings \
  --generic-patch-grid 14 \
  --generic-media-key images \
  --max-samples 5
```

Baseline:

```bash
python -m autogaze.eval.run_benchmark \
  --task videomme \
  --mllm generic_mllm \
  --model-path <hf-or-local-mllm> \
  --no-autogaze \
  --generic-vision-hook vision_model.embeddings \
  --generic-patch-grid 14 \
  --generic-media-key images \
  --max-samples 5
```

## Full Pipeline Example

```bash
python autogaze/infer_full.py assets/example_input.mp4 \
  --mllm generic_mllm \
  --model-path <hf-or-local-mllm> \
  --autogaze-path weights/AutoGaze \
  --generic-vision-hook vision_model.embeddings \
  --generic-patch-grid 14 \
  --generic-media-key images \
  --compare-autogaze
```

## How To Find The Hook Path

Inspect likely modules:

```python
for name, module in model.named_modules():
    lname = name.lower()
    if "embed" in lname or "patch" in lname or "vision" in lname:
        print(name, type(module).__name__)
```

Good hook candidates usually output a tensor shaped like:

- `(B, N, C)`
- `(N, C)`
- a tuple/list whose first element is one of those tensors

Avoid hooking after the LLM projection unless you only want to measure downstream text behavior; the best first hook is normally the visual patch embedding output.

## Limitations

- Hook mode zeroes tokens but does not shorten the sequence.
- Patch ordering must match row-major spatial order for the mask to be meaningful.
- Some processors require model-specific chat templates or video token strings; use `--generic-prompt-template`.
- Some MLLMs expose no standard `generate()` path through generic HF auto classes; those need model-specific runners.
