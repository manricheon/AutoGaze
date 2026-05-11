from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from autogaze_ext.data import DummyActionRecognitionDataset, DummyVideoVQADataset, FrameSampler
from autogaze_ext.models import AutoGazeWrapper
from autogaze_ext.models.decoders import DummyActionRecognitionDecoder
from autogaze_ext.models.mllm import GenericMLLMAdapter
from autogaze_ext.models.vision import GenericViTAdapter
from autogaze_ext.pipeline.runner import DEFAULT_CONFIG_DIR, load_config


@dataclass(frozen=True)
class InferenceResult:
    task_type: str
    logs: dict[str, Any]
    outputs: dict[str, Any]


def _metadata_to_plain(metadata: Any) -> dict[str, Any]:
    if metadata is None:
        return {}
    plain: dict[str, Any] = {}
    for key, value in dict(metadata).items():
        if isinstance(value, torch.Tensor):
            plain[key] = value.detach().cpu().tolist()
        else:
            plain[key] = value
    return plain


def _first_sample_indices(metadata: dict[str, Any]) -> list[int]:
    indices = metadata.get("original_frame_indices", metadata.get("frame_indices", []))
    if isinstance(indices, torch.Tensor):
        indices = indices.detach().cpu().tolist()
    if indices and isinstance(indices[0], torch.Tensor):
        return [int(x[0].item()) for x in indices]
    if indices and isinstance(indices[0], list):
        return [int(x) for x in indices[0]]
    return [int(x) for x in indices]


def build_frame_sampler(cfg: DictConfig) -> FrameSampler:
    sampler_cfg = cfg.data.dummy_video.get("sampler", {})
    mode = sampler_cfg.get("mode", "fixed")
    if mode == "max":
        return FrameSampler(mode="max", max_frames=int(sampler_cfg.get("max_frames", cfg.data.dummy_video.frames)))
    return FrameSampler(mode="fixed", num_frames=int(sampler_cfg.get("num_frames", cfg.data.dummy_video.frames)))


def build_dataset(cfg: DictConfig):
    frame_sampler = build_frame_sampler(cfg)
    common = {
        "num_samples": int(cfg.data.dummy_video.num_samples),
        "total_frames": int(cfg.data.dummy_video.frames),
        "height": int(cfg.data.dummy_video.height),
        "width": int(cfg.data.dummy_video.width),
        "frame_sampler": frame_sampler,
    }
    if cfg.task.type == "video_vqa":
        return DummyVideoVQADataset(**common)
    if cfg.task.type == "action_recognition":
        return DummyActionRecognitionDataset(**common)
    raise ValueError(f"Unsupported dummy task type: {cfg.task.type}")


def run_inference(cfg: DictConfig, batch_size: int = 2) -> InferenceResult:
    if bool(cfg.model.autogaze.enabled):
        raise NotImplementedError("Minimal dummy pipeline supports AutoGaze OFF mode only")

    dataset = build_dataset(cfg)
    batch = next(iter(DataLoader(dataset, batch_size=batch_size)))
    video = batch["video"]
    metadata = _metadata_to_plain(batch.get("metadata", {}))

    vision = GenericViTAdapter(mode="dummy")
    vision_output = vision(video, metadata=metadata)

    autogaze = AutoGazeWrapper(enabled=False)
    autogaze_output = autogaze(visual_tokens=vision_output.visual_tokens, metadata=vision_output.metadata)

    selected_tokens = vision_output.visual_tokens
    logs: dict[str, Any] = {
        "input video shape": tuple(video.shape),
        "sampled frame indices": _first_sample_indices(metadata),
        "AutoGaze": "ON" if autogaze.enabled else "OFF",
        "visual token count before AutoGaze": autogaze_output.metadata["original_visual_token_count"],
        "visual token count after AutoGaze": autogaze_output.metadata["selected_visual_token_count"],
    }

    if cfg.task.type == "video_vqa":
        questions = list(batch["question"])
        mllm = GenericMLLMAdapter(mode="dummy", answer="dummy")
        mllm_output = mllm.generate(selected_tokens, questions=questions, metadata=autogaze_output.metadata)
        logs["generated dummy answer"] = mllm_output.generated_text
        return InferenceResult(
            task_type="video_vqa",
            logs=logs,
            outputs={"generated_text": mllm_output.generated_text, "metadata": mllm_output.metadata},
        )

    decoder = DummyActionRecognitionDecoder()
    decoder_output = decoder(selected_tokens, metadata=autogaze_output.metadata)
    logs["output shape"] = tuple(decoder_output.logits.shape)
    return InferenceResult(
        task_type="action_recognition",
        logs=logs,
        outputs={
            "logits": decoder_output.logits,
            "predicted_labels": decoder_output.predicted_labels,
            "metadata": decoder_output.metadata,
        },
    )


def print_inference_result(result: InferenceResult) -> None:
    print(f"task type: {result.task_type}")
    for key, value in result.logs.items():
        print(f"{key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run checkpoint-free dummy AutoGaze extension inference")
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--config-name", default="dummy_video_vqa")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    cfg = load_config(Path(args.config_dir), args.config_name)
    result = run_inference(cfg, batch_size=args.batch_size)
    print_inference_result(result)


if __name__ == "__main__":
    main()
