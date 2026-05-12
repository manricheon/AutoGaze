from __future__ import annotations

import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from autogaze_ext.pipeline.hf_official_processor_baseline import run_hf_official_processor_baseline
from autogaze_ext.pipeline.runner import load_config


class FakeBatch(dict):
    def to(self, device: str) -> "FakeBatch":
        return FakeBatch({key: value.to(device) if hasattr(value, "to") else value for key, value in self.items()})


class FakeTokenizer:
    video_token = "<video>"


class FakeProcessor:
    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if "videos" in kwargs:
            raise TypeError("fake processor only accepts images")
        return FakeBatch(
            {
                "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
                "pixel_values": torch.zeros(1, 3, 8, 8),
            }
        )

    def batch_decode(self, outputs, skip_special_tokens: bool = True):
        return ["mock generated answer"]


class FakeProcessorAcceptsVideoPath(FakeProcessor):
    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if "videos" not in kwargs:
            raise TypeError("fake processor expects videos")
        return FakeBatch({"input_ids": torch.tensor([[1, 2]], dtype=torch.long)})


class FakeModel:
    def __init__(self) -> None:
        self.eval_called = False

    def to(self, device: str):
        return self

    def eval(self) -> None:
        self.eval_called = True

    def generate(self, **kwargs):
        return torch.tensor([[1, 2, 3, 4]], dtype=torch.long)


class FakeModelLoader:
    def __init__(self, model: FakeModel | None = None) -> None:
        self.model = model or FakeModel()
        self.calls: list[dict[str, object]] = []

    def load_model(self, model_id, revision=None, device=None, dtype=None):
        self.calls.append({"model_id": model_id, "revision": revision, "device": device, "dtype": dtype})
        return self.model


class FakeProcessorLoader:
    def __init__(self, processor: FakeProcessor | None = None) -> None:
        self.processor = processor or FakeProcessor()
        self.calls: list[dict[str, object]] = []

    def load_processor(self, model_id, revision=None):
        self.calls.append({"model_id": model_id, "revision": revision})
        return self.processor


def test_hf_official_processor_baseline_config_loads() -> None:
    cfg = load_config(config_name="hf_benchmark/hf_official_processor_baseline")

    assert cfg.benchmark.huggingface.integration_mode == "official_processor"
    assert cfg.benchmark.huggingface.autogaze_token_injection is False
    assert cfg.model.huggingface.local_files_only is True


def test_hf_official_processor_baseline_dry_run_writes_report(tmp_path: Path) -> None:
    cfg = load_config(config_name="hf_benchmark/hf_official_processor_baseline")

    path = run_hf_official_processor_baseline(
        cfg,
        output_dir=tmp_path,
        dry_run=True,
        query_text="What happens?",
    )
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["integration_mode"] == "official_processor"
    assert data["autogaze_token_injection"] is False
    assert data["generation_status"] == "dry_run"
    assert data["processor_loaded"] is False
    assert data["model_loaded"] is False
    assert data["query_text"] == "What happens?"


def test_hf_official_processor_baseline_dummy_video_generation_with_mocks(tmp_path: Path) -> None:
    cfg = load_config(config_name="hf_benchmark/hf_official_processor_baseline")
    cfg = OmegaConf.merge(
        cfg,
        {
            "model": {
                "huggingface": {
                    "model_id": "local/mock-mllm",
                    "revision": "abc123",
                    "local_files_only": True,
                }
            }
        },
    )
    processor_loader = FakeProcessorLoader()
    model_loader = FakeModelLoader()

    path = run_hf_official_processor_baseline(
        cfg,
        output_dir=tmp_path,
        dry_run=False,
        query_text="What is in the video?",
        num_frames=2,
        resolution=32,
        processor_loader=processor_loader,
        model_loader=model_loader,
    )
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["model_id"] == "local/mock-mllm"
    assert data["revision"] == "abc123"
    assert data["processor_id"] == "local/mock-mllm"
    assert data["dataset_input_source"] == "local_dummy_video"
    assert data["generated_answer"] == "mock generated answer"
    assert data["generation_status"] == "generated"
    assert data["latency_ms"] >= 0
    assert data["peak_vram_mb"] == "N/A"
    assert data["metadata"]["processor_metadata"]["processor_attempt"] == "images_frames"
    assert "<video>" in processor_loader.processor.calls[-1]["text"]


def test_hf_official_processor_baseline_local_video_path_with_mocks(tmp_path: Path) -> None:
    cfg = load_config(config_name="hf_benchmark/hf_official_processor_baseline")
    cfg = OmegaConf.merge(cfg, {"model": {"huggingface": {"model_id": "local/mock-mllm"}}})
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"not a real video; fake processor only checks the path")
    processor_loader = FakeProcessorLoader(FakeProcessorAcceptsVideoPath())

    path = run_hf_official_processor_baseline(
        cfg,
        output_dir=tmp_path / "out",
        dry_run=False,
        video="path",
        video_path=str(video_path),
        query_text="Summarize.",
        processor_loader=processor_loader,
        model_loader=FakeModelLoader(),
    )
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["dataset_input_source"] == "local_video"
    assert data["video_path"] == str(video_path)
    assert data["generation_status"] == "generated"
    assert processor_loader.processor.calls[-1]["videos"] == str(video_path)
    assert data["autogaze_token_injection"] is False


def test_hf_official_processor_baseline_does_not_use_text_only_fallback(tmp_path: Path) -> None:
    class TextOnlyProcessor(FakeProcessor):
        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            raise TypeError("video input not supported")

    cfg = load_config(config_name="hf_benchmark/hf_official_processor_baseline")
    cfg = OmegaConf.merge(cfg, {"model": {"huggingface": {"model_id": "local/mock-mllm"}}})

    path = run_hf_official_processor_baseline(
        cfg,
        output_dir=tmp_path,
        dry_run=False,
        query_text="Do not ignore this query.",
        processor_loader=FakeProcessorLoader(TextOnlyProcessor()),
        model_loader=FakeModelLoader(),
    )
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["generation_status"] == "failed"
    assert "silently ignored" in data["skipped_reason"]
    assert data["query_text"] == "Do not ignore this query."
