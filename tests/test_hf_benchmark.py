from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from autogaze_ext.pipeline.hf_benchmark import run_hf_benchmark
from autogaze_ext.pipeline.runner import load_config
from autogaze_ext.utils import SUPPORTED_HF_MODES


class FakeModelLoader:
    def __init__(self) -> None:
        self.calls = []

    def load_model(self, model_id):
        self.calls.append(model_id)
        return {"model_id": model_id}


class FakeProcessorLoader:
    def __init__(self) -> None:
        self.calls = []

    def load_processor(self, model_id):
        self.calls.append(model_id)
        return {"processor_id": model_id}


@pytest.mark.parametrize(
    "config_name",
    [
        "hf_benchmark/hf_model_only",
        "hf_benchmark/hf_dataset_only",
        "hf_benchmark/hf_model_and_dataset",
        "hf_benchmark/local_model_hf_dataset",
        "hf_benchmark/hf_model_local_dataset",
        "hf_benchmark/offline_hf_cache",
    ],
)
def test_hf_benchmark_config_examples_dry_run(config_name: str, tmp_path: Path) -> None:
    cfg = load_config(config_name=config_name)
    path = run_hf_benchmark(cfg, output_dir=tmp_path)
    data = json.loads(path.read_text())

    assert data["mode"] in SUPPORTED_HF_MODES
    assert data["integration_mode"] == "official_processor"
    assert data["dry_run"] is True
    assert data["metadata"]["model_loaded"] is False
    assert data["metadata"]["processor_loaded"] is False
    assert "secret" not in json.dumps(data["metadata"]["redacted_hf_config"]).lower()


def test_hf_dataset_only_local_file_smoke(tmp_path: Path) -> None:
    data_path = tmp_path / "samples.jsonl"
    data_path.write_text(
        '{"video": "a.mp4", "question": "q1", "answer": "yes"}\n'
        '{"video": "b.mp4", "question": "q2", "answer": "no"}\n',
        encoding="utf-8",
    )
    cfg = load_config(config_name="hf_benchmark/hf_dataset_only")
    cfg = OmegaConf.merge(
        cfg,
        {
            "benchmark": {"huggingface": {"dry_run": False}},
            "data": {
                "huggingface": {
                    "dataset_id": str(data_path),
                    "dataset_split": "validation",
                    "max_samples": 1,
                }
            },
        },
    )

    path = run_hf_benchmark(cfg, output_dir=tmp_path)
    data = json.loads(path.read_text())

    assert data["dataset_id"] == str(data_path)
    assert data["dataset_split"] == "validation"
    assert data["evaluated_samples"] == 1
    assert data["metric_implementation_source"] in {"internal_fallback", "huggingface_evaluate"}
    assert data["metric_result"]["num_samples"] == 1


def test_hf_model_only_uses_mock_loaders(tmp_path: Path) -> None:
    cfg = load_config(config_name="hf_benchmark/hf_model_only")
    cfg = OmegaConf.merge(
        cfg,
        {
            "benchmark": {"huggingface": {"dry_run": False}},
            "model": {
                "huggingface": {
                    "model_id": "local/tiny-model",
                    "revision": "abc123",
                    "local_files_only": True,
                    "trust_remote_code": False,
                }
            },
        },
    )
    model_loader = FakeModelLoader()
    processor_loader = FakeProcessorLoader()

    path = run_hf_benchmark(
        cfg,
        output_dir=tmp_path,
        model_loader=model_loader,
        processor_loader=processor_loader,
    )
    data = json.loads(path.read_text())

    assert model_loader.calls == ["local/tiny-model"]
    assert processor_loader.calls == ["local/tiny-model"]
    assert data["model_id"] == "local/tiny-model"
    assert data["model_revision"] == "abc123"
    assert data["processor_tokenizer_id"] == "local/tiny-model"
    assert data["metadata"]["model_loaded"] is True
    assert data["metadata"]["processor_loaded"] is True


def test_offline_hf_cache_report_flags(tmp_path: Path) -> None:
    cfg = load_config(config_name="hf_benchmark/offline_hf_cache")
    path = run_hf_benchmark(cfg, output_dir=tmp_path)
    data = json.loads(path.read_text())

    assert data["mode"] == "offline_hf_cache"
    assert data["cache_mode"] == "offline_hf_cache"
    assert data["offline_mode"] is True
    assert data["local_files_only"] is True
