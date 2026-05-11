from __future__ import annotations

import json
import os
import sys
import types

from autogaze_ext.data import HFDatasetLoader
from autogaze_ext.metrics import HFEvaluateMetric
from autogaze_ext.models.huggingface import HFModelLoader, HFProcessorLoader
from autogaze_ext.utils import HFLoadConfig, hf_offline_env, hf_offline_mode, redacted_hf_config


class _FakeAuto:
    calls = []

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        cls.calls.append((model_id, kwargs))
        return {"model_id": model_id, "kwargs": kwargs, "class": cls.__name__}


class _FakeAutoModel(_FakeAuto):
    pass


class _FakeAutoProcessor(_FakeAuto):
    pass


class _FakeAutoTokenizer(_FakeAuto):
    pass


def test_hf_model_loader_uses_transformers_mock_without_logging_token(monkeypatch) -> None:
    fake_transformers = types.SimpleNamespace(
        AutoModel=_FakeAutoModel,
        AutoModelForCausalLM=_FakeAutoModel,
        AutoModelForVision2Seq=_FakeAutoModel,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setenv("HF_TOKEN_FOR_TEST", "secret-token")

    loader = HFModelLoader(
        {
            "model_id": "local-model",
            "revision": "abc123",
            "token_env_var": "HF_TOKEN_FOR_TEST",
            "local_files_only": True,
            "model_class": "AutoModel",
        }
    )
    model = loader.load_model()

    assert model["model_id"] == "local-model"
    assert model["kwargs"]["token"] == "secret-token"
    assert loader.last_load_info["token_present"] is True
    assert "secret-token" not in json.dumps(loader.last_load_info)


def test_hf_processor_and_tokenizer_loader_with_mock(monkeypatch) -> None:
    fake_transformers = types.SimpleNamespace(
        AutoProcessor=_FakeAutoProcessor,
        AutoTokenizer=_FakeAutoTokenizer,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    loader = HFProcessorLoader({"model_id": "processor-model", "local_files_only": True})

    processor = loader.load_processor()
    tokenizer = loader.load_tokenizer()

    assert processor["class"] == "_FakeAutoProcessor"
    assert tokenizer["class"] == "_FakeAutoTokenizer"


def test_local_json_jsonl_csv_dataset_loading_and_field_mapping(tmp_path) -> None:
    json_path = tmp_path / "data.json"
    json_path.write_text(json.dumps([{"video_path": "a.mp4", "q": "Q", "a": "A"}]), encoding="utf-8")
    jsonl_path = tmp_path / "data.jsonl"
    jsonl_path.write_text('{"video_path": "b.mp4", "q": "Q2", "a": "A2"}\n', encoding="utf-8")
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("video_path,q,a\nc.mp4,Q3,A3\n", encoding="utf-8")

    cfg = {"field_mapping": {"video": "video_path", "question": "q", "answer": "a"}, "max_samples": 1}
    for path in [json_path, jsonl_path, csv_path]:
        dataset = HFDatasetLoader(cfg).load_dataset(str(path))
        assert len(dataset) == 1
        assert dataset[0]["video"] in {"a.mp4", "b.mp4", "c.mp4"}
        assert "question" in dataset[0]
        assert "answer" in dataset[0]


def test_hf_evaluate_wrapper_fallback_compute() -> None:
    metric = HFEvaluateMetric(
        "__missing_metric__",
        fallback_compute=lambda predictions, references: {"exact_match": float(predictions == references)},
    )
    metric.add_batch(predictions=["a"], references=["a"])

    result = metric.compute()

    assert result["metric_source"] == "internal_fallback"
    assert result["hf_evaluate_available"] is False
    assert result["exact_match"] == 1.0


def test_offline_env_helpers_restore_environment(monkeypatch) -> None:
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    assert hf_offline_env(True)["HF_HUB_OFFLINE"] == "1"

    with hf_offline_mode(True):
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
        assert os.environ["HF_DATASETS_OFFLINE"] == "1"

    assert "HF_HUB_OFFLINE" not in os.environ


def test_redacted_config_never_contains_token(monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN_FOR_TEST", "secret-token")
    cfg = HFLoadConfig(model_id="m", token_env_var="HF_TOKEN_FOR_TEST")
    redacted = redacted_hf_config(cfg)

    assert redacted["token_present"] is True
    assert "secret-token" not in json.dumps(redacted)
