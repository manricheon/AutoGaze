from __future__ import annotations

import json

import pytest
from omegaconf import OmegaConf

from autogaze_ext.data import HFDatasetLoader
from autogaze_ext.models.huggingface import HFModelLoader, HFProcessorLoader
from autogaze_ext.pipeline.runner import load_config
from autogaze_ext.utils import HFLoadConfig


def _hf_config_from_nodes(cfg) -> HFLoadConfig:
    runtime = dict(OmegaConf.to_container(cfg.runtime.huggingface, resolve=True))
    model = dict(OmegaConf.to_container(cfg.model.huggingface, resolve=True))
    merged = {**runtime, **model}
    return HFLoadConfig.from_mapping(merged)


def _skip_if_cache_miss(exc: Exception) -> None:
    message = str(exc)
    cache_only_markers = (
        "local_files_only",
        "offline mode",
        "couldn't find",
        "cannot find",
        "not the path to a directory",
        "does not appear to have",
        "We couldn't connect",
        "unrecognized processing class",
        "does not have a tokenizer",
    )
    if any(marker.lower() in message.lower() for marker in cache_only_markers):
        pytest.skip(
            "Optional HF smoke model or processor/tokenizer assets are not available in the local cache. "
            "The test uses local_files_only=true/offline=true and does not download models."
        )
    raise exc


def test_optional_hf_tiny_local_files_only_loader_smoke(monkeypatch) -> None:
    pytest.importorskip("transformers")
    monkeypatch.setenv("HF_TOKEN", "optional-smoke-secret-token")
    cfg = load_config(config_name="hf_smoke/tiny_local_files_only")
    hf_config = _hf_config_from_nodes(cfg)

    assert hf_config.model_id == "hf-internal-testing/tiny-random-CLIPModel"
    assert hf_config.local_files_only is True
    assert hf_config.offline is True
    assert hf_config.trust_remote_code is False

    dataset = HFDatasetLoader(cfg.data.huggingface).load_dataset()
    assert len(dataset) == 2
    assert dataset[0]["video"] == "dummy_000.mp4"
    assert dataset[0]["answer"] == "dummy"

    model_loader = HFModelLoader(hf_config)
    processor_loader = HFProcessorLoader(hf_config)
    try:
        model = model_loader.load_model()
        processor = processor_loader.load_processor()
        tokenizer = processor_loader.load_tokenizer()
    except Exception as exc:
        _skip_if_cache_miss(exc)

    assert model is not None
    assert processor is not None
    assert tokenizer is not None
    assert tokenizer("dummy prompt")["input_ids"]
    assert model_loader.last_load_info is not None
    assert processor_loader.last_load_info is not None
    assert "optional-smoke-secret-token" not in json.dumps(model_loader.last_load_info)
    assert "optional-smoke-secret-token" not in json.dumps(processor_loader.last_load_info)
