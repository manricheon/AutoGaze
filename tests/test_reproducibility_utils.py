from __future__ import annotations

import json
import random

import numpy as np
import torch
from omegaconf import OmegaConf

from autogaze_ext.utils import (
    create_reproducibility_manifest,
    redact_sensitive_values,
    save_reproducibility_manifest,
    set_seed,
)


def _sample_config():
    return OmegaConf.create(
        {
            "experiment": {"id": "repro_smoke"},
            "runtime": {
                "precision": {"dtype": "float32"},
                "huggingface": {
                    "offline": True,
                    "cache_dir": "/tmp/hf-cache",
                    "token_env_var": "HF_TOKEN",
                    "token": "must-not-appear",
                },
            },
            "model": {
                "autogaze": {"checkpoint": None},
                "vision_encoder": {"checkpoint": "/models/vision.pt"},
                "mllm": {"checkpoint": "/models/mllm.pt"},
                "task_decoder": {"checkpoint": None},
                "huggingface": {
                    "model_id": "org/model",
                    "revision": "model-rev",
                    "trust_remote_code": False,
                    "token": "also-secret",
                },
            },
            "data": {
                "huggingface": {
                    "dataset_id": "org/dataset",
                    "dataset_config": "subset",
                    "dataset_split": "validation",
                    "revision": "dataset-rev",
                    "local_files_only": True,
                }
            },
        }
    )


def test_set_seed_repeats_random_numpy_and_torch_values():
    state = set_seed(123, deterministic=False)
    values_a = (random.random(), np.random.rand(), torch.rand(2))

    set_seed(123, deterministic=False)
    values_b = (random.random(), np.random.rand(), torch.rand(2))

    assert state.seed == 123
    assert state.deterministic is False
    assert values_a[0] == values_b[0]
    assert values_a[1] == values_b[1]
    assert torch.equal(values_a[2], values_b[2])


def test_reproducibility_manifest_creation_captures_required_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "env-secret-token")
    output_path = tmp_path / "manifest.json"

    saved_path = save_reproducibility_manifest(
        _sample_config(),
        output_path,
        repo_root=tmp_path,
        packages=["torch", "definitely-not-installed-autogaze-test-package"],
        timestamp="2026-05-11T00:00:00+00:00",
        metadata={"api_key": "hidden", "note": "dummy"},
    )

    data = json.loads(saved_path.read_text(encoding="utf-8"))
    raw_text = saved_path.read_text(encoding="utf-8")

    assert data["benchmark_timestamp"] == "2026-05-11T00:00:00+00:00"
    assert data["precision_setting"] == "float32"
    assert data["package_versions"]["torch"] != "not_installed"
    assert data["package_versions"]["definitely-not-installed-autogaze-test-package"] == "not_installed"
    assert "device_information" in data
    assert data["cuda_available"] in {True, False}
    assert data["mps_available"] in {True, False}
    assert data["model_checkpoints_used"]["vision_encoder"] == "/models/vision.pt"
    assert data["huggingface"]["model_id"] == "org/model"
    assert data["huggingface"]["model_revision"] == "model-rev"
    assert data["huggingface"]["dataset_id"] == "org/dataset"
    assert data["huggingface"]["dataset_revision"] == "dataset-rev"
    assert data["huggingface"]["cache_dir"] == "/tmp/hf-cache"
    assert data["huggingface"]["offline"] is True
    assert data["huggingface"]["local_files_only"] is True
    assert data["huggingface"]["trust_remote_code"] is False
    assert data["resolved_config"]["runtime"]["huggingface"]["token"] == "<REDACTED>"
    assert data["metadata"]["api_key"] == "<REDACTED>"
    assert "env-secret-token" not in raw_text
    assert "must-not-appear" not in raw_text
    assert "also-secret" not in raw_text


def test_redact_sensitive_values_preserves_safe_hf_token_metadata():
    redacted = redact_sensitive_values(
        {
            "token": "secret",
            "access_token": "secret",
            "nested": {"password": "secret", "token_env_var": "HF_TOKEN", "token_present": True},
            "normal": "value",
        }
    )

    assert redacted["token"] == "<REDACTED>"
    assert redacted["access_token"] == "<REDACTED>"
    assert redacted["nested"]["password"] == "<REDACTED>"
    assert redacted["nested"]["token_env_var"] == "HF_TOKEN"
    assert redacted["nested"]["token_present"] is True
    assert redacted["normal"] == "value"


def test_manifest_to_dict_is_redacted(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "another-env-secret")
    manifest = create_reproducibility_manifest(
        _sample_config(),
        repo_root=tmp_path,
        packages=[],
        timestamp="2026-05-11T00:00:00+00:00",
        metadata={"secret_note": "hidden"},
    )

    data = manifest.to_dict()
    text = json.dumps(data)

    assert data["resolved_config"]["model"]["huggingface"]["token"] == "<REDACTED>"
    assert data["metadata"]["secret_note"] == "<REDACTED>"
    assert "another-env-secret" not in text

