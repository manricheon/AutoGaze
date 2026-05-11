from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "download_hf_assets.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("download_hf_assets", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["download_hf_assets"] = module
    spec.loader.exec_module(module)
    return module


def test_dry_run_manifest_without_download(tmp_path, monkeypatch) -> None:
    module = _load_script_module()

    def fail_download(**kwargs):
        raise AssertionError("dry run must not call snapshot_download")

    monkeypatch.setattr(module, "_snapshot_download", fail_download)
    manifest = module.build_manifest(
        model_id="org/model",
        dataset_id="org/dataset",
        revision="abc123",
        cache_dir=str(tmp_path / "cache"),
        token_env_var="HF_TOKEN_FOR_TEST",
        include_processor_tokenizer=True,
        dry_run=True,
    )
    path = module.write_manifest(manifest, tmp_path / "manifest.json")
    data = json.loads(path.read_text())

    assert data["dry_run"] is True
    assert data["models"][0]["repo_id"] == "org/model"
    assert data["models"][0]["cache_path"] is None
    assert data["models"][0]["include_processor_tokenizer"] is True
    assert data["datasets"][0]["repo_id"] == "org/dataset"


def test_real_mode_uses_snapshot_mock_and_redacts_token(tmp_path, monkeypatch) -> None:
    module = _load_script_module()
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path / "cache" / kwargs["repo_id"].replace("/", "--"))

    monkeypatch.setattr(module, "_snapshot_download", fake_download)
    monkeypatch.setenv("HF_TOKEN_FOR_TEST", "secret-token")

    manifest = module.build_manifest(
        model_id="org/model",
        dataset_id=None,
        revision="abc123",
        cache_dir=str(tmp_path / "cache"),
        token_env_var="HF_TOKEN_FOR_TEST",
        include_processor_tokenizer=True,
        dry_run=False,
    )
    data = manifest.to_dict()

    assert calls[0]["repo_id"] == "org/model"
    assert calls[0]["token"] == "secret-token"
    assert data["token_present"] is True
    assert "secret-token" not in json.dumps(data)
    assert data["models"][0]["cache_path"].endswith("org--model")


def test_cli_dry_run_writes_manifest(tmp_path) -> None:
    module = _load_script_module()
    manifest = module.build_manifest(
        model_id="org/model",
        dataset_id=None,
        revision=None,
        cache_dir=None,
        token_env_var="HF_TOKEN",
        include_processor_tokenizer=False,
        dry_run=True,
    )
    out = module.write_manifest(manifest, tmp_path / "manifest.json")

    assert out.exists()
    assert json.loads(out.read_text())["models"][0]["repo_id"] == "org/model"
