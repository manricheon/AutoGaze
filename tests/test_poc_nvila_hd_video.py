from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "poc_nvila_hd_video.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("poc_nvila_hd_video", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["poc_nvila_hd_video"] = module
    spec.loader.exec_module(module)
    return module


def _write_config(
    tmp_path: Path,
    *,
    missing_nvila_checkpoint: bool = False,
    autogaze_enabled: bool = True,
) -> Path:
    autogaze_dir = tmp_path / "AutoGaze"
    siglip_dir = tmp_path / "siglip"
    nvila_dir = tmp_path / "NVILA-8B-HD-Video"
    for directory in (autogaze_dir, siglip_dir, nvila_dir):
        directory.mkdir()
        (directory / "config.json").write_text("{}", encoding="utf-8")

    nvila_checkpoint = tmp_path / "missing_nvila" if missing_nvila_checkpoint else nvila_dir
    autogaze_cfg = {
        "enabled": True,
        "module_path": "fake_autogaze",
        "class_or_factory": "AutoGaze",
        "checkpoint": str(autogaze_dir),
        "config_path": str(autogaze_dir / "config.json"),
        "model_config_path": str(autogaze_dir),
        "processor_path": str(autogaze_dir),
        "tokenizer_or_processor_path": str(autogaze_dir),
        "local_files_only": True,
        "trust_remote_code": False,
        "construction_kwargs": {},
        "original_cli_args": {"gazing_ratio": 0.75, "task_loss_requirement": 0.7},
    }
    if not autogaze_enabled:
        autogaze_cfg = {
            "enabled": False,
            "mode": "full",
            "status": "disabled_full_token_path",
        }

    cfg = {
        "experiment": {"id": "A2_real" if autogaze_enabled else "A1_real"},
        "model": {
            "autogaze": autogaze_cfg,
            "vision_encoder": {
                "type": "modified_siglip",
                "module_path": "fake_siglip",
                "class_or_factory": "SiglipVisionModel",
                "checkpoint": str(siglip_dir),
                "config_path": str(siglip_dir / "config.json"),
                "model_config_path": str(siglip_dir),
                "processor_path": str(siglip_dir),
                "tokenizer_or_processor_path": str(siglip_dir),
                "local_files_only": True,
                "trust_remote_code": False,
                "construction_kwargs": {},
            },
            "mllm": {
                "type": "nvila",
                "module_path": "fake_transformers",
                "class_or_factory": "AutoModel",
                "checkpoint": str(nvila_checkpoint),
                "config_path": str(nvila_dir / "config.json"),
                "model_config_path": str(nvila_dir),
                "processor_path": str(nvila_dir),
                "tokenizer_path": str(nvila_dir),
                "tokenizer_or_processor_path": str(nvila_dir),
                "local_files_only": True,
                "trust_remote_code": True,
                "construction_kwargs": {},
                "nvila_hd_video_model_id": "nvidia/NVILA-8B-HD-Video",
                "nvila_hd_video_processor_module_path": "fake_transformers",
                "nvila_hd_video_processor_class_name": "AutoProcessor",
                "nvila_hd_video_processor_factory_name": "from_pretrained",
                "nvila_hd_video_module_path": "fake_transformers",
                "nvila_hd_video_class_name": "AutoModel",
                "nvila_hd_video_factory_name": "from_pretrained",
                "nvila_hd_video_checkpoint_path": str(nvila_checkpoint),
                "nvila_hd_video_config_path": str(nvila_dir / "config.json"),
                "prompt_template": "{video_token}\n\n{prompt}",
                "video_preprocess_mode": "official_processor",
                "num_video_frames": 2,
                "num_video_frames_thumbnail": 1,
                "max_tiles_video": 1,
                "gazing_ratio_tile": [0.2, 0.06],
                "gazing_ratio_thumbnail": 1,
                "task_loss_requirement_tile": 0.6,
                "task_loss_requirement_thumbnail": None,
                "max_batch_size_autogaze": 1,
                "autogaze_enabled": autogaze_enabled,
            },
        },
        "inference": {
            "query_text": "Question: What is happening?\nPlease answer directly.",
            "output_dir": str(tmp_path / "outputs"),
        },
    }
    path = tmp_path / "A2_real.yaml"
    OmegaConf.save(OmegaConf.create(cfg), path)
    return path


def _install_fake_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_autogaze = types.ModuleType("fake_autogaze")
    fake_siglip = types.ModuleType("fake_siglip")
    fake_transformers = types.ModuleType("fake_transformers")

    class AutoGaze:
        num_vision_tokens_each_frame = 4

        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

        def eval(self):
            return self

        def to(self, *_args, **_kwargs):
            return self

        def __call__(self, _inputs, **_kwargs):
            return {
                "gazing_pos": torch.tensor([[0, 1, 4, 4]]),
                "if_padded_gazing": torch.tensor([[False, False, True, True]]),
            }

    class SiglipVisionModel:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

        def eval(self):
            return self

        def __call__(self, video, gazing_info=None):
            return torch.ones((video.shape[0], 2, 8))

    class _Tokenizer:
        video_token = "<video>"

    class AutoProcessor:
        tokenizer = _Tokenizer()

        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

        def __call__(self, *, text, videos, return_tensors):
            assert text.startswith("<video>")
            assert videos is not None
            assert return_tensors == "pt"
            return {"input_ids": torch.tensor([[1, 2]])}

        def batch_decode(self, *_args, **_kwargs):
            return ["mock answer"]

    class AutoModel:
        device = torch.device("cpu")

        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

        def eval(self):
            return self

        def generate(self, **_kwargs):
            return torch.tensor([[1, 2, 3]])

    fake_autogaze.AutoGaze = AutoGaze
    fake_siglip.SiglipVisionModel = SiglipVisionModel
    fake_transformers.AutoProcessor = AutoProcessor
    fake_transformers.AutoModel = AutoModel
    monkeypatch.setitem(sys.modules, "fake_autogaze", fake_autogaze)
    monkeypatch.setitem(sys.modules, "fake_siglip", fake_siglip)
    monkeypatch.setitem(sys.modules, "fake_transformers", fake_transformers)


def test_check_mode_with_fake_modules(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script_module()
    _install_fake_modules(monkeypatch)
    config = _write_config(tmp_path)
    output_dir = tmp_path / "poc_check"

    summary = module.run_poc(mode="check", config=config, output_dir=output_dir)

    assert summary.status == "passed"
    assert (output_dir / "logs" / "poc_summary.json").exists()
    assert {stage.name for stage in summary.stages} >= {
        "autogaze_import",
        "siglip_import",
        "nvila_model_import",
        "nvila_processor",
    }


def test_check_mode_allows_autogaze_off_a1_baseline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script_module()
    _install_fake_modules(monkeypatch)
    config = _write_config(tmp_path, autogaze_enabled=False)

    summary = module.run_poc(mode="check", config=config, output_dir=tmp_path / "a1_check")

    assert summary.status == "passed"
    autogaze_stage = next(stage for stage in summary.stages if stage.name == "autogaze_import")
    assert autogaze_stage.status == "disabled"
    assert all(
        item.exists or not item.required
        for item in summary.path_checks
        if item.name.startswith("autogaze_")
    )


def test_missing_checkpoint_path_reports_blocker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script_module()
    _install_fake_modules(monkeypatch)
    config = _write_config(tmp_path, missing_nvila_checkpoint=True)

    summary = module.run_poc(mode="check", config=config, output_dir=tmp_path / "missing")

    assert summary.status == "blocked"
    assert any("nvila_checkpoint missing" in item["reason"] for item in summary.skipped_stages)


def test_query_text_is_not_silently_ignored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script_module()
    _install_fake_modules(monkeypatch)
    config = _write_config(tmp_path)

    summary = module.run_poc(
        mode="full_pipeline",
        config=config,
        output_dir=tmp_path / "full_stub",
        query_text="Question: What happens?",
    )

    assert summary.query_text == "Question: What happens?"
    assert any(item["stage"] == "nvila_generation" for item in summary.skipped_stages)
    assert any("query text was accepted" in item["reason"] for item in summary.skipped_stages)


def test_output_directory_is_created(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script_module()
    _install_fake_modules(monkeypatch)
    config = _write_config(tmp_path)
    output_dir = tmp_path / "created_output"

    module.run_poc(mode="autogaze_only", config=config, output_dir=output_dir)

    assert output_dir.exists()
    assert (output_dir / "logs" / "poc_summary.json").exists()


def test_stub_stages_are_clearly_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script_module()
    _install_fake_modules(monkeypatch)
    config = _write_config(tmp_path)

    summary = module.run_poc(mode="autogaze_only", config=config, output_dir=tmp_path / "stub")

    autogaze_stage = next(stage for stage in summary.stages if stage.name == "autogaze")
    assert autogaze_stage.status == "skipped"
    assert "checkpoint loading disabled" in str(autogaze_stage.skipped_reason)
