from __future__ import annotations

from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

from autogaze_ext.investigation.canonical_smoke_inference import run_canonical_smoke_inference


class MockAutoGaze:
    config = SimpleNamespace(scales=[32, 64, 112, 224])
    num_vision_tokens_each_frame = 196

    @classmethod
    def from_pretrained(cls, *_args: Any, **_kwargs: Any) -> "MockAutoGaze":
        return cls()

    def eval(self) -> "MockAutoGaze":
        return self

    def to(self, *_args: Any, **_kwargs: Any) -> "MockAutoGaze":
        return self

    def __call__(self, inputs: dict[str, torch.Tensor], **kwargs: Any) -> dict[str, torch.Tensor | list[int]]:
        video = inputs["video"]
        assert video.ndim == 5
        return {
            "gazing_pos": torch.tensor([[0, 1, 2, 3]], device=video.device),
            "if_padded_gazing": torch.tensor([[False, False, True, False]], device=video.device),
            "num_gazing_each_frame": torch.tensor([2, 2], device=video.device),
            "selected_scales": kwargs.get("target_scales", [32, 64, 112, 224]),
        }


class MockSigLIP:
    @classmethod
    def from_pretrained(cls, *_args: Any, **_kwargs: Any) -> "MockSigLIP":
        return cls()

    def eval(self) -> "MockSigLIP":
        return self

    def to(self, *_args: Any, **_kwargs: Any) -> "MockSigLIP":
        return self

    def __call__(self, video: torch.Tensor, gazing_info: dict[str, torch.Tensor] | None = None) -> SimpleNamespace:
        if gazing_info is not None:
            token_count = int(gazing_info["gazing_pos"].shape[-1])
        else:
            token_count = int(video.shape[1] * (video.shape[-1] // 16) * (video.shape[-2] // 16))
        return SimpleNamespace(last_hidden_state=torch.ones(video.shape[0], token_count, 8, device=video.device))


class MockNVILA:
    @classmethod
    def from_pretrained(cls, *_args: Any, **_kwargs: Any) -> "MockNVILA":
        return cls()

    def eval(self) -> "MockNVILA":
        return self

    def to(self, *_args: Any, **_kwargs: Any) -> "MockNVILA":
        return self

    def generate(self, visual_features: torch.Tensor, query_text: str, max_new_tokens: int = 1) -> dict[str, str]:
        assert max_new_tokens == 1
        assert visual_features.ndim == 3
        return {"generated_text": f"mock answer: {query_text}"}


def _module(**attrs: Any) -> ModuleType:
    module = ModuleType("mock_module")
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _fake_import(name: str) -> ModuleType:
    modules = {
        "fake_autogaze": _module(AutoGaze=MockAutoGaze),
        "fake_siglip": _module(SiglipVisionModel=MockSigLIP),
        "fake_nvila": _module(NVILA=MockNVILA),
    }
    if name not in modules:
        raise ModuleNotFoundError(name)
    return modules[name]


def _write_reference_files(root: Path) -> Path:
    (root / "configs").mkdir()
    (root / "docs").mkdir()
    quick_start = root / "QUICK_START.md"
    quick_start.write_text("# Quick Start\n", encoding="utf-8")
    (root / "docs" / "QUICK_START_reference.md").write_text("# QUICK_START Reference\n", encoding="utf-8")
    return root / "configs"


def _cfg(tmp_path: Path, *, experiment: str = "A2_real", autogaze_enabled: bool = True, mllm_module: str = "fake_nvila") -> dict[str, Any]:
    return {
        "experiment": {"id": experiment},
        "inference": {"output_dir": str(tmp_path / "outputs" / experiment)},
        "model": {
            "autogaze": {
                "enabled": autogaze_enabled,
                "module_path": "fake_autogaze",
                "class_or_factory": "AutoGaze",
                "model_config_path": "bfshi/AutoGaze",
                "local_files_only": True,
                "original_cli_args": {
                    "gazing_ratio": 0.75,
                    "task_loss_requirement": 0.7,
                    "target_scales": None,
                    "target_patch_size": None,
                },
            },
            "vision_encoder": {
                "module_path": "fake_siglip",
                "class_or_factory": "SiglipVisionModel",
                "model_config_path": "google/siglip2-base-patch16-224",
                "local_files_only": True,
                "construction_kwargs": {"attn_implementation": "sdpa"},
                "original_cli_args": {
                    "gazing_info_argument": "gazing_info",
                    "high_resolution_example": {
                        "target_scales": [56, 112, 196, 392],
                        "target_patch_size": 14,
                    },
                },
            },
            "mllm": {
                "module_path": mllm_module,
                "class_or_factory": "NVILA",
                "model_config_path": "mock-nvila",
                "local_files_only": True,
            },
        },
    }


def test_autogaze_only_smoke_writes_outputs(tmp_path: Path) -> None:
    config_dir = _write_reference_files(tmp_path)

    report = run_canonical_smoke_inference(
        experiment="A2_real",
        mode="autogaze_only",
        num_frames=2,
        resolution=224,
        output_dir=tmp_path / "out",
        config_dir=config_dir,
        cfg=_cfg(tmp_path),
        import_module_fn=_fake_import,
    )

    assert report.status == "passed"
    assert report.quick_start_found is True
    assert report.input_shape == [1, 2, 3, 224, 224]
    assert report.selected_token_count == 3
    assert Path(report.artifacts["selected_patch_indices"]).exists()
    assert Path(report.artifacts["token_counts"]).exists()
    assert report.visualization_dir is not None


def test_full_pipeline_accepts_query_and_skips_missing_mllm(tmp_path: Path) -> None:
    config_dir = _write_reference_files(tmp_path)

    report = run_canonical_smoke_inference(
        experiment="A1_real",
        mode="full_pipeline",
        query_text="Describe the video.",
        num_frames=2,
        resolution=224,
        output_dir=tmp_path / "out",
        config_dir=config_dir,
        cfg=_cfg(tmp_path, experiment="A1_real", autogaze_enabled=False, mllm_module="missing_nvila"),
        import_module_fn=_fake_import,
    )

    assert report.status == "partial"
    assert report.vision_feature_shape == [1, 392, 8]
    assert report.output_text is None
    assert any("query text was accepted" in item["reason"] for item in report.skipped_stages)


def test_full_pipeline_generates_with_mock_mllm(tmp_path: Path) -> None:
    config_dir = _write_reference_files(tmp_path)

    report = run_canonical_smoke_inference(
        experiment="A2_real",
        mode="full_pipeline",
        query_text="What is happening?",
        num_frames=2,
        resolution=224,
        max_new_tokens=1,
        allow_mllm_load=True,
        output_dir=tmp_path / "out",
        config_dir=config_dir,
        cfg=_cfg(tmp_path),
        import_module_fn=_fake_import,
    )

    assert report.status == "passed"
    assert report.vision_feature_shape == [1, 4, 8]
    assert report.mllm_input_shape == [1, 4, 8]
    assert report.output_text == "mock answer: What is happening?"
    assert Path(report.artifacts["answer"]).exists()


def test_quick_start_scaling_fields_are_reflected(tmp_path: Path) -> None:
    config_dir = _write_reference_files(tmp_path)

    report = run_canonical_smoke_inference(
        experiment="A2_real",
        mode="autogaze_only",
        num_frames=2,
        resolution=392,
        scale_resolution="quick_start_target_scales",
        output_dir=tmp_path / "out",
        config_dir=config_dir,
        cfg=_cfg(tmp_path),
        import_module_fn=_fake_import,
    )

    assert report.scaling["status"] == "quick_start_target_scales_applied"
    assert report.scaling["target_scales"] == [56, 112, 196, 392]
    autogaze_stage = next(stage for stage in report.stages if stage.name == "autogaze")
    assert autogaze_stage.details["call_kwargs"]["target_patch_size"] == 14


def test_invalid_video_mode_errors(tmp_path: Path) -> None:
    config_dir = _write_reference_files(tmp_path)

    with pytest.raises(ValueError, match="Only --video dummy"):
        run_canonical_smoke_inference(
            experiment="A2_real",
            mode="autogaze_only",
            video="unsupported",
            config_dir=config_dir,
            cfg=_cfg(tmp_path),
            import_module_fn=_fake_import,
        )
