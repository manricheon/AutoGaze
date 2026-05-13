from __future__ import annotations

import importlib.util
import json
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
    dummy_frames: int = 4,
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
        "data": {"dummy_video": {"frames": dummy_frames}},
        "inference": {
            "query_text": "Question: What is happening?\nPlease answer directly.",
            "output_dir": str(tmp_path / "outputs"),
            "frame_selection_mode": "sample",
            "frame_interval": 1,
            "max_windows": None,
            "drop_last": False,
            "pad_last": False,
            "scaling_mode": "resize",
            "patch_size": 16,
            "spatial_tile_size": None,
            "gaze_ratio": 0.75 if autogaze_enabled else None,
            "task_loss_requirement": 0.7 if autogaze_enabled else None,
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
            batch = int(_inputs["video"].shape[0])
            return {
                "gazing_pos": torch.tensor([[0, 1, 4, 4]] * batch),
                "if_padded_gazing": torch.tensor([[False, False, True, True]] * batch),
                "selected_scales": torch.tensor([[32, 64, 32, 64]] * batch),
            }

    class StrictAutoGaze(AutoGaze):
        def __call__(self, _inputs):
            batch = int(_inputs["video"].shape[0])
            return {
                "gazing_pos": torch.tensor([[0, 1]] * batch),
                "if_padded_gazing": torch.tensor([[False, False]] * batch),
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
    fake_autogaze.StrictAutoGaze = StrictAutoGaze
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


def test_run_poc_default_num_frames_matches_canonical_setting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script_module()
    _install_fake_modules(monkeypatch)
    config = _write_config(tmp_path, dummy_frames=20)

    summary = module.run_poc(
        mode="autogaze_only",
        config=config,
        output_dir=tmp_path / "default_num_frames",
    )

    assert module.DEFAULT_NUM_FRAMES == 16
    assert summary.frame_selection["num_frames"] == 16
    assert len(summary.frame_selection["window_frame_indices"][0]) == 16


def test_autogaze_only_video_exports_with_fake_modules(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script_module()
    _install_fake_modules(monkeypatch)
    config = _write_config(tmp_path)
    output_dir = tmp_path / "video_export"

    summary = module.run_poc(
        mode="autogaze_only",
        config=config,
        output_dir=output_dir,
        num_frames=2,
        resolution=32,
        no_checkpoint_load=False,
        save_overlay_video=True,
        save_side_by_side_video=True,
        video_fps=2.0,
    )

    assert summary.status == "passed"
    overlay_video = output_dir / "visualizations" / "autogaze_only" / "videos" / "autogaze_overlay_sampled_only.mp4"
    side_by_side_video = output_dir / "visualizations" / "autogaze_only" / "videos" / "autogaze_side_by_side_sampled_only.mp4"
    metadata_path = (
        output_dir
        / "visualizations"
        / "autogaze_only"
        / "metadata"
        / "visualization_video_metadata.json"
    )
    assert overlay_video.exists() and overlay_video.stat().st_size > 0
    assert side_by_side_video.exists() and side_by_side_video.stat().st_size > 0
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["patch_grid_source"] == "inferred_from_processed_resolution_and_patch16"
    assert summary.artifacts["overlay_video"] == str(overlay_video)
    assert summary.artifacts["side_by_side_video"] == str(side_by_side_video)


def test_frame_selection_metadata_is_saved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script_module()
    _install_fake_modules(monkeypatch)
    config = _write_config(tmp_path, dummy_frames=10)
    output_dir = tmp_path / "frame_selection_metadata"

    summary = module.run_poc(
        mode="autogaze_only",
        config=config,
        output_dir=output_dir,
        num_frames=4,
        frame_selection_mode="interval",
        frame_interval=2,
        no_checkpoint_load=False,
    )

    metadata_path = output_dir / "autogaze" / "frame_selection_metadata.json"
    assert summary.status == "passed"
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["mode"] == "interval"
    assert metadata["frame_interval"] == 2
    assert metadata["window_frame_indices"] == [[0, 2, 4, 6]]


def test_autogaze_only_runs_over_multiple_windows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script_module()
    _install_fake_modules(monkeypatch)
    config = _write_config(tmp_path, dummy_frames=5)
    output_dir = tmp_path / "multi_window"

    summary = module.run_poc(
        mode="autogaze_only",
        config=config,
        output_dir=output_dir,
        num_frames=2,
        frame_selection_mode="chunk",
        no_checkpoint_load=False,
    )

    assert summary.status == "passed"
    assert summary.frame_selection["number_of_windows"] == 3
    assert {stage.name for stage in summary.stages} >= {
        "autogaze_window_000",
        "autogaze_window_001",
        "autogaze_window_002",
    }
    assert (output_dir / "autogaze" / "windows" / "window_000" / "selected_patch_indices.json").exists()
    assert (output_dir / "autogaze" / "windows" / "window_001" / "token_counts.json").exists()
    assert (output_dir / "visualizations" / "autogaze_only" / "windows" / "window_000" / "frames").exists()


def test_sampled_only_video_export_over_multiple_windows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script_module()
    _install_fake_modules(monkeypatch)
    config = _write_config(tmp_path, dummy_frames=4)
    output_dir = tmp_path / "multi_window_video"

    summary = module.run_poc(
        mode="autogaze_only",
        config=config,
        output_dir=output_dir,
        num_frames=2,
        resolution=32,
        frame_selection_mode="chunk",
        no_checkpoint_load=False,
        save_overlay_video=True,
        save_side_by_side_video=True,
    )

    overlay_video = output_dir / "visualizations" / "autogaze_only" / "videos" / "autogaze_overlay_sampled_only.mp4"
    side_by_side_video = output_dir / "visualizations" / "autogaze_only" / "videos" / "autogaze_side_by_side_sampled_only.mp4"
    metadata_path = (
        output_dir
        / "visualizations"
        / "autogaze_only"
        / "metadata"
        / "visualization_video_metadata.json"
    )

    assert summary.status == "passed"
    assert overlay_video.exists() and overlay_video.stat().st_size > 0
    assert side_by_side_video.exists() and side_by_side_video.stat().st_size > 0
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["sampled_frame_indices"] == [0, 1, 2, 3]
    canonical_metadata = json.loads(
        (
            output_dir
            / "visualizations"
            / "autogaze"
            / "metadata"
            / "visualization_video_metadata.json"
        ).read_text(encoding="utf-8")
    )
    assert canonical_metadata["comparison_layout"] == "processed_overlay"
    assert (output_dir / "visualizations" / "autogaze" / "videos" / "autogaze_side_by_side.mp4").exists()


def test_full_length_video_export_preserves_frame_count_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script_module()
    _install_fake_modules(monkeypatch)
    config = _write_config(tmp_path, dummy_frames=4)
    output_dir = tmp_path / "full_length_video"

    summary = module.run_poc(
        mode="autogaze_only",
        config=config,
        output_dir=output_dir,
        num_frames=2,
        resolution=32,
        frame_selection_mode="sample",
        no_checkpoint_load=False,
        save_overlay_video=True,
        save_side_by_side_video=True,
        video_export_mode="full_length",
        video_fps=3.0,
    )

    metadata = json.loads(
        (
            output_dir
            / "visualizations"
            / "autogaze"
            / "metadata"
            / "visualization_video_metadata.json"
        ).read_text(encoding="utf-8")
    )
    assert summary.status == "passed"
    assert (output_dir / "visualizations" / "autogaze" / "videos" / "autogaze_overlay_full_length.mp4").exists()
    assert (output_dir / "visualizations" / "autogaze" / "videos" / "autogaze_side_by_side_full_length.mp4").exists()
    assert metadata["video_export_mode"] == "full_length"
    assert metadata["original_frame_count"] == 4
    assert metadata["full_length_export"]["output_frame_count"] == 4
    assert metadata["full_length_export"]["processed_frame_indices"] == [0, 3]
    assert metadata["full_length_export"]["status"] == "implemented"


def test_original_space_overlay_via_poc_resize(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script_module()
    _install_fake_modules(monkeypatch)
    config = _write_config(tmp_path, dummy_frames=2)
    output_dir = tmp_path / "original_overlay"

    summary = module.run_poc(
        mode="autogaze_only",
        config=config,
        output_dir=output_dir,
        num_frames=2,
        resolution=32,
        scaling_mode="resize",
        no_checkpoint_load=False,
        save_side_by_side_video=True,
        comparison_layout="original_overlay",
    )

    metadata = json.loads(
        (
            output_dir
            / "visualizations"
            / "autogaze"
            / "metadata"
            / "visualization_video_metadata.json"
        ).read_text(encoding="utf-8")
    )
    assert summary.status == "passed"
    assert (output_dir / "visualizations" / "autogaze" / "videos" / "autogaze_original_overlay.mp4").exists()
    assert metadata["comparison_layout"] == "original_overlay"
    assert metadata["coordinate_mapping"]["mapping_exact"] is True


def test_runtime_controls_and_scaling_metadata_are_saved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script_module()
    _install_fake_modules(monkeypatch)
    config = _write_config(tmp_path, dummy_frames=4)
    output_dir = tmp_path / "runtime_controls"

    summary = module.run_poc(
        mode="autogaze_only",
        config=config,
        output_dir=output_dir,
        num_frames=2,
        resolution=32,
        scaling_mode="fit_short_side",
        gaze_ratio=0.5,
        task_loss_requirement=0.6,
        no_checkpoint_load=False,
    )

    token_counts = json.loads(
        (output_dir / "autogaze" / "windows" / "window_000" / "token_counts.json").read_text(encoding="utf-8")
    )
    runtime = json.loads((output_dir / "autogaze" / "runtime_metadata.json").read_text(encoding="utf-8"))
    token_summary = json.loads((output_dir / "autogaze" / "token_counts_summary.json").read_text(encoding="utf-8"))
    metrics = json.loads((output_dir / "logs" / "metrics.json").read_text(encoding="utf-8"))
    scaling = json.loads((output_dir / "scaling" / "scaling_metadata.json").read_text(encoding="utf-8"))

    assert summary.autogaze_runtime["gazing_ratio"] == 0.5
    assert summary.autogaze_runtime["task_loss_requirement"] == 0.6
    assert token_counts["autogaze_runtime"]["gazing_ratio"] == 0.5
    assert runtime["requested_gaze_ratio"] == 0.5
    assert runtime["effective_gaze_ratio"] == 0.5
    assert token_summary["original_visual_token_count"] == summary.original_visual_token_count
    assert token_summary["selected_visual_token_count"] == summary.selected_token_count
    assert metrics["mode"] == "autogaze_only"
    assert metrics["frame_selection_mode"] == "sample"
    assert metrics["scaling_mode"] == "fit_short_side"
    assert metrics["selected_visual_token_count"] == summary.selected_token_count
    assert (output_dir / "logs" / "metrics.csv").exists()
    assert scaling["scaling_mode"] == "fit_short_side"
    assert scaling["windows"][0]["aspect_ratio_preserved"] is True


def test_strict_autogaze_params_records_signature_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script_module()
    _install_fake_modules(monkeypatch)
    config = _write_config(tmp_path, dummy_frames=4)
    output_dir = tmp_path / "strict_params_pass"

    summary = module.run_poc(
        mode="autogaze_only",
        config=config,
        output_dir=output_dir,
        num_frames=2,
        resolution=32,
        gaze_ratio=0.5,
        task_loss_requirement=0.6,
        strict_autogaze_params=True,
        no_checkpoint_load=False,
    )

    runtime = json.loads((output_dir / "autogaze" / "runtime_metadata.json").read_text(encoding="utf-8"))
    metrics = json.loads((output_dir / "logs" / "metrics.json").read_text(encoding="utf-8"))

    assert summary.status == "passed"
    assert runtime["strict_autogaze_params"] is True
    assert runtime["autogaze_param_validation"]["accepted"] is True
    assert runtime["autogaze_param_validation"]["accepted_via_var_kwargs"] is True
    assert runtime["unsupported_runtime_params"] == []
    assert metrics["strict_autogaze_params"] is True
    assert metrics["unsupported_runtime_params"] == []


def test_strict_autogaze_params_reports_unsupported_kwargs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script_module()
    _install_fake_modules(monkeypatch)
    config = _write_config(tmp_path, dummy_frames=4)
    cfg = OmegaConf.load(config)
    cfg.model.autogaze.class_or_factory = "StrictAutoGaze"
    OmegaConf.save(cfg, config)
    output_dir = tmp_path / "strict_params_fail"

    summary = module.run_poc(
        mode="autogaze_only",
        config=config,
        output_dir=output_dir,
        num_frames=2,
        resolution=32,
        gaze_ratio=0.5,
        task_loss_requirement=0.6,
        strict_autogaze_params=True,
        no_checkpoint_load=False,
    )

    runtime = json.loads((output_dir / "autogaze" / "runtime_metadata.json").read_text(encoding="utf-8"))

    assert summary.status == "partial"
    assert runtime["strict_autogaze_params"] is True
    assert runtime["autogaze_param_validation"]["accepted"] is False
    assert set(runtime["unsupported_runtime_params"]) >= {"gazing_ratio", "task_loss_requirement"}
    assert any("strict AutoGaze parameter validation failed" in item["reason"] for item in summary.skipped_stages)


def test_full_pipeline_saves_autogaze_visualization_under_full_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script_module()
    _install_fake_modules(monkeypatch)
    config = _write_config(tmp_path, dummy_frames=4)
    output_dir = tmp_path / "full_pipeline_visuals"

    summary = module.run_poc(
        mode="full_pipeline",
        config=config,
        output_dir=output_dir,
        query_text="Question: What happens?",
        num_frames=2,
        resolution=32,
        no_checkpoint_load=False,
        save_overlay_video=True,
        save_side_by_side_video=True,
        save_scale_panel_video=True,
        show_patch_indices=False,
        show_scale_labels=True,
    )

    assert summary.status == "passed"
    assert (output_dir / "visualizations" / "full_pipeline" / "videos" / "autogaze_overlay_sampled_only.mp4").exists()
    assert (output_dir / "visualizations" / "full_pipeline" / "videos" / "autogaze_side_by_side_sampled_only.mp4").exists()
    assert (output_dir / "visualizations" / "full_pipeline" / "videos" / "autogaze_scale_panels_sampled_only.mp4").exists()
    assert (output_dir / "visualizations" / "autogaze" / "videos" / "autogaze_overlay.mp4").exists()
    assert (output_dir / "visualizations" / "autogaze" / "videos" / "autogaze_side_by_side.mp4").exists()
    assert (output_dir / "visualizations" / "autogaze" / "videos" / "autogaze_scale_panels.mp4").exists()
    assert (output_dir / "visualizations" / "autogaze" / "windows" / "window_000" / "scale_panels").exists()
    metadata = json.loads(
        (
            output_dir
            / "visualizations"
            / "full_pipeline"
            / "metadata"
            / "visualization_video_metadata.json"
        ).read_text(encoding="utf-8")
    )
    assert metadata["show_patch_indices"] is False
    assert metadata["show_scale_labels"] is True
    assert metadata["missing_scale_metadata"] is False
    assert metadata["info_panel_mode"] == "external"
    assert (output_dir / "predictions" / "answer.json").exists()
    metrics = json.loads((output_dir / "logs" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["mode"] == "full_pipeline"
    assert metrics["generated_answer"] == "mock answer"
    assert metrics["vision_encoder_latency_ms"] != "N/A"
    assert metrics["mllm_decode_latency_ms"] != "N/A"


def test_chop_mode_records_chop_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script_module()
    _install_fake_modules(monkeypatch)
    config = _write_config(tmp_path, dummy_frames=2)
    output_dir = tmp_path / "chop"

    summary = module.run_poc(
        mode="autogaze_only",
        config=config,
        output_dir=output_dir,
        num_frames=2,
        resolution=16,
        scaling_mode="chop",
        spatial_tile_size=16,
        save_chop_frames=True,
        no_checkpoint_load=False,
    )

    assert summary.status == "passed"
    assert summary.scaling["scaling_mode"] == "chop"
    assert summary.scaling["windows"][0]["status"] == "partial_quick_start_chop"
    assert "chop" in summary.scaling["windows"][0]
    assert (output_dir / "autogaze" / "windows" / "window_000" / "selected_patch_mask.json").exists()
    chop_metadata = json.loads((output_dir / "chops" / "chop_metadata.json").read_text(encoding="utf-8"))
    assert chop_metadata["coordinate_space"] == "chop_local"
    assert chop_metadata["original_space_overlay_supported"] is False
    assert chop_metadata["number_of_chops"] >= 1
    first = chop_metadata["chops"][0]
    assert first["window_id"] == 0
    assert first["chop_overlap"] == 0
    assert "selected_patch_indices" in first
    assert (
        output_dir
        / "chops"
        / "windows"
        / "window_000"
        / f"frame_{first['source_frame_index']:03d}"
        / "chop_000"
        / "token_counts.json"
    ).exists()
    assert (
        output_dir
        / "visualizations"
        / "autogaze"
        / "chops"
        / "window_000"
        / f"frame_{first['source_frame_index']:03d}"
        / "chop_000"
        / "frames"
    ).exists()


def test_chop_overlay_union_writes_merged_overlay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script_module()
    _install_fake_modules(monkeypatch)
    config = _write_config(tmp_path, dummy_frames=2)
    output_dir = tmp_path / "chop_overlay_union"

    summary = module.run_poc(
        mode="autogaze_only",
        config=config,
        output_dir=output_dir,
        num_frames=2,
        resolution=16,
        scaling_mode="chop",
        spatial_tile_size=16,
        chop_merge_mode="overlay_union",
        save_chop_overlay_video=True,
        no_checkpoint_load=False,
    )

    metadata = json.loads(
        (
            output_dir
            / "visualizations"
            / "autogaze"
            / "metadata"
            / "chop_overlay_metadata.json"
        ).read_text(encoding="utf-8")
    )
    assert summary.status == "passed"
    assert (output_dir / "visualizations" / "autogaze" / "videos" / "autogaze_chop_overlay.mp4").exists()
    assert (
        output_dir
        / "visualizations"
        / "autogaze"
        / "windows"
        / "window_000"
        / "frames"
        / "frame_000_chop_merged_overlay.png"
    ).exists()
    assert metadata["merge_mode"] == "overlay_union"
    assert metadata["windows"][0]["coordinate_space"] == "full_processed_frame"
    assert metadata["windows"][0]["scale_conflict_handling"] == "last_scale_wins"


def test_chop_overlay_union_rejects_overlap_until_mapping_is_defined(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script_module()
    _install_fake_modules(monkeypatch)
    config = _write_config(tmp_path, dummy_frames=2)

    with pytest.raises(NotImplementedError, match="non-overlapping"):
        module.run_poc(
            mode="autogaze_only",
            config=config,
            output_dir=tmp_path / "chop_overlap",
            num_frames=2,
            resolution=16,
            scaling_mode="chop",
            spatial_tile_size=16,
            chop_overlap=4,
            chop_merge_mode="overlay_union",
            save_chop_overlay_video=True,
            no_checkpoint_load=False,
        )


def test_stub_stages_are_clearly_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script_module()
    _install_fake_modules(monkeypatch)
    config = _write_config(tmp_path)

    summary = module.run_poc(mode="autogaze_only", config=config, output_dir=tmp_path / "stub")

    autogaze_stage = next(stage for stage in summary.stages if stage.name == "autogaze")
    assert autogaze_stage.status == "skipped"
    assert "checkpoint loading disabled" in str(autogaze_stage.skipped_reason)


def test_visualization_skip_metadata_is_saved_when_autogaze_is_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script_module()
    _install_fake_modules(monkeypatch)
    config = _write_config(tmp_path)
    output_dir = tmp_path / "visualization_skip"

    summary = module.run_poc(
        mode="autogaze_only",
        config=config,
        output_dir=output_dir,
        num_frames=2,
        resolution=32,
        save_overlay_video=True,
        save_side_by_side_video=True,
        save_scale_panel_video=True,
    )

    metadata_path = output_dir / "visualizations" / "autogaze" / "metadata" / "visualization_skip_metadata.json"
    metrics = json.loads((output_dir / "logs" / "metrics.json").read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert summary.status == "partial"
    assert summary.artifacts["visualization_skip_metadata"] == str(metadata_path)
    assert metadata["status"] == "skipped"
    assert "checkpoint loading disabled" in metadata["reason"]
    assert metadata["requested_outputs"]["overlay_video"] is True
    assert metadata["requested_outputs"]["side_by_side_video"] is True
    assert metadata["requested_outputs"]["scale_panel_video"] is True
    assert metadata["frame_selection"]["mode"] == "sample"
    assert metrics["visualization_status"] == "skipped"
    assert "checkpoint loading disabled" in metrics["visualization_skip_reason"]
    assert metrics["visualization_skip_metadata"]["status"] == "skipped"
    assert not (output_dir / "visualizations" / "autogaze" / "videos" / "autogaze_overlay.mp4").exists()


@pytest.mark.parametrize("export_mode", ["hold_last"])
def test_poc_video_export_unsupported_modes_raise(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    export_mode: str,
) -> None:
    module = _load_script_module()
    _install_fake_modules(monkeypatch)
    config = _write_config(tmp_path)

    with pytest.raises(NotImplementedError, match=export_mode):
        module.run_poc(
            mode="autogaze_only",
            config=config,
            output_dir=tmp_path / export_mode,
            save_overlay_video=True,
            video_export_mode=export_mode,
        )


def test_quickstart_scaling_metadata_supported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script_module()
    _install_fake_modules(monkeypatch)
    config = _write_config(tmp_path)
    output_dir = tmp_path / "quickstart_supported"

    summary = module.run_poc(
        mode="autogaze_only",
        config=config,
        output_dir=output_dir,
        num_frames=2,
        resolution=224,
        scaling_mode="quickstart",
        patch_size=16,
        no_checkpoint_load=False,
    )

    scaling = json.loads((output_dir / "scaling" / "scaling_metadata.json").read_text(encoding="utf-8"))
    assert summary.status == "passed"
    assert scaling["scaling_mode"] == "quickstart"
    assert scaling["windows"][0]["quickstart_reference_used"] == "docs/QUICK_START_reference.md"
    assert scaling["windows"][0]["quickstart_exact_match"] is True
    assert scaling["windows"][0]["unsupported_reason"] is None


def test_quickstart_scaling_metadata_unsupported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script_module()
    _install_fake_modules(monkeypatch)
    config = _write_config(tmp_path)
    output_dir = tmp_path / "quickstart_unsupported"

    summary = module.run_poc(
        mode="autogaze_only",
        config=config,
        output_dir=output_dir,
        num_frames=2,
        resolution=448,
        scaling_mode="quickstart",
        patch_size=16,
        no_checkpoint_load=False,
    )

    scaling = json.loads((output_dir / "scaling" / "scaling_metadata.json").read_text(encoding="utf-8"))
    assert summary.status in {"blocked", "partial"}
    assert scaling["scaling_mode"] == "quickstart"
    assert scaling["quickstart_exact_match"] is False
    assert "Unsupported" in scaling["unsupported_reason"]


def test_priority2_benchmark_configs_load() -> None:
    required_fields = {
        "mode",
        "full_pipeline_plugin_mode",
        "component_plugins",
        "frame_selection_mode",
        "num_frames",
        "scaling_mode",
        "resolution",
        "gaze_ratio",
        "overlay_style",
        "multi_scale_overlay",
        "scale_color_mode",
        "show_patch_index",
        "show_scale_label",
        "save_overlay_video",
        "save_side_by_side_video",
        "save_scale_panel_video",
        "scale_panel_layout",
        "comparison_layout",
        "metadata_placement",
        "info_panel_position",
        "output_dir",
    }
    config_paths = [
        ROOT / "configs" / "benchmark" / "poc_default.yaml",
        ROOT / "configs" / "benchmark" / "poc_feature_matrix_smoke.yaml",
        ROOT / "configs" / "benchmark" / "poc_autogaze_only_visualization.yaml",
        ROOT / "configs" / "benchmark" / "poc_full_pipeline_visualization.yaml",
        ROOT / "configs" / "benchmark" / "poc_chop_mode_smoke.yaml",
        ROOT / "configs" / "benchmark" / "poc_multiscale_visualization.yaml",
        ROOT / "configs" / "benchmark" / "poc_scale_panel_video.yaml",
    ]
    for path in config_paths:
        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        assert isinstance(cfg, dict)
        benchmark = cfg["benchmark"]
        assert required_fields.issubset(benchmark.keys())
        assert benchmark["full_pipeline_plugin_mode"] == "experiment_config"
        assert benchmark["component_plugins"]["status"] == "config_driven_guarded"
        assert benchmark["component_plugins"]["vision_encoder"]["config_section"] == "model.vision_encoder"
        assert benchmark["component_plugins"]["mllm"]["config_section"] == "model.mllm"
        assert benchmark["component_plugins"]["mllm"]["official_processor_path"] is True
        assert benchmark["heavy_benchmark"] is False
        assert benchmark["run_by_default"] is False
        if benchmark["preset_id"] == "poc_default":
            assert benchmark["num_frames"] == 16


def test_priority3_high_resolution_configs_load_with_safety_limits() -> None:
    config_paths = [
        ROOT / "configs" / "benchmark" / "poc_high_resolution_chop_smoke.yaml",
        ROOT / "configs" / "benchmark" / "poc_high_resolution_chop_medium.yaml",
        ROOT / "configs" / "benchmark" / "poc_full_length_video_export_smoke.yaml",
    ]
    for path in config_paths:
        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        assert isinstance(cfg, dict)
        benchmark = cfg["benchmark"]
        assert benchmark["full_pipeline_plugin_mode"] == "experiment_config"
        assert benchmark["component_plugins"]["vision_encoder"]["enabled_in_full_pipeline"] is True
        assert benchmark["component_plugins"]["mllm"]["generation_method"] == "generate"
        assert benchmark["batch_size"] == 1
        assert benchmark["num_frames"] <= 32
        assert benchmark["max_chops"] <= 48
        assert benchmark["benchmark_iterations"] <= 3
        assert benchmark["heavy_benchmark"] is False
        assert benchmark["run_by_default"] is False
