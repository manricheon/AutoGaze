from __future__ import annotations

from pathlib import Path

from autogaze_ext.investigation import locate_quick_start
from autogaze_ext.pipeline.runner import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_quick_start_can_be_located_from_configured_path(tmp_path: Path) -> None:
    source_dir = tmp_path / "original_autogaze"
    source_dir.mkdir()
    quick_start = source_dir / "QUICK_START.md"
    quick_start.write_text("# Quick Start\n", encoding="utf-8")

    located = locate_quick_start(source_dir, repo_root=tmp_path)

    assert located.path == quick_start.resolve()
    assert located.source == "configured_path"


def test_inference_guide_and_quick_start_reference_exist() -> None:
    inference_guide = ROOT / "docs" / "INFERENCE_GUIDE.md"
    quick_start_reference = ROOT / "docs" / "QUICK_START_reference.md"
    nvila_source = ROOT / "docs" / "nvila-hd-video-readme.md"
    nvila_reference = ROOT / "docs" / "NVILA_HD_VIDEO_REFERENCE.md"
    poc_guide = ROOT / "docs" / "POC_NVILA_HD_VIDEO_GUIDE.md"

    assert inference_guide.exists()
    assert quick_start_reference.exists()
    assert nvila_source.exists()
    assert nvila_reference.exists()
    assert poc_guide.exists()

    guide_text = inference_guide.read_text(encoding="utf-8")
    for section in [
        "AutoGaze-Only Inference",
        "Full Pipeline Inference",
        "NVILA-HD-Video Canonical Path",
        "Query Text Handling",
        "Resolution Scaling",
        "Current Limitations",
    ]:
        assert section in guide_text
    assert "stub-only / future work" in guide_text
    assert "AutoProcessor" in guide_text
    assert "AutoModel" in guide_text
    assert "docs/POC_NVILA_HD_VIDEO_GUIDE.md" in guide_text

    poc_text = poc_guide.read_text(encoding="utf-8")
    for expected in [
        "scripts/poc_nvila_hd_video.py",
        "check",
        "autogaze_only",
        "full_pipeline",
        "full_length",
        "overlay_union",
        "autogaze_chop_overlay.mp4",
        "poc_summary.json",
        "Output Type Samples by Mode",
        "ASCII patch mask example",
        "Scale panel video",
        "Full-length video export",
        "chop_overlay_metadata.json",
        "Frame Selection ASCII Examples",
        "window_000 = [0, 3, 6, 9]",
        "padded_frame_mask",
        "frame_selection_metadata.json",
        "num_frames = 16",
        "configs/benchmark/poc_default.yaml",
        "Full Pipeline Component Plug-In Mode",
        "full_pipeline_plugin_mode: experiment_config",
        "module/class/checkpoint overrides are config-driven, not CLI flags",
        "Current CLI Surface",
        "A1 and A2 Canonical Config Guide",
        "configs/experiment/A1_real.yaml",
        "configs/experiment/A2_real.yaml",
        "A1 Full-Token Full Pipeline",
        "A2 AutoGaze-Only Visualization",
        "A2 Full Pipeline With Query Text",
        "Do not claim encoder-side acceleration unless A2 reduces tokens before the intended encoder compute stage.",
    ]:
        assert expected in poc_text

    assert "default is `16`" in guide_text
    assert "configs/benchmark/poc_default.yaml" in guide_text


def test_nvila_hd_video_reference_contains_required_extractions() -> None:
    text = (ROOT / "docs" / "NVILA_HD_VIDEO_REFERENCE.md").read_text(encoding="utf-8")

    for expected in [
        "nvidia/NVILA-8B-HD-Video",
        "AutoProcessor.from_pretrained",
        "AutoModel.from_pretrained",
        "weights/NVILA-8B-HD-Video",
        "processor.tokenizer.video_token",
        "videos=video_path",
        "gazing_ratio_tile",
        "max_tiles_video",
        "processor.batch_decode",
        "not an official NVILA-HD-Video guide command",
        "scripts/poc_nvila_hd_video.py",
    ]:
        assert expected in text


def test_quick_start_reference_contains_required_extractions() -> None:
    text = (ROOT / "docs" / "QUICK_START_reference.md").read_text(encoding="utf-8")

    for expected in [
        "AutoGaze.from_pretrained",
        "bfshi/AutoGaze",
        "SiglipVisionModel",
        "gazing_ratio",
        "task_loss_requirement",
        "target_scales",
        "target_patch_size",
        "Query text or prompt arguments are not present",
        "No shell CLI is described",
    ]:
        assert expected in text


def test_a1_a2_real_configs_include_quick_start_alignment_fields() -> None:
    for config_name in ["experiment/A1_real", "experiment/A2_real"]:
        cfg = load_config(ROOT / "configs", config_name)

        assert cfg.inference.input_resolution == 224
        assert cfg.inference.scaling_mode == "resize"
        assert cfg.inference.frame_count == 16
        assert cfg.inference.patch_size == 16
        assert "target_scales" in cfg.inference
        assert "target_patch_size" in cfg.inference
        assert "spatial_tile_size" in cfg.inference
        assert "checkpoint_root" in cfg.inference
        assert "inference_script_path" in cfg.inference
        assert "resolution" in cfg.inference
        assert "num_frames" in cfg.inference
        assert cfg.inference.frame_selection_mode == "sample"
        assert cfg.inference.frame_interval == 1
        assert "max_windows" in cfg.inference
        assert cfg.inference.drop_last is False
        assert cfg.inference.pad_last is False
        assert "query_text" in cfg.inference
        assert "prompt_template" in cfg.inference
        assert "video_preprocess_mode" in cfg.inference
        assert "nvila_hd_video_model_id" in cfg.inference
        assert "output_dir" in cfg.inference
        assert "visualization_dir" in cfg.inference
        assert cfg.model.mllm.nvila_hd_video_model_id == "nvidia/NVILA-8B-HD-Video"
        assert cfg.model.mllm.nvila_hd_video_module_path == "transformers"
        assert cfg.model.mllm.nvila_hd_video_class_name == "AutoModel"
        assert cfg.model.mllm.nvila_hd_video_processor_class_name == "AutoProcessor"
        assert cfg.model.mllm.nvila_hd_video_checkpoint_path == "weights/NVILA-8B-HD-Video"
        assert cfg.model.mllm.nvila_hd_video_config_path == "weights/NVILA-8B-HD-Video/config.json"
        assert cfg.model.mllm.processor_path == "weights/NVILA-8B-HD-Video"
        assert cfg.model.mllm.tokenizer_path == "weights/NVILA-8B-HD-Video"
        assert cfg.model.mllm.video_preprocess_mode == "official_processor"
        assert cfg.model.mllm.prompt_template == "{video_token}\n\n{prompt}"

    a2 = load_config(ROOT / "configs", "experiment/A2_real")
    assert a2.model.autogaze.input_resolution == 224
    assert a2.model.autogaze.frame_count == 16
    assert a2.model.autogaze.original_cli_args.gazing_ratio == 0.75
    assert a2.model.autogaze.original_cli_args.task_loss_requirement == 0.7

    a1 = load_config(ROOT / "configs", "experiment/A1_real")
    assert a1.model.vision_encoder.original_cli_args.gazing_info_argument == "gazing_info"
    assert a1.model.vision_encoder.original_cli_args.high_resolution_example.target_patch_size == 14
    assert a1.model.mllm.autogaze_enabled is False
    assert a2.model.mllm.autogaze_enabled is True
