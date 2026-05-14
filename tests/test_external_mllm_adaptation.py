from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import infer_full
from poc_infer_utils import load_config
from poc_model_registry import (
    MLLMS,
    build_mllm,
    build_vision_encoder,
    get_mllm_registry_metadata,
    get_vision_encoder_registry_metadata,
)


TARGET_MODELS = {
    "llava_ov",
    "longva",
    "longvila_r1",
    "apollo",
    "videollama3",
    "videochat_flash",
    "internvl3_5",
    "qwen2_5_vl",
}

EXTERNAL_CONFIGS = [
    "llava_ov_autogaze_input_selection.yaml",
    "llava_ov_siglip_sparse_candidate.yaml",
    "longva_siglip_candidate.yaml",
    "longvila_r1_siglip_candidate.yaml",
    "apollo_siglip_branch_candidate.yaml",
    "videollama3_siglip_candidate.yaml",
    "videochat_flash_input_selection.yaml",
    "internvl3_5_input_selection.yaml",
    "qwen2_5_vl_input_selection.yaml",
]

VJEPA2_CONFIGS = [
    "vjepa2_official_dense.yaml",
    "vjepa2_video_classification.yaml",
    "vjepa2_feature_extraction.yaml",
    "vjepa2_autogaze_frame_selection.yaml",
    "vjepa2_autogaze_chop_selection.yaml",
    "vjepa2_zero_mask.yaml",
    "vjepa2_context_mask_probe.yaml",
    "vjepa2_sparse_tubelet_experimental.yaml",
    "vjepa2_generic_decoder_stub.yaml",
    "vjepa2_mllm_projector_blocked.yaml",
]

REGISTRY_SCHEMA_FIELDS = {
    "model_name",
    "default_model_id",
    "adapter_class",
    "vision_encoder_type",
    "siglip_based",
    "positional_encoding_type",
    "positional_encoding_status",
    "dense_grid_dependency",
    "variable_visual_token_support",
    "visual_placeholder_dynamic",
    "direct_sparse_autogaze_supported",
    "requires_training_or_trainable_adapter",
    "recommended_modes",
    "risk_level",
    "status",
}


def _external_cfg(name: str) -> Path:
    return ROOT / "configs" / "poc_inference" / "external" / name


def test_external_registry_contains_all_target_models() -> None:
    metadata = get_mllm_registry_metadata()
    assert TARGET_MODELS.issubset(MLLMS)
    assert TARGET_MODELS.issubset(metadata)
    for required_name in ("nvila", "generic_mllm"):
        assert required_name in metadata
    for name in TARGET_MODELS:
        item = metadata[name]
        assert REGISTRY_SCHEMA_FIELDS.issubset(item)
        assert item["model_name"] == name
        assert item["adapter_class"]
        assert item["supported_integration_modes"]
        assert item["recommended_mode"]
        assert item["risk_level"] in {"low", "medium", "high"}
        assert item["status"] in {"stub-only", "blocked", "future work"}


def test_external_adapters_load_lazily_without_heavy_model_fallbacks() -> None:
    for name in TARGET_MODELS:
        adapter = build_mllm(name, {"model_id": f"local/{name}"})
        status = adapter.load(allow_real_model_loading=False)
        assert status.status == "stub-only"
        assert adapter.name == name
        assert adapter.model is None
        assert "nvila" not in str(status.metadata.get("actual_model_loaded", "")).lower()
        assert "modified_siglip" not in str(status.metadata).lower()


def test_external_adapters_do_not_silently_fallback_to_nvila_or_modified_siglip() -> None:
    for name in TARGET_MODELS:
        adapter = build_mllm(name)
        assert adapter.name == name
        assert adapter.name != "nvila"
    with pytest.raises(ValueError):
        build_mllm("llava_ov_typo")


def test_canonical_registry_and_configs_are_preserved() -> None:
    expected = {
        "A0_vanilla_siglip_nvila_off.yaml": ("A0", "vanilla_siglip", "nvila"),
        "A1_modified_siglip_nvila_off.yaml": ("A1", "modified_siglip", "nvila"),
        "A2_modified_siglip_nvila_on.yaml": ("A2", "modified_siglip", "nvila"),
        "A3_vanilla_siglip_nvila_on.yaml": ("A3", "vanilla_siglip", "nvila"),
    }
    for filename, (experiment_id, vision_name, mllm_name) in expected.items():
        cfg = load_config(ROOT / "configs" / "poc_inference" / filename)
        assert cfg["experiment"]["id"] == experiment_id
        assert cfg["vision_encoder"]["name"] == vision_name
        assert cfg["mllm"]["name"] == mllm_name
        assert cfg["mllm"]["generation_input_mode"] == "official_processor"
    assert build_mllm("nvila").name == "nvila"
    assert get_mllm_registry_metadata("nvila")["status"] == "implemented"


def test_no_silent_fallback_for_qwen_or_external_vision_encoder() -> None:
    assert build_mllm("qwen").name == "qwen"
    assert build_mllm("qwen2_5_vl").name == "qwen2_5_vl"
    assert build_vision_encoder("vjepa2").name == "vjepa2"
    assert build_vision_encoder("external").name == "external"
    assert build_vision_encoder("generic_vit").name == "generic_vit"
    assert build_vision_encoder("external").name != "modified_siglip"
    vision_metadata = get_vision_encoder_registry_metadata()
    assert "external" in vision_metadata
    assert vision_metadata["external"]["adapter_class"] == "ExternalVisionEncoderAdapter"


def test_unsupported_modes_raise_clear_not_implemented_errors() -> None:
    blocked_cases = {
        "llava_ov": "siglip_sparse_patch",
        "longva": "siglip_sparse_patch",
        "longvila_r1": "siglip_sparse_patch",
        "apollo": "siglip_sparse_patch",
        "videollama3": "siglip_sparse_patch",
        "videochat_flash": "direct_visual_token_injection",
        "internvl3_5": "direct_visual_token_injection",
        "qwen2_5_vl": "direct_visual_token_injection",
    }
    method_by_mode = {
        "siglip_sparse_patch": "run_siglip_sparse_patch_path",
        "rope_sparse_patch": "run_rope_sparse_patch_path",
        "direct_visual_token_injection": "run_direct_visual_token_injection_path",
    }
    for name, mode in blocked_cases.items():
        adapter = build_mllm(name)
        method = getattr(adapter, method_by_mode[mode])
        with pytest.raises(NotImplementedError, match="No fallback"):
            method()

    for name in TARGET_MODELS:
        adapter = build_mllm(name)
        with pytest.raises(NotImplementedError, match="No fallback"):
            adapter.run_rope_sparse_patch_path()


def test_direct_visual_token_injection_disabled_for_non_sparse_models() -> None:
    for name in ("qwen2_5_vl", "internvl3_5", "videochat_flash"):
        metadata = get_mllm_registry_metadata(name)
        adapter = build_mllm(name)
        assert metadata["direct_token_injection_supported"] is False
        assert adapter.supports_direct_visual_tokens() is False
        with pytest.raises(NotImplementedError):
            adapter.run_direct_visual_token_injection_path()


def test_siglip_candidates_have_explicit_candidate_status() -> None:
    for name in ("llava_ov", "longvila_r1", "videollama3"):
        status = get_mllm_registry_metadata(name)["compatibility_status"]
        assert status in {"siglip_candidate", "native_candidate", "needs_code_inspection"}


def test_external_configs_load_and_are_not_marked_runnable() -> None:
    for name in EXTERNAL_CONFIGS:
        cfg = load_config(_external_cfg(name))
        assert cfg["mllm"]["name"] in TARGET_MODELS
        assert cfg["mllm"]["generation_input_mode"] in infer_full.MLLM_GENERATION_INPUT_MODES
        assert cfg["mllm"]["status"] in {"stub-only", "blocked", "future work"}
        assert cfg["runtime"]["allow_real_model_loading"] is False
        assert cfg["vision_encoder"]["name"] == "generic_vit"
        assert cfg["vision_encoder"]["required_for_full_pipeline"] is False


def test_generation_input_mode_normalization() -> None:
    assert infer_full._normalize_generation_input_mode("official_processor") == "official_processor"
    assert infer_full._normalize_generation_input_mode("rope_sparse_patch") == "rope_sparse_patch"
    assert infer_full._normalize_generation_input_mode("direct_visual_tokens") == "direct_visual_token_injection"
    with pytest.raises(ValueError):
        infer_full._normalize_generation_input_mode("silent_nvila_fallback")


def test_infer_full_cli_routes_external_model_to_requested_adapter(tmp_path: Path) -> None:
    args = infer_full.parse_args(
        [
            "--config",
            str(ROOT / "configs" / "poc_inference" / "A2_modified_siglip_nvila_on.yaml"),
            "--video-path",
            "dummy",
            "--query-text",
            "Describe the video.",
            "--output-dir",
            str(tmp_path / "llava_route"),
            "--mllm",
            "llava_ov",
            "--integration-mode",
            "autogaze_frame_selection",
            "--model-id",
            "local/llava-ov-test",
            "--vision-encoder",
            "external",
            "--num-frames",
            "2",
            "--resolution",
            "32",
            "--no-progress",
        ]
    )
    summary = infer_full.run(args)
    assert summary["adapter_statuses"]["mllm"]["name"] == "llava_ov"
    assert summary["adapter_statuses"]["vision_encoder"]["name"] == "external"
    assert summary["metrics"]["actual_mllm"] == "llava_ov"
    assert summary["metrics"]["actual_vision_encoder"] == "external"
    assert summary["metrics"]["generation_input_mode"] == "autogaze_frame_selection"
    assert "nvila" not in str(summary["adapter_statuses"]["mllm"]).lower()


def test_mllm_adapt_report_exists_and_covers_all_models() -> None:
    report = ROOT / "docs" / "mllm_adapt_report.md"
    text = report.read_text(encoding="utf-8")
    for display_name in (
        "LLaVA-OV / LLaVA-OneVision 8B",
        "LongVA-7B",
        "LongVILA-R1-7B",
        "Apollo-7B",
        "VideoLLaMA3-7B",
        "VideoChat-Flash",
        "InternVL3.5-8B",
        "Qwen2.5-VL-7B",
    ):
        section_start = text.index(f"### {display_name}")
        section = text[section_start : text.find("\n### ", section_start + 1) if "\n### " in text[section_start + 1 :] else len(text)]
        assert "Recommendation:" in section
        assert "Risk:" in section
    assert "Direct sparse feasibility" in text
    assert "rope_sparse_candidate" in text
    assert "positional encoding" in text.lower()


def test_generic_vit_compatibility_document_exists_and_covers_required_topics() -> None:
    doc = ROOT / "docs" / "generic_vit_autogaze_compatibility.md"
    text = doc.read_text(encoding="utf-8")
    for required in (
        "What AutoGaze Provides",
        "Application Levels",
        "Positional Encoding Compatibility",
        "Direct Sparse Integration Criteria",
        "Non-Training Rule",
        "rope_sparse_candidate",
        "M-RoPE",
        "Perceiver / Q-Former / resampler",
        "direct visual token injection",
        "native_sparse_patch",
        "light_modified_sparse",
        "autogaze_zero_mask",
        "post_encoder_zero_mask",
        "input_selection_only",
        "unsupported_for_now",
        "V-JEPA2 Implications",
    ):
        assert required in text


def test_adapter_compatibility_reporting_methods() -> None:
    for name in TARGET_MODELS:
        adapter = build_mllm(name)
        assert adapter.inspect_vision_encoder()["vision_encoder_type"]
        assert adapter.inspect_positional_encoding()["positional_encoding_type"]
        assert "variable_visual_token_support" in adapter.inspect_projector()
        assert "visual_placeholder_dynamic" in adapter.inspect_placeholder_handling()
        assert adapter.supports_direct_sparse_autogaze() is False
        assert adapter.supports_input_level_selection() is True
        assert isinstance(adapter.supports_post_encoder_pruning(), bool)
        report = adapter.status_report()
        assert report["inspect_positional_encoding"]["positional_encoding_type"]
        assert "supports_direct_sparse_autogaze" in report


def test_vjepa2_registry_entry_and_lazy_adapter_status() -> None:
    metadata = get_vision_encoder_registry_metadata("vjepa2")
    assert metadata["model_type"] == "video_encoder"
    assert metadata["positional_encoding_type"] == "3d_rope"
    assert metadata["patch_structure"] == "tubelet"
    assert metadata["patch_size"] == 16
    assert metadata["crop_size"] == 256
    assert metadata["frames_per_clip"] == 64
    assert metadata["tubelet_size"] == 2
    assert metadata["supports_official_dense"] is True
    assert metadata["supports_autogaze_frame_selection"] is True
    assert metadata["supports_autogaze_chop_selection"] is True
    assert metadata["supports_autogaze_zero_mask"] is True
    assert metadata["supports_context_mask_probe"] == "unknown"
    assert metadata["supports_sparse_tubelet"] == "unknown"
    assert metadata["supports_direct_mllm_projection"] is False
    assert metadata["status"] == "needs_code_inspection"

    adapter = build_vision_encoder("vjepa2")
    status = adapter.load(allow_real_model_loading=False)
    assert status.status == "stub"
    assert adapter.model is None
    report = adapter.status_report()
    assert report["position_encoding"]["positional_encoding_type"] == "3d_rope"
    assert report["position_encoding"]["patch_structure"] == "tubelet"
    assert report["tubelet_grid"] == (32, 16, 16)
    assert report["supports_official_dense"] is True
    assert report["supports_autogaze_zero_mask"] is True


def test_vjepa2_registered_modes_and_blocked_sparse_paths() -> None:
    adapter = build_vision_encoder("vjepa2")
    for method_name in (
        "run_official_dense",
        "run_video_classification",
        "run_feature_extraction",
        "run_autogaze_frame_selection",
        "run_autogaze_chop_selection",
        "run_autogaze_zero_mask",
    ):
        result = getattr(adapter, method_name)()
        assert result["status"] == "stub-only"
        assert result["metadata"]["adapter"] == "vjepa2"

    with pytest.raises(NotImplementedError, match="context_mask"):
        adapter.run_context_mask_probe()
    with pytest.raises(NotImplementedError, match="sparse_tubelet"):
        adapter.run_sparse_tubelet()


def test_vjepa2_mllm_projection_blocked_without_verified_frozen_projector() -> None:
    adapter = build_vision_encoder("vjepa2")
    assert adapter.supports_mllm_projection("nvila")["supported"] is False
    for target in ("qwen2_5_vl", "internvl3_5", "videochat_flash"):
        projection = adapter.supports_mllm_projection(target)
        assert projection["status"] == "blocked_without_training"
        assert projection["requires_training"] is True


def test_vjepa2_configs_load_and_are_not_marked_runnable() -> None:
    for name in VJEPA2_CONFIGS:
        cfg = load_config(_external_cfg(name))
        assert cfg["vision_encoder"]["name"] == "vjepa2"
        assert cfg["vision_encoder"]["model_type"] == "video_encoder"
        assert cfg["vision_encoder"]["patch_size"] == 16
        assert cfg["vision_encoder"]["crop_size"] == 256
        assert cfg["vision_encoder"]["frames_per_clip"] == 64
        assert cfg["vision_encoder"]["tubelet_size"] == 2
        assert cfg["runtime"]["allow_real_model_loading"] is False
        assert cfg["vision_encoder"]["status"] in {"stub-only", "blocked", "future work"}


def test_vjepa2_report_sections_and_tables_exist() -> None:
    report = ROOT / "docs" / "mllm_adapt_report.md"
    text = report.read_text(encoding="utf-8")
    assert "Section A: AutoGaze Integration Feasibility for Table-Model MLLMs" in text
    assert "Section B: V-JEPA2 as an Alternative Video Encoder" in text
    assert "V-JEPA2 Decoder Recommendations" in text
    assert "V-JEPA2 as Vision Encoder for Candidate MLLMs" in text
    assert "context_mask" in text
    assert "sparse tubelet" in text.lower()
    assert "blocked_without_training" in text
