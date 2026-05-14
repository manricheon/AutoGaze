from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import infer_autogaze
import infer_full
from poc_infer_utils import load_config
from poc_model_registry import build_mllm, build_vision_encoder, get_mllm_registry_metadata


CANONICAL_CONFIGS = {
    "A0_vanilla_siglip_nvila_off.yaml": ("A0", "vanilla_siglip", "nvila"),
    "A1_modified_siglip_nvila_off.yaml": ("A1", "modified_siglip", "nvila"),
    "A2_modified_siglip_nvila_on.yaml": ("A2", "modified_siglip", "nvila"),
    "A3_vanilla_siglip_nvila_on.yaml": ("A3", "vanilla_siglip", "nvila"),
}


def _cfg(name: str) -> Path:
    return ROOT / "configs" / "poc_inference" / name


def _external_cfg(name: str) -> Path:
    return ROOT / "configs" / "poc_inference" / "external" / name


def test_canonical_a0_to_a3_configs_still_load_and_route_to_nvila() -> None:
    for filename, (experiment_id, vision_name, mllm_name) in CANONICAL_CONFIGS.items():
        cfg = load_config(_cfg(filename))
        assert cfg["experiment"]["id"] == experiment_id
        assert cfg["vision_encoder"]["name"] == vision_name
        assert cfg["mllm"]["name"] == mllm_name
        assert cfg["mllm"]["generation_input_mode"] == "official_processor"
        assert build_mllm(cfg["mllm"]["name"]).name == "nvila"
        assert build_vision_encoder(cfg["vision_encoder"]["name"]).name == vision_name
    assert get_mllm_registry_metadata("nvila")["status"] == "implemented"


def test_infer_full_canonical_configs_route_to_nvila_without_real_loading(tmp_path: Path) -> None:
    for filename, (_experiment_id, _vision_name, _mllm_name) in CANONICAL_CONFIGS.items():
        args = infer_full.parse_args(
            [
                "--config",
                str(_cfg(filename)),
                "--video-path",
                "dummy",
                "--query-text",
                "Describe the video.",
                "--output-dir",
                str(tmp_path / filename.replace(".yaml", "")),
                "--num-frames",
                "2",
                "--resolution",
                "32",
                "--no-progress",
            ]
        )
        summary = infer_full.run(args)
        assert summary["adapter_statuses"]["mllm"]["name"] == "nvila"
        assert summary["metrics"]["actual_mllm"] == "nvila"
        assert summary["metrics"]["generation_input_mode"] == "official_processor"
        assert "answer" in summary["artifacts"]
        assert "metrics_json" in summary["artifacts"]
        assert "poc_summary" in summary["artifacts"]


def test_infer_autogaze_dummy_path_still_writes_flat_artifacts(tmp_path: Path) -> None:
    args = infer_autogaze.parse_args(
        [
            "--config",
            str(_cfg("A2_modified_siglip_nvila_on.yaml")),
            "--video-path",
            "dummy",
            "--output-dir",
            str(tmp_path / "autogaze_only"),
            "--num-frames",
            "2",
            "--resolution",
            "32",
            "--no-progress",
        ]
    )
    summary = infer_autogaze.run(args)
    assert summary["mode"] == "autogaze_only"
    assert summary["status"] in {"partial", "completed"}
    assert "token_counts_summary" in summary["artifacts"]
    assert "selected_patch_indices" in summary["artifacts"]
    assert "metrics_json" in summary["artifacts"]


def test_first_external_mllm_smoke_config_loads() -> None:
    cfg = load_config(_external_cfg("first_external_mllm_smoke.yaml"))
    assert cfg["mllm"]["name"] == "qwen2_5_vl"
    assert cfg["mllm"]["model_id"] == "weights/Qwen2.5-VL-7B-Instruct"
    assert cfg["mllm"]["generation_input_mode"] == "autogaze_frame_selection"
    assert cfg["mllm"]["direct_visual_token_injection_supported"] is False
    assert cfg["autogaze"]["enabled"] is True
    assert cfg["output"]["status"] == "stub-only"


def test_first_vjepa2_smoke_config_loads() -> None:
    cfg = load_config(_external_cfg("first_vjepa2_smoke.yaml"))
    assert cfg["vision_encoder"]["name"] == "vjepa2"
    assert cfg["vision_encoder"]["integration_mode"] == "vjepa2_feature_extraction"
    assert cfg["vision_encoder"]["decoder_type"] == "temporal_pooling_feature_probe_stub"
    assert cfg["vision_encoder"]["required_for_full_pipeline"] is True
    assert cfg["mllm"]["name"] == "generic_mllm"
    assert cfg["output"]["status"] == "stub-only"


def test_selected_external_mllm_routes_to_qwen25vl_adapter_without_fallback(tmp_path: Path) -> None:
    args = infer_full.parse_args(
        [
            "--config",
            str(_external_cfg("first_external_mllm_smoke.yaml")),
            "--video-path",
            "dummy",
            "--query-text",
            "Describe the video.",
            "--output-dir",
            str(tmp_path / "qwen_smoke"),
            "--num-frames",
            "2",
            "--resolution",
            "32",
            "--no-progress",
        ]
    )
    summary = infer_full.run(args)
    assert summary["adapter_statuses"]["mllm"]["name"] == "qwen2_5_vl"
    assert summary["metrics"]["actual_mllm"] == "qwen2_5_vl"
    assert summary["metrics"]["generation_input_mode"] == "autogaze_frame_selection"
    assert "nvila" not in str(summary["adapter_statuses"]["mllm"]).lower()
    assert "modified_siglip" not in str(summary["adapter_statuses"]["vision_encoder"]).lower()


def test_selected_vjepa2_smoke_routes_to_vjepa2_adapter_without_fallback(tmp_path: Path) -> None:
    args = infer_full.parse_args(
        [
            "--config",
            str(_external_cfg("first_vjepa2_smoke.yaml")),
            "--video-path",
            "dummy",
            "--query-text",
            "Extract features.",
            "--output-dir",
            str(tmp_path / "vjepa2_smoke"),
            "--num-frames",
            "2",
            "--resolution",
            "32",
            "--no-progress",
        ]
    )
    summary = infer_full.run(args)
    assert summary["adapter_statuses"]["vision_encoder"]["name"] == "vjepa2"
    assert summary["metrics"]["actual_vision_encoder"] == "vjepa2"
    assert summary["adapter_statuses"]["mllm"]["name"] == "generic_mllm"
    assert "modified_siglip" not in str(summary["adapter_statuses"]["vision_encoder"]).lower()
    assert "nvila" not in str(summary["adapter_statuses"]["mllm"]).lower()


def test_selected_unsupported_modes_raise_clear_errors() -> None:
    qwen = build_mllm("qwen2_5_vl")
    assert qwen.supports_direct_visual_tokens() is False
    with pytest.raises(NotImplementedError, match="No fallback"):
        qwen.run_direct_visual_token_injection_path()

    vjepa2 = build_vision_encoder("vjepa2")
    with pytest.raises(NotImplementedError, match="sparse_tubelet"):
        vjepa2.run_sparse_tubelet()
    with pytest.raises(NotImplementedError, match="context_mask"):
        vjepa2.run_context_mask_probe()


def test_first_external_smoke_targets_doc_exists_and_labels_commands() -> None:
    doc = ROOT / "docs" / "FIRST_EXTERNAL_SMOKE_TARGETS.md"
    text = doc.read_text(encoding="utf-8")
    assert "Selected Table MLLM Target" in text
    assert "Qwen2.5-VL-7B" in text
    assert "Selected V-JEPA2 Target" in text
    assert "vjepa2_feature_extraction" in text
    assert "stub-only" in text
    assert "blocked" in text.lower()
    assert "direct_visual_token_injection" in text
