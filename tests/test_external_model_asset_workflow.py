from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from external_model_asset_utils import load_asset_manifest, prioritized_model_names, select_model_names, select_models_by_tier
import infer_full
from poc_infer_utils import load_config, nested_get
from poc_model_registry import (
    build_mllm,
    build_vision_encoder,
    get_mllm_registry_metadata,
    get_vision_encoder_registry_metadata,
)


TARGET_MODELS = [
    "llava_ov",
    "longva",
    "longvila_r1",
    "apollo",
    "videollama3",
    "videochat_flash",
    "internvl3_5",
    "qwen2_5_vl",
    "vjepa2",
]


def run_cmd(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_asset_manifest_loads_all_target_models() -> None:
    manifest = load_asset_manifest("configs/poc_inference/model_asset_manifest.yaml")
    assert select_model_names(manifest, ["all"]) == TARGET_MODELS
    for name in TARGET_MODELS:
        entry = manifest["models"][name]
        assert entry["expected_adapter_name"]
        assert entry["local_target_directory"].startswith("weights/")


def test_model_tiers_are_defined_and_prioritized() -> None:
    manifest = load_asset_manifest("configs/poc_inference/model_asset_manifest.yaml")
    assert select_models_by_tier(manifest, ["tier1"]) == ["longvila_r1", "longva", "llava_ov", "videollama3", "apollo"]
    assert select_models_by_tier(manifest, ["tier1b"]) == ["vjepa2"]
    assert select_models_by_tier(manifest, ["tier2"]) == ["qwen2_5_vl", "internvl3_5", "videochat_flash"]
    assert prioritized_model_names(manifest, select_models_by_tier(manifest, ["tier1"]))[0] == "longvila_r1"


def test_downloader_dry_run_writes_report(tmp_path: Path) -> None:
    report = tmp_path / "download_report.md"
    result = run_cmd(
        [
            "scripts/prepare_external_model_assets.py",
            "--manifest",
            "configs/poc_inference/model_asset_manifest.yaml",
            "--model",
            "qwen2_5_vl",
            "--dry-run",
            "--local-files-only",
            "--write-report",
            str(report),
        ]
    )
    assert result.returncode == 0, result.stderr
    text = report.read_text(encoding="utf-8")
    assert "qwen2_5_vl" in text
    assert "dry-run" in text


def test_downloader_tier_topk_selects_longvila_without_download(tmp_path: Path) -> None:
    report = tmp_path / "tier1_report.md"
    result = run_cmd(
        [
            "scripts/prepare_external_model_assets.py",
            "--manifest",
            "configs/poc_inference/model_asset_manifest.yaml",
            "--tier",
            "tier1",
            "--select-top-k",
            "1",
            "--dry-run",
            "--write-report",
            str(report),
        ]
    )
    assert result.returncode == 0, result.stderr
    text = report.read_text(encoding="utf-8")
    assert "longvila_r1" in text
    assert "longva" not in text
    assert "| dry-run |" in text


def test_model_all_does_not_download_by_default(tmp_path: Path) -> None:
    report = tmp_path / "all_default_report.md"
    result = run_cmd(
        [
            "scripts/prepare_external_model_assets.py",
            "--manifest",
            "configs/poc_inference/model_asset_manifest.yaml",
            "--model",
            "all",
            "--write-report",
            str(report),
        ]
    )
    assert result.returncode == 0, result.stderr
    text = report.read_text(encoding="utf-8")
    assert "External Model Asset Download Dry-Run Report" in text
    assert "| download |" not in text


def test_verifier_reports_missing_paths_clearly(tmp_path: Path) -> None:
    report = tmp_path / "verify_report.md"
    result = run_cmd(
        [
            "scripts/verify_external_model_assets.py",
            "--manifest",
            "configs/poc_inference/model_asset_manifest.yaml",
            "--model",
            "llava_ov",
            "--weights-root",
            str(tmp_path / "missing_weights"),
            "--local-files-only",
            "--write-report",
            str(report),
        ]
    )
    assert result.returncode == 0, result.stderr
    text = report.read_text(encoding="utf-8")
    assert "llava_ov" in text
    assert "local directory missing" in text


def test_config_inspector_handles_missing_files_clearly(tmp_path: Path) -> None:
    report = tmp_path / "inspect_report.md"
    result = run_cmd(
        [
            "scripts/inspect_external_model_configs.py",
            "--manifest",
            "configs/poc_inference/model_asset_manifest.yaml",
            "--model",
            "llava_ov",
            "--weights-root",
            str(tmp_path / "missing_weights"),
            "--local-files-only",
            "--write-report",
            str(report),
        ]
    )
    assert result.returncode == 0, result.stderr
    text = report.read_text(encoding="utf-8")
    assert "llava_ov" in text
    assert "config.json not found" in text


@pytest.mark.parametrize("name", TARGET_MODELS)
def test_smoke_configs_load(name: str) -> None:
    cfg = load_config(REPO_ROOT / "configs" / "poc_inference" / "external" / f"smoke_{name}.yaml")
    if name == "vjepa2":
        assert nested_get(cfg, "vision_encoder.name") == "vjepa2"
    else:
        assert nested_get(cfg, "mllm.name") == name
    assert nested_get(cfg, "runtime.allow_real_model_loading") is False
    assert nested_get(cfg, "runtime.local_files_only") is True


def test_selected_smoke_configs_load_and_route_to_selected_adapters() -> None:
    tier1 = load_config(REPO_ROOT / "configs" / "poc_inference" / "external" / "selected_tier1_smoke.yaml")
    assert nested_get(tier1, "selection.selected_model") == "longvila_r1"
    assert nested_get(tier1, "mllm.name") == "longvila_r1"
    assert nested_get(tier1, "mllm.generation_input_mode") == "autogaze_zero_mask"
    assert build_mllm("longvila_r1", nested_get(tier1, "mllm", {})).name == "longvila_r1"

    vjepa2 = load_config(REPO_ROOT / "configs" / "poc_inference" / "external" / "selected_vjepa2_smoke.yaml")
    assert nested_get(vjepa2, "selection.selected_model") == "vjepa2"
    assert nested_get(vjepa2, "vision_encoder.name") == "vjepa2"
    assert build_vision_encoder("vjepa2", nested_get(vjepa2, "vision_encoder", {})).name == "vjepa2"


def test_registry_contains_asset_status_for_all_targets() -> None:
    mllm_meta = get_mllm_registry_metadata()
    vision_meta = get_vision_encoder_registry_metadata()
    for name in TARGET_MODELS:
        meta = vision_meta["vjepa2"] if name == "vjepa2" else mllm_meta[name]
        assert "local_asset_status" in meta
        assert "recommended_first_smoke_mode" in meta
        assert "official_processor_support_status" in meta


def test_adapters_load_lazily_without_real_import_or_fallback() -> None:
    qwen = build_mllm("qwen2_5_vl", {"checkpoint_path": "weights/Qwen2.5-VL-7B-Instruct"})
    status = qwen.load(allow_real_model_loading=False)
    assert qwen.name == "qwen2_5_vl"
    assert status.status == "stub-only"
    assert status.metadata["adapter"] == "qwen2_5_vl"

    vjepa2 = build_vision_encoder("vjepa2", {"checkpoint_path": "weights/vjepa2-vitl-fpc64-256"})
    status = vjepa2.load(allow_real_model_loading=False)
    assert vjepa2.name == "vjepa2"
    assert status.status in {"stub", "stub-only"}
    assert vjepa2.name != "modified_siglip"


def test_no_silent_fallback_for_external_models() -> None:
    assert build_mllm("longva", {}).name == "longva"
    assert build_mllm("qwen2_5_vl", {}).name == "qwen2_5_vl"
    assert build_vision_encoder("vjepa2", {}).name == "vjepa2"
    with pytest.raises(ValueError):
        build_mllm("llava_ov_missing_alias", {})


@pytest.mark.parametrize("name", ["qwen2_5_vl", "internvl3_5", "videochat_flash"])
def test_direct_visual_token_injection_disabled_by_default(name: str) -> None:
    adapter = build_mllm(name, {})
    assert adapter.supports_direct_visual_tokens() is False
    with pytest.raises(NotImplementedError):
        adapter.run_direct_visual_token_injection_path()


def test_rope_sparse_patch_not_enabled_unless_verified() -> None:
    for name in ("qwen2_5_vl", "videollama3", "longvila_r1"):
        adapter = build_mllm(name, {})
        assert adapter.supports_rope_sparse_patch() is False
        with pytest.raises(NotImplementedError):
            adapter.run_rope_sparse_patch_path()


def test_vjepa2_to_mllm_projector_blocked_without_verified_projector() -> None:
    adapter = build_vision_encoder("vjepa2", {})
    report = adapter.supports_mllm_projection("qwen2_5_vl")
    assert report["supported"] is False
    assert report["status"] == "blocked_without_training"


def test_smoke_runner_dry_run_reports_blocked_for_missing_assets(tmp_path: Path) -> None:
    output_dir = tmp_path / "smoke"
    result = run_cmd(
        [
            "scripts/run_external_model_smoke.py",
            "--config",
            "configs/poc_inference/external/smoke_llava_ov.yaml",
            "--video-path",
            "dummy",
            "--query-text",
            "Describe the video.",
            "--output-dir",
            str(output_dir),
            "--dry-run",
            "--local-files-only",
        ]
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads((output_dir / "logs" / "poc_summary.json").read_text(encoding="utf-8"))
    assert summary["model"] == "llava_ov"
    assert summary["status"] == "blocked"
    assert summary["adapter_route"]["adapter"] == "llava_ov"
    assert summary["assets"]["download_status"] == "missing"


def test_selected_tier1_zero_mask_dry_run_does_not_claim_acceleration(tmp_path: Path) -> None:
    output_dir = tmp_path / "selected_tier1"
    result = run_cmd(
        [
            "scripts/run_external_model_smoke.py",
            "--config",
            "configs/poc_inference/external/selected_tier1_smoke.yaml",
            "--video-path",
            "dummy",
            "--query-text",
            "Describe the video.",
            "--output-dir",
            str(output_dir),
            "--dry-run",
            "--local-files-only",
        ]
    )
    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "logs" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["integration_mode"] == "autogaze_zero_mask"
    assert metrics["zero_mask_stage"] == "pixel"
    assert metrics["zero_mask_encoder_compute_reduction"] is False
    assert metrics["zero_mask_expected_speedup"] == "none"


def test_selected_tier1_dummy_weight_smoke_runs_without_real_checkpoint(tmp_path: Path) -> None:
    output_dir = tmp_path / "selected_tier1_dummy"
    result = run_cmd(
        [
            "scripts/run_external_model_smoke.py",
            "--config",
            "configs/poc_inference/external/selected_tier1_smoke.yaml",
            "--video-path",
            "dummy",
            "--query-text",
            "Describe the video.",
            "--output-dir",
            str(output_dir),
            "--allow-dummy-weights",
            "--local-files-only",
            "--max-new-tokens",
            "8",
        ]
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads((output_dir / "logs" / "poc_summary.json").read_text(encoding="utf-8"))
    answer = json.loads((output_dir / "predictions" / "answer.json").read_text(encoding="utf-8"))
    assert summary["adapter_statuses"]["mllm"]["status"] == "dummy"
    assert summary["metrics"]["dummy_weights_enabled"] is True
    assert answer["status"] == "dummy"
    assert answer["answer"].startswith("[dummy:longvila_r1]")
    assert answer["generation_metadata"]["real_checkpoint_loaded"] is False


def test_infer_full_zero_mask_metrics_do_not_claim_encoder_acceleration(tmp_path: Path) -> None:
    args = infer_full.parse_args(
        [
            "--config",
            "configs/poc_inference/external/selected_tier1_smoke.yaml",
            "--video-path",
            "dummy",
            "--query-text",
            "Describe the video.",
            "--output-dir",
            str(tmp_path / "infer_full_zero_mask"),
            "--num-frames",
            "2",
            "--resolution",
            "32",
            "--no-progress",
        ]
    )
    summary = infer_full.run(args)
    metrics = summary["metrics"]
    assert metrics["generation_input_mode"] == "autogaze_zero_mask"
    assert metrics["zero_mask_stage"] == "pixel"
    assert metrics["zero_mask_encoder_compute_reduction"] is False
    assert metrics["zero_mask_expected_speedup"] == "none"


def test_report_documents_exist() -> None:
    assert (REPO_ROOT / "docs" / "MODEL_ASSET_MANIFEST.md").exists()
    assert (REPO_ROOT / "docs" / "EXTERNAL_MODEL_SMOKE_PLAN.md").exists()
    assert (REPO_ROOT / "docs" / "mllm_adapt_report.md").exists()
    assert (REPO_ROOT / "docs" / "generic_vit_autogaze_compatibility.md").exists()
