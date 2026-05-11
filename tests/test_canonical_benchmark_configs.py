from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
PRESETS = [
    "canonical_a1_small",
    "canonical_a2_small",
    "canonical_a1_medium",
    "canonical_a2_medium",
]


def _load_preset(name: str):
    return OmegaConf.load(ROOT / "configs" / "benchmark" / f"{name}.yaml").benchmark


def test_canonical_benchmark_presets_exist_and_include_configurable_axes() -> None:
    required_axes = {
        "num_frames",
        "resolution",
        "scale_resolution",
        "token_budget",
        "warmup_iterations",
        "benchmark_iterations",
        "max_new_tokens",
        "batch_size",
        "device",
        "dtype",
    }

    for name in PRESETS:
        cfg = _load_preset(name)

        assert cfg.preset_id == name
        assert cfg.experiment in {"A1_real", "A2_real"}
        assert required_axes.issubset(set(cfg.keys()))
        assert set(cfg.modes) == {"autogaze_only", "full_pipeline"}
        assert cfg.status == "config_template_only_not_paper_reproduction"


def test_canonical_benchmark_presets_keep_safe_defaults() -> None:
    for name in PRESETS:
        cfg = _load_preset(name)

        assert cfg.batch_size == 1
        assert cfg.num_frames <= 16
        assert cfg.resolution <= 392
        assert cfg.max_new_tokens == 1
        assert cfg.allow_mllm_load is False
        assert cfg.auto_download_datasets is False
        assert cfg.safety_limits.no_4k_default is True
        assert cfg.safety_limits.no_1k_frame_default is True
        assert cfg.safety_limits.requires_explicit_mllm_load is True
        assert "not a paper reproduction" in cfg.warning


def test_canonical_medium_presets_follow_quick_start_scaling_template() -> None:
    for name in ["canonical_a1_medium", "canonical_a2_medium"]:
        cfg = _load_preset(name)

        assert cfg.resolution == 392
        assert cfg.scale_resolution == "quick_start_target_scales"
        assert list(cfg.target_scales) == [56, 112, 196, 392]
        assert cfg.target_patch_size == 14
        assert cfg.quick_start_alignment.scaling_behavior == "quick_start_target_scales"


def test_canonical_small_presets_use_default_quick_start_resolution() -> None:
    for name in ["canonical_a1_small", "canonical_a2_small"]:
        cfg = _load_preset(name)

        assert cfg.num_frames == 16
        assert cfg.resolution == 224
        assert cfg.scale_resolution is None
        assert cfg.target_scales is None
        assert cfg.target_patch_size is None
        assert cfg.quick_start_alignment.scaling_behavior == "default_224x224_no_target_scaling"


def test_canonical_benchmark_output_tables_are_declared() -> None:
    required_tables = {
        "latency",
        "vram",
        "token_reduction",
        "throughput",
        "stage_level_timing",
        "skipped_stages",
    }

    for name in PRESETS:
        cfg = _load_preset(name)

        assert required_tables.issubset(set(cfg.output_tables.keys()))
        for table_name in required_tables:
            table = cfg.output_tables[table_name]
            assert table.enabled is True
            assert len(table.columns) > 0
