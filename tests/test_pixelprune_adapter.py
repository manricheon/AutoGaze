import os

from repro.plugins.pixelprune_adapter import (
    PixelPruneConfig,
    build_pixelprune_environment,
    pixelprune_model_key,
)


def test_pixelprune_model_key_maps_qwen3_families():
    assert pixelprune_model_key("qwen3-vl") == "qwen3_vl"
    assert pixelprune_model_key("qwen3-vl-moe") == "qwen3_vl"


def test_pixelprune_environment_matches_upstream_controls():
    env = build_pixelprune_environment(
        PixelPruneConfig(model_key="qwen3_vl", enabled=True, threshold=0.05, verbose=True)
    )

    assert env == {
        "PIXELPRUNE_ENABLED": "true",
        "PIXELPRUNE_THRESHOLD": "0.05",
        "PIXELPRUNE_VERBOSE": "true",
    }


def test_pixelprune_config_can_apply_environment_without_importing_package(monkeypatch):
    for key in ["PIXELPRUNE_ENABLED", "PIXELPRUNE_THRESHOLD", "PIXELPRUNE_VERBOSE"]:
        monkeypatch.delenv(key, raising=False)

    PixelPruneConfig(model_key="qwen3_vl", threshold=0.0).apply_environment()

    assert os.environ["PIXELPRUNE_ENABLED"] == "true"
    assert os.environ["PIXELPRUNE_THRESHOLD"] == "0.0"
    assert os.environ["PIXELPRUNE_VERBOSE"] == "false"
