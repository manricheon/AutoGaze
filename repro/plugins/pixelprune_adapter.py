from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class PixelPruneConfig:
    model_key: str
    enabled: bool = True
    threshold: float = 0.0
    verbose: bool = False

    def apply_environment(self) -> dict[str, str]:
        env = build_pixelprune_environment(self)
        os.environ.update(env)
        return env


def pixelprune_model_key(model_family: str) -> str:
    if model_family in {"qwen3-vl", "qwen3-vl-moe"}:
        return "qwen3_vl"
    if model_family in {"qwen3.5-vl", "qwen3_5"}:
        return "qwen3_5"
    raise ValueError(f"PixelPrune does not declare support for model family: {model_family}")


def build_pixelprune_environment(config: PixelPruneConfig) -> dict[str, str]:
    return {
        "PIXELPRUNE_ENABLED": "true" if config.enabled else "false",
        "PIXELPRUNE_THRESHOLD": str(config.threshold),
        "PIXELPRUNE_VERBOSE": "true" if config.verbose else "false",
    }


def apply_pixelprune_if_available(config: PixelPruneConfig) -> dict[str, str | bool]:
    env = config.apply_environment()
    try:
        from pixelprune import apply_pixelprune  # type: ignore
    except ModuleNotFoundError:
        return {
            "applied": False,
            "reason": "pixelprune package is not installed",
            "model_key": config.model_key,
            "environment": env,
        }
    apply_pixelprune(model=config.model_key)
    return {
        "applied": True,
        "reason": None,
        "model_key": config.model_key,
        "environment": env,
    }
