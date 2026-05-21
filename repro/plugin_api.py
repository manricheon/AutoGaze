from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


ModelFamily = Literal[
    "nvila-hd-video-autogaze",
    "nvila-video-baseline",
    "nvila-video-plugin",
    "longvila",
    "qwen2-vl",
    "qwen2.5-vl",
    "qwen3-vl",
    "qwen3-vl-moe",
    "llava-onevision",
    "internvl3",
]
TokenSelectorKind = Literal["none", "keep-all", "autogaze", "external-mask"]
IntegrationLevel = Literal[
    "none",
    "native_processor",
    "pre_encoder_sparse",
    "post_encoder_token_prune",
    "planned_plugin",
]

PAPER_BASELINE_FAMILY = "nvila-video-baseline"
PLUGIN_EXPERIMENT_FAMILIES = {
    "nvila-video-plugin",
    "longvila",
    "qwen2-vl",
    "qwen2.5-vl",
    "qwen3-vl",
    "qwen3-vl-moe",
    "llava-onevision",
    "internvl3",
}
AUTOGAZE_INTEGRATION_LEVELS = {"native_processor", "pre_encoder_sparse", "post_encoder_token_prune", "planned_plugin"}
PLUGIN_AUTOGAZE_INTEGRATION_LEVELS = {"pre_encoder_sparse", "post_encoder_token_prune", "planned_plugin"}


class PluginSpecError(ValueError):
    """Raised when a runner configuration violates the plugin experiment contract."""


@dataclass(frozen=True)
class MetricStatus:
    value: str
    reason: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {"value": self.value, "reason": self.reason}


@dataclass(frozen=True)
class PluginResult:
    plugin_name: str
    status: MetricStatus
    metrics: dict[str, Any]
    artifacts: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "plugin_name": self.plugin_name,
            "status": self.status.to_dict(),
            "metrics": _json_ready(self.metrics),
            "artifacts": _json_ready(self.artifacts),
        }


@dataclass(frozen=True)
class ExperimentSpec:
    model_family: ModelFamily
    model_path: str
    token_selector_kind: TokenSelectorKind
    token_selector_path: str | None
    vision_encoder_kind: str
    vision_encoder_path: str | None
    mllm_kind: str
    mllm_path: str
    integration_level: IntegrationLevel
    num_video_frames: int
    num_thumbnail_frames: int
    max_tiles_video: int
    resize_longest_edge: int | None
    resize_shortest_edge: int | None
    gazing_ratio: float | None
    output_dir: Path
    pre_encoder_prune_adapter: str = "none"

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ExperimentSpec":
        output_dir = _output_dir_from_args(args)
        spec = cls(
            model_family=str(getattr(args, "model_family")),  # type: ignore[arg-type]
            model_path=str(getattr(args, "model_path")),
            token_selector_kind=str(getattr(args, "token_selector_adapter")),  # type: ignore[arg-type]
            token_selector_path=_optional_path_string(getattr(args, "token_selector_path", None)),
            vision_encoder_kind=str(getattr(args, "vision_encoder_adapter")),
            vision_encoder_path=_optional_path_string(getattr(args, "vision_encoder_path", None)),
            mllm_kind=str(getattr(args, "mllm_adapter")),
            mllm_path=str(getattr(args, "mllm_path", getattr(args, "model_path"))),
            integration_level=str(getattr(args, "autogaze_integration_level", "none")),  # type: ignore[arg-type]
            num_video_frames=int(getattr(args, "num_video_frames")),
            num_thumbnail_frames=int(getattr(args, "num_video_frames_thumbnail", 0) or 0),
            max_tiles_video=int(getattr(args, "max_tiles_video", 0) or 0),
            resize_longest_edge=_optional_int(getattr(args, "video_resize_longest_edge", None)),
            resize_shortest_edge=_optional_int(getattr(args, "video_resize_shortest_edge", None)),
            gazing_ratio=_optional_float(getattr(args, "gazing_ratio", None)),
            output_dir=output_dir,
            pre_encoder_prune_adapter=str(getattr(args, "pre_encoder_prune_adapter", "none")),
        )
        spec.validate()
        return spec

    @property
    def uses_autogaze(self) -> bool:
        return self.token_selector_kind == "autogaze" and self.integration_level in AUTOGAZE_INTEGRATION_LEVELS

    @property
    def is_paper_baseline_candidate(self) -> bool:
        return self.model_family == PAPER_BASELINE_FAMILY

    def validate(self) -> None:
        if self.model_family == PAPER_BASELINE_FAMILY:
            if self.token_selector_kind != "none" or self.integration_level != "none":
                raise PluginSpecError(
                    "NVILA-8B-Video paper baseline must use token_selector_kind='none' "
                    "and integration_level='none'. Use model_family='nvila-video-plugin' "
                    "for AutoGaze on/off experiments."
                )
        if self.model_family in PLUGIN_EXPERIMENT_FAMILIES and self.token_selector_kind == "autogaze":
            if self.integration_level not in PLUGIN_AUTOGAZE_INTEGRATION_LEVELS:
                raise PluginSpecError(
                    "AutoGaze plugin experiments require integration_level to be one of "
                    "planned_plugin, pre_encoder_sparse, or post_encoder_token_prune."
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_family": self.model_family,
            "model_path": self.model_path,
            "token_selector_kind": self.token_selector_kind,
            "token_selector_path": self.token_selector_path,
            "vision_encoder_kind": self.vision_encoder_kind,
            "vision_encoder_path": self.vision_encoder_path,
            "mllm_kind": self.mllm_kind,
            "mllm_path": self.mllm_path,
            "integration_level": self.integration_level,
            "num_video_frames": self.num_video_frames,
            "num_thumbnail_frames": self.num_thumbnail_frames,
            "max_tiles_video": self.max_tiles_video,
            "resize_longest_edge": self.resize_longest_edge,
            "resize_shortest_edge": self.resize_shortest_edge,
            "gazing_ratio": self.gazing_ratio,
            "output_dir": str(self.output_dir),
            "pre_encoder_prune_adapter": self.pre_encoder_prune_adapter,
            "uses_autogaze": self.uses_autogaze,
            "is_paper_baseline_candidate": self.is_paper_baseline_candidate,
        }


def _output_dir_from_args(args: argparse.Namespace) -> Path:
    for attr in ("output_json", "summary", "summary_json", "predictions"):
        value = getattr(args, attr, None)
        if value:
            return Path(str(value)).parent
    return Path("outputs/autogaze_repro")


def _optional_path_string(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    return value
