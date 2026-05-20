import argparse
from pathlib import Path

import pytest

from repro.plugin_api import ExperimentSpec, MetricStatus, PluginResult, PluginSpecError


def make_args(**overrides):
    values = {
        "model_family": "nvila-video-plugin",
        "model_path": "weight/NVILA-8B-Video",
        "token_selector_adapter": "autogaze",
        "token_selector_path": "weight/AutoGaze",
        "vision_encoder_adapter": "nvila-video-vision",
        "vision_encoder_path": "auto",
        "mllm_adapter": "nvila-video",
        "mllm_path": "weight/NVILA-8B-Video",
        "autogaze_integration_level": "planned_plugin",
        "num_video_frames": 128,
        "num_video_frames_thumbnail": 64,
        "max_tiles_video": 48,
        "video_resize_longest_edge": None,
        "video_resize_shortest_edge": 720,
        "output_json": "outputs/autogaze_repro/plugin.json",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_experiment_spec_from_runner_args_records_component_contract():
    spec = ExperimentSpec.from_args(make_args())

    assert spec.model_family == "nvila-video-plugin"
    assert spec.model_path == "weight/NVILA-8B-Video"
    assert spec.token_selector_kind == "autogaze"
    assert spec.token_selector_path == "weight/AutoGaze"
    assert spec.vision_encoder_kind == "nvila-video-vision"
    assert spec.vision_encoder_path == "auto"
    assert spec.mllm_kind == "nvila-video"
    assert spec.mllm_path == "weight/NVILA-8B-Video"
    assert spec.integration_level == "planned_plugin"
    assert spec.num_video_frames == 128
    assert spec.num_thumbnail_frames == 64
    assert spec.max_tiles_video == 48
    assert spec.resize_longest_edge is None
    assert spec.resize_shortest_edge == 720
    assert spec.output_dir == Path("outputs/autogaze_repro")


def test_experiment_spec_rejects_paper_baseline_with_autogaze_selector():
    with pytest.raises(PluginSpecError, match="paper baseline"):
        ExperimentSpec.from_args(
            make_args(
                model_family="nvila-video-baseline",
                token_selector_adapter="autogaze",
                autogaze_integration_level="planned_plugin",
            )
        )


def test_experiment_spec_requires_plugin_integration_for_autogaze_on_experiments():
    with pytest.raises(PluginSpecError, match="integration_level"):
        ExperimentSpec.from_args(
            make_args(
                model_family="longvila",
                token_selector_adapter="autogaze",
                autogaze_integration_level="none",
            )
        )


def test_experiment_spec_applies_plugin_autogaze_rule_to_qwen_family():
    with pytest.raises(PluginSpecError, match="integration_level"):
        ExperimentSpec.from_args(
            make_args(
                model_family="qwen2-vl",
                model_path="weight/Qwen2-VL",
                token_selector_adapter="autogaze",
                vision_encoder_adapter="qwen2-vl-vision",
                mllm_adapter="qwen2-vl",
                mllm_path="weight/Qwen2-VL",
                autogaze_integration_level="none",
            )
        )


def test_experiment_spec_applies_plugin_autogaze_rule_to_qwen3_family():
    with pytest.raises(PluginSpecError, match="integration_level"):
        ExperimentSpec.from_args(
            make_args(
                model_family="qwen3-vl",
                model_path="weight/Qwen3-VL-8B-Instruct",
                token_selector_adapter="autogaze",
                vision_encoder_adapter="qwen3-vl-vision",
                mllm_adapter="qwen3-vl",
                mllm_path="weight/Qwen3-VL-8B-Instruct",
                autogaze_integration_level="none",
            )
        )


def test_experiment_spec_applies_plugin_autogaze_rule_to_other_mllm_families():
    for family, vision, mllm in [
        ("qwen2.5-vl", "qwen2.5-vl-vision", "qwen2.5-vl"),
        ("llava-onevision", "llava-onevision-siglip", "llava-onevision"),
        ("internvl3", "internvl-dynamic-vision", "internvl3"),
    ]:
        with pytest.raises(PluginSpecError, match="integration_level"):
            ExperimentSpec.from_args(
                make_args(
                    model_family=family,
                    model_path=f"weight/{family}",
                    token_selector_adapter="autogaze",
                    vision_encoder_adapter=vision,
                    mllm_adapter=mllm,
                    mllm_path=f"weight/{family}",
                    autogaze_integration_level="none",
                )
            )


def test_experiment_spec_normalizes_native_off_plugin_comparison():
    spec = ExperimentSpec.from_args(
        make_args(
            model_family="longvila",
            model_path="weight/longvila",
            token_selector_adapter="keep-all",
            token_selector_path=None,
            vision_encoder_adapter="longvila-siglip",
            mllm_adapter="longvila",
            mllm_path="weight/longvila",
            autogaze_integration_level="none",
        )
    )

    assert spec.token_selector_kind == "keep-all"
    assert spec.integration_level == "none"
    assert spec.uses_autogaze is False


def test_experiment_spec_to_dict_is_json_ready():
    spec = ExperimentSpec.from_args(make_args())

    data = spec.to_dict()

    assert data["output_dir"] == "outputs/autogaze_repro"
    assert data["uses_autogaze"] is True
    assert data["is_paper_baseline_candidate"] is False


def test_plugin_result_records_metric_status_reasons():
    result = PluginResult(
        plugin_name="longvila",
        status=MetricStatus(value="planned", reason="adapter contract only"),
        metrics={"visual_tokens": None},
        artifacts={"report": Path("outputs/report.md")},
    )

    assert result.to_dict() == {
        "plugin_name": "longvila",
        "status": {"value": "planned", "reason": "adapter contract only"},
        "metrics": {"visual_tokens": None},
        "artifacts": {"report": "outputs/report.md"},
    }
