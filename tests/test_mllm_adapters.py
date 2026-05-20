import os

from repro.plugins.mllm_adapters import (
    InternVL3CliMllmAdapter,
    MllmRunRequest,
    PlannedMllmAdapter,
    QwenGridMllmAdapter,
    VilaCliMllmAdapter,
    build_metric_skeleton,
    extract_assistant_text,
    resolve_mllm_adapter,
)


def make_request(**overrides):
    values = {
        "model_family": "qwen3-vl",
        "model_path": "weight/Qwen3-VL-8B-Instruct",
        "mllm_adapter": "qwen3-vl",
        "prompt": "What happens in the video?",
        "video": "inputs/example.mp4",
        "image": None,
        "device_map": "auto",
        "dtype": "auto",
        "attn_implementation": None,
        "trust_remote_code": True,
        "max_new_tokens": 32,
        "token_selector_kind": "keep-all",
        "integration_level": "none",
        "num_video_frames": 128,
        "max_tiles_video": 1,
        "external_mllm_command": "vila-infer",
    }
    values.update(overrides)
    return MllmRunRequest(**values)


def test_resolve_mllm_adapter_maps_qwen_grid_families_to_runtime_adapter():
    for name in ["qwen2-vl", "qwen2.5-vl", "qwen3-vl", "qwen3-vl-moe"]:
        adapter = resolve_mllm_adapter(name)

        assert isinstance(adapter, QwenGridMllmAdapter)
        assert adapter.runtime_status == "implemented"


def test_resolve_mllm_adapter_maps_vila_family_to_vila_cli_adapter():
    for name in ["nvila-video", "longvila"]:
        adapter = resolve_mllm_adapter(name)

        assert isinstance(adapter, VilaCliMllmAdapter)
        assert adapter.runtime_status == "external_cli_ready"
        assert adapter.name == name


def test_resolve_mllm_adapter_marks_probe_required_families_as_planned():
    adapter = resolve_mllm_adapter("internvl3")

    assert not isinstance(adapter, PlannedMllmAdapter)
    assert isinstance(adapter, InternVL3CliMllmAdapter)
    assert adapter.runtime_status == "external_cli_ready"


def test_qwen_grid_adapter_builds_video_chat_message():
    adapter = QwenGridMllmAdapter()

    messages = adapter.build_messages(make_request())

    assert messages == [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": "inputs/example.mp4"},
                {"type": "text", "text": "What happens in the video?"},
            ],
        }
    ]


def test_metric_skeleton_contains_latency_token_memory_slots():
    skeleton = build_metric_skeleton(make_request(prompt="one two three"))

    assert skeleton["latency_ms"] == {
        "model_load": None,
        "processor_load": None,
        "input_build": None,
        "generate": None,
        "total": None,
    }
    assert skeleton["tokens"]["prompt_tokens_estimated"] == 3
    assert skeleton["tokens"]["input_ids_tokens"] is None
    assert skeleton["tokens"]["visual_tokens_before_prune"] is None
    assert skeleton["tokens"]["visual_tokens_after_prune"] is None
    assert skeleton["memory_bytes"]["peak_cuda_allocated"] is None


def test_runtime_description_includes_metric_schema():
    adapter = resolve_mllm_adapter("qwen3-vl")

    description = adapter.describe_runtime(make_request())

    assert description["status"] == "implemented"
    assert "latency_ms" in description["metric_schema"]
    assert "tokens" in description["metric_schema"]


def test_qwen_grid_adapter_describes_post_encoder_probe_for_autogaze_request():
    adapter = resolve_mllm_adapter("qwen3-vl")

    description = adapter.describe_runtime(
        make_request(
            model_family="qwen3-vl",
            mllm_adapter="qwen3-vl",
            token_selector_kind="autogaze",
            integration_level="post_encoder_token_prune",
        )
    )

    assert description["feature_packing_probe"]["family_group"] == "qwen_grid_vl"
    assert description["feature_packing_probe"]["required_inputs"] == [
        "pixel_values_videos",
        "video_grid_thw",
        "input_ids",
    ]
    assert description["feature_packing_probe"]["post_encoder_hook"] == (
        "after get_video_features output and before visual token insertion into MLLM context"
    )


def test_qwen_grid_adapter_keeps_autogaze_request_as_probe_required():
    adapter = resolve_mllm_adapter("qwen3-vl")

    result = adapter.run(
        make_request(
            model_family="qwen3-vl",
            mllm_adapter="qwen3-vl",
            token_selector_kind="autogaze",
            integration_level="post_encoder_token_prune",
        )
    )

    assert result.status == "probe_required"
    assert result.metrics["feature_packing_probe"]["family_group"] == "qwen_grid_vl"


def test_llava_onevision_adapter_uses_path_style_video_content():
    adapter = resolve_mllm_adapter("llava-onevision")

    messages = adapter.build_messages(make_request(model_family="llava-onevision", mllm_adapter="llava-onevision"))

    assert messages == [
        {
            "role": "user",
            "content": [
                {"type": "video", "path": "inputs/example.mp4"},
                {"type": "text", "text": "What happens in the video?"},
            ],
        }
    ]


def test_planned_adapter_describes_model_specific_feature_probe():
    adapter = resolve_mllm_adapter("internvl3")

    description = adapter.describe_runtime(make_request(model_family="internvl3", mllm_adapter="internvl3"))

    assert description["status"] == "external_cli_ready"
    assert description["feature_packing_probe"]["family_group"] == "internvl"
    assert description["feature_packing_probe"]["required_inputs"] == ["pixel_values", "num_patches_list"]
    assert description["feature_packing_probe"]["post_encoder_hook"] == (
        "after dynamic visual feature extraction and before language model packing"
    )


def test_internvl3_adapter_reports_missing_external_helper_for_off_execution():
    adapter = resolve_mllm_adapter("internvl3")

    result = adapter.run(
        make_request(
            model_family="internvl3",
            mllm_adapter="internvl3",
            external_mllm_command="definitely_missing_internvl3_helper_for_test",
        )
    )

    assert result.status == "failed_missing_dependency"
    assert result.text is None
    assert result.metrics["latency_ms"]["generate"] is None
    assert result.metrics["external_cli"]["command"][0] == "definitely_missing_internvl3_helper_for_test"


def test_nvila_video_planned_adapter_records_paper_baseline_safe_probe():
    adapter = resolve_mllm_adapter("nvila-video")

    result = adapter.run(
        make_request(
            model_family="nvila-video-baseline",
            model_path="weight/NVILA-8B-Video",
            mllm_adapter="nvila-video",
            external_mllm_command="definitely_missing_vila_infer_for_test",
        )
    )

    assert result.status == "failed_missing_dependency"
    assert result.metrics["external_cli"]["command"][0] == "definitely_missing_vila_infer_for_test"
    assert result.metrics["metric_status"]["reason"] == "vila-infer command was not found"
    assert result.metrics["tokens"]["visual_tokens_before_prune"] is None


def test_extract_assistant_text_prefers_assistant_markers_and_json_answer():
    assert extract_assistant_text("loading\nAssistant: Answer: C\n") == "Answer: C"
    assert extract_assistant_text('{"answer": "B", "latency_ms": 123}\n') == "B"
    assert extract_assistant_text("User: Q\nfinal answer\n") == "final answer"


def test_vila_cli_adapter_builds_off_generation_command():
    adapter = resolve_mllm_adapter("nvila-video")

    command = adapter.build_command(
        make_request(
            model_family="nvila-video-plugin",
            model_path="weight/NVILA-8B-Video",
            mllm_adapter="nvila-video",
            external_mllm_command="/opt/vila/bin/vila-infer",
            prompt="Describe the video.",
            video="inputs/example.mp4",
            num_video_frames=256,
            max_tiles_video=8,
        )
    )

    assert command == [
        "/opt/vila/bin/vila-infer",
        "--model-path",
        "weight/NVILA-8B-Video",
        "--conv-mode",
        "auto",
        "--text",
        "Describe the video.",
        "--media",
        "inputs/example.mp4",
        "--num_video_frames",
        "256",
        "--video_max_tiles",
        "8",
    ]


def test_vila_cli_adapter_keeps_autogaze_requested_run_as_probe_required():
    adapter = resolve_mllm_adapter("longvila")

    result = adapter.run(
        make_request(
            model_family="longvila",
            mllm_adapter="longvila",
            token_selector_kind="autogaze",
            integration_level="post_encoder_token_prune",
        )
    )

    assert result.status == "probe_required"
    assert result.metrics["feature_packing_probe"]["autogaze_applicability"] == "plugin_on_off_experiment"
    assert result.metrics["feature_packing_probe"]["family_group"] == "longvila"
    assert result.metrics["metric_status"]["reason"] == "AutoGaze-on VILA-family integration still requires a feature packing probe"


def test_vila_cli_adapter_executes_available_command_and_extracts_last_output_line(tmp_path):
    script = tmp_path / "vila-infer"
    script.write_text("#!/bin/sh\necho loading model\necho final answer\n")
    os.chmod(script, 0o755)
    adapter = resolve_mllm_adapter("nvila-video")

    result = adapter.run(
        make_request(
            model_family="nvila-video-plugin",
            mllm_adapter="nvila-video",
            external_mllm_command=str(script),
        )
    )

    assert result.status == "executed"
    assert result.text == "final answer"
    assert result.metrics["latency_ms"]["total"] is not None
    assert result.metrics["external_cli"]["returncode"] == 0


def test_internvl3_cli_adapter_builds_off_generation_command():
    adapter = resolve_mllm_adapter("internvl3")

    command = adapter.build_command(
        make_request(
            model_family="internvl3",
            model_path="weight/InternVL3-8B",
            mllm_adapter="internvl3",
            external_mllm_command="python -m repro.internvl3_off_infer",
            prompt="What happens?",
            video="inputs/example.mp4",
            num_video_frames=8,
            max_tiles_video=4,
        )
    )

    assert command == [
        "python",
        "-m",
        "repro.internvl3_off_infer",
        "--model-path",
        "weight/InternVL3-8B",
        "--prompt",
        "What happens?",
        "--video",
        "inputs/example.mp4",
        "--num-video-frames",
        "8",
        "--max-tiles-video",
        "4",
        "--max-new-tokens",
        "32",
    ]


def test_internvl3_cli_adapter_keeps_autogaze_requested_run_as_probe_required():
    adapter = resolve_mllm_adapter("internvl3")

    result = adapter.run(
        make_request(
            model_family="internvl3",
            mllm_adapter="internvl3",
            token_selector_kind="autogaze",
            integration_level="post_encoder_token_prune",
        )
    )

    assert result.status == "probe_required"
    assert result.metrics["feature_packing_probe"]["family_group"] == "internvl"
    assert result.metrics["metric_status"]["reason"] == "AutoGaze-on InternVL3 integration still requires a dynamic tiling probe"
