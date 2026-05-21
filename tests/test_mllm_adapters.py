import os

from repro.plugins.mllm_adapters import (
    _build_qwen_grid_inputs,
    build_qwen_pruned_visual_inputs,
    build_qwen_pre_vit_sparse_visual_inputs,
    InternVL3CliMllmAdapter,
    install_qwen_pre_vit_sparse_hook,
    MllmRunRequest,
    MllmRunResult,
    PlannedMllmAdapter,
    QwenGridMllmAdapter,
    VilaCliMllmAdapter,
    build_metric_skeleton,
    extract_assistant_text,
    resolve_mllm_adapter,
    resolve_qwen_visual_keep_indices,
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
        "pre_encoder_prune_adapter": "none",
        "gazing_ratio": None,
        "num_video_frames": 128,
        "max_tiles_video": 1,
        "external_mllm_command": "vila-infer",
        "enable_qwen_prune_generate": False,
        "sparse_selection_plan_path": None,
        "qwen_video_nframes": None,
        "qwen_video_fps": None,
        "qwen_video_max_pixels": None,
        "qwen_video_min_pixels": None,
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


def test_qwen_grid_video_input_error_mentions_qwen_vl_utils_install(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "qwen_vl_utils":
            raise ModuleNotFoundError("No module named 'qwen_vl_utils'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    try:
        _build_qwen_grid_inputs(object(), [{"role": "user", "content": []}], make_request())
    except RuntimeError as exc:
        assert "pip install qwen-vl-utils" in str(exc)
    else:
        raise AssertionError("expected missing qwen_vl_utils RuntimeError")


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


def test_qwen_grid_adapter_builds_autogaze_post_encoder_poc_plan():
    adapter = resolve_mllm_adapter("qwen3-vl")

    result = adapter.run(
        make_request(
            model_family="qwen3-vl",
            mllm_adapter="qwen3-vl",
            token_selector_kind="autogaze",
            integration_level="post_encoder_token_prune",
            gazing_ratio=0.1,
        )
    )

    assert result.status == "poc_ready"
    assert result.metrics["feature_packing_probe"]["family_group"] == "qwen_grid_vl"
    assert result.metrics["sparse_selection_plan"]["selector_name"] == "autogaze"
    assert result.metrics["tokens"]["visual_tokens_before_prune"] == 25088
    assert result.metrics["tokens"]["visual_tokens_after_prune"] == 2509


def test_qwen_pruned_visual_inputs_keep_selected_visual_placeholders_only():
    torch = __import__("torch")

    class FakeQwenModel:
        def __init__(self):
            self.config = type("Config", (), {"video_token_id": 999})()
            self.embedding = torch.nn.Embedding(1200, 4)

        def get_input_embeddings(self):
            return self.embedding

        def get_video_features(self, pixel_values_videos=None, video_grid_thw=None):
            return torch.tensor(
                [
                    [10.0, 0.0, 0.0, 0.0],
                    [20.0, 0.0, 0.0, 0.0],
                    [30.0, 0.0, 0.0, 0.0],
                    [40.0, 0.0, 0.0, 0.0],
                ]
            )

    inputs = {
        "input_ids": torch.tensor([[1, 999, 999, 999, 999, 2]]),
        "attention_mask": torch.ones((1, 6), dtype=torch.long),
        "pixel_values_videos": torch.zeros((1, 3, 2, 2)),
        "video_grid_thw": torch.tensor([[1, 2, 2]]),
    }

    pruned = build_qwen_pruned_visual_inputs(FakeQwenModel(), inputs, keep_indices=[0, 2])

    assert pruned["input_ids"].tolist() == [[1, 999, 999, 2]]
    assert pruned["attention_mask"].tolist() == [[1, 1, 1, 1]]
    assert list(pruned["inputs_embeds"].shape) == [1, 4, 4]
    assert pruned["inputs_embeds"][0, 1].tolist() == [10.0, 0.0, 0.0, 0.0]
    assert pruned["inputs_embeds"][0, 2].tolist() == [30.0, 0.0, 0.0, 0.0]
    assert "pixel_values_videos" not in pruned
    assert "video_grid_thw" not in pruned
    assert pruned["qwen_prune_generate_metadata"]["visual_tokens_before_prune"] == 4
    assert pruned["qwen_prune_generate_metadata"]["visual_tokens_after_prune"] == 2


def test_qwen_grid_adapter_runs_prune_generate_path_when_explicitly_enabled(monkeypatch):
    adapter = resolve_mllm_adapter("qwen3-vl")

    def fake_run(self, request):
        return MllmRunResult(
            text="pruned answer",
            prompt=request.prompt,
            video=request.video,
            image=request.image,
            adapter=self.name,
            status="executed",
            metrics={
                "latency_ms": {"total": 22.0},
                "tokens": {
                    "visual_tokens_before_prune": 100,
                    "visual_tokens_after_prune": 10,
                },
                "memory_bytes": {},
                "qwen_prune_generate": {"enabled": True},
            },
        )

    monkeypatch.setattr(QwenGridMllmAdapter, "_run_qwen_autogaze_prune_generate", fake_run)

    result = adapter.run(
        make_request(
            model_family="qwen3-vl",
            mllm_adapter="qwen3-vl",
            token_selector_kind="autogaze",
            integration_level="post_encoder_token_prune",
            gazing_ratio=0.1,
            enable_qwen_prune_generate=True,
        )
    )

    assert result.status == "executed"
    assert result.text == "pruned answer"
    assert result.metrics["qwen_prune_generate"]["enabled"] is True


def test_qwen_grid_adapter_runs_pre_vit_sparse_path_when_explicitly_enabled(monkeypatch):
    adapter = resolve_mllm_adapter("qwen3-vl")

    def fake_run(self, request):
        return MllmRunResult(
            text="pre vit sparse answer",
            prompt=request.prompt,
            video=request.video,
            image=request.image,
            adapter=self.name,
            status="executed",
            metrics={
                "latency_ms": {"total": 33.0},
                "tokens": {
                    "visual_tokens_before_prune": 100,
                    "visual_tokens_after_prune": 7,
                },
                "memory_bytes": {},
                "qwen_pre_vit_sparse": {"enabled": True},
            },
        )

    monkeypatch.setattr(QwenGridMllmAdapter, "_run_qwen_autogaze_pre_vit_sparse_generate", fake_run)

    result = adapter.run(
        make_request(
            model_family="qwen3-vl",
            mllm_adapter="qwen3-vl",
            token_selector_kind="autogaze",
            integration_level="pre_encoder_sparse",
            pre_encoder_prune_adapter="autogaze-sparse",
            enable_qwen_prune_generate=True,
            sparse_selection_plan_path="outputs/plan.json",
        )
    )

    assert result.status == "executed"
    assert result.text == "pre vit sparse answer"
    assert result.metrics["qwen_pre_vit_sparse"]["enabled"] is True


def test_qwen_pre_vit_sparse_visual_inputs_use_pruned_features_and_original_placeholders():
    torch = __import__("torch")

    class FakeQwenModel:
        def __init__(self):
            self.config = type("Config", (), {"video_token_id": 999})()
            self.embedding = torch.nn.Embedding(1200, 4)

        def get_input_embeddings(self):
            return self.embedding

        def get_video_features(self, pixel_values_videos=None, video_grid_thw=None):
            return torch.tensor(
                [
                    [10.0, 0.0, 0.0, 0.0],
                    [30.0, 0.0, 0.0, 0.0],
                ]
            )

    inputs = {
        "input_ids": torch.tensor([[1, 999, 999, 999, 999, 2]]),
        "attention_mask": torch.ones((1, 6), dtype=torch.long),
        "pixel_values_videos": torch.zeros((1, 3, 2, 2)),
        "video_grid_thw": torch.tensor([[1, 2, 2]]),
    }

    pruned = build_qwen_pre_vit_sparse_visual_inputs(FakeQwenModel(), inputs, original_keep_indices=[0, 2])

    assert pruned["input_ids"].tolist() == [[1, 999, 999, 2]]
    assert list(pruned["inputs_embeds"].shape) == [1, 4, 4]
    assert pruned["inputs_embeds"][0, 1].tolist() == [10.0, 0.0, 0.0, 0.0]
    assert pruned["inputs_embeds"][0, 2].tolist() == [30.0, 0.0, 0.0, 0.0]
    assert "pixel_values_videos" not in pruned
    assert "video_grid_thw" not in pruned
    assert pruned["qwen_pre_vit_sparse_metadata"]["visual_tokens_before_prune"] == 4
    assert pruned["qwen_pre_vit_sparse_metadata"]["visual_tokens_after_prune"] == 2
    assert pruned["qwen_pre_vit_sparse_metadata"]["kept_original_visual_indices"] == [0, 2]


def test_install_qwen_pre_vit_sparse_hook_runs_sparse_visual_forward():
    torch = __import__("torch")

    class FakeMerger(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states.reshape(-1, 4, 3).sum(dim=1)

    class FakeVisual(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dtype = torch.float32
            self.spatial_merge_unit = 4
            self.patch_embed = torch.nn.Identity()
            self.blocks = torch.nn.ModuleList([])
            self.merger = FakeMerger()

        def rot_pos_emb(self, grid_thw):
            return torch.zeros((8, 2), dtype=torch.float32)

    class FakeInner:
        def __init__(self):
            self.visual = FakeVisual()

        def get_video_features(self, pixel_values_videos=None, video_grid_thw=None):
            raise AssertionError("should be patched")

    class FakeModel:
        def __init__(self):
            self.model = FakeInner()

        def get_video_features(self, pixel_values_videos=None, video_grid_thw=None):
            raise AssertionError("should be patched")

    model = FakeModel()

    status = install_qwen_pre_vit_sparse_hook(model, keep_indices=[1])
    features = model.get_video_features(
        pixel_values_videos=torch.arange(24, dtype=torch.float32).reshape(8, 3),
        video_grid_thw=torch.tensor([[1, 2, 4]]),
    )

    assert status["selected_merged_tokens"] == 1
    assert features.tolist() == [[66.0, 70.0, 74.0]]


def test_resolve_qwen_visual_keep_indices_prefers_sparse_plan_json(tmp_path):
    torch = __import__("torch")
    plan_json = tmp_path / "sparse_plan.json"
    plan_json.write_text(
        """
        {
          "selector_name": "autogaze",
          "source_video": {"path": "inputs/example.mp4", "sampled_frame_indices": [0]},
          "preprocess_space": {"resized_width": 64, "resized_height": 64},
          "patch_space": {"autogaze_patch_size": 16, "encoder_patch_size": 16, "scale_sizes": [64]},
          "selected_patches": [
            {
              "frame_index": 0,
              "frame_order": 0,
              "tile_id": 0,
              "scale_id": 0,
              "scale_size": 64,
              "patch_index": 10,
              "bbox_resized_xyxy": [32, 32, 48, 48],
              "bbox_original_xyxy": [32.0, 32.0, 48.0, 48.0],
              "autoregressive_order": 1
            }
          ],
          "encoder_mapping": {"status": "not_mapped"},
          "mllm_mapping": {"status": "not_mapped"}
        }
        """
    )
    inputs = {
        "input_ids": torch.tensor([[1, 999, 999, 999, 999, 2]]),
        "video_grid_thw": torch.tensor([[1, 4, 4]]),
    }

    keep_indices, metadata = resolve_qwen_visual_keep_indices(
        make_request(
            gazing_ratio=0.25,
            sparse_selection_plan_path=str(plan_json),
        ),
        model=object(),
        inputs=inputs,
        visual_count=16,
    )

    assert keep_indices == [10]
    assert metadata["selection_source"] == "sparse_selection_plan"
    assert metadata["mllm_mapping"]["status"] == "exact_grid"


def test_qwen_grid_adapter_runs_pixelprune_pre_vit_path(monkeypatch):
    adapter = resolve_mllm_adapter("qwen3-vl")

    def fake_run(self, request):
        return MllmRunResult(
            text="pixelprune answer",
            prompt=request.prompt,
            video=request.video,
            image=request.image,
            adapter=self.name,
            status="executed",
            metrics={
                "latency_ms": {"total": 12.0},
                "tokens": {"input_ids_tokens": 42},
                "memory_bytes": {},
            },
        )

    monkeypatch.setattr(QwenGridMllmAdapter, "_run_qwen_generate", fake_run)

    result = adapter.run(
        make_request(
            model_family="qwen3-vl",
            mllm_adapter="qwen3-vl",
            integration_level="pre_encoder_sparse",
            pre_encoder_prune_adapter="pixelprune",
        )
    )

    assert result.status == "executed"
    assert result.text == "pixelprune answer"
    assert result.metrics["pre_encoder_prune"]["adapter"] == "pixelprune"
    assert result.metrics["pre_encoder_prune"]["integration_level"] == "pre_encoder_sparse"


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
    assert result.metrics["feature_packing_probe"]["next_probe_command"]["goal"] == "capture_vila_feature_packing"
    assert result.metrics["feature_packing_probe"]["next_probe_command"]["requires_code_probe"] is True
    assert result.metrics["metric_status"]["reason"] == "AutoGaze-on VILA-family integration still requires a feature packing probe"


def test_vila_cli_adapter_collects_static_feature_probe_when_config_exists(tmp_path):
    model_dir = tmp_path / "LongVILA"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        '{"model_type": "llava", "architectures": ["LongVILAForCausalLM"], "vision_tower": "siglip"}'
    )
    adapter = resolve_mllm_adapter("longvila")

    result = adapter.run(
        make_request(
            model_family="longvila",
            model_path=str(model_dir),
            mllm_adapter="longvila",
            token_selector_kind="autogaze",
            integration_level="post_encoder_token_prune",
        )
    )

    assert result.status == "probe_collected"
    assert result.metrics["vila_feature_probe"]["status"] == "static_probe_collected"
    assert result.metrics["vila_feature_probe"]["config_summary"]["architectures"] == ["LongVILAForCausalLM"]
    assert result.metrics["metric_status"]["value"] == "probe_collected"


def test_nvila_video_autogaze_probe_includes_next_runtime_probe_command(tmp_path):
    adapter = resolve_mllm_adapter("nvila-video")

    result = adapter.run(
        make_request(
            model_family="nvila-video-plugin",
            model_path=str(tmp_path / "missing_nvila_model"),
            mllm_adapter="nvila-video",
            token_selector_kind="autogaze",
            integration_level="post_encoder_token_prune",
        )
    )

    command = result.metrics["feature_packing_probe"]["next_probe_command"]
    assert result.status == "probe_required"
    assert command["goal"] == "capture_vila_feature_packing"
    assert command["adapter"] == "nvila-video"
    assert command["expected_outputs"] == [
        "processor video tensor/frame contract",
        "vision feature shape",
        "projector output shape",
        "LLM visual token insertion boundary",
    ]


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
