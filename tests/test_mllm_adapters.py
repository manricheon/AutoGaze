import os
import sys
import types

from repro.plugins.mllm_adapters import (
    _build_qwen_grid_inputs,
    build_qwen_inputs_from_video_features,
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
    build_mllm_processing_budget_summary,
    extract_assistant_text,
    qwen_grid_chunk_slices,
    qwen_chunked_video_features,
    qwen_thumbnail_visual_keep_indices,
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
        "qwen_vit_mode": "qwen_full_vit",
        "qwen_vit_chunk_frames": 16,
        "qwen_vit_max_spatial_chunks": 1,
        "num_video_frames_thumbnail": 0,
        "qwen_thumbnail_mode": "none",
        "video_resize_shortest_edge": None,
        "video_resize_longest_edge": None,
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


def test_qwen_grid_runtime_description_reports_runner_resize_request():
    adapter = QwenGridMllmAdapter()

    description = adapter.describe_runtime(
        make_request(video_resize_shortest_edge=448, qwen_video_max_pixels=262144)
    )

    runner_resize = description["qwen_video_input"]["runner_resize"]
    assert runner_resize["enabled"] is True
    assert runner_resize["shortest_edge"] == 448
    assert runner_resize["longest_edge"] is None
    assert runner_resize["mode"] == "preloaded_resized_frames"
    assert description["qwen_video_input"]["max_pixels"] == 262144


def test_qwen_grid_runtime_description_reports_append_video_thumbnails():
    adapter = QwenGridMllmAdapter()

    description = adapter.describe_runtime(
        make_request(num_video_frames_thumbnail=2, qwen_thumbnail_mode="append-video")
    )

    assert description["qwen_video_input"]["thumbnail"] == {
        "mode": "append-video",
        "requested_frames": 2,
        "effective_frames": 2,
        "placement": "appended_after_main_video_frames",
        "pruning_policy": "keep_all",
    }


def test_build_qwen_grid_inputs_uses_preloaded_resized_frames_when_runner_resize_requested(monkeypatch):
    frames = [object(), object()]
    calls = {}

    def fake_load_sampled_video_frames(video, sample_count, resize, *, decode_strategy="auto"):
        calls["load"] = {
            "video": video,
            "sample_count": sample_count,
            "resize": resize,
            "decode_strategy": decode_strategy,
        }
        return frames, {"decode_strategy": "seek", "decode_frames_read": 2}

    def fail_process_vision_info(*args, **kwargs):
        raise AssertionError("runner-side resize should not ask qwen_vl_utils to decode the video path")

    fake_qwen_utils = types.SimpleNamespace(process_vision_info=fail_process_vision_info)
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", fake_qwen_utils)
    monkeypatch.setattr("repro.plugins.mllm_adapters.load_sampled_video_frames", fake_load_sampled_video_frames)
    monkeypatch.setattr(
        "repro.plugins.mllm_adapters.apply_resize_to_dimensions",
        lambda **kwargs: {"width": 448, "height": 252, "mode": "longest_edge"},
    )
    monkeypatch.setattr(
        "repro.plugins.mllm_adapters.read_video_metadata",
        lambda video: {"width": 1920, "height": 1080, "frames": 60},
    )

    class FakeProcessor:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            calls["template"] = {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
            }
            return "<video> prompt"

        def __call__(self, **kwargs):
            calls["processor"] = kwargs
            return {"input_ids": [1, 2, 3]}

    request = make_request(
        qwen_video_nframes=2,
        qwen_video_max_pixels=262144,
        video_resize_longest_edge=448,
    )

    output = _build_qwen_grid_inputs(FakeProcessor(), QwenGridMllmAdapter().build_messages(request), request)

    assert output == {"input_ids": [1, 2, 3]}
    assert calls["load"] == {
        "video": "inputs/example.mp4",
        "sample_count": 2,
        "resize": {"width": 448, "height": 252, "mode": "longest_edge"},
        "decode_strategy": "auto",
    }
    assert calls["processor"]["videos"] == [frames]
    assert "max_pixels" not in calls["processor"]
    assert "nframes" not in calls["processor"]


def test_build_qwen_grid_inputs_appends_thumbnail_frames_to_preloaded_video(monkeypatch):
    frames = [object(), object(), object(), object()]
    calls = {}

    def fake_load_sampled_video_frames(video, sample_count, resize, *, decode_strategy="auto"):
        calls["load"] = {"sample_count": sample_count, "resize": resize}
        return frames, {"decode_strategy": "seek", "decode_frames_read": 4}

    monkeypatch.setattr("repro.plugins.mllm_adapters.load_sampled_video_frames", fake_load_sampled_video_frames)
    monkeypatch.setattr(
        "repro.plugins.mllm_adapters.apply_resize_to_dimensions",
        lambda **kwargs: {"width": 448, "height": 252, "mode": "longest_edge"},
    )
    monkeypatch.setattr(
        "repro.plugins.mllm_adapters.read_video_metadata",
        lambda video: {"width": 1920, "height": 1080, "frames": 60},
    )
    monkeypatch.setattr(
        "repro.plugins.mllm_adapters.uniform_sample_indices",
        lambda total, count: [0, 10, 20, 30],
    )

    class FakeProcessor:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return "<video> prompt"

        def __call__(self, **kwargs):
            calls["processor"] = kwargs
            return {"input_ids": [1, 2, 3]}

    request = make_request(
        qwen_video_nframes=4,
        num_video_frames_thumbnail=2,
        qwen_thumbnail_mode="append-video",
        video_resize_longest_edge=448,
    )

    _build_qwen_grid_inputs(FakeProcessor(), QwenGridMllmAdapter().build_messages(request), request)

    assert calls["load"]["sample_count"] == 4
    assert calls["processor"]["videos"] == [[frames[0], frames[1], frames[2], frames[3], frames[0], frames[2]]]


def test_build_qwen_grid_inputs_still_passes_qwen_video_kwargs_without_runner_resize(monkeypatch):
    calls = {}

    def fake_process_vision_info(messages, return_video_kwargs=False):
        calls["messages"] = messages
        return ["image"], ["video_tensor"], {"fps": 2.0, "max_pixels": 262144}

    fake_qwen_utils = types.SimpleNamespace(process_vision_info=fake_process_vision_info)
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", fake_qwen_utils)

    class FakeProcessor:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return "<video> prompt"

        def __call__(self, **kwargs):
            calls["processor"] = kwargs
            return {"input_ids": [1, 2, 3]}

    request = make_request(qwen_video_max_pixels=262144, qwen_video_fps=2.0)

    _build_qwen_grid_inputs(FakeProcessor(), QwenGridMllmAdapter().build_messages(request), request)

    assert calls["messages"][0]["content"][0]["max_pixels"] == 262144
    assert calls["processor"]["videos"] == ["video_tensor"]
    assert calls["processor"]["max_pixels"] == 262144


def test_qwen_thumbnail_visual_keep_indices_keeps_appended_temporal_tail():
    indices, metadata = qwen_thumbnail_visual_keep_indices(
        make_request(
            qwen_video_nframes=4,
            num_video_frames_thumbnail=2,
            qwen_thumbnail_mode="append-video",
        ),
        video_grid_thw=[6, 2, 2],
        spatial_merge_size=1,
    )

    assert indices == list(range(16, 24))
    assert metadata["thumbnail_temporal_start"] == 4
    assert metadata["thumbnail_visual_tokens"] == 8


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
        "qwen_vit_prepare": None,
        "generate": None,
        "total": None,
    }
    assert skeleton["tokens"]["prompt_tokens_estimated"] == 3
    assert skeleton["tokens"]["input_ids_tokens"] is None
    assert skeleton["tokens"]["visual_tokens_before_prune"] is None
    assert skeleton["tokens"]["visual_tokens_after_prune"] is None
    assert skeleton["memory_bytes"]["peak_cuda_allocated"] is None
    assert skeleton["qwen_vit"]["mode"] == "qwen_full_vit"
    assert skeleton["qwen_thumbnail"]["mode"] == "none"
    assert skeleton["processing_budget_summary"]["video"]["requested_video_frames"] == 128


def test_build_mllm_processing_budget_summary_reports_qwen_resize_thumbnail_and_expected_tokens():
    summary = build_mllm_processing_budget_summary(
        make_request(
            qwen_video_nframes=32,
            num_video_frames_thumbnail=8,
            qwen_thumbnail_mode="append-video",
            max_tiles_video=4,
            video_resize_longest_edge=448,
            qwen_vit_mode="qwen_chunked_vit_autogaze_sparse",
            qwen_vit_max_spatial_chunks=4,
        )
    )

    assert summary["video"]["requested_video_frames"] == 32
    assert summary["video"]["runner_resize"]["longest_edge"] == 448
    assert summary["thumbnail"]["enabled"] is True
    assert summary["thumbnail"]["mode"] == "append-video"
    assert summary["model_processing_unit"]["name"] == "qwen_video_grid_thw"
    assert summary["tiling"]["spatial_chunks_per_frame_limit"] == 4
    assert summary["patch_budget_before_vit"]["estimated_total_frames_in_processor"] == 40
    assert summary["patch_budget_before_vit"]["estimated_visual_tokens_before_prune"] is not None
    assert summary["patch_budget_before_vit"]["multiscale_policy"] == "not_applicable_qwen_native_grid"


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


def test_qwen_grid_chunks_split_flat_video_tokens_by_temporal_grid():
    torch = __import__("torch")

    chunks = qwen_grid_chunk_slices(
        torch.tensor([[4, 2, 4]]),
        spatial_merge_size=2,
        chunk_frames=2,
    )

    assert len(chunks) == 2
    assert chunks[0]["t_start"] == 0
    assert chunks[0]["t_end"] == 2
    assert chunks[0]["spatial_tiles"] == 1
    assert chunks[0]["raw_token_start"] == 0
    assert chunks[0]["raw_token_end"] == 16
    assert chunks[0]["merged_token_indices"] == [0, 1, 2, 3]
    assert chunks[0]["raw_token_indices"] == [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15]
    assert chunks[1]["t_start"] == 2
    assert chunks[1]["t_end"] == 4
    assert chunks[1]["merged_token_indices"] == [4, 5, 6, 7]


def test_qwen_grid_chunks_can_split_spatial_grid_like_nvila_tiles():
    torch = __import__("torch")

    chunks = qwen_grid_chunk_slices(
        torch.tensor([[2, 4, 4]]),
        spatial_merge_size=2,
        chunk_frames=1,
        max_spatial_chunks=4,
    )

    assert len(chunks) == 8
    assert chunks[0]["tile_grid_cols"] == 2
    assert chunks[0]["tile_grid_rows"] == 2
    assert chunks[0]["spatial_tile_index"] == 0
    assert chunks[0]["raw_token_indices"] == [0, 1, 4, 5]
    assert chunks[0]["merged_token_indices"] == [0]
    assert chunks[1]["spatial_tile_index"] == 1
    assert chunks[1]["raw_token_indices"] == [2, 3, 6, 7]
    assert chunks[1]["merged_token_indices"] == [1]
    assert chunks[4]["t_start"] == 1
    assert chunks[4]["raw_token_indices"] == [16, 17, 20, 21]
    assert chunks[4]["merged_token_indices"] == [4]


def test_build_qwen_inputs_from_video_features_keeps_dense_placeholders():
    torch = __import__("torch")

    class FakeQwenModel:
        def __init__(self):
            self.config = type("Config", (), {"video_token_id": 999})()
            self.embedding = torch.nn.Embedding(1200, 4)

        def get_input_embeddings(self):
            return self.embedding

    inputs = {
        "input_ids": torch.tensor([[1, 999, 999, 2]]),
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
        "pixel_values_videos": torch.zeros((4, 3)),
        "video_grid_thw": torch.tensor([[1, 1, 4]]),
    }
    features = torch.tensor([[10.0, 0.0, 0.0, 0.0], [20.0, 0.0, 0.0, 0.0]])

    packed = build_qwen_inputs_from_video_features(FakeQwenModel(), inputs, features)

    assert packed["input_ids"].tolist() == [[1, 999, 999, 2]]
    assert packed["attention_mask"].tolist() == [[1, 1, 1, 1]]
    assert packed["inputs_embeds"][0, 1].tolist() == [10.0, 0.0, 0.0, 0.0]
    assert packed["inputs_embeds"][0, 2].tolist() == [20.0, 0.0, 0.0, 0.0]
    assert "pixel_values_videos" not in packed
    assert "video_grid_thw" not in packed
    assert packed["qwen_video_feature_metadata"]["visual_tokens_before_prune"] == 2
    assert packed["qwen_video_feature_metadata"]["visual_tokens_after_prune"] == 2


def test_build_qwen_inputs_from_video_features_keeps_sparse_placeholders():
    torch = __import__("torch")

    class FakeQwenModel:
        def __init__(self):
            self.config = type("Config", (), {"video_token_id": 999})()
            self.embedding = torch.nn.Embedding(1200, 4)

        def get_input_embeddings(self):
            return self.embedding

    inputs = {
        "input_ids": torch.tensor([[1, 999, 999, 999, 999, 2]]),
        "attention_mask": torch.ones((1, 6), dtype=torch.long),
        "pixel_values_videos": torch.zeros((16, 3)),
        "video_grid_thw": torch.tensor([[1, 4, 4]]),
    }
    features = torch.tensor([[20.0, 0.0, 0.0, 0.0], [40.0, 0.0, 0.0, 0.0]])

    packed = build_qwen_inputs_from_video_features(
        FakeQwenModel(),
        inputs,
        features,
        original_keep_indices=[1, 3],
        metadata_key="qwen_chunked_vit_sparse_metadata",
    )

    assert packed["input_ids"].tolist() == [[1, 999, 999, 2]]
    assert packed["inputs_embeds"][0, 1].tolist() == [20.0, 0.0, 0.0, 0.0]
    assert packed["inputs_embeds"][0, 2].tolist() == [40.0, 0.0, 0.0, 0.0]
    assert packed["qwen_chunked_vit_sparse_metadata"]["kept_original_visual_indices"] == [1, 3]
    assert packed["qwen_chunked_vit_sparse_metadata"]["visual_tokens_before_prune"] == 4
    assert packed["qwen_chunked_vit_sparse_metadata"]["visual_tokens_after_prune"] == 2


def test_qwen_chunked_video_features_runs_dense_and_sparse_chunks():
    torch = __import__("torch")

    class FakeMerger(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states.reshape(-1, 4, 1).sum(dim=1)

    class FakeVisual(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dtype = torch.float32
            self.spatial_merge_unit = 4
            self.patch_embed = torch.nn.Identity()
            self.blocks = torch.nn.ModuleList([])
            self.merger = FakeMerger()

        def rot_pos_emb(self, grid_thw):
            t, h, w = [int(item) for item in grid_thw[0].tolist()]
            return torch.zeros((t * h * w, 1), dtype=torch.float32)

    class FakeModel:
        def __init__(self):
            self.visual = FakeVisual()
            self.config = type("Config", (), {"vision_config": type("VisionConfig", (), {"spatial_merge_size": 2})()})()

    values = {
        "pixel_values_videos": torch.arange(32, dtype=torch.float32).reshape(32, 1),
        "video_grid_thw": torch.tensor([[4, 2, 4]]),
    }

    dense_features, dense_metadata = qwen_chunked_video_features(FakeModel(), values, chunk_frames=2)
    sparse_features, sparse_metadata = qwen_chunked_video_features(
        FakeModel(),
        values,
        chunk_frames=2,
        keep_indices=[1, 6],
    )

    assert dense_features.flatten().tolist() == [10.0, 18.0, 42.0, 50.0, 74.0, 82.0, 106.0, 114.0]
    assert dense_metadata["chunk_count"] == 2
    assert dense_metadata["visual_tokens_after_prune"] == 8
    assert sparse_features.flatten().tolist() == [18.0, 106.0]
    assert sparse_metadata["executed_chunk_count"] == 2
    assert sparse_metadata["visual_tokens_after_prune"] == 2


def test_qwen_chunked_video_features_runs_spatial_tiles_after_processor():
    torch = __import__("torch")

    class FakeMerger(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states.reshape(-1, 4, 1).sum(dim=1)

    class FakeVisual(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dtype = torch.float32
            self.spatial_merge_unit = 4
            self.patch_embed = torch.nn.Identity()
            self.blocks = torch.nn.ModuleList([])
            self.merger = FakeMerger()

        def rot_pos_emb(self, grid_thw):
            t, h, w = [int(item) for item in grid_thw[0].tolist()]
            return torch.zeros((t * h * w, 1), dtype=torch.float32)

    class FakeModel:
        def __init__(self):
            self.visual = FakeVisual()
            self.config = type("Config", (), {"vision_config": type("VisionConfig", (), {"spatial_merge_size": 2})()})()

    values = {
        "pixel_values_videos": torch.arange(32, dtype=torch.float32).reshape(32, 1),
        "video_grid_thw": torch.tensor([[2, 4, 4]]),
    }

    features, metadata = qwen_chunked_video_features(
        FakeModel(),
        values,
        chunk_frames=1,
        max_spatial_chunks=4,
    )
    sparse_features, sparse_metadata = qwen_chunked_video_features(
        FakeModel(),
        values,
        chunk_frames=1,
        max_spatial_chunks=4,
        keep_indices=[1, 4],
    )

    assert features.flatten().tolist() == [10.0, 18.0, 42.0, 50.0, 74.0, 82.0, 106.0, 114.0]
    assert metadata["spatial_chunking"]["tile_grid"] == {"cols": 2, "rows": 2, "tiles": 4}
    assert metadata["spatial_chunking"]["mode"] == "qwen_processor_grid_spatial_tiles"
    assert sparse_features.flatten().tolist() == [18.0, 74.0]
    assert sparse_metadata["executed_chunk_count"] == 2
    assert sparse_metadata["spatial_chunking"]["max_spatial_chunks"] == 4


def test_qwen_grid_adapter_routes_chunked_vit_mode(monkeypatch):
    adapter = resolve_mllm_adapter("qwen3-vl")

    def fake_run(self, request):
        return MllmRunResult(
            text="chunked answer",
            prompt=request.prompt,
            video=request.video,
            image=request.image,
            adapter=self.name,
            status="executed",
            metrics={"qwen_vit": {"mode": request.qwen_vit_mode, "chunk_frames": request.qwen_vit_chunk_frames}},
        )

    monkeypatch.setattr(QwenGridMllmAdapter, "_run_qwen_chunked_vit_generate", fake_run)

    result = adapter.run(make_request(qwen_vit_mode="qwen_chunked_vit", qwen_vit_chunk_frames=8))

    assert result.text == "chunked answer"
    assert result.metrics["qwen_vit"] == {"mode": "qwen_chunked_vit", "chunk_frames": 8}


def test_qwen_grid_adapter_routes_chunked_vit_autogaze_sparse_mode(monkeypatch):
    adapter = resolve_mllm_adapter("qwen3-vl")

    def fake_run(self, request):
        return MllmRunResult(
            text="sparse chunked answer",
            prompt=request.prompt,
            video=request.video,
            image=request.image,
            adapter=self.name,
            status="executed",
            metrics={"qwen_vit": {"mode": request.qwen_vit_mode}},
        )

    monkeypatch.setattr(QwenGridMllmAdapter, "_run_qwen_chunked_vit_autogaze_sparse_generate", fake_run)

    result = adapter.run(
        make_request(
            qwen_vit_mode="qwen_chunked_vit_autogaze_sparse",
            token_selector_kind="autogaze",
            integration_level="pre_encoder_sparse",
            pre_encoder_prune_adapter="autogaze-sparse",
            enable_qwen_prune_generate=True,
            sparse_selection_plan_path="outputs/plan.json",
        )
    )

    assert result.text == "sparse chunked answer"
    assert result.metrics["qwen_vit"]["mode"] == "qwen_chunked_vit_autogaze_sparse"


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
