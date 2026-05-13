from __future__ import annotations

import json
import logging
import sys
import types
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import infer_autogaze
import infer_full
from poc_infer_utils import SCALE_COLORS, load_config, prepare_video, resolve_frame_selection_max_windows
from poc_model_registry import build_mllm, build_vision_encoder
from poc_model_adapters import NVILAAdapter, QwenAdapter, VJEPA2Adapter


POC_CONFIGS = [
    "A0_vanilla_siglip_nvila_off.yaml",
    "A1_modified_siglip_nvila_off.yaml",
    "A2_modified_siglip_nvila_on.yaml",
    "A3_vanilla_siglip_nvila_on.yaml",
    "E1_vjepa2_encoder.yaml",
    "E2_qwen_mllm.yaml",
    "E3_vjepa2_qwen.yaml",
]


def _cfg(name: str) -> Path:
    return ROOT / "configs" / "poc_inference" / name


def _write_cfg(tmp_path: Path, cfg: dict, name: str = "config.yaml") -> Path:
    cfg_path = tmp_path / name
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    return cfg_path


def test_priority1_configs_load_and_name_required_models() -> None:
    seen = set()
    for name in POC_CONFIGS:
        cfg = load_config(_cfg(name))
        seen.add(cfg["experiment"]["id"])
        assert cfg["vision_encoder"]["name"] in {"modified_siglip", "vanilla_siglip", "vjepa2", "generic_vit"}
        assert cfg["mllm"]["name"] in {"nvila", "qwen", "generic_mllm"}
        assert "checkpoint_path" in cfg["vision_encoder"]
        assert "processor_path" in cfg["mllm"]
    assert seen == {"A0", "A1", "A2", "A3", "E1", "E2", "E3"}


def test_configs_reference_local_weight_cache_when_available() -> None:
    a2 = load_config(_cfg("A2_modified_siglip_nvila_on.yaml"))
    assert a2["autogaze"]["checkpoint_path"] == "weights/AutoGaze"
    assert a2["vision_encoder"]["checkpoint_path"] == "weights/siglip2-base-patch16-224"
    assert a2["vision_encoder"]["from_pretrained_kwargs"]["scales"] == "32+64+112+224"
    assert a2["mllm"]["checkpoint_path"] == "weights/NVILA-8B-HD-Video"
    assert a2["mllm"]["processor_from_pretrained_kwargs"]["use_fast"] is False
    assert a2["mllm"]["processor_from_pretrained_kwargs"]["autogaze_model_id"] == "weights/AutoGaze"
    assert a2["mllm"]["processor_from_pretrained_kwargs"]["num_video_frames"] == 16
    assert a2["mllm"]["processor_from_pretrained_kwargs"]["num_video_frames_thumbnail"] == 16
    assert a2["mllm"]["local_files_only"] is True
    assert a2["mllm"]["trust_remote_code"] is True
    assert a2["mllm"]["class_name"] == "AutoModel"
    assert a2["mllm"]["official_processor_owns_vision"] is True
    assert a2["mllm"]["sync_autogaze_controls_from_config"] is True
    assert a2["runtime"]["warmup_runs"] == 1
    assert a2["runtime"]["progress"] is True

    e1 = load_config(_cfg("E1_vjepa2_encoder.yaml"))
    assert e1["vision_encoder"]["checkpoint_path"] == "weights/vjepa2-vitl-fpc64-256"
    assert e1["vision_encoder"]["processor_path"] == "weights/vjepa2-vitl-fpc64-256"
    assert e1["vision_encoder"]["processor_from_pretrained_kwargs"]["use_fast"] is False
    assert e1["vision_encoder"]["resolution"] == 256

    e2 = load_config(_cfg("E2_qwen_mllm.yaml"))
    assert e2["mllm"]["name"] == "qwen"
    assert e2["mllm"]["checkpoint_path"] == "weights/Qwen2.5-VL-7B-Instruct"
    assert e2["mllm"]["processor_path"] == "weights/Qwen2.5-VL-7B-Instruct"
    assert e2["mllm"]["prompt_template"] == "Question: {prompt}"
    assert e2["mllm"]["processor_from_pretrained_kwargs"]["use_fast"] is False


def test_cli_parsing_and_model_overrides() -> None:
    args = infer_full.parse_args(
        [
            "--config",
            str(_cfg("A2_modified_siglip_nvila_on.yaml")),
            "--video-path",
            "dummy",
            "--query-text",
            "Do not ignore this.",
            "--vision-encoder",
            "vjepa2",
            "--vision-encoder-ckpt",
            "/tmp/vjepa2.pt",
            "--mllm",
            "qwen",
            "--model-id",
            "local-qwen",
            "--processor-path",
            "/tmp/processor",
        ]
    )
    cfg = infer_full._with_model_overrides(load_config(args.config), args)
    assert cfg["vision_encoder"]["name"] == "vjepa2"
    assert cfg["vision_encoder"]["checkpoint_path"] == "/tmp/vjepa2.pt"
    assert cfg["mllm"]["name"] == "qwen"
    assert cfg["mllm"]["model_id"] == "local-qwen"
    assert cfg["mllm"]["checkpoint_path"] is None
    assert cfg["mllm"]["processor_path"] == "/tmp/processor"

    with pytest.raises(SystemExit):
        infer_autogaze.parse_args(
            [
                "--config",
                str(_cfg("A2_modified_siglip_nvila_on.yaml")),
                "--video-path",
                "dummy",
                "--frame-stride",
                "2",
            ]
        )


def test_no_silent_fallback_between_model_types() -> None:
    assert build_vision_encoder("vjepa2").name == "vjepa2"
    assert build_mllm("qwen").name == "qwen"
    with pytest.raises(ValueError):
        build_vision_encoder("not_a_model")
    with pytest.raises(ValueError):
        build_mllm("not_a_model")


def test_frame_selection_scaling_and_chop_metadata() -> None:
    cfg = load_config(_cfg("A2_modified_siglip_nvila_on.yaml"))
    assert resolve_frame_selection_max_windows(
        cli_max_windows=None,
        cfg=cfg,
        frame_selection_mode="all",
        cli_frame_selection_mode="all",
    ) is None
    assert resolve_frame_selection_max_windows(
        cli_max_windows=0,
        cfg=cfg,
        frame_selection_mode="all",
        cli_frame_selection_mode="all",
    ) is None
    assert resolve_frame_selection_max_windows(
        cli_max_windows=None,
        cfg=cfg,
        frame_selection_mode="chunk",
        cli_frame_selection_mode="chunk",
    ) == 1

    for mode in ("sample", "chunk", "interval", "all"):
        prepared = prepare_video(
            cfg,
            video_path="dummy",
            frame_selection_mode=mode,
            num_frames=2,
            frame_interval=2,
            max_windows=2,
            scaling_mode="resize",
            resolution=32,
            chop_size=16,
            chop_overlap=0,
            max_chops=None,
            chop_merge_mode="metadata_only",
        )
        assert prepared.frame_selection.mode == mode
        assert prepared.processed_video.shape[-2:] == (32, 32)

    for scaling_mode in ("fit_short_side", "fit_long_side"):
        prepared = prepare_video(
            cfg,
            video_path="dummy",
            frame_selection_mode="sample",
            num_frames=2,
            frame_interval=1,
            max_windows=1,
            scaling_mode=scaling_mode,
            resolution=32,
            chop_size=16,
            chop_overlap=0,
            max_chops=None,
            chop_merge_mode="metadata_only",
        )
        assert prepared.scaling_metadata["windows"][0]["scaling_mode"] == scaling_mode

    chopped = prepare_video(
        cfg,
        video_path="dummy",
        frame_selection_mode="sample",
        num_frames=2,
        frame_interval=1,
        max_windows=1,
        scaling_mode="chop",
        resolution=32,
        chop_size=16,
        chop_overlap=4,
        max_chops=3,
        chop_merge_mode="metadata_only",
    )
    assert chopped.chop_metadata is not None
    assert chopped.chop_metadata["windows"][0]["records"]
    assert len(chopped.chop_metadata["windows"][0]["records"]) == 3
    assert chopped.chop_metadata["windows"][0]["status"] == "actual_spatial_chops"
    assert chopped.chop_metadata["windows"][0]["processed_frame_count"] == 6
    assert chopped.processed_video.shape == (3, 2, 3, 32, 32)
    assert len(chopped.frame_records) == 6
    assert chopped.frame_records[0]["chop_index"] == 0
    assert chopped.frame_records[2]["chop_index"] == 1


def test_chop_mode_expands_processed_frames_and_token_counts(tmp_path: Path) -> None:
    output_dir = tmp_path / "autogaze_chop"
    args = infer_autogaze.parse_args(
        [
            "--config",
            str(_cfg("A2_modified_siglip_nvila_on.yaml")),
            "--video-path",
            "dummy",
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--frame-selection-mode",
            "sample",
            "--num-frames",
            "2",
            "--scaling-mode",
            "chop",
            "--resolution",
            "32",
            "--chop-size",
            "16",
            "--max-chops",
            "3",
            "--gaze-ratio",
            "0.25",
            "--no-progress",
        ]
    )
    summary = infer_autogaze.run(args)
    assert summary["status"] == "partial"
    metrics = json.loads((output_dir / "logs" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["number_of_source_frames"] == 2
    assert metrics["number_of_processed_frames"] == 6
    assert metrics["spatial_chops_per_window"] == {"0": 3}
    assert metrics["original_token_count"] == 6 * 265
    selected = json.loads((output_dir / "autogaze" / "selected_patch_indices.json").read_text(encoding="utf-8"))
    assert len(selected["frames"]) == 6
    assert selected["frames"][0]["chop_index"] == 0
    assert selected["frames"][2]["chop_index"] == 1
    viz = json.loads((output_dir / "visualizations" / "autogaze" / "metadata" / "visualization_metadata.json").read_text(encoding="utf-8"))
    assert viz["visualization_mode"] == "merged_chop_source_frames"
    assert viz["frame_count"] == 2
    assert viz["processed_crop_frame_count"] == 6
    assert viz["frame_records"][0]["chop_count"] == 3
    assert (output_dir / "visualizations" / "autogaze" / "frames" / "frame_000001_overlay.png").exists()
    assert not (output_dir / "visualizations" / "autogaze" / "frames" / "frame_000005_overlay.png").exists()


def test_cli_all_frame_selection_saves_all_dummy_frames(tmp_path: Path) -> None:
    output_dir = tmp_path / "autogaze_all"
    args = infer_autogaze.parse_args(
        [
            "--config",
            str(_cfg("A2_modified_siglip_nvila_on.yaml")),
            "--video-path",
            "dummy",
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--frame-selection-mode",
            "all",
            "--num-frames",
            "3",
            "--scaling-mode",
            "resize",
            "--resolution",
            "32",
            "--no-progress",
        ]
    )
    infer_autogaze.run(args)
    frame_selection = json.loads((output_dir / "autogaze" / "frame_selection_metadata.json").read_text(encoding="utf-8"))
    visualization = json.loads(
        (output_dir / "visualizations" / "autogaze" / "metadata" / "visualization_metadata.json").read_text(encoding="utf-8")
    )
    assert frame_selection["mode"] == "all"
    assert frame_selection["max_windows"] is None
    assert frame_selection["original_frame_count"] == 8
    assert visualization["frame_count"] == 8
    assert (output_dir / "visualizations" / "autogaze" / "frames" / "frame_000007_overlay.png").exists()


def test_chop_mllm_generation_falls_back_to_source_video_for_text() -> None:
    calls: list[str | None] = []

    class FakeMLLM:
        def supports_direct_visual_tokens(self):
            return False

        def generate(self, *, query_text, video, visual_tokens=None, max_new_tokens=32, video_path=None):
            calls.append(video_path)
            if video_path is None:
                return {
                    "status": "blocked",
                    "answer": None,
                    "reason": "processor rejected flat chop tensor",
                    "query_text_used": True,
                    "metadata": {"video_input_kind": "processed_tensor_pil_frames"},
                }
            return {
                "status": "real",
                "answer": "source video answer",
                "reason": None,
                "query_text_used": True,
                "metadata": {"video_input_kind": "video_reference"},
            }

    result = infer_full._generate_mllm_with_video_policy(
        FakeMLLM(),
        query_text="What changed?",
        video=torch.zeros(2, 16, 3, 8, 8),
        visual_tokens=None,
        max_new_tokens=4,
        video_path="/tmp/source.mp4",
        has_chop_metadata=True,
        allow_chop_source_fallback=True,
    )
    assert calls == [None, "/tmp/source.mp4"]
    assert result["status"] == "real"
    assert result["answer"] == "source video answer"
    assert result["metadata"]["actual_video_input_source"] == "source_video_path_after_chop_tensor_failure"
    assert result["metadata"]["chop_tensor_attempted"] is True
    assert result["metadata"]["chop_source_fallback_used"] is True
    assert result["metadata"]["chop_tensor_failure_reason"] == "processor rejected flat chop tensor"


def test_autogaze_dummy_run_writes_flat_outputs_and_metrics(tmp_path: Path) -> None:
    output_dir = tmp_path / "autogaze"
    args = infer_autogaze.parse_args(
        [
            "--config",
            str(_cfg("A2_modified_siglip_nvila_on.yaml")),
            "--video-path",
            "dummy",
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--frame-selection-mode",
            "sample",
            "--num-frames",
            "2",
            "--scaling-mode",
            "resize",
            "--resolution",
            "32",
            "--gaze-ratio",
            "0.25",
            "--save-side-by-side-video",
        ]
    )
    summary = infer_autogaze.run(args)
    assert summary["status"] == "partial"
    assert (output_dir / "autogaze" / "selected_patch_indices.json").exists()
    assert (output_dir / "visualizations" / "autogaze" / "frames" / "frame_000000_overlay.png").exists()
    assert (output_dir / "visualizations" / "autogaze" / "scale_panels" / "frame_000000_scale_panel.png").exists()
    assert (output_dir / "visualizations" / "autogaze" / "videos" / "autogaze_side_by_side.mp4").exists()
    assert not (output_dir / "visualizations" / "autogaze" / "windows").exists()
    metrics = json.loads((output_dir / "logs" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["real_stub_blocked_status"] == "stub_dummy_autogaze"
    assert metrics["gaze_ratio"] == 0.25
    assert metrics["warmup_runs"] == 1
    assert metrics["autogaze_latency_ms"] == 0.0
    assert metrics["module_processing_latency_ms"] == metrics["end_to_end_latency_ms"]
    assert metrics["visualization_latency_ms"] is not None
    assert metrics["wall_clock_latency_ms"] >= metrics["module_processing_latency_ms"]
    selected = json.loads((output_dir / "autogaze" / "selected_patch_indices.json").read_text(encoding="utf-8"))
    records = selected["frames"][0]["selected_patch_records"]
    widths_by_scale = {
        record["scale"]: record["normalized_box"][2] - record["normalized_box"][0]
        for record in records
    }
    assert widths_by_scale[0] > widths_by_scale[1] > widths_by_scale[2] > widths_by_scale[3]
    viz = json.loads((output_dir / "visualizations" / "autogaze" / "metadata" / "visualization_metadata.json").read_text(encoding="utf-8"))
    assert viz["flat_output_structure"] is True
    assert [item["scale_resolution"] for item in viz["scale_layout"]] == [32, 64, 112, 224]
    assert viz["scale_colors"] == {str(key): list(value) for key, value in SCALE_COLORS.items()}


def test_full_pipeline_preserves_query_and_writes_reports(tmp_path: Path) -> None:
    output_dir = tmp_path / "full"
    args = infer_full.parse_args(
        [
            "--config",
            str(_cfg("A2_modified_siglip_nvila_on.yaml")),
            "--video-path",
            "dummy",
            "--query-text",
            "What is happening?",
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--frame-selection-mode",
            "sample",
            "--num-frames",
            "2",
            "--scaling-mode",
            "resize",
            "--resolution",
            "32",
            "--max-new-tokens",
            "4",
        ]
    )
    summary = infer_full.run(args)
    assert summary["status"] == "partial"
    answer = json.loads((output_dir / "predictions" / "answer.json").read_text(encoding="utf-8"))
    assert answer["query_text"] == "What is happening?"
    assert answer["query_text_used"] is True
    assert answer["status"] == "stub"
    metrics = json.loads((output_dir / "logs" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["query_text"] == "What is happening?"
    assert metrics["requested_vision_encoder"] == "modified_siglip"
    assert metrics["requested_mllm"] == "nvila"
    assert metrics["warmup_runs"] == 1
    assert metrics["mllm_generation_latency_ms"] is not None
    assert metrics["module_processing_latency_ms"] == metrics["end_to_end_latency_ms"]
    assert metrics["visualization_latency_ms"] is not None
    assert (output_dir / "logs" / "metrics.csv").exists()


def test_allow_real_autogaze_missing_checkpoint_blocks_even_with_dummy_video(tmp_path: Path) -> None:
    output_dir = tmp_path / "autogaze_blocked"
    cfg = load_config(_cfg("A2_modified_siglip_nvila_on.yaml"))
    cfg["autogaze"]["checkpoint_path"] = str(tmp_path / "missing_autogaze")
    cfg["autogaze"]["processor_path"] = str(tmp_path / "missing_autogaze")
    cfg_path = _write_cfg(tmp_path, cfg)
    args = infer_autogaze.parse_args(
        [
            "--config",
            str(cfg_path),
            "--video-path",
            "dummy",
            "--output-dir",
            str(output_dir),
            "--allow-real-model-loading",
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--frame-selection-mode",
            "sample",
            "--num-frames",
            "2",
            "--scaling-mode",
            "resize",
            "--resolution",
            "32",
        ]
    )
    summary = infer_autogaze.run(args)
    assert summary["status"] == "blocked"
    metrics = json.loads((output_dir / "logs" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["real_stub_blocked_status"] == "blocked"
    assert "checkpoint/model is missing" in metrics["failure_reason"]


def test_vjepa2_real_loading_with_available_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = types.ModuleType("fake_vjepa2")

    class AutoModel:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs):
            instance = cls()
            instance.model_id = model_id
            instance.kwargs = kwargs
            return instance

        def to(self, device: str):
            self.device = device
            return self

        def eval(self):
            return self

        def __call__(self, pixel_values=None, videos=None):
            video = pixel_values if pixel_values is not None else videos
            batch, frames = int(video.shape[0]), int(video.shape[1])
            return types.SimpleNamespace(last_hidden_state=torch.ones(batch, frames, 5))

    fake_module.AutoModel = AutoModel
    monkeypatch.setitem(sys.modules, "fake_vjepa2", fake_module)

    adapter = VJEPA2Adapter(
        {
            "module_path": "fake_vjepa2",
            "class_name": "AutoModel",
            "model_id": "fake/vjepa2",
            "local_files_only": True,
        }
    )
    status = adapter.load(allow_real_model_loading=True, device="cpu", dtype="float32")
    assert status.status == "real"
    assert status.metadata["processor_status"] == "not_configured_tensor_input"
    assert adapter.model.kwargs["dtype"] is torch.float32
    assert "torch_dtype" not in adapter.model.kwargs
    output = adapter.forward(torch.zeros(1, 2, 3, 8, 8))
    assert output["status"] == "real"
    assert output["visual_tokens"].shape == (1, 2, 5)


def test_transformers_torch_dtype_warning_is_suppressed_during_model_load(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake_module = types.ModuleType("fake_warning_model")

    class AutoModel:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs):
            logging.getLogger("transformers.configuration_utils").warning(
                "`torch_dtype` is deprecated! Use `dtype` instead!"
            )
            instance = cls()
            instance.model_id = model_id
            instance.kwargs = kwargs
            return instance

        def to(self, device: str):
            return self

        def eval(self):
            return self

    fake_module.AutoModel = AutoModel
    monkeypatch.setitem(sys.modules, "fake_warning_model", fake_module)
    adapter = VJEPA2Adapter(
        {
            "module_path": "fake_warning_model",
            "class_name": "AutoModel",
            "model_id": "fake/warning-model",
            "local_files_only": True,
        }
    )
    with caplog.at_level(logging.WARNING, logger="transformers.configuration_utils"):
        status = adapter.load(allow_real_model_loading=True, device="cpu", dtype="float32")
    assert status.status == "real"
    assert "`torch_dtype` is deprecated" not in caplog.text


def test_qwen_official_processor_path_with_available_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = types.ModuleType("fake_qwen")

    class AutoModelForVision2Seq:
        device = torch.device("cpu")

        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs):
            instance = cls()
            instance.model_id = model_id
            instance.kwargs = kwargs
            return instance

        def to(self, device: str):
            self.device = torch.device(device)
            return self

        def eval(self):
            return self

        def generate(self, **inputs):
            assert "input_ids" in inputs
            return torch.tensor([[10, 11, 12, 13]])

    class AutoProcessor:
        @classmethod
        def from_pretrained(cls, processor_path: str, **kwargs):
            instance = cls()
            instance.processor_path = processor_path
            instance.kwargs = kwargs
            return instance

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            assert tokenize is False
            assert add_generation_prompt is True
            assert messages[0]["content"][0]["type"] == "video"
            assert messages[0]["content"][1]["text"] == "Question: What is happening?"
            return "<qwen-chat>Question: What is happening?</qwen-chat>"

        def __call__(self, *, text, videos=None, return_tensors=None, **_kwargs):
            assert text == ["<qwen-chat>Question: What is happening?</qwen-chat>"]
            assert isinstance(videos, list)
            assert len(videos) == 1
            assert isinstance(videos[0], list)
            assert return_tensors == "pt"
            return {"input_ids": torch.tensor([[10, 11]])}

        def batch_decode(self, outputs, skip_special_tokens=True):
            assert skip_special_tokens is True
            assert outputs.tolist() == [[12, 13]]
            return ["official qwen answer"]

    fake_module.AutoModelForVision2Seq = AutoModelForVision2Seq
    fake_module.AutoProcessor = AutoProcessor
    monkeypatch.setitem(sys.modules, "fake_qwen", fake_module)

    adapter = QwenAdapter(
        {
            "module_path": "fake_qwen",
            "class_name": "AutoModelForVision2Seq",
            "processor_module_path": "fake_qwen",
            "processor_class_name": "AutoProcessor",
            "model_id": "fake/qwen",
            "processor_path": "fake/qwen",
            "local_files_only": True,
            "prompt_template": "Question: {prompt}",
        }
    )
    status = adapter.load(allow_real_model_loading=True, device="cpu", dtype="float32")
    assert status.status == "real"
    assert status.metadata["processor_status"] == "real"
    assert adapter.model.kwargs["dtype"] is torch.float32
    assert "torch_dtype" not in adapter.model.kwargs
    assert adapter.processor.kwargs["use_fast"] is False
    result = adapter.generate(
        query_text="What is happening?",
        video=torch.zeros(1, 2, 3, 8, 8),
        max_new_tokens=4,
    )
    assert result["status"] == "real"
    assert result["answer"] == "official qwen answer"
    assert result["official_processor_path"] is True
    assert result["metadata"]["chat_template_path"] is True
    assert result["metadata"]["vision_preprocess_path"] == "processor_direct_video_payload"
    assert result["metadata"]["video_input_kind"] == "processed_tensor_pil_frames"


def test_qwen_blocks_without_chat_template(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = types.ModuleType("fake_qwen_no_chat_template")

    class AutoModelForVision2Seq:
        device = torch.device("cpu")

        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

        def eval(self):
            return self

        def generate(self, **_inputs):
            raise AssertionError("Qwen generation should block before model.generate")

    class AutoProcessor:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

        def __call__(self, **_kwargs):
            raise AssertionError("Qwen generation should block before processor call")

    fake_module.AutoModelForVision2Seq = AutoModelForVision2Seq
    fake_module.AutoProcessor = AutoProcessor
    monkeypatch.setitem(sys.modules, "fake_qwen_no_chat_template", fake_module)

    adapter = QwenAdapter(
        {
            "module_path": "fake_qwen_no_chat_template",
            "class_name": "AutoModelForVision2Seq",
            "processor_module_path": "fake_qwen_no_chat_template",
            "processor_class_name": "AutoProcessor",
            "model_id": "fake/qwen",
            "processor_path": "fake/qwen",
        }
    )
    status = adapter.load(allow_real_model_loading=True, device="cpu", dtype="float32")
    assert status.status == "real"
    assert status.metadata["chat_template_path"] is False
    result = adapter.generate(query_text="What is happening?", video=torch.zeros(1, 2, 3, 8, 8))
    assert result["status"] == "blocked"
    assert "apply_chat_template" in str(result["reason"])


def test_qwen_incomplete_local_shards_block_before_loading(tmp_path: Path) -> None:
    checkpoint = tmp_path / "qwen"
    checkpoint.mkdir()
    (checkpoint / "model-00002-of-00002.safetensors").write_bytes(b"stub")
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 10},
                "weight_map": {
                    "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                    "lm_head.weight": "model-00002-of-00002.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )
    adapter = QwenAdapter(
        {
            "module_path": "missing_module_should_not_import",
            "class_name": "AutoModelForVision2Seq",
            "checkpoint_path": str(checkpoint),
            "processor_path": str(checkpoint),
        }
    )
    status = adapter.load(allow_real_model_loading=True, device="cpu", dtype="float32")
    assert status.status == "blocked"
    assert "checkpoint is incomplete" in str(status.reason)
    assert status.metadata["missing_shards"] == ["model-00001-of-00002.safetensors"]


def test_nvila_official_processor_path_and_autogaze_controls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_module = types.ModuleType("fake_nvila")

    class AutoModel:
        device = torch.device("cpu")

        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs):
            instance = cls()
            instance.model_id = model_id
            instance.kwargs = kwargs
            return instance

        def eval(self):
            return self

        def generate(self, **inputs):
            assert "input_ids" in inputs
            return torch.tensor([[21, 22, 23]])

    class AutoProcessor:
        tokenizer = types.SimpleNamespace(video_token="<video>")

        @classmethod
        def from_pretrained(cls, processor_path: str, **kwargs):
            instance = cls()
            instance.processor_path = processor_path
            instance.kwargs = kwargs
            instance.num_video_frames = kwargs.get("num_video_frames")
            return instance

        def __call__(self, *, text, videos=None, return_tensors=None, **_kwargs):
            assert text == "<video>\n\nQuestion: What changed?"
            assert isinstance(videos, list)
            assert len(videos) == 16
            assert return_tensors == "pt"
            return {"input_ids": torch.tensor([[21, 22]])}

        def batch_decode(self, outputs, skip_special_tokens=True):
            assert outputs.tolist() == [[23]]
            return ["nvila answer"]

    fake_module.AutoModel = AutoModel
    fake_module.AutoProcessor = AutoProcessor
    monkeypatch.setitem(sys.modules, "fake_nvila", fake_module)
    autogaze_dir = tmp_path / "AutoGaze"
    autogaze_dir.mkdir()
    (autogaze_dir / "config.json").write_text(json.dumps({"max_num_frames": 16}), encoding="utf-8")

    adapter = NVILAAdapter(
        {
            "module_path": "fake_nvila",
            "class_name": "AutoModel",
            "processor_module_path": "fake_nvila",
            "processor_class_name": "AutoProcessor",
            "model_id": "fake/nvila",
            "processor_path": "fake/nvila",
            "trust_remote_code": True,
            "prompt_template": "{video_token}\n\nQuestion: {prompt}",
            "sync_autogaze_controls_from_config": True,
            "poc_autogaze_enabled": False,
            "poc_gaze_ratio": 0.75,
            "poc_task_loss_requirement": 0.7,
            "poc_autogaze_checkpoint_path": str(autogaze_dir),
        }
    )
    status = adapter.load(allow_real_model_loading=True, device="cpu", dtype="float32")
    assert status.status == "real"
    assert status.metadata["processor_status"] == "real"
    assert adapter.model.kwargs["dtype"] is torch.float32
    assert "torch_dtype" not in adapter.model.kwargs
    assert adapter.processor.kwargs["use_fast"] is False
    assert status.metadata["processor_autogaze_controls"]["autogaze_model_id"] == str(autogaze_dir)
    assert status.metadata["processor_autogaze_controls"]["use_fast"] is False
    assert status.metadata["processor_autogaze_controls"]["num_video_frames"] == 16
    assert status.metadata["processor_autogaze_controls"]["num_video_frames_thumbnail"] == 16
    assert status.metadata["processor_autogaze_controls"]["gazing_ratio_tile"] is None
    assert status.metadata["processor_autogaze_controls"]["gazing_ratio_thumbnail"] is None
    result = adapter.generate(
        query_text="What changed?",
        video=torch.zeros(1, 2, 3, 8, 8),
        max_new_tokens=3,
        video_path="dummy",
    )
    assert result["status"] == "real"
    assert result["answer"] == "nvila answer"
    assert result["metadata"]["autogaze_visual_tokens_injected"] is False
    assert result["metadata"]["video_input_kind"] == "processed_tensor_pil_frames_to_16"


def test_nvila_tensor_video_input_flattens_chop_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = types.ModuleType("fake_nvila_chop")

    class AutoModel:
        device = torch.device("cpu")

        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs):
            return cls()

        def eval(self):
            return self

        def generate(self, **inputs):
            assert "input_ids" in inputs
            return torch.tensor([[31, 32, 33]])

    class AutoProcessor:
        tokenizer = types.SimpleNamespace(video_token="<video>")

        @classmethod
        def from_pretrained(cls, processor_path: str, **kwargs):
            instance = cls()
            instance.num_video_frames = kwargs.get("num_video_frames")
            return instance

        def __call__(self, *, text, videos=None, return_tensors=None, **_kwargs):
            assert isinstance(videos, list)
            assert len(videos) == 48
            return {"input_ids": torch.tensor([[31, 32]])}

        def batch_decode(self, outputs, skip_special_tokens=True):
            assert outputs.tolist() == [[33]]
            return ["chop nvila answer"]

    fake_module.AutoModel = AutoModel
    fake_module.AutoProcessor = AutoProcessor
    monkeypatch.setitem(sys.modules, "fake_nvila_chop", fake_module)

    adapter = NVILAAdapter(
        {
            "module_path": "fake_nvila_chop",
            "class_name": "AutoModel",
            "processor_module_path": "fake_nvila_chop",
            "processor_class_name": "AutoProcessor",
            "model_id": "fake/nvila",
            "processor_path": "fake/nvila",
            "prompt_template": "{video_token}\n\n{prompt}",
            "processor_from_pretrained_kwargs": {"num_video_frames": 16},
        }
    )
    status = adapter.load(allow_real_model_loading=True, device="cpu", dtype="float32")
    assert status.status == "real"
    result = adapter.generate(
        query_text="Use chops.",
        video=torch.zeros(3, 16, 3, 8, 8),
        max_new_tokens=3,
        video_path=None,
    )
    assert result["status"] == "real"
    assert result["metadata"]["video_input_kind"] == "processed_tensor_pil_frames_to_48"


def test_real_loading_blocked_does_not_fall_back_to_stub_tokens(tmp_path: Path) -> None:
    output_dir = tmp_path / "blocked"
    cfg = load_config(_cfg("E1_vjepa2_encoder.yaml"))
    cfg["vision_encoder"]["checkpoint_path"] = str(tmp_path / "missing_vjepa2")
    cfg["vision_encoder"]["processor_path"] = str(tmp_path / "missing_vjepa2")
    cfg_path = _write_cfg(tmp_path, cfg)
    args = infer_full.parse_args(
        [
            "--config",
            str(cfg_path),
            "--video-path",
            "dummy",
            "--query-text",
            "What is happening?",
            "--output-dir",
            str(output_dir),
            "--allow-real-model-loading",
            "--local-files-only",
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--frame-selection-mode",
            "sample",
            "--num-frames",
            "2",
            "--scaling-mode",
            "resize",
            "--resolution",
            "32",
        ]
    )
    summary = infer_full.run(args)
    assert summary["status"] == "blocked"
    assert "vision_encoder" in summary["blocked_stages"]
    metrics = json.loads((output_dir / "logs" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["adapter_statuses"]["vision_encoder"]["status"] == "blocked"
    assert "used stub output" not in json.dumps(metrics["skipped_stages"])


def test_infer_full_qwen_real_official_processor_integration(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_module = types.ModuleType("fake_qwen_full")

    class AutoModelForVision2Seq:
        device = torch.device("cpu")

        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

        def to(self, device: str):
            self.device = torch.device(device)
            return self

        def eval(self):
            return self

        def generate(self, **_inputs):
            return torch.tensor([[1, 2, 3]])

    class AutoProcessor:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            assert tokenize is False
            assert add_generation_prompt is True
            assert messages[0]["content"][0]["type"] == "video"
            assert messages[0]["content"][1]["text"] == "Question: Audit qwen?"
            return "<qwen-chat>Question: Audit qwen?</qwen-chat>"

        def __call__(self, *, text, videos=None, return_tensors=None, **_kwargs):
            assert text == ["<qwen-chat>Question: Audit qwen?</qwen-chat>"]
            assert isinstance(videos, list)
            assert len(videos) == 1
            assert isinstance(videos[0], list)
            return {"input_ids": torch.tensor([[1, 2]])}

        def batch_decode(self, outputs, skip_special_tokens=True):
            assert outputs.tolist() == [[3]]
            return ["qwen full answer"]

    fake_module.AutoModelForVision2Seq = AutoModelForVision2Seq
    fake_module.AutoProcessor = AutoProcessor
    monkeypatch.setitem(sys.modules, "fake_qwen_full", fake_module)

    cfg = load_config(_cfg("E2_qwen_mllm.yaml"))
    cfg["mllm"].update(
        {
            "module_path": "fake_qwen_full",
            "class_name": "AutoModelForVision2Seq",
            "processor_module_path": "fake_qwen_full",
            "processor_class_name": "AutoProcessor",
            "model_id": "fake/qwen",
            "processor_path": "fake/qwen",
            "prompt_template": "Question: {prompt}",
        }
    )
    cfg_path = _write_cfg(tmp_path, cfg, "qwen_real.yaml")
    output_dir = tmp_path / "qwen"
    args = infer_full.parse_args(
        [
            "--config",
            str(cfg_path),
            "--video-path",
            "dummy",
            "--query-text",
            "Audit qwen?",
            "--output-dir",
            str(output_dir),
            "--allow-real-model-loading",
            "--local-files-only",
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--frame-selection-mode",
            "sample",
            "--num-frames",
            "2",
            "--scaling-mode",
            "resize",
            "--resolution",
            "32",
        ]
    )
    summary = infer_full.run(args)
    assert summary["status"] == "completed"
    answer = json.loads((output_dir / "predictions" / "answer.json").read_text(encoding="utf-8"))
    assert answer["answer"] == "qwen full answer"
    assert answer["adapter_statuses"]["vision_encoder"]["status"] == "skipped"
    assert answer["adapter_statuses"]["mllm"]["status"] == "real"
    metrics = json.loads((output_dir / "logs" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["adapter_statuses"]["vision_encoder"]["status"] == "skipped"
    assert metrics["adapter_statuses"]["mllm"]["metadata"]["official_processor_path"] is True
    assert metrics["warmup_runs"] == 1
    assert metrics["mllm_generation_latency_ms"] is not None
    assert metrics["module_processing_latency_ms"] == metrics["end_to_end_latency_ms"]


def test_infer_full_nvila_real_official_processor_integration(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_module = types.ModuleType("fake_nvila_full")

    class AutoModel:
        device = torch.device("cpu")

        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

        def eval(self):
            return self

        def generate(self, **_inputs):
            return torch.tensor([[7, 8, 9]])

    class AutoProcessor:
        tokenizer = types.SimpleNamespace(video_token="<video>")

        @classmethod
        def from_pretrained(cls, *_args, **kwargs):
            instance = cls()
            instance.kwargs = kwargs
            return instance

        def __call__(self, *, text, videos=None, return_tensors=None, **_kwargs):
            assert text == "<video>\n\nNVILA question?"
            assert isinstance(videos, list)
            return {"input_ids": torch.tensor([[7, 8]])}

        def batch_decode(self, outputs, skip_special_tokens=True):
            assert outputs.tolist() == [[9]]
            return ["nvila full answer"]

    fake_module.AutoModel = AutoModel
    fake_module.AutoProcessor = AutoProcessor
    monkeypatch.setitem(sys.modules, "fake_nvila_full", fake_module)

    cfg = load_config(_cfg("A1_modified_siglip_nvila_off.yaml"))
    cfg["mllm"].update(
        {
            "module_path": "fake_nvila_full",
            "class_name": "AutoModel",
            "processor_module_path": "fake_nvila_full",
            "processor_class_name": "AutoProcessor",
            "model_id": "fake/nvila",
            "processor_path": "fake/nvila",
            "prompt_template": "{video_token}\n\n{prompt}",
        }
    )
    cfg_path = _write_cfg(tmp_path, cfg, "nvila_real.yaml")
    output_dir = tmp_path / "nvila"
    args = infer_full.parse_args(
        [
            "--config",
            str(cfg_path),
            "--video-path",
            "dummy",
            "--query-text",
            "NVILA question?",
            "--output-dir",
            str(output_dir),
            "--allow-real-model-loading",
            "--local-files-only",
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--frame-selection-mode",
            "sample",
            "--num-frames",
            "2",
            "--scaling-mode",
            "resize",
            "--resolution",
            "32",
        ]
    )
    summary = infer_full.run(args)
    assert summary["status"] == "completed"
    answer = json.loads((output_dir / "predictions" / "answer.json").read_text(encoding="utf-8"))
    assert answer["answer"] == "nvila full answer"
    assert answer["adapter_statuses"]["vision_encoder"]["status"] == "skipped"
    assert answer["adapter_statuses"]["mllm"]["status"] == "real"
    metrics = json.loads((output_dir / "logs" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["adapter_statuses"]["mllm"]["metadata"]["processor_autogaze_controls"]["gazing_ratio_tile"] is None
    assert metrics["adapter_statuses"]["mllm"]["metadata"]["official_processor_path"] is True
    assert metrics["warmup_runs"] == 1
    assert metrics["mllm_generation_latency_ms"] is not None
    assert metrics["module_processing_latency_ms"] == metrics["end_to_end_latency_ms"]
