from __future__ import annotations

import json
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
from poc_infer_utils import SCALE_COLORS, load_config, prepare_video
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
    assert a2["mllm"]["local_files_only"] is True
    assert a2["mllm"]["trust_remote_code"] is True
    assert a2["mllm"]["class_name"] == "AutoModel"
    assert a2["mllm"]["official_processor_owns_vision"] is True
    assert a2["mllm"]["sync_autogaze_controls_from_config"] is True

    e1 = load_config(_cfg("E1_vjepa2_encoder.yaml"))
    assert e1["vision_encoder"]["checkpoint_path"] == "weights/vjepa2-vitl-fpc64-256"
    assert e1["vision_encoder"]["processor_path"] == "weights/vjepa2-vitl-fpc64-256"
    assert e1["vision_encoder"]["resolution"] == 256

    e2 = load_config(_cfg("E2_qwen_mllm.yaml"))
    assert e2["mllm"]["name"] == "qwen"
    assert e2["mllm"]["checkpoint_path"] == "weights/Qwen2.5-VL-7B-Instruct"
    assert e2["mllm"]["processor_path"] == "weights/Qwen2.5-VL-7B-Instruct"


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
    output = adapter.forward(torch.zeros(1, 2, 3, 8, 8))
    assert output["status"] == "real"
    assert output["visual_tokens"].shape == (1, 2, 5)


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

        def __call__(self, *, text, videos=None, return_tensors=None, **_kwargs):
            assert "Question" in (text[0] if isinstance(text, list) else text)
            assert videos is not None
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
    result = adapter.generate(
        query_text="What is happening?",
        video=torch.zeros(1, 2, 3, 8, 8),
        max_new_tokens=4,
    )
    assert result["status"] == "real"
    assert result["answer"] == "official qwen answer"
    assert result["official_processor_path"] is True


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


def test_nvila_official_processor_path_and_autogaze_controls(monkeypatch: pytest.MonkeyPatch) -> None:
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
            return instance

        def __call__(self, *, text, videos=None, return_tensors=None, **_kwargs):
            assert text == "<video>\n\nQuestion: What changed?"
            assert isinstance(videos, list)
            assert return_tensors == "pt"
            return {"input_ids": torch.tensor([[21, 22]])}

        def batch_decode(self, outputs, skip_special_tokens=True):
            assert outputs.tolist() == [[23]]
            return ["nvila answer"]

    fake_module.AutoModel = AutoModel
    fake_module.AutoProcessor = AutoProcessor
    monkeypatch.setitem(sys.modules, "fake_nvila", fake_module)

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
        }
    )
    status = adapter.load(allow_real_model_loading=True, device="cpu", dtype="float32")
    assert status.status == "real"
    assert status.metadata["processor_status"] == "real"
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
    assert result["metadata"]["video_input_kind"] == "processed_tensor_pil_frames"


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

        def __call__(self, *, text, videos=None, return_tensors=None, **_kwargs):
            assert text == "Question: Audit qwen?"
            assert videos is not None
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
