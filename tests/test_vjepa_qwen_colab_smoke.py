from pathlib import Path

from repro.vjepa_qwen_colab_smoke import (
    DEFAULT_QWEN_MODEL,
    DEFAULT_VJEPA_MODEL,
    _ensure_vjepa_video_axis_order,
    build_parser,
    qwen_model_class_candidates,
    synthetic_vjepa_pixel_values,
)


def test_colab_smoke_defaults_are_cuda_and_qwen25():
    args = build_parser().parse_args([])

    assert args.require_cuda is True
    assert args.device == "cuda"
    assert args.qwen_model == DEFAULT_QWEN_MODEL
    assert args.qwen_model == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert args.vjepa_model == DEFAULT_VJEPA_MODEL
    assert args.output_json == "outputs/autogaze_vjepa/colab_vjepa_qwen_smoke.json"


def test_colab_smoke_can_point_to_local_checkpoint_paths(tmp_path):
    qwen = tmp_path / "Qwen2.5-VL-3B-Instruct"
    vjepa = tmp_path / "vjepa2-vitl-fpc64-256"

    args = build_parser().parse_args(
        [
            "--qwen-model",
            str(qwen),
            "--vjepa-model",
            str(vjepa),
            "--frames-per-clip",
            "4",
            "--crop-size",
            "224",
        ]
    )

    assert Path(args.qwen_model) == qwen
    assert Path(args.vjepa_model) == vjepa
    assert args.frames_per_clip == 4
    assert args.crop_size == 224


def test_qwen_model_class_candidates_prefer_image_text_to_text():
    names = [name for name, _ in qwen_model_class_candidates()]

    assert names[0] == "AutoModelForImageTextToText"
    assert "AutoModelForVision2Seq" in names


def test_synthetic_vjepa_pixel_values_use_transformers_video_axis_order():
    import pytest

    torch = pytest.importorskip("torch")

    values = synthetic_vjepa_pixel_values(
        frames_per_clip=4,
        crop_size=224,
        dtype=torch.float32,
        device="cpu",
    )

    assert list(values.shape) == [1, 4, 3, 224, 224]


def test_ensure_vjepa_video_axis_order_accepts_legacy_channel_first_tensors():
    import pytest

    torch = pytest.importorskip("torch")

    legacy = torch.zeros((1, 3, 4, 16, 16), dtype=torch.float32)
    normalized = _ensure_vjepa_video_axis_order(legacy)

    assert list(normalized.shape) == [1, 4, 3, 16, 16]
    assert normalized.is_contiguous()


def test_ensure_vjepa_video_axis_order_keeps_temporal_first_tensors():
    import pytest

    torch = pytest.importorskip("torch")

    values = torch.zeros((1, 4, 3, 16, 16), dtype=torch.float32)
    normalized = _ensure_vjepa_video_axis_order(values)

    assert list(normalized.shape) == [1, 4, 3, 16, 16]
