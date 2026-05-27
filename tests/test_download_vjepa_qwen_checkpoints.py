from pathlib import Path

from scripts.download_vjepa_qwen_checkpoints import (
    DEFAULT_AUTOGAZE_MODEL,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_QWEN_MODEL,
    DEFAULT_VJEPA_MODEL,
    build_download_plan,
    download_checkpoints,
    parse_args,
)


def test_vjepa_qwen_download_defaults_target_colab_weight_root():
    args = parse_args([])

    assert args.autogaze_model == DEFAULT_AUTOGAZE_MODEL
    assert args.vjepa_model == DEFAULT_VJEPA_MODEL
    assert args.qwen_model == DEFAULT_QWEN_MODEL
    assert args.output_root == DEFAULT_OUTPUT_ROOT


def test_vjepa_qwen_download_plan_separates_model_dirs():
    plan = build_download_plan(
        autogaze_model="nvidia/AutoGaze",
        vjepa_model="facebook/vjepa2-vitl-fpc64-256",
        qwen_model="Qwen/Qwen2.5-VL-3B-Instruct",
        output_root=Path("/content/autogaze_weights"),
        revision="main",
        max_workers=4,
    )

    assert plan["models"]["autogaze"]["repo_id"] == "nvidia/AutoGaze"
    assert plan["models"]["autogaze"]["local_dir"] == "/content/autogaze_weights/nvidia__AutoGaze"
    assert plan["models"]["vjepa"]["repo_id"] == "facebook/vjepa2-vitl-fpc64-256"
    assert plan["models"]["vjepa"]["local_dir"] == "/content/autogaze_weights/facebook__vjepa2-vitl-fpc64-256"
    assert plan["models"]["qwen"]["repo_id"] == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert plan["models"]["qwen"]["local_dir"] == "/content/autogaze_weights/Qwen__Qwen2.5-VL-3B-Instruct"


def test_download_checkpoints_calls_snapshot_download(monkeypatch, tmp_path):
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return kwargs["local_dir"]

    monkeypatch.setattr("scripts.download_vjepa_qwen_checkpoints.snapshot_download", fake_snapshot_download)

    result = download_checkpoints(
        autogaze_model="nvidia/AutoGaze",
        vjepa_model="facebook/vjepa2-vitl-fpc64-256",
        qwen_model="Qwen/Qwen2.5-VL-3B-Instruct",
        output_root=tmp_path,
        revision="main",
        max_workers=2,
    )

    assert [call["repo_id"] for call in calls] == [
        "nvidia/AutoGaze",
        "facebook/vjepa2-vitl-fpc64-256",
        "Qwen/Qwen2.5-VL-3B-Instruct",
    ]
    assert all(call["repo_type"] == "model" for call in calls)
    assert result["models"]["autogaze"]["snapshot_path"].endswith("nvidia__AutoGaze")
    assert result["models"]["vjepa"]["snapshot_path"].endswith("facebook__vjepa2-vitl-fpc64-256")
    assert result["models"]["qwen"]["snapshot_path"].endswith("Qwen__Qwen2.5-VL-3B-Instruct")
