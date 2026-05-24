from pathlib import Path

from scripts.prepare_video_task_assets import (
    DATASET_PRESETS,
    MODEL_PRESETS,
    build_asset_plan,
    download_asset_plan,
    parse_args,
    parse_asset_spec,
)


def test_parse_asset_spec_uses_name_as_local_subdir_and_optional_revision():
    asset = parse_asset_spec("caption_set=org/video-caption@v1", repo_type="dataset")

    assert asset == {
        "name": "caption_set",
        "repo_id": "org/video-caption",
        "repo_type": "dataset",
        "revision": "v1",
        "local_subdir": "caption_set",
    }


def test_default_model_preset_prepares_qwen_and_autogaze_weights():
    args = parse_args([])
    plan = build_asset_plan(args)
    model_names = [row["name"] for row in plan["models"]]

    assert args.model_preset == "qwen-video-task"
    assert model_names == ["qwen3-vl-8b", "autogaze"]
    assert plan["models"][0]["repo_id"] == "Qwen/Qwen3-VL-8B-Instruct"
    assert plan["models"][0]["local_dir"] == "weight/Qwen3-VL-8B-Instruct"
    assert plan["datasets"] == []


def test_custom_dataset_and_expand_model_plan_use_separate_roots():
    args = parse_args(
        [
            "--local-root",
            "/data/video_tasks",
            "--weight-root",
            "/models/weight",
            "--dataset",
            "my_caption=org/caption-dataset",
            "--model-preset",
            "qwen-compare",
        ]
    )
    plan = build_asset_plan(args)

    assert plan["datasets"] == [
        {
            "name": "my_caption",
            "repo_id": "org/caption-dataset",
            "repo_type": "dataset",
            "revision": "main",
            "local_dir": "/data/video_tasks/my_caption",
            "allow_patterns": None,
            "ignore_patterns": None,
            "max_workers": 8,
        }
    ]
    assert [row["name"] for row in plan["models"]] == ["qwen2.5-vl-7b", "qwen3-vl-8b", "autogaze"]
    assert plan["models"][0]["local_dir"] == "/models/weight/Qwen2.5-VL-7B-Instruct"


def test_caption_action_dataset_preset_uses_selected_hf_video_datasets():
    args = parse_args(
        [
            "--local-root",
            "/data/video_tasks",
            "--dataset-preset",
            "caption-action-smoke",
            "--model-preset",
            "none",
        ]
    )
    plan = build_asset_plan(args)

    assert DATASET_PRESETS["caption-action-smoke"][0]["repo_id"] == "VLM2Vec/MSR-VTT"
    assert DATASET_PRESETS["caption-action-smoke"][1]["repo_id"] == "bitmind/UCF101-Videos"
    assert [row["name"] for row in plan["datasets"]] == ["msrvtt-caption", "ucf101-action"]
    assert plan["datasets"][0]["local_dir"] == "/data/video_tasks/msrvtt"
    assert plan["datasets"][1]["local_dir"] == "/data/video_tasks/ucf101-videos"


def test_download_asset_plan_calls_snapshot_download_with_repo_types(monkeypatch, tmp_path):
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(Path(kwargs["local_dir"]) / ".snapshot")

    monkeypatch.setattr("scripts.prepare_video_task_assets.snapshot_download", fake_snapshot_download)
    args = parse_args(
        [
            "--local-root",
            str(tmp_path / "inputs"),
            "--weight-root",
            str(tmp_path / "weight"),
            "--dataset",
            "caption=org/caption",
            "--model-preset",
            "none",
            "--model",
            "qwen_local=Qwen/Qwen3-VL-8B-Instruct",
            "--include",
            "*.json",
            "--exclude",
            "*.md",
            "--max-workers",
            "2",
        ]
    )
    plan = build_asset_plan(args)
    result = download_asset_plan(plan)

    assert [call["repo_type"] for call in calls] == ["dataset", "model"]
    assert calls[0]["repo_id"] == "org/caption"
    assert calls[0]["local_dir"] == str(tmp_path / "inputs" / "caption")
    assert calls[1]["repo_id"] == "Qwen/Qwen3-VL-8B-Instruct"
    assert calls[1]["local_dir"] == str(tmp_path / "weight" / "qwen_local")
    assert calls[1]["allow_patterns"] == ["*.json"]
    assert calls[1]["ignore_patterns"] == ["*.md"]
    assert calls[1]["max_workers"] == 2
    assert result["downloaded"][0]["snapshot_path"].endswith(".snapshot")


def test_expand_smoke_model_preset_lists_required_families():
    preset_names = [asset["name"] for asset in MODEL_PRESETS["expand-smoke"]]

    assert "qwen2.5-vl-7b" in preset_names
    assert "qwen3-vl-8b" in preset_names
    assert "autogaze" in preset_names
    assert "nvila-video" in preset_names
    assert "llava-onevision" in preset_names
    assert "internvl3-8b" in preset_names
