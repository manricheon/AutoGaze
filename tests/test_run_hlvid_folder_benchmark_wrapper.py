import json
from pathlib import Path

import scripts.run_hlvid_folder_benchmark as wrapper


def write_minimal_hlvid_dataset(root: Path) -> None:
    videos = root / "videos"
    videos.mkdir(parents=True)
    (videos / "clip.mp4").write_bytes(b"fake-video")
    (root / "manifest_test.json").write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "category": "av",
                    "video_path": "clip.mp4",
                    "question": "What happens? A. one B. two C. three D. four",
                    "answer": "A",
                }
            ]
        )
    )


def test_qwen_plugin_suite_routes_through_unified_hlvid_wrapper(monkeypatch, tmp_path):
    dataset = tmp_path / "hlvid"
    write_minimal_hlvid_dataset(dataset)
    captured = {}

    def fake_run_plugin_hlvid_benchmark(**kwargs):
        captured.update(kwargs)
        return {"summary": {"ok": True}}

    monkeypatch.setattr(wrapper, "run_plugin_hlvid_benchmark", fake_run_plugin_hlvid_benchmark)

    wrapper.main(
        [
            "--dataset-dir",
            str(dataset),
            "--output-dir",
            str(tmp_path / "out"),
            "--plugin-suite",
            "qwen",
            "--plugin-model",
            "qwen3-vl=/models/qwen",
            "--autogaze-model",
            "/models/autogaze",
            "--video-decode-strategy",
            "seek",
            "--autogaze-target-patch-size",
            "14",
            "--max-batch-size-autogaze",
            "4",
            "--gazing-ratio-tile",
            "0.2",
            "--limit",
            "3",
            "--num-video-frames",
            "16",
            "--max-tiles-video",
            "4",
            "--video-resize-longest-edge",
            "224",
            "--continue-on-error",
        ]
    )

    assert captured["manifest"] == str(dataset / "manifest_test.json")
    assert captured["video_root"] == str(dataset / "videos")
    assert captured["output_dir"] == str(tmp_path / "out")
    assert captured["modes"] == [
        "qwen_full_vit",
        "qwen_chunked_vit",
        "qwen_chunked_vit_autogaze_sparse",
    ]
    assert captured["models"] == {"qwen3-vl": "/models/qwen"}
    assert captured["autogaze_model"] == "/models/autogaze"
    assert captured["video_decode_strategy"] == "seek"
    assert captured["autogaze_target_patch_size"] == 14
    assert captured["max_batch_size_autogaze"] == 4
    assert captured["gazing_ratio"] == 0.2
    assert captured["limit"] == 3
    assert captured["num_video_frames"] == 16
    assert captured["qwen_video_nframes"] == 16
    assert captured["max_tiles_video"] == 4
    assert captured["qwen_vit_max_spatial_chunks"] == 4
    assert captured["video_resize_longest_edge"] == 224


def test_unified_hlvid_wrapper_uses_explicit_plugin_modes(monkeypatch, tmp_path):
    dataset = tmp_path / "hlvid"
    write_minimal_hlvid_dataset(dataset)
    captured = {}

    def fake_run_plugin_hlvid_benchmark(**kwargs):
        captured.update(kwargs)
        return {"summary": {"ok": True}}

    monkeypatch.setattr(wrapper, "run_plugin_hlvid_benchmark", fake_run_plugin_hlvid_benchmark)

    wrapper.main(
        [
            "--dataset-dir",
            str(dataset),
            "--plugin-suite",
            "custom",
            "--plugin-modes",
            "qwen_full_vit,qwen_chunked_vit",
            "--qwen-video-nframes",
            "8",
            "--qwen-vit-max-spatial-chunks",
            "2",
        ]
    )

    assert captured["modes"] == ["qwen_full_vit", "qwen_chunked_vit"]
    assert captured["output_dir"] == "outputs/autogaze_repro/plugin_hlvid_custom"
    assert captured["qwen_video_nframes"] == 8
    assert captured["qwen_vit_max_spatial_chunks"] == 2


def test_qwen_plugin_suite_defaults_autogaze_sparse_to_four_scales(monkeypatch, tmp_path):
    dataset = tmp_path / "hlvid"
    write_minimal_hlvid_dataset(dataset)
    captured = {}

    def fake_run_plugin_hlvid_benchmark(**kwargs):
        captured.update(kwargs)
        return {"summary": {"ok": True}}

    monkeypatch.setattr(wrapper, "run_plugin_hlvid_benchmark", fake_run_plugin_hlvid_benchmark)

    wrapper.main(
        [
            "--dataset-dir",
            str(dataset),
            "--plugin-suite",
            "qwen",
            "--num-video-frames",
            "16",
            "--video-resize-longest-edge",
            "224",
        ]
    )

    assert captured["modes"] == [
        "qwen_full_vit",
        "qwen_chunked_vit",
        "qwen_chunked_vit_autogaze_sparse",
    ]
    assert captured["autogaze_target_scales"] == "64+128+192+224"
    assert captured["autogaze_target_patch_size"] == 16
    assert captured["autogaze_tile_size"] == 224


def test_qwen_plugin_suite_scales_follow_resize_longest_edge(monkeypatch, tmp_path):
    dataset = tmp_path / "hlvid"
    write_minimal_hlvid_dataset(dataset)
    captured = {}

    def fake_run_plugin_hlvid_benchmark(**kwargs):
        captured.update(kwargs)
        return {"summary": {"ok": True}}

    monkeypatch.setattr(wrapper, "run_plugin_hlvid_benchmark", fake_run_plugin_hlvid_benchmark)

    wrapper.main(
        [
            "--dataset-dir",
            str(dataset),
            "--plugin-suite",
            "qwen",
            "--video-resize-longest-edge",
            "448",
        ]
    )

    assert captured["autogaze_target_scales"] == "112+224+336+448"
    assert captured["autogaze_target_patch_size"] == 16
    assert captured["autogaze_tile_size"] == 448


def test_unified_hlvid_wrapper_keeps_nvila_default_path(monkeypatch):
    called = {}

    def fake_nvila_main():
        called["nvila"] = True

    monkeypatch.setattr(wrapper, "run_nvila_hlvid_main", fake_nvila_main)

    wrapper.main(["--dataset-dir", "/data/hlvid", "--limit", "1"])

    assert called == {"nvila": True}
