from pathlib import Path

from scripts.download_qwen_model import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REPO_ID,
    build_download_plan,
    download_qwen_model,
    parse_args,
)


def test_repro_requirements_include_qwen_video_input_helper():
    requirements = Path("requirements-repro.txt").read_text()

    assert "qwen-vl-utils" in requirements


def test_qwen_model_download_defaults_match_flexible_runner_local_path():
    args = parse_args([])

    assert args.repo_id == "Qwen/Qwen3-VL-8B-Instruct"
    assert args.output_dir == "weight/Qwen3-VL-8B-Instruct"
    assert DEFAULT_REPO_ID == args.repo_id
    assert DEFAULT_OUTPUT_DIR == args.output_dir


def test_qwen_model_download_plan_reports_exact_repo_and_local_dir():
    plan = build_download_plan(
        repo_id="Qwen/Qwen3-VL-8B-Instruct",
        output_dir=Path("weight/Qwen3-VL-8B-Instruct"),
        revision="main",
        allow_patterns=["*.json"],
        ignore_patterns=["*.md"],
        max_workers=4,
    )

    assert plan == {
        "repo_id": "Qwen/Qwen3-VL-8B-Instruct",
        "revision": "main",
        "output_dir": "weight/Qwen3-VL-8B-Instruct",
        "repo_type": "model",
        "allow_patterns": ["*.json"],
        "ignore_patterns": ["*.md"],
        "max_workers": 4,
    }


def test_download_qwen_model_calls_snapshot_download_with_local_dir(monkeypatch, tmp_path):
    calls = {}

    def fake_snapshot_download(**kwargs):
        calls.update(kwargs)
        return str(tmp_path / "Qwen3-VL-8B-Instruct")

    monkeypatch.setattr("scripts.download_qwen_model.snapshot_download", fake_snapshot_download)

    result = download_qwen_model(
        repo_id="Qwen/Qwen3-VL-8B-Instruct",
        output_dir=tmp_path / "Qwen3-VL-8B-Instruct",
        revision="main",
        allow_patterns=["*.json"],
        ignore_patterns=None,
        max_workers=2,
    )

    assert calls["repo_id"] == "Qwen/Qwen3-VL-8B-Instruct"
    assert calls["repo_type"] == "model"
    assert calls["revision"] == "main"
    assert calls["local_dir"] == str(tmp_path / "Qwen3-VL-8B-Instruct")
    assert calls["allow_patterns"] == ["*.json"]
    assert calls["ignore_patterns"] is None
    assert calls["max_workers"] == 2
    assert result["snapshot_path"] == str(tmp_path / "Qwen3-VL-8B-Instruct")
