import sys

import pytest

import autogaze.eval.run_benchmark as run_benchmark


def _run_main_and_capture(monkeypatch, argv):
    captured = {}

    def fake_evaluate(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(run_benchmark, "evaluate", fake_evaluate)
    monkeypatch.setattr(sys, "argv", ["run_benchmark.py", *argv])

    run_benchmark.main()
    return captured


def test_parser_exposes_current_branch_runner_keys():
    parser = run_benchmark._build_parser()
    help_text = parser.format_help()

    assert "siglip_qwen25" in help_text
    assert "vjepa2_nvila" in help_text
    assert "vjepa2_qwen25" in help_text
    assert "qwen25vl_full" in help_text


def test_vjepa2_qwen25_cli_uses_vjepa2_path_as_model_path(monkeypatch):
    captured = _run_main_and_capture(
        monkeypatch,
        [
            "--task",
            "videomme",
            "--mllm",
            "vjepa2_qwen25",
            "--integration",
            "full",
            "--model-path",
            "weights/NVILA-8B-HD-Video",
            "--vjepa2-path",
            "weights/vjepa2-vitl-fpc64-256",
            "--lm-path",
            "weights/Qwen2.5-7B-Instruct",
            "--autogaze-path",
            "weights/AutoGaze",
            "--max-samples",
            "5",
        ],
    )

    assert captured["mllm"] == "vjepa2_qwen25"
    assert captured["model_path"] == "weights/vjepa2-vitl-fpc64-256"
    assert captured["autogaze_path"] == "weights/AutoGaze"
    assert captured["max_samples"] == 5
    assert captured["runner_kwargs"] == {
        "integration": "full",
        "lm_path": "weights/Qwen2.5-7B-Instruct",
    }


def test_vjepa2_nvila_cli_keeps_nvila_model_path_and_forwards_vjepa2_path(monkeypatch):
    captured = _run_main_and_capture(
        monkeypatch,
        [
            "--task",
            "videomme",
            "--mllm",
            "vjepa2_nvila",
            "--model-path",
            "weights/NVILA-8B-HD-Video",
            "--vjepa2-path",
            "weights/vjepa2-vitl-fpc64-256",
            "--autogaze-path",
            "weights/AutoGaze",
        ],
    )

    assert captured["mllm"] == "vjepa2_nvila"
    assert captured["model_path"] == "weights/NVILA-8B-HD-Video"
    assert captured["runner_kwargs"] == {"vjepa2_path": "weights/vjepa2-vitl-fpc64-256"}


def test_nvila_native_no_autogaze_keeps_autogaze_path_and_forces_ratio(monkeypatch):
    captured = _run_main_and_capture(
        monkeypatch,
        [
            "--task",
            "videomme",
            "--mllm",
            "nvila",
            "--model-path",
            "weights/NVILA-8B-HD-Video",
            "--autogaze-path",
            "weights/AutoGaze",
            "--no-autogaze",
        ],
    )

    assert captured["mllm"] == "nvila"
    assert captured["autogaze_path"] == "weights/AutoGaze"
    assert captured["gazing_ratio"] == 1.0


def test_non_native_no_autogaze_sets_autogaze_path_none(monkeypatch):
    captured = _run_main_and_capture(
        monkeypatch,
        [
            "--task",
            "videomme",
            "--mllm",
            "siglip_qwen25",
            "--model-path",
            "weights/Qwen2.5-VL-7B-Instruct",
            "--no-autogaze",
        ],
    )

    assert captured["mllm"] == "siglip_qwen25"
    assert captured["autogaze_path"] is None


def test_parser_accepts_actionatlas_task():
    parser = run_benchmark._build_parser()

    args = parser.parse_args(["--task", "actionatlas"])

    assert args.task == "actionatlas"


def test_invalid_runner_exits_before_evaluation(monkeypatch):
    called = False

    def fake_evaluate(**kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(run_benchmark, "evaluate", fake_evaluate)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_benchmark.py", "--task", "videomme", "--mllm", "not_a_runner"],
    )

    with pytest.raises(SystemExit):
        run_benchmark.main()

    assert called is False
