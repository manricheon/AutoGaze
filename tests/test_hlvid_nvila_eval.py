from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import evaluate_hlvid_nvila


def test_hlvid_prompt_and_choice_parsing() -> None:
    prompt = evaluate_hlvid_nvila.build_hlvid_prompt(
        "What does the sign say?",
        {"A": "Hampden St", "B": "Hampden Ave", "C": "HampdenBlvd", "D": "Hampden Rd"},
    )
    assert "Question: What does the sign say?" in prompt
    assert "B. Hampden Ave" in prompt
    assert prompt.endswith("Please answer directly with the letter of the correct answer.")

    assert evaluate_hlvid_nvila.extract_choice("The answer is B.") == "B"
    assert evaluate_hlvid_nvila.extract_choice("C. HampdenBlvd") == "C"
    assert evaluate_hlvid_nvila.normalize_choice("Hampden Rd", {"D": "Hampden Rd"}) == "D"


def test_hlvid_dry_run_with_local_json(tmp_path: Path) -> None:
    dataset = tmp_path / "hlvid_sample.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "question_id": "sample-1",
                    "video_path": "videos/sample.mp4",
                    "question": "What does the white text on the green road sign say?",
                    "options": {
                        "A": "Hampden St",
                        "B": "Hampden Ave",
                        "C": "HampdenBlvd",
                        "D": "Hampden Rd",
                    },
                    "answer": "B",
                }
            ]
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "outputs"
    args = evaluate_hlvid_nvila.parse_args(
        [
            "--config",
            str(ROOT / "configs" / "poc_inference" / "hlvid_nvila_hd_eval.yaml"),
            "--dataset-path",
            str(dataset),
            "--video-root",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ]
    )
    summary = evaluate_hlvid_nvila.run(args)
    assert summary["status"] == "dry_run"
    assert summary["processor_kwargs"]["num_video_frames"] == 128
    assert summary["processor_kwargs"]["num_video_frames_thumbnail"] == 64
    assert summary["processor_kwargs"]["max_tiles_video"] == 48
    assert summary["processor_kwargs"]["gazing_ratio_tile"] == [0.2] + [0.06] * 15

    predictions = json.loads((output_dir / "predictions" / "hlvid_predictions.json").read_text(encoding="utf-8"))
    assert predictions["predictions"][0]["status"] == "dry_run"
    assert predictions["predictions"][0]["target_choice"] == "B"
    assert "Please answer directly" in predictions["predictions"][0]["prompt"]


def test_hlvid_safe_infer_full_configs_load() -> None:
    resize_cfg = evaluate_hlvid_nvila.load_config(ROOT / "configs" / "poc_inference" / "hlvid_infer_full_resize_safe.yaml")
    chop_cfg = evaluate_hlvid_nvila.load_config(ROOT / "configs" / "poc_inference" / "hlvid_infer_full_resize_then_chop_safe.yaml")

    assert resize_cfg["video_input"]["read_mode"] == "streaming"
    assert resize_cfg["memory"]["max_processed_frames_per_window"] == 16
    assert resize_cfg["scaling"]["mode"] == "resize"
    assert resize_cfg["mllm"]["processor_from_pretrained_kwargs"]["num_video_frames"] == 16

    assert chop_cfg["video_input"]["read_mode"] == "streaming"
    assert chop_cfg["scaling"]["mode"] == "resize_then_chop"
    assert chop_cfg["scaling"]["max_chops"] == 4
    assert chop_cfg["memory"]["max_processed_frames_per_window"] == 64


def test_hlvid_eval_script_cli_dry_run(tmp_path: Path) -> None:
    dataset = tmp_path / "hlvid_sample.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "question_id": "sample-2",
                "video_path": "https://example.com/video.mp4",
                "question": "Which option is visible?",
                "A": "red",
                "B": "blue",
                "C": "green",
                "D": "yellow",
                "answer": "A. red",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "cli_outputs"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_hlvid_nvila.py",
            "--config",
            "configs/poc_inference/hlvid_nvila_hd_eval.yaml",
            "--dataset-path",
            str(dataset),
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "logs" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["num_predictions"] == 1
    assert metrics["accuracy"] is None
