import json
from pathlib import Path

from repro.video_task_benchmark import build_markdown_report, run_video_task_benchmark


def test_run_caption_benchmark_writes_predictions_summary_and_markdown(monkeypatch, tmp_path):
    manifest = tmp_path / "caption.jsonl"
    manifest.write_text(json.dumps({"sample_id": "c1", "video_path": "clip.mp4", "caption": "a person cooks"}) + "\n")
    video_root = tmp_path / "videos"
    video_root.mkdir()
    (video_root / "clip.mp4").write_text("fake")

    def fake_run_single(args):
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        return {
            "implementation_status": "executed",
            "generation": {
                "text": "A person cooks in a kitchen.",
                "status": "executed",
                "metrics": {
                    "latency_ms": {"total": 12.0, "generate": 8.0},
                    "tokens": {"visual_tokens_before_prune": 100, "visual_tokens_after_prune": 40},
                    "memory_bytes": {"peak_cuda_reserved": 2048},
                    "metric_status": {"value": "executed"},
                },
            },
        }

    monkeypatch.setattr("repro.video_task_benchmark.run_single", fake_run_single)

    result = run_video_task_benchmark(
        manifest=manifest,
        video_root=video_root,
        output_dir=tmp_path / "out",
        task_type="captioning",
        modes=["qwen3_full_vit"],
        limit=1,
    )

    prediction = result["predictions"][0]
    assert prediction["task_type"] == "captioning"
    assert prediction["prompt"] == "Describe the video."
    assert prediction["references"] == ["a person cooks"]
    assert prediction["raw_output"] == "A person cooks in a kitchen."
    assert result["summary"]["modes"]["qwen3_full_vit"]["scoring_status"] == "not_scored"
    assert Path(result["artifacts"]["predictions"]).is_file()
    assert Path(result["artifacts"]["scored"]).is_file()
    assert Path(result["artifacts"]["summary"]).is_file()
    assert Path(result["artifacts"]["markdown"]).is_file()


def test_run_action_benchmark_scores_exact_match_and_records_failure(monkeypatch, tmp_path):
    manifest = tmp_path / "action.json"
    manifest.write_text(
        json.dumps(
            [
                {"sample_id": "a1", "video_path": "a.mp4", "prompt": "Action? A. run B. cook", "label": "A"},
                {"sample_id": "a2", "video_path": "b.mp4", "prompt": "Action? A. run B. cook", "label": "B"},
            ]
        )
    )
    video_root = tmp_path / "videos"
    video_root.mkdir()
    (video_root / "a.mp4").write_text("fake")
    (video_root / "b.mp4").write_text("fake")
    calls = []

    def fake_run_single(args):
        calls.append(args)
        if "b.mp4" in args.video:
            return {
                "implementation_status": "oom",
                "failure": {"kind": "oom", "stage": "mllm_generate"},
                "generation": {
                    "text": None,
                    "status": "oom",
                    "metrics": {"failure": {"kind": "oom", "stage": "mllm_generate"}},
                },
            }
        return {
            "implementation_status": "executed",
            "generation": {
                "text": "A",
                "status": "executed",
                "metrics": {"latency_ms": {"total": 5.0}, "memory_bytes": {"peak_cuda_reserved": 1024}},
            },
        }

    monkeypatch.setattr("repro.video_task_benchmark.run_single", fake_run_single)

    result = run_video_task_benchmark(
        manifest=manifest,
        video_root=video_root,
        output_dir=tmp_path / "out",
        task_type="action_classification",
        modes=["qwen3_full_vit"],
        limit=2,
    )

    summary = result["summary"]["modes"]["qwen3_full_vit"]
    assert len(calls) == 2
    assert calls[0].prompt == "Action? A. run B. cook"
    assert summary["total"] == 2
    assert summary["correct"] == 1
    assert summary["failed"] == 1
    assert summary["accuracy_total"] == 0.5
    assert result["predictions"][1]["failure_stage"] == "mllm_generate"


def test_run_videoqa_benchmark_scores_multiple_choice(monkeypatch, tmp_path):
    manifest = tmp_path / "videoqa.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "vq1",
                "video_path": "clip.mp4",
                "question": "What happens? A. one B. two",
                "answer": "A",
                "choices": ["one", "two"],
            }
        )
        + "\n"
    )
    video_root = tmp_path / "videos"
    video_root.mkdir()
    (video_root / "clip.mp4").write_text("fake")

    def fake_run_single(args):
        return {
            "implementation_status": "executed",
            "generation": {
                "text": "A",
                "status": "executed",
                "metrics": {
                    "latency_ms": {"total": 10.0, "generate": 4.0},
                    "tokens": {"visual_tokens_before_prune": 100, "visual_tokens_after_prune": 50},
                    "memory_bytes": {"peak_cuda_reserved": 1234},
                },
            },
        }

    monkeypatch.setattr("repro.video_task_benchmark.run_single", fake_run_single)

    result = run_video_task_benchmark(
        manifest=manifest,
        video_root=video_root,
        output_dir=tmp_path / "out",
        task_type="videoqa",
        modes=["qwen3_full_vit"],
        limit=1,
    )

    summary = result["summary"]["modes"]["qwen3_full_vit"]
    assert result["predictions"][0]["task_type"] == "videoqa"
    assert result["predictions"][0]["question"] == "What happens? A. one B. two"
    assert summary["task_type"] == "videoqa"
    assert summary["correct"] == 1
    assert summary["accuracy_total"] == 1.0
    assert Path(result["artifacts"]["markdown"]).name == "videoqa_report.md"


def test_build_markdown_report_lists_modes_and_task_type():
    markdown = build_markdown_report(
        {
            "task_type": "action_classification",
            "modes": {
                "qwen3_full_vit": {
                    "total": 2,
                    "correct": 1,
                    "failed": 0,
                    "parse_failed": 0,
                    "accuracy_total": 0.5,
                    "accuracy_scored": 0.5,
                    "status_counts": {"executed": 2},
                }
            },
        }
    )

    assert "# Video Task Benchmark" in markdown
    assert "action_classification" in markdown
    assert "qwen3_full_vit" in markdown
