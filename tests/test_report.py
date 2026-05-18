import json
from pathlib import Path

from repro.report import summarize_autogaze_bench, summarize_hlvid


def test_summarize_autogaze_bench_extracts_key_fields(tmp_path: Path):
    path = tmp_path / "bench.json"
    path.write_text(
        json.dumps(
            {
                "gaze": {
                    "token_reduction_ratio": 4.0,
                    "selected_non_padded_patches": 100,
                    "raw_patch_budget": 400,
                },
                "latency_ms": {
                    "autogaze": {"mean": 5.0},
                    "siglip_full": {"mean": 40.0},
                    "siglip_gazed": {"mean": 12.0},
                },
            }
        )
    )
    row = summarize_autogaze_bench(path)
    assert row["token_reduction_ratio"] == 4.0
    assert row["siglip_speedup_excluding_autogaze"] == 40.0 / 12.0
    assert row["siglip_speedup_including_autogaze"] == 40.0 / 17.0


def test_summarize_hlvid_extracts_accuracy(tmp_path: Path):
    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps(
            {
                "total": 268,
                "scored": 260,
                "correct": 137,
                "parse_failed": 8,
                "accuracy_scored": 0.5269,
            }
        )
    )
    row = summarize_hlvid(path)
    assert row["total"] == 268
    assert row["accuracy_scored"] == 0.5269
    assert row["paper_target_hlvid"] == 0.526
