import json
from pathlib import Path

from repro.common import (
    BenchmarkTimer,
    append_jsonl,
    compute_stats,
    resolve_device,
    write_json,
    write_jsonl,
)


def test_resolve_device_cpu_is_always_available():
    assert resolve_device("cpu").type == "cpu"


def test_compute_stats_reports_mean_and_median():
    stats = compute_stats([3.0, 1.0, 2.0])
    assert stats["count"] == 3
    assert stats["mean"] == 2.0
    assert stats["median"] == 2.0
    assert stats["min"] == 1.0
    assert stats["max"] == 3.0


def test_json_writers_create_parent_dirs(tmp_path: Path):
    json_path = tmp_path / "nested" / "result.json"
    jsonl_path = tmp_path / "nested" / "rows.jsonl"

    write_json(json_path, {"ok": True})
    append_jsonl(jsonl_path, [{"idx": 1}, {"idx": 2}])

    assert json.loads(json_path.read_text()) == {"ok": True}
    assert [json.loads(line) for line in jsonl_path.read_text().splitlines()] == [
        {"idx": 1},
        {"idx": 2},
    ]


def test_write_jsonl_overwrites_existing_rows(tmp_path: Path):
    jsonl_path = tmp_path / "rows.jsonl"
    append_jsonl(jsonl_path, [{"idx": 1}])

    write_jsonl(jsonl_path, [{"idx": 2}])

    assert [json.loads(line) for line in jsonl_path.read_text().splitlines()] == [{"idx": 2}]


def test_benchmark_timer_records_elapsed_ms():
    timer = BenchmarkTimer("cpu")
    with timer.measure():
        sum(range(100))
    assert len(timer.elapsed_ms) == 1
    assert timer.elapsed_ms[0] >= 0
