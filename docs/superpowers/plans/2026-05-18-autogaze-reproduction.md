# AutoGaze Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a clean, reproducible AutoGaze benchmark harness that validates MPS execution locally, prepares CUDA/NVILA execution, and includes an HLVid benchmark path aligned with the paper-facing setup.

**Architecture:** Keep official repositories and downloaded data outside tracked source while storing reproducible harness code, tests, and runbooks in this repo. The harness has shared metadata/timing helpers, an AutoGaze/SigLIP benchmark runner, an NVILA single-sample runner, and an HLVid manifest/scoring/full-run pipeline. MPS runs are correctness and plumbing checks; CUDA runs are the source for leader-facing performance and HLVid quality claims.

**Tech Stack:** Python 3.11, PyTorch, Transformers, Hugging Face Datasets/Hub, PyAV, pandas/CSV/JSONL, pytest, official `NVlabs/AutoGaze`, optional official `NVlabs/VILA`, `nvidia/AutoGaze`, `nvidia/NVILA-8B-HD-Video`, `bfshi/HLVid`.

---

## File Structure

- Create `.gitignore` to exclude local environments, official clones, model/data caches, and generated benchmark outputs.
- Create `requirements-repro.txt` for local harness dependencies that are not supplied by the official AutoGaze editable install.
- Create `scripts/bootstrap_official_repos.sh` to clone or update official `NVlabs/AutoGaze` and optionally `NVlabs/VILA`.
- Create `repro/__init__.py` as the local harness package marker.
- Create `repro/common.py` for device resolution, timing synchronization, JSON/JSONL/CSV writing, git revision capture, and environment metadata.
- Create `repro/autogaze_bench.py` for AutoGaze-only and SigLIP baseline-vs-gazed benchmarking.
- Create `repro/hlvid.py` for HLVid manifest generation, answer parsing, scoring, prediction IO, and summaries.
- Create `repro/nvila_runner.py` for NVILA-HD-Video quickstart, HLVid dry-run, and full CUDA execution.
- Create `repro/report.py` for benchmark summary tables from JSON/JSONL outputs.
- Create `tests/test_common.py`, `tests/test_hlvid.py`, and `tests/test_report.py` for unit coverage that does not require model downloads.
- Create `docs/AUTOGAZE_REPRO_RUNBOOK.md` with MPS and CUDA commands, expected outputs, and HLVid operating notes.

Generated directories:

- `external/AutoGaze/`: official AutoGaze clone, not tracked.
- `external/VILA/`: official VILA clone, not tracked unless NVILA runner needs repo entrypoints.
- `data/hlvid/`: HLVid manifests and video cache pointers. Full video files remain untracked.
- `outputs/autogaze_repro/`: benchmark outputs, prediction JSONL files, score summaries, and logs.

## Task 1: Repo Hygiene And Official Source Bootstrap

**Files:**
- Create: `.gitignore`
- Create: `requirements-repro.txt`
- Create: `scripts/bootstrap_official_repos.sh`
- Test: shell syntax via `bash -n scripts/bootstrap_official_repos.sh`

- [ ] **Step 1: Add `.gitignore` for generated and downloaded artifacts**

```gitignore
.venv/
__pycache__/
*.pyc
.pytest_cache/
.mypy_cache/
.ruff_cache/

external/
data/hlvid/videos/
data/hlvid/cache/
outputs/autogaze_repro/

.DS_Store
```

- [ ] **Step 2: Add `requirements-repro.txt`**

```txt
accelerate
av
datasets
huggingface_hub
pandas
pyarrow
pytest
tqdm
transformers
```

- [ ] **Step 3: Add `scripts/bootstrap_official_repos.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERNAL_DIR="${ROOT_DIR}/external"
AUTOGAZE_DIR="${EXTERNAL_DIR}/AutoGaze"
VILA_DIR="${EXTERNAL_DIR}/VILA"

mkdir -p "${EXTERNAL_DIR}" "${ROOT_DIR}/outputs/autogaze_repro"

clone_or_update() {
  local repo_url="$1"
  local target_dir="$2"
  if [[ -d "${target_dir}/.git" ]]; then
    git -C "${target_dir}" fetch --prune origin
    git -C "${target_dir}" switch main
    git -C "${target_dir}" pull --ff-only origin main
  else
    git clone "${repo_url}" "${target_dir}"
  fi
}

clone_or_update "https://github.com/NVlabs/AutoGaze.git" "${AUTOGAZE_DIR}"

if [[ "${1:-}" == "--with-vila" ]]; then
  clone_or_update "https://github.com/NVlabs/VILA.git" "${VILA_DIR}"
fi

python - <<'PY'
import json
import subprocess
from pathlib import Path

root = Path.cwd()
sources = {}
for name in ["AutoGaze", "VILA"]:
    repo = root / "external" / name
    if (repo / ".git").exists():
        sources[name] = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
        ).strip()

out = root / "outputs" / "autogaze_repro" / "source_revisions.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(sources, indent=2) + "\n")
print(json.dumps(sources, indent=2))
PY
```

- [ ] **Step 4: Verify shell syntax**

Run:

```bash
bash -n scripts/bootstrap_official_repos.sh
```

Expected: exit code 0 with no output.

- [ ] **Step 5: Commit bootstrap files**

```bash
git add .gitignore requirements-repro.txt scripts/bootstrap_official_repos.sh
git commit -m "chore: add AutoGaze source bootstrap"
```

## Task 2: Core Device, Timing, And Output Helpers

**Files:**
- Create: `repro/__init__.py`
- Create: `repro/common.py`
- Create: `tests/test_common.py`

- [ ] **Step 1: Write failing tests for common helpers**

Create `tests/test_common.py`:

```python
import json
from pathlib import Path

from repro.common import (
    BenchmarkTimer,
    append_jsonl,
    compute_stats,
    resolve_device,
    write_json,
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


def test_benchmark_timer_records_elapsed_ms():
    timer = BenchmarkTimer("cpu")
    with timer.measure():
        sum(range(100))
    assert len(timer.elapsed_ms) == 1
    assert timer.elapsed_ms[0] >= 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_common.py -q
```

Expected: FAIL because `repro.common` does not exist.

- [ ] **Step 3: Implement `repro/__init__.py`**

```python
"""Local AutoGaze reproduction harness."""
```

- [ ] **Step 4: Implement `repro/common.py`**

```python
from __future__ import annotations

import csv
import json
import platform
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import torch


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    return torch.device(name)


def synchronize(device: str | torch.device) -> None:
    device_type = torch.device(device).type
    if device_type == "cuda":
        torch.cuda.synchronize()
    elif device_type == "mps":
        torch.mps.synchronize()


class BenchmarkTimer:
    def __init__(self, device: str | torch.device) -> None:
        self.device = torch.device(device)
        self.elapsed_ms: list[float] = []

    @contextmanager
    def measure(self):
        synchronize(self.device)
        start = time.perf_counter()
        try:
            yield
        finally:
            synchronize(self.device)
            self.elapsed_ms.append((time.perf_counter() - start) * 1000.0)


def compute_stats(values: Iterable[float]) -> dict[str, float | int]:
    data = list(values)
    if not data:
        return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(data),
        "mean": statistics.fmean(data),
        "median": statistics.median(data),
        "min": min(data),
        "max": max(data),
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        target.write_text("")
        return
    with target.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def git_revision(path: str | Path) -> str | None:
    repo = Path(path)
    if not (repo / ".git").exists():
        return None
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def environment_metadata(device: torch.device, external_root: str | Path = "external") -> dict[str, Any]:
    external = Path(external_root)
    metadata: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": device.type,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "autogaze_revision": git_revision(external / "AutoGaze"),
        "vila_revision": git_revision(external / "VILA"),
    }
    if device.type == "cuda":
        metadata["cuda_device_name"] = torch.cuda.get_device_name(device)
    return metadata
```

- [ ] **Step 5: Run tests to verify they pass**

Run:

```bash
pytest tests/test_common.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit common helpers**

```bash
git add repro/__init__.py repro/common.py tests/test_common.py
git commit -m "test: add benchmark helper coverage"
```

## Task 3: HLVid Parsing, Scoring, And Manifest Helpers

**Files:**
- Create: `repro/hlvid.py`
- Create: `tests/test_hlvid.py`

- [ ] **Step 1: Write failing tests for answer parsing and scoring**

Create `tests/test_hlvid.py`:

```python
from repro.hlvid import (
    REQUIRED_COLUMNS,
    parse_choice,
    score_predictions,
    validate_manifest_rows,
)


def test_parse_choice_accepts_direct_letters_and_prefixed_text():
    assert parse_choice("A") == "A"
    assert parse_choice("Answer: c.") == "C"
    assert parse_choice("The correct answer is D because the sign says Duke.") == "D"


def test_parse_choice_returns_none_for_ambiguous_output():
    assert parse_choice("A or B") is None
    assert parse_choice("No idea") is None


def test_validate_manifest_rows_requires_official_columns():
    row = {
        "question_id": 1,
        "category": "av",
        "video_path": "clip_av_video_5_001.mp4",
        "question": "Question? A. One B. Two C. Three D. Four",
        "answer": "A",
    }
    validate_manifest_rows([row])
    assert set(REQUIRED_COLUMNS).issubset(row)


def test_score_predictions_tracks_parse_failures_separately():
    rows = [
        {"answer": "A", "raw_output": "A"},
        {"answer": "B", "raw_output": "Answer: C"},
        {"answer": "D", "raw_output": "A or D"},
    ]
    summary, scored = score_predictions(rows)
    assert summary["total"] == 3
    assert summary["scored"] == 2
    assert summary["correct"] == 1
    assert summary["parse_failed"] == 1
    assert summary["accuracy_scored"] == 0.5
    assert scored[2]["parse_status"] == "failed"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_hlvid.py -q
```

Expected: FAIL because `repro.hlvid` does not exist.

- [ ] **Step 3: Implement `repro/hlvid.py` parsing and scoring**

```python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from datasets import load_dataset

from repro.common import append_jsonl, write_json

REQUIRED_COLUMNS = ("question_id", "category", "video_path", "question", "answer")
CHOICE_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)


def parse_choice(text: str | None) -> str | None:
    if text is None:
        return None
    matches = [m.group(1).upper() for m in CHOICE_RE.finditer(text)]
    unique = sorted(set(matches))
    if len(unique) == 1:
        return unique[0]
    return None


def validate_manifest_rows(rows: list[dict[str, Any]]) -> None:
    missing_by_row = []
    for idx, row in enumerate(rows):
        missing = [name for name in REQUIRED_COLUMNS if name not in row]
        if missing:
            missing_by_row.append((idx, missing))
    if missing_by_row:
        raise ValueError(f"HLVid manifest rows missing required columns: {missing_by_row[:3]}")


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in REQUIRED_COLUMNS}


def load_hlvid_manifest(split: str = "test", limit: int | None = None) -> list[dict[str, Any]]:
    dataset = load_dataset("bfshi/HLVid", split=split)
    rows = [normalize_row(dict(row)) for row in dataset]
    if limit is not None:
        rows = rows[:limit]
    validate_manifest_rows(rows)
    return rows


def score_predictions(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scored_rows: list[dict[str, Any]] = []
    correct = 0
    scored = 0
    parse_failed = 0

    for row in rows:
        parsed = parse_choice(row.get("raw_output"))
        expected = parse_choice(str(row.get("answer", "")))
        out = dict(row)
        out["parsed_answer"] = parsed
        out["expected_answer"] = expected
        if parsed is None or expected is None:
            parse_failed += 1
            out["correct"] = False
            out["parse_status"] = "failed"
        else:
            scored += 1
            out["correct"] = parsed == expected
            out["parse_status"] = "parsed"
            correct += int(out["correct"])
        scored_rows.append(out)

    summary = {
        "total": len(rows),
        "scored": scored,
        "correct": correct,
        "parse_failed": parse_failed,
        "accuracy_scored": correct / scored if scored else 0.0,
        "accuracy_total": correct / len(rows) if rows else 0.0,
    }
    return summary, scored_rows


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    return [json.loads(line) for line in source.read_text().splitlines() if line.strip()]


def build_manifest(args: argparse.Namespace) -> None:
    rows = load_hlvid_manifest(split=args.split, limit=args.limit)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(rows)} HLVid rows to {output}")


def score_file(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.predictions)
    summary, scored_rows = score_predictions(rows)
    write_json(args.summary, summary)
    append_jsonl(args.scored, scored_rows)
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="HLVid manifest and scoring helpers")
    sub = parser.add_subparsers(dest="command", required=True)

    manifest = sub.add_parser("manifest")
    manifest.add_argument("--split", default="test")
    manifest.add_argument("--limit", type=int)
    manifest.add_argument("--output", default="data/hlvid/manifest_test.json")
    manifest.set_defaults(func=build_manifest)

    score = sub.add_parser("score")
    score.add_argument("--predictions", required=True)
    score.add_argument("--summary", default="outputs/autogaze_repro/hlvid_score_summary.json")
    score.add_argument("--scored", default="outputs/autogaze_repro/hlvid_scored_predictions.jsonl")
    score.set_defaults(func=score_file)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
pytest tests/test_hlvid.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit HLVid helpers**

```bash
git add repro/hlvid.py tests/test_hlvid.py
git commit -m "test: add HLVid scoring helpers"
```

## Task 4: AutoGaze And SigLIP Benchmark Runner

**Files:**
- Create: `repro/autogaze_bench.py`
- Modify: `requirements-repro.txt` if an imported package is missing during execution.
- Test: local CLI help, then model smoke after official repo bootstrap.

- [ ] **Step 1: Add benchmark runner with isolated imports**

Create `repro/autogaze_bench.py`:

```python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import av
import torch
from transformers import AutoImageProcessor

from repro.common import BenchmarkTimer, compute_stats, environment_metadata, resolve_device, write_csv, write_json


def add_external_autogaze(path: str) -> None:
    repo = Path(path).resolve()
    if not (repo / "autogaze").exists():
        raise FileNotFoundError(f"AutoGaze repo not found at {repo}")
    sys.path.insert(0, str(repo))


def load_video_frames(video_path: str, frame_count: int) -> Any:
    from autogaze.datasets.video_utils import read_video_pyav

    container = av.open(video_path)
    try:
        return read_video_pyav(container=container, indices=list(range(frame_count)))
    finally:
        container.close()


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def summarize_gaze(gaze_outputs: dict[str, Any], raw_patch_budget: int) -> dict[str, Any]:
    padded = gaze_outputs["if_padded_gazing"]
    selected = int((~padded).sum().item())
    padded_count = int(padded.sum().item())
    total_gaze_slots = int(padded.numel())
    return {
        "raw_patch_budget": raw_patch_budget,
        "selected_non_padded_patches": selected,
        "padded_gazing_positions": padded_count,
        "total_gaze_slots": total_gaze_slots,
        "token_reduction_ratio": raw_patch_budget / selected if selected else 0.0,
        "num_gazing_each_frame": [int(x) for x in gaze_outputs["num_gazing_each_frame"]],
    }


def run(args: argparse.Namespace) -> None:
    add_external_autogaze(args.autogaze_repo)

    from autogaze.datasets.video_utils import transform_video_for_pytorch
    from autogaze.models.autogaze import AutoGaze, AutoGazeImageProcessor
    from autogaze.vision_encoders.siglip import SiglipVisionModel

    device = resolve_device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    autogaze_transform = AutoGazeImageProcessor.from_pretrained(args.autogaze_model)
    autogaze_model = AutoGaze.from_pretrained(args.autogaze_model).to(device)
    autogaze_model.eval()

    frame_count = args.frames or int(autogaze_model.config.max_num_frames)
    raw_video = load_video_frames(args.video, frame_count)
    video_input_autogaze = transform_video_for_pytorch(raw_video, autogaze_transform)[None].to(device=device, dtype=dtype)

    siglip_transform = AutoImageProcessor.from_pretrained(args.siglip_model)
    siglip_model = SiglipVisionModel.from_pretrained(
        args.siglip_model,
        scales=autogaze_model.config.scales,
        attn_implementation=args.attn_implementation,
    ).to(device=device, dtype=dtype)
    siglip_model.eval()
    video_input_siglip = transform_video_for_pytorch(raw_video, siglip_transform)[None].to(device=device, dtype=dtype)

    raw_patch_budget = int(frame_count * autogaze_model.num_vision_tokens_each_frame)

    with torch.inference_mode():
        for _ in range(args.warmup):
            gaze_outputs = autogaze_model(
                {"video": video_input_autogaze},
                gazing_ratio=args.gazing_ratio,
                task_loss_requirement=args.task_loss_requirement,
            )
            _ = siglip_model(video_input_siglip)
            _ = siglip_model(video_input_siglip, gazing_info=gaze_outputs)

        autogaze_timer = BenchmarkTimer(device)
        siglip_full_timer = BenchmarkTimer(device)
        siglip_gazed_timer = BenchmarkTimer(device)
        final_gaze_outputs = None
        final_siglip_full = None
        final_siglip_gazed = None

        for _ in range(args.repeat):
            with autogaze_timer.measure():
                final_gaze_outputs = autogaze_model(
                    {"video": video_input_autogaze},
                    gazing_ratio=args.gazing_ratio,
                    task_loss_requirement=args.task_loss_requirement,
                )
            with siglip_full_timer.measure():
                final_siglip_full = siglip_model(video_input_siglip)
            with siglip_gazed_timer.measure():
                final_siglip_gazed = siglip_model(video_input_siglip, gazing_info=final_gaze_outputs)

    assert final_gaze_outputs is not None
    assert final_siglip_full is not None
    assert final_siglip_gazed is not None

    gaze_summary = summarize_gaze(final_gaze_outputs, raw_patch_budget)
    result = {
        "metadata": environment_metadata(device),
        "input": {
            "video": args.video,
            "frames": frame_count,
            "gazing_ratio": args.gazing_ratio,
            "task_loss_requirement": args.task_loss_requirement,
            "dtype": args.dtype,
        },
        "models": {
            "autogaze": args.autogaze_model,
            "siglip": args.siglip_model,
        },
        "gaze": gaze_summary,
        "latency_ms": {
            "autogaze": compute_stats(autogaze_timer.elapsed_ms),
            "siglip_full": compute_stats(siglip_full_timer.elapsed_ms),
            "siglip_gazed": compute_stats(siglip_gazed_timer.elapsed_ms),
        },
        "shapes": {
            "video_input_autogaze": list(video_input_autogaze.shape),
            "video_input_siglip": list(video_input_siglip.shape),
            "siglip_full_hidden": list(final_siglip_full.last_hidden_state.shape),
            "siglip_gazed_hidden": list(final_siglip_gazed.last_hidden_state.shape),
        },
        "input_tensor_bytes": {
            "autogaze": tensor_bytes(video_input_autogaze),
            "siglip": tensor_bytes(video_input_siglip),
        },
    }

    out_json = Path(args.output_json)
    out_csv = Path(args.output_csv)
    write_json(out_json, result)
    write_csv(
        out_csv,
        [
            {
                "stage": stage,
                **stats,
                "token_reduction_ratio": gaze_summary["token_reduction_ratio"],
                "selected_non_padded_patches": gaze_summary["selected_non_padded_patches"],
                "raw_patch_budget": gaze_summary["raw_patch_budget"],
            }
            for stage, stats in result["latency_ms"].items()
        ],
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark AutoGaze and AutoGaze-compatible SigLIP")
    parser.add_argument("--autogaze-repo", default="external/AutoGaze")
    parser.add_argument("--autogaze-model", default="nvidia/AutoGaze")
    parser.add_argument("--siglip-model", default="google/siglip2-base-patch16-224")
    parser.add_argument("--video", default="external/AutoGaze/assets/example_input.mp4")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    parser.add_argument("--frames", type=int)
    parser.add_argument("--gazing-ratio", type=float, default=0.75)
    parser.add_argument("--task-loss-requirement", type=float, default=0.7)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--output-json", default="outputs/autogaze_repro/autogaze_siglip_bench.json")
    parser.add_argument("--output-csv", default="outputs/autogaze_repro/autogaze_siglip_bench.csv")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify CLI parsing before model downloads**

Run:

```bash
python -m repro.autogaze_bench --help
```

Expected: usage text that includes `--gazing-ratio`, `--task-loss-requirement`, `--device`, and `--output-json`.

- [ ] **Step 3: Bootstrap official AutoGaze source**

Run:

```bash
bash scripts/bootstrap_official_repos.sh
```

Expected: `external/AutoGaze` exists and `outputs/autogaze_repro/source_revisions.json` contains an `AutoGaze` commit hash.

- [ ] **Step 4: Install local and official dependencies in the active environment**

Run:

```bash
python -m pip install -r requirements-repro.txt
python -m pip install -e external/AutoGaze
```

Expected: packages install successfully. If network sandboxing blocks downloads, rerun with approved escalation.

- [ ] **Step 5: Run MPS smoke benchmark**

Run:

```bash
python -m repro.autogaze_bench \
  --device mps \
  --dtype float32 \
  --warmup 1 \
  --repeat 3 \
  --output-json outputs/autogaze_repro/mps_autogaze_siglip_bench.json \
  --output-csv outputs/autogaze_repro/mps_autogaze_siglip_bench.csv
```

Expected: JSON output includes positive `token_reduction_ratio`, positive latency means for `autogaze`, `siglip_full`, and `siglip_gazed`, and SigLIP gazed hidden state length equal to AutoGaze total gaze slots.

- [ ] **Step 6: Commit benchmark runner**

```bash
git add repro/autogaze_bench.py requirements-repro.txt
git commit -m "feat: add AutoGaze SigLIP benchmark runner"
```

## Task 5: NVILA-HD-Video Single-Sample Runner

**Files:**
- Create: `repro/nvila_runner.py`
- Test: CLI help locally, optional MPS import/config check, CUDA single-sample run on GPU machine.

- [ ] **Step 1: Add NVILA runner with quickstart defaults**

Create `repro/nvila_runner.py`:

```python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_url
from transformers import AutoModel, AutoProcessor

from repro.common import append_jsonl, environment_metadata, resolve_device, synchronize, write_json
from repro.hlvid import load_hlvid_manifest, parse_choice, score_predictions

DEFAULT_MODEL = "nvidia/NVILA-8B-HD-Video"
DEFAULT_EXAMPLE_VIDEO = "https://huggingface.co/datasets/bfshi/HLVid/resolve/main/example/clip_av_video_5_001.mp4"
DEFAULT_PROMPT = (
    "Question: What does the white text on the green road sign say?\n"
    "A. Hampden St\n"
    "B. Hampden Ave\n"
    "C. HampdenBlvd\n"
    "D. Hampden Rd\n"
    "Please answer directly with the letter of the correct answer."
)


def processor_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "num_video_frames": args.num_video_frames,
        "num_video_frames_thumbnail": args.num_video_frames_thumbnail,
        "max_tiles_video": args.max_tiles_video,
        "gazing_ratio_tile": [0.2] + [0.06] * 15,
        "gazing_ratio_thumbnail": 1,
        "task_loss_requirement_tile": args.task_loss_requirement_tile,
        "task_loss_requirement_thumbnail": None,
        "max_batch_size_autogaze": args.max_batch_size_autogaze,
        "trust_remote_code": True,
    }


def load_model_and_processor(args: argparse.Namespace):
    processor = AutoProcessor.from_pretrained(args.model_path, **processor_kwargs(args))
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        device_map=args.device_map,
        max_batch_size_siglip=args.max_batch_size_siglip,
    )
    model.eval()
    return model, processor


def input_device(model, fallback: torch.device) -> torch.device:
    model_device = getattr(model, "device", None)
    if model_device is not None:
        return torch.device(model_device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return fallback


def resolve_video(video: str, args: argparse.Namespace) -> str:
    if video.startswith("http://") or video.startswith("https://"):
        return video
    local = Path(video)
    if local.exists():
        return str(local)
    cached = Path(args.hlvid_video_root) / video
    if cached.exists():
        return str(cached)
    return hf_hub_url(repo_id=args.hlvid_repo, filename=video, repo_type="dataset")


def tensor_shapes(payload: dict[str, Any]) -> dict[str, list[int]]:
    return {key: list(value.shape) for key, value in payload.items() if isinstance(value, torch.Tensor)}


def extract_gaze_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = {
        "autogaze_selected_patches": None,
        "autogaze_padded_patches": None,
        "autogaze_total_gaze_slots": None,
        "autogaze_token_reduction_ratio": None,
        "available_input_keys": sorted(payload.keys()),
    }
    candidates = []
    for value in payload.values():
        if isinstance(value, dict):
            candidates.append(value)
    for candidate in candidates:
        if "if_padded_gazing" in candidate:
            padded = candidate["if_padded_gazing"]
            if isinstance(padded, torch.Tensor):
                selected = int((~padded.bool()).sum().item())
                padded_count = int(padded.bool().sum().item())
                total = int(padded.numel())
                metrics.update({
                    "autogaze_selected_patches": selected,
                    "autogaze_padded_patches": padded_count,
                    "autogaze_total_gaze_slots": total,
                    "autogaze_token_reduction_ratio": None,
                })
                return metrics
    return metrics


def move_tensors(payload: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in payload.items()}


def timed_generate(model, inputs: dict[str, Any], processor, device: torch.device, max_new_tokens: int) -> dict[str, Any]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    synchronize(device)
    generate_ms = (time.perf_counter() - start) * 1000.0
    generated = outputs[:, inputs["input_ids"].shape[1]:]
    response = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    peak_memory_bytes = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    return {
        "raw_output": response,
        "parsed_answer": parse_choice(response),
        "generate_ms": generate_ms,
        "generated_tokens": int(generated.shape[1]),
        "peak_memory_bytes": peak_memory_bytes,
    }


def generate_one(
    model,
    processor,
    video: str,
    prompt: str,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    video_token = processor.tokenizer.video_token
    resolved_video = resolve_video(video, args)

    start = time.perf_counter()
    inputs = processor(text=f"{video_token}\n\n{prompt}", videos=resolved_video, return_tensors="pt")
    preprocess_ms = (time.perf_counter() - start) * 1000.0

    target_device = input_device(model, device)
    inputs = move_tensors(dict(inputs), target_device)

    ttft_ms = None
    if args.measure_ttft:
        one_token = timed_generate(model, inputs, processor, device, max_new_tokens=1)
        ttft_ms = one_token["generate_ms"]

    synchronize(device)
    result = timed_generate(model, inputs, processor, device, max_new_tokens=args.max_new_tokens)
    decode_estimated_ms = max(result["generate_ms"] - ttft_ms, 0.0) if ttft_ms is not None else None
    gaze_metrics = extract_gaze_metrics(inputs)

    return {
        **result,
        **gaze_metrics,
        "video_input": video,
        "video_resolved": resolved_video,
        "input_token_count": int(inputs["input_ids"].shape[1]),
        "input_shapes": tensor_shapes(inputs),
        "video_preprocess_ms": preprocess_ms,
        "ttft_ms": ttft_ms,
        "decode_estimated_ms": decode_estimated_ms,
        "total_ms": preprocess_ms + result["generate_ms"],
        "vision_encoder_ms": None,
        "mllm_prefill_ms": ttft_ms,
        "metric_note": "Remote NVILA code may not expose separate vision encoder and AutoGaze internals; null fields mean unavailable from public generate outputs.",
    }


def run_single(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    model, processor = load_model_and_processor(args)
    result = generate_one(model, processor, args.video, args.prompt, device, args)
    payload = {
        "metadata": environment_metadata(device),
        "model_path": args.model_path,
        "video": args.video,
        "prompt": args.prompt,
        "result": result,
    }
    write_json(args.output_json, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def run_hlvid(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    rows = load_hlvid_manifest(split=args.split, limit=args.limit)
    model, processor = load_model_and_processor(args)
    output_path = Path(args.predictions)
    completed_ids = set()
    if output_path.exists() and args.resume:
        for line in output_path.read_text().splitlines():
            if line.strip():
                completed_ids.add(json.loads(line)["question_id"])

    predictions = []
    for row in rows:
        if row["question_id"] in completed_ids:
            continue
        result = generate_one(model, processor, row["video_path"], row["question"], device, args)
        prediction = {
            **row,
            **result,
            "model_path": args.model_path,
            "num_video_frames": args.num_video_frames,
            "num_video_frames_thumbnail": args.num_video_frames_thumbnail,
            "max_tiles_video": args.max_tiles_video,
            "task_loss_requirement_tile": args.task_loss_requirement_tile,
        }
        append_jsonl(output_path, [prediction])
        predictions.append(prediction)

    all_rows = []
    if output_path.exists():
        all_rows = [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()]
    summary, scored = score_predictions(all_rows)
    write_json(args.summary, summary)
    append_jsonl(args.scored_predictions, scored)
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NVILA-HD-Video quickstart and HLVid benchmark")
    parser.add_argument("--mode", choices=["single", "hlvid"], default="single")
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--video", default=DEFAULT_EXAMPLE_VIDEO)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--num-video-frames", type=int, default=128)
    parser.add_argument("--num-video-frames-thumbnail", type=int, default=64)
    parser.add_argument("--max-tiles-video", type=int, default=48)
    parser.add_argument("--task-loss-requirement-tile", type=float, default=0.6)
    parser.add_argument("--max-batch-size-autogaze", type=int, default=16)
    parser.add_argument("--max-batch-size-siglip", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--measure-ttft", action="store_true")
    parser.add_argument("--hlvid-repo", default="bfshi/HLVid")
    parser.add_argument("--hlvid-video-root", default="data/hlvid/videos")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-json", default="outputs/autogaze_repro/nvila_single.json")
    parser.add_argument("--predictions", default="outputs/autogaze_repro/hlvid_predictions.jsonl")
    parser.add_argument("--summary", default="outputs/autogaze_repro/hlvid_summary.json")
    parser.add_argument("--scored-predictions", default="outputs/autogaze_repro/hlvid_scored_predictions.jsonl")
    args = parser.parse_args()
    if args.mode == "single":
        run_single(args)
    else:
        run_hlvid(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify CLI help locally**

Run:

```bash
python -m repro.nvila_runner --help
```

Expected: usage text includes `--mode`, `--num-video-frames`, `--max-tiles-video`, `--task-loss-requirement-tile`, and `--measure-ttft`.

- [ ] **Step 3: Run CUDA single-sample quickstart on GPU machine**

Run on CUDA hardware:

```bash
python -m repro.nvila_runner \
  --mode single \
  --device cuda \
  --num-video-frames 128 \
  --num-video-frames-thumbnail 64 \
  --max-tiles-video 48 \
  --measure-ttft \
  --output-json outputs/autogaze_repro/cuda_nvila_single_128f.json
```

Expected: JSON output contains a raw model answer, parsed answer if the answer is a single letter, preprocessing latency, generation latency, total latency, input token count, generated token count, peak CUDA memory if available, and null-marked internal metrics when the public remote code does not expose them.

- [ ] **Step 4: Commit NVILA runner**

```bash
git add repro/nvila_runner.py
git commit -m "feat: add NVILA HD video runner"
```

## Task 6: HLVid Manifest, Dry Run, And Full CUDA Commands

**Files:**
- Modify: `repro/hlvid.py`
- Modify: `repro/nvila_runner.py` if HLVid video path resolution needs a local cache prefix.
- Test: `pytest tests/test_hlvid.py -q`, manifest generation with `--limit 5`, CUDA dry run with `--limit 1`.

- [ ] **Step 1: Generate a small HLVid manifest locally**

Run:

```bash
python -m repro.hlvid manifest \
  --split test \
  --limit 5 \
  --output data/hlvid/manifest_test_5.json
```

Expected: `data/hlvid/manifest_test_5.json` exists with five rows and each row has `question_id`, `category`, `video_path`, `question`, and `answer`.

- [ ] **Step 2: Run HLVid unit tests again**

Run:

```bash
pytest tests/test_hlvid.py -q
```

Expected: PASS.

- [ ] **Step 3: Run HLVid CUDA dry run**

Run on CUDA hardware:

```bash
python -m repro.nvila_runner \
  --mode hlvid \
  --device cuda \
  --limit 1 \
  --num-video-frames 128 \
  --num-video-frames-thumbnail 64 \
  --max-tiles-video 48 \
  --measure-ttft \
  --resume \
  --predictions outputs/autogaze_repro/hlvid_dry_run_predictions.jsonl \
  --summary outputs/autogaze_repro/hlvid_dry_run_summary.json \
  --scored-predictions outputs/autogaze_repro/hlvid_dry_run_scored.jsonl
```

Expected: prediction JSONL has one row, summary JSON reports `total` 1, and failures are explicit if the sample video cannot be downloaded or decoded.

- [ ] **Step 4: Run full HLVid CUDA benchmark with paper-facing scale**

Run on the target CUDA machine after storage is confirmed:

```bash
python -m repro.nvila_runner \
  --mode hlvid \
  --device cuda \
  --limit 268 \
  --num-video-frames 1024 \
  --num-video-frames-thumbnail 64 \
  --max-tiles-video 48 \
  --measure-ttft \
  --resume \
  --predictions outputs/autogaze_repro/hlvid_full_predictions.jsonl \
  --summary outputs/autogaze_repro/hlvid_full_summary.json \
  --scored-predictions outputs/autogaze_repro/hlvid_full_scored.jsonl
```

Expected: run is resumable, summary reports `accuracy_scored`, prediction rows include total latency, preprocessing latency, generation latency, optional TTFT, memory, generated tokens, and available gaze metrics, and report notes compare the result to HLVid 52.6 for NVILA-8B-HD-Video at 1024 frames and max resolution 3584.

- [ ] **Step 5: Commit HLVid execution integration**

```bash
git add repro/hlvid.py repro/nvila_runner.py
git commit -m "feat: add HLVid benchmark execution path"
```

## Task 7: Result Summaries And Leader-Facing Report Tables

**Files:**
- Create: `repro/report.py`
- Create: `tests/test_report.py`

- [ ] **Step 1: Write failing tests for report summaries**

Create `tests/test_report.py`:

```python
import json
from pathlib import Path

from repro.report import summarize_autogaze_bench, summarize_hlvid


def test_summarize_autogaze_bench_extracts_key_fields(tmp_path: Path):
    path = tmp_path / "bench.json"
    path.write_text(json.dumps({
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
    }))
    row = summarize_autogaze_bench(path)
    assert row["token_reduction_ratio"] == 4.0
    assert row["siglip_speedup_excluding_autogaze"] == 40.0 / 12.0
    assert row["siglip_speedup_including_autogaze"] == 40.0 / 17.0


def test_summarize_hlvid_extracts_accuracy(tmp_path: Path):
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({
        "total": 268,
        "scored": 260,
        "correct": 137,
        "parse_failed": 8,
        "accuracy_scored": 0.5269,
    }))
    row = summarize_hlvid(path)
    assert row["total"] == 268
    assert row["accuracy_scored"] == 0.5269
    assert row["paper_target_hlvid"] == 0.526
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_report.py -q
```

Expected: FAIL because `repro.report` does not exist.

- [ ] **Step 3: Implement `repro/report.py`**

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from repro.common import write_csv, write_json


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def summarize_autogaze_bench(path: str | Path) -> dict[str, Any]:
    payload = load_json(path)
    autogaze_ms = float(payload["latency_ms"]["autogaze"]["mean"])
    siglip_full_ms = float(payload["latency_ms"]["siglip_full"]["mean"])
    siglip_gazed_ms = float(payload["latency_ms"]["siglip_gazed"]["mean"])
    return {
        "source": str(path),
        "token_reduction_ratio": payload["gaze"]["token_reduction_ratio"],
        "selected_non_padded_patches": payload["gaze"]["selected_non_padded_patches"],
        "raw_patch_budget": payload["gaze"]["raw_patch_budget"],
        "autogaze_ms_mean": autogaze_ms,
        "siglip_full_ms_mean": siglip_full_ms,
        "siglip_gazed_ms_mean": siglip_gazed_ms,
        "siglip_speedup_excluding_autogaze": safe_div(siglip_full_ms, siglip_gazed_ms),
        "siglip_speedup_including_autogaze": safe_div(siglip_full_ms, siglip_gazed_ms + autogaze_ms),
    }


def summarize_hlvid(path: str | Path) -> dict[str, Any]:
    payload = load_json(path)
    return {
        "source": str(path),
        "total": payload["total"],
        "scored": payload["scored"],
        "correct": payload["correct"],
        "parse_failed": payload["parse_failed"],
        "accuracy_scored": payload["accuracy_scored"],
        "paper_target_hlvid": 0.526,
        "paper_delta": payload["accuracy_scored"] - 0.526,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize AutoGaze reproduction outputs")
    parser.add_argument("--autogaze-json", action="append", default=[])
    parser.add_argument("--hlvid-summary", action="append", default=[])
    parser.add_argument("--output-json", default="outputs/autogaze_repro/report_summary.json")
    parser.add_argument("--output-csv", default="outputs/autogaze_repro/report_summary.csv")
    args = parser.parse_args()

    rows = []
    for path in args.autogaze_json:
        rows.append({"kind": "autogaze_siglip", **summarize_autogaze_bench(path)})
    for path in args.hlvid_summary:
        rows.append({"kind": "hlvid", **summarize_hlvid(path)})
    write_json(args.output_json, {"rows": rows})
    write_csv(args.output_csv, rows)
    print(json.dumps({"rows": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
pytest tests/test_report.py -q
```

Expected: PASS.

- [ ] **Step 5: Generate a summary from existing outputs**

Run after MPS benchmark:

```bash
python -m repro.report \
  --autogaze-json outputs/autogaze_repro/mps_autogaze_siglip_bench.json \
  --output-json outputs/autogaze_repro/mps_report_summary.json \
  --output-csv outputs/autogaze_repro/mps_report_summary.csv
```

Expected: JSON and CSV summarize token reduction and SigLIP speedups with and without AutoGaze overhead.

- [ ] **Step 6: Commit report helpers**

```bash
git add repro/report.py tests/test_report.py
git commit -m "feat: add benchmark report summaries"
```

## Task 8: Runbook And CUDA Handoff Documentation

**Files:**
- Create: `docs/AUTOGAZE_REPRO_RUNBOOK.md`

- [ ] **Step 1: Write the runbook**

Create `docs/AUTOGAZE_REPRO_RUNBOOK.md`:

```markdown
# AutoGaze Reproduction Runbook

## Official Sources

- AutoGaze code: https://github.com/NVlabs/AutoGaze
- AutoGaze project page: https://autogaze.github.io/
- AutoGaze paper: https://arxiv.org/abs/2603.12254
- AutoGaze collection: https://huggingface.co/collections/bfshi/autogaze
- HLVid dataset: https://huggingface.co/datasets/bfshi/HLVid
- NVILA-HD-Video README path: https://github.com/NVlabs/VILA/tree/main/vila_hd/nvila_hd_video

## Local MPS Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-repro.txt
bash scripts/bootstrap_official_repos.sh
python -m pip install -e external/AutoGaze
```

## MPS AutoGaze And SigLIP Smoke Benchmark

```bash
python -m repro.autogaze_bench \
  --device mps \
  --dtype float32 \
  --warmup 1 \
  --repeat 3 \
  --output-json outputs/autogaze_repro/mps_autogaze_siglip_bench.json \
  --output-csv outputs/autogaze_repro/mps_autogaze_siglip_bench.csv
```

Use this result to confirm that the official AutoGaze model loads, emits gazing metadata, and drives the customized SigLIP path on Apple MPS.

## CUDA Single-Sample NVILA Check

```bash
python -m repro.nvila_runner \
  --mode single \
  --device cuda \
  --num-video-frames 128 \
  --num-video-frames-thumbnail 64 \
  --max-tiles-video 48 \
  --measure-ttft \
  --output-json outputs/autogaze_repro/cuda_nvila_single_128f.json
```

This mirrors the official NVILA-HD-Video quickstart scale and validates the model and processor path before the full HLVid run.

## HLVid Manifest

```bash
python -m repro.hlvid manifest \
  --split test \
  --output data/hlvid/manifest_test.json
```

The Hugging Face dataset card currently exposes the `test` split with 268 rows and about 152 GB of files. Generate the manifest before downloading all videos so sample ids and expected answers are pinned.

## HLVid Dry Run

```bash
python -m repro.nvila_runner \
  --mode hlvid \
  --device cuda \
  --limit 1 \
  --num-video-frames 128 \
  --num-video-frames-thumbnail 64 \
  --max-tiles-video 48 \
  --measure-ttft \
  --resume \
  --predictions outputs/autogaze_repro/hlvid_dry_run_predictions.jsonl \
  --summary outputs/autogaze_repro/hlvid_dry_run_summary.json \
  --scored-predictions outputs/autogaze_repro/hlvid_dry_run_scored.jsonl
```

## HLVid Paper-Facing Run

```bash
python -m repro.nvila_runner \
  --mode hlvid \
  --device cuda \
  --limit 268 \
  --num-video-frames 1024 \
  --num-video-frames-thumbnail 64 \
  --max-tiles-video 48 \
  --measure-ttft \
  --resume \
  --predictions outputs/autogaze_repro/hlvid_full_predictions.jsonl \
  --summary outputs/autogaze_repro/hlvid_full_summary.json \
  --scored-predictions outputs/autogaze_repro/hlvid_full_scored.jsonl
```

Compare `accuracy_scored` with the project-page HLVid target of 52.6 for NVILA-8B-HD-Video. Report skipped, failed, and parse-failed samples separately.
```

- [ ] **Step 2: Commit runbook**

```bash
git add docs/AUTOGAZE_REPRO_RUNBOOK.md
git commit -m "docs: add AutoGaze reproduction runbook"
```

## Task 9: Full Local Verification

**Files:**
- Modify files from prior tasks only if verification exposes a concrete defect.

- [ ] **Step 1: Run unit tests**

Run:

```bash
pytest tests -q
```

Expected: PASS for tests that do not require model downloads.

- [ ] **Step 2: Run CLI help checks**

Run:

```bash
python -m repro.autogaze_bench --help
python -m repro.hlvid --help
python -m repro.nvila_runner --help
python -m repro.report --help
```

Expected: each command prints usage text and exits 0.

- [ ] **Step 3: Run MPS benchmark after dependencies and official repo are installed**

Run:

```bash
python -m repro.autogaze_bench \
  --device mps \
  --dtype float32 \
  --warmup 1 \
  --repeat 3 \
  --output-json outputs/autogaze_repro/mps_autogaze_siglip_bench.json \
  --output-csv outputs/autogaze_repro/mps_autogaze_siglip_bench.csv
```

Expected: output JSON and CSV exist and include token reduction, AutoGaze latency, full SigLIP latency, and gazed SigLIP latency.

- [ ] **Step 4: Generate local report summary**

Run:

```bash
python -m repro.report \
  --autogaze-json outputs/autogaze_repro/mps_autogaze_siglip_bench.json \
  --output-json outputs/autogaze_repro/mps_report_summary.json \
  --output-csv outputs/autogaze_repro/mps_report_summary.csv
```

Expected: report summary includes `siglip_speedup_excluding_autogaze` and `siglip_speedup_including_autogaze`.

- [ ] **Step 5: Commit verification fixes if any were needed**

If no fixes were needed, do not create an empty commit. If fixes were needed:

```bash
git add repro tests docs scripts requirements-repro.txt .gitignore
git commit -m "fix: stabilize AutoGaze reproduction harness"
```

## Task 10: CUDA And HLVid Handoff Package

**Files:**
- Modify: `docs/AUTOGAZE_REPRO_RUNBOOK.md`
- Create only if needed: `docs/CUDA_RESULTS_TEMPLATE.md`

- [ ] **Step 1: Add CUDA result template if the first CUDA run is not available in this workspace**

Create `docs/CUDA_RESULTS_TEMPLATE.md`:

```markdown
# CUDA AutoGaze And HLVid Results Template

## Hardware

- GPU:
- GPU memory:
- Driver:
- CUDA:
- PyTorch:
- Transformers:
- AutoGaze commit:
- VILA commit:
- Model revisions:

## AutoGaze/SigLIP Efficiency

- Input video:
- Frames:
- Resolution:
- Raw patch budget:
- Selected non-padded patches:
- Token reduction ratio:
- AutoGaze mean latency:
- SigLIP full mean latency:
- SigLIP gazed mean latency:
- SigLIP speedup excluding AutoGaze:
- SigLIP speedup including AutoGaze:

## NVILA Single-Sample

- Frames:
- Max tiles:
- Total latency:
- Generated tokens:
- Parsed answer:
- Raw output:

## HLVid

- Split:
- Samples attempted:
- Samples scored:
- Correct:
- Parse failed:
- Runtime failed:
- Accuracy scored:
- Paper target:
- Delta from paper target:

## Notes

- Deviations from paper-facing setup:
- Failure categories:
- Next measurement:
```

- [ ] **Step 2: Commit handoff template**

```bash
git add docs/CUDA_RESULTS_TEMPLATE.md docs/AUTOGAZE_REPRO_RUNBOOK.md
git commit -m "docs: add CUDA HLVid results template"
```

## Self-Review Checklist

- Spec coverage: MPS AutoGaze, SigLIP token/latency comparison, CUDA-ready NVILA, HLVid manifest/dry/full paths, per-sample scoring, and leader-facing report outputs are each covered by tasks.
- Placeholder scan: The plan contains no unfinished marker text or unspecified file paths.
- Type consistency: `raw_output`, `parsed_answer`, `answer`, `question_id`, `latency_ms`, and summary field names are consistent across tests, runner code, scoring, and reporting.
- Risk handling: Network/model/data downloads are isolated to explicit commands, generated content is ignored, and HLVid full video scale is separated from local dry-run validation.
