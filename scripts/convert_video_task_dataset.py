from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DATASET_PRESETS: dict[str, dict[str, str]] = {
    "msrvtt-caption": {
        "repo_id": "VLM2Vec/MSR-VTT",
        "task_type": "captioning",
        "source": "VLM2Vec/MSR-VTT",
    },
    "ucf101-action": {
        "repo_id": "bitmind/UCF101-Videos",
        "task_type": "action_classification",
        "source": "bitmind/UCF101-Videos",
    },
}

SUPPORTED_SUFFIXES = {".jsonl", ".json", ".csv", ".parquet"}


def convert_dataset_rows(rows: list[dict[str, Any]], *, dataset_preset: str) -> list[dict[str, Any]]:
    if dataset_preset == "msrvtt-caption":
        return _convert_msrvtt_caption(rows)
    if dataset_preset == "ucf101-action":
        return _convert_ucf101_action(rows)
    raise ValueError(f"unsupported dataset preset: {dataset_preset}")


def convert_dataset_to_manifest(
    *,
    input_path: str | Path,
    output_path: str | Path,
    dataset_preset: str,
    limit: int | None = None,
) -> dict[str, Any]:
    rows = read_source_rows(input_path)
    converted = convert_dataset_rows(rows, dataset_preset=dataset_preset)
    if limit is not None:
        converted = converted[:limit]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in converted:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    preset = DATASET_PRESETS[dataset_preset]
    return {
        "dataset_preset": dataset_preset,
        "source_repo": preset["repo_id"],
        "task_type": preset["task_type"],
        "input_path": str(input_path),
        "output_path": str(output),
        "rows_written": len(converted),
    }


def read_source_rows(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if source.is_dir():
        rows: list[dict[str, Any]] = []
        for child in sorted(item for item in source.rglob("*") if item.suffix.lower() in SUPPORTED_SUFFIXES):
            rows.extend(read_source_rows(child))
        return rows
    suffix = source.suffix.lower()
    if suffix == ".jsonl":
        return [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("rows", payload.get("data", []))
        if not isinstance(payload, list):
            raise ValueError(f"JSON source must be a row list or contain rows/data: {source}")
        return [dict(row) for row in payload]
    if suffix == ".csv":
        with source.open(newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if suffix == ".parquet":
        try:
            import pandas as pd
        except ModuleNotFoundError as exc:  # pragma: no cover - optional local dataset format
            raise RuntimeError("Reading parquet dataset metadata requires pandas/pyarrow.") from exc
        return pd.read_parquet(source).to_dict(orient="records")
    raise ValueError(f"unsupported input suffix: {source.suffix}")


def _convert_msrvtt_caption(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_video: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        video_path = _first_present(row, "video_path", "video", "video_name", "filename", "file_name")
        caption = _first_present(row, "caption", "text", "sentence", "query")
        if video_path is None or caption is None or not str(caption).strip():
            continue
        normalized_path = _normalize_video_path(video_path)
        key = normalized_path
        item = by_video.setdefault(
            key,
            {
                "sample_id": Path(normalized_path).stem or str(index),
                "video_path": normalized_path,
                "references": [],
                "source": DATASET_PRESETS["msrvtt-caption"]["source"],
            },
        )
        if row.get("category") is not None and "category" not in item:
            item["category"] = str(row["category"])
        caption_text = str(caption).strip()
        if caption_text not in item["references"]:
            item["references"].append(caption_text)
    return list(by_video.values())


def _convert_ucf101_action(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        video_path = _first_present(row, "clip_path", "video_path", "video", "path", "filename", "file_name")
        label = _first_present(row, "label", "action", "class", "class_name", "action_label")
        if video_path is None or label is None or str(label).strip() == "":
            continue
        converted.append(
            {
                "sample_id": str(_first_present(row, "clip_name", "video_id", "id") or Path(str(video_path)).stem or index),
                "video_path": _normalize_video_path(video_path),
                "label": str(label).strip(),
                "source": DATASET_PRESETS["ucf101-action"]["source"],
            }
        )
    return converted


def _first_present(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip() != "":
            return value
    return None


def _normalize_video_path(value: Any) -> str:
    return str(value).strip().lstrip("/")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert local HF video dataset metadata to video_task_benchmark JSONL manifests.")
    parser.add_argument("--input", required=True, help="Local HF snapshot metadata file or directory.")
    parser.add_argument("--output", required=True, help="Output JSONL manifest path.")
    parser.add_argument("--dataset-preset", choices=sorted(DATASET_PRESETS), default="msrvtt-caption")
    parser.add_argument("--limit", type=int)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    result = convert_dataset_to_manifest(
        input_path=args.input,
        output_path=args.output,
        dataset_preset=args.dataset_preset,
        limit=args.limit,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
