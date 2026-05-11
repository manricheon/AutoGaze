from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from autogaze_ext.utils.hf_cache import HFLoadConfig, redacted_hf_config
from autogaze_ext.utils.hf_offline import hf_offline_mode


class LocalListDataset(list):
    def select(self, indices: Iterable[int]) -> "LocalListDataset":
        return LocalListDataset([self[i] for i in indices])


class HFDatasetLoader:
    def __init__(self, config: HFLoadConfig | dict[str, Any] | None = None) -> None:
        self.config = config if isinstance(config, HFLoadConfig) else HFLoadConfig.from_mapping(config or {})
        self.last_load_info: dict[str, Any] | None = None

    def load_dataset(
        self,
        dataset_id: str | None = None,
        config: str | None = None,
        split: str | None = None,
        streaming: bool | None = None,
        field_mapping: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> Any:
        resolved_id = dataset_id or self.config.dataset_id
        if not resolved_id:
            raise ValueError("dataset_id is required")
        path = Path(resolved_id)
        mapping = field_mapping or self.config.field_mapping
        if path.exists():
            dataset = self._load_local(path, field_mapping=mapping)
            dataset = self._limit(dataset)
            self.last_load_info = {**redacted_hf_config(self.config), "dataset_id": str(path), "source": "local_file"}
            return dataset
        return self._load_hub(resolved_id, config=config, split=split, streaming=streaming, **kwargs)

    def _load_hub(self, dataset_id: str, config: str | None = None, split: str | None = None, streaming: bool | None = None, **kwargs: Any) -> Any:
        try:
            import datasets  # type: ignore
        except ImportError as exc:
            raise ImportError("datasets is required for Hugging Face Hub datasets") from exc

        load_kwargs = {
            "path": dataset_id,
            "name": config or self.config.dataset_config,
            "split": split or self.config.dataset_split,
            "revision": self.config.revision,
            "cache_dir": self.config.cache_dir,
            "streaming": self.config.streaming if streaming is None else streaming,
            "num_proc": None if (self.config.streaming if streaming is None else streaming) else self.config.num_proc,
            **kwargs,
        }
        token = self.config.token
        if token:
            load_kwargs["token"] = token
        load_kwargs = {key: value for key, value in load_kwargs.items() if value is not None}
        with hf_offline_mode(self.config.offline):
            dataset = datasets.load_dataset(**load_kwargs)
        if self.config.max_samples is not None and hasattr(dataset, "select"):
            dataset = dataset.select(range(min(self.config.max_samples, len(dataset))))
        self.last_load_info = {**redacted_hf_config(self.config), "dataset_id": dataset_id, "source": "hub"}
        return dataset

    def _load_local(self, path: Path, field_mapping: dict[str, str] | None = None) -> LocalListDataset:
        suffix = path.suffix.lower()
        if suffix == ".json":
            rows = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(rows, dict):
                rows = rows.get("data", [rows])
        elif suffix == ".jsonl":
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        elif suffix == ".csv":
            with path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        else:
            raise ValueError(f"Unsupported local dataset metadata format: {suffix}")
        return LocalListDataset([self._map_fields(row, field_mapping) for row in rows])

    def _limit(self, dataset: LocalListDataset) -> LocalListDataset:
        if self.config.max_samples is None:
            return dataset
        return LocalListDataset(dataset[: self.config.max_samples])

    @staticmethod
    def _map_fields(row: dict[str, Any], field_mapping: dict[str, str] | None) -> dict[str, Any]:
        if not field_mapping:
            return row
        mapped = dict(row)
        for target, source in field_mapping.items():
            if source in row:
                mapped[target] = row[source]
        return mapped
