from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SUPPORTED_HF_MODES = {
    "hf_model_only",
    "hf_dataset_only",
    "hf_model_and_dataset",
    "local_model_hf_dataset",
    "hf_model_local_dataset",
    "offline_hf_cache",
}


@dataclass(frozen=True)
class HFLoadConfig:
    model_id: str | None = None
    dataset_id: str | None = None
    dataset_config: str | None = None
    dataset_split: str | None = "validation"
    revision: str | None = None
    trust_remote_code: bool = False
    token_env_var: str = "HF_TOKEN"
    cache_dir: str | None = None
    local_files_only: bool = False
    offline: bool = False
    streaming: bool = False
    max_samples: int | None = None
    num_proc: int = 1
    model_class: str | None = None
    field_mapping: dict[str, str] | None = None

    @classmethod
    def from_mapping(cls, config: dict[str, Any] | Any) -> "HFLoadConfig":
        raw = dict(config or {})
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: raw[key] for key in allowed if key in raw})

    @property
    def token(self) -> str | None:
        return os.environ.get(self.token_env_var) if self.token_env_var else None

    def common_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "revision": self.revision,
            "cache_dir": self.cache_dir,
            "local_files_only": self.local_files_only or self.offline,
            "trust_remote_code": self.trust_remote_code,
        }
        token = self.token
        if token:
            kwargs["token"] = token
        return {key: value for key, value in kwargs.items() if value is not None}


def redacted_hf_config(config: HFLoadConfig) -> dict[str, Any]:
    data = {
        "model_id": config.model_id,
        "dataset_id": config.dataset_id,
        "dataset_config": config.dataset_config,
        "dataset_split": config.dataset_split,
        "revision": config.revision,
        "trust_remote_code": config.trust_remote_code,
        "token_env_var": config.token_env_var,
        "cache_dir": config.cache_dir,
        "local_files_only": config.local_files_only,
        "offline": config.offline,
        "streaming": config.streaming,
        "max_samples": config.max_samples,
        "num_proc": config.num_proc,
        "model_class": config.model_class,
    }
    data["token_present"] = config.token is not None
    return data


def resolve_cache_dir(cache_dir: str | None = None) -> Path | None:
    if cache_dir:
        return Path(cache_dir).expanduser()
    env_cache = os.environ.get("HF_HOME") or os.environ.get("HF_HUB_CACHE")
    return Path(env_cache).expanduser() if env_cache else None
