from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf


DEFAULT_PACKAGES = (
    "torch",
    "torchvision",
    "transformers",
    "datasets",
    "evaluate",
    "huggingface_hub",
    "accelerate",
    "hydra-core",
    "omegaconf",
    "numpy",
    "timm",
)

SENSITIVE_KEY_PARTS = (
    "token",
    "access_token",
    "api_key",
    "apikey",
    "secret",
    "password",
    "authorization",
    "credential",
)

SAFE_TOKEN_KEYS = {"token_env_var", "token_present"}
REDACTED = "<REDACTED>"


@dataclass(frozen=True)
class HFRevisionInfo:
    model_id: str | None = None
    model_revision: str | None = None
    dataset_id: str | None = None
    dataset_config: str | None = None
    dataset_split: str | None = None
    dataset_revision: str | None = None
    cache_dir: str | None = None
    offline: bool = False
    local_files_only: bool = False
    trust_remote_code: bool = False


@dataclass(frozen=True)
class ReproducibilityManifest:
    benchmark_timestamp: str
    resolved_config: dict[str, Any]
    git_commit_hash: str | None
    git_dirty: bool | None
    package_versions: dict[str, str]
    device_information: dict[str, Any]
    cuda_available: bool
    mps_available: bool
    precision_setting: str | None
    model_checkpoints_used: dict[str, Any]
    huggingface: HFRevisionInfo
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return redact_sensitive_values(data)


def _to_plain_config(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, (DictConfig, ListConfig)):
        data = OmegaConf.to_container(config, resolve=True)
    elif isinstance(config, Mapping):
        data = dict(config)
    else:
        raise TypeError(f"Unsupported config type for reproducibility manifest: {type(config)!r}")
    if not isinstance(data, dict):
        raise TypeError("Reproducibility config must resolve to a mapping")
    return data


def _is_sensitive_key(key: str) -> bool:
    normalized = key.lower()
    if normalized in SAFE_TOKEN_KEYS:
        return False
    return any(part in normalized for part in SENSITIVE_KEY_PARTS)


def redact_sensitive_values(value: Any) -> Any:
    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_str = str(key)
            redacted[key_str] = REDACTED if _is_sensitive_key(key_str) and item is not None else redact_sensitive_values(item)
        return redacted
    if isinstance(value, list):
        return [redact_sensitive_values(item) for item in value]
    if isinstance(value, tuple):
        return [redact_sensitive_values(item) for item in value]
    return value


def capture_git_commit_hash(repo_root: str | Path = ".") -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def capture_git_dirty(repo_root: str | Path = ".") -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(result.stdout.strip())


def capture_package_versions(packages: Iterable[str] = DEFAULT_PACKAGES) -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    return versions


def capture_device_info() -> dict[str, Any]:
    cuda_devices: list[dict[str, Any]] = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            cuda_devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_mb": int(props.total_memory // (1024 * 1024)),
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
            )

    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "cuda_devices": cuda_devices,
        "mps_available": bool(torch.backends.mps.is_available()),
        "mps_built": bool(torch.backends.mps.is_built()),
    }


def _get_nested(config: Mapping[str, Any], *keys: str) -> Any:
    cursor: Any = config
    for key in keys:
        if not isinstance(cursor, Mapping) or key not in cursor:
            return None
        cursor = cursor[key]
    return cursor


def _node(config: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    value = _get_nested(config, *keys)
    return dict(value) if isinstance(value, Mapping) else {}


def capture_huggingface_revisions(config: Any) -> HFRevisionInfo:
    plain = _to_plain_config(config)
    runtime_hf = _node(plain, "runtime", "huggingface")
    model_hf = _node(plain, "model", "huggingface")
    data_hf = _node(plain, "data", "huggingface")

    model_revision = model_hf.get("revision") or runtime_hf.get("revision")
    dataset_revision = data_hf.get("revision") or runtime_hf.get("revision")
    cache_dir = model_hf.get("cache_dir") or data_hf.get("cache_dir") or runtime_hf.get("cache_dir")

    return HFRevisionInfo(
        model_id=model_hf.get("model_id"),
        model_revision=model_revision,
        dataset_id=data_hf.get("dataset_id"),
        dataset_config=data_hf.get("dataset_config"),
        dataset_split=data_hf.get("dataset_split"),
        dataset_revision=dataset_revision,
        cache_dir=cache_dir,
        offline=bool(model_hf.get("offline") or data_hf.get("offline") or runtime_hf.get("offline")),
        local_files_only=bool(
            model_hf.get("local_files_only") or data_hf.get("local_files_only") or runtime_hf.get("local_files_only")
        ),
        trust_remote_code=bool(model_hf.get("trust_remote_code") or runtime_hf.get("trust_remote_code")),
    )


def capture_model_checkpoints(config: Any) -> dict[str, Any]:
    plain = _to_plain_config(config)
    return redact_sensitive_values(
        {
            "autogaze": _get_nested(plain, "model", "autogaze", "checkpoint"),
            "vision_encoder": _get_nested(plain, "model", "vision_encoder", "checkpoint"),
            "mllm": _get_nested(plain, "model", "mllm", "checkpoint"),
            "task_decoder": _get_nested(plain, "model", "task_decoder", "checkpoint"),
        }
    )


def create_reproducibility_manifest(
    config: Any,
    *,
    repo_root: str | Path = ".",
    packages: Iterable[str] = DEFAULT_PACKAGES,
    timestamp: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ReproducibilityManifest:
    plain = _to_plain_config(config)
    timestamp = timestamp or datetime.now(timezone.utc).isoformat()
    device_info = capture_device_info()
    precision = _get_nested(plain, "runtime", "precision", "dtype")

    return ReproducibilityManifest(
        benchmark_timestamp=timestamp,
        resolved_config=redact_sensitive_values(plain),
        git_commit_hash=capture_git_commit_hash(repo_root),
        git_dirty=capture_git_dirty(repo_root),
        package_versions=capture_package_versions(packages),
        device_information=device_info,
        cuda_available=bool(device_info["cuda_available"]),
        mps_available=bool(device_info["mps_available"]),
        precision_setting=str(precision) if precision is not None else None,
        model_checkpoints_used=capture_model_checkpoints(plain),
        huggingface=capture_huggingface_revisions(plain),
        metadata=redact_sensitive_values(dict(metadata or {})),
    )


def save_reproducibility_manifest(
    config: Any,
    output_path: str | Path,
    *,
    repo_root: str | Path = ".",
    packages: Iterable[str] = DEFAULT_PACKAGES,
    timestamp: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    manifest = create_reproducibility_manifest(
        config,
        repo_root=repo_root,
        packages=packages,
        timestamp=timestamp,
        metadata=metadata,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path

