from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class QuickStartLocation:
    path: Path
    source: str


def locate_quick_start(
    configured_path: str | Path | None = None,
    *,
    repo_root: str | Path = ".",
    env_var: str = "AUTOGAZE_ORIGINAL_SOURCE_PATH",
) -> QuickStartLocation:
    """Locate the original QUICK_START.md without modifying it."""
    candidates: list[tuple[Path, str]] = []
    if configured_path:
        configured = Path(configured_path).expanduser()
        candidates.append((configured / "QUICK_START.md" if configured.is_dir() else configured, "configured_path"))

    env_path = os.environ.get(env_var)
    if env_path:
        env_candidate = Path(env_path).expanduser()
        candidates.append((env_candidate / "QUICK_START.md" if env_candidate.is_dir() else env_candidate, f"env:{env_var}"))

    root = Path(repo_root)
    candidates.extend(
        [
            (root / "QUICK_START.md", "repo_root"),
            (root / "autogaze" / "QUICK_START.md", "repo_autogaze_subdir"),
        ]
    )

    for path, source in candidates:
        if path.exists() and path.is_file():
            return QuickStartLocation(path=path.resolve(), source=source)

    searched = ", ".join(str(path) for path, _ in candidates)
    raise FileNotFoundError(f"Could not locate QUICK_START.md. Searched: {searched}")

