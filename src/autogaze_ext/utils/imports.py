from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable


ImportModuleFn = Callable[[str], ModuleType]


@dataclass(frozen=True)
class ImportResolution:
    module_path: str | None
    object_name: str | None
    module_available: bool
    object_available: bool
    error: str | None = None

    @property
    def ready(self) -> bool:
        return self.module_available and (self.object_name is None or self.object_available)


def resolve_import(
    module_path: str | None,
    object_name: str | None = None,
    *,
    import_module_fn: ImportModuleFn = importlib.import_module,
) -> ImportResolution:
    """Resolve a module and optional class/factory name without instantiating it."""
    if not module_path:
        return ImportResolution(
            module_path=None,
            object_name=object_name,
            module_available=False,
            object_available=False,
            error="module_path is not configured",
        )

    try:
        module = import_module_fn(module_path)
    except Exception as exc:
        return ImportResolution(
            module_path=module_path,
            object_name=object_name,
            module_available=False,
            object_available=False,
            error=f"failed to import module '{module_path}': {exc}",
        )

    if not object_name:
        return ImportResolution(
            module_path=module_path,
            object_name=None,
            module_available=True,
            object_available=True,
            error=None,
        )

    cursor: Any = module
    for part in object_name.split("."):
        if not hasattr(cursor, part):
            return ImportResolution(
                module_path=module_path,
                object_name=object_name,
                module_available=True,
                object_available=False,
                error=f"module '{module_path}' does not provide '{object_name}'",
            )
        cursor = getattr(cursor, part)

    return ImportResolution(
        module_path=module_path,
        object_name=object_name,
        module_available=True,
        object_available=True,
        error=None,
    )

