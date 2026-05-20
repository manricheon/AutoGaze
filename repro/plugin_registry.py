from __future__ import annotations

from collections import defaultdict
from typing import Any


class PluginRegistryError(ValueError):
    """Raised when plugin registration or resolution fails."""


class PluginRegistry:
    def __init__(self) -> None:
        self._plugins: dict[str, dict[str, Any]] = defaultdict(dict)

    def register(self, role: str, name: str, plugin: Any) -> None:
        role_plugins = self._plugins[str(role)]
        name = str(name)
        if name in role_plugins:
            raise PluginRegistryError(f"Plugin {name!r} is already registered for role {role!r}.")
        role_plugins[name] = plugin

    def resolve(self, role: str, name: str) -> Any:
        role = str(role)
        name = str(name)
        role_plugins = self._plugins.get(role, {})
        if name not in role_plugins:
            available = ", ".join(sorted(role_plugins)) or "none"
            raise PluginRegistryError(f"Unknown plugin {name!r} for role {role!r}. Available for {role}: {available}")
        return role_plugins[name]

    def available(self, role: str) -> list[str]:
        return sorted(self._plugins.get(str(role), {}))
