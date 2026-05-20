import pytest

from repro.plugin_registry import PluginRegistry, PluginRegistryError


class DummyPlugin:
    role = "token_selector"
    name = "dummy"


def test_registry_registers_and_resolves_plugin_by_role_and_name():
    registry = PluginRegistry()

    registry.register("token_selector", "dummy", DummyPlugin)

    assert registry.resolve("token_selector", "dummy") is DummyPlugin
    assert registry.available("token_selector") == ["dummy"]


def test_registry_rejects_duplicate_plugin_names_for_same_role():
    registry = PluginRegistry()
    registry.register("token_selector", "dummy", DummyPlugin)

    with pytest.raises(PluginRegistryError, match="already registered"):
        registry.register("token_selector", "dummy", DummyPlugin)


def test_registry_reports_available_plugins_when_resolution_fails():
    registry = PluginRegistry()
    registry.register("mllm", "nvila-video", DummyPlugin)

    with pytest.raises(PluginRegistryError, match="Available for mllm: nvila-video"):
        registry.resolve("mllm", "longvila")
