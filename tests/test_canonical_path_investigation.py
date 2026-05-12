from __future__ import annotations

import sys
import types
from pathlib import Path

from autogaze_ext.investigation import check_canonical_path
from autogaze_ext.utils import resolve_import


def _probe(available: set[str]):
    def inner(module_name: str) -> bool:
        return module_name in available

    return inner


def test_canonical_path_reports_stub_only_when_modules_and_checkpoints_missing() -> None:
    report = check_canonical_path(import_probe=_probe(set()))

    assert report.components["original_autogaze"].module_available is False
    assert report.components["modified_siglip"].module_available is False
    assert report.components["nvila"].module_available is False
    assert report.experiments["A1"].can_run_real is False
    assert report.experiments["A1"].can_run_stub is True
    assert report.experiments["A2"].can_run_real is False
    assert any("module import unavailable" in item for item in report.experiments["A2"].missing)


def test_canonical_path_reports_a1_real_when_modified_siglip_and_nvila_ready(tmp_path: Path) -> None:
    modified_ckpt = tmp_path / "modified_siglip.pt"
    nvila_ckpt = tmp_path / "nvila.pt"
    modified_ckpt.write_text("dummy", encoding="utf-8")
    nvila_ckpt.write_text("dummy", encoding="utf-8")

    report = check_canonical_path(
        import_probe=_probe({"autogaze.models.modified_siglip", "nvila"}),
        checkpoint_overrides={
            "modified_siglip": modified_ckpt,
            "nvila": nvila_ckpt,
        },
    )

    assert report.components["modified_siglip"].real_ready is True
    assert report.components["nvila"].real_ready is True
    assert report.components["original_autogaze"].real_ready is False
    assert report.experiments["A1"].can_run_real is True
    assert report.experiments["A2"].can_run_real is False
    assert any("original_autogaze" in item for item in report.experiments["A2"].missing)


def test_canonical_path_reports_a2_real_when_all_components_ready(tmp_path: Path) -> None:
    autogaze_ckpt = tmp_path / "autogaze.pt"
    modified_ckpt = tmp_path / "modified_siglip.pt"
    nvila_ckpt = tmp_path / "nvila.pt"
    for path in (autogaze_ckpt, modified_ckpt, nvila_ckpt):
        path.write_text("dummy", encoding="utf-8")

    report = check_canonical_path(
        import_probe=_probe({"autogaze.models.autogaze", "autogaze.models.modified_siglip", "nvila"}),
        checkpoint_overrides={
            "original_autogaze": autogaze_ckpt,
            "modified_siglip": modified_ckpt,
            "nvila": nvila_ckpt,
        },
    )

    assert report.experiments["A1"].can_run_real is True
    assert report.experiments["A2"].can_run_real is True
    assert report.experiments["A2"].missing == []


def test_canonical_path_report_to_dict_contains_expected_paths(tmp_path: Path) -> None:
    missing_ckpt = tmp_path / "missing.pt"

    report = check_canonical_path(
        import_probe=_probe({"autogaze.models.autogaze"}),
        checkpoint_overrides={"original_autogaze": missing_ckpt},
    )
    data = report.to_dict()

    component = data["components"]["original_autogaze"]
    assert component["module_available"] is True
    assert component["checkpoint_exists"] is False
    assert str(missing_ckpt) in component["expected_paths"]
    assert "config:model.autogaze.checkpoint" in component["expected_paths"]
    assert data["experiments"]["A2"]["can_run_real"] is False


def test_resolve_import_reports_missing_module_and_object() -> None:
    missing_module = resolve_import("__missing_autogaze_test_module__", "Factory")
    assert missing_module.module_available is False
    assert missing_module.object_available is False
    assert "failed to import module" in str(missing_module.error)

    module = types.ModuleType("_mock_resolve_import_module")
    sys.modules[module.__name__] = module
    try:
        missing_object = resolve_import(module.__name__, "Factory")
        assert missing_object.module_available is True
        assert missing_object.object_available is False
        assert "does not provide" in str(missing_object.error)
    finally:
        sys.modules.pop(module.__name__, None)


def test_real_canonical_configs_validate_mocked_modules_and_fake_checkpoints(monkeypatch, tmp_path: Path) -> None:
    autogaze_module = types.ModuleType("autogaze.models.autogaze")
    siglip_module = types.ModuleType("autogaze.vision_encoders.siglip.modeling_siglip")
    transformers_module = types.ModuleType("transformers")

    class AutoGaze:
        pass

    class SiglipVisionModel:
        pass

    class AutoModel:
        pass

    autogaze_module.AutoGaze = AutoGaze
    siglip_module.SiglipVisionModel = SiglipVisionModel
    transformers_module.AutoModel = AutoModel
    monkeypatch.setitem(sys.modules, "autogaze.models.autogaze", autogaze_module)
    monkeypatch.setitem(sys.modules, "autogaze.vision_encoders.siglip.modeling_siglip", siglip_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)

    autogaze_ckpt = tmp_path / "autogaze.pt"
    modified_ckpt = tmp_path / "modified_siglip.pt"
    nvila_ckpt = tmp_path / "nvila.pt"
    for path in (autogaze_ckpt, modified_ckpt, nvila_ckpt):
        path.write_text("dummy", encoding="utf-8")

    report = check_canonical_path(
        experiment_ids=("A1_real", "A2_real"),
        checkpoint_overrides={
            "original_autogaze": autogaze_ckpt,
            "modified_siglip": modified_ckpt,
            "nvila": nvila_ckpt,
        },
    )

    assert report.components["original_autogaze"].configured_module_path == "autogaze.models.autogaze"
    assert report.components["original_autogaze"].configured_class_or_factory == "AutoGaze"
    assert report.components["original_autogaze"].class_or_factory_exists is True
    assert report.components["modified_siglip"].configured_module_path == "autogaze.vision_encoders.siglip.modeling_siglip"
    assert report.components["modified_siglip"].class_or_factory_exists is True
    assert report.components["nvila"].configured_module_path == "transformers"
    assert report.components["nvila"].configured_class_or_factory == "AutoModel"
    assert report.components["nvila"].class_or_factory_exists is True
    assert report.components["nvila"].ready_for_model_construction is True
    assert report.experiments["A1_real"].can_run_real is True
    assert report.experiments["A2_real"].can_run_real is True


def test_real_canonical_config_reports_missing_factory(monkeypatch, tmp_path: Path) -> None:
    autogaze_module = types.ModuleType("autogaze.models.autogaze")
    siglip_module = types.ModuleType("autogaze.vision_encoders.siglip.modeling_siglip")
    transformers_module = types.ModuleType("transformers")
    autogaze_module.AutoGaze = object
    siglip_module.SiglipVisionModel = object
    monkeypatch.setitem(sys.modules, "autogaze.models.autogaze", autogaze_module)
    monkeypatch.setitem(sys.modules, "autogaze.vision_encoders.siglip.modeling_siglip", siglip_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)

    autogaze_ckpt = tmp_path / "autogaze.pt"
    modified_ckpt = tmp_path / "modified_siglip.pt"
    nvila_ckpt = tmp_path / "nvila.pt"
    for path in (autogaze_ckpt, modified_ckpt, nvila_ckpt):
        path.write_text("dummy", encoding="utf-8")

    report = check_canonical_path(
        experiment_ids=("A1_real", "A2_real"),
        checkpoint_overrides={
            "original_autogaze": autogaze_ckpt,
            "modified_siglip": modified_ckpt,
            "nvila": nvila_ckpt,
        },
    )

    assert report.components["nvila"].module_available is True
    assert report.components["nvila"].class_or_factory_exists is False
    assert report.components["nvila"].ready_for_model_construction is False
    assert any("class/factory unavailable" in item for item in report.experiments["A1_real"].missing)
