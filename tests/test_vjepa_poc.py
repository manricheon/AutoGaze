import json

import pytest

from repro.vjepa_poc import build_synthetic_sparse_selection_plan, main, run_mapping_probe


def test_build_synthetic_sparse_selection_plan_contains_multiscale_patches():
    plan = build_synthetic_sparse_selection_plan()

    assert plan.selector_name == "autogaze-direct-synthetic"
    assert plan.source_video.sampled_frame_indices == [0, 1, 2, 3]
    assert plan.patch_space.scale_sizes == [112, 224]
    assert {patch.scale_id for patch in plan.selected_patches} == {0, 1}


def test_run_mapping_probe_returns_vjepa_counts_and_markdown(tmp_path):
    output_json = tmp_path / "vjepa_poc.json"
    output_md = tmp_path / "vjepa_poc.md"

    payload = run_mapping_probe(
        sparse_selection_plan=build_synthetic_sparse_selection_plan(),
        frames_per_clip=4,
        tubelet_size=2,
        crop_size=224,
        patch_size=16,
        output_json=output_json,
        output_md=output_md,
    )

    assert payload["implementation_status"] == "mapping_probe_ready"
    assert payload["vjepa"]["raw_token_count"] == 392
    assert payload["vjepa"]["selected_token_count"] < 392
    assert payload["vjepa"]["mapping_policy"]["tubelet"] == "any_frame_selected"
    assert json.loads(output_json.read_text())["vjepa"]["status"] == "mapped"
    assert "AutoGaze + V-JEPA PoC" in output_md.read_text()


def test_run_mapping_probe_can_include_scale_aware_probe(tmp_path):
    payload = run_mapping_probe(
        sparse_selection_plan=build_synthetic_sparse_selection_plan(),
        frames_per_clip=4,
        tubelet_size=2,
        crop_size=224,
        patch_size=16,
        include_scale_aware=True,
        output_json=tmp_path / "scale_aware.json",
        output_md=tmp_path / "scale_aware.md",
    )

    assert payload["scale_aware_vjepa"]["status"] == "mapped"
    assert payload["scale_aware_vjepa"]["raw_token_count"] > payload["vjepa"]["raw_token_count"]
    assert payload["scale_aware_vjepa"]["selected_token_count"] <= payload["vjepa"]["selected_token_count"]
    assert payload["scale_aware_vjepa"]["mapping_policy"]["multiscale"] == "separate_vjepa_pass_per_autogaze_scale"


def test_run_mapping_probe_can_run_tiny_sparse_encoder_smoke(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    payload = run_mapping_probe(
        sparse_selection_plan=build_synthetic_sparse_selection_plan(),
        frames_per_clip=4,
        tubelet_size=2,
        crop_size=224,
        patch_size=16,
        tiny_encoder_smoke=True,
        output_json=tmp_path / "tiny_encoder.json",
        output_md=tmp_path / "tiny_encoder.md",
    )

    assert payload["vjepa_sparse_encoder_smoke"]["status"] == "passed"
    assert payload["vjepa_sparse_encoder_smoke"]["metrics"]["raw_token_count"] == 392
    assert payload["vjepa_sparse_encoder_smoke"]["metrics"]["selected_token_count"] == payload["vjepa"]["selected_token_count"]


def test_vjepa_poc_main_writes_synthetic_outputs(tmp_path):
    output_json = tmp_path / "main.json"
    output_md = tmp_path / "main.md"

    exit_code = main(
        [
            "--synthetic",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--scale-aware",
            "--tiny-encoder-smoke",
        ]
    )

    assert exit_code == 0
    assert output_json.is_file()
    assert output_md.is_file()
