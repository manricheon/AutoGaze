import json

from repro.internvl_dynamic_tile_probe import run_internvl_dynamic_tile_probe


def test_internvl_dynamic_tile_probe_collects_tile_contract(tmp_path):
    model_dir = tmp_path / "InternVL3"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "internvl_chat",
                "vision_config": {
                    "image_size": 448,
                    "patch_size": 14,
                },
                "downsample_ratio": 0.5,
                "dynamic_image_size": True,
                "use_thumbnail": True,
                "min_dynamic_patch": 1,
                "max_dynamic_patch": 12,
            }
        )
    )

    probe = run_internvl_dynamic_tile_probe(
        model_path=str(model_dir),
        model_family="internvl3",
        video="inputs/example.mp4",
        num_video_frames=32,
        max_tiles_video=8,
    )

    assert probe["status"] == "static_probe_collected"
    assert probe["dynamic_tile_probe"]["status"] == "dynamic_tile_probe_required"
    assert probe["dynamic_tile_probe"]["model_grid_fields"] == ["pixel_values", "num_patches_list"]
    assert probe["dynamic_tile_probe"]["tile_level_strategy"] == "select_dynamic_tiles_before_vit"
    assert probe["dynamic_tile_probe"]["patch_level_strategy"] == "requires_patch_row_col_within_each_dynamic_tile"
    assert probe["dynamic_tile_probe"]["estimated_tokens_per_tile_before_downsample"] == 1024
    assert probe["dynamic_tile_probe"]["thumbnail_policy"] == "keep_all_until_thumbnail_mapping_is_verified"


def test_internvl_dynamic_tile_probe_handles_missing_config(tmp_path):
    model_dir = tmp_path / "InternVL3"
    model_dir.mkdir()

    probe = run_internvl_dynamic_tile_probe(
        model_path=str(model_dir),
        model_family="internvl3",
        video=None,
        num_video_frames=16,
        max_tiles_video=4,
    )

    assert probe["status"] == "config_missing"
    assert probe["dynamic_tile_probe"]["status"] == "config_missing"
    assert probe["runtime_probe_required"] is True
