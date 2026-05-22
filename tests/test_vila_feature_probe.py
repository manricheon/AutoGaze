import json

from repro.vila_feature_probe import run_vila_feature_probe


def test_vila_feature_probe_collects_static_config_contract(tmp_path):
    model_dir = tmp_path / "NVILA-8B-Video"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llava",
                "architectures": ["LlavaQwenForCausalLM"],
                "vision_tower": "google/siglip-so400m-patch14-384",
                "mm_projector_type": "mlp2x_gelu",
                "mm_vision_select_layer": -2,
                "video_token_len": 196,
            }
        )
    )

    probe = run_vila_feature_probe(
        model_path=str(model_dir),
        model_family="nvila-video-plugin",
        video="inputs/example.mp4",
        prompt="What happens?",
        num_video_frames=256,
        max_tiles_video=8,
    )

    assert probe["status"] == "static_probe_collected"
    assert probe["model_family"] == "nvila-video-plugin"
    assert probe["config_summary"]["model_type"] == "llava"
    assert probe["config_summary"]["architectures"] == ["LlavaQwenForCausalLM"]
    assert "vision_tower" in probe["config_summary"]["feature_packing_related_keys"]
    assert "mm_projector_type" in probe["config_summary"]["feature_packing_related_keys"]
    assert probe["probe_targets"] == [
        "processor video tensor/frame contract",
        "vision feature shape",
        "projector output shape",
        "LLM visual token insertion boundary",
    ]
    assert probe["runtime_probe_required"] is True
    assert probe["next_action"] == "instrument_vila_remote_code_feature_packing"
    assert probe["pre_vit_sparse_probe"]["status"] == "in_process_probe_required"
    assert probe["pre_vit_sparse_probe"]["integration_level"] == "pre_encoder_sparse"
    assert probe["pre_vit_sparse_probe"]["first_prunable_boundary"] == "before_vision_tower_forward"
    assert "vision_tower.forward input tensor" in probe["pre_vit_sparse_probe"]["required_hooks"]
    assert probe["pre_vit_sparse_probe"]["external_cli_limitation"] is True


def test_vila_feature_probe_reports_patch_position_alignment_risks(tmp_path):
    model_dir = tmp_path / "LongVILA"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llava",
                "architectures": ["LongVILAForCausalLM"],
                "vision_tower": "siglip-so400m-patch14-384",
                "mm_projector_type": "mlp2x_gelu",
                "image_aspect_ratio": "anyres",
            }
        )
    )

    probe = run_vila_feature_probe(
        model_path=str(model_dir),
        model_family="longvila",
        video="inputs/long.mp4",
        prompt="What happens?",
        num_video_frames=128,
        max_tiles_video=4,
    )

    assert probe["pre_vit_sparse_probe"]["position_alignment"] == {
        "status": "must_preserve_or_rebuild",
        "fields": ["frame_order", "tile_id", "patch_row", "patch_col", "scale_id"],
    }
    assert probe["pre_vit_sparse_probe"]["next_actions"] == [
        "load model in-process instead of external CLI",
        "capture processor output pixel tensor and frame/tile metadata",
        "capture vision tower input/output shape",
        "map SparseSelectionPlan patch coordinates to vision tower token order",
    ]
