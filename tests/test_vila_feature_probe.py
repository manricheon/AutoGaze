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
