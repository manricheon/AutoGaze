from pathlib import Path

import av
from omegaconf import OmegaConf


def test_example_input_autogaze_config_matches_packaged_video():
    root = Path(__file__).resolve().parents[1]
    config = OmegaConf.load(root / "configs" / "repro" / "example_input_autogaze.yaml")
    video_path = root / config.video.path

    assert video_path.is_file()
    assert config.video.kind == "regular_mp4"
    assert config.video.source.width == 448
    assert config.video.source.height == 448
    assert config.video.source.frames == 64
    assert config.video.sampling.frames == 16
    assert config.autogaze.model == "nvidia/AutoGaze"
    assert config.siglip.model == "google/siglip2-base-patch16-224"

    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        assert stream.width == config.video.source.width
        assert stream.height == config.video.source.height
        assert stream.frames == config.video.source.frames
    finally:
        container.close()


def test_hlvid_like_nvila_config_uses_fixed_total_frame_sampling():
    root = Path(__file__).resolve().parents[1]
    config = OmegaConf.load(root / "configs" / "repro" / "hlvid_like_nvila_1024.yaml")

    assert config.nvila_runner.args.mode == "hlvid"
    assert config.nvila_runner.args.num_video_frames == 1024
    assert config.nvila_runner.args.num_video_frames_thumbnail == 64
    assert config.nvila_runner.args.max_tiles_video == 48
    assert config.nvila_runner.args.max_batch_size_autogaze == 16
    assert config.nvila_runner.args.max_batch_size_siglip == 32
    assert (root / "scripts" / "download_hlvid_example_video.py").is_file()


def test_hf_space_autogaze_examples_config_matches_downloaded_videos():
    root = Path(__file__).resolve().parents[1]
    config = OmegaConf.load(root / "configs" / "repro" / "hf_space_autogaze_examples.yaml")

    assert config.source.space == "bfshi/AutoGaze"
    assert config.space_autogaze_settings.ui_gazing_ratio == 0.75
    assert round(float(config.space_autogaze_settings.model_gazing_ratio), 6) == round(0.75 * 196 / 265, 6)
    assert config.space_autogaze_settings.task_loss_requirement == 0.7
    assert config.space_autogaze_settings.chunk_frames == 16
    assert config.space_autogaze_settings.spatial_chunk_size == 224
    assert config.nvila_runner.args.num_video_frames == 128
    assert (root / "scripts" / "download_hf_space_examples.py").is_file()

    for example in config.examples:
        video_path = root / example.local_path
        assert video_path.is_file()
        container = av.open(str(video_path))
        try:
            stream = container.streams.video[0]
            assert stream.width == example.source.width
            assert stream.height == example.source.height
            assert stream.frames == example.source.frames
        finally:
            container.close()


def test_streaming_pipeline_profiles_include_mps_and_cuda_recommendations():
    root = Path(__file__).resolve().parents[1]
    config = OmegaConf.load(root / "configs" / "repro" / "streaming_pipeline_profiles.yaml")

    assert config.defaults.stream_chunk_frames == 16
    assert config.defaults.stream_decode_strategy == "seek"
    assert config.local_mps.fast_448p.args.max_batch_size_autogaze == 1
    assert config.local_mps.siglip_patch16_probe.args.stream_run_siglip is True
    assert config.local_mps.siglip_patch16_probe.args.autogaze_target_patch_size == 16
    assert config.local_mps.balanced_720p.args.max_tiles_video == 4
    assert config.cuda.latency_4k.args.max_tiles_video == 8
    assert config.cuda.quality_4k.args.max_tiles_video == 16
    assert config.cuda.paper_probe_4k_256f.args.num_video_frames == 256
    assert config.cuda.paper_stress_4k_1024f.args.num_video_frames == 1024


def test_plugin_experiment_configs_mark_on_off_identity():
    root = Path(__file__).resolve().parents[1]
    config_dir = root / "configs" / "repro"

    expected = {
        "nvila_video_plugin_off.yaml": ("nvila-video-plugin", "keep-all", "plugin_off_native", "none"),
        "nvila_video_plugin_autogaze_requested.yaml": (
            "nvila-video-plugin",
            "autogaze",
            "experimental_plugin_requested",
            "planned_plugin",
        ),
        "longvila_plugin_off.yaml": ("longvila", "keep-all", "plugin_off_native", "none"),
        "longvila_plugin_autogaze_requested.yaml": (
            "longvila",
            "autogaze",
            "experimental_plugin_requested",
            "planned_plugin",
        ),
        "qwen3_vl_plugin_off.yaml": ("qwen3-vl", "keep-all", "plugin_off_native", "none"),
        "qwen3_vl_plugin_autogaze_post_prune.yaml": (
            "qwen3-vl",
            "autogaze",
            "experimental_plugin_requested",
            "post_encoder_token_prune",
        ),
        "qwen3_vl_pixelprune_pre_vit.yaml": (
            "qwen3-vl",
            "keep-all",
            "not_autogaze_pixelprune_pre_vit_reference",
            "pre_encoder_sparse",
        ),
        "qwen2_vl_plugin_off.yaml": ("qwen2-vl", "keep-all", "plugin_off_native", "none"),
        "qwen2_vl_plugin_autogaze_post_prune.yaml": (
            "qwen2-vl",
            "autogaze",
            "experimental_plugin_requested",
            "post_encoder_token_prune",
        ),
        "qwen25_vl_plugin_off.yaml": ("qwen2.5-vl", "keep-all", "plugin_off_native", "none"),
        "qwen25_vl_plugin_autogaze_post_prune.yaml": (
            "qwen2.5-vl",
            "autogaze",
            "experimental_plugin_requested",
            "post_encoder_token_prune",
        ),
        "llava_onevision_plugin_off.yaml": ("llava-onevision", "keep-all", "plugin_off_native", "none"),
        "llava_onevision_plugin_autogaze_post_prune.yaml": (
            "llava-onevision",
            "autogaze",
            "experimental_plugin_requested",
            "post_encoder_token_prune",
        ),
        "internvl3_plugin_off.yaml": ("internvl3", "keep-all", "plugin_off_native", "none"),
        "internvl3_plugin_autogaze_post_prune.yaml": (
            "internvl3",
            "autogaze",
            "experimental_plugin_requested",
            "post_encoder_token_prune",
        ),
    }

    for filename, (family, selector, applicability, integration_level) in expected.items():
        config = OmegaConf.load(config_dir / filename)
        assert config.experiment_identity.model_family == family
        assert config.experiment_identity.token_selector == selector
        assert config.experiment_identity.autogaze_applicability == applicability
        assert config.experiment_identity.autogaze_integration_level == integration_level
        assert config.flexible_runner.args.mode in {"inspect", "single"}
        assert config.flexible_runner.args.model_family == family
        assert config.flexible_runner.args.token_selector_adapter == selector
        assert config.flexible_runner.args.autogaze_integration_level == integration_level


def test_plugin_hlvid_limit3_config_defines_comparison_modes():
    root = Path(__file__).resolve().parents[1]
    config = OmegaConf.load(root / "configs" / "repro" / "plugin_hlvid_limit3.yaml")

    assert config.plugin_hlvid_benchmark.args.limit == 3
    assert "nvila-video-off" in config.plugin_hlvid_benchmark.args.modes
    assert "longvila-off" in config.plugin_hlvid_benchmark.args.modes
    assert "internvl3-off" in config.plugin_hlvid_benchmark.args.modes
    assert "qwen3-vl-pixelprune-pre-vit" in config.plugin_hlvid_benchmark.args.modes
    assert "qwen3-vl-autogaze-probe" in config.plugin_hlvid_benchmark.args.modes
    assert "qwen3-vl-autogaze-prune-generate" in config.plugin_hlvid_benchmark.args.modes
    assert (root / "scripts" / "download_qwen_model.py").is_file()


def test_plugin_hlvid_qwen_vit_limit3_config_defines_matched_qwen_modes():
    root = Path(__file__).resolve().parents[1]
    config = OmegaConf.load(root / "configs" / "repro" / "plugin_hlvid_qwen_vit_limit3.yaml")

    args = config.plugin_hlvid_benchmark.args
    assert args.limit == 3
    assert args.modes == [
        "qwen_full_vit",
        "qwen_chunked_vit",
        "qwen_chunked_vit_autogaze_sparse",
    ]
    assert args.models["qwen3-vl"] == "weight/Qwen3-VL-8B-Instruct"
    assert args.num_video_frames == 32
    assert args.qwen_video_nframes == 32
    assert args.num_video_frames_thumbnail == 8
    assert args.qwen_thumbnail_mode == "append-video"
    assert args.video_resize_longest_edge == 448
    assert args.qwen_vit_chunk_frames == 16
    assert args.qwen_vit_max_spatial_chunks == 4
    assert args.autogaze_target_scales == "112+224+336+448"
    assert args.autogaze_target_patch_size == 16
    assert args.autogaze_tile_size == 448


def test_autogaze_priority_validation_config_lists_four_tracks():
    root = Path(__file__).resolve().parents[1]
    config = OmegaConf.load(root / "configs" / "repro" / "autogaze_priority_validation.yaml")

    assert [track.name for track in config.priority_tracks] == [
        "nvila_hd_autogaze_profile",
        "nvila_video_baseline_autogaze_on_off",
        "longvila_autogaze_poc",
        "qwen3_vl_autogaze_poc",
    ]
    assert config.priority_tracks[0].runner.module == "repro.hlvid_batch_benchmark"
    assert "hd_autogaze" in config.priority_tracks[0].runner.modes
    assert "nvila-video-off" in config.priority_tracks[1].runner.modes
    assert "nvila-video-autogaze-probe" in config.priority_tracks[1].runner.modes
    assert "longvila-autogaze-probe" in config.priority_tracks[2].runner.modes
    assert "qwen3-vl-autogaze-poc" in config.priority_tracks[3].runner.modes
    assert "qwen3-vl-autogaze-prune-generate" in config.priority_tracks[3].runner.optional_modes
