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
