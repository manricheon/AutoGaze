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
