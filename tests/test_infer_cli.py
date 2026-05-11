import sys
from pathlib import Path

import numpy as np
import torch

import autogaze.infer as infer


def test_full_patch_outputs_selects_all_tokens():
    out = infer.full_patch_outputs(num_frames=2)

    assert out["scales"] == [32, 64, 112, 224]
    assert out["num_vision_tokens_each_frame"] == 265
    assert out["gazing_pos"].shape == (1, 530)
    assert not out["if_padded_gazing"].any()
    assert torch.equal(out["num_gazing_each_frame"], torch.tensor([265, 265]))
    assert all(mask.all() for mask in out["gazing_mask"])


def test_no_autogaze_mode_does_not_load_model(monkeypatch, tmp_path):
    video_path = Path("synthetic.mp4")
    output_dir = tmp_path / "out"

    def fail_load(*_args, **_kwargs):
        raise AssertionError("AutoGaze should not be loaded in --no-autogaze mode")

    monkeypatch.setattr(infer.AutoGaze, "from_pretrained", fail_load)
    monkeypatch.setattr(infer.AutoGazeImageProcessor, "from_pretrained", fail_load)
    monkeypatch.setattr(infer, "collect_video_paths", lambda _input: [video_path])
    monkeypatch.setattr(
        infer,
        "load_video",
        lambda _path, _num_frames: (np.zeros((2, 8, 8, 3), dtype=np.uint8), [0, 1]),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "infer.py",
            str(video_path),
            "--no-autogaze",
            "--output-format",
            "npy",
            "--output-dir",
            str(output_dir),
            "--num-frames",
            "2",
        ],
    )

    infer.main()

    assert (output_dir / "synthetic_gaze.npz").exists()
