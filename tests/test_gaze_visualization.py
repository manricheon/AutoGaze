import json

import av
import torch
from PIL import Image

from repro.gaze_visualization import (
    build_gaze_overlay_records,
    render_overlay_frame,
    write_gaze_visualization_artifacts,
)


def test_build_gaze_overlay_records_maps_multiscale_positions_to_frame_boxes():
    patches_per_frame = 1060
    gazing_info = {
        "gazing_pos_tiles": [torch.tensor([[0, patches_per_frame + 16]])],
        "if_padded_gazing_tiles": [torch.tensor([[False, False]])],
        "num_gazing_each_frame_tiles": [torch.tensor([[1, 1]])],
    }

    records = build_gaze_overlay_records(
        gazing_info=gazing_info,
        video_index=0,
        num_video_frames=2,
        spatial_tiles=1,
        grid_cols=1,
        scales=[56, 112, 196, 392],
        patch_size=14,
        tile_size=392,
    )

    assert len(records[0]) == 1
    assert records[0][0]["scale"] == 56
    assert records[0][0]["bbox"] == [0, 0, 98, 98]
    assert len(records[1]) == 1
    assert records[1][0]["scale"] == 112
    assert records[1][0]["bbox"] == [0, 0, 49, 49]


def test_build_gaze_overlay_records_offsets_spatial_tiles_on_canvas():
    gazing_info = {
        "gazing_pos_tiles": [
            torch.tensor(
                [
                    [0],
                    [16],
                ]
            )
        ],
        "if_padded_gazing_tiles": [torch.tensor([[False], [False]])],
        "num_gazing_each_frame_tiles": [torch.tensor([[1], [1]])],
    }

    records = build_gaze_overlay_records(
        gazing_info=gazing_info,
        video_index=0,
        num_video_frames=1,
        spatial_tiles=2,
        grid_cols=2,
        scales=[56, 112, 196, 392],
        patch_size=14,
        tile_size=392,
    )

    assert [record["scale"] for record in records[0]] == [56, 112]
    assert records[0][0]["bbox"] == [0, 0, 98, 98]
    assert records[0][1]["bbox"] == [392, 0, 441, 49]


def test_render_overlay_frame_draws_scale_colored_patch_masks():
    frame = Image.new("RGB", (392, 392), "white")
    records = [
        {
            "scale": 56,
            "scale_index": 0,
            "bbox": [0, 0, 98, 98],
            "color": [255, 69, 58],
        }
    ]

    rendered = render_overlay_frame(frame, records, alpha=0.5)

    assert rendered.size == frame.size
    assert rendered.getpixel((20, 20)) != (255, 255, 255)
    assert rendered.getpixel((0, 0)) != (255, 69, 58)
    assert rendered.getpixel((180, 180)) == (255, 255, 255)


def video_size(path):
    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        return int(stream.width), int(stream.height)
    finally:
        container.close()


def test_write_gaze_visualization_artifacts_saves_selected_sized_overlay_and_manifest(tmp_path):
    frames = [
        Image.new("RGB", (196, 196), "white"),
        Image.new("RGB", (196, 196), "gray"),
    ]
    processor_frames = [
        Image.new("RGB", (392, 392), "white"),
        Image.new("RGB", (392, 392), "gray"),
    ]
    patches_per_frame = 1060
    gazing_info = {
        "gazing_pos_tiles": [torch.tensor([[0, patches_per_frame + 16]])],
        "if_padded_gazing_tiles": [torch.tensor([[False, False]])],
        "num_gazing_each_frame_tiles": [torch.tensor([[1, 1]])],
        "gazing_pos_thumbnails": [torch.tensor([[0]])],
        "if_padded_gazing_thumbnails": [torch.tensor([[False]])],
        "num_gazing_each_frame_thumbnails": [torch.tensor([[1]])],
    }

    manifest = write_gaze_visualization_artifacts(
        selected_frames=frames,
        overlay_base_frames=processor_frames,
        output_dir=tmp_path,
        label="single_clip",
        video="clip.mp4",
        sampled_frame_indices=[0, 9],
        gazing_mode="autogaze",
        gazing_info=gazing_info,
        spatial_tiles=1,
        grid_cols=1,
        grid_rows=1,
        scales=[56, 112, 196, 392],
        patch_size=14,
        tile_size=392,
        fps=4,
    )

    assert manifest["status"] == "written"
    assert manifest["selected_frames_video"].endswith("single_clip_selected_frames.mp4")
    assert manifest["overlay_video"].endswith("single_clip_autogaze_overlay.mp4")
    assert manifest["gazing_info_json"].endswith("single_clip_gazing_info.json")
    assert (tmp_path / "single_clip_selected_frames.mp4").exists()
    assert (tmp_path / "single_clip_autogaze_overlay.mp4").exists()
    assert video_size(tmp_path / "single_clip_selected_frames.mp4") == (196, 196)
    assert video_size(tmp_path / "single_clip_autogaze_overlay.mp4") == (196, 196)
    raw = json.loads((tmp_path / "single_clip_gazing_info.json").read_text())
    assert raw["sampled_frame_indices"] == [0, 9]
    assert raw["overlay_render_size"] == [196, 196]
    assert raw["overlay_coordinate_space"] == "selected_frame"
    assert raw["overlay_records_by_frame"]["0"][0]["scale"] == 56
    assert raw["overlay_records_by_frame"]["0"][0]["bbox"] == [0, 0, 49, 49]
