from __future__ import annotations

from fractions import Fraction
import json
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import av
from PIL import Image, ImageDraw


DEFAULT_SCALE_COLORS = (
    (255, 69, 58),
    (52, 199, 89),
    (0, 122, 255),
    (255, 204, 0),
    (191, 90, 242),
    (255, 149, 0),
)


def safe_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return label.strip("._") or "visualization"


def _to_python(value: Any) -> Any:
    if hasattr(value, "detach") and callable(value.detach):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _to_python(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_python(item) for item in value]
    return value


def serialize_gazing_info(gazing_info: dict[str, Any] | None) -> dict[str, Any] | None:
    if gazing_info is None:
        return None
    return _to_python(gazing_info)


def _per_video_value(gazing_info: dict[str, Any], key: str, video_index: int) -> Any:
    value = gazing_info.get(key)
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if video_index >= len(value):
            return None
        return value[video_index]
    if video_index == 0:
        return value
    return None


def _matrix(value: Any) -> list[list[Any]]:
    data = _to_python(value)
    if data is None:
        return []
    if not isinstance(data, list):
        return [[data]]
    if not data:
        return []
    if isinstance(data[0], list):
        return data
    return [data]


def _scale_offsets(scales: Sequence[int], patch_size: int) -> list[dict[str, int]]:
    offsets: list[dict[str, int]] = []
    cursor = 0
    for index, scale in enumerate(scales):
        grid = int(scale) // int(patch_size)
        count = grid * grid
        offsets.append(
            {
                "scale": int(scale),
                "scale_index": index,
                "start": cursor,
                "end": cursor + count,
                "grid": grid,
            }
        )
        cursor += count
    return offsets


def _patch_location(
    *,
    position_in_frame: int,
    scales: Sequence[int],
    patch_size: int,
) -> dict[str, int] | None:
    for item in _scale_offsets(scales, patch_size):
        if item["start"] <= position_in_frame < item["end"]:
            patch_index = position_in_frame - item["start"]
            grid = item["grid"]
            return {
                "scale": item["scale"],
                "scale_index": item["scale_index"],
                "grid": grid,
                "patch_index_in_scale": patch_index,
                "patch_row": patch_index // grid,
                "patch_col": patch_index % grid,
            }
    return None


def palette_for_scales(scales: Sequence[int]) -> dict[int, list[int]]:
    colors: dict[int, list[int]] = {}
    for index, scale in enumerate(scales):
        colors[int(scale)] = list(DEFAULT_SCALE_COLORS[index % len(DEFAULT_SCALE_COLORS)])
    return colors


def build_gaze_overlay_records(
    *,
    gazing_info: dict[str, Any],
    video_index: int,
    num_video_frames: int,
    spatial_tiles: int,
    grid_cols: int,
    scales: Sequence[int],
    patch_size: int,
    tile_size: int,
) -> dict[int, list[dict[str, Any]]]:
    records_by_frame: dict[int, list[dict[str, Any]]] = {index: [] for index in range(num_video_frames)}
    positions = _matrix(_per_video_value(gazing_info, "gazing_pos_tiles", video_index))
    padded = _matrix(_per_video_value(gazing_info, "if_padded_gazing_tiles", video_index))
    counts = _matrix(_per_video_value(gazing_info, "num_gazing_each_frame_tiles", video_index))
    if not positions or not counts:
        return records_by_frame

    spatial_tiles = max(int(spatial_tiles), 1)
    grid_cols = max(int(grid_cols), 1)
    patches_per_frame = sum((int(scale) // int(patch_size)) ** 2 for scale in scales)
    colors = palette_for_scales(scales)

    for tile_sequence_index, pos_row in enumerate(positions):
        count_row = counts[tile_sequence_index] if tile_sequence_index < len(counts) else [len(pos_row)]
        pad_row = padded[tile_sequence_index] if tile_sequence_index < len(padded) else [False] * len(pos_row)
        if not count_row:
            continue

        temporal_chunk = tile_sequence_index // spatial_tiles
        spatial_index = tile_sequence_index % spatial_tiles
        tile_col = spatial_index % grid_cols
        tile_row = spatial_index // grid_cols
        chunk_frames = len(count_row)
        offset = 0

        for frame_in_chunk, count_value in enumerate(count_row):
            count = int(count_value)
            sampled_frame_offset = temporal_chunk * chunk_frames + frame_in_chunk
            if sampled_frame_offset >= num_video_frames:
                offset += count
                continue

            for segment_index in range(offset, min(offset + count, len(pos_row))):
                if segment_index < len(pad_row) and bool(pad_row[segment_index]):
                    continue
                position = int(pos_row[segment_index])
                position_in_frame = position % patches_per_frame
                patch = _patch_location(
                    position_in_frame=position_in_frame,
                    scales=scales,
                    patch_size=patch_size,
                )
                if patch is None:
                    continue

                cell = float(tile_size) / float(patch["grid"])
                x0 = tile_col * tile_size + int(round(patch["patch_col"] * cell))
                y0 = tile_row * tile_size + int(round(patch["patch_row"] * cell))
                x1 = tile_col * tile_size + int(round((patch["patch_col"] + 1) * cell))
                y1 = tile_row * tile_size + int(round((patch["patch_row"] + 1) * cell))
                records_by_frame[sampled_frame_offset].append(
                    {
                        "sampled_frame_offset": sampled_frame_offset,
                        "tile_sequence_index": tile_sequence_index,
                        "temporal_chunk": temporal_chunk,
                        "frame_in_chunk": frame_in_chunk,
                        "spatial_tile_index": spatial_index,
                        "tile_col": tile_col,
                        "tile_row": tile_row,
                        "scale": patch["scale"],
                        "scale_index": patch["scale_index"],
                        "patch_index_in_scale": patch["patch_index_in_scale"],
                        "patch_row": patch["patch_row"],
                        "patch_col": patch["patch_col"],
                        "bbox": [x0, y0, x1, y1],
                        "color": colors[patch["scale"]],
                    }
                )
            offset += count

    return records_by_frame


def render_overlay_frame(
    frame: Image.Image,
    records: Sequence[dict[str, Any]],
    *,
    alpha: float = 0.35,
) -> Image.Image:
    base = frame.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    fill_alpha = max(0, min(255, int(round(alpha * 255))))
    for record in records:
        color = [int(component) for component in record.get("color", [255, 69, 58])[:3]]
        bbox = [int(value) for value in record["bbox"]]
        draw.rectangle(bbox, fill=tuple(color + [fill_alpha]))
    return Image.alpha_composite(base, overlay).convert("RGB")


def _even_frame(frame: Image.Image) -> Image.Image:
    width, height = frame.size
    even_width = width + (width % 2)
    even_height = height + (height % 2)
    if (even_width, even_height) == frame.size:
        return frame
    return frame.resize((even_width, even_height))


def write_video(
    frames: Iterable[Image.Image],
    path: str | Path,
    *,
    fps: float,
) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_list = [_even_frame(frame.convert("RGB")) for frame in frames]
    if not frame_list:
        raise ValueError("At least one frame is required to write a visualization video.")
    if fps <= 0:
        raise ValueError("fps must be positive")

    width, height = frame_list[0].size
    rate = Fraction(str(float(fps))).limit_denominator(1000)
    container = av.open(str(path), mode="w")
    try:
        stream = container.add_stream("mpeg4", rate=rate)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for frame in frame_list:
            if frame.size != (width, height):
                frame = frame.resize((width, height))
            video_frame = av.VideoFrame.from_image(frame)
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()
    return str(path)


def scale_overlay_records(
    records: Sequence[dict[str, Any]],
    *,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> list[dict[str, Any]]:
    source_width, source_height = source_size
    target_width, target_height = target_size
    if source_width <= 0 or source_height <= 0:
        raise ValueError("source_size must be positive")
    x_scale = target_width / source_width
    y_scale = target_height / source_height
    scaled: list[dict[str, Any]] = []
    for record in records:
        x0, y0, x1, y1 = [int(value) for value in record["bbox"]]
        scaled_record = dict(record)
        scaled_record["source_bbox"] = [x0, y0, x1, y1]
        scaled_record["bbox"] = [
            int(round(x0 * x_scale)),
            int(round(y0 * y_scale)),
            int(round(x1 * x_scale)),
            int(round(y1 * y_scale)),
        ]
        scaled.append(scaled_record)
    return scaled


def write_gaze_visualization_artifacts(
    *,
    selected_frames: Sequence[Image.Image],
    overlay_base_frames: Sequence[Image.Image],
    output_dir: str | Path,
    label: str,
    video: str,
    sampled_frame_indices: Sequence[int],
    gazing_mode: str,
    gazing_info: dict[str, Any] | None,
    spatial_tiles: int,
    grid_cols: int,
    grid_rows: int,
    scales: Sequence[int],
    patch_size: int,
    tile_size: int,
    fps: float,
    alpha: float = 0.35,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    label = safe_label(label)
    selected_path = output_dir / f"{label}_selected_frames.mp4"
    overlay_path = output_dir / f"{label}_autogaze_overlay.mp4"
    processor_path = output_dir / f"{label}_processor_frames.mp4"
    processor_overlay_path = output_dir / f"{label}_processor_autogaze_overlay.mp4"
    gazing_json_path = output_dir / f"{label}_gazing_info.json"

    selected_video = write_video(selected_frames, selected_path, fps=fps)
    processor_video = write_video(overlay_base_frames, processor_path, fps=fps) if overlay_base_frames else None
    overlay_status = "skipped"
    overlay_video = None
    processor_overlay_status = "skipped"
    processor_overlay_video = None
    overlay_records_by_frame: dict[int, list[dict[str, Any]]] = {index: [] for index in range(len(selected_frames))}
    processor_overlay_records_by_frame: dict[int, list[dict[str, Any]]] = {
        index: [] for index in range(len(overlay_base_frames))
    }
    tile_canvas_size = (int(grid_cols) * int(tile_size), int(grid_rows) * int(tile_size))
    overlay_render_size = list(selected_frames[0].size) if selected_frames else None
    processor_overlay_render_size = list(overlay_base_frames[0].size) if overlay_base_frames else None

    if gazing_mode == "autogaze" and gazing_info is not None:
        tile_records_by_frame = build_gaze_overlay_records(
            gazing_info=gazing_info,
            video_index=0,
            num_video_frames=len(selected_frames),
            spatial_tiles=spatial_tiles,
            grid_cols=grid_cols,
            scales=scales,
            patch_size=patch_size,
            tile_size=tile_size,
        )
        overlay_records_by_frame = {
            index: scale_overlay_records(
                tile_records_by_frame.get(index, []),
                source_size=tile_canvas_size,
                target_size=frame.size,
            )
            for index, frame in enumerate(selected_frames)
        }
        processor_overlay_records_by_frame = {
            index: scale_overlay_records(
                tile_records_by_frame.get(index, []),
                source_size=tile_canvas_size,
                target_size=frame.size,
            )
            for index, frame in enumerate(overlay_base_frames)
        }
        overlay_frames = [
            render_overlay_frame(
                frame.convert("RGB"),
                overlay_records_by_frame.get(index, []),
                alpha=alpha,
            )
            for index, frame in enumerate(selected_frames)
        ]
        overlay_video = write_video(overlay_frames, overlay_path, fps=fps)
        overlay_status = "written"
        if overlay_base_frames:
            processor_overlay_frames = [
                render_overlay_frame(
                    frame.convert("RGB"),
                    processor_overlay_records_by_frame.get(index, []),
                    alpha=alpha,
                )
                for index, frame in enumerate(overlay_base_frames)
            ]
            processor_overlay_video = write_video(processor_overlay_frames, processor_overlay_path, fps=fps)
            processor_overlay_status = "written"
        else:
            processor_overlay_status = "skipped_missing_processor_frames"
    elif gazing_mode != "autogaze":
        overlay_status = "skipped_keep_all"
        processor_overlay_status = "skipped_keep_all"
    elif gazing_info is None:
        overlay_status = "skipped_missing_gazing_info"
        processor_overlay_status = "skipped_missing_gazing_info"

    raw = {
        "video": video,
        "label": label,
        "gazing_mode": gazing_mode,
        "sampled_frame_indices": [int(index) for index in sampled_frame_indices],
        "selected_frames_video": selected_video,
        "overlay_video": overlay_video,
        "overlay_status": overlay_status,
        "processor_frames_video": processor_video,
        "processor_overlay_video": processor_overlay_video,
        "processor_overlay_status": processor_overlay_status,
        "spatial_tiles": int(spatial_tiles),
        "grid_cols": int(grid_cols),
        "grid_rows": int(grid_rows),
        "tile_size": int(tile_size),
        "overlay_coordinate_space": "selected_frame",
        "overlay_render_size": overlay_render_size,
        "processor_overlay_coordinate_space": "processor_input_frame",
        "processor_overlay_render_size": processor_overlay_render_size,
        "overlay_tile_canvas_size": [tile_canvas_size[0], tile_canvas_size[1]],
        "overlay_base_frame_count": len(overlay_base_frames),
        "scales": [int(scale) for scale in scales],
        "patch_size": int(patch_size),
        "scale_colors": palette_for_scales(scales),
        "overlay_records_by_frame": {str(key): value for key, value in overlay_records_by_frame.items()},
        "processor_overlay_records_by_frame": {
            str(key): value for key, value in processor_overlay_records_by_frame.items()
        },
        "gazing_info": serialize_gazing_info(gazing_info),
        "note": (
            "Overlay boxes are computed on the processor tile canvas, then mapped onto both selected-frame "
            "and processor-input-frame videos. Thumbnail patches are keep-all in this runner and are saved "
            "in gazing_info when present, but are not drawn on the tile overlay videos."
        ),
    }
    gazing_json_path.write_text(json.dumps(raw, indent=2, sort_keys=True, ensure_ascii=False))

    return {
        "status": "written",
        "selected_frames_video": selected_video,
        "overlay_video": overlay_video,
        "overlay_status": overlay_status,
        "processor_frames_video": processor_video,
        "processor_overlay_video": processor_overlay_video,
        "processor_overlay_status": processor_overlay_status,
        "gazing_info_json": str(gazing_json_path),
        "sampled_frame_count": len(selected_frames),
        "overlay_frame_count": len(selected_frames),
        "overlay_coordinate_space": "selected_frame",
        "overlay_render_size": overlay_render_size,
        "processor_overlay_frame_count": len(overlay_base_frames),
        "processor_overlay_coordinate_space": "processor_input_frame",
        "processor_overlay_render_size": processor_overlay_render_size,
        "scale_colors": palette_for_scales(scales),
    }
