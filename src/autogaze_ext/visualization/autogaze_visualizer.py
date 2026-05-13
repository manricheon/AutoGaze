from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from autogaze_ext.visualization.base_visualizer import BaseVisualizer


class AutoGazeVisualizer(BaseVisualizer):
    """Dummy AutoGaze patch and scale visualizer."""

    _CATEGORICAL_COLORS = [
        (255, 56, 56),
        (34, 197, 94),
        (59, 130, 246),
        (245, 158, 11),
        (168, 85, 247),
        (20, 184, 166),
    ]
    _GRADIENT_COLORS = [
        (254, 240, 138),  # light yellow
        (251, 146, 60),  # orange
        (244, 114, 182),  # pink
        (168, 85, 247),  # purple
    ]
    _SCALE_ID_FACTORS = {
        0: 1.0 / 7.0,
        1: 2.0 / 7.0,
        2: 0.5,
        3: 1.0,
    }

    def visualize_selected_patches(
        self,
        video: torch.Tensor,
        selected_patch_indices: torch.Tensor | list[int],
        patch_grid: tuple[int, int],
        scales: torch.Tensor | list[int] | None = None,
        mode: str = "autogaze_only",
        prefix: str = "patches",
        overlay_style: str = "mask",
        overlay_alpha: float = 0.35,
        show_patch_indices: bool = False,
        show_scale_labels: bool = False,
        multi_scale_overlay: bool = True,
        scale_color_mode: str = "gradient",
    ) -> list[Path]:
        frames = self._first_video(video)
        output_dir = self._frames_dir(mode)
        frame_indices = list(range(int(frames.shape[0])))
        overlay_frames = self._render_overlay_frames(
            frames,
            selected_patch_indices,
            patch_grid,
            scales=scales,
            sampled_frame_indices=frame_indices,
            overlay_style=overlay_style,
            overlay_alpha=overlay_alpha,
            show_patch_indices=show_patch_indices,
            show_scale_labels=show_scale_labels,
            multi_scale_overlay=multi_scale_overlay,
            scale_color_mode=scale_color_mode,
        )

        paths: list[Path] = []
        for frame_idx, image in enumerate(overlay_frames):
            path = output_dir / f"{prefix}_frame_{frame_idx:03d}.png"
            image.save(path)
            paths.append(path)
        return paths

    def export_autogaze_videos(
        self,
        video: torch.Tensor,
        selected_patch_indices: torch.Tensor | list[int],
        patch_grid: tuple[int, int],
        *,
        scales: torch.Tensor | list[int] | None = None,
        sampled_frame_indices: list[int] | None = None,
        original_frame_count: int | None = None,
        original_resolution: tuple[int, int] | None = None,
        processed_resolution: tuple[int, int] | None = None,
        original_video: torch.Tensor | None = None,
        full_original_video: torch.Tensor | None = None,
        original_fps: float | None = None,
        original_visual_token_count: int | None = None,
        selected_visual_token_count: int | None = None,
        patch_grid_source: str = "provided",
        mode: str = "autogaze_only",
        video_export_mode: str = "sampled_only",
        fps: float = 4.0,
        overlay_alpha: float = 0.35,
        overlay_line_width: int = 2,
        save_overlay_video: bool = True,
        save_side_by_side_video: bool = True,
        save_scale_panel_video: bool = False,
        prefix: str = "autogaze",
        output_video_suffix: str | None = None,
        overlay_style: str = "mask",
        show_patch_boxes: bool | None = None,
        show_patch_indices: bool = False,
        show_scale_labels: bool = False,
        multi_scale_overlay: bool = True,
        scale_color_mode: str = "gradient",
        scale_panel_layout: str = "2x2",
        comparison_layout: str = "processed_overlay",
        info_panel_mode: str = "external",
        scaling_mode: str = "resize",
    ) -> dict[str, Path]:
        if video_export_mode == "hold_last":
            raise NotImplementedError(
                "video_export_mode='hold_last' is not implemented; it requires a policy for carrying "
                "AutoGaze state across unprocessed frames"
            )
        if video_export_mode not in {"sampled_only", "full_length"}:
            raise ValueError("video_export_mode must be sampled_only, full_length, or hold_last")
        if fps <= 0:
            raise ValueError("fps must be > 0")
        if not 0 <= overlay_alpha <= 1:
            raise ValueError("overlay_alpha must be between 0 and 1")
        if overlay_line_width <= 0:
            raise ValueError("overlay_line_width must be > 0")
        if info_panel_mode not in {"external", "inline", "none"}:
            raise ValueError("info_panel_mode must be external, inline, or none")
        if overlay_style not in {"mask", "box", "both"}:
            raise ValueError("overlay_style must be mask, box, or both")
        if scale_color_mode not in {"gradient", "categorical"}:
            raise ValueError("scale_color_mode must be gradient or categorical")
        if scale_panel_layout != "2x2":
            raise NotImplementedError("only scale_panel_layout='2x2' is currently supported")
        if comparison_layout == "chop_overlay":
            raise NotImplementedError("comparison_layout='chop_overlay' is handled by the PoC chop overlay-union path")
        if comparison_layout not in {"processed_overlay", "original_overlay", "original_processed_overlay"}:
            raise ValueError("comparison_layout must be processed_overlay, original_overlay, original_processed_overlay, or chop_overlay")
        if comparison_layout in {"original_overlay", "original_processed_overlay"}:
            if original_video is None:
                raise NotImplementedError("original-space overlay requires original_video frames")
            if scaling_mode not in {"none", "resize", "fit_short_side", "fit_long_side", "quickstart"}:
                raise NotImplementedError(
                    f"original-space overlay for scaling_mode={scaling_mode!r} is not supported here; "
                    "chop coordinates must use the explicit overlay_union path"
                )
        if show_patch_boxes is not None:
            overlay_style = "both" if show_patch_boxes and overlay_style == "mask" else overlay_style
            overlay_style = "mask" if not show_patch_boxes and overlay_style in {"box", "both"} else overlay_style

        frames = self._first_video(video)
        frame_count = int(frames.shape[0])
        sampled = sampled_frame_indices or list(range(frame_count))
        if len(sampled) != frame_count:
            raise ValueError("sampled_frame_indices length must match processed frame count")

        base_dir = self.ensure_dir(mode)
        frames_dir = self._frames_dir(mode)
        videos_dir = base_dir / "videos"
        metadata_dir = base_dir / "metadata"
        videos_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)

        original_frames = frames if comparison_layout == "processed_overlay" else (
            self._first_video(original_video) if original_video is not None else frames
        )
        originals = [self._frame_to_image(frame).convert("RGB") for frame in original_frames]
        if len(originals) != frame_count:
            originals = [self._frame_to_image(frame).convert("RGB") for frame in frames]
        output_fps = original_fps if original_fps is not None and video_export_mode == "full_length" else fps
        scale_metadata = self._scale_metadata(
            scales,
            selected_patch_indices,
            frame_count=frame_count,
            patch_grid=patch_grid,
            multi_scale_overlay=multi_scale_overlay,
            scale_color_mode=scale_color_mode,
        )
        overlay_frames = self._render_overlay_frames(
            frames,
            selected_patch_indices,
            patch_grid,
            scales=scales,
            sampled_frame_indices=sampled,
            original_visual_token_count=original_visual_token_count,
            selected_visual_token_count=selected_visual_token_count,
            overlay_style=overlay_style,
            overlay_alpha=overlay_alpha,
            overlay_line_width=overlay_line_width,
            show_patch_indices=show_patch_indices,
            show_scale_labels=show_scale_labels,
            multi_scale_overlay=multi_scale_overlay,
            scale_color_mode=scale_color_mode,
            info_panel_mode=info_panel_mode,
        )

        artifacts: dict[str, Path] = {}
        for frame_idx, image in enumerate(overlay_frames):
            path = frames_dir / f"{prefix}_overlay_frame_{frame_idx:03d}.png"
            image.save(path)
            if frame_idx == 0:
                artifacts["first_overlay_frame"] = path

        output_video_paths: dict[str, str] = {}
        suffix = f"_{output_video_suffix}" if output_video_suffix else ("_full_length" if video_export_mode == "full_length" else "")
        full_original_frames = self._first_video(full_original_video) if full_original_video is not None else None
        full_length_metadata: dict[str, Any] = {
            "status": "not_requested",
            "processed_frame_indices": sampled,
            "unprocessed_frame_policy": None,
            "exact": None,
        }
        if save_overlay_video:
            overlay_video_path = videos_dir / f"autogaze_overlay{suffix}.mp4"
            export_frames, full_length_metadata = self._apply_video_export_mode(
                overlay_frames,
                sampled_frame_indices=sampled,
                original_frame_count=original_frame_count,
                video_export_mode=video_export_mode,
                full_original_frames=full_original_frames,
                label="AutoGaze Overlay",
            )
            self._write_mp4(overlay_video_path, export_frames, fps=output_fps)
            artifacts["overlay_video"] = overlay_video_path
            output_video_paths["overlay_video"] = str(overlay_video_path)

        if save_side_by_side_video:
            if comparison_layout == "processed_overlay":
                side_by_side_frames = [
                    self._side_by_side_frame(original, overlay)
                    for original, overlay in zip(originals, overlay_frames)
                ]
                side_by_side_path = videos_dir / f"autogaze_side_by_side{suffix}.mp4"
                export_frames, full_length_metadata = self._apply_video_export_mode(
                    side_by_side_frames,
                    sampled_frame_indices=sampled,
                    original_frame_count=original_frame_count,
                    video_export_mode=video_export_mode,
                    full_original_frames=full_original_frames,
                    label="Processed / Overlay",
                )
                self._write_mp4(side_by_side_path, export_frames, fps=output_fps)
                artifacts["side_by_side_video"] = side_by_side_path
                output_video_paths["side_by_side_video"] = str(side_by_side_path)
            else:
                original_overlay_frames = self._render_original_overlay_frames(
                    original_frames,
                    selected_patch_indices,
                    patch_grid,
                    processed_resolution=processed_resolution or (int(frames.shape[-2]), int(frames.shape[-1])),
                    scales=scales,
                    sampled_frame_indices=sampled,
                    original_visual_token_count=original_visual_token_count,
                    selected_visual_token_count=selected_visual_token_count,
                    overlay_style=overlay_style,
                    overlay_alpha=overlay_alpha,
                    overlay_line_width=overlay_line_width,
                    show_patch_indices=show_patch_indices,
                    show_scale_labels=show_scale_labels,
                    multi_scale_overlay=multi_scale_overlay,
                    scale_color_mode=scale_color_mode,
                    info_panel_mode=info_panel_mode,
                )
                if comparison_layout == "original_overlay":
                    original_overlay_path = videos_dir / f"autogaze_original_overlay{suffix}.mp4"
                    export_frames, full_length_metadata = self._apply_video_export_mode(
                        original_overlay_frames,
                        sampled_frame_indices=sampled,
                        original_frame_count=original_frame_count,
                        video_export_mode=video_export_mode,
                        full_original_frames=full_original_frames,
                        label="Original Overlay",
                    )
                    self._write_mp4(original_overlay_path, export_frames, fps=output_fps)
                    artifacts["original_overlay_video"] = original_overlay_path
                    output_video_paths["original_overlay_video"] = str(original_overlay_path)
                elif comparison_layout == "original_processed_overlay":
                    comparison_frames = [
                        self._side_by_side_frame(original_overlay, processed_overlay)
                        for original_overlay, processed_overlay in zip(original_overlay_frames, overlay_frames)
                    ]
                    comparison_path = videos_dir / f"autogaze_original_processed_overlay{suffix}.mp4"
                    export_frames, full_length_metadata = self._apply_video_export_mode(
                        comparison_frames,
                        sampled_frame_indices=sampled,
                        original_frame_count=original_frame_count,
                        video_export_mode=video_export_mode,
                        full_original_frames=full_original_frames,
                        label="Original / Processed Overlay",
                    )
                    self._write_mp4(comparison_path, export_frames, fps=output_fps)
                    artifacts["original_processed_overlay_video"] = comparison_path
                    output_video_paths["original_processed_overlay_video"] = str(comparison_path)

        if save_scale_panel_video:
            scale_panel_frames = self._render_scale_panel_frames(
                frames,
                selected_patch_indices,
                patch_grid,
                scales=scales,
                sampled_frame_indices=sampled,
                original_visual_token_count=original_visual_token_count,
                selected_visual_token_count=selected_visual_token_count,
                overlay_style=overlay_style,
                overlay_alpha=overlay_alpha,
                overlay_line_width=overlay_line_width,
                show_patch_indices=show_patch_indices,
                show_scale_labels=show_scale_labels,
                multi_scale_overlay=multi_scale_overlay,
                scale_color_mode=scale_color_mode,
                scale_panel_layout=scale_panel_layout,
                info_panel_mode=info_panel_mode,
            )
            scale_panels_dir = self._scale_panels_dir(mode)
            for frame_idx, image in enumerate(scale_panel_frames):
                path = scale_panels_dir / f"{prefix}_scale_panel_frame_{frame_idx:03d}.png"
                image.save(path)
                if frame_idx == 0:
                    artifacts["first_scale_panel_frame"] = path
            scale_panel_path = videos_dir / f"autogaze_scale_panels{suffix}.mp4"
            export_frames, full_length_metadata = self._apply_video_export_mode(
                scale_panel_frames,
                sampled_frame_indices=sampled,
                original_frame_count=original_frame_count,
                video_export_mode=video_export_mode,
                full_original_frames=full_original_frames,
                label="Scale Panels",
            )
            self._write_mp4(scale_panel_path, export_frames, fps=output_fps)
            legacy_scale_panel_path = videos_dir / f"autogaze_scale_panel{suffix}.mp4"
            if legacy_scale_panel_path != scale_panel_path:
                self._write_mp4(legacy_scale_panel_path, export_frames, fps=output_fps)
            artifacts["scale_panel_video"] = scale_panel_path
            output_video_paths["scale_panel_video"] = str(scale_panel_path)

        processed_h, processed_w = int(frames.shape[-2]), int(frames.shape[-1])
        processed_resolution = processed_resolution or (processed_h, processed_w)
        original_resolution = original_resolution or processed_resolution
        grid_h, grid_w = self._validate_patch_grid(patch_grid)
        patch_size = (processed_resolution[0] / grid_h, processed_resolution[1] / grid_w)
        reduction = None
        if original_visual_token_count and selected_visual_token_count is not None:
            reduction = 1.0 - (float(selected_visual_token_count) / float(original_visual_token_count))

        metadata_path = metadata_dir / "visualization_video_metadata.json"
        metadata = {
            "video_export_mode": video_export_mode,
            "fps": fps,
            "original_fps": original_fps,
            "output_fps": output_fps,
            "sampled_frame_indices": sampled,
            "original_frame_count": original_frame_count,
            "processed_frame_count": frame_count,
            "original_resolution": list(original_resolution),
            "processed_resolution": list(processed_resolution),
            "patch_grid": [grid_h, grid_w],
            "patch_grid_source": patch_grid_source,
            "patch_size": [patch_size[0], patch_size[1]],
            "original_visual_token_count": original_visual_token_count,
            "selected_visual_token_count": selected_visual_token_count,
            "token_reduction_ratio": reduction,
            "output_video_paths": output_video_paths,
            "overlay_style": overlay_style,
            "show_patch_boxes": overlay_style in {"box", "both"},
            "show_patch_indices": show_patch_indices,
            "show_scale_labels": show_scale_labels,
            "multi_scale_overlay": multi_scale_overlay,
            "scale_color_mode": scale_color_mode,
            "scale_color_map": scale_metadata["scale_color_map"],
            "available_scale_ids": scale_metadata["available_scale_ids"],
            "missing_scale_metadata": scale_metadata["missing_scale_metadata"],
            "scale_panel_layout": scale_panel_layout,
            "comparison_layout": comparison_layout,
            "info_panel_mode": info_panel_mode,
            "coordinate_mapping": self._coordinate_mapping_metadata(
                original_resolution=original_resolution,
                processed_resolution=processed_resolution,
                scaling_mode=scaling_mode,
                comparison_layout=comparison_layout,
            ),
            "full_length_export": full_length_metadata,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        artifacts["video_metadata"] = metadata_path
        return artifacts

    def visualize_scale_indicators(
        self,
        scales: torch.Tensor | list[int],
        mode: str = "autogaze_only",
        filename: str = "scale_indicators.txt",
    ) -> Path:
        output_dir = self.ensure_dir(mode)
        path = output_dir / filename
        counts: dict[str, int] = {}
        for scale in self._as_list(scales):
            key = str(int(scale))
            counts[key] = counts.get(key, 0) + 1
        path.write_text("\n".join(f"{key}: {value}" for key, value in sorted(counts.items())), encoding="utf-8")
        return path

    def visualize(
        self,
        video: torch.Tensor,
        selected_patch_indices: torch.Tensor | list[int],
        patch_grid: tuple[int, int],
        scales: torch.Tensor | list[int] | None = None,
        **metadata: Any,
    ) -> list[Path]:
        paths = self.visualize_selected_patches(video, selected_patch_indices, patch_grid, scales=scales)
        if scales is not None:
            paths.append(self.visualize_scale_indicators(scales))
        return paths

    def _frames_dir(self, mode: str) -> Path:
        path = self.ensure_dir(mode) / "frames"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _scale_panels_dir(self, mode: str) -> Path:
        path = self.ensure_dir(mode) / "scale_panels"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _validate_patch_grid(patch_grid: tuple[int, int]) -> tuple[int, int]:
        grid_h, grid_w = int(patch_grid[0]), int(patch_grid[1])
        if grid_h <= 0 or grid_w <= 0:
            raise ValueError("patch_grid values must be > 0")
        return grid_h, grid_w

    def _render_overlay_frames(
        self,
        frames: torch.Tensor,
        selected_patch_indices: torch.Tensor | list[int],
        patch_grid: tuple[int, int],
        *,
        scales: torch.Tensor | list[int] | None = None,
        sampled_frame_indices: list[int] | None = None,
        original_visual_token_count: int | None = None,
        selected_visual_token_count: int | None = None,
        overlay_style: str = "mask",
        overlay_alpha: float = 0.35,
        overlay_line_width: int = 2,
        show_patch_indices: bool = False,
        show_scale_labels: bool = False,
        multi_scale_overlay: bool = True,
        scale_color_mode: str = "gradient",
        info_panel_mode: str = "external",
        scale_values_for_layout: list[int] | None = None,
    ) -> list[Image.Image]:
        grid_h, grid_w = self._validate_patch_grid(patch_grid)
        frame_count = int(frames.shape[0])
        sampled = sampled_frame_indices or list(range(frame_count))
        selected_by_frame = self._patches_by_frame(selected_patch_indices, frame_count=frame_count, patch_grid=(grid_h, grid_w))
        scales_by_frame = self._scales_by_frame(scales, selected_by_frame)
        scale_values_for_layout = scale_values_for_layout or sorted({scale for row in scales_by_frame for scale in row})
        scale_color_map = self._build_scale_color_map(
            scales_by_frame,
            multi_scale_overlay=multi_scale_overlay,
            scale_color_mode=scale_color_mode,
        )

        images: list[Image.Image] = []
        for frame_idx, frame in enumerate(frames):
            image = self._frame_to_image(frame).convert("RGB")
            overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            patches = selected_by_frame[frame_idx]
            frame_scales = scales_by_frame[frame_idx]
            for i, patch_idx in enumerate(patches):
                if patch_idx < 0 or patch_idx >= grid_h * grid_w:
                    raise ValueError("selected_patch_indices contain values outside patch_grid")
                scale_value = frame_scales[i] if i < len(frame_scales) else None
                x0n, y0n, x1n, y1n = self._scale_aware_patch_box(
                    patch_idx,
                    (grid_h, grid_w),
                    scale_value,
                    scale_values_for_layout=scale_values_for_layout,
                )
                x0, y0 = x0n * image.width, y0n * image.height
                x1, y1 = x1n * image.width, y1n * image.height
                color = self._color_for_scale(scale_value, scale_color_map=scale_color_map)
                fill = (*color, int(255 * overlay_alpha)) if overlay_style in {"mask", "both"} else None
                outline = (*color, 255) if overlay_style in {"box", "both"} else None
                width = overlay_line_width if outline is not None else 1
                draw.rectangle((x0, y0, x1, y1), fill=fill, outline=outline, width=width)
                if show_patch_indices or show_scale_labels:
                    labels: list[str] = []
                    if show_patch_indices:
                        labels.append(str(patch_idx))
                    if show_scale_labels and scale_value is not None:
                        labels.append(f"s{scale_value}")
                    label = "@".join(labels)
                    draw.text((x0 + 2, y0 + 2), label, fill=(255, 255, 255, 255))

            composited = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
            composited = self._draw_frame_text(
                composited,
                frame_idx=frame_idx,
                sampled_frame_index=sampled[frame_idx],
                selected_count=len(patches),
                original_visual_token_count=original_visual_token_count,
                selected_visual_token_count=selected_visual_token_count,
                info_panel_mode=info_panel_mode,
            )
            images.append(composited)
        return images

    def _render_scale_panel_frames(
        self,
        frames: torch.Tensor,
        selected_patch_indices: torch.Tensor | list[int],
        patch_grid: tuple[int, int],
        *,
        scales: torch.Tensor | list[int] | None = None,
        sampled_frame_indices: list[int] | None = None,
        original_visual_token_count: int | None = None,
        selected_visual_token_count: int | None = None,
        overlay_style: str = "mask",
        overlay_alpha: float = 0.35,
        overlay_line_width: int = 2,
        show_patch_indices: bool = False,
        show_scale_labels: bool = False,
        multi_scale_overlay: bool = True,
        scale_color_mode: str = "gradient",
        scale_panel_layout: str = "2x2",
        info_panel_mode: str = "external",
    ) -> list[Image.Image]:
        grid_h, grid_w = self._validate_patch_grid(patch_grid)
        frame_count = int(frames.shape[0])
        selected_by_frame = self._patches_by_frame(selected_patch_indices, frame_count=frame_count, patch_grid=(grid_h, grid_w))
        scales_by_frame = self._scales_by_frame(scales, selected_by_frame)
        scale_values = sorted({scale for row in scales_by_frame for scale in row})
        if not scale_values:
            scale_values = [0]

        panels: list[Image.Image] = []
        for scale_value in scale_values:
            filtered_rows: list[list[int]] = []
            for patches, frame_scales in zip(selected_by_frame, scales_by_frame):
                if frame_scales:
                    filtered_rows.append([patch for patch, scale in zip(patches, frame_scales) if scale == scale_value])
                else:
                    filtered_rows.append(patches if scale_value == 0 else [])
            panel = self._render_overlay_frames(
                frames,
                filtered_rows,
                patch_grid,
                scales=[[scale_value] * len(row) for row in filtered_rows],
                sampled_frame_indices=sampled_frame_indices,
                original_visual_token_count=original_visual_token_count,
                selected_visual_token_count=selected_visual_token_count,
                overlay_style=overlay_style,
                overlay_alpha=overlay_alpha,
                overlay_line_width=overlay_line_width,
                show_patch_indices=show_patch_indices,
                show_scale_labels=show_scale_labels,
                multi_scale_overlay=multi_scale_overlay,
                scale_color_mode=scale_color_mode,
                info_panel_mode=info_panel_mode,
                scale_values_for_layout=scale_values,
            )
            for image in panel:
                draw = ImageDraw.Draw(image)
                bbox = draw.textbbox((6, 6), f"scale {scale_value}")
                draw.rectangle((bbox[0] - 4, bbox[1] - 4, bbox[2] + 4, bbox[3] + 4), fill=(0, 0, 0))
                draw.text((6, 6), f"scale {scale_value}", fill=(255, 255, 255))
            panels.append(panel)

        output_frames: list[Image.Image] = []
        for frame_idx in range(frame_count):
            row_images = [panel[frame_idx].convert("RGB") for panel in panels]
            if scale_panel_layout == "2x2" and len(row_images) <= 4:
                output_frames.append(self._grid_2x2(row_images))
            else:
                output_frames.append(self._concat_h(row_images))
        return output_frames

    def _render_original_overlay_frames(
        self,
        original_frames: torch.Tensor,
        selected_patch_indices: torch.Tensor | list[int],
        patch_grid: tuple[int, int],
        *,
        processed_resolution: tuple[int, int],
        scales: torch.Tensor | list[int] | None = None,
        sampled_frame_indices: list[int] | None = None,
        original_visual_token_count: int | None = None,
        selected_visual_token_count: int | None = None,
        overlay_style: str = "mask",
        overlay_alpha: float = 0.35,
        overlay_line_width: int = 2,
        show_patch_indices: bool = False,
        show_scale_labels: bool = False,
        multi_scale_overlay: bool = True,
        scale_color_mode: str = "gradient",
        info_panel_mode: str = "external",
        scale_values_for_layout: list[int] | None = None,
    ) -> list[Image.Image]:
        grid_h, grid_w = self._validate_patch_grid(patch_grid)
        frame_count = int(original_frames.shape[0])
        sampled = sampled_frame_indices or list(range(frame_count))
        selected_by_frame = self._patches_by_frame(selected_patch_indices, frame_count=frame_count, patch_grid=(grid_h, grid_w))
        scales_by_frame = self._scales_by_frame(scales, selected_by_frame)
        scale_values_for_layout = scale_values_for_layout or sorted({scale for row in scales_by_frame for scale in row})
        scale_color_map = self._build_scale_color_map(
            scales_by_frame,
            multi_scale_overlay=multi_scale_overlay,
            scale_color_mode=scale_color_mode,
        )
        processed_h, processed_w = int(processed_resolution[0]), int(processed_resolution[1])
        if processed_h <= 0 or processed_w <= 0:
            raise ValueError("processed_resolution must be positive for original-space overlay")

        images: list[Image.Image] = []
        for frame_idx, frame in enumerate(original_frames):
            image = self._frame_to_image(frame).convert("RGB")
            overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            patches = selected_by_frame[frame_idx]
            frame_scales = scales_by_frame[frame_idx]
            for i, patch_idx in enumerate(patches):
                if patch_idx < 0 or patch_idx >= grid_h * grid_w:
                    raise ValueError("selected_patch_indices contain values outside patch_grid")
                scale_value = frame_scales[i] if i < len(frame_scales) else None
                x0n, y0n, x1n, y1n = self._scale_aware_patch_box(
                    patch_idx,
                    (grid_h, grid_w),
                    scale_value,
                    scale_values_for_layout=scale_values_for_layout,
                )
                x0, y0 = x0n * image.width, y0n * image.height
                x1, y1 = x1n * image.width, y1n * image.height
                color = self._color_for_scale(scale_value, scale_color_map=scale_color_map)
                fill = (*color, int(255 * overlay_alpha)) if overlay_style in {"mask", "both"} else None
                outline = (*color, 255) if overlay_style in {"box", "both"} else None
                width = overlay_line_width if outline is not None else 1
                draw.rectangle((x0, y0, x1, y1), fill=fill, outline=outline, width=width)
                if show_patch_indices or show_scale_labels:
                    labels: list[str] = []
                    if show_patch_indices:
                        labels.append(str(patch_idx))
                    if show_scale_labels and scale_value is not None:
                        labels.append(f"s{scale_value}")
                    draw.text((x0 + 2, y0 + 2), "@".join(labels), fill=(255, 255, 255, 255))
            composited = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
            composited = self._draw_frame_text(
                composited,
                frame_idx=frame_idx,
                sampled_frame_index=sampled[frame_idx],
                selected_count=len(patches),
                original_visual_token_count=original_visual_token_count,
                selected_visual_token_count=selected_visual_token_count,
                info_panel_mode=info_panel_mode,
            )
            images.append(composited)
        return images

    def _apply_video_export_mode(
        self,
        rendered_frames: list[Image.Image],
        *,
        sampled_frame_indices: list[int],
        original_frame_count: int | None,
        video_export_mode: str,
        full_original_frames: torch.Tensor | None,
        label: str,
    ) -> tuple[list[Image.Image], dict[str, Any]]:
        if video_export_mode == "sampled_only":
            return rendered_frames, {
                "status": "sampled_only",
                "processed_frame_indices": sampled_frame_indices,
                "unprocessed_frame_policy": "not_applicable",
                "exact": True,
            }
        if video_export_mode != "full_length":
            raise NotImplementedError(f"video_export_mode={video_export_mode!r} is not supported here")
        if original_frame_count is None:
            raise NotImplementedError("full_length export requires original_frame_count metadata")
        if original_frame_count <= 0:
            raise ValueError("original_frame_count must be > 0")
        if len(rendered_frames) != len(sampled_frame_indices):
            raise ValueError("rendered frame count must match sampled_frame_indices length")

        template = rendered_frames[0].convert("RGB")
        full_frames = [self._unprocessed_frame(template, frame_idx, full_original_frames, label) for frame_idx in range(original_frame_count)]
        duplicate_policy = "last_overlay_wins"
        for rendered, source_idx in zip(rendered_frames, sampled_frame_indices):
            if source_idx < 0 or source_idx >= original_frame_count:
                raise ValueError("sampled_frame_indices contain values outside original frame count")
            full_frames[source_idx] = rendered.convert("RGB")
        exact = full_original_frames is not None and int(full_original_frames.shape[0]) >= original_frame_count
        return full_frames, {
            "status": "implemented",
            "processed_frame_indices": sampled_frame_indices,
            "unprocessed_frame_policy": "original_frame_if_available_else_black_frame",
            "duplicate_processed_frame_policy": duplicate_policy,
            "original_frame_count": original_frame_count,
            "output_frame_count": len(full_frames),
            "exact": exact,
            "approximate_reason": None if exact else "full original frame tensor was unavailable; black placeholders used for unprocessed frames",
        }

    def _unprocessed_frame(
        self,
        template: Image.Image,
        frame_idx: int,
        full_original_frames: torch.Tensor | None,
        label: str,
    ) -> Image.Image:
        if full_original_frames is not None and frame_idx < int(full_original_frames.shape[0]):
            image = self._frame_to_image(full_original_frames[frame_idx]).convert("RGB")
            if image.size != template.size:
                image = image.resize(template.size)
        else:
            image = Image.new("RGB", template.size, color=(0, 0, 0))
        draw = ImageDraw.Draw(image)
        text = f"{label} | frame {frame_idx} | unprocessed"
        bbox = draw.textbbox((6, 6), text)
        draw.rectangle((bbox[0] - 4, bbox[1] - 4, bbox[2] + 4, bbox[3] + 4), fill=(0, 0, 0))
        draw.text((6, 6), text, fill=(255, 255, 255))
        return image

    @staticmethod
    def _coordinate_mapping_metadata(
        *,
        original_resolution: tuple[int, int],
        processed_resolution: tuple[int, int],
        scaling_mode: str,
        comparison_layout: str,
    ) -> dict[str, Any]:
        original_h, original_w = int(original_resolution[0]), int(original_resolution[1])
        processed_h, processed_w = int(processed_resolution[0]), int(processed_resolution[1])
        supported = comparison_layout in {"processed_overlay", "original_overlay", "original_processed_overlay"} and scaling_mode in {
            "none",
            "resize",
            "fit_short_side",
            "fit_long_side",
            "quickstart",
        }
        return {
            "mode": "processed_to_original_affine" if comparison_layout != "processed_overlay" else "processed_frame_coordinates",
            "scaling_mode": scaling_mode,
            "original_resolution": [original_h, original_w],
            "processed_resolution": [processed_h, processed_w],
            "scale_factors": {
                "x": original_w / float(processed_w) if processed_w else None,
                "y": original_h / float(processed_h) if processed_h else None,
            },
            "padding": {"left": 0, "top": 0, "right": 0, "bottom": 0},
            "crop_offsets": {"x": 0, "y": 0},
            "chop_offsets": None,
            "mapping_exact": supported,
            "unsupported_reason": None if supported else f"comparison_layout={comparison_layout} is not supported for scaling_mode={scaling_mode}",
        }

    def _patches_by_frame(
        self,
        selected_patch_indices: torch.Tensor | list[int],
        *,
        frame_count: int,
        patch_grid: tuple[int, int],
    ) -> list[list[int]]:
        patches_per_frame = patch_grid[0] * patch_grid[1]
        raw = selected_patch_indices.detach().cpu() if isinstance(selected_patch_indices, torch.Tensor) else selected_patch_indices
        if isinstance(raw, torch.Tensor):
            if raw.ndim == 3:
                raw = raw[0]
            if raw.ndim == 2 and raw.shape[0] == frame_count:
                return [[int(value) for value in row.flatten().tolist()] for row in raw]
            values = [int(value) for value in raw.flatten().tolist()]
        elif raw and isinstance(raw, (list, tuple)) and all(isinstance(row, (list, tuple)) for row in raw):
            rows = [[int(value) for value in row] for row in raw]  # type: ignore[arg-type]
            if len(rows) == frame_count:
                return rows
            values = [value for row in rows for value in row]
        else:
            values = [int(value) for value in self._as_list(raw)]

        if any(value >= patches_per_frame for value in values):
            grouped = [[] for _ in range(frame_count)]
            for value in values:
                if value < 0:
                    raise ValueError("selected_patch_indices contain negative values")
                frame_idx, patch_idx = divmod(value, patches_per_frame)
                if frame_idx < frame_count:
                    grouped[frame_idx].append(patch_idx)
            return grouped
        return [list(values) for _ in range(frame_count)]

    def _scales_by_frame(self, scales: torch.Tensor | list[int] | None, selected_by_frame: list[list[int]]) -> list[list[int]]:
        if scales is None:
            return [[] for _ in selected_by_frame]
        raw = scales.detach().cpu() if isinstance(scales, torch.Tensor) else scales
        if isinstance(raw, torch.Tensor):
            if raw.ndim == 3:
                raw = raw[0]
            if raw.ndim == 2 and raw.shape[0] == len(selected_by_frame):
                return [[int(value) for value in row.flatten().tolist()] for row in raw]
            values = [int(value) for value in raw.flatten().tolist()]
        elif raw and isinstance(raw, (list, tuple)) and all(isinstance(row, (list, tuple)) for row in raw):
            rows = [[int(value) for value in row] for row in raw]  # type: ignore[arg-type]
            if len(rows) == len(selected_by_frame):
                return rows
            values = [value for row in rows for value in row]
        else:
            values = [int(value) for value in self._as_list(raw)]

        result: list[list[int]] = []
        cursor = 0
        for patches in selected_by_frame:
            count = len(patches)
            result.append(values[cursor : cursor + count])
            cursor += count
        return result

    def _scale_metadata(
        self,
        scales: torch.Tensor | list[int] | None,
        selected_patch_indices: torch.Tensor | list[int],
        *,
        frame_count: int,
        patch_grid: tuple[int, int],
        multi_scale_overlay: bool,
        scale_color_mode: str,
    ) -> dict[str, Any]:
        selected = self._patches_by_frame(selected_patch_indices, frame_count=frame_count, patch_grid=patch_grid)
        scales_by_frame = self._scales_by_frame(scales, selected)
        color_map = self._build_scale_color_map(
            scales_by_frame,
            multi_scale_overlay=multi_scale_overlay,
            scale_color_mode=scale_color_mode,
        )
        return {
            "available_scale_ids": sorted({scale for row in scales_by_frame for scale in row}),
            "missing_scale_metadata": not any(scales_by_frame),
            "scale_color_map": {
                str(scale): list(color)
                for scale, color in color_map.items()
            },
        }

    @classmethod
    def _scale_aware_patch_box(
        cls,
        patch_idx: int,
        patch_grid: tuple[int, int],
        scale_value: int | None,
        *,
        scale_values_for_layout: list[int] | None = None,
    ) -> tuple[float, float, float, float]:
        grid_h, grid_w = patch_grid
        row, col = divmod(int(patch_idx), grid_w)
        scale_h, scale_w = cls._scale_grid_for_value(
            scale_value,
            patch_grid,
            scale_values_for_layout=scale_values_for_layout,
        )
        if scale_h == grid_h and scale_w == grid_w:
            return col / grid_w, row / grid_h, (col + 1) / grid_w, (row + 1) / grid_h

        scale_row = min(scale_h - 1, int(row * scale_h / grid_h))
        scale_col = min(scale_w - 1, int(col * scale_w / grid_w))
        return (
            scale_col / scale_w,
            scale_row / scale_h,
            (scale_col + 1) / scale_w,
            (scale_row + 1) / scale_h,
        )

    @classmethod
    def _scale_grid_for_value(
        cls,
        scale_value: int | None,
        patch_grid: tuple[int, int],
        *,
        scale_values_for_layout: list[int] | None = None,
    ) -> tuple[int, int]:
        grid_h, grid_w = patch_grid
        factor = cls._scale_factor_for_value(scale_value, scale_values_for_layout=scale_values_for_layout)
        if factor is None:
            return grid_h, grid_w
        return max(1, int(round(grid_h * factor))), max(1, int(round(grid_w * factor)))

    @classmethod
    def _scale_factor_for_value(
        cls,
        scale_value: int | None,
        *,
        scale_values_for_layout: list[int] | None = None,
    ) -> float | None:
        if scale_value is None:
            return None
        scale = int(scale_value)
        context = sorted({int(value) for value in scale_values_for_layout or []})
        if context and max(context) <= 3 and scale in cls._SCALE_ID_FACTORS:
            return cls._SCALE_ID_FACTORS[scale]
        if context and max(context) > 3 and scale in context:
            context_set = set(context)
            canonical_scales = {32, 64, 112, 224}
            quickstart_scales = {56, 112, 196, 392}
            if context_set.issubset(canonical_scales) and scale in canonical_scales:
                return scale / 224.0
            if context_set.issubset(quickstart_scales) and scale in quickstart_scales:
                return scale / 392.0
            max_scale = max(context)
            return scale / float(max_scale) if max_scale > 0 else None
        if scale in cls._SCALE_ID_FACTORS:
            return cls._SCALE_ID_FACTORS[scale]
        if scale in {32, 64, 112, 224}:
            return scale / 224.0
        if scale in {56, 196, 392}:
            return scale / 392.0
        return None

    def _build_scale_color_map(
        self,
        scales_by_frame: list[list[int]],
        *,
        multi_scale_overlay: bool,
        scale_color_mode: str,
    ) -> dict[int, tuple[int, int, int]]:
        if not multi_scale_overlay:
            return {}
        values = sorted({int(scale) for row in scales_by_frame for scale in row})
        palette = self._GRADIENT_COLORS if scale_color_mode == "gradient" else self._CATEGORICAL_COLORS
        return {scale: palette[index % len(palette)] for index, scale in enumerate(values)}

    def _color_for_scale(
        self,
        scale: int | None,
        *,
        scale_color_map: dict[int, tuple[int, int, int]] | None = None,
    ) -> tuple[int, int, int]:
        if scale is None:
            return self._GRADIENT_COLORS[0]
        if scale_color_map and int(scale) in scale_color_map:
            return scale_color_map[int(scale)]
        return self._GRADIENT_COLORS[0]

    @staticmethod
    def _draw_frame_text(
        image: Image.Image,
        *,
        frame_idx: int,
        sampled_frame_index: int,
        selected_count: int,
        original_visual_token_count: int | None,
        selected_visual_token_count: int | None,
        info_panel_mode: str = "external",
    ) -> Image.Image:
        if info_panel_mode == "none":
            return image
        lines = [f"frame {frame_idx} | source index {sampled_frame_index}", f"selected patches: {selected_count}"]
        if original_visual_token_count is not None and selected_visual_token_count is not None:
            lines.append(f"tokens: {selected_visual_token_count}/{original_visual_token_count}")
        text = "\n".join(lines)
        pad = 4
        if info_panel_mode == "external":
            scratch = Image.new("RGB", (1, 1))
            draw = ImageDraw.Draw(scratch)
            bbox = draw.multiline_textbbox((0, 0), text)
            panel_h = (bbox[3] - bbox[1]) + pad * 4
            canvas = Image.new("RGB", (image.width, image.height + panel_h), color=(0, 0, 0))
            canvas.paste(image, (0, 0))
            draw = ImageDraw.Draw(canvas)
            draw.multiline_text((pad * 2, image.height + pad * 2), text, fill=(255, 255, 255))
            return canvas
        draw = ImageDraw.Draw(image)
        bbox = draw.multiline_textbbox((6, 6), text)
        draw.rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), fill=(0, 0, 0))
        draw.multiline_text((6, 6), text, fill=(255, 255, 255))
        return image

    @staticmethod
    def _side_by_side_frame(original: Image.Image, overlay: Image.Image) -> Image.Image:
        original = original.convert("RGB")
        overlay = overlay.convert("RGB")
        if original.size != overlay.size:
            overlay = overlay.resize(original.size)
        width, height = original.size
        canvas = Image.new("RGB", (width * 2, height), color=(0, 0, 0))
        canvas.paste(original, (0, 0))
        canvas.paste(overlay, (width, 0))
        draw = ImageDraw.Draw(canvas)
        for x0, label in [(0, "Original / Processed"), (width, "AutoGaze Overlay")]:
            bbox = draw.textbbox((x0 + 6, 6), label)
            draw.rectangle((bbox[0] - 4, bbox[1] - 4, bbox[2] + 4, bbox[3] + 4), fill=(0, 0, 0))
            draw.text((x0 + 6, 6), label, fill=(255, 255, 255))
        return canvas

    @staticmethod
    def _grid_2x2(images: list[Image.Image]) -> Image.Image:
        if not images:
            raise ValueError("cannot build scale panel with no images")
        prepared = [image.convert("RGB") for image in images]
        width = max(image.width for image in prepared)
        height = max(image.height for image in prepared)
        normalized: list[Image.Image] = []
        for image in prepared:
            canvas = Image.new("RGB", (width, height), color=(0, 0, 0))
            canvas.paste(image, (0, 0))
            normalized.append(canvas)
        while len(normalized) < 4:
            normalized.append(Image.new("RGB", (width, height), color=(0, 0, 0)))
        canvas = Image.new("RGB", (width * 2, height * 2), color=(0, 0, 0))
        for index, image in enumerate(normalized[:4]):
            x = (index % 2) * width
            y = (index // 2) * height
            canvas.paste(image, (x, y))
        return canvas

    @staticmethod
    def _concat_h(images: list[Image.Image]) -> Image.Image:
        if not images:
            raise ValueError("cannot concatenate no images")
        heights = [image.height for image in images]
        max_height = max(heights)
        prepared: list[Image.Image] = []
        for image in images:
            image = image.convert("RGB")
            if image.height != max_height:
                canvas = Image.new("RGB", (image.width, max_height), color=(0, 0, 0))
                canvas.paste(image, (0, 0))
                image = canvas
            prepared.append(image)
        canvas = Image.new("RGB", (sum(image.width for image in prepared), max_height), color=(0, 0, 0))
        cursor = 0
        for image in prepared:
            canvas.paste(image, (cursor, 0))
            cursor += image.width
        return canvas

    @staticmethod
    def _even_sized(image: Image.Image) -> Image.Image:
        width, height = image.size
        even_width = width if width % 2 == 0 else width + 1
        even_height = height if height % 2 == 0 else height + 1
        if (even_width, even_height) == image.size:
            return image.convert("RGB")
        canvas = Image.new("RGB", (even_width, even_height), color=(0, 0, 0))
        canvas.paste(image.convert("RGB"), (0, 0))
        return canvas

    @classmethod
    def _write_mp4(cls, path: Path, frames: list[Image.Image], *, fps: float) -> None:
        if not frames:
            raise ValueError("cannot write a video with no frames")
        try:
            import av  # type: ignore
        except ImportError as exc:
            raise ImportError("PyAV is required for AutoGaze MP4 visualization export") from exc

        rate = Fraction(fps).limit_denominator(1000)
        prepared = [cls._even_sized(frame) for frame in frames]
        width, height = prepared[0].size
        path.parent.mkdir(parents=True, exist_ok=True)
        with av.open(str(path), mode="w") as container:
            stream = container.add_stream("mpeg4", rate=rate)
            stream.width = width
            stream.height = height
            stream.pix_fmt = "yuv420p"
            for image in prepared:
                frame = av.VideoFrame.from_ndarray(np.asarray(image), format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
