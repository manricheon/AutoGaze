from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from repro.hlvid_example_autogaze import read_video_metadata
from repro.nvila_runner import apply_resize_to_dimensions, load_sampled_video_frames
from repro.plugins.gaze_plan import SparseSelectionPlan, sparse_selection_plan_from_dict
from repro.vjepa_qwen_runner import _scale_bbox_to_frame


DEFAULT_VIDEO = "inputs/hlvid_example/clip_av_video_5_001.mp4"
DEFAULT_PLAN = "outputs/autogaze_repro/qwen_modes_smoke/qwen_chunked_vit_autogaze_sparse_actual_cpu_224_g002_autogaze_sparse_plan.json"
DEFAULT_OUTPUT_DIR = "docs/assets/colab_v2"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build local 16-frame visualization assets for COLAB_VERIFICATION_REPORT_V2_KO.")
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--sparse-plan-json", default=DEFAULT_PLAN)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-video-frames", type=int, default=16)
    parser.add_argument("--video-resize-longest-edge", type=int, default=224)
    parser.add_argument("--video-decode-strategy", choices=["auto", "seek", "scan"], default="seek")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output = build_assets(args)
    print(json.dumps(output, indent=2, sort_keys=True))


def build_assets(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = read_video_metadata(args.video)
    effective = apply_resize_to_dimensions(
        width=int(metadata["width"]),
        height=int(metadata["height"]),
        shortest_edge=None,
        longest_edge=int(args.video_resize_longest_edge),
        exact_width=None,
        exact_height=None,
    )
    resize = {
        "width": int(effective["width"]),
        "height": int(effective["height"]),
        "mode": "longest_edge",
    }
    frames, decode_stats = load_sampled_video_frames(
        args.video,
        int(args.num_video_frames),
        resize,
        decode_strategy=str(args.video_decode_strategy),
    )
    frames = [frame.convert("RGB") for frame in frames]

    selected_grid = output_dir / "hlvid_example_16f_selected_frames.png"
    make_frame_grid(frames).save(selected_grid)

    artifacts: dict[str, Any] = {
        "video": str(args.video),
        "source_metadata": metadata,
        "num_video_frames": int(args.num_video_frames),
        "decode_stats": decode_stats,
        "selected_frames_grid_image": str(selected_grid),
    }
    plan_path = Path(args.sparse_plan_json)
    if plan_path.exists():
        plan = sparse_selection_plan_from_dict(json.loads(plan_path.read_text(encoding="utf-8")))
        overlay = output_dir / "qwen_autogaze_sparse_overlay_16f.png"
        make_sparse_overlay_grid(frames, plan).save(overlay)
        artifacts.update(
            {
                "sparse_plan_json": str(plan_path),
                "autogaze_overlay_image": str(overlay),
                "autogaze_selected_patch_tokens": plan.selected_patch_tokens,
                "autogaze_raw_patch_tokens": plan.raw_patch_tokens,
                "autogaze_overlay_status": "written",
            }
        )
    else:
        artifacts.update(
            {
                "sparse_plan_json": str(plan_path),
                "autogaze_overlay_status": "missing_sparse_plan",
            }
        )

    manifest = output_dir / "manifest.json"
    manifest.write_text(json.dumps(artifacts, indent=2, sort_keys=True), encoding="utf-8")
    artifacts["manifest_json"] = str(manifest)
    return artifacts


def make_frame_grid(frames: list[Image.Image], *, columns: int = 4) -> Image.Image:
    if not frames:
        return Image.new("RGB", (1, 1), color=(255, 255, 255))
    width = max(frame.width for frame in frames)
    height = max(frame.height for frame in frames)
    cols = max(1, min(int(columns), len(frames)))
    rows = (len(frames) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * width, rows * height), color=(18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    for index, frame in enumerate(frames):
        image = frame.resize((width, height))
        x = (index % cols) * width
        y = (index // cols) * height
        canvas.paste(image, (x, y))
        draw.rectangle([x, y, x + 32, y + 18], fill=(0, 0, 0))
        draw.text((x + 4, y + 3), f"{index:02d}", fill=(255, 255, 255))
    return canvas


def make_sparse_overlay_grid(frames: list[Image.Image], plan: SparseSelectionPlan) -> Image.Image:
    selected_by_frame: dict[int, list[Any]] = {}
    for patch in plan.selected_patches:
        selected_by_frame.setdefault(int(patch.frame_order), []).append(patch)
    colors = [
        (255, 64, 64, 90),
        (64, 160, 255, 90),
        (64, 220, 120, 90),
        (255, 190, 64, 90),
        (190, 96, 255, 90),
    ]
    overlays: list[Image.Image] = []
    for order, frame in enumerate(frames):
        base = frame.convert("RGBA")
        layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer, "RGBA")
        for patch in selected_by_frame.get(order, []):
            color = colors[int(patch.scale_id) % len(colors)]
            bbox = _scale_bbox_to_frame(patch.bbox_resized_xyxy, plan, base.size)
            draw.rectangle(bbox, fill=color)
        composite = Image.alpha_composite(base, layer).convert("RGB")
        overlays.append(composite)
    return make_frame_grid(overlays)


if __name__ == "__main__":
    main()
