from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def get_uniform_frame_indices(max_frame: int, num_segments: int) -> list[int]:
    if num_segments <= 0:
        return []
    seg_size = float(max_frame + 1) / num_segments
    return [int((seg_size / 2) + round(seg_size * idx)) for idx in range(num_segments)]


def build_video_question(num_patches_list: list[int], prompt: str) -> str:
    video_prefix = "".join([f"Frame{i + 1}: <image>\n" for i in range(len(num_patches_list))])
    return video_prefix + prompt


def run_internvl3_off(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import torch
        import torchvision.transforms as transforms
        from decord import VideoReader, cpu
        from PIL import Image
        from torchvision.transforms.functional import InterpolationMode
        from transformers import AutoModel, AutoTokenizer
    except ModuleNotFoundError as exc:
        return {
            "status": "failed_missing_dependency",
            "answer": None,
            "error": f"missing dependency: {exc.name}",
        }

    if args.video is None and args.image is None:
        return {"status": "failed", "answer": None, "error": "either --video or --image is required"}

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16 if args.dtype == "float16" else torch.float32
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_flash_attn=bool(args.use_flash_attn),
        trust_remote_code=not args.no_trust_remote_code,
        device_map=args.device_map if args.device_map != "none" else None,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=not args.no_trust_remote_code, use_fast=False)
    generation_config = {"max_new_tokens": args.max_new_tokens, "do_sample": False}

    if args.video:
        pixel_values, num_patches_list = load_video_pixels(
            args.video,
            VideoReader=VideoReader,
            cpu=cpu,
            Image=Image,
            transforms=transforms,
            InterpolationMode=InterpolationMode,
            torch=torch,
            input_size=args.input_size,
            max_num=args.max_tiles_video,
            num_segments=args.num_video_frames,
        )
        question = build_video_question(num_patches_list, args.prompt)
    else:
        pixel_values = load_image_pixels(
            args.image,
            Image=Image,
            transforms=transforms,
            InterpolationMode=InterpolationMode,
            torch=torch,
            input_size=args.input_size,
            max_num=args.max_tiles_video,
        )
        num_patches_list = [int(pixel_values.shape[0])]
        question = "<image>\n" + args.prompt

    if args.device != "cpu":
        pixel_values = pixel_values.to(dtype=dtype).cuda()
    else:
        pixel_values = pixel_values.to(dtype=dtype)
    response = model.chat(
        tokenizer,
        pixel_values,
        question,
        generation_config,
        num_patches_list=num_patches_list,
        history=None,
        return_history=False,
    )
    return {
        "status": "executed",
        "answer": response[0] if isinstance(response, tuple) else response,
        "num_patches_list": num_patches_list,
        "input_frames": len(num_patches_list),
        "model_path": args.model_path,
    }


def load_video_pixels(
    video_path: str,
    *,
    VideoReader: Any,
    cpu: Any,
    Image: Any,
    transforms: Any,
    InterpolationMode: Any,
    torch: Any,
    input_size: int,
    max_num: int,
    num_segments: int,
) -> tuple[Any, list[int]]:
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    transform = build_transform(transforms, InterpolationMode, input_size)
    pixel_values_list = []
    num_patches_list = []
    for frame_index in get_uniform_frame_indices(max_frame=max_frame, num_segments=num_segments):
        image = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
        tiles = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = torch.stack([transform(tile) for tile in tiles])
        num_patches_list.append(int(pixel_values.shape[0]))
        pixel_values_list.append(pixel_values)
    return torch.cat(pixel_values_list), num_patches_list


def load_image_pixels(
    image_path: str,
    *,
    Image: Any,
    transforms: Any,
    InterpolationMode: Any,
    torch: Any,
    input_size: int,
    max_num: int,
) -> Any:
    image = Image.open(image_path).convert("RGB")
    transform = build_transform(transforms, InterpolationMode, input_size)
    tiles = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    return torch.stack([transform(tile) for tile in tiles])


def build_transform(transforms: Any, InterpolationMode: Any, input_size: int) -> Any:
    return transforms.Compose(
        [
            transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            transforms.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def dynamic_preprocess(image: Any, *, image_size: int, max_num: int, use_thumbnail: bool) -> list[Any]:
    width, height = image.size
    aspect_ratio = width / height
    target_ratios = sorted(
        {(i, j) for n in range(1, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if 1 <= i * j <= max_num},
        key=lambda item: item[0] * item[1],
    )
    best_ratio = min(target_ratios, key=lambda item: abs(aspect_ratio - (item[0] / item[1])))
    target_width = image_size * best_ratio[0]
    target_height = image_size * best_ratio[1]
    resized = image.resize((target_width, target_height))
    tiles = []
    for idx in range(best_ratio[0] * best_ratio[1]):
        left = (idx % best_ratio[0]) * image_size
        upper = (idx // best_ratio[0]) * image_size
        tiles.append(resized.crop((left, upper, left + image_size, upper + image_size)))
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run InternVL3 native/off video inference")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--video")
    parser.add_argument("--image")
    parser.add_argument("--num-video-frames", type=int, default=8)
    parser.add_argument("--max-tiles-video", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--use-flash-attn", action="store_true")
    parser.add_argument("--no-trust-remote-code", action="store_true")
    return parser


def main() -> None:
    payload = run_internvl3_off(build_parser().parse_args())
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
