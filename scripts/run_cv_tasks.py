#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Run AutoGaze CV task comparisons on an image or video and save results.

Image mode  → PNG comparison grids per task + metrics.json
Video mode  → per-task MP4 with 3-panel side-by-side [original+gaze | full | ag_result]

Usage:
    python scripts/run_cv_tasks.py --input path/to/image.jpg --output-dir results/cv_tasks
    python scripts/run_cv_tasks.py --input path/to/video.mp4 --output-dir results/cv_tasks
    python scripts/run_cv_tasks.py --input path/to/image.jpg --tasks depth yolos --ratios 0.75 0.5
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# Cross-platform font helpers
# ─────────────────────────────────────────────────────────────────────────────

_PIL_FONT_CACHE: dict = {}


def _get_pil_font(size=14):
    """Return a PIL ImageFont that renders CJK characters on both macOS and Linux."""
    from PIL import ImageFont
    if size in _PIL_FONT_CACHE:
        return _PIL_FONT_CACHE[size]
    candidates = [
        # macOS
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/Library/Fonts/AppleGothic.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        # Linux — nanum
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
        # Linux — noto CJK
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        # Linux — DejaVu (ASCII-only fallback but clean)
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                font = ImageFont.truetype(p, size)
                _PIL_FONT_CACHE[size] = font
                return font
            except Exception:
                pass
    try:
        import matplotlib.font_manager as fm
        for name in ["Apple SD Gothic Neo", "NanumGothic", "Noto Sans CJK KR",
                     "Noto Sans CJK JP", "DejaVu Sans"]:
            try:
                path = fm.findfont(fm.FontProperties(family=name), fallback_to_default=False)
                if path and os.path.exists(path):
                    font = ImageFont.truetype(path, size)
                    _PIL_FONT_CACHE[size] = font
                    return font
            except Exception:
                pass
    except Exception:
        pass
    font = ImageFont.load_default()
    _PIL_FONT_CACHE[size] = font
    return font


def _configure_mpl_cjk():
    """Configure matplotlib to use an available CJK-capable font."""
    import matplotlib
    matplotlib.rcParams["axes.unicode_minus"] = False
    try:
        import matplotlib.font_manager as fm
        for name in ["Apple SD Gothic Neo", "AppleGothic", "NanumGothic",
                     "Noto Sans CJK KR", "Noto Sans CJK JP"]:
            try:
                path = fm.findfont(fm.FontProperties(family=name), fallback_to_default=False)
                if path and os.path.exists(path):
                    matplotlib.rcParams["font.family"] = name
                    return
            except Exception:
                pass
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

ALL_TASKS = ["depth", "yolos", "dinov2", "segformer", "siglip", "videomae_cls", "xclip"]

DEFAULTS = {
    "ag_path":   "weights/AutoGaze",
    "ag_ratio":  0.5,
    "ratios":    [0.75, 0.5, 0.25],
    "tasks":     ALL_TASKS,
    "score_thr": 0.3,
    "stride":    1,
}


def parse_args():
    p = argparse.ArgumentParser(
        description="AutoGaze CV task comparison script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, help="Image or video file path")
    p.add_argument("--output-dir", default="results/cv_tasks", help="Root output directory")
    p.add_argument("--ag-path", default=DEFAULTS["ag_path"], help="AutoGaze weights directory")
    p.add_argument("--ag-ratio", type=float, default=DEFAULTS["ag_ratio"],
                   help="Default AutoGaze gazing ratio")
    p.add_argument("--ratios", nargs="+", type=float, default=DEFAULTS["ratios"],
                   help="AutoGaze gazing ratios to test")
    p.add_argument("--tasks", nargs="+", choices=ALL_TASKS, default=DEFAULTS["tasks"],
                   help="Tasks to run")
    p.add_argument("--score-thr", type=float, default=DEFAULTS["score_thr"],
                   help="Detection score threshold")
    p.add_argument("--stride", type=int, default=DEFAULTS["stride"],
                   help="Frame stride for video input")
    p.add_argument("--temporal-window", type=int, default=16,
                   help="Frames per AutoGaze chunk (model was trained with T=16; "
                        "last chunk is zero-padded to this size)")
    p.add_argument("--save-frames", action="store_true",
                   help="Also save each video frame as PNG under frames/<task>/frame_XXXX.png")
    p.add_argument("--videomae-recon", action="store_true",
                   help="Add VideoMAE reconstruction panel: non-gaze patches masked → "
                        "VideoMAE reconstructs them → run CV tasks on reconstructed frame. "
                        "Produces 4-panel output [original+gaze | full | ag | recon]")
    p.add_argument("--device", default=None,
                   help="Device: cuda / cpu (default: auto, MPS excluded)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Lazy model loading (load only once per script run)
# ─────────────────────────────────────────────────────────────────────────────

_loaded = {}


def _load_ag(ag_path, device):
    if "ag" in _loaded:
        return _loaded["ag"]
    from autogaze.models.autogaze import AutoGaze
    from autogaze.models.autogaze.processing_autogaze import AutoGazeImageProcessor
    from autogaze.models.autogaze.autogaze_cv import AutoGazeTokenSelector

    print(f"[AutoGaze] loading from {ag_path}…")
    ag_model = AutoGaze.from_pretrained(ag_path).to(device).eval()
    ag_proc   = AutoGazeImageProcessor.from_pretrained(ag_path)
    selector  = AutoGazeTokenSelector(ag_model, gazing_ratio=0.5)
    _loaded["ag"] = (ag_model, ag_proc, selector)
    print("[AutoGaze] ready")
    return _loaded["ag"]


def _load_depth(device):
    if "depth" in _loaded:
        return _loaded["depth"]
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    mid = "depth-anything/Depth-Anything-V2-Small-hf"
    print(f"[Depth] loading {mid}…")
    proc  = AutoImageProcessor.from_pretrained(mid)
    model = AutoModelForDepthEstimation.from_pretrained(mid).to(device).eval()
    _loaded["depth"] = (proc, model)
    return _loaded["depth"]


def _load_yolos(device):
    if "yolos" in _loaded:
        return _loaded["yolos"]
    from transformers import AutoImageProcessor, AutoModelForObjectDetection
    mid = "hustvl/yolos-tiny"
    print(f"[YOLOS] loading {mid}…")
    proc  = AutoImageProcessor.from_pretrained(mid)
    model = AutoModelForObjectDetection.from_pretrained(mid).to(device).eval()
    _loaded["yolos"] = (proc, model)
    return _loaded["yolos"]


def _load_dinov2(device):
    if "dinov2" in _loaded:
        return _loaded["dinov2"]
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    mid = "facebook/dinov2-base-imagenet1k-1-layer"
    print(f"[DINOv2] loading {mid}…")
    proc  = AutoImageProcessor.from_pretrained(mid)
    model = AutoModelForImageClassification.from_pretrained(mid).to(device).eval()
    _loaded["dinov2"] = (proc, model)
    return _loaded["dinov2"]


def _load_segformer(device):
    if "segformer" in _loaded:
        return _loaded["segformer"]
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
    mid = "nvidia/segformer-b2-finetuned-ade-512-512"
    print(f"[SegFormer] loading {mid}…")
    proc  = AutoImageProcessor.from_pretrained(mid)
    model = SegformerForSemanticSegmentation.from_pretrained(mid).to(device).eval()
    _loaded["segformer"] = (proc, model)
    return _loaded["segformer"]


def _load_siglip(device):
    if "siglip" in _loaded:
        return _loaded["siglip"]
    from transformers import SiglipModel, AutoProcessor
    mid = "google/siglip-base-patch16-224"
    print(f"[SigLIP] loading {mid}…")
    proc  = AutoProcessor.from_pretrained(mid)
    model = SiglipModel.from_pretrained(mid).to(device).eval()
    _loaded["siglip"] = (proc, model)
    return _loaded["siglip"]


def _load_videomae(device):
    if "videomae" in _loaded:
        return _loaded["videomae"]
    from transformers import VideoMAEForPreTraining, VideoMAEImageProcessor
    mid = "MCG-NJU/videomae-base"
    print(f"[VideoMAE] loading {mid}…")
    proc  = VideoMAEImageProcessor.from_pretrained(mid)
    model = VideoMAEForPreTraining.from_pretrained(mid).to(device).eval()
    _loaded["videomae"] = (proc, model)
    print("[VideoMAE] ready")
    return _loaded["videomae"]


XCLIP_ACTION_TEXTS = [
    "a person playing basketball",
    "a person running",
    "a person swimming",
    "a person cooking",
    "a person dancing",
    "a person playing guitar",
    "a person riding a bike",
    "a person playing soccer",
]


def _load_videomae_cls(device):
    if "videomae_cls" in _loaded:
        return _loaded["videomae_cls"]
    from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
    mid = "MCG-NJU/videomae-base-finetuned-kinetics"
    print(f"[VideoMAE-CLS] loading {mid}…")
    proc  = VideoMAEImageProcessor.from_pretrained(mid)
    model = VideoMAEForVideoClassification.from_pretrained(mid).to(device).eval()
    _loaded["videomae_cls"] = (proc, model)
    print("[VideoMAE-CLS] ready")
    return _loaded["videomae_cls"]


def _load_xclip(device):
    if "xclip" in _loaded:
        return _loaded["xclip"]
    from transformers import XCLIPModel, XCLIPProcessor
    mid = "microsoft/xclip-base-patch32"
    print(f"[X-CLIP] loading {mid}…")
    proc  = XCLIPProcessor.from_pretrained(mid)
    model = XCLIPModel.from_pretrained(mid).to(device).eval()
    _loaded["xclip"] = (proc, model)
    print("[X-CLIP] ready")
    return _loaded["xclip"]


def _pad_frames_to_T(frames, T):
    """Pad/truncate a list of PIL images to exactly T frames."""
    frames = list(frames)
    if len(frames) >= T:
        return frames[:T]
    return frames + [frames[-1]] * (T - len(frames))


# ─────────────────────────────────────────────────────────────────────────────
# AutoGaze helpers
# ─────────────────────────────────────────────────────────────────────────────

def prep_for_autogaze(pil_image, ag_proc, device):
    out   = ag_proc(images=[pil_image.resize((224, 224))], return_tensors="pt")
    video = out["pixel_values"].to(device)   # (1, 1, C, 224, 224)
    return video


def compute_gaze_map(ag_model, ag_video, ag_ratio=0.5):
    """Return (14, 14) float numpy gaze map."""
    from autogaze.models.autogaze.autogaze_cv import AutoGazeTokenSelector
    sel = AutoGazeTokenSelector(ag_model, gazing_ratio=ag_ratio)
    with torch.no_grad():
        gaze_out = ag_model({"video": ag_video}, gazing_ratio=ag_ratio, generate_only=True)
    mask = gaze_out["gazing_mask"][-1][0, 0]   # (196,)
    return mask.float().cpu().numpy().reshape(14, 14)


def make_selector(ag_model, ratio):
    from autogaze.models.autogaze.autogaze_cv import AutoGazeTokenSelector
    return AutoGazeTokenSelector(ag_model, gazing_ratio=ratio)


# ─────────────────────────────────────────────────────────────────────────────
# Task runners  (image mode: return dict of numpy arrays)
# ─────────────────────────────────────────────────────────────────────────────

def run_depth(pil_img, ag_video, ag_model, ratios, device):
    proc, model = _load_depth(device)
    inputs = proc(images=pil_img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    ih = inputs["pixel_values"].shape[-2]
    iw = inputs["pixel_values"].shape[-1]
    gh, gw = ih // 14, iw // 14

    results = {}
    with torch.no_grad():
        results["full"] = model(**inputs).predicted_depth.squeeze().cpu().float().numpy()

    for r in ratios:
        sel = make_selector(ag_model, r)
        m = sel.compute_gaze_mask(ag_video, target_h=gh, target_w=gw)
        with sel.token_mask_context(model.backbone.embeddings, m, has_cls_token=True):
            with torch.no_grad():
                d = model(**inputs).predicted_depth.squeeze().cpu().float().numpy()
        results[f"ag{int(r*100)}"] = d

    ref = results["full"]
    metrics = {f"rmse_ag{int(r*100)}": float(np.sqrt(((results[f'ag{int(r*100)}'] - ref)**2).mean()))
               for r in ratios}
    return results, metrics, (ih, iw, gh, gw)


COCO_COLORS = None

def _get_coco_colors():
    global COCO_COLORS
    if COCO_COLORS is None:
        import matplotlib.pyplot as plt
        COCO_COLORS = list(plt.cm.tab20.colors)
    return COCO_COLORS


def _detect_to_rgba(pil_img, logits, pred_boxes, id2label, thresh):
    """Return RGBA numpy (H, W, 4) with detection overlay."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from io import BytesIO
    colors = _get_coco_colors()
    fig, ax = plt.subplots(1, 1, figsize=(pil_img.width / 100, pil_img.height / 100), dpi=100)
    ax.imshow(pil_img)
    W, H = pil_img.size
    probs = logits.softmax(-1)[0, :, :-1]
    keep  = probs.max(-1).values > thresh
    boxes  = pred_boxes[0, keep].cpu()
    labels = probs[keep].argmax(-1).cpu()
    scores = probs[keep].max(-1).values.cpu()
    n_det  = int(keep.sum())
    for (cx, cy, w, h), lbl, sc in zip(boxes, labels, scores):
        x0 = (cx - w/2) * W;  y0 = (cy - h/2) * H
        bw = w * W;             bh = h * H
        c = colors[lbl % len(colors)]
        rect = mpatches.FancyBboxPatch(
            (x0, y0), bw, bh, boxstyle="round,pad=2",
            linewidth=2, edgecolor=c, facecolor="none"
        )
        ax.add_patch(rect)
        name = id2label.get(lbl.item(), str(lbl.item()))
        ax.text(x0, y0 - 4, f"{name} {sc:.2f}", color="white", fontsize=7,
                fontweight="bold", bbox=dict(facecolor=c, alpha=0.8, pad=1))
    ax.axis("off")
    fig.tight_layout(pad=0)
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    arr = np.array(Image.open(buf).convert("RGBA"))
    return arr, n_det


def run_yolos(pil_img, ag_video, ag_model, ratios, device, score_thr):
    proc, model = _load_yolos(device)
    inputs = proc(images=pil_img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    ih = inputs["pixel_values"].shape[-2]
    iw = inputs["pixel_values"].shape[-1]
    gh, gw = ih // 16, iw // 16

    results = {}
    with torch.no_grad():
        out = model(**inputs)
    results["full"], n_full = _detect_to_rgba(pil_img, out.logits, out.pred_boxes,
                                               model.config.id2label, score_thr)
    n_boxes = {"full": n_full}

    for r in ratios:
        sel = make_selector(ag_model, r)
        m = sel.compute_gaze_mask(ag_video, target_h=gh, target_w=gw)
        with sel.token_mask_context(model.vit.embeddings, m, has_cls_token=True):
            with torch.no_grad():
                out = model(**inputs)
        key = f"ag{int(r*100)}"
        results[key], n_boxes[key] = _detect_to_rgba(
            pil_img, out.logits, out.pred_boxes, model.config.id2label, score_thr
        )

    metrics = {f"n_boxes_ag{int(r*100)}": n_boxes[f'ag{int(r*100)}'] for r in ratios}
    metrics["n_boxes_full"] = n_full
    return results, metrics, (ih, iw, gh, gw)


SIGLIP_TEXTS = [
    "a cat resting on a sofa",
    "a dog playing in a park",
    "a remote control on a table",
    "a person sitting in a chair",
    "a book on a shelf",
]


def run_dinov2(pil_img, ag_video, ag_model, ratios, device):
    proc, model = _load_dinov2(device)
    inputs = proc(images=pil_img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    ih = inputs["pixel_values"].shape[-2]
    iw = inputs["pixel_values"].shape[-1]
    gh, gw = ih // 14, iw // 14

    def topk(logits, k=5):
        probs = logits.softmax(-1)[0]
        top = probs.topk(k)
        return [(model.config.id2label[i.item()], p.item())
                for i, p in zip(top.indices, top.values)]

    with torch.no_grad():
        base_top5 = topk(model(**inputs).logits)

    ratio_top5 = {}
    for r in ratios:
        sel = make_selector(ag_model, r)
        m = sel.compute_gaze_mask(ag_video, target_h=gh, target_w=gw)
        with sel.token_mask_context(model.dinov2.embeddings, m, has_cls_token=True):
            with torch.no_grad():
                ratio_top5[r] = dict(topk(model(**inputs).logits, k=10))

    # Build union of top labels across all models for a complete chart
    seen = {}
    for lbl, p in base_top5:
        seen[lbl] = seen.get(lbl, 0) + p
    for r in ratios:
        for lbl, p in ratio_top5[r].items():
            seen[lbl] = seen.get(lbl, 0) + p
    # sort by total probability weight, keep top 8
    union_labels = [lbl for lbl, _ in sorted(seen.items(), key=lambda x: -x[1])][:8]

    # full probs dict for chart lookups
    full_top_dict = dict(base_top5)

    metrics = {"top1_full": base_top5[0][0]}
    for r in ratios:
        rd = ratio_top5[r]
        metrics[f"top1_ag{int(r*100)}"] = max(rd, key=rd.get) if rd else "?"
    return full_top_dict, ratio_top5, union_labels, metrics, (ih, iw, gh, gw)


np.random.seed(42)
ADE20K_PALETTE = np.random.randint(0, 255, (150, 3), dtype=np.uint8)


def run_segformer(pil_img, ag_video, ag_model, ratios, device):
    from autogaze.models.autogaze.autogaze_cv import ConvFeatureSelector
    proc, model = _load_segformer(device)
    inputs = proc(images=pil_img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    ih = inputs["pixel_values"].shape[-2]
    iw = inputs["pixel_values"].shape[-1]
    gh, gw = ih // 4, iw // 4

    def to_rgb(logits_batch):
        up = F.interpolate(logits_batch, size=(ih, iw), mode="bilinear", align_corners=False)
        seg = up.argmax(dim=1)[0].cpu().numpy()
        return ADE20K_PALETTE[seg]

    stage0_proj = model.segformer.encoder.patch_embeddings[0].proj
    results = {}
    with torch.no_grad():
        results["full"] = to_rgb(model(**inputs).logits)

    for r in ratios:
        sel = make_selector(ag_model, r)
        conv_sel = ConvFeatureSelector(sel, ag_video, gh, gw)
        with conv_sel.apply(stage0_proj):
            with torch.no_grad():
                results[f"ag{int(r*100)}"] = to_rgb(model(**inputs).logits)

    metrics = {}
    ref = results["full"]
    for r in ratios:
        diff = (results[f"ag{int(r*100)}"].astype(float) - ref.astype(float))
        metrics[f"pixel_diff_ag{int(r*100)}"] = float(np.abs(diff).mean())
    return results, metrics, (ih, iw, gh, gw)


def run_siglip(pil_img, ag_video, ag_model, ratios, device, texts=None):
    if texts is None:
        texts = SIGLIP_TEXTS
    proc, model = _load_siglip(device)
    GRID = 14  # 224 // 16

    def infer(mask_context=None):
        inputs = proc(text=texts, images=[pil_img], return_tensors="pt",
                      padding="max_length").to(device)
        if mask_context:
            with mask_context:
                with torch.no_grad():
                    out = model(**inputs)
        else:
            with torch.no_grad():
                out = model(**inputs)
        return out.logits_per_image.softmax(dim=-1)[0].cpu().numpy()

    probs_full = infer()
    ratio_probs = {}
    for r in ratios:
        sel = make_selector(ag_model, r)
        m = sel.compute_gaze_mask(ag_video, target_h=GRID, target_w=GRID)
        ctx = sel.token_mask_context(model.vision_model.embeddings, m, has_cls_token=False)
        ratio_probs[r] = infer(ctx)

    metrics = {"top1_full": texts[int(np.argmax(probs_full))]}
    for r in ratios:
        metrics[f"top1_ag{int(r*100)}"] = texts[int(np.argmax(ratio_probs[r]))]
    return probs_full, ratio_probs, texts, metrics


def run_videomae_cls(chunk_pil, ag_video, ag_model, ratios, device):
    """Video action recognition with VideoMAE-CLS (Kinetics-400).

    chunk_pil: list of T PIL images (repeat single frame in image mode)
    ag_video:  (1, 1, C, 224, 224) or (1, T, C, 224, 224) AutoGaze input tensor
    """
    proc, model = _load_videomae_cls(device)
    T = 16
    frames = _pad_frames_to_T(chunk_pil, T)
    inputs = proc(frames, return_tensors="pt")
    pv = inputs.pixel_values.to(device)                   # (1, 16, 3, 224, 224)
    t_dim = pv.shape[1] // model.config.tubelet_size       # 16 // 2 = 8

    def topk_cls(logits, k=5):
        probs = logits.softmax(-1)[0]
        top = probs.topk(k)
        return [(model.config.id2label[i.item()], p.item())
                for i, p in zip(top.indices, top.values)]

    with torch.no_grad():
        top_full = topk_cls(model(pv).logits)

    ratio_top5 = {}
    for r in ratios:
        sel = make_selector(ag_model, r)
        m = sel.compute_gaze_mask(ag_video, target_h=14, target_w=14)  # (1, 196) bool
        spatial = m[0].cpu().numpy()                                     # (196,) bool
        full_mask = (
            torch.from_numpy(spatial).unsqueeze(0).expand(t_dim, -1).reshape(1, t_dim * 196)
        ).to(device)                                                     # (1, 1568)

        def _hook(module, inp, out, mask=full_mask):
            return out * mask.float().unsqueeze(-1)

        handle = model.videomae.embeddings.register_forward_hook(_hook)
        try:
            with torch.no_grad():
                ratio_top5[r] = topk_cls(model(pv).logits)
        finally:
            handle.remove()

    metrics = {"top1_full": top_full[0][0]}
    for r in ratios:
        metrics[f"top1_ag{int(r*100)}"] = ratio_top5[r][0][0]
    return top_full, ratio_top5, metrics


def run_xclip(chunk_pil, ag_video, ag_model, ratios, device, texts=None):
    """Zero-shot video action recognition with X-CLIP.

    chunk_pil: list of T PIL images (repeat single frame in image mode)
    ag_video:  (1, 1, C, 224, 224) or (1, T, C, 224, 224) AutoGaze input tensor
    texts:     list of action description strings (default: XCLIP_ACTION_TEXTS)
    """
    if texts is None:
        texts = XCLIP_ACTION_TEXTS
    proc, model = _load_xclip(device)
    T = 8
    frames = _pad_frames_to_T(chunk_pil, T)

    def _infer(spatial_mask=None):
        inputs_x = proc(text=texts, videos=frames, return_tensors="pt",
                        padding=True).to(device)
        if spatial_mask is not None:
            def _hook(module, inp, out):
                cls_tok = out[:, :1]
                spatial = out[:, 1:] * spatial_mask.float().to(out.device).unsqueeze(0).unsqueeze(-1)
                return torch.cat([cls_tok, spatial], dim=1)
            handle = model.vision_model.vision_model.embeddings.register_forward_hook(_hook)
        else:
            handle = None
        try:
            with torch.no_grad():
                logits = model(**inputs_x).logits_per_video  # (1, n_texts)
        finally:
            if handle is not None:
                handle.remove()
        probs = logits.softmax(-1)[0].cpu().numpy()
        return sorted(zip(texts, probs.tolist()), key=lambda x: -x[1])

    top_full = _infer()
    ratio_top5 = {}
    for r in ratios:
        sel = make_selector(ag_model, r)
        m = sel.compute_gaze_mask(ag_video, target_h=14, target_w=14)  # (1, 196) bool
        ratio_top5[r] = _infer(m[0].float())

    metrics = {"top1_full": top_full[0][0]}
    for r in ratios:
        metrics[f"top1_ag{int(r*100)}"] = ratio_top5[r][0][0]
    return top_full, ratio_top5, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Visualization helpers
# ─────────────────────────────────────────────────────────────────────────────

def gaze_overlay_pil(pil_img, gaze_map):
    """Return PIL Image with gaze overlay (same size as pil_img)."""
    import cv2
    img_arr = np.array(pil_img.resize((224, 224)))
    overlay = cv2.resize(gaze_map, (224, 224), interpolation=cv2.INTER_LINEAR)
    import matplotlib.cm as cm
    heat = (cm.hot(overlay)[:, :, :3] * 255).astype(np.uint8)
    blended = (img_arr * 0.6 + heat * 0.4).clip(0, 255).astype(np.uint8)
    return Image.fromarray(blended).resize(pil_img.size, Image.LANCZOS)


def depth_to_rgb(depth):
    import matplotlib.cm as cm
    d_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    return (cm.Spectral_r(d_norm)[:, :, :3] * 255).astype(np.uint8)


def save_comparison_grid(out_path, pil_img, gaze_map, task_results, task_name,
                         ratios, label_fn=None):
    """Save a side-by-side comparison PNG for one task.

    Columns: [original | gaze overlay | full | ag75 | ag50 | ag25 ...]
    """
    import matplotlib.pyplot as plt
    _configure_mpl_cjk()

    n_cols = 2 + 1 + len(ratios)   # original, gaze, full, *ratios
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    axes[0].imshow(pil_img)
    axes[0].set_title("원본", fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(gaze_overlay_pil(pil_img, gaze_map))
    n_sel = int((gaze_map > 0.5).sum())
    axes[1].set_title(f"AutoGaze gaze\n({n_sel}/196 선택)", fontweight="bold")
    axes[1].axis("off")

    def _show(ax, data, title):
        if isinstance(data, np.ndarray):
            if data.dtype != np.uint8:
                data = depth_to_rgb(data)
            if data.ndim == 3 and data.shape[2] == 4:
                data = data[..., :3]
            ax.imshow(data)
        elif isinstance(data, Image.Image):
            ax.imshow(data)
        else:
            ax.imshow(data)
        ax.set_title(title, fontweight="bold")
        ax.axis("off")

    _show(axes[2], task_results["full"],
          label_fn("full") if label_fn else "전체 (100%)")

    for i, r in enumerate(ratios):
        key = f"ag{int(r*100)}"
        lbl = label_fn(key) if label_fn else f"AutoGaze {int(r*100)}%"
        _show(axes[3 + i], task_results[key], lbl)

    fig.suptitle(task_name, fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


def save_dinov2_chart(out_path, full_top_dict, ratio_top5, union_labels, ratios):
    import matplotlib.pyplot as plt
    _configure_mpl_cjk()

    labels = [lbl[:22] for lbl in union_labels]
    x = np.arange(len(labels))
    n_groups = 1 + len(ratios)
    width = 0.72 / n_groups
    offsets = np.linspace(-(n_groups - 1) / 2, (n_groups - 1) / 2, n_groups) * width
    fig, ax = plt.subplots(figsize=(max(13, len(labels) * 1.6), 5))
    ax.bar(x + offsets[0], [full_top_dict.get(lbl, 0.0) for lbl in union_labels],
           width, label="전체 (100%)", color="steelblue")
    colors = ["#e67e22", "#e74c3c", "#8e44ad", "#27ae60"]
    for i, r in enumerate(ratios):
        probs_r = [ratio_top5[r].get(lbl, 0.0) for lbl in union_labels]
        ax.bar(x + offsets[1 + i], probs_r, width,
               label=f"AutoGaze {int(r*100)}%", color=colors[i % len(colors)])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("확률")
    ax.set_title("DINOv2 ImageNet Top-8 확률 (전체+AG 합산 기준) — AutoGaze ratio별",
                 fontweight="bold")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


def save_siglip_chart(out_path, probs_full, ratio_probs, texts, ratios):
    import matplotlib.pyplot as plt
    _configure_mpl_cjk()

    x = np.arange(len(texts))
    w = 0.18
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - 1.5 * w, probs_full, w, label="전체 (100%)", color="steelblue")
    clrs = ["#e67e22", "#e74c3c", "#8e44ad"]
    for i, r in enumerate(ratios):
        ax.bar(x + (i - 0.5) * w, ratio_probs[r], w,
               label=f"AutoGaze {int(r*100)}%", color=clrs[i % len(clrs)])
    ax.set_xticks(x)
    ax.set_xticklabels([t[:30] for t in texts], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("유사도 확률")
    ax.set_title("SigLIP Zero-shot 분류 — AutoGaze ratio별", fontweight="bold")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


def save_action_chart(out_path, top_full, ratio_top5, ratios, title):
    """Bar chart comparing top-K action recognition probabilities across ratios."""
    import matplotlib.pyplot as plt
    _configure_mpl_cjk()

    # Union of top labels from full + all ratios (capped at 8)
    union_labels = [lbl for lbl, _ in top_full[:8]]
    for r in ratios:
        for lbl, _ in ratio_top5[r]:
            if lbl not in union_labels:
                union_labels.append(lbl)
    union_labels = union_labels[:8]

    def _prob(lst, lbl):
        return next((p for l, p in lst if l == lbl), 0.0)

    x = np.arange(len(union_labels))
    n_groups = 1 + len(ratios)
    width = 0.72 / n_groups
    offsets = np.linspace(-(n_groups - 1) / 2, (n_groups - 1) / 2, n_groups) * width

    fig, ax = plt.subplots(figsize=(max(13, len(union_labels) * 1.6), 5))
    ax.bar(x + offsets[0], [_prob(top_full, lbl) for lbl in union_labels],
           width, label="전체 (100%)", color="steelblue")
    colors = ["#e67e22", "#e74c3c", "#8e44ad", "#27ae60"]
    for i, r in enumerate(ratios):
        ax.bar(x + offsets[1 + i], [_prob(ratio_top5[r], lbl) for lbl in union_labels],
               width, label=f"AutoGaze {int(r*100)}%", color=colors[i % len(colors)])
    ax.set_xticks(x)
    ax.set_xticklabels([lbl[:28] for lbl in union_labels], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("확률")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


def save_summary_chart(out_path, all_metrics, ratios):
    import matplotlib.pyplot as plt
    _configure_mpl_cjk()

    has_depth = "depth" in all_metrics and "rmse_ag50" in all_metrics["depth"]
    has_det   = "yolos" in all_metrics and "n_boxes_full" in all_metrics["yolos"]
    n_plots = int(has_depth) + int(has_det)
    if n_plots == 0:
        return

    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    idx = 0
    if has_depth:
        dm = all_metrics["depth"]
        rmse_vals = [0.0] + [dm.get(f"rmse_ag{int(r*100)}", 0) for r in ratios]
        ax = axes[idx]; idx += 1
        ax.plot([100] + [int(r*100) for r in ratios], rmse_vals, "o-",
                color="#2e86c1", lw=2, ms=7)
        ax.set_xlabel("AutoGaze ratio (%)")
        ax.set_ylabel("RMSE vs 전체 토큰")
        ax.set_title("Depth Anything V2\n토큰 감소 → depth RMSE", fontweight="bold")
        ax.invert_xaxis()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if has_det:
        ym = all_metrics["yolos"]
        n_full = max(ym["n_boxes_full"], 1)
        det_vals = [1.0] + [ym.get(f"n_boxes_ag{int(r*100)}", n_full) / n_full for r in ratios]
        ax = axes[idx]; idx += 1
        ax.plot([100] + [int(r*100) for r in ratios], det_vals, "s-",
                color="#e67e22", lw=2, ms=7)
        ax.set_xlabel("AutoGaze ratio (%)")
        ax.set_ylabel("검출 박스 수 (전체 대비)")
        ax.set_title("YOLOS 객체 탐지\n토큰 감소 → 검출 박스 수", fontweight="bold")
        ax.set_ylim(0, 1.3)
        ax.axhline(1.0, color="gray", ls="--", lw=1, alpha=0.6)
        ax.invert_xaxis()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("AutoGaze Token Ratio → Task 성능 영향", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Video helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_video_frames_all(video_path, stride=1):
    """Return list of (frame_idx, PIL Image) for every stride-th frame."""
    import cv2
    frames = []
    cap = cv2.VideoCapture(str(video_path))
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            frames.append((idx, Image.fromarray(frame[:, :, ::-1])))
        idx += 1
    cap.release()
    return frames


def _get_video_fps(video_path):
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps or 30.0


def _frames_to_mp4(out_path, frames, fps):
    """Write list of PIL Images to MP4, normalizing all frames to the first frame's size."""
    import cv2
    if not frames:
        return
    w, h = frames[0].size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    for f in frames:
        if f.size != (w, h):
            f = f.resize((w, h), Image.LANCZOS)
        writer.write(np.array(f.convert("RGB"))[:, :, ::-1])
    writer.release()


def prep_video_chunk(pil_frames, ag_proc, device):
    """List of T PIL images → (1, T, C, 224, 224) tensor for AutoGaze."""
    tensors = []
    for img in pil_frames:
        t = ag_proc(images=[img.resize((224, 224))], return_tensors="pt")["pixel_values"][0]
        # pixel_values shape: (1, C, 224, 224) — drop the outer batch dim per frame
        tensors.append(t)  # (1, C, 224, 224)
    video = torch.cat(tensors, dim=0).unsqueeze(0)  # (1, T, C, 224, 224)
    return video.to(device)


def gaze_mask_for_grid(gaze_map_14, target_h, target_w, threshold=0.5, device="cpu"):
    """(14, 14) numpy float array → bool tensor (1, target_h * target_w).

    Uses the already-computed gaze map from a temporal AutoGaze pass —
    avoids re-running AutoGaze per frame.
    """
    raw = torch.from_numpy(gaze_map_14).float().unsqueeze(0).unsqueeze(0)  # (1,1,14,14)
    if target_h != 14 or target_w != 14:
        raw = F.interpolate(raw, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return (raw.squeeze() > threshold).reshape(1, -1).to(device)  # (1, N)


def make_npanel(panel_data, target_size=None):
    """Horizontal N-panel PIL Image with top label bar burned in.

    Args:
        panel_data:  list of (pil_or_ndarray, label_str)
        target_size: optional (W, H) — all panels are resized to this size so
                     aspect ratios stay consistent (e.g. pass pil_img.size in
                     video mode to match original frame dimensions).
    """
    from PIL import ImageDraw

    def _to_pil(x):
        if isinstance(x, np.ndarray):
            if x.dtype != np.uint8:
                x = depth_to_rgb(x)
            if x.ndim == 3 and x.shape[2] == 4:
                x = x[..., :3]
            return Image.fromarray(x)
        return x

    panels = [(_to_pil(img), lbl) for img, lbl in panel_data]

    if target_size is not None:
        # Normalize all panels to the same (W, H) so aspect ratios are consistent
        panels = [(p.resize(target_size, Image.LANCZOS), lbl) for p, lbl in panels]
    else:
        target_h = max(p.height for p, _ in panels)
        panels = [(p.resize((int(p.width * target_h / p.height), target_h), Image.LANCZOS), lbl)
                  for p, lbl in panels]

    target_h = panels[0][0].height
    label_h = max(24, target_h // 20)
    font = _get_pil_font(max(12, label_h - 4))
    total_w = sum(p.width for p, _ in panels)
    canvas = Image.new("RGB", (total_w, target_h + label_h), (15, 15, 15))
    draw = ImageDraw.Draw(canvas)

    x_off = 0
    for panel, label in panels:
        canvas.paste(panel, (x_off, label_h))
        draw.rectangle([x_off, 0, x_off + panel.width, label_h], fill=(40, 40, 40))
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]
        except AttributeError:
            tw = len(label) * (label_h // 2)
        tx = x_off + max(4, (panel.width - tw) // 2)
        draw.text((tx, 3), label, fill=(240, 240, 60), font=font)
        if x_off > 0:
            draw.line([(x_off, 0), (x_off, target_h + label_h)], fill=(80, 80, 80), width=2)
        x_off += panel.width

    return canvas


def make_3panel(pil_gaze_ov, arr_full, arr_ag, labels=("원본+gaze", "전체 토큰", "AutoGaze 토큰")):
    return make_npanel([(pil_gaze_ov, labels[0]), (arr_full, labels[1]), (arr_ag, labels[2])])


def _cls_overlay_pil(pil_img, label_probs, top_label=""):
    """Draw top-K classification results as text overlay on a copy of pil_img.

    Args:
        label_probs: list of (label_str, prob_float) sorted by prob desc.
        top_label:   optional tag shown in the top-right corner (e.g. "top-1").
    """
    from PIL import ImageDraw
    img = pil_img.copy().convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img)

    row_h = max(18, H // 20)
    font = _get_pil_font(max(11, row_h - 4))
    box_h = row_h * len(label_probs) + 8
    draw.rectangle([0, H - box_h, W, H], fill=(10, 10, 10))

    bar_max_w = W - 10
    for i, (label, prob) in enumerate(label_probs):
        y = H - box_h + 4 + i * row_h
        bar_w = int(bar_max_w * prob)
        bar_color = (60, 200, 80) if i == 0 else (60, 100, 200)
        draw.rectangle([0, y, bar_w, y + row_h - 2], fill=bar_color)
        txt = f"{prob:.3f}  {label[:35]}"
        draw.text((4, y + 1), txt, fill=(255, 255, 255), font=font)

    if top_label:
        draw.text((4, 4), top_label, fill=(240, 240, 60), font=font)

    return img


# ─────────────────────────────────────────────────────────────────────────────
# VideoMAE reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_with_videomae(chunk_pil, gaze_map_14, videomae_proc, videomae_model,
                               device, frame_t=0):
    """Reconstruct frame_t using VideoMAE with gaze-selected patches as visible.

    AutoGaze selects the "important" spatial patches (gaze_map_14 > 0.5).
    These are kept visible; all other patches are masked → VideoMAE reconstructs them.

    Pipeline:
        chunk (T=16 PIL) → VideoMAEImageProcessor → pixel_values (1,16,3,224,224)
        bool_masked_pos: ~gaze_bool broadcast over 8 temporal positions  → (1,1568)
        VideoMAEForPreTraining forward → logits (1, N_masked, 1536)
        un-normalize per-patch + un-normalize ImageNet → reconstructed frames
        return PIL image at frame_t

    Args:
        chunk_pil:  list of T PIL images (will be padded/truncated to 16)
        gaze_map_14: (14,14) float numpy — AutoGaze finest-scale map
        frame_t:    which frame index in the chunk to return (0-based)

    Returns:
        PIL Image (224×224) of the reconstructed frame
    """
    T_model = 16
    frames = list(chunk_pil[:T_model])
    if len(frames) < T_model:
        frames += [frames[-1]] * (T_model - len(frames))

    inputs = videomae_proc(frames, return_tensors="pt")
    pixel_values = inputs.pixel_values.to(device)   # (1, 16, 3, 224, 224)

    cfg = videomae_model.config
    tp = cfg.tubelet_size   # 2
    ps = cfg.patch_size     # 16
    B, T, C, H, W = pixel_values.shape
    t_dim  = T  // tp          # 8  temporal positions
    h_dim  = H  // ps          # 14 spatial rows
    w_dim  = W  // ps          # 14 spatial cols
    N      = t_dim * h_dim * w_dim   # 1568 total patches
    D      = tp * ps * ps * C        # 1536 values per patch

    # Build bool_masked_pos:
    # AutoGaze True = selected = VISIBLE → VideoMAE False (not masked)
    # AutoGaze False = background = MASKED → VideoMAE True (reconstruct)
    gaze_spatial = torch.from_numpy(gaze_map_14 > 0.5).reshape(h_dim * w_dim)  # (196,)
    spatial_mask = ~gaze_spatial                                                 # True = mask
    bool_masked_pos = spatial_mask.unsqueeze(0).expand(t_dim, -1)               # (8, 196)
    bool_masked_pos = bool_masked_pos.reshape(1, N).to(device)                  # (1, 1568)

    with torch.no_grad():
        outputs = videomae_model(pixel_values=pixel_values,
                                  bool_masked_pos=bool_masked_pos)
    logits = outputs.logits  # (1, N_masked, D)

    # Patchify pixel_values → (N, D) for per-patch stats
    pv = pixel_values[0]                                                  # (T, C, H, W)
    pv_p = pv.reshape(t_dim, tp, C, h_dim, ps, w_dim, ps)
    pv_p = pv_p.permute(0, 3, 5, 1, 4, 6, 2)                            # (t,h,w,tp,ps,ps,C)
    pv_p = pv_p.reshape(N, D)                                             # (N, D)

    # Per-patch mean/std (in ImageNet-normalised space)
    patch_mean = pv_p.mean(dim=-1, keepdim=True)                          # (N, 1)
    patch_std  = (pv_p.var(dim=-1, keepdim=True) + 1e-6).sqrt()

    # Fill masked patches with un-normalised predictions
    recon_patches = pv_p.clone()
    mask_idx = bool_masked_pos[0].nonzero(as_tuple=True)[0]              # (N_masked,)
    pred = logits[0] * patch_std[mask_idx] + patch_mean[mask_idx]        # un-norm per patch
    recon_patches[mask_idx] = pred

    # Depatchify → (T, C, H, W)
    recon = recon_patches.reshape(t_dim, h_dim, w_dim, tp, ps, ps, C)
    recon = recon.permute(0, 3, 6, 1, 4, 2, 5)                          # (t,tp,C,h,ps,w,ps)
    recon = recon.reshape(T, C, H, W)

    # Un-normalize ImageNet
    img_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    img_std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    recon = (recon * img_std + img_mean).clamp(0, 1)

    frame_arr = (recon[min(frame_t, T-1)].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(frame_arr)   # (224, 224, 3)


def run_video(input_path, out_dir, ag_model, ag_proc,
              tasks, score_thr, stride, device, ag_ratio, temporal_window=16,
              save_frames=False, videomae_recon=False):
    """
    Proper temporal video pipeline:

    1. Read all frames (strided).
    2. Process in chunks of `temporal_window` frames.
    3. One AutoGaze forward per chunk → per-frame gaze maps (T, 14, 14).
    4. For each frame: task inference (full + AG-masked) using the pre-computed mask.
    5. Compose 3-panel frame: [original+gaze_overlay | full_result | ag_result].
    6. Write per-task MP4.
    """
    from autogaze.models.autogaze.autogaze_cv import ConvFeatureSelector

    print(f"  reading frames (stride={stride})…")
    all_frames = _read_video_frames_all(input_path, stride)
    if not all_frames:
        print("  no frames found")
        return

    n_frames = len(all_frames)
    src_fps   = _get_video_fps(input_path)
    out_fps   = max(src_fps / stride, 1.0)
    print(f"  {n_frames} frames @ {out_fps:.1f} fps out  |  temporal_window={temporal_window}")

    video_tasks = [t for t in tasks if t in [
        "depth", "yolos", "dinov2", "segformer", "siglip", "videomae_cls", "xclip",
    ]]
    video_frames = {task: [] for task in video_tasks}

    # Create per-task frame directories if requested
    frame_dirs = {}
    if save_frames:
        for task in video_tasks:
            d = out_dir / "frames" / task
            d.mkdir(parents=True, exist_ok=True)
            frame_dirs[task] = d

    # ── chunk loop ────────────────────────────────────────────────────────────
    for chunk_start in range(0, n_frames, temporal_window):
        chunk = all_frames[chunk_start: chunk_start + temporal_window]
        actual_T = len(chunk)
        chunk_pil = [img for _, img in chunk]

        # Zero-pad last chunk to temporal_window so model sees its trained input size
        pad_T = temporal_window - actual_T
        if pad_T > 0:
            chunk_pil = chunk_pil + [chunk_pil[-1]] * pad_T

        # ── AutoGaze temporal inference: one forward for temporal_window frames ─
        ag_chunk = prep_video_chunk(chunk_pil, ag_proc, device)  # (1, temporal_window, C, 224, 224)
        with torch.no_grad():
            gaze_out = ag_model({"video": ag_chunk}, gazing_ratio=ag_ratio, generate_only=True)
        # gazing_mask[-1]: (B=1, temporal_window, 196) — finest 14×14 scale
        # only keep actual_T frames (discard padding)
        gaze_logits = gaze_out["gazing_mask"][-1][0, :actual_T]  # (actual_T, 196)
        gaze_maps = gaze_logits.float().cpu().numpy().reshape(actual_T, 14, 14)

        # ── Chunk-level action recognition (one result per clip) ──────────────
        # Both tasks need the full clip; result is displayed on every frame of the chunk.
        chunk_action = {}

        if "videomae_cls" in tasks:
            vmae_proc, vmae_model = _load_videomae_cls(device)
            vmae_frames = _pad_frames_to_T(chunk_pil, 16)
            vmae_pv = vmae_proc(vmae_frames, return_tensors="pt").pixel_values.to(device)
            t_dim_v = vmae_pv.shape[1] // vmae_model.config.tubelet_size  # 16//2=8

            def _topk_v(logits, k=5):
                probs = logits.softmax(-1)[0]
                top = probs.topk(k)
                return [(vmae_model.config.id2label[i.item()], p.item())
                        for i, p in zip(top.indices, top.values)]

            with torch.no_grad():
                vmae_top_full = _topk_v(vmae_model(vmae_pv).logits)

            # Aggregate spatial gaze over time → select top ag_ratio fraction
            agg = gaze_maps.mean(axis=0).reshape(196)
            n_keep = max(1, int(ag_ratio * 196))
            idx_keep = np.argsort(agg)[-n_keep:]
            sp_bool = np.zeros(196, dtype=bool); sp_bool[idx_keep] = True
            vmae_mask = (
                torch.from_numpy(sp_bool).unsqueeze(0).expand(t_dim_v, -1)
                .reshape(1, t_dim_v * 196).to(device)
            )

            def _vmae_hook(module, inp, out, mask=vmae_mask):
                return out * mask.float().unsqueeze(-1)

            hv = vmae_model.videomae.embeddings.register_forward_hook(_vmae_hook)
            try:
                with torch.no_grad():
                    vmae_top_ag = _topk_v(vmae_model(vmae_pv).logits)
            finally:
                hv.remove()
            chunk_action["videomae_cls"] = (vmae_top_full, vmae_top_ag)

        if "xclip" in tasks:
            xclip_proc, xclip_model = _load_xclip(device)
            xclip_frames = _pad_frames_to_T(chunk_pil, 8)

            def _xclip_infer(sp_mask=None):
                xinp = xclip_proc(text=XCLIP_ACTION_TEXTS, videos=xclip_frames,
                                  return_tensors="pt", padding=True).to(device)
                if sp_mask is not None:
                    def _xh(module, inp, out):
                        cls_tok = out[:, :1]
                        spt = out[:, 1:] * sp_mask.float().to(out.device).unsqueeze(0).unsqueeze(-1)
                        return torch.cat([cls_tok, spt], dim=1)
                    hx = xclip_model.vision_model.vision_model.embeddings.register_forward_hook(_xh)
                else:
                    hx = None
                try:
                    with torch.no_grad():
                        log = xclip_model(**xinp).logits_per_video
                finally:
                    if hx is not None: hx.remove()
                probs = log.softmax(-1)[0].cpu().numpy()
                return sorted(zip(XCLIP_ACTION_TEXTS, probs.tolist()), key=lambda x: -x[1])

            xclip_top_full = _xclip_infer()
            agg = gaze_maps.mean(axis=0).reshape(196)
            n_keep = max(1, int(ag_ratio * 196))
            idx_keep = np.argsort(agg)[-n_keep:]
            sp_bool_x = np.zeros(196, dtype=bool); sp_bool_x[idx_keep] = True
            xclip_top_ag = _xclip_infer(torch.from_numpy(sp_bool_x).float())
            chunk_action["xclip"] = (xclip_top_full, xclip_top_ag)

        # ── VideoMAE reconstruction: one pass per chunk, per frame ────────────
        # Reconstruct each frame using only the gaze-selected patches as visible.
        # Non-gaze patches are masked → VideoMAE fills them in.
        recon_pils = {}  # frame_idx → PIL Image (original frame resolution)
        if videomae_recon:
            vm_proc, vm_model = _load_videomae(device)
            for t, (frame_idx, orig_pil) in enumerate(chunk):
                recon_224 = reconstruct_with_videomae(
                    chunk_pil, gaze_maps[t], vm_proc, vm_model, device, frame_t=t
                )
                # Resize back to original frame resolution so aspect ratio matches other panels
                recon_pils[frame_idx] = recon_224.resize(orig_pil.size, Image.LANCZOS)

        # ── per-frame task inference (only actual frames, not padding) ──────────
        for t, (frame_idx, pil_img) in enumerate(chunk):
            gmap = gaze_maps[t]  # (14, 14) pre-computed gaze map for this frame
            n_sel = int((gmap > 0.5).sum())
            pil_gaze_ov = gaze_overlay_pil(pil_img, gmap)

            print(f"  chunk {chunk_start//temporal_window+1}  frame {frame_idx:04d}"
                  f"  gaze {n_sel}/196 ({n_sel/196*100:.0f}%)",
                  end="\r")

            # ── Depth ─────────────────────────────────────────────────────────
            if "depth" in tasks:
                proc, model = _load_depth(device)
                inp = proc(images=pil_img, return_tensors="pt")
                inp = {k: v.to(device) for k, v in inp.items()}
                ih, iw = inp["pixel_values"].shape[-2:]
                gh, gw = ih // 14, iw // 14
                # full tokens
                with torch.no_grad():
                    d_full = model(**inp).predicted_depth.squeeze().cpu().float().numpy()
                # ag-masked tokens — use pre-computed gaze map
                m = gaze_mask_for_grid(gmap, gh, gw, device=device)
                d_ag = _run_with_mask(
                    model.backbone.embeddings, m, True,
                    lambda: model(**inp).predicted_depth.squeeze().cpu().float().numpy(),
                )
                panels_d = [
                    (pil_gaze_ov,          "원본 + gaze"),
                    (depth_to_rgb(d_full), "Depth 전체 토큰"),
                    (depth_to_rgb(d_ag),   f"Depth AG {int(ag_ratio*100)}%"),
                ]
                if videomae_recon and frame_idx in recon_pils:
                    panels_d.append((recon_pils[frame_idx], "VideoMAE 재구성 원본"))
                    recon_inp = proc(images=recon_pils[frame_idx], return_tensors="pt")
                    recon_inp = {k: v.to(device) for k, v in recon_inp.items()}
                    with torch.no_grad():
                        d_recon = model(**recon_inp).predicted_depth.squeeze().cpu().float().numpy()
                    panels_d.append((depth_to_rgb(d_recon), "Depth VideoMAE 재구성"))
                panel = make_npanel(panels_d, target_size=pil_img.size)
                video_frames["depth"].append(panel)
                if save_frames:
                    panel.save(frame_dirs["depth"] / f"frame_{frame_idx:04d}.png")

            # ── YOLOS ─────────────────────────────────────────────────────────
            if "yolos" in tasks:
                proc, model = _load_yolos(device)
                inp = proc(images=pil_img, return_tensors="pt")
                inp = {k: v.to(device) for k, v in inp.items()}
                ih, iw = inp["pixel_values"].shape[-2:]
                gh, gw = ih // 16, iw // 16
                with torch.no_grad():
                    out_full = model(**inp)
                full_rgba, n_full = _detect_to_rgba(
                    pil_img, out_full.logits, out_full.pred_boxes,
                    model.config.id2label, score_thr)
                m = gaze_mask_for_grid(gmap, gh, gw, device=device)
                out_ag_result = _run_with_mask(model.vit.embeddings, m, True,
                                               lambda: model(**inp))
                ag_rgba, n_ag = _detect_to_rgba(
                    pil_img, out_ag_result.logits, out_ag_result.pred_boxes,
                    model.config.id2label, score_thr)
                panels_y = [
                    (pil_gaze_ov, "원본 + gaze"),
                    (full_rgba,   f"탐지 전체 ({n_full})"),
                    (ag_rgba,     f"탐지 AG ({n_ag}) {int(ag_ratio*100)}%"),
                ]
                if videomae_recon and frame_idx in recon_pils:
                    panels_y.append((recon_pils[frame_idx], "VideoMAE 재구성 원본"))
                    recon_inp_y = proc(images=recon_pils[frame_idx], return_tensors="pt")
                    recon_inp_y = {k: v.to(device) for k, v in recon_inp_y.items()}
                    with torch.no_grad():
                        out_recon_y = model(**recon_inp_y)
                    recon_rgba, n_recon_y = _detect_to_rgba(
                        recon_pils[frame_idx], out_recon_y.logits, out_recon_y.pred_boxes,
                        model.config.id2label, score_thr)
                    panels_y.append((recon_rgba, f"탐지 VideoMAE 재구성 ({n_recon_y})"))
                panel = make_npanel(panels_y, target_size=pil_img.size)
                video_frames["yolos"].append(panel)
                if save_frames:
                    panel.save(frame_dirs["yolos"] / f"frame_{frame_idx:04d}.png")

            # ── SegFormer ─────────────────────────────────────────────────────
            if "segformer" in tasks:
                proc, model = _load_segformer(device)
                inp = proc(images=pil_img, return_tensors="pt")
                inp = {k: v.to(device) for k, v in inp.items()}
                ih, iw = inp["pixel_values"].shape[-2:]
                gh, gw = ih // 4, iw // 4
                with torch.no_grad():
                    logits_full = model(**inp).logits
                up_full = F.interpolate(logits_full, size=(ih, iw), mode="bilinear", align_corners=False)
                seg_full = ADE20K_PALETTE[up_full.argmax(1)[0].cpu().numpy()]
                # ConvFeatureSelector needs a gaze mask tensor — build from pre-computed map
                mask_2d = torch.from_numpy(gmap).float()
                mask_2d = (mask_2d > 0.5).float().unsqueeze(0).unsqueeze(0)  # (1,1,14,14)
                if gh != 14 or gw != 14:
                    mask_2d = F.interpolate(mask_2d, size=(gh, gw), mode="bilinear",
                                            align_corners=False)
                stage0_proj = model.segformer.encoder.patch_embeddings[0].proj

                def _seg_hook(m_mod, inp_h, out_h):
                    B, C, H, W = out_h.shape
                    m2 = F.interpolate(mask_2d.to(device), size=(H, W), mode="nearest")
                    return out_h * m2

                handle = stage0_proj.register_forward_hook(_seg_hook)
                try:
                    with torch.no_grad():
                        logits_ag = model(**inp).logits
                finally:
                    handle.remove()
                up_ag = F.interpolate(logits_ag, size=(ih, iw), mode="bilinear", align_corners=False)
                seg_ag = ADE20K_PALETTE[up_ag.argmax(1)[0].cpu().numpy()]
                panels_s = [
                    (pil_gaze_ov, "원본 + gaze"),
                    (seg_full,    "세그 전체 토큰"),
                    (seg_ag,      f"세그 AG {int(ag_ratio*100)}%"),
                ]
                if videomae_recon and frame_idx in recon_pils:
                    panels_s.append((recon_pils[frame_idx], "VideoMAE 재구성 원본"))
                    recon_inp_s = proc(images=recon_pils[frame_idx], return_tensors="pt")
                    recon_inp_s = {k: v.to(device) for k, v in recon_inp_s.items()}
                    with torch.no_grad():
                        logits_recon = model(**recon_inp_s).logits
                    recon_up = F.interpolate(logits_recon,
                                             size=recon_pils[frame_idx].size[::-1],
                                             mode="bilinear", align_corners=False)
                    seg_recon = ADE20K_PALETTE[recon_up.argmax(1)[0].cpu().numpy()]
                    panels_s.append((seg_recon, "세그 VideoMAE 재구성"))
                panel = make_npanel(panels_s, target_size=pil_img.size)
                video_frames["segformer"].append(panel)
                if save_frames:
                    panel.save(frame_dirs["segformer"] / f"frame_{frame_idx:04d}.png")

            # ── DINOv2 ────────────────────────────────────────────────────────
            if "dinov2" in tasks:
                proc, model = _load_dinov2(device)
                inp = proc(images=pil_img, return_tensors="pt")
                inp = {k: v.to(device) for k, v in inp.items()}
                ih, iw = inp["pixel_values"].shape[-2:]
                gh, gw = ih // 14, iw // 14

                def _topk_dino(logits, k=5):
                    probs = logits.softmax(-1)[0]
                    top = probs.topk(k)
                    return [(model.config.id2label[i.item()], p.item())
                            for i, p in zip(top.indices, top.values)]

                with torch.no_grad():
                    top_full = _topk_dino(model(**inp).logits)
                m = gaze_mask_for_grid(gmap, gh, gw, device=device)
                top_ag = _topk_dino(
                    _run_with_mask(model.dinov2.embeddings, m, True,
                                   lambda: model(**inp).logits)
                )
                full_img = _cls_overlay_pil(pil_img, top_full,
                                            f"top-1: {top_full[0][0][:20]}")
                ag_img   = _cls_overlay_pil(pil_img, top_ag,
                                            f"top-1: {top_ag[0][0][:20]}")
                panels_dn = [
                    (pil_gaze_ov, "원본 + gaze"),
                    (full_img,    "DINOv2 전체 토큰"),
                    (ag_img,      f"DINOv2 AG {int(ag_ratio*100)}%"),
                ]
                if videomae_recon and frame_idx in recon_pils:
                    panels_dn.append((recon_pils[frame_idx], "VideoMAE 재구성 원본"))
                    recon_inp_dn = proc(images=recon_pils[frame_idx], return_tensors="pt")
                    recon_inp_dn = {k: v.to(device) for k, v in recon_inp_dn.items()}
                    top_recon_dn = _topk_dino(
                        _run_with_mask(model.dinov2.embeddings,
                                       gaze_mask_for_grid(gmap, gh, gw, device=device),
                                       True, lambda: model(**recon_inp_dn).logits)
                    )
                    recon_img_dn = _cls_overlay_pil(recon_pils[frame_idx], top_recon_dn,
                                                    f"top-1: {top_recon_dn[0][0][:20]}")
                    panels_dn.append((recon_img_dn, "DINOv2 VideoMAE 재구성"))
                panel = make_npanel(panels_dn, target_size=pil_img.size)
                video_frames["dinov2"].append(panel)
                if save_frames:
                    panel.save(frame_dirs["dinov2"] / f"frame_{frame_idx:04d}.png")

            # ── SigLIP ────────────────────────────────────────────────────────
            if "siglip" in tasks:
                proc, model = _load_siglip(device)
                GRID = 14  # 224 // 16

                def _siglip_probs(mask_ctx=None):
                    inputs_s = proc(text=SIGLIP_TEXTS, images=[pil_img],
                                    return_tensors="pt", padding="max_length").to(device)
                    if mask_ctx:
                        with mask_ctx:
                            with torch.no_grad():
                                out = model(**inputs_s)
                    else:
                        with torch.no_grad():
                            out = model(**inputs_s)
                    return out.logits_per_image.softmax(-1)[0].cpu().numpy()

                probs_full = _siglip_probs()
                m = gaze_mask_for_grid(gmap, GRID, GRID, device=device)
                from autogaze.models.autogaze.autogaze_cv import AutoGazeTokenSelector as _AGS
                inst = object.__new__(_AGS)
                probs_ag = _siglip_probs(
                    _AGS.token_mask_context(inst, model.vision_model.embeddings, m,
                                            has_cls_token=False)
                )
                top_full_s = sorted(zip(SIGLIP_TEXTS, probs_full.tolist()),
                                    key=lambda x: -x[1])
                top_ag_s   = sorted(zip(SIGLIP_TEXTS, probs_ag.tolist()),
                                    key=lambda x: -x[1])
                full_img_s = _cls_overlay_pil(pil_img, top_full_s,
                                              f"top-1: {top_full_s[0][0][:25]}")
                ag_img_s   = _cls_overlay_pil(pil_img, top_ag_s,
                                              f"top-1: {top_ag_s[0][0][:25]}")
                panels_sl = [
                    (pil_gaze_ov, "원본 + gaze"),
                    (full_img_s,  "SigLIP 전체 토큰"),
                    (ag_img_s,    f"SigLIP AG {int(ag_ratio*100)}%"),
                ]
                if videomae_recon and frame_idx in recon_pils:
                    panels_sl.append((recon_pils[frame_idx], "VideoMAE 재구성 원본"))
                    def _siglip_recon_probs():
                        inputs_sr = proc(text=SIGLIP_TEXTS, images=[recon_pils[frame_idx]],
                                         return_tensors="pt", padding="max_length").to(device)
                        with torch.no_grad():
                            out_sr = model(**inputs_sr)
                        return out_sr.logits_per_image.softmax(-1)[0].cpu().numpy()
                    probs_recon_sl = _siglip_recon_probs()
                    top_recon_sl = sorted(zip(SIGLIP_TEXTS, probs_recon_sl.tolist()),
                                          key=lambda x: -x[1])
                    recon_img_sl = _cls_overlay_pil(recon_pils[frame_idx], top_recon_sl,
                                                    f"top-1: {top_recon_sl[0][0][:25]}")
                    panels_sl.append((recon_img_sl, "SigLIP VideoMAE 재구성"))
                panel = make_npanel(panels_sl, target_size=pil_img.size)
                video_frames["siglip"].append(panel)
                if save_frames:
                    panel.save(frame_dirs["siglip"] / f"frame_{frame_idx:04d}.png")

            # ── VideoMAE-CLS ──────────────────────────────────────────────────
            if "videomae_cls" in tasks and "videomae_cls" in chunk_action:
                top_fv, top_av = chunk_action["videomae_cls"]
                full_img_v = _cls_overlay_pil(pil_img, top_fv,
                                              f"top-1: {top_fv[0][0][:25]}")
                ag_img_v   = _cls_overlay_pil(pil_img, top_av,
                                              f"top-1: {top_av[0][0][:25]}")
                panel = make_npanel([
                    (pil_gaze_ov, "원본 + gaze"),
                    (full_img_v,  "VideoMAE-CLS 전체"),
                    (ag_img_v,    f"VideoMAE-CLS AG {int(ag_ratio*100)}%"),
                ], target_size=pil_img.size)
                video_frames["videomae_cls"].append(panel)
                if save_frames:
                    panel.save(frame_dirs["videomae_cls"] / f"frame_{frame_idx:04d}.png")

            # ── X-CLIP ────────────────────────────────────────────────────────
            if "xclip" in tasks and "xclip" in chunk_action:
                top_fx, top_ax = chunk_action["xclip"]
                full_img_x = _cls_overlay_pil(pil_img, top_fx,
                                              f"top-1: {top_fx[0][0][:25]}")
                ag_img_x   = _cls_overlay_pil(pil_img, top_ax,
                                              f"top-1: {top_ax[0][0][:25]}")
                panel = make_npanel([
                    (pil_gaze_ov, "원본 + gaze"),
                    (full_img_x,  "X-CLIP 전체"),
                    (ag_img_x,    f"X-CLIP AG {int(ag_ratio*100)}%"),
                ], target_size=pil_img.size)
                video_frames["xclip"].append(panel)
                if save_frames:
                    panel.save(frame_dirs["xclip"] / f"frame_{frame_idx:04d}.png")

    print()

    # ── write output videos ───────────────────────────────────────────────────
    for task_key, frame_list in video_frames.items():
        if not frame_list:
            continue
        mp4_path = out_dir / f"{task_key}_video.mp4"
        _frames_to_mp4(mp4_path, frame_list, out_fps)
        print(f"  saved: {mp4_path}  ({len(frame_list)} frames)")

    if save_frames:
        for task_key, d in frame_dirs.items():
            n = len(list(d.glob("*.png")))
            print(f"  frames/{task_key}/  ({n} PNGs)")


def _run_with_mask(embed_module, mask, has_cls_token, fn):
    """Apply forward hook mask to embed_module, run fn() under no_grad, return result."""
    from autogaze.models.autogaze.autogaze_cv import AutoGazeTokenSelector
    inst = object.__new__(AutoGazeTokenSelector)
    with AutoGazeTokenSelector.token_mask_context(inst, embed_module, mask, has_cls_token):
        with torch.no_grad():
            return fn()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def is_video(path):
    return Path(path).suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def main():
    args = parse_args()

    # device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # output dir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output: {out_dir}")

    # load AutoGaze
    ag_model, ag_proc, selector = _load_ag(args.ag_path, device)

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"Input not found: {input_path}")

    if is_video(input_path):
        print(f"[video mode] stride={args.stride}  temporal_window={args.temporal_window}")
        run_video(input_path, out_dir, ag_model, ag_proc,
                  args.tasks, args.score_thr, args.stride,
                  device, args.ag_ratio, args.temporal_window,
                  save_frames=args.save_frames,
                  videomae_recon=args.videomae_recon)
        return

    # ── Image mode ───────────────────────────────────────────────────────────
    pil_img  = Image.open(input_path).convert("RGB")
    ag_video = prep_for_autogaze(pil_img, ag_proc, device)
    gaze_map = compute_gaze_map(ag_model, ag_video, args.ag_ratio)

    # save gaze overlay
    gaze_out = out_dir / "gaze_map.png"
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].imshow(pil_img); axes[0].set_title("원본"); axes[0].axis("off")
    im = axes[1].imshow(gaze_map, cmap="hot", vmin=0, vmax=1)
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    axes[1].set_title("AutoGaze 14×14 gaze map"); axes[1].set_xticks([]); axes[1].set_yticks([])
    axes[2].imshow(gaze_overlay_pil(pil_img, gaze_map))
    axes[2].set_title("이미지 + gaze 오버레이"); axes[2].axis("off")
    n_sel = int((gaze_map > 0.5).sum())
    fig.suptitle(f"AutoGaze gaze map  |  선택 토큰: {n_sel}/196 ({n_sel/196*100:.0f}%)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(); fig.savefig(gaze_out, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"  saved: {gaze_out}")

    all_metrics = {}

    if "depth" in args.tasks:
        print("[depth] running…")
        depth_res, depth_metrics, _ = run_depth(pil_img, ag_video, ag_model, args.ratios, device)
        all_metrics["depth"] = depth_metrics
        save_comparison_grid(
            out_dir / "depth_comparison.png", pil_img, gaze_map, depth_res,
            "Depth Anything V2 — AutoGaze ratio별 depth map 비교", args.ratios,
        )

    if "yolos" in args.tasks:
        print("[yolos] running…")
        yolos_res, yolos_metrics, _ = run_yolos(
            pil_img, ag_video, ag_model, args.ratios, device, args.score_thr
        )
        all_metrics["yolos"] = yolos_metrics
        save_comparison_grid(
            out_dir / "detection_comparison.png", pil_img, gaze_map, yolos_res,
            "YOLOS 객체 탐지 — AutoGaze ratio별 비교", args.ratios,
        )

    if "dinov2" in args.tasks:
        print("[dinov2] running…")
        full_top_dict, ratio_top5, union_labels, dino_metrics, _ = run_dinov2(
            pil_img, ag_video, ag_model, args.ratios, device
        )
        all_metrics["dinov2"] = dino_metrics
        save_dinov2_chart(out_dir / "recognition_comparison.png",
                          full_top_dict, ratio_top5, union_labels, args.ratios)

    if "segformer" in args.tasks:
        print("[segformer] running…")
        seg_res, seg_metrics, _ = run_segformer(pil_img, ag_video, ag_model, args.ratios, device)
        all_metrics["segformer"] = seg_metrics
        save_comparison_grid(
            out_dir / "segmentation_comparison.png", pil_img, gaze_map, seg_res,
            "SegFormer-B2 ADE20K — AutoGaze ratio별 세그멘테이션", args.ratios,
        )

    if "siglip" in args.tasks:
        print("[siglip] running…")
        probs_full, ratio_probs, texts, siglip_metrics = run_siglip(
            pil_img, ag_video, ag_model, args.ratios, device
        )
        all_metrics["siglip"] = siglip_metrics
        save_siglip_chart(out_dir / "siglip_comparison.png", probs_full, ratio_probs, texts, args.ratios)

    if "videomae_cls" in args.tasks:
        print("[videomae_cls] running…")
        chunk_pil_v = [pil_img] * 16   # simulate 16-frame clip from a still image
        top_full_v, ratio_top5_v, vmae_metrics = run_videomae_cls(
            chunk_pil_v, ag_video, ag_model, args.ratios, device
        )
        all_metrics["videomae_cls"] = vmae_metrics
        save_action_chart(
            out_dir / "videomae_cls_comparison.png",
            top_full_v, ratio_top5_v, args.ratios,
            "VideoMAE-CLS Kinetics-400 동작 인식 — AutoGaze ratio별",
        )

    if "xclip" in args.tasks:
        print("[xclip] running…")
        chunk_pil_x = [pil_img] * 8   # simulate 8-frame clip from a still image
        top_full_x, ratio_top5_x, xclip_metrics = run_xclip(
            chunk_pil_x, ag_video, ag_model, args.ratios, device
        )
        all_metrics["xclip"] = xclip_metrics
        save_action_chart(
            out_dir / "xclip_comparison.png",
            top_full_x, ratio_top5_x, args.ratios,
            "X-CLIP Zero-shot 동작 인식 — AutoGaze ratio별",
        )

    save_summary_chart(out_dir / "summary.png", all_metrics, args.ratios)

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"  saved: {metrics_path}")

    print(f"\nDone. Results in: {out_dir}")


if __name__ == "__main__":
    main()
