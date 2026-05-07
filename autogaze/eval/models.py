# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
MLLM runner registry for video QA benchmarks.

Built-in runners:
  - nvila           NVILA-8B with native AutoGaze processor integration
  - qwen25vl        Qwen2.5-VL-7B with AutoGaze applied via zero-shot forward hook
  - qwen25vl_full   Qwen2.5-VL-7B with AutoGaze full ViT integration (per-temporal-chunk)
  - vjepa2          V-JEPA2 encoder with AutoGaze applied via zero-shot forward hook
  - vjepa2_full     V-JEPA2 encoder with AutoGaze full integration (per-temporal-group)
  - siglip          Vanilla HuggingFace SigLIP, feature extraction (optional AutoGaze hook)

Usage (API)
-----------
    from autogaze.eval.models import load_runner, RUNNERS

    runner = load_runner(
        mllm="nvila",
        model_path="weights/NVILA-8B-HD-Video",
        autogaze_path="weights/AutoGaze",     # None → AutoGaze OFF
        gazing_ratio=0.75,
    )
    answer = runner.run(frames, prompt, max_new_tokens=16)

Adding a new MLLM
-----------------
    1. Subclass BaseMLLMRunner and implement load() + run().
    2. Register it: RUNNERS["your-key"] = YourRunner

Architecture notes (Qwen2.5-VL)
--------------------------------
Qwen2.5-VL uses a 3D patch embedding (temporal_patch_size=2, patch_size=14).
After `model.visual.patch_embed`, patch tokens are shaped (N_total, C) where
N_total = T_p * H_p * W_p (no leading batch dimension).

AutoGaze gaze masks are computed at 14×14 spatial resolution and averaged
across all T input frames.  The spatial mask is then:
  1. Bilinearly interpolated to the Qwen patch grid (H_p × W_p).
  2. Tiled across T_p temporal chunks.
  3. Applied via a custom forward hook on model.visual.patch_embed — zeroing
     non-gazed patches so that the rest of the vision encoder only attends to
     selected tokens (the same zero-shot mechanism as AutoGazeTokenSelector).

Architecture notes (V-JEPA2)
------------------------------
V-JEPA2 is a pure video encoder (no LLM component).  It uses a 3D patch
embedding (tubelet_size=2, patch_size=16).  Output shape: (B, N, C) where
N = T_p * H_p * W_p (T_p = T / tubelet_size).

Unlike Qwen2.5-VL, V-JEPA2 has full cross-temporal attention across all
patches.  AutoGaze zeroes non-gazed patch embeddings before the transformer
layers (zeroed value vectors contribute nothing to attention outputs).

For MCQ benchmarks, VJEPA2Runner requires a trained projector + LLM head
via the `lm_path` constructor argument.  Feature-extraction usage (encode
video to patch features) is available via ``runner.encode_video()``.
"""

from __future__ import annotations

import io
import logging
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from PIL import Image

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Base class
# ─────────────────────────────────────────────────────────────────────────────

class BaseMLLMRunner:
    """Abstract base for MLLM runners."""

    #: Name shown in log / output JSON
    name: str = "base"

    #: True if run() supports MCQ text generation (has a paired LLM).
    #: False for feature-extraction-only runners (e.g. SigLIPRunner, VJEPA2Runner).
    supports_mcq: bool = True

    def run(
        self,
        frames: List[Image.Image],
        prompt: str,
        max_new_tokens: int = 16,
    ) -> str:
        """Run one sample inference, return the raw generated string."""
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# NVILA runner  (native AutoGaze processor)
# ─────────────────────────────────────────────────────────────────────────────

class NVILARunner(BaseMLLMRunner):
    """NVILA with native AutoGaze processor integration.

    When *autogaze_path* is None the processor is loaded without AutoGaze,
    which produces a full-patch baseline result.
    """

    name = "nvila"

    def __init__(
        self,
        model_path: str,
        autogaze_path: Optional[str],
        gazing_ratio: float,
        dtype: torch.dtype = torch.bfloat16,
    ):
        from transformers import AutoProcessor, AutoModel

        proc_kwargs: Dict[str, Any] = dict(trust_remote_code=True)
        if autogaze_path is not None:
            proc_kwargs.update(
                autogaze_model_id=autogaze_path,
                gazing_ratio_tile=gazing_ratio,
                gazing_ratio_thumbnail=gazing_ratio,
            )

        log.info("NVILARunner: loading processor from %s", model_path)
        self.processor = AutoProcessor.from_pretrained(model_path, **proc_kwargs)

        log.info("NVILARunner: loading model from %s (dtype=%s)", model_path, dtype)
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto",
        )
        self.model.eval()

    def run(
        self,
        frames: List[Image.Image],
        prompt: str,
        max_new_tokens: int = 16,
    ) -> str:
        inputs = self.processor(
            text=[prompt],
            videos=[frames],
            return_tensors="pt",
            padding=True,
        )
        device = next(self.model.parameters()).device
        inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        with torch.inference_mode():
            gen_ids = self.model.generate(
                **{k: v for k, v in inputs.items() if k != "labels"},
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        prompt_len = inputs["input_ids"].shape[1]
        return self.processor.tokenizer.decode(
            gen_ids[0][prompt_len:], skip_special_tokens=True
        )


# ─────────────────────────────────────────────────────────────────────────────
# Qwen2.5-VL runner  (two AutoGaze integration modes)
# ─────────────────────────────────────────────────────────────────────────────

_AG_GRID = 14          # AutoGaze produces 14×14 = 196 spatial gaze scores
_AG_THRESHOLD = 0.5


class Qwen25VLRunner(BaseMLLMRunner):
    """Qwen2.5-VL with AutoGaze, supporting two integration modes.

    integration='hook'  (zero-shot, default)
        AutoGaze runs on all frames, the mean gaze map is bilinearly interpolated
        to the Qwen patch grid (H_p × W_p), tiled across T_p temporal chunks,
        and applied via a forward hook on ``model.visual.patch_embed``.
        No model modification required; works as a zero-shot drop-in.

    integration='full'  (full integration, follows INTEGRATION.md)
        ``model.visual`` is replaced with ``AutoGazeQwen25VisionTransformer``,
        which applies gaze masks inside ``forward()`` after ``patch_embed``.

        Key improvement over the hook:
          - Per-temporal-chunk gaze maps: AutoGaze's T input frames are grouped
            into T_p = T / temporal_patch_size chunks.  Each chunk gets its own
            gaze map by averaging the corresponding input-frame gaze scores.
            This respects temporal variation in scene saliency.
          - Gaze applied before window-reordering, matching the patch layout.

        Architecture note: Qwen2.5-VL's visual encoder has NO cross-temporal
        attention (each chunk is processed independently by cu_seqlens), so no
        block-causal attention mask is needed — unlike the NVILA/SigLIP integration.

    When *autogaze_path* is None all patches are kept (baseline mode); the
    integration flag is ignored.
    """

    name = "qwen25vl"

    TEMPORAL_PATCH_SIZE = 2   # Qwen2.5-VL default
    PATCH_SIZE = 14

    def __init__(
        self,
        model_path: str,
        autogaze_path: Optional[str],
        gazing_ratio: float,
        dtype: torch.dtype = torch.bfloat16,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1280 * 28 * 28,
        integration: str = "hook",
    ):
        """
        Args:
            integration: ``'hook'`` (zero-shot, default) or ``'full'``
                         (modified visual encoder with per-temporal-chunk gaze).
        """
        if integration not in ("hook", "full"):
            raise ValueError(f"integration must be 'hook' or 'full', got '{integration}'")

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        except ImportError:
            raise ImportError(
                "transformers>=4.45 required for Qwen2.5-VL. "
                "pip install transformers>=4.45"
            )

        self.integration = integration

        log.info("Qwen25VLRunner: loading processor from %s", model_path)
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

        log.info("Qwen25VLRunner: loading model from %s (dtype=%s, integration=%s)",
                 model_path, dtype, integration)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
        )
        self.model.eval()

        # For integration='full', swap model.visual to the AutoGaze-aware subclass.
        # This is a class monkey-patch: no weights change, only forward() is overridden.
        if integration == "full":
            from autogaze.vision_encoders.qwen25vl import AutoGazeQwen25VisionTransformer
            self.model.visual.__class__ = AutoGazeQwen25VisionTransformer
            self.model.visual._gazing_info = None
            log.info("Qwen25VLRunner: model.visual patched → AutoGazeQwen25VisionTransformer")

        self.autogaze_path = autogaze_path
        self.gazing_ratio = gazing_ratio
        self.selector = None

        if autogaze_path is not None:
            self._load_autogaze(autogaze_path, gazing_ratio)

    # ------------------------------------------------------------------ #
    # AutoGaze model loading
    # ------------------------------------------------------------------ #

    def _load_autogaze(self, autogaze_path: str, gazing_ratio: float) -> None:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))

        from autogaze.models.autogaze import AutoGaze, AutoGazeImageProcessor
        from autogaze.models.autogaze.autogaze_cv import AutoGazeTokenSelector

        log.info("Qwen25VLRunner: loading AutoGaze from %s", autogaze_path)
        ag_model = AutoGaze.from_pretrained(autogaze_path)
        ag_model.eval()
        ag_model = ag_model.to(next(self.model.parameters()).device)

        self.selector = AutoGazeTokenSelector(ag_model, gazing_ratio=gazing_ratio)
        self.ag_processor = AutoGazeImageProcessor.from_pretrained(autogaze_path)

        log.info("Qwen25VLRunner: AutoGaze ready (gazing_ratio=%.2f, integration=%s)",
                 gazing_ratio, self.integration)

    # ------------------------------------------------------------------ #
    # AutoGaze input preprocessing
    # ------------------------------------------------------------------ #

    def _frames_to_ag_tensor(self, frames: List[Image.Image]) -> torch.Tensor:
        """Preprocess PIL frames for AutoGaze: (1, T, C, 224, 224)."""
        processed = self.ag_processor(images=frames, return_tensors="pt")
        pv = processed["pixel_values"]  # (T, C, H, W) at 224×224
        return pv.unsqueeze(0).to(next(self.model.parameters()).device)

    @torch.no_grad()
    def _run_autogaze(self, frames: List[Image.Image]) -> torch.Tensor:
        """Run AutoGaze on frames → raw gaze map (1, T, 14, 14) float."""
        ag_video = self._frames_to_ag_tensor(frames)
        gaze_out = self.selector.ag(
            {'video': ag_video},
            gazing_ratio=self.selector.gazing_ratio,
            generate_only=True,
        )
        raw = gaze_out['gazing_mask'][-1].float()         # (1, T, 196)
        T = raw.shape[1]
        return raw.reshape(1, T, _AG_GRID, _AG_GRID)      # (1, T, 14, 14)

    # ------------------------------------------------------------------ #
    # Gaze mask — hook mode (single spatial map, tiled over all T_p chunks)
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _flat_gaze_mask_hook(
        self,
        frames: List[Image.Image],
        T_p: int,
        H_p: int,
        W_p: int,
    ) -> torch.Tensor:
        """(T_p * H_p * W_p,) float mask — mean of all T frames, tiled.

        Used by integration='hook'.
        """
        gaze_map = self._run_autogaze(frames)          # (1, T, 14, 14)
        mean_map = gaze_map.mean(dim=1)                # (1, 14, 14)

        if H_p != _AG_GRID or W_p != _AG_GRID:
            mean_map = F.interpolate(
                mean_map.unsqueeze(1), size=(H_p, W_p),
                mode='bilinear', align_corners=False,
            ).squeeze(1)                               # (1, H_p, W_p)

        spatial = (mean_map > _AG_THRESHOLD).float().reshape(-1)        # (H_p * W_p,)
        return spatial.unsqueeze(0).expand(T_p, -1).reshape(-1)         # (T_p * H_p * W_p,)

    # ------------------------------------------------------------------ #
    # Gaze mask — full mode (per-temporal-chunk maps)
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _gazing_info_full(
        self,
        frames: List[Image.Image],
        T_p: int,
        H_p: int,
        W_p: int,
    ) -> dict:
        """Per-temporal-chunk gaze mask as expected by AutoGazeQwen25VisionTransformer.

        Strategy: AutoGaze outputs T_ag per-frame gaze maps.  These are grouped
        into T_p = T_ag / temporal_patch_size Qwen temporal chunks by averaging
        the gaze scores of the frames belonging to each chunk.  Each chunk's map
        is interpolated to (H_p, W_p) and thresholded.

        Returns:
            dict with key ``'patch_mask'``: ``(T_p * H_p * W_p,)`` float32 tensor
            where 1.0 = keep and 0.0 = zero, in row-major (t, h, w) order.
        """
        gaze_map = self._run_autogaze(frames)          # (1, T_ag, 14, 14)
        T_ag = gaze_map.shape[1]
        chunk_size = max(1, T_ag // T_p)

        chunk_masks = []
        for t_p in range(T_p):
            # Average gaze over the AutoGaze frames belonging to this Qwen chunk
            start = min(t_p * chunk_size, T_ag - 1)
            end   = min((t_p + 1) * chunk_size, T_ag)
            chunk = gaze_map[0, start:end].mean(dim=0, keepdim=True)  # (1, 14, 14)

            if H_p != _AG_GRID or W_p != _AG_GRID:
                chunk = F.interpolate(
                    chunk.unsqueeze(0), size=(H_p, W_p),
                    mode='bilinear', align_corners=False,
                ).squeeze(0)                            # (1, H_p, W_p)

            chunk_masks.append((chunk.squeeze(0) > _AG_THRESHOLD).float().reshape(-1))

        # Concatenate in row-major (t, h, w) order: (T_p * H_p * W_p,)
        return {'patch_mask': torch.cat(chunk_masks, dim=0)}

    # ------------------------------------------------------------------ #
    # Hook context manager  (hook mode only)
    # ------------------------------------------------------------------ #

    @contextmanager
    def _patch_embed_hook(self, flat_mask: torch.Tensor):
        """Hook model.visual.patch_embed output → zero non-gazed patches.

        Qwen2.5-VL's patch_embed outputs (N_total, C) with no batch dim.
        """
        mask = flat_mask

        def _hook(module, input, output):
            N = min(output.shape[0], mask.shape[0])
            result = output.clone()
            result[:N] = result[:N] * mask[:N].float().unsqueeze(-1)
            return result

        handle = self.model.visual.patch_embed.register_forward_hook(_hook)
        try:
            yield
        finally:
            handle.remove()

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def run(
        self,
        frames: List[Image.Image],
        prompt: str,
        max_new_tokens: int = 16,
    ) -> str:
        messages = [
            {
                "role": "user",
                "content": [{"type": "video"}, {"type": "text", "text": prompt}],
            }
        ]
        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text_input], videos=[frames],
            return_tensors="pt", padding=True,
        )
        device = next(self.model.parameters()).device
        inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        gen_kwargs = {k: v for k, v in inputs.items() if k != "labels"}
        gen_kwargs.update(max_new_tokens=max_new_tokens, do_sample=False,
                          temperature=None, top_p=None)

        # ── dispatch to correct AutoGaze mode ─────────────────────────── #
        if self.selector is None:
            # Baseline: no AutoGaze
            with torch.inference_mode():
                gen_ids = self.model.generate(**gen_kwargs)

        elif self.integration == "full":
            # Full integration: inject gazing_info into model.visual
            grid_thw = inputs.get("video_grid_thw")
            if grid_thw is not None and len(grid_thw) > 0:
                T_p = int(grid_thw[0][0])
                H_p = int(grid_thw[0][1])
                W_p = int(grid_thw[0][2])
                self.model.visual._gazing_info = self._gazing_info_full(
                    frames, T_p, H_p, W_p
                )
            else:
                log.warning("Qwen25VLRunner[full]: video_grid_thw not found — no gaze applied")
            with torch.inference_mode():
                gen_ids = self.model.generate(**gen_kwargs)

        else:
            # Hook mode: apply gaze via patch_embed forward hook
            grid_thw = inputs.get("video_grid_thw")
            if grid_thw is not None and len(grid_thw) > 0:
                T_p = int(grid_thw[0][0])
                H_p = int(grid_thw[0][1])
                W_p = int(grid_thw[0][2])
                flat_mask = self._flat_gaze_mask_hook(frames, T_p, H_p, W_p)
                ctx = self._patch_embed_hook(flat_mask)
            else:
                log.warning("Qwen25VLRunner[hook]: video_grid_thw not found — no gaze applied")
                from contextlib import nullcontext
                ctx = nullcontext()
            with ctx, torch.inference_mode():
                gen_ids = self.model.generate(**gen_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        return self.processor.tokenizer.decode(
            gen_ids[0][prompt_len:], skip_special_tokens=True
        )


# ─────────────────────────────────────────────────────────────────────────────
# V-JEPA2 runner  (pure video encoder, two AutoGaze integration modes)
# ─────────────────────────────────────────────────────────────────────────────

class VJEPA2Runner(BaseMLLMRunner):
    """V-JEPA2 video encoder with AutoGaze, supporting two integration modes.

    V-JEPA2 is a pure video encoder (no LLM component).  For MCQ benchmarks it
    requires a trained projector + LLM specified via *lm_path*.  For feature-
    extraction usage (e.g., retrieval or probe tasks) use ``encode_video()``.

    integration='hook'  (zero-shot, default)
        AutoGaze runs on all frames; the mean gaze map is interpolated to the
        V-JEPA2 patch grid (H_p × W_p) and tiled across T_p temporal groups.
        Applied via a forward hook on ``model.encoder.embeddings.patch_embeddings``.
        No model modification required.

    integration='full'  (full integration, follows INTEGRATION.md)
        ``model.encoder`` is replaced with ``AutoGazeVJEPA2Encoder``, which
        applies per-temporal-group gaze masks inside ``forward()`` after
        ``self.embeddings()``.

        Key improvement over the hook:
          - Per-temporal-group gaze maps: T_ag AutoGaze frames are grouped into
            T_p = T / tubelet_size groups.  Each group gets its own gaze map by
            averaging the corresponding input-frame gaze scores.
          - Zeroing applied before the transformer layer stack.

        Architecture note: V-JEPA2 has full cross-temporal attention.  We use
        the zeroing approach rather than an attention mask, since the attention
        layers do not expose an additive attention-mask parameter.  Zeroed value
        vectors contribute nothing to the attention output.

    When *autogaze_path* is None all patches are kept (baseline mode); the
    integration flag is ignored.
    """

    name = "vjepa2"
    supports_mcq = False   # feature-extraction only; use VJEPA2LLMRunner for MCQ

    TUBELET_SIZE = 2    # V-JEPA2 default
    PATCH_SIZE   = 16

    def __init__(
        self,
        model_path: str,
        autogaze_path: Optional[str],
        gazing_ratio: float,
        dtype: torch.dtype = torch.bfloat16,
        lm_path: Optional[str] = None,
        integration: str = "hook",
    ):
        """
        Args:
            model_path:    HF hub ID or local path to V-JEPA2 weights.
            autogaze_path: Path to AutoGaze weights, or None for baseline.
            gazing_ratio:  Fraction of patches retained per frame (0–1).
            dtype:         Torch dtype for V-JEPA2 (default: bfloat16).
            lm_path:       Optional: path / HF ID for a paired LLM + projector
                           (required for MCQ ``run()`` calls; None = features-only).
            integration:   ``'hook'`` (zero-shot, default) or ``'full'``
                           (modified encoder with per-temporal-group gaze).
        """
        if integration not in ("hook", "full"):
            raise ValueError(f"integration must be 'hook' or 'full', got '{integration}'")

        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError:
            raise ImportError("transformers>=4.53 required for V-JEPA2.")

        self.integration = integration

        log.info("VJEPA2Runner: loading model from %s (dtype=%s, integration=%s)",
                 model_path, dtype, integration)
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
        )
        self.model.eval()

        # Try to load a processor/image processor for frame preprocessing
        try:
            self.processor = AutoProcessor.from_pretrained(model_path)
        except Exception:
            self.processor = None
            log.warning("VJEPA2Runner: no processor found at %s — use _frames_to_tensor directly",
                        model_path)

        # For integration='full', swap model.encoder to AutoGaze-aware subclass.
        if integration == "full":
            from autogaze.vision_encoders.vjepa2 import AutoGazeVJEPA2Encoder
            self.model.encoder.__class__ = AutoGazeVJEPA2Encoder
            self.model.encoder._gazing_info = None
            log.info("VJEPA2Runner: model.encoder patched → AutoGazeVJEPA2Encoder")

        self.autogaze_path = autogaze_path
        self.gazing_ratio = gazing_ratio
        self.selector = None

        if autogaze_path is not None:
            self._load_autogaze(autogaze_path, gazing_ratio)

    # ------------------------------------------------------------------ #
    # AutoGaze model loading
    # ------------------------------------------------------------------ #

    def _load_autogaze(self, autogaze_path: str, gazing_ratio: float) -> None:
        from autogaze.models.autogaze import AutoGaze, AutoGazeImageProcessor
        from autogaze.models.autogaze.autogaze_cv import AutoGazeTokenSelector

        log.info("VJEPA2Runner: loading AutoGaze from %s", autogaze_path)
        ag_model = AutoGaze.from_pretrained(autogaze_path)
        ag_model.eval()
        ag_model = ag_model.to(next(self.model.parameters()).device)

        self.selector = AutoGazeTokenSelector(ag_model, gazing_ratio=gazing_ratio)
        self.ag_processor = AutoGazeImageProcessor.from_pretrained(autogaze_path)

        log.info("VJEPA2Runner: AutoGaze ready (gazing_ratio=%.2f, integration=%s)",
                 gazing_ratio, self.integration)

    # ------------------------------------------------------------------ #
    # Input preprocessing
    # ------------------------------------------------------------------ #

    def _frames_to_ag_tensor(self, frames: List[Image.Image]) -> torch.Tensor:
        """Preprocess PIL frames for AutoGaze: (1, T, C, 224, 224)."""
        processed = self.ag_processor(images=frames, return_tensors="pt")
        pv = processed["pixel_values"]                        # (T, C, H, W)
        return pv.unsqueeze(0).to(next(self.model.parameters()).device)

    def _frames_to_vjepa_tensor(self, frames: List[Image.Image]) -> torch.Tensor:
        """Preprocess PIL frames for V-JEPA2: (1, T, C, H, W).

        Uses the model's processor when available; falls back to manual resize
        to 256×256 (V-JEPA2's training resolution).
        """
        device = next(self.model.parameters()).device

        if self.processor is not None:
            inputs = self.processor(videos=[frames], return_tensors="pt")
            key = "pixel_values_videos" if "pixel_values_videos" in inputs else "pixel_values"
            return inputs[key].to(device=device, dtype=next(self.model.parameters()).dtype)

        import torchvision.transforms.functional as TF
        resize = 256
        pil_frames = [f.convert("RGB").resize((resize, resize)) for f in frames]
        tensor_frames = torch.stack([TF.to_tensor(f) for f in pil_frames])  # (T, C, H, W)
        mean = torch.tensor([0.485, 0.456, 0.406])
        std  = torch.tensor([0.229, 0.224, 0.225])
        tensor_frames = (tensor_frames - mean[None, :, None, None]) / std[None, :, None, None]
        return tensor_frames.unsqueeze(0).to(
            device=device, dtype=next(self.model.parameters()).dtype
        )  # (1, T, C, H, W)

    @torch.no_grad()
    def _run_autogaze(self, frames: List[Image.Image]) -> torch.Tensor:
        """Run AutoGaze on frames → raw gaze map (1, T, 14, 14) float."""
        ag_video = self._frames_to_ag_tensor(frames)
        gaze_out = self.selector.ag(
            {'video': ag_video},
            gazing_ratio=self.selector.gazing_ratio,
            generate_only=True,
        )
        raw = gaze_out['gazing_mask'][-1].float()              # (1, T, 196)
        T = raw.shape[1]
        return raw.reshape(1, T, _AG_GRID, _AG_GRID)           # (1, T, 14, 14)

    # ------------------------------------------------------------------ #
    # Gaze mask helpers
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _flat_gaze_mask_hook(
        self,
        frames: List[Image.Image],
        T_p: int,
        H_p: int,
        W_p: int,
    ) -> torch.Tensor:
        """(T_p * H_p * W_p,) float mask — mean of all frames, tiled.

        Used by integration='hook'.
        """
        gaze_map = self._run_autogaze(frames)               # (1, T, 14, 14)
        mean_map = gaze_map.mean(dim=1)                     # (1, 14, 14)

        if H_p != _AG_GRID or W_p != _AG_GRID:
            mean_map = F.interpolate(
                mean_map.unsqueeze(1), size=(H_p, W_p),
                mode='bilinear', align_corners=False,
            ).squeeze(1)                                    # (1, H_p, W_p)

        spatial = (mean_map > _AG_THRESHOLD).float().reshape(-1)     # (H_p * W_p,)
        return spatial.unsqueeze(0).expand(T_p, -1).reshape(-1)      # (T_p * H_p * W_p,)

    @torch.no_grad()
    def _gazing_info_full(
        self,
        frames: List[Image.Image],
        T_p: int,
        H_p: int,
        W_p: int,
    ) -> dict:
        """Per-temporal-group gazing_info for AutoGazeVJEPA2Encoder.

        Follows INTEGRATION.md Step 1: computes the indices of the top-k gazed
        patches within each temporal group.  Using top-k gives a uniform count
        k per group, so N_gazed = T_p × k with no padding required.

        Strategy:
          1. Run AutoGaze → (1, T_ag, 14, 14) gaze maps.
          2. Group T_ag frames into T_p temporal groups; average within each.
          3. Resize each averaged map to (H_p, W_p).
          4. Pick the top-k spatial positions per group (k = gazing_ratio × H_p × W_p).
          5. Convert spatial indices to V-JEPA2 flat indices:
               flat = t_p * H_p * W_p + spatial_idx

        Returns:
            dict with keys:
              ``'gazing_pos'``            (1, N_gazed) long  — original flat indices
              ``'num_gazing_each_frame'`` (T_p,)       long  — k per group (uniform)
              ``'if_padded_gazing'``      (1, N_gazed) bool  — all False (no padding)
        """
        gaze_map = self._run_autogaze(frames)           # (1, T_ag, 14, 14)
        T_ag   = gaze_map.shape[1]
        device = gaze_map.device
        chunk_size = max(1, T_ag // T_p)

        k = max(1, min(int(self.selector.gazing_ratio * H_p * W_p), H_p * W_p))

        gazing_pos_list = []
        for t_p in range(T_p):
            start = min(t_p * chunk_size, T_ag - 1)
            end   = min((t_p + 1) * chunk_size, T_ag)
            chunk = gaze_map[0, start:end].mean(dim=0)      # (14, 14)

            if H_p != _AG_GRID or W_p != _AG_GRID:
                chunk = F.interpolate(
                    chunk.unsqueeze(0).unsqueeze(0),         # (1, 1, 14, 14)
                    size=(H_p, W_p),
                    mode='bilinear', align_corners=False,
                ).squeeze()                                  # (H_p, W_p)

            # Top-k spatial indices within this temporal group
            spatial_idx = chunk.reshape(-1).topk(k).indices     # (k,)
            flat_idx    = t_p * H_p * W_p + spatial_idx         # (k,) V-JEPA2 flat indices
            gazing_pos_list.append(flat_idx)

        gazing_pos = torch.cat(gazing_pos_list).unsqueeze(0)    # (1, T_p*k)
        N_gazed    = gazing_pos.shape[1]
        return {
            'gazing_pos':            gazing_pos,
            'num_gazing_each_frame': torch.full((T_p,), k, dtype=torch.long, device=device),
            'if_padded_gazing':      torch.zeros(1, N_gazed, dtype=torch.bool, device=device),
        }

    # ------------------------------------------------------------------ #
    # Hook context manager  (hook mode only)
    # ------------------------------------------------------------------ #

    @contextmanager
    def _patch_embed_hook(self, flat_mask: torch.Tensor):
        """Hook V-JEPA2 patch_embeddings output → zero non-gazed patches.

        ``patch_embeddings`` outputs (B, N, C); flat_mask is (T_p*H_p*W_p,)
        and is broadcast over the batch dimension.
        """
        mask = flat_mask  # (N,)

        def _hook(module, input, output):
            N = min(output.shape[1], mask.shape[0])
            result = output.clone()
            result[:, :N, :] = result[:, :N, :] * mask[:N].float().unsqueeze(0).unsqueeze(-1)
            return result

        handle = self.model.encoder.embeddings.patch_embeddings.register_forward_hook(_hook)
        try:
            yield
        finally:
            handle.remove()

    # ------------------------------------------------------------------ #
    # Feature extraction API
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def encode_video(
        self,
        frames: List[Image.Image],
    ) -> torch.Tensor:
        """Encode frames to V-JEPA2 patch features.

        Returns:
            (B, N, C) float tensor of patch features after the encoder.
        """
        video = self._frames_to_vjepa_tensor(frames)            # (1, T, C, H, W)

        if self.selector is None:
            outputs = self.model.encoder(pixel_values_videos=video)
            return outputs.last_hidden_state

        T = video.shape[1]
        T_p = max(1, T // self.TUBELET_SIZE)
        # H_p and W_p depend on input resolution; derive from tensor
        H = video.shape[3]
        W = video.shape[4]
        H_p = H // self.PATCH_SIZE
        W_p = W // self.PATCH_SIZE

        if self.integration == "full":
            self.model.encoder._gazing_info = self._gazing_info_full(frames, T_p, H_p, W_p)
            outputs = self.model.encoder(pixel_values_videos=video)
            return outputs.last_hidden_state
        else:
            flat_mask = self._flat_gaze_mask_hook(frames, T_p, H_p, W_p)
            with self._patch_embed_hook(flat_mask):
                outputs = self.model.encoder(pixel_values_videos=video)
            return outputs.last_hidden_state

    # ------------------------------------------------------------------ #
    # MCQ inference
    # ------------------------------------------------------------------ #

    def run(
        self,
        frames: List[Image.Image],
        prompt: str,
        max_new_tokens: int = 16,
    ) -> str:
        """Feature extraction runner does not support MCQ.

        Use ``VJEPA2LLMRunner`` (--mllm vjepa2_llm) for full video QA.
        """
        raise NotImplementedError(
            "VJEPA2Runner only extracts features. "
            "For MCQ video QA use VJEPA2LLMRunner (--mllm vjepa2_llm)."
        )


# ─────────────────────────────────────────────────────────────────────────────
# V-JEPA2 + LLM runner  (V-JEPA2 ViT + projector + any causal LM)
# ─────────────────────────────────────────────────────────────────────────────

class VJEPA2LLMRunner(VJEPA2Runner):
    """V-JEPA2 ViT backbone + MLP projector + causal LLM for video QA.

    Architecture
    ------------
    Video (T frames)
      ↓  V-JEPA2 encoder  [with optional AutoGaze zeroing]
    (B, T_p × H_p × W_p, 1024) patch features
      ↓  temporal mean pooling  → (B, T_p, 1024)
      ↓  VJEPA2Projector (LayerNorm + 2-layer MLP)
    (B, T_p, lm_hidden) video tokens   ← T_p = num_frames / tubelet_size
      ↓  prepend to text embeddings
    LLM (any causal LM) → answer

    Projector
    ---------
    The projector maps V-JEPA2 (1024-dim) features to the LLM's embedding
    space.  It must be fine-tuned before meaningful QA is possible.

    *Without* a trained projector (projector_path=None):
      - A randomly initialised projector is created.
      - ``run()`` still executes but output is random text.
      - Use this configuration to start a training run.

    *With* a trained projector (projector_path="weights/vjepa2_projector"):
      - Loaded via ``VJEPA2Projector.from_pretrained(projector_path)``.
      - ``run()`` produces meaningful answers.

    Saving / Loading the projector
    --------------------------------
    After fine-tuning::

        runner.projector.save_pretrained("weights/vjepa2_projector/")

    Loading for inference::

        runner = load_runner(
            mllm           = "vjepa2_llm",
            model_path     = "facebook/vjepa2-vitl-fpc64-256",
            lm_path        = "Qwen/Qwen2.5-7B-Instruct",
            projector_path = "weights/vjepa2_projector/",
            autogaze_path  = "weights/AutoGaze",
            gazing_ratio   = 0.75,
        )

    Training setup (minimal example)
    ---------------------------------
    Freeze V-JEPA2 and LLM; train only the projector::

        for p in runner.model.parameters(): p.requires_grad = False
        for p in runner.lm.parameters():    p.requires_grad = False
        # projector parameters are trainable by default
        optimizer = torch.optim.AdamW(runner.projector.parameters(), lr=1e-4)

    Input format
    ------------
    ``run()`` prepends video tokens before the text prompt::

        [<v0>, <v1>, ..., <v_{T_p-1}>]  [text tokens]  →  LLM  → answer

    For MCQ benchmarks, the prompt should end with the choice letters so the
    LLM generates ``"A"``, ``"B"``, ``"C"``, or ``"D"`` immediately.
    """

    name = "vjepa2_llm"
    supports_mcq = True    # has a paired LLM — overrides VJEPA2Runner.supports_mcq

    def __init__(
        self,
        model_path: str,
        autogaze_path: Optional[str],
        gazing_ratio: float,
        lm_path: str,
        dtype: torch.dtype = torch.bfloat16,
        projector_path: Optional[str] = None,
        integration: str = "full",
    ):
        """
        Args:
            model_path:      HF ID or path to V-JEPA2 weights.
            autogaze_path:   AutoGaze weights, or None for baseline.
            gazing_ratio:    Fraction of patches to keep (0–1).
            lm_path:         HF ID or path to the causal LLM (e.g. Qwen2.5-7B).
            dtype:           Torch dtype (default: bfloat16).
            projector_path:  Path to a saved ``VJEPA2Projector`` checkpoint, or
                             None to use a randomly-initialised projector
                             (useful for starting a training run).
            integration:     ``'full'`` (default) or ``'hook'``.
        """
        # V-JEPA2 encoder + AutoGaze
        super().__init__(
            model_path    = model_path,
            autogaze_path = autogaze_path,
            gazing_ratio  = gazing_ratio,
            dtype         = dtype,
            integration   = integration,
        )

        # LLM
        self._load_lm(lm_path, dtype)

        # Projector: loaded or freshly initialised
        self._load_projector(projector_path)

    # ------------------------------------------------------------------ #
    # Loading helpers
    # ------------------------------------------------------------------ #

    def _load_lm(self, lm_path: str, dtype: torch.dtype) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        log.info("VJEPA2LLMRunner: loading LLM from %s", lm_path)
        self.lm = AutoModelForCausalLM.from_pretrained(
            lm_path, torch_dtype=dtype, device_map="auto"
        )
        self.lm.eval()
        self.lm_tokenizer = AutoTokenizer.from_pretrained(lm_path)
        if self.lm_tokenizer.pad_token is None:
            self.lm_tokenizer.pad_token = self.lm_tokenizer.eos_token
        log.info("VJEPA2LLMRunner: LLM ready  hidden_size=%d",
                 self.lm.config.hidden_size)

    def _load_projector(self, projector_path: Optional[str]) -> None:
        from autogaze.vision_encoders.vjepa2 import VJEPA2Projector

        vit_hidden = self.model.config.hidden_size   # 1024 for ViT-L

        if projector_path is not None:
            log.info("VJEPA2LLMRunner: loading projector from %s", projector_path)
            self.projector = VJEPA2Projector.from_pretrained(projector_path)
        else:
            log.warning(
                "VJEPA2LLMRunner: projector_path not set — "
                "using randomly-initialised projector.  "
                "Fine-tune before expecting meaningful answers."
            )
            self.projector = VJEPA2Projector.new_for_lm(self.lm, vit_hidden=vit_hidden)

        lm_device = next(self.lm.parameters()).device
        self.projector = self.projector.to(device=lm_device,
                                           dtype=next(self.lm.parameters()).dtype)
        log.info("VJEPA2LLMRunner: projector  vit=%d → lm=%d",
                 self.projector.vit_hidden, self.projector.lm_hidden)

    # ------------------------------------------------------------------ #
    # Video encoding + projection
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def encode_and_project(self, frames: List[Image.Image]) -> torch.Tensor:
        """Encode frames and project to LLM embedding space.

        Returns:
            (1, T_p, lm_hidden) video token tensor on the LLM's device.
        """
        video = self._frames_to_vjepa_tensor(frames)    # (1, T, C, H, W)
        T     = video.shape[1]
        T_p   = max(1, T // self.TUBELET_SIZE)
        H_p   = video.shape[3] // self.PATCH_SIZE
        W_p   = video.shape[4] // self.PATCH_SIZE

        # ── Encode with optional AutoGaze ─────────────────────────────
        k_per_group = H_p * W_p   # default (no AutoGaze): all patches per group
        if self.selector is not None and self.integration == "full":
            gazing_info = self._gazing_info_full(frames, T_p, H_p, W_p)
            k_per_group = gazing_info['num_gazing_each_frame'][0].item()
            self.model.encoder._gazing_info = gazing_info
            patch_features = self.model.encoder(
                pixel_values_videos=video
            ).last_hidden_state
            # patch_features: (1, T_p*k, vit_hidden)
        elif self.selector is not None:   # hook mode
            flat_mask = self._flat_gaze_mask_hook(frames, T_p, H_p, W_p)
            with self._patch_embed_hook(flat_mask):
                patch_features = self.model.encoder(
                    pixel_values_videos=video
                ).last_hidden_state
            # patch_features: (1, T_p*H_p*W_p, vit_hidden) — zeroed, not shortened
        else:
            patch_features = self.model.encoder(
                pixel_values_videos=video
            ).last_hidden_state
            # patch_features: (1, T_p*H_p*W_p, vit_hidden)

        # ── Project to LLM space ──────────────────────────────────────
        # grid_thw encodes how to reshape for temporal mean pooling.
        # Full integration: N = T_p * k → (T_p, k, 1) so H_p*W_p = k.
        # Hook / baseline:  N = T_p * H_p * W_p → (T_p, H_p, W_p).
        if self.selector is not None and self.integration == "full":
            proj_grid = (T_p, k_per_group, 1)
        else:
            proj_grid = (T_p, H_p, W_p)

        lm_device = next(self.lm.parameters()).device
        patch_features = patch_features.to(lm_device)
        return self.projector(patch_features, grid_thw=proj_grid)
        # → (1, T_p, lm_hidden)

    # ------------------------------------------------------------------ #
    # MCQ inference
    # ------------------------------------------------------------------ #

    def run(
        self,
        frames: List[Image.Image],
        prompt: str,
        max_new_tokens: int = 16,
    ) -> str:
        """Video QA with V-JEPA2 ViT + VJEPA2Projector + causal LLM.

        Video tokens are prepended before the text tokens in embedding space::

            [v_0, ..., v_{T_p-1}] [text] → LLM → answer

        Args:
            frames:         List of PIL frames (uniform or stride sampled).
            prompt:         Question / MCQ prompt string.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Generated answer string (e.g. ``"A"`` for MCQ).
        """
        lm_device = next(self.lm.parameters()).device

        # ── Video tokens (1, T_p, lm_hidden) ─────────────────────────
        video_tokens = self.encode_and_project(frames)     # (1, T_p, D)

        # ── Text embeddings ───────────────────────────────────────────
        tok_out = self.lm_tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True
        )
        text_ids = tok_out["input_ids"].to(lm_device)       # (1, L)
        embed_fn = self.lm.get_input_embeddings()
        text_embeds = embed_fn(text_ids)                    # (1, L, D)

        # ── Concat: [video | text] ────────────────────────────────────
        inputs_embeds = torch.cat([video_tokens, text_embeds], dim=1)
        # attention_mask: all 1s (no padding)
        attn_mask = torch.ones(
            inputs_embeds.shape[:2], dtype=torch.long, device=lm_device
        )

        # ── Generate ──────────────────────────────────────────────────
        with torch.inference_mode():
            gen_ids = self.lm.generate(
                inputs_embeds  = inputs_embeds,
                attention_mask = attn_mask,
                max_new_tokens = max_new_tokens,
                do_sample      = False,
                temperature    = None,
                top_p          = None,
            )

        # gen_ids includes the prompt tokens for some LLMs when using
        # inputs_embeds; take only the newly generated part.
        # The LLM generates from position inputs_embeds.shape[1] onward.
        if gen_ids.shape[1] > inputs_embeds.shape[1]:
            new_ids = gen_ids[:, inputs_embeds.shape[1]:]
        else:
            new_ids = gen_ids

        return self.lm_tokenizer.decode(new_ids[0], skip_special_tokens=True).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Vanilla SigLIP runner  (HuggingFace transformers, unmodified)
# ─────────────────────────────────────────────────────────────────────────────

class SigLIPRunner(BaseMLLMRunner):
    """Vanilla HuggingFace SigLIP vision encoder for video.

    Uses ``transformers.SiglipVisionModel`` exactly as released — **no**
    modification to the model code.  Frames are processed as a batch of
    independent images ``(T, C, H, W)``; temporal order is preserved but
    there is no cross-frame attention.

    Contrast with the AutoGaze-modified SigLIP inside NVILA:
      - NVILA's SigLIP: custom code in ``autogaze/vision_encoders/siglip/``,
        accepts ``(B, T, C, H, W)``, block-causal inter-frame attention,
        multi-scale patch support.
      - This runner: standard HF code, per-frame independent processing.

    AutoGaze integration  (when *autogaze_path* is provided):
        Hook mode only — identical zero-shot approach to VJEPA2Runner/hook:
          1. Run AutoGaze on all frames → per-frame 14×14 gaze maps.
          2. Resize to the SigLIP patch grid ``(H_p, W_p)`` and threshold.
          3. A forward hook on ``vision_model.embeddings`` zeroes the
             embeddings of non-gazed patches before the transformer layers.

    Since each frame is processed independently there is no cross-frame
    gaze aggregation — each frame keeps only its own gazed patches.

    Feature output: ``encode_video()`` returns ``(1, T*N, C)`` — all frames
    concatenated along the token axis (same convention as VJEPA2Runner).
    ``run()`` raises ``NotImplementedError``; pair with a projector + LLM for MCQ.
    """

    name = "siglip"
    supports_mcq = False   # feature-extraction only; no paired LLM

    def __init__(
        self,
        model_path: str,
        autogaze_path: Optional[str],
        gazing_ratio: float,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Args:
            model_path:    HF hub ID or local path to a SigLIP model,
                           e.g. ``"google/siglip-so400m-patch14-224"``.
            autogaze_path: Path to AutoGaze weights, or None for baseline.
            gazing_ratio:  Fraction of patches to keep per frame (0–1).
            dtype:         Torch dtype (default: bfloat16).
        """
        try:
            from transformers import SiglipProcessor, SiglipVisionModel
        except ImportError:
            raise ImportError("transformers>=4.44 required for SigLIP.")

        self.gazing_ratio = gazing_ratio

        log.info("SigLIPRunner: loading model from %s (dtype=%s)", model_path, dtype)
        self.processor = SiglipProcessor.from_pretrained(model_path)
        self.model = SiglipVisionModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
        )
        self.model.eval()

        # Patch grid dimensions from config
        cfg = self.model.config
        self.H_p = cfg.image_size // cfg.patch_size
        self.W_p = cfg.image_size // cfg.patch_size

        self.selector = None
        if autogaze_path is not None:
            self._load_autogaze(autogaze_path)

    # ------------------------------------------------------------------ #
    # AutoGaze model loading
    # ------------------------------------------------------------------ #

    def _load_autogaze(self, autogaze_path: str) -> None:
        from autogaze.models.autogaze import AutoGaze, AutoGazeImageProcessor
        from autogaze.models.autogaze.autogaze_cv import AutoGazeTokenSelector

        log.info("SigLIPRunner: loading AutoGaze from %s", autogaze_path)
        ag_model = AutoGaze.from_pretrained(autogaze_path)
        ag_model.eval()
        ag_model = ag_model.to(next(self.model.parameters()).device)

        self.selector = AutoGazeTokenSelector(ag_model, gazing_ratio=self.gazing_ratio)
        self.ag_processor = AutoGazeImageProcessor.from_pretrained(autogaze_path)
        log.info("SigLIPRunner: AutoGaze ready (gazing_ratio=%.2f)", self.gazing_ratio)

    # ------------------------------------------------------------------ #
    # Input preprocessing
    # ------------------------------------------------------------------ #

    def _frames_to_ag_tensor(self, frames: List[Image.Image]) -> torch.Tensor:
        """Preprocess PIL frames for AutoGaze: (1, T, C, 224, 224)."""
        processed = self.ag_processor(images=frames, return_tensors="pt")
        pv = processed["pixel_values"]           # (T, C, H, W)
        return pv.unsqueeze(0).to(next(self.model.parameters()).device)

    def _frames_to_siglip_tensor(self, frames: List[Image.Image]) -> torch.Tensor:
        """Preprocess PIL frames for SigLIP: (T, C, H, W)."""
        device = next(self.model.parameters()).device
        dtype  = next(self.model.parameters()).dtype
        inputs = self.processor(images=frames, return_tensors="pt")
        return inputs["pixel_values"].to(device=device, dtype=dtype)

    # ------------------------------------------------------------------ #
    # Gaze computation
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _run_autogaze(self, frames: List[Image.Image]) -> torch.Tensor:
        """Run AutoGaze → (1, T, 14, 14) float gaze map."""
        ag_video = self._frames_to_ag_tensor(frames)
        gaze_out = self.selector.ag(
            {'video': ag_video},
            gazing_ratio=self.selector.gazing_ratio,
            generate_only=True,
        )
        raw = gaze_out['gazing_mask'][-1].float()   # (1, T, 196)
        T = raw.shape[1]
        return raw.reshape(1, T, _AG_GRID, _AG_GRID)

    @torch.no_grad()
    def _per_frame_mask(self, frames: List[Image.Image]) -> torch.Tensor:
        """Per-frame binary mask for SigLIP patch grid.

        Returns ``(T, H_p * W_p)`` float — 1 = keep, 0 = zero out.
        """
        gaze_map = self._run_autogaze(frames)   # (1, T, 14, 14)
        gaze_map = gaze_map.squeeze(0)          # (T, 14, 14)

        if self.H_p != _AG_GRID or self.W_p != _AG_GRID:
            gaze_map = F.interpolate(
                gaze_map.unsqueeze(1),           # (T, 1, 14, 14)
                size=(self.H_p, self.W_p),
                mode='bilinear', align_corners=False,
            ).squeeze(1)                         # (T, H_p, W_p)

        return (gaze_map > _AG_THRESHOLD).float().reshape(gaze_map.shape[0], -1)  # (T, N)

    # ------------------------------------------------------------------ #
    # Hook context manager
    # ------------------------------------------------------------------ #

    @contextmanager
    def _embed_hook(self, per_frame_mask: torch.Tensor):
        """Zero non-gazed patch embeddings via a hook on vision_model.embeddings.

        per_frame_mask: ``(T, N)`` float — applied to the ``(T, N, C)`` output
        of the SigLIP embeddings module before the transformer encoder.
        """
        mask = per_frame_mask

        def _hook(module, input, output):
            T, N, C = output.shape
            m = mask[:T, :N].to(output.device)
            return output * m.unsqueeze(-1)

        handle = self.model.vision_model.embeddings.register_forward_hook(_hook)
        try:
            yield
        finally:
            handle.remove()

    # ------------------------------------------------------------------ #
    # Feature extraction API
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def encode_video(self, frames: List[Image.Image]) -> torch.Tensor:
        """Encode video frames to SigLIP patch features.

        Each frame is processed independently (no cross-frame attention).
        AutoGaze gaze masking (if loaded) is applied per-frame.

        Returns:
            ``(1, T * N, C)`` — all frames concatenated along the token axis,
            where T = number of frames, N = patches per frame (H_p * W_p),
            C = SigLIP hidden size.
        """
        pixel_values = self._frames_to_siglip_tensor(frames)  # (T, C, H, W)

        if self.selector is None:
            outputs = self.model(pixel_values=pixel_values)
        else:
            mask = self._per_frame_mask(frames)                # (T, N)
            with self._embed_hook(mask):
                outputs = self.model(pixel_values=pixel_values)

        feats = outputs.last_hidden_state   # (T, N, C)
        T, N, C = feats.shape
        return feats.reshape(1, T * N, C)  # (1, T*N, C)

    # ------------------------------------------------------------------ #
    # MCQ inference — not available without LLM
    # ------------------------------------------------------------------ #

    def run(
        self,
        frames: List[Image.Image],
        prompt: str,
        max_new_tokens: int = 16,
    ) -> str:
        raise NotImplementedError(
            "SigLIPRunner is feature-extraction only (no paired LLM). "
            "Use encode_video() for feature extraction."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Runner registry
# ─────────────────────────────────────────────────────────────────────────────

RUNNERS: Dict[str, type] = {
    "nvila"         : NVILARunner,
    "qwen25vl"      : Qwen25VLRunner,      # zero-shot hook (integration='hook')
    "qwen25vl_full" : Qwen25VLRunner,      # full modified-ViT integration
    "vjepa2"        : VJEPA2Runner,        # V-JEPA2 encoder, feature extraction only
    "vjepa2_full"   : VJEPA2Runner,        # V-JEPA2 full modified-encoder, feature extraction
    "vjepa2_llm"    : VJEPA2LLMRunner,     # V-JEPA2 ViT + projector + LLM, MCQ video QA
    "siglip"        : SigLIPRunner,        # vanilla HF SigLIP, feature extraction (+ optional AutoGaze hook)
}

# Default integration mode per runner key
_RUNNER_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "qwen25vl"      : {"integration": "hook"},
    "qwen25vl_full" : {"integration": "full"},
    "vjepa2"        : {"integration": "hook"},
    "vjepa2_full"   : {"integration": "full"},
    "vjepa2_llm"    : {"integration": "full"},
}


def load_runner(
    mllm: str,
    model_path: str,
    autogaze_path: Optional[str],
    gazing_ratio: float,
    dtype: torch.dtype = torch.bfloat16,
    **kwargs: Any,
) -> BaseMLLMRunner:
    """Instantiate and return a runner by name.

    Args:
        mllm:          Runner key — one of ``RUNNERS``.
                       ``'qwen25vl'``      → Qwen2.5-VL, zero-shot hook.
                       ``'qwen25vl_full'`` → Qwen2.5-VL, full ViT integration.
                       ``'vjepa2'``        → V-JEPA2 encoder, feature extraction (hook).
                       ``'vjepa2_full'``   → V-JEPA2 encoder, feature extraction (full).
                       ``'vjepa2_llm'``    → V-JEPA2 ViT + projector + LLM, video QA.
                       ``'siglip'``        → Vanilla HF SigLIP, feature extraction.
        model_path:    Path or HF hub ID for the MLLM weights.
        autogaze_path: Path to AutoGaze weights, or None for baseline.
        gazing_ratio:  Fraction of patches retained per frame (0–1).
        dtype:         Torch dtype for the MLLM (default: bfloat16).
        **kwargs:      Extra keyword arguments forwarded to the runner constructor.
    """
    if mllm not in RUNNERS:
        raise ValueError(
            f"Unknown MLLM '{mllm}'.  Available: {sorted(RUNNERS.keys())}"
        )
    cls = RUNNERS[mllm]
    # Merge runner-specific defaults (e.g. integration mode) then user kwargs
    merged = {**_RUNNER_DEFAULTS.get(mllm, {}), **kwargs}
    return cls(
        model_path=model_path,
        autogaze_path=autogaze_path,
        gazing_ratio=gazing_ratio,
        dtype=dtype,
        **merged,
    )
