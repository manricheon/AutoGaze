from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class AutoGazeOutput:
    selected_patch_indices: torch.Tensor
    selected_scales: torch.Tensor | None
    attention_maps: torch.Tensor | None
    token_budget: int | None
    metadata: dict[str, Any]


def detect_original_autogaze() -> bool:
    try:
        from autogaze.models.autogaze import AutoGaze  # noqa: F401
    except Exception:
        return False
    return True


class AutoGazeWrapper:
    """Thin interface around AutoGaze as a patch/token selector."""

    def __init__(
        self,
        enabled: bool,
        original_model: Any | None = None,
        token_budget: int | None = None,
        num_patches_per_frame: int | None = None,
        scales: list[int] | None = None,
    ) -> None:
        self.enabled = enabled
        self.original_model = original_model
        self.token_budget = token_budget
        self.num_patches_per_frame = num_patches_per_frame
        self.scales = scales
        self.original_autogaze_available = detect_original_autogaze()

    def __call__(
        self,
        inputs: dict[str, Any] | None = None,
        *,
        video: torch.Tensor | None = None,
        visual_tokens: torch.Tensor | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AutoGazeOutput:
        inputs = dict(inputs or {})
        if video is not None:
            inputs["video"] = video
        if metadata is None:
            metadata = inputs.get("metadata", {})

        if self.enabled:
            return self._forward_on(inputs, metadata=metadata, **kwargs)
        return self._forward_off(video=inputs.get("video"), visual_tokens=visual_tokens, metadata=metadata)

    def _forward_off(
        self,
        *,
        video: torch.Tensor | None,
        visual_tokens: torch.Tensor | None,
        metadata: dict[str, Any] | None,
    ) -> AutoGazeOutput:
        metadata = dict(metadata or {})
        batch_size, frame_count, patches_per_frame = self._infer_full_shape(
            video=video,
            visual_tokens=visual_tokens,
            metadata=metadata,
        )

        device = self._infer_device(video=video, visual_tokens=visual_tokens)
        frame_indices = self._frame_indices(metadata, frame_count)
        patch_indices = self._patch_indices(metadata, patches_per_frame, device)
        selected_patch_indices = patch_indices.view(1, 1, patches_per_frame).expand(
            batch_size,
            frame_count,
            patches_per_frame,
        )

        original_token_count = frame_count * patches_per_frame
        output_metadata = {
            **metadata,
            "autogaze_enabled": False,
            "mode": "full",
            "frame_indices": frame_indices,
            "patch_indices": patch_indices.cpu().tolist(),
            "scales": metadata.get("scales", self.scales),
            "original_visual_token_count": original_token_count,
            "selected_visual_token_count": original_token_count,
            "original_visual_token_count_total": original_token_count * batch_size,
            "selected_visual_token_count_total": original_token_count * batch_size,
        }

        selected_scales = self._selected_scales(output_metadata, selected_patch_indices.shape)
        return AutoGazeOutput(
            selected_patch_indices=selected_patch_indices,
            selected_scales=selected_scales,
            attention_maps=None,
            token_budget=self.token_budget,
            metadata=output_metadata,
        )

    def _forward_on(
        self,
        inputs: dict[str, Any],
        *,
        metadata: dict[str, Any] | None,
        **kwargs: Any,
    ) -> AutoGazeOutput:
        if self.original_model is None:
            availability = "detected" if self.original_autogaze_available else "not detected"
            raise NotImplementedError(
                "AutoGaze ON mode requires an explicit original_model instance. "
                f"Original AutoGaze package is {availability}; checkpoint/model loading is outside this wrapper scope."
            )

        raw_output = self.original_model(inputs, **kwargs)
        return self._adapt_original_output(raw_output, metadata=dict(metadata or {}))

    def _adapt_original_output(self, raw_output: dict[str, Any], metadata: dict[str, Any]) -> AutoGazeOutput:
        selected_patch_indices = raw_output.get("gazing_pos")
        if selected_patch_indices is None:
            raise KeyError("Original AutoGaze output did not include 'gazing_pos'")

        if_padded = raw_output.get("if_padded_gazing")
        if if_padded is not None:
            selected_count = int((~if_padded.bool()).sum(dim=1).max().item())
        else:
            selected_count = int(selected_patch_indices.shape[-1])

        original_count = raw_output.get("num_vision_tokens_each_frame")
        frame_rate = int(raw_output.get("frame_sampling_rate", 1))
        frame_indices = metadata.get("frame_indices")
        if frame_indices is not None and original_count is not None:
            original_count = int(original_count) * max(1, len(frame_indices) // frame_rate)

        output_metadata = {
            **metadata,
            "autogaze_enabled": True,
            "mode": "original_autogaze",
            "frame_indices": frame_indices,
            "patch_indices": selected_patch_indices.detach().cpu().tolist(),
            "scales": raw_output.get("scales", metadata.get("scales", self.scales)),
            "original_visual_token_count": original_count,
            "selected_visual_token_count": selected_count,
            "raw_output_keys": sorted(raw_output.keys()),
        }
        return AutoGazeOutput(
            selected_patch_indices=selected_patch_indices,
            selected_scales=None,
            attention_maps=raw_output.get("gazing_mask"),
            token_budget=self.token_budget,
            metadata=output_metadata,
        )

    def _infer_full_shape(
        self,
        *,
        video: torch.Tensor | None,
        visual_tokens: torch.Tensor | None,
        metadata: dict[str, Any],
    ) -> tuple[int, int, int]:
        batch_size = 1
        frame_count = len(metadata["frame_indices"]) if "frame_indices" in metadata else None
        patches_per_frame = self.num_patches_per_frame

        if visual_tokens is not None:
            if visual_tokens.ndim == 4:
                batch_size, token_frames, token_patches = visual_tokens.shape[:3]
                frame_count = frame_count or token_frames
                patches_per_frame = patches_per_frame or token_patches
            elif visual_tokens.ndim == 3:
                batch_size, token_patches = visual_tokens.shape[:2]
                frame_count = frame_count or 1
                patches_per_frame = patches_per_frame or token_patches
            else:
                raise ValueError(
                    f"Expected visual_tokens shape [B, T, N, D] or [B, N, D], got {tuple(visual_tokens.shape)}"
                )

        if video is not None:
            if video.ndim != 5:
                raise ValueError(f"Expected video shape [B, T, C, H, W], got {tuple(video.shape)}")
            batch_size = int(video.shape[0])
            frame_count = frame_count or int(video.shape[1])

        if "patch_indices" in metadata:
            patches_per_frame = patches_per_frame or len(metadata["patch_indices"])

        if frame_count is None or patches_per_frame is None:
            raise ValueError(
                "AutoGaze OFF mode needs frame and patch counts from video/visual_tokens, "
                "metadata, or num_patches_per_frame."
            )

        return batch_size, int(frame_count), int(patches_per_frame)

    @staticmethod
    def _infer_device(video: torch.Tensor | None, visual_tokens: torch.Tensor | None) -> torch.device:
        if visual_tokens is not None:
            return visual_tokens.device
        if video is not None:
            return video.device
        return torch.device("cpu")

    @staticmethod
    def _frame_indices(metadata: dict[str, Any], frame_count: int) -> list[int]:
        if "frame_indices" in metadata:
            return list(metadata["frame_indices"])
        if "original_frame_indices" in metadata:
            return list(metadata["original_frame_indices"])
        return list(range(frame_count))

    @staticmethod
    def _patch_indices(metadata: dict[str, Any], patches_per_frame: int, device: torch.device) -> torch.Tensor:
        if "patch_indices" in metadata:
            return torch.as_tensor(metadata["patch_indices"], dtype=torch.long, device=device)
        return torch.arange(patches_per_frame, dtype=torch.long, device=device)

    @staticmethod
    def _selected_scales(metadata: dict[str, Any], selected_shape: torch.Size) -> torch.Tensor | None:
        scales = metadata.get("scales")
        if scales is None:
            return None
        if isinstance(scales, torch.Tensor):
            return scales
        if len(scales) == selected_shape[-1]:
            scale_tensor = torch.as_tensor(scales, dtype=torch.long)
            return scale_tensor.view(1, 1, selected_shape[-1]).expand(selected_shape)
        return None
