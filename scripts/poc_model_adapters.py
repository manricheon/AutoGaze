#!/usr/bin/env python3
from __future__ import annotations

import importlib
import json
import logging
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TORCH_DTYPE_DEPRECATION_WARNING = "`torch_dtype` is deprecated! Use `dtype` instead!"


@dataclass
class AdapterStatus:
    name: str
    status: str
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "reason": self.reason,
            "metadata": self.metadata,
        }


@contextmanager
def _suppress_transformers_torch_dtype_warning():
    class _Filter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return TORCH_DTYPE_DEPRECATION_WARNING not in record.getMessage()

    warning_filter = _Filter()
    loggers = [
        logging.getLogger("transformers"),
        logging.getLogger("transformers.configuration_utils"),
        logging.getLogger("transformers.modeling_utils"),
    ]
    for logger in loggers:
        logger.addFilter(warning_filter)
    try:
        yield
    finally:
        for logger in loggers:
            logger.removeFilter(warning_filter)


class VisionEncoderAdapter:
    name = "generic_vit"

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.status = AdapterStatus(self.name, "not_loaded")
        self.model: Any = None
        self.device = "cpu"

    def load(self, *, allow_real_model_loading: bool = False, device: str = "cpu", dtype: str = "float32") -> AdapterStatus:
        if not allow_real_model_loading:
            self.status = AdapterStatus(self.name, "stub", "real vision loading disabled")
            return self.status
        self.device = device
        module_path = self.config.get("module_path") or self.default_module_path()
        class_name = self.config.get("class_name") or self.config.get("class_or_factory") or self.default_class_name()
        checkpoint = self._checkpoint()
        if not module_path or not class_name:
            self.status = AdapterStatus(self.name, "blocked", "module_path and class_name are required for real loading")
            return self.status
        if not checkpoint:
            self.status = AdapterStatus(self.name, "blocked", f"{self.name} checkpoint_path or model_id is required for real loading")
            return self.status
        if _looks_like_local_path(str(checkpoint)):
            local_checkpoint = _resolve_local_path(checkpoint)
            if not local_checkpoint.exists():
                self.status = AdapterStatus(self.name, "blocked", f"{self.name} checkpoint path does not exist: {checkpoint}")
                return self.status
            missing_shards = _missing_sharded_checkpoint_files(local_checkpoint)
            if missing_shards:
                self.status = AdapterStatus(
                    self.name,
                    "blocked",
                    f"{self.name} checkpoint is incomplete; missing shard files: {', '.join(missing_shards[:8])}",
                    metadata={"checkpoint": checkpoint, "resolved_checkpoint": str(local_checkpoint), "missing_shards": missing_shards},
                )
                return self.status
        checkpoint_ref = _model_reference_for_loading(checkpoint)
        try:
            factory = getattr(importlib.import_module(str(module_path)), str(class_name))
            load_kwargs = _from_pretrained_kwargs(self.config, dtype=dtype)
            with _suppress_transformers_torch_dtype_warning():
                self.model = factory.from_pretrained(checkpoint_ref, **load_kwargs) if hasattr(factory, "from_pretrained") else factory()
            if hasattr(self.model, "to"):
                self.model.to(device)
            if hasattr(self.model, "eval"):
                self.model.eval()
            self.status = AdapterStatus(
                self.name,
                "real",
                metadata={
                    "checkpoint": checkpoint,
                    "resolved_checkpoint": checkpoint_ref,
                    "module_path": module_path,
                    "class_name": class_name,
                    "local_files_only": bool(self.config.get("local_files_only", False)),
                    "trust_remote_code": bool(self.config.get("trust_remote_code", False)),
                },
            )
        except Exception as exc:  # pragma: no cover - real model path is environment-dependent.
            self.status = AdapterStatus(self.name, "blocked", f"real vision loading failed: {exc}")
        return self.status

    def load_dummy_weights(self, *, device: str = "cpu", dtype: str = "float32") -> AdapterStatus:
        self.device = device
        self.model = None
        self.status = AdapterStatus(
            self.name,
            "dummy",
            "explicit dummy-weight smoke mode; no real vision checkpoint was loaded",
            metadata={
                "dummy_weights": True,
                "real_checkpoint_loaded": False,
                "device": device,
                "dtype": dtype,
                "checkpoint": self._checkpoint(),
            },
        )
        return self.status

    def preprocess_or_accept_features(self, video: torch.Tensor, autogaze: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return {"video": video, "autogaze": autogaze}

    def prepare_video_inputs(self, video: torch.Tensor, autogaze: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return self.preprocess_or_accept_features(video, autogaze=autogaze)

    def forward(self, video: torch.Tensor, autogaze: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if video.ndim != 5:
            raise ValueError(f"expected video shape [B,T,C,H,W], got {tuple(video.shape)}")
        if self.status.status == "blocked":
            raise RuntimeError(self.status.reason or f"{self.name} is blocked")
        if self.model is not None:
            return self._forward_real(video, autogaze=autogaze)
        batch, frames, _channels, height, width = [int(dim) for dim in video.shape]
        grid_h, grid_w = self.patch_grid((height, width))
        token_count = frames * grid_h * grid_w
        output_dim = self.output_dim()
        tokens = torch.zeros((batch, token_count, output_dim), dtype=video.dtype)
        return {
            "status": self.status.status,
            "visual_tokens": tokens,
            "metadata": {
                "adapter": self.name,
                "stub_status": self.status.status,
                "reason": self.status.reason,
                "autogaze_tokens_used": bool(autogaze is not None and self.supports_autogaze_tokens()),
            },
        }

    def _forward_real(self, video: torch.Tensor, autogaze: Mapping[str, Any] | None = None) -> dict[str, Any]:
        model_input = video.to(self.device) if hasattr(video, "to") else video
        kwargs = {}
        if autogaze is not None and self.supports_autogaze_tokens():
            kwargs["gazing_info"] = _move_tensor_mapping(autogaze, self.device)
        errors: list[str] = []
        batch_frames: tuple[int, int] | None = None
        if (
            isinstance(model_input, torch.Tensor)
            and model_input.ndim == 5
            and self.config.get("video_input_mode") == "flatten_frames"
        ):
            batch, frames, channels, height, width = [int(dim) for dim in model_input.shape]
            model_candidates = [model_input.reshape(batch * frames, channels, height, width), model_input]
            batch_frames = (batch, frames)
        else:
            model_candidates = [model_input]
        with torch.inference_mode():
            for candidate in model_candidates:
                for call in (
                    lambda candidate=candidate: self.model(pixel_values=candidate, **kwargs),
                    lambda candidate=candidate: self.model(videos=candidate, **kwargs),
                    lambda candidate=candidate: self.model(candidate, **kwargs),
                ):
                    try:
                        output = call()
                        break
                    except TypeError as exc:
                        errors.append(str(exc))
                else:
                    continue
                break
            else:
                raise RuntimeError(f"{self.name} real forward failed for supported call signatures: {errors}")
        tokens = getattr(output, "last_hidden_state", output)
        if isinstance(tokens, Mapping):
            tokens = tokens.get("last_hidden_state") or tokens.get("pooler_output") or tokens.get("hidden_states")
        if isinstance(tokens, (list, tuple)):
            tokens = tokens[0]
        if not isinstance(tokens, torch.Tensor):
            raise RuntimeError(f"{self.name} real forward did not return a tensor-like visual output")
        if batch_frames is not None and tokens.ndim >= 3 and int(tokens.shape[0]) == batch_frames[0] * batch_frames[1]:
            batch, frames = batch_frames
            tokens = tokens.reshape(batch, frames * int(tokens.shape[1]), *tokens.shape[2:])
        return {
            "status": "real",
            "visual_tokens": tokens,
            "metadata": {
                "adapter": self.name,
                "status": self.status.to_dict(),
                "autogaze_tokens_used": bool(autogaze is not None and self.supports_autogaze_tokens()),
            },
        }

    def output_dim(self) -> int:
        return int(self.config.get("output_dim", 768))

    def patch_grid(self, resolution: tuple[int, int] | None = None) -> tuple[int, int]:
        patch_size = int(self.config.get("patch_size", 16))
        if resolution is None:
            resolution = (int(self.config.get("resolution", 224)), int(self.config.get("resolution", 224)))
        return max(1, int(resolution[0]) // patch_size), max(1, int(resolution[1]) // patch_size)

    def supports_autogaze_tokens(self) -> bool:
        return False

    def supports_chop_mode(self) -> bool:
        return False

    def default_module_path(self) -> str | None:
        return None

    def default_class_name(self) -> str | None:
        return None

    def _checkpoint(self) -> Any:
        return self.config.get("checkpoint_path") or self.config.get("model_id")

    def status_report(self) -> dict[str, Any]:
        return {
            **self.status.to_dict(),
            "adapter": self.name,
            "supports_autogaze_tokens": self.supports_autogaze_tokens(),
            "supports_chop_mode": self.supports_chop_mode(),
            "output_dim": self.output_dim(),
        }


class ModifiedSiglipAdapter(VisionEncoderAdapter):
    name = "modified_siglip"

    def supports_autogaze_tokens(self) -> bool:
        return True

    def supports_chop_mode(self) -> bool:
        return True


class VanillaSiglipAdapter(VisionEncoderAdapter):
    name = "vanilla_siglip"

    def supports_autogaze_tokens(self) -> bool:
        return bool(self.config.get("experimental_autogaze_adapter", False))


class VJEPA2Adapter(VisionEncoderAdapter):
    name = "vjepa2"

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(config)
        self.processor: Any = None

    def default_module_path(self) -> str | None:
        return "transformers"

    def default_class_name(self) -> str | None:
        return "AutoModel"

    def load(self, *, allow_real_model_loading: bool = False, device: str = "cpu", dtype: str = "float32") -> AdapterStatus:
        status = super().load(allow_real_model_loading=allow_real_model_loading, device=device, dtype=dtype)
        if status.status != "real":
            return status
        processor_path = self.config.get("processor_path") or self._checkpoint()
        processor_class = self.config.get("processor_class_name") or self.config.get("processor_class_or_factory")
        if not processor_class and not self.config.get("processor_path"):
            status.metadata["processor_status"] = "not_configured_tensor_input"
            return status
        if _looks_like_local_path(str(processor_path)) and not _resolve_local_path(processor_path).exists():
            self.status = AdapterStatus(
                self.name,
                "blocked",
                f"real V-JEPA2 processor path does not exist: {processor_path}",
                metadata=status.metadata,
            )
            return self.status
        processor_module = self.config.get("processor_module_path") or "transformers"
        processor_class = processor_class or "AutoProcessor"
        try:
            factory = getattr(importlib.import_module(str(processor_module)), str(processor_class))
            processor_ref = _model_reference_for_loading(processor_path)
            with _suppress_transformers_torch_dtype_warning():
                self.processor = factory.from_pretrained(processor_ref, **_processor_from_pretrained_kwargs(self.config))
            status.metadata["processor_status"] = "real"
            status.metadata["processor_path"] = processor_path
            status.metadata["resolved_processor_path"] = processor_ref
            status.metadata["processor_module_path"] = processor_module
            status.metadata["processor_class_name"] = processor_class
        except Exception as exc:
            if bool(self.config.get("allow_tensor_input_without_processor", True)):
                status.metadata["processor_status"] = "unavailable_tensor_input_allowed"
                status.metadata["processor_error"] = str(exc)
                status.metadata["processor_path"] = processor_path
                status.metadata["resolved_processor_path"] = _model_reference_for_loading(processor_path)
                status.metadata["processor_module_path"] = processor_module
                status.metadata["processor_class_name"] = processor_class
                self.status = status
                return status
            self.status = AdapterStatus(
                self.name,
                "blocked",
                f"real V-JEPA2 processor loading failed: {exc}",
                metadata=status.metadata,
            )
            return self.status
        return status

    def preprocess_or_accept_features(self, video: torch.Tensor, autogaze: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if video.ndim != 5:
            raise ValueError(f"expected video shape [B,T,C,H,W], got {tuple(video.shape)}")
        return {
            "video": video,
            "autogaze": None,
            "metadata": {
                "semantics": "[B,T,C,H,W]",
                "note": "AutoGaze patch indices are not assumed to map directly to V-JEPA2 tokens.",
            },
        }

    def _forward_real(self, video: torch.Tensor, autogaze: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if video.ndim != 5:
            raise ValueError(f"expected video shape [B,T,C,H,W], got {tuple(video.shape)}")
        model_input = video.to(self.device) if hasattr(video, "to") else video
        skip_predictor = bool(self.config.get("skip_predictor", True))
        call_errors: list[str] = []
        with torch.inference_mode():
            for call_name, call in (
                (
                    "pixel_values_videos+skip_predictor",
                    lambda: self.model(pixel_values_videos=model_input, skip_predictor=skip_predictor),
                ),
                ("pixel_values_videos", lambda: self.model(pixel_values_videos=model_input)),
                ("pixel_values", lambda: self.model(pixel_values=model_input)),
                ("videos", lambda: self.model(videos=model_input)),
                ("positional", lambda: self.model(model_input)),
            ):
                try:
                    output = call()
                    forward_call = call_name
                    break
                except TypeError as exc:
                    call_errors.append(f"{call_name}: {exc}")
            else:
                raise RuntimeError("V-JEPA2 real forward failed for known input signatures: " + " | ".join(call_errors))
        tokens = getattr(output, "last_hidden_state", output)
        if isinstance(tokens, Mapping):
            tokens = tokens.get("last_hidden_state") or tokens.get("hidden_states")
        if isinstance(tokens, (list, tuple)):
            tokens = tokens[0]
        if not isinstance(tokens, torch.Tensor):
            raise RuntimeError("V-JEPA2 real forward did not return last_hidden_state tensor")
        pooled = tokens.mean(dim=1) if tokens.ndim >= 3 else None
        return {
            "status": "real",
            "visual_tokens": tokens,
            "pooled_features": pooled,
            "metadata": {
                "adapter": self.name,
                "status": self.status.to_dict(),
                "input_shape": [int(dim) for dim in video.shape],
                "feature_shape": [int(dim) for dim in tokens.shape],
                "pooled_feature_shape": [int(dim) for dim in pooled.shape] if isinstance(pooled, torch.Tensor) else None,
                "forward_call": forward_call,
                "skip_predictor": skip_predictor,
                "autogaze_tokens_used": False,
                "mllm_projection": self.supports_mllm_projection(),
            },
        }

    def output_dim(self) -> int:
        return self._config_int("hidden_size", self.config.get("output_dim", 1024))

    def patch_grid(self, resolution: tuple[int, int] | None = None) -> tuple[int, int]:
        patch_size = self._config_int("patch_size", 16)
        if resolution is None:
            crop_size = self._config_int("crop_size", self.config.get("image_size", 256))
            resolution = (crop_size, crop_size)
        return max(1, int(resolution[0]) // patch_size), max(1, int(resolution[1]) // patch_size)

    def supports_chop_mode(self) -> bool:
        return True

    def supports_autogaze_tokens(self) -> bool:
        return False

    def run_official_dense(self, *, video: torch.Tensor | None = None, **_: Any) -> dict[str, Any]:
        if self.model is not None and video is not None:
            result = self.forward(video)
            result["metadata"] = {**dict(result.get("metadata") or {}), **self._vjepa2_report("vjepa2_official_dense", generation_ran=False)}
            return result
        return self._vjepa2_mode_status(
            "vjepa2_official_dense",
            "V-JEPA2 dense official path is registered; real feature extraction requires explicit model loading and video input.",
        )

    def run_autogaze_frame_selection(self, **_: Any) -> dict[str, Any]:
        return self._vjepa2_mode_status(
            "autogaze_frame_selection_vjepa2",
            "AutoGaze frame/window selection can feed dense selected clips to the V-JEPA2 official processor; no sparse tubelet injection is claimed.",
        )

    def run_autogaze_chop_selection(self, **_: Any) -> dict[str, Any]:
        return self._vjepa2_mode_status(
            "autogaze_chop_selection_vjepa2",
            "AutoGaze crop/chop selection can feed dense selected crops or clips to V-JEPA2; no encoder-side sparse acceleration is claimed.",
        )

    def run_autogaze_zero_mask(self, **_: Any) -> dict[str, Any]:
        return self._vjepa2_mode_status(
            "autogaze_zero_mask_vjepa2",
            "AutoGaze zero-mask mode can preserve the dense V-JEPA2 input grid while masking unselected image regions; this is a probing mode, not encoder acceleration.",
        )

    def run_context_mask_probe(self, **_: Any) -> dict[str, Any]:
        raise NotImplementedError(
            "vjepa2_context_mask_probe needs source inspection: context_mask/target_mask semantics and compute impact are not verified. "
            "No fallback to SigLIP, NVILA, or trainable projection adapter is allowed."
        )

    def run_sparse_tubelet(self, **_: Any) -> dict[str, Any]:
        raise NotImplementedError(
            "vjepa2_sparse_tubelet is blocked until V-JEPA2 patchify/forward accepts selected tubelets with correct 3D-RoPE positions. "
            "No fallback to SigLIP, NVILA, or trainable projection adapter is allowed."
        )

    def run_video_classification(self, **_: Any) -> dict[str, Any]:
        return self._vjepa2_mode_status(
            "vjepa2_video_classification",
            "Use an official VJEPA2ForVideoClassification checkpoint for action-recognition smoke tests; the local base encoder checkpoint is not treated as a trained classifier.",
        )

    def run_feature_extraction(self, *, video: torch.Tensor | None = None, **_: Any) -> dict[str, Any]:
        if self.model is not None and video is not None:
            return self.forward(video)
        return self._vjepa2_mode_status(
            "vjepa2_feature_extraction",
            "Frozen V-JEPA2 feature extraction is a safe PoC path after explicit real model loading.",
        )

    def get_patch_grid(self, resolution: tuple[int, int] | None = None) -> tuple[int, int]:
        return self.patch_grid(resolution)

    def get_tubelet_grid(self) -> tuple[int, int, int]:
        frames = self._config_int("frames_per_clip", 64)
        tubelet = self._config_int("tubelet_size", 2)
        grid_h, grid_w = self.patch_grid()
        return max(1, frames // max(1, tubelet)), grid_h, grid_w

    def get_position_encoding_status(self) -> dict[str, Any]:
        return {
            "positional_encoding_type": "3d_rope",
            "patch_structure": "tubelet",
            "patch_size": self._config_int("patch_size", 16),
            "crop_size": self._config_int("crop_size", self.config.get("image_size", 256)),
            "frames_per_clip": self._config_int("frames_per_clip", 64),
            "tubelet_size": self._config_int("tubelet_size", 2),
            "deterministic_sparse_position_status": "needs_source_inspection",
            "context_mask_status": "needs_source_inspection",
            "sparse_tubelet_status": "blocked_without_forward_patchification_verification",
        }

    def get_output_dim(self) -> int:
        return self.output_dim()

    def supports_mllm_projection(self, target_mllm: str | None = None) -> dict[str, Any]:
        verified = bool(self.config.get("compatible_frozen_projector_verified", False))
        return {
            "target_mllm": target_mllm or self.config.get("target_mllm") or "unspecified",
            "supported": verified,
            "status": "verified_frozen_projector" if verified else "blocked_without_training",
            "requires_training": not verified,
            "reason": (
                "A compatible frozen projector was explicitly marked verified."
                if verified
                else "V-JEPA2 hidden size/features are not a drop-in replacement for reviewed MLLM vision projectors."
            ),
        }

    def recommend_decoder(self) -> list[dict[str, Any]]:
        return [
            {
                "decoder": "VJEPA2ForVideoClassification",
                "priority": "high",
                "training_needed": "no if using an already trained classification checkpoint; yes for new labels",
                "supports_query_text": False,
                "recommended_status": "feasible",
            },
            {
                "decoder": "frozen_feature_extraction_plus_probe",
                "priority": "high",
                "training_needed": "optional small non-MLLM probe; no trainable MLLM adapter",
                "supports_query_text": False,
                "recommended_status": "feasible",
            },
            {
                "decoder": "temporal_pooling_retrieval",
                "priority": "medium",
                "training_needed": False,
                "supports_query_text": "only with an external text embedding/retrieval setup",
                "recommended_status": "feasible_input_only",
            },
            {
                "decoder": "qformer_perceiver_or_mllm_connector",
                "priority": "low",
                "training_needed": True,
                "supports_query_text": True,
                "recommended_status": "blocked_without_training",
            },
        ]

    def status_report(self) -> dict[str, Any]:
        return {
            **super().status_report(),
            "model_type": "video_encoder",
            "patch_structure": "tubelet",
            "input_tensor_format": "[B,T,C,H,W]",
            "position_encoding": self.get_position_encoding_status(),
            "tubelet_grid": self.get_tubelet_grid(),
            "supports_official_dense": True,
            "supports_autogaze_frame_selection": True,
            "supports_autogaze_chop_selection": True,
            "supports_autogaze_zero_mask": True,
            "supports_context_mask_probe": "unknown",
            "supports_sparse_tubelet": "unknown",
            "supports_direct_mllm_projection": False,
            "local_asset_status": "local_exists_if_weights/vjepa2-vitl-fpc64-256_is_present",
            "official_processor_support_status": "registered_dense_feature_path",
            "input_selection_support_status": "supported_as_dense_clip_or_chop_selection_only",
            "zero_mask_support_status": "supported_as_image_space_probe_only_no_encoder_acceleration",
            "mllm_projection": self.supports_mllm_projection(),
            "decoder_recommendations": self.recommend_decoder(),
        }

    def _vjepa2_mode_status(self, integration_mode: str, reason: str) -> dict[str, Any]:
        return {
            "status": "stub-only",
            "reason": reason,
            "metadata": self._vjepa2_report(integration_mode, generation_ran=False),
        }

    def _vjepa2_report(self, integration_mode: str, *, generation_ran: bool) -> dict[str, Any]:
        checkpoint = self._checkpoint() or "weights/vjepa2-vitl-fpc64-256"
        return {
            "requested_model": checkpoint,
            "actual_model_loaded": checkpoint if self.status.status == "real" else None,
            "adapter": self.name,
            "integration_mode": integration_mode,
            "real_checkpoint_loaded": self.status.status == "real",
            "generation_ran": generation_ran,
            "model_type": "video_encoder",
            "positional_encoding_type": "3d_rope",
            "patch_structure": "tubelet",
            "patch_size": self._config_int("patch_size", 16),
            "crop_size": self._config_int("crop_size", self.config.get("image_size", 256)),
            "frames_per_clip": self._config_int("frames_per_clip", 64),
            "tubelet_size": self._config_int("tubelet_size", 2),
            "output_dim": self.output_dim(),
            "mllm_projection_supported": self.supports_mllm_projection()["supported"],
        }

    def _config_int(self, key: str, default: Any) -> int:
        try:
            return int(self.config.get(key, default))
        except (TypeError, ValueError):
            return int(default)


class GenericVitAdapter(VisionEncoderAdapter):
    name = "generic_vit"


class ExternalVisionEncoderAdapter(VisionEncoderAdapter):
    name = "external"

    def load(self, *, allow_real_model_loading: bool = False, device: str = "cpu", dtype: str = "float32") -> AdapterStatus:
        if not allow_real_model_loading:
            self.status = AdapterStatus(
                self.name,
                "stub-only",
                "external vision encoder loading disabled; no fallback to modified_siglip",
                metadata={"fallback_adapter": None},
            )
            return self.status
        return super().load(allow_real_model_loading=allow_real_model_loading, device=device, dtype=dtype)


class MLLMAdapter:
    name = "generic_mllm"

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.status = AdapterStatus(self.name, "not_loaded")
        self.model: Any = None
        self.processor: Any = None
        self.device = "cpu"

    def load(self, *, allow_real_model_loading: bool = False, device: str = "cpu", dtype: str = "float32") -> AdapterStatus:
        if not allow_real_model_loading:
            self.status = AdapterStatus(self.name, "stub", "real MLLM loading disabled")
            return self.status
        self.device = device
        module_path = self.config.get("module_path") or self.default_module_path()
        class_name = self.config.get("class_name") or self.config.get("class_or_factory") or self.default_class_name()
        checkpoint = self._checkpoint()
        if not module_path or not class_name:
            self.status = AdapterStatus(self.name, "blocked", "module_path and class_name are required for real loading")
            return self.status
        if not checkpoint:
            self.status = AdapterStatus(self.name, "blocked", f"{self.name} checkpoint_path or model_id is required for real loading")
            return self.status
        if _looks_like_local_path(str(checkpoint)):
            local_checkpoint = _resolve_local_path(checkpoint)
            if not local_checkpoint.exists():
                self.status = AdapterStatus(self.name, "blocked", f"{self.name} checkpoint path does not exist: {checkpoint}")
                return self.status
            missing_shards = _missing_sharded_checkpoint_files(local_checkpoint)
            if missing_shards:
                self.status = AdapterStatus(
                    self.name,
                    "blocked",
                    f"{self.name} checkpoint is incomplete; missing shard files: {', '.join(missing_shards[:8])}",
                    metadata={"checkpoint": checkpoint, "resolved_checkpoint": str(local_checkpoint), "missing_shards": missing_shards},
                )
                return self.status
        checkpoint_ref = _model_reference_for_loading(checkpoint)
        try:
            factory = getattr(importlib.import_module(str(module_path)), str(class_name))
            load_kwargs = _from_pretrained_kwargs(self.config, dtype=dtype)
            with _suppress_transformers_torch_dtype_warning():
                self.model = factory.from_pretrained(checkpoint_ref, **load_kwargs) if hasattr(factory, "from_pretrained") else factory()
            if hasattr(self.model, "to") and "device_map" not in load_kwargs:
                self.model.to(device)
            if hasattr(self.model, "eval"):
                self.model.eval()
            self.status = AdapterStatus(
                self.name,
                "real",
                metadata={
                    "checkpoint": checkpoint,
                    "resolved_checkpoint": checkpoint_ref,
                    "module_path": module_path,
                    "class_name": class_name,
                    "local_files_only": bool(self.config.get("local_files_only", False)),
                    "trust_remote_code": bool(self.config.get("trust_remote_code", False)),
                },
            )
        except Exception as exc:  # pragma: no cover - real model path is environment-dependent.
            self.status = AdapterStatus(self.name, "blocked", f"real MLLM loading failed: {exc}")
        return self.status

    def load_dummy_weights(self, *, device: str = "cpu", dtype: str = "float32") -> AdapterStatus:
        self.device = device
        self.model = None
        self.processor = None
        self.status = AdapterStatus(
            self.name,
            "dummy",
            "explicit dummy-weight smoke mode; no real MLLM checkpoint was loaded",
            metadata={
                "dummy_weights": True,
                "real_checkpoint_loaded": False,
                "device": device,
                "dtype": dtype,
                "checkpoint": self._checkpoint(),
            },
        )
        return self.status

    def prepare_inputs(
        self,
        *,
        query_text: str,
        video: torch.Tensor,
        visual_tokens: torch.Tensor | None = None,
        video_path: str | None = None,
        autogaze: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "query_text": query_text,
            "video": video,
            "visual_tokens": visual_tokens,
            "video_path": video_path,
            "autogaze": autogaze,
        }

    def generate(
        self,
        *,
        query_text: str,
        video: torch.Tensor,
        visual_tokens: torch.Tensor | None = None,
        max_new_tokens: int = 32,
        video_path: str | None = None,
        autogaze: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not query_text:
            return {"status": "blocked", "answer": None, "reason": "query text is required"}
        if self.status.status == "dummy":
            return self._dummy_generation(
                query_text=query_text,
                max_new_tokens=max_new_tokens,
                integration_mode=str(self.config.get("generation_input_mode") or "official_processor"),
            )
        if self.model is None:
            return {
                "status": self.status.status,
                "answer": None,
                "reason": self.status.reason or "MLLM generation unavailable",
                "query_text_used": True,
            }
        return {
            "status": "blocked",
            "answer": None,
            "reason": f"real generation is not implemented for {self.name}; add a model-specific generation adapter",
            "query_text_used": True,
        }

    def _dummy_generation(self, *, query_text: str, max_new_tokens: int, integration_mode: str) -> dict[str, Any]:
        answer = f"[dummy:{self.name}] placeholder response for: {query_text}"
        if max_new_tokens > 0:
            answer = " ".join(answer.split()[:max_new_tokens])
        return {
            "status": "dummy",
            "answer": answer,
            "reason": "dummy-weight smoke generation; output is not a model prediction",
            "query_text_used": True,
            "metadata": {
                "adapter": self.name,
                "integration_mode": integration_mode,
                "dummy_weights": True,
                "real_checkpoint_loaded": False,
                "generation_ran": True,
                "official_processor_path": False,
                "direct_visual_token_injection": False,
                "autogaze_visual_tokens_injected": False,
            },
        }

    def count_visual_tokens(self, visual_tokens: torch.Tensor | None) -> int | None:
        if visual_tokens is None:
            return None
        if visual_tokens.ndim < 2:
            return None
        return int(visual_tokens.shape[-2])

    def supports_direct_visual_tokens(self) -> bool:
        return False

    def supports_native_sparse_patch(self) -> bool:
        return False

    def supports_light_modified_sparse(self) -> bool:
        return False

    def supports_rope_sparse_patch(self) -> bool:
        return False

    def supports_direct_visual_token_injection(self) -> bool:
        return self.supports_direct_visual_tokens()

    def supports_official_processor_path(self) -> bool:
        return False

    def inspect_vision_encoder(self) -> dict[str, Any]:
        return {
            "adapter": self.name,
            "vision_encoder_type": getattr(self, "encoder_type", "unknown"),
            "siglip_based": getattr(self, "siglip_based", "unknown"),
            "patch_grid_accessible": getattr(self, "patch_grid_accessible", "unknown"),
            "dense_grid_dependency": getattr(self, "dense_grid_dependency", "unknown"),
        }

    def inspect_positional_encoding(self) -> dict[str, Any]:
        return {
            "adapter": self.name,
            "positional_encoding_type": getattr(self, "positional_encoding_type", "unknown"),
            "positional_encoding_status": getattr(self, "positional_encoding_status", "unknown"),
            "deterministic_position_adaptation": getattr(self, "deterministic_position_adaptation", "unknown"),
        }

    def inspect_projector(self) -> dict[str, Any]:
        return {
            "adapter": self.name,
            "projector_status": getattr(self, "projector_status", "unknown"),
            "variable_visual_token_support": getattr(self, "variable_visual_token_support", "unknown"),
            "requires_training_or_trainable_adapter": getattr(self, "requires_training_or_trainable_adapter", "unknown"),
        }

    def inspect_placeholder_handling(self) -> dict[str, Any]:
        return {
            "adapter": self.name,
            "visual_placeholder_dynamic": getattr(self, "visual_placeholder_dynamic", "unknown"),
            "placeholder_status": getattr(self, "placeholder_status", "unknown"),
        }

    def supports_direct_sparse_autogaze(self) -> bool:
        return getattr(self, "direct_sparse_autogaze_supported", False) is True

    def supports_input_level_selection(self) -> bool:
        return False

    def supports_autogaze_zero_mask(self) -> bool:
        return False

    def supports_post_encoder_zero_mask(self) -> bool:
        return False

    def supports_post_encoder_pruning(self) -> bool:
        return False

    def run_autogaze_zero_mask_path(self, **_: Any) -> dict[str, Any]:
        raise NotImplementedError(
            f"{self.name} does not support autogaze_zero_mask. "
            "No fallback to NVILA, modified SigLIP, or trainable projection adapter is allowed."
        )

    def run_native_sparse_patch_path(self, **_: Any) -> dict[str, Any]:
        raise NotImplementedError(
            f"{self.name} does not support native_sparse_patch. "
            "No fallback to NVILA, modified SigLIP, or trainable projection adapter is allowed."
        )

    def run_light_modified_sparse_path(self, **_: Any) -> dict[str, Any]:
        raise NotImplementedError(
            f"{self.name} does not support light_modified_sparse. "
            "No fallback to NVILA, modified SigLIP, or trainable projection adapter is allowed."
        )

    def run_post_encoder_zero_mask_path(self, **_: Any) -> dict[str, Any]:
        raise NotImplementedError(
            f"{self.name} does not support post_encoder_zero_mask. "
            "No fallback to NVILA, modified SigLIP, or trainable projection adapter is allowed."
        )

    def default_module_path(self) -> str | None:
        return None

    def default_class_name(self) -> str | None:
        return None

    def _checkpoint(self) -> Any:
        return self.config.get("checkpoint_path") or self.config.get("model_id")

    def status_report(self) -> dict[str, Any]:
        return {
            **self.status.to_dict(),
            "adapter": self.name,
            "supports_direct_visual_tokens": self.supports_direct_visual_tokens(),
            "supports_native_sparse_patch": self.supports_native_sparse_patch(),
            "supports_light_modified_sparse": self.supports_light_modified_sparse(),
            "supports_rope_sparse_patch": self.supports_rope_sparse_patch(),
            "supports_direct_visual_token_injection": self.supports_direct_visual_token_injection(),
            "supports_official_processor_path": self.supports_official_processor_path(),
            "inspect_vision_encoder": self.inspect_vision_encoder(),
            "inspect_positional_encoding": self.inspect_positional_encoding(),
            "inspect_projector": self.inspect_projector(),
            "inspect_placeholder_handling": self.inspect_placeholder_handling(),
            "supports_direct_sparse_autogaze": self.supports_direct_sparse_autogaze(),
            "supports_input_level_selection": self.supports_input_level_selection(),
            "supports_autogaze_zero_mask": self.supports_autogaze_zero_mask(),
            "supports_post_encoder_zero_mask": self.supports_post_encoder_zero_mask(),
            "supports_post_encoder_pruning": self.supports_post_encoder_pruning(),
        }


class NVILAAdapter(MLLMAdapter):
    name = "nvila"

    def load(self, *, allow_real_model_loading: bool = False, device: str = "cpu", dtype: str = "float32") -> AdapterStatus:
        status = super().load(allow_real_model_loading=allow_real_model_loading, device=device, dtype=dtype)
        if status.status != "real":
            return status
        processor_path = self.config.get("processor_path") or self._checkpoint()
        if not processor_path:
            self.status = AdapterStatus(self.name, "blocked", "NVILA processor_path or checkpoint_path is required for official processor loading")
            return self.status
        if _looks_like_local_path(str(processor_path)) and not _resolve_local_path(processor_path).exists():
            self.status = AdapterStatus(
                self.name,
                "blocked",
                f"NVILA official processor path does not exist: {processor_path}",
                metadata=status.metadata,
            )
            return self.status
        processor_module = self.config.get("processor_module_path") or "transformers"
        processor_class = self.config.get("processor_class_name") or self.config.get("processor_class_or_factory") or "AutoProcessor"
        try:
            factory = getattr(importlib.import_module(str(processor_module)), str(processor_class))
            processor_ref = _model_reference_for_loading(processor_path)
            processor_kwargs = _nvila_processor_from_pretrained_kwargs(self.config)
            with _suppress_transformers_torch_dtype_warning():
                self.processor = factory.from_pretrained(processor_ref, **processor_kwargs)
            status.metadata["processor_status"] = "real"
            status.metadata["processor_path"] = processor_path
            status.metadata["resolved_processor_path"] = processor_ref
            status.metadata["processor_module_path"] = processor_module
            status.metadata["processor_class_name"] = processor_class
            status.metadata["official_processor_path"] = True
            status.metadata["processor_autogaze_controls"] = _jsonable_processor_kwargs(processor_kwargs)
            self.status = status
        except Exception as exc:
            self.status = AdapterStatus(
                self.name,
                "blocked",
                f"NVILA official processor loading failed: {exc}",
                metadata=status.metadata,
            )
            return self.status
        return status

    def default_module_path(self) -> str | None:
        return "transformers"

    def default_class_name(self) -> str | None:
        return "AutoModel"

    def supports_direct_visual_tokens(self) -> bool:
        return False

    def supports_official_processor_path(self) -> bool:
        return True

    def generate(
        self,
        *,
        query_text: str,
        video: torch.Tensor,
        visual_tokens: torch.Tensor | None = None,
        max_new_tokens: int = 32,
        video_path: str | None = None,
        autogaze: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if visual_tokens is not None:
            return {
                "status": "blocked",
                "answer": None,
                "reason": "NVILA direct visual token injection is not verified in this PoC; use official processor video input.",
                "query_text_used": True,
            }
        if self.model is not None and self.processor is not None:
            return self._generate_with_official_processor(
                query_text=query_text,
                video=video,
                video_path=video_path,
                max_new_tokens=max_new_tokens,
            )
        return super().generate(
            query_text=query_text,
            video=video,
            visual_tokens=visual_tokens,
            max_new_tokens=max_new_tokens,
            video_path=video_path,
            autogaze=autogaze,
        )

    def _generate_with_official_processor(
        self,
        *,
        query_text: str,
        video: torch.Tensor,
        video_path: str | None,
        max_new_tokens: int,
    ) -> dict[str, Any]:
        if not query_text:
            return {"status": "blocked", "answer": None, "reason": "query text is required", "query_text_used": False}
        try:
            prompt_template = str(self.config.get("prompt_template") or "{video_token}\n\n{prompt}")
            video_token = getattr(getattr(self.processor, "tokenizer", None), "video_token", "<video>")
            prompt = prompt_template.format(prompt=query_text, query=query_text, video_token=video_token)
            processor_kwargs = _nvila_processor_from_pretrained_kwargs(self.config)
            target_frame_count = _positive_int_or_none(processor_kwargs.get("num_video_frames")) or _positive_int_or_none(
                getattr(self.processor, "num_video_frames", None)
            )
            video_input, video_input_kind = _official_video_input(
                video=video,
                video_path=video_path,
                target_frame_count=target_frame_count,
                batch_pil_frames=True,
            )
            inputs = self.processor(text=prompt, videos=video_input, return_tensors="pt")
            if isinstance(inputs, Mapping):
                model_device = getattr(self.model, "device", self.device)
                inputs = {
                    key: value.to(model_device) if isinstance(value, torch.Tensor) else value
                    for key, value in inputs.items()
                }
            with torch.inference_mode():
                outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            input_ids = inputs.get("input_ids") if isinstance(inputs, Mapping) else None
            decode_input = outputs[:, input_ids.shape[1] :] if isinstance(outputs, torch.Tensor) and isinstance(input_ids, torch.Tensor) else outputs
            decoded = self.processor.batch_decode(decode_input, skip_special_tokens=True)
            answer = str(decoded[0]).strip() if decoded else ""
            return {
                "status": "real",
                "answer": answer,
                "reason": None,
                "query_text_used": True,
                "official_processor_path": True,
                "metadata": {
                    "prompt_template": prompt_template,
                    "video_input_kind": video_input_kind,
                    "max_new_tokens": max_new_tokens,
                    "autogaze_visualizer_status": self.config.get("poc_autogaze_status"),
                    "autogaze_enabled_for_processor": self.config.get("poc_autogaze_enabled"),
                    "processor_autogaze_controls": _jsonable_processor_kwargs(processor_kwargs),
                    "target_frame_count": target_frame_count,
                    "autogaze_visual_tokens_injected": False,
                    "note": (
                        "NVILA generation uses the official processor video path. "
                        "The separate PoC AutoGaze stage is retained for visualization and metrics; "
                        "direct visual-token injection into NVILA is not claimed."
                    ),
                },
            }
        except Exception as exc:  # pragma: no cover - real model path is environment-dependent.
            return {
                "status": "blocked",
                "answer": None,
                "reason": f"NVILA official processor generation failed: {exc}",
                "query_text_used": True,
                "official_processor_path": True,
            }


class QwenAdapter(MLLMAdapter):
    name = "qwen"

    def default_module_path(self) -> str | None:
        return "transformers"

    def default_class_name(self) -> str | None:
        return "AutoModelForVision2Seq"

    def load(self, *, allow_real_model_loading: bool = False, device: str = "cpu", dtype: str = "float32") -> AdapterStatus:
        status = super().load(allow_real_model_loading=allow_real_model_loading, device=device, dtype=dtype)
        if status.status != "real":
            return status
        processor_path = self.config.get("processor_path") or self._checkpoint()
        if not processor_path:
            self.status = AdapterStatus(self.name, "blocked", "Qwen processor_path or model_id is required for official processor loading")
            return self.status
        if _looks_like_local_path(str(processor_path)) and not _resolve_local_path(processor_path).exists():
            self.status = AdapterStatus(
                self.name,
                "blocked",
                f"Qwen official processor path does not exist: {processor_path}",
                metadata=status.metadata,
            )
            return self.status
        processor_module = self.config.get("processor_module_path") or "transformers"
        processor_class = self.config.get("processor_class_name") or self.config.get("processor_class_or_factory") or "AutoProcessor"
        try:
            factory = getattr(importlib.import_module(str(processor_module)), str(processor_class))
            processor_ref = _model_reference_for_loading(processor_path)
            with _suppress_transformers_torch_dtype_warning():
                self.processor = factory.from_pretrained(processor_ref, **_processor_from_pretrained_kwargs(self.config))
            status.metadata["processor_status"] = "real"
            status.metadata["processor_path"] = processor_path
            status.metadata["resolved_processor_path"] = processor_ref
            status.metadata["processor_module_path"] = processor_module
            status.metadata["processor_class_name"] = processor_class
            status.metadata["official_processor_path"] = True
            status.metadata["chat_template_path"] = bool(hasattr(self.processor, "apply_chat_template"))
            status.metadata["qwen_vl_utils_available"] = _qwen_vl_utils_available()
            status.metadata["processor_kwargs"] = _jsonable_processor_kwargs(_processor_from_pretrained_kwargs(self.config))
        except Exception as exc:
            self.status = AdapterStatus(
                self.name,
                "blocked",
                f"Qwen official processor loading failed: {exc}",
                metadata=status.metadata,
            )
            return self.status
        return status

    def supports_official_processor_path(self) -> bool:
        return True

    def generate(
        self,
        *,
        query_text: str,
        video: torch.Tensor,
        visual_tokens: torch.Tensor | None = None,
        max_new_tokens: int = 32,
        video_path: str | None = None,
        autogaze: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if visual_tokens is not None and not self.supports_direct_visual_tokens():
            return {
                "status": "blocked",
                "answer": None,
                "reason": "Qwen direct visual token injection is unsupported; use official processor/input-level selection.",
                "query_text_used": True,
            }
        if self.model is not None and self.processor is not None:
            return self._generate_with_official_processor(
                query_text=query_text,
                video=video,
                video_path=video_path,
                max_new_tokens=max_new_tokens,
                autogaze=autogaze,
            )
        return super().generate(
            query_text=query_text,
            video=video,
            visual_tokens=visual_tokens,
            max_new_tokens=max_new_tokens,
            video_path=video_path,
            autogaze=autogaze,
        )

    def _generate_with_official_processor(
        self,
        *,
        query_text: str,
        video: torch.Tensor,
        video_path: str | None,
        max_new_tokens: int,
        autogaze: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not query_text:
            return {"status": "blocked", "answer": None, "reason": "query text is required", "query_text_used": False}
        try:
            use_autogaze_mask = self._qwen_autogaze_integration_mode() == "qwen_vision_mask"
            prefer_processed_tensor = use_autogaze_mask or bool(
                self.config.get("prefer_processed_tensor", self.config.get("qwen_prefer_processed_tensor", False))
            )
            inputs, input_metadata = self._prepare_official_qwen_inputs(
                query_text=query_text,
                video=video,
                video_path=None if prefer_processed_tensor else video_path,
                prefer_processed_tensor=prefer_processed_tensor,
            )
            if input_metadata.get("status") == "blocked":
                return {
                    "status": "blocked",
                    "answer": None,
                    "reason": str(input_metadata.get("reason")),
                    "query_text_used": True,
                    "official_processor_path": True,
                    "metadata": input_metadata,
                }
            if isinstance(inputs, Mapping):
                model_device = getattr(self.model, "device", self.device)
                inputs = {
                    key: value.to(model_device) if isinstance(value, torch.Tensor) else value
                    for key, value in inputs.items()
                }
            mask_context, mask_metadata = self._qwen_autogaze_mask_context(
                inputs=inputs,
                autogaze=autogaze,
            )
            if mask_metadata.get("status") == "blocked":
                return {
                    "status": "blocked",
                    "answer": None,
                    "reason": str(mask_metadata.get("reason")),
                    "query_text_used": True,
                    "official_processor_path": True,
                    "metadata": {**input_metadata, **mask_metadata},
                }
            with mask_context, torch.inference_mode():
                outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            input_ids = inputs.get("input_ids") if isinstance(inputs, Mapping) else None
            decode_input = outputs[:, input_ids.shape[1] :] if isinstance(outputs, torch.Tensor) and isinstance(input_ids, torch.Tensor) else outputs
            decoded = self.processor.batch_decode(decode_input, skip_special_tokens=True)
            answer = str(decoded[0]).strip() if decoded else ""
            return {
                "status": "real",
                "answer": answer,
                "reason": None,
                "query_text_used": True,
                "official_processor_path": True,
                "metadata": {
                    **input_metadata,
                    **mask_metadata,
                    "max_new_tokens": max_new_tokens,
                },
            }
        except Exception as exc:  # pragma: no cover - real model path is environment-dependent.
            return {
                "status": "blocked",
                "answer": None,
                "reason": f"Qwen official processor generation failed: {exc}",
                "query_text_used": True,
                "official_processor_path": True,
            }

    def _prepare_official_qwen_inputs(
        self,
        *,
        query_text: str,
        video: torch.Tensor,
        video_path: str | None,
        prefer_processed_tensor: bool = False,
    ) -> tuple[Any, dict[str, Any]]:
        if not hasattr(self.processor, "apply_chat_template"):
            return None, {
                "status": "blocked",
                "reason": "Qwen official processor path requires apply_chat_template; loaded processor does not expose it.",
                "chat_template_path": False,
            }
        prompt_template = str(self.config.get("prompt_template") or "Question: {prompt}")
        prompt_text = prompt_template.format(prompt=query_text, query=query_text, video_token="").strip()
        video_input, video_input_kind = _official_video_input(
            video=video,
            video_path=None if prefer_processed_tensor else video_path,
        )
        message_video_ref = _qwen_message_video_reference(video_input, video_input_kind)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": message_video_ref},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        qwen_vl_utils_available = _qwen_vl_utils_available()
        use_qwen_vl_utils = bool(self.config.get("use_qwen_vl_utils", True))
        require_qwen_vl_utils = bool(self.config.get("require_qwen_vl_utils", False))
        call_kwargs = dict(self.config.get("processor_call_kwargs") or {})
        metadata: dict[str, Any] = {
            "prompt_template": prompt_template,
            "prompt_text": prompt_text,
            "video_input_kind": video_input_kind,
            "chat_template_path": True,
            "qwen_vl_utils_available": qwen_vl_utils_available,
            "qwen_vl_utils_requested": use_qwen_vl_utils,
            "vision_preprocess_path": "processor_direct_video_payload",
            "processor_call_kwargs": _jsonable_processor_kwargs(call_kwargs),
            "direct_visual_token_injection": False,
            "qwen_autogaze_prefer_processed_tensor": bool(prefer_processed_tensor),
        }

        if use_qwen_vl_utils and qwen_vl_utils_available and video_input_kind in {"video_path", "video_reference"}:
            try:
                from qwen_vl_utils import process_vision_info

                try:
                    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
                except TypeError:
                    image_inputs, video_inputs = process_vision_info(messages)
                    video_kwargs = {}
                metadata["vision_preprocess_path"] = "qwen_vl_utils"
                metadata["qwen_vl_utils_video_kwargs"] = _jsonable_processor_kwargs(video_kwargs)
                processor_kwargs = {**video_kwargs, **call_kwargs}
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    return_tensors="pt",
                    **processor_kwargs,
                )
                return inputs, metadata
            except Exception as exc:
                return None, {
                    **metadata,
                    "status": "blocked",
                    "reason": f"Qwen qwen_vl_utils preprocessing failed: {exc}",
                }
        if require_qwen_vl_utils and video_input_kind in {"video_path", "video_reference"}:
            return None, {
                **metadata,
                "status": "blocked",
                "reason": "Qwen qwen_vl_utils preprocessing is required by config but qwen_vl_utils is not available.",
            }

        processor_videos = _qwen_processor_video_payload(video_input)
        metadata["processor_video_batch_count"] = len(processor_videos) if isinstance(processor_videos, list) else 1
        try:
            inputs = self.processor(text=[text], videos=processor_videos, return_tensors="pt", **call_kwargs)
        except Exception as exc:
            return None, {
                **metadata,
                "status": "blocked",
                "reason": f"Qwen official processor generation input preparation failed: {exc}",
            }
        return inputs, metadata

    def _qwen_autogaze_integration_mode(self) -> str:
        return str(self.config.get("autogaze_integration") or self.config.get("qwen_autogaze_integration") or "none")

    def _qwen_autogaze_mask_context(
        self,
        *,
        inputs: Mapping[str, Any] | Any,
        autogaze: Mapping[str, Any] | None,
    ) -> tuple[Any, dict[str, Any]]:
        from contextlib import nullcontext

        mode = self._qwen_autogaze_integration_mode()
        base_metadata: dict[str, Any] = {
            "qwen_autogaze_integration": mode,
            "qwen_visual_mask_applied": False,
            "qwen_visual_tokens_shortened": False,
            "qwen_encoder_side_acceleration_claimed": False,
            "direct_visual_token_injection": False,
        }
        if mode in {"", "none", "disabled"}:
            return nullcontext(), base_metadata
        if mode != "qwen_vision_mask":
            return nullcontext(), {
                **base_metadata,
                "status": "blocked",
                "reason": f"Unsupported Qwen AutoGaze integration mode: {mode}",
            }
        if not isinstance(inputs, Mapping):
            return nullcontext(), {
                **base_metadata,
                "status": "blocked",
                "reason": "Qwen AutoGaze vision mask requires mapping-style processor inputs",
            }
        if not isinstance(autogaze, Mapping):
            return nullcontext(), {
                **base_metadata,
                "status": "blocked",
                "reason": "Qwen AutoGaze vision mask requested but AutoGaze metadata was not provided",
            }
        if not bool(autogaze.get("autogaze_enabled", False)):
            return nullcontext(), {
                **base_metadata,
                "status": "blocked",
                "reason": "Qwen AutoGaze vision mask requested but AutoGaze is disabled",
            }
        autogaze_status = str(autogaze.get("status") or "unknown")
        if autogaze_status == "blocked":
            return nullcontext(), {
                **base_metadata,
                "status": "blocked",
                "reason": f"Qwen AutoGaze vision mask requested but AutoGaze is blocked: {autogaze.get('reason')}",
            }
        grid_thw = inputs.get("video_grid_thw")
        if not isinstance(grid_thw, torch.Tensor) or grid_thw.numel() < 3:
            return nullcontext(), {
                **base_metadata,
                "status": "blocked",
                "reason": "Qwen AutoGaze vision mask requires video_grid_thw from the official Qwen processor",
            }
        if grid_thw.ndim == 1:
            grid = grid_thw
        else:
            grid = grid_thw[0]
        qwen_t = int(grid[0].item())
        qwen_h = int(grid[1].item())
        qwen_w = int(grid[2].item())
        if qwen_t <= 0 or qwen_h <= 0 or qwen_w <= 0:
            return nullcontext(), {
                **base_metadata,
                "status": "blocked",
                "reason": f"Qwen processor returned invalid video_grid_thw: {[qwen_t, qwen_h, qwen_w]}",
            }

        mask, mask_stats = _qwen_patch_mask_from_autogaze(
            autogaze,
            qwen_t=qwen_t,
            qwen_h=qwen_h,
            qwen_w=qwen_w,
            min_keep_tokens=int(self.config.get("qwen_autogaze_min_keep_tokens", 1)),
            empty_chunk_policy=str(self.config.get("qwen_autogaze_empty_chunk_policy", "keep_center")),
        )
        if mask is None:
            return nullcontext(), {
                **base_metadata,
                "status": "blocked",
                "reason": str(mask_stats.get("reason") or "failed to build Qwen AutoGaze patch mask"),
                **mask_stats,
            }
        ctx, hook_metadata = self._qwen_visual_patch_mask_hook(mask)
        return ctx, {
            **base_metadata,
            **mask_stats,
            **hook_metadata,
            "qwen_visual_mask_applied": True,
            "status": "ready",
        }

    def _qwen_visual_patch_mask_hook(self, flat_mask: torch.Tensor) -> tuple[Any, dict[str, Any]]:
        visual = getattr(self.model, "visual", None)
        patch_embed = getattr(visual, "patch_embed", None)
        if patch_embed is None or not hasattr(patch_embed, "register_forward_hook"):
            from contextlib import nullcontext

            return nullcontext(), {
                "status": "blocked",
                "reason": "Qwen model.visual.patch_embed is not available for AutoGaze vision masking",
            }

        hook_state: dict[str, Any] = {"qwen_patch_embed_hook_registered": True}

        @contextmanager
        def _hook_context():
            mask = flat_mask.detach().float()

            def _hook(_module: Any, _input: Any, output: Any) -> Any:
                if not isinstance(output, torch.Tensor):
                    hook_state["qwen_patch_embed_hook_error"] = "patch_embed output is not a tensor"
                    return output
                mask_device = mask.to(device=output.device, dtype=output.dtype)
                if output.ndim == 2:
                    n = min(int(output.shape[0]), int(mask_device.shape[0]))
                    result = output.clone()
                    result[:n] = result[:n] * mask_device[:n].unsqueeze(-1)
                    hook_state["qwen_patch_embed_output_tokens"] = int(output.shape[0])
                    hook_state["qwen_patch_embed_mask_tokens_applied"] = n
                    return result
                if output.ndim == 3:
                    n = min(int(output.shape[1]), int(mask_device.shape[0]))
                    result = output.clone()
                    result[:, :n, :] = result[:, :n, :] * mask_device[:n].view(1, n, 1)
                    hook_state["qwen_patch_embed_output_tokens"] = int(output.shape[1])
                    hook_state["qwen_patch_embed_mask_tokens_applied"] = n
                    return result
                hook_state["qwen_patch_embed_hook_error"] = f"unsupported patch_embed output shape: {tuple(output.shape)}"
                return output

            handle = patch_embed.register_forward_hook(_hook)
            try:
                yield
            finally:
                handle.remove()

        return _hook_context(), hook_state


class GenericMLLMAdapter(MLLMAdapter):
    name = "generic_mllm"


class ExternalMLLMAdapter(MLLMAdapter):
    default_model_id = ""
    encoder_type = "unknown"
    llm_type = "unknown"
    supported_integration_modes: tuple[str, ...] = (
        "official_processor",
        "autogaze_frame_selection",
        "autogaze_chop_selection",
        "autogaze_zero_mask",
    )
    direct_token_injection_supported: bool | str = False
    native_sparse_patch_supported: bool | str = False
    light_modified_sparse_supported: bool | str = False
    rope_sparse_patch_supported: bool | str = False
    autogaze_zero_mask_supported: bool | str = True
    post_encoder_zero_mask_supported: bool | str = "unknown"
    input_selection_only_supported: bool | str = True
    native_sparse_patch_supported: bool | str = False
    light_modified_sparse_supported: bool | str = False
    rope_sparse_patch_supported: bool | str = False
    autogaze_zero_mask_supported: bool | str = True
    post_encoder_zero_mask_supported: bool | str = "unknown"
    input_selection_only_supported: bool | str = True
    siglip_based: bool | str = "unknown"
    positional_encoding_status = "needs_code_inspection"
    token_count_compatibility = "needs_code_inspection"
    positional_encoding_type = "unknown"
    dense_grid_dependency: bool | str = "unknown"
    variable_visual_token_support: bool | str = "unknown"
    visual_placeholder_dynamic: bool | str = "unknown"
    direct_sparse_autogaze_supported: bool | str = "unknown"
    patch_grid_accessible: bool | str = "unknown"
    deterministic_position_adaptation: bool | str = "unknown"
    projector_status = "needs_code_inspection"
    placeholder_status = "needs_code_inspection"
    requires_official_processor = True
    requires_training_or_trainable_adapter: bool | str = False
    recommended_mode = "autogaze_frame_selection"
    recommended_first_smoke_mode = "autogaze_frame_selection"
    risk_level = "high"
    compatibility_status = "needs_code_inspection"
    local_asset_status = "not_checked"
    official_processor_support_status = "stub_until_assets_verified"
    input_selection_support_status = "supported_as_input_level_selection_only"
    zero_mask_support_status = "supported_as_image_space_probe_only"
    unsupported_mode_reasons: dict[str, str] = {}

    def load(self, *, allow_real_model_loading: bool = False, device: str = "cpu", dtype: str = "float32") -> AdapterStatus:
        self.device = device
        metadata = self._adapter_report(integration_mode="load", generation_ran=False)
        if not allow_real_model_loading:
            self.status = AdapterStatus(
                self.name,
                "stub-only",
                f"{self.name} external MLLM adapter is metadata/stub-only; real loading disabled",
                metadata=metadata,
            )
            return self.status
        checkpoint = self._checkpoint()
        if not checkpoint:
            self.status = AdapterStatus(
                self.name,
                "blocked",
                f"{self.name} real loading requires an explicit model_id or checkpoint_path; default model IDs are metadata only",
                metadata=metadata,
            )
            return self.status
        if _looks_like_local_path(str(checkpoint)):
            local_checkpoint = _resolve_local_path(checkpoint)
            if not local_checkpoint.exists():
                self.status = AdapterStatus(self.name, "blocked", f"{self.name} checkpoint path does not exist: {checkpoint}", metadata=metadata)
                return self.status
            missing_shards = _missing_sharded_checkpoint_files(local_checkpoint)
            if missing_shards:
                self.status = AdapterStatus(
                    self.name,
                    "blocked",
                    f"{self.name} checkpoint is incomplete; missing shard files: {', '.join(missing_shards[:8])}",
                    metadata={**metadata, "missing_shards": missing_shards},
                )
                return self.status
        checkpoint_ref = _model_reference_for_loading(checkpoint)
        processor_path = self.config.get("processor_path") or checkpoint
        processor_ref = _model_reference_for_loading(processor_path)
        module_path = self.config.get("module_path") or "transformers"
        class_candidates = self._model_class_candidates()
        processor_module = self.config.get("processor_module_path") or "transformers"
        processor_class_name = self.config.get("processor_class_name") or self.config.get("processor_class_or_factory") or "AutoProcessor"
        errors: list[str] = []
        try:
            module = importlib.import_module(str(module_path))
            load_kwargs = _from_pretrained_kwargs(self.config, dtype=dtype)
            with _suppress_transformers_torch_dtype_warning():
                for class_name in class_candidates:
                    try:
                        factory = getattr(module, str(class_name))
                        self.model = factory.from_pretrained(checkpoint_ref, **load_kwargs)
                        metadata["model_class_name"] = class_name
                        break
                    except Exception as exc:  # pragma: no cover - environment/model dependent.
                        errors.append(f"{class_name}: {exc}")
                else:
                    self.status = AdapterStatus(
                        self.name,
                        "blocked",
                        f"{self.name} real model loading failed for candidate classes: {' | '.join(errors)}",
                        metadata=metadata,
                    )
                    return self.status
            if hasattr(self.model, "to") and "device_map" not in load_kwargs:
                self.model.to(device)
            if hasattr(self.model, "eval"):
                self.model.eval()
            processor_factory = getattr(importlib.import_module(str(processor_module)), str(processor_class_name))
            with _suppress_transformers_torch_dtype_warning():
                self.processor = processor_factory.from_pretrained(processor_ref, **_processor_from_pretrained_kwargs(self.config))
            self.status = AdapterStatus(
                self.name,
                "real",
                metadata={
                    **metadata,
                    "checkpoint": checkpoint,
                    "resolved_checkpoint": checkpoint_ref,
                    "processor_path": processor_path,
                    "resolved_processor_path": processor_ref,
                    "module_path": module_path,
                    "processor_module_path": processor_module,
                    "processor_class_name": processor_class_name,
                    "local_files_only": bool(self.config.get("local_files_only", False)),
                    "trust_remote_code": bool(self.config.get("trust_remote_code", False)),
                    "official_processor_path": True,
                },
            )
        except Exception as exc:  # pragma: no cover - real model path is environment-dependent.
            self.status = AdapterStatus(self.name, "blocked", f"{self.name} official processor/model loading failed: {exc}", metadata=metadata)
        return self.status

    def _model_class_candidates(self) -> tuple[str, ...]:
        configured = self.config.get("class_name") or self.config.get("class_or_factory")
        if configured:
            return (str(configured),)
        return ("AutoModelForVision2Seq", "AutoModelForCausalLM", "AutoModel")

    def supports_official_processor_path(self) -> bool:
        return "official_processor" in self.supported_integration_modes

    def supports_direct_visual_tokens(self) -> bool:
        return self.direct_token_injection_supported is True

    def supports_native_sparse_patch(self) -> bool:
        return self.native_sparse_patch_supported is True

    def supports_light_modified_sparse(self) -> bool:
        return self.light_modified_sparse_supported is True

    def supports_rope_sparse_patch(self) -> bool:
        return self.rope_sparse_patch_supported is True

    def supports_direct_visual_token_injection(self) -> bool:
        return self.supports_direct_visual_tokens()

    def supports_direct_sparse_autogaze(self) -> bool:
        return self.direct_sparse_autogaze_supported is True

    def supports_input_level_selection(self) -> bool:
        return bool({"autogaze_frame_selection", "autogaze_chop_selection"} & set(self.supported_integration_modes))

    def supports_autogaze_zero_mask(self) -> bool:
        return self.autogaze_zero_mask_supported is True

    def supports_post_encoder_zero_mask(self) -> bool:
        return self.post_encoder_zero_mask_supported is True

    def supports_post_encoder_pruning(self) -> bool:
        return "post_encoder_pruning" in self.supported_integration_modes

    def run_official_processor_path(self, **_: Any) -> dict[str, Any]:
        if self.model is not None and self.processor is not None:
            return self.generate(**_)
        return self._mode_status("official_processor", **_)

    def run_autogaze_frame_selection_path(self, **_: Any) -> dict[str, Any]:
        if self.model is not None and self.processor is not None:
            return self._generate_with_selection_mode("autogaze_frame_selection", **_)
        return self._mode_status("autogaze_frame_selection", **_)

    def run_autogaze_chop_selection_path(self, **_: Any) -> dict[str, Any]:
        if self.model is not None and self.processor is not None:
            return self._generate_with_selection_mode("autogaze_chop_selection", **_)
        return self._mode_status("autogaze_chop_selection", **_)

    def run_autogaze_zero_mask_path(self, **_: Any) -> dict[str, Any]:
        if self.model is not None and self.processor is not None:
            return self._generate_with_selection_mode("autogaze_zero_mask", **_)
        return self._mode_status("autogaze_zero_mask", **_)

    def run_siglip_sparse_patch_path(self, **_: Any) -> dict[str, Any]:
        return self._mode_status("siglip_sparse_patch", **_)

    def run_native_sparse_patch_path(self, **_: Any) -> dict[str, Any]:
        return self._mode_status("native_sparse_patch", **_)

    def run_light_modified_sparse_path(self, **_: Any) -> dict[str, Any]:
        return self._mode_status("light_modified_sparse", **_)

    def run_rope_sparse_patch_path(self, **_: Any) -> dict[str, Any]:
        return self._mode_status("rope_sparse_patch", **_)

    def run_post_encoder_pruning_path(self, **_: Any) -> dict[str, Any]:
        return self._mode_status("post_encoder_pruning", **_)

    def run_post_encoder_zero_mask_path(self, **_: Any) -> dict[str, Any]:
        return self._mode_status("post_encoder_zero_mask", **_)

    def run_direct_visual_token_injection_path(self, **_: Any) -> dict[str, Any]:
        return self._mode_status("direct_visual_token_injection", **_)

    def _mode_status(self, integration_mode: str, **kwargs: Any) -> dict[str, Any]:
        if integration_mode not in self.supported_integration_modes or integration_mode == "direct_visual_token_injection":
            reason = self.unsupported_mode_reasons.get(
                integration_mode,
                f"{integration_mode} is not verified for {self.name}",
            )
            raise NotImplementedError(self._unsupported_message(integration_mode, reason))
        if self.status.status == "dummy":
            return self._dummy_generation(
                query_text=str(kwargs.get("query_text") or ""),
                max_new_tokens=int(kwargs.get("max_new_tokens") or 32),
                integration_mode=integration_mode,
            )
        return {
            "status": "stub-only",
            "answer": None,
            "reason": (
                f"{self.name} {integration_mode} is registered as a lightweight plan/stub. "
                "Implement the model official processor path and run a real smoke test before marking it runnable."
            ),
            "query_text_used": True,
            "metadata": self._adapter_report(integration_mode=integration_mode, generation_ran=False),
        }

    def generate(
        self,
        *,
        query_text: str,
        video: torch.Tensor,
        visual_tokens: torch.Tensor | None = None,
        max_new_tokens: int = 32,
        video_path: str | None = None,
        autogaze: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if visual_tokens is not None:
            return {
                "status": "blocked",
                "answer": None,
                "reason": f"{self.name} direct visual token injection is not verified; use official_processor or input-selection modes",
                "query_text_used": True,
                "metadata": self._adapter_report(
                    integration_mode="direct_visual_token_injection",
                    generation_ran=False,
                    unsupported_reason="direct visual token injection is not verified",
                ),
            }
        if self.model is not None and self.processor is not None:
            return self._generate_with_standard_processor(
                query_text=query_text,
                video=video,
                video_path=video_path,
                max_new_tokens=max_new_tokens,
                integration_mode="official_processor",
            )
        return super().generate(
            query_text=query_text,
            video=video,
            visual_tokens=visual_tokens,
            max_new_tokens=max_new_tokens,
            video_path=video_path,
            autogaze=autogaze,
        )

    def _generate_with_selection_mode(self, integration_mode: str, **kwargs: Any) -> dict[str, Any]:
        result = self.generate(**kwargs)
        metadata = dict(result.get("metadata") or {})
        metadata["integration_mode"] = integration_mode
        metadata["autogaze_input_selection_mode"] = integration_mode
        metadata["autogaze_visual_tokens_injected"] = False
        result["metadata"] = metadata
        return result

    def _generate_with_standard_processor(
        self,
        *,
        query_text: str,
        video: torch.Tensor,
        video_path: str | None,
        max_new_tokens: int,
        integration_mode: str,
    ) -> dict[str, Any]:
        if not query_text:
            return {"status": "blocked", "answer": None, "reason": "query text is required", "query_text_used": False}
        try:
            prompt_template = str(self.config.get("prompt_template") or "{prompt}")
            prompt_text = prompt_template.format(prompt=query_text, query=query_text, video_token="<video>").strip()
            video_input, video_input_kind = _official_video_input(video=video, video_path=video_path)
            call_kwargs = dict(self.config.get("processor_call_kwargs") or {})
            inputs, input_metadata = self._prepare_standard_processor_inputs(
                prompt_text=prompt_text,
                video_input=video_input,
                video_input_kind=video_input_kind,
                call_kwargs=call_kwargs,
            )
            if input_metadata.get("status") == "blocked":
                return {
                    "status": "blocked",
                    "answer": None,
                    "reason": str(input_metadata.get("reason")),
                    "query_text_used": True,
                    "metadata": {
                        **self._adapter_report(integration_mode=integration_mode, generation_ran=False),
                        **input_metadata,
                    },
                }
            if isinstance(inputs, Mapping):
                model_device = getattr(self.model, "device", self.device)
                inputs = {
                    key: value.to(model_device) if isinstance(value, torch.Tensor) else value
                    for key, value in inputs.items()
                }
            with torch.inference_mode():
                outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            input_ids = inputs.get("input_ids") if isinstance(inputs, Mapping) else None
            decode_input = outputs[:, input_ids.shape[1] :] if isinstance(outputs, torch.Tensor) and isinstance(input_ids, torch.Tensor) else outputs
            decoded = self.processor.batch_decode(decode_input, skip_special_tokens=True) if hasattr(self.processor, "batch_decode") else [str(decode_input)]
            answer = str(decoded[0]).strip() if decoded else ""
            return {
                "status": "real",
                "answer": answer,
                "reason": None,
                "query_text_used": True,
                "official_processor_path": True,
                "metadata": {
                    **self._adapter_report(integration_mode=integration_mode, generation_ran=True),
                    **input_metadata,
                    "prompt_template": prompt_template,
                    "prompt_text": prompt_text,
                    "video_input_kind": video_input_kind,
                    "max_new_tokens": max_new_tokens,
                    "autogaze_visual_tokens_injected": False,
                },
            }
        except Exception as exc:  # pragma: no cover - real model path is environment-dependent.
            return {
                "status": "blocked",
                "answer": None,
                "reason": f"{self.name} official processor generation failed: {exc}",
                "query_text_used": True,
                "official_processor_path": True,
                "metadata": self._adapter_report(integration_mode=integration_mode, generation_ran=False),
            }

    def _prepare_standard_processor_inputs(
        self,
        *,
        prompt_text: str,
        video_input: Any,
        video_input_kind: str,
        call_kwargs: Mapping[str, Any],
    ) -> tuple[Any, dict[str, Any]]:
        metadata: dict[str, Any] = {
            "video_input_kind": video_input_kind,
            "processor_call_kwargs": _jsonable_processor_kwargs(call_kwargs),
            "processor_input_attempts": [],
        }
        if hasattr(self.processor, "apply_chat_template"):
            message_video_ref = _qwen_message_video_reference(video_input, video_input_kind)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": message_video_ref},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            try:
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                metadata["chat_template_path"] = True
            except Exception as exc:
                return None, {**metadata, "status": "blocked", "reason": f"{self.name} chat template failed: {exc}"}
        else:
            text = prompt_text
            metadata["chat_template_path"] = False

        video_payload = _qwen_processor_video_payload(video_input)
        attempts = (
            ("batched_text_videos", {"text": [text], "videos": video_payload, "return_tensors": "pt", **dict(call_kwargs)}),
            ("single_text_videos", {"text": text, "videos": video_input, "return_tensors": "pt", **dict(call_kwargs)}),
            ("batched_text_images", {"text": [text], "images": video_payload, "return_tensors": "pt", **dict(call_kwargs)}),
        )
        errors: list[str] = []
        for label, kwargs in attempts:
            metadata["processor_input_attempts"].append(label)
            try:
                return self.processor(**kwargs), metadata
            except Exception as exc:
                errors.append(f"{label}: {exc}")
        return None, {
            **metadata,
            "status": "blocked",
            "reason": f"{self.name} official processor input preparation failed: {' | '.join(errors)}",
        }

    def _unsupported_message(self, integration_mode: str, reason: str) -> str:
        return (
            f"{self.name} does not support {integration_mode}: {reason}. "
            "No fallback to NVILA, modified SigLIP, or trainable projection adapter is allowed."
        )

    def _adapter_report(
        self,
        *,
        integration_mode: str,
        generation_ran: bool,
        unsupported_reason: str | None = None,
    ) -> dict[str, Any]:
        requested_model = self.config.get("model_id") or self.config.get("checkpoint_path") or self.default_model_id
        return {
            "requested_model": requested_model,
            "actual_model_loaded": requested_model if self.status.status == "real" else None,
            "adapter": self.name,
            "integration_mode": integration_mode,
            "real_checkpoint_loaded": self.status.status == "real",
            "generation_ran": generation_ran,
            "unsupported_reason": unsupported_reason,
            "encoder_type": self.encoder_type,
            "llm_type": self.llm_type,
            "supported_integration_modes": list(self.supported_integration_modes),
            "direct_token_injection_supported": self.direct_token_injection_supported,
            "native_sparse_patch_supported": self.native_sparse_patch_supported,
            "light_modified_sparse_supported": self.light_modified_sparse_supported,
            "rope_sparse_patch_supported": self.rope_sparse_patch_supported,
            "autogaze_zero_mask_supported": self.autogaze_zero_mask_supported,
            "post_encoder_zero_mask_supported": self.post_encoder_zero_mask_supported,
            "input_selection_only_supported": self.input_selection_only_supported,
            "siglip_based": self.siglip_based,
            "positional_encoding_compatibility": self.positional_encoding_status,
            "positional_encoding_type": self.positional_encoding_type,
            "dense_grid_dependency": self.dense_grid_dependency,
            "variable_visual_token_support": self.variable_visual_token_support,
            "visual_placeholder_dynamic": self.visual_placeholder_dynamic,
            "direct_sparse_autogaze_supported": self.direct_sparse_autogaze_supported,
            "patch_grid_accessible": self.patch_grid_accessible,
            "deterministic_position_adaptation": self.deterministic_position_adaptation,
            "projector_status": self.projector_status,
            "placeholder_status": self.placeholder_status,
            "token_count_compatibility": self.token_count_compatibility,
            "requires_official_processor": self.requires_official_processor,
            "requires_training_or_trainable_adapter": self.requires_training_or_trainable_adapter,
            "recommended_mode": self.recommended_mode,
            "recommended_first_smoke_mode": self.recommended_first_smoke_mode,
            "risk_level": self.risk_level,
            "compatibility_status": self.compatibility_status,
            "local_asset_status": self.local_asset_status,
            "official_processor_support_status": self.official_processor_support_status,
            "input_selection_support_status": self.input_selection_support_status,
            "zero_mask_support_status": self.zero_mask_support_status,
        }


class LlavaOVAdapter(ExternalMLLMAdapter):
    name = "llava_ov"
    default_model_id = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    encoder_type = "SigLIP vision encoder, patch14, anyres/video pooling"
    llm_type = "Qwen2-7B"
    supported_integration_modes = (
        "official_processor",
        "autogaze_frame_selection",
        "autogaze_chop_selection",
        "autogaze_zero_mask",
        "post_encoder_pruning",
    )
    direct_token_injection_supported = "unknown"
    siglip_based = True
    positional_encoding_type = "absolute_2d_siglip_plus_qwen2_rope"
    dense_grid_dependency = True
    variable_visual_token_support = "unknown"
    visual_placeholder_dynamic = "unknown"
    direct_sparse_autogaze_supported = "unknown"
    patch_grid_accessible = "unknown"
    deterministic_position_adaptation = "unknown"
    projector_status = "needs_code_inspection: projector after anyres/video pooling"
    placeholder_status = "needs_code_inspection: image/video token placeholders must match packed token count"
    positional_encoding_status = "needs_code_inspection: anyres/newline and video pooling must be preserved"
    token_count_compatibility = "needs_code_inspection: video path pools to fixed per-frame sequence"
    recommended_mode = "autogaze_frame_selection"
    recommended_first_smoke_mode = "autogaze_frame_selection"
    risk_level = "medium"
    compatibility_status = "siglip_candidate"
    unsupported_mode_reasons = {
        "siglip_sparse_patch": "candidate only; anyres/video pooling, positional remapping, projector, and placeholder counts are not verified",
        "rope_sparse_patch": "LLaVA-OneVision sparse RoPE path is not verified; video pooling and placeholder counts still require dense/packed tokens",
        "direct_visual_token_injection": "placeholder count, pooled video token count, and position IDs have not been verified",
    }


class LongVAAdapter(ExternalMLLMAdapter):
    name = "longva"
    default_model_id = "lmms-lab/LongVA-7B"
    encoder_type = "CLIP ViT-L/14-336 vision tower with anyres/unires processing"
    llm_type = "Qwen2-7B"
    supported_integration_modes = ("official_processor", "autogaze_frame_selection", "autogaze_chop_selection", "autogaze_zero_mask", "post_encoder_pruning")
    direct_token_injection_supported = False
    siglip_based = False
    positional_encoding_type = "clip_absolute_2d_plus_qwen2_rope"
    dense_grid_dependency = True
    variable_visual_token_support = "unknown"
    visual_placeholder_dynamic = "unknown"
    direct_sparse_autogaze_supported = False
    patch_grid_accessible = "unknown"
    deterministic_position_adaptation = False
    projector_status = "blocked_for_direct_sparse: CLIP/unires projector contract is model-specific"
    placeholder_status = "needs_code_inspection: LLaVA-style placeholder expansion"
    positional_encoding_status = "blocked_for_direct_sparse: CLIP/anyres token layout differs from AutoGaze SigLIP path"
    token_count_compatibility = "blocked_for_direct_sparse: unires pooling/projector contract needs model-specific code"
    recommended_mode = "autogaze_frame_selection"
    recommended_first_smoke_mode = "autogaze_frame_selection"
    risk_level = "medium"
    compatibility_status = "input_selection_only"
    unsupported_mode_reasons = {
        "siglip_sparse_patch": "LongVA-7B config uses a CLIP vision tower, not SigLIP",
        "rope_sparse_patch": "LongVA direct sparse RoPE path is not verified and still depends on CLIP dense/anyres visual tokens",
        "direct_visual_token_injection": "visual token placeholder and projector compatibility are not verified",
    }


class LongVILAAdapter(ExternalMLLMAdapter):
    name = "longvila_r1"
    default_model_id = "Efficient-Large-Model/LongVILA-R1-7B"
    encoder_type = "VILA SigLIP SO400M patch14-448 with TSP video encoder"
    llm_type = "Qwen2-7B"
    supported_integration_modes = (
        "official_processor",
        "autogaze_frame_selection",
        "autogaze_chop_selection",
        "autogaze_zero_mask",
        "post_encoder_pruning",
    )
    direct_token_injection_supported = "unknown"
    siglip_based = True
    positional_encoding_type = "siglip_absolute_2d_interpolation_plus_tsp_video_pooling_and_qwen2_rope"
    dense_grid_dependency = "unknown"
    variable_visual_token_support = "unknown"
    visual_placeholder_dynamic = "unknown"
    direct_sparse_autogaze_supported = "unknown"
    patch_grid_accessible = "unknown"
    deterministic_position_adaptation = "unknown"
    projector_status = "needs_code_inspection: VILA projector/TSP pipeline"
    placeholder_status = "needs_code_inspection: VILA _embed media placeholder alignment"
    positional_encoding_status = "needs_code_inspection: VILA SigLIP interpolation and TSP video pooling must be preserved"
    token_count_compatibility = "needs_code_inspection: projector/placeholder behavior must match sparse token count"
    recommended_mode = "autogaze_frame_selection"
    recommended_first_smoke_mode = "autogaze_frame_selection"
    risk_level = "medium"
    compatibility_status = "native_candidate"
    unsupported_mode_reasons = {
        "siglip_sparse_patch": "candidate only; VILA SigLIP grid mapping, TSP pooling, projector, and placeholder alignment are not verified",
        "rope_sparse_patch": "LongVILA sparse RoPE path is not verified; SigLIP/TSP media pipeline should be inspected first",
        "direct_visual_token_injection": "VILA _embed/projector placeholder alignment has not been verified for externally selected tokens",
    }


class ApolloAdapter(ExternalMLLMAdapter):
    name = "apollo"
    default_model_id = "GoodiesHere/Apollo-LMMs-Apollo-7B-t32"
    encoder_type = "hybrid SigLIP SO400M plus InternVideo2 vision tower with Perceiver connector"
    llm_type = "Qwen2-7B"
    supported_integration_modes = ("official_processor", "autogaze_frame_selection", "autogaze_chop_selection", "autogaze_zero_mask", "post_encoder_pruning")
    direct_token_injection_supported = False
    siglip_based = "partial"
    positional_encoding_type = "hybrid_siglip_absolute_2d_plus_internvideo_temporal_plus_perceiver_latents"
    dense_grid_dependency = True
    variable_visual_token_support = False
    visual_placeholder_dynamic = "unknown"
    direct_sparse_autogaze_supported = False
    patch_grid_accessible = "partial"
    deterministic_position_adaptation = False
    projector_status = "blocked_for_direct_sparse: Perceiver connector emits fixed resampled tokens"
    placeholder_status = "blocked_until_connector_output_and_placeholders_are_verified"
    positional_encoding_status = "blocked_for_direct_sparse: hybrid tower and Perceiver resampler hide a simple SigLIP grid"
    token_count_compatibility = "blocked_for_direct_sparse: connector emits fixed resampled token counts"
    recommended_mode = "autogaze_chop_selection"
    recommended_first_smoke_mode = "autogaze_chop_selection"
    risk_level = "high"
    compatibility_status = "post_encoder_only"
    unsupported_mode_reasons = {
        "siglip_sparse_patch": "Apollo uses a hybrid SigLIP/InternVideo2 tower and Perceiver connector; isolated SigLIP sparse path is not verified",
        "rope_sparse_patch": "Apollo does not expose a verified explicit sparse spatial/temporal RoPE path before its hybrid connector",
        "direct_visual_token_injection": "fixed connector output and placeholders are not verified for arbitrary selected tokens",
    }


class VideoLLaMA3Adapter(ExternalMLLMAdapter):
    name = "videollama3"
    default_model_id = "DAMO-NLP-SG/VideoLLaMA3-7B"
    encoder_type = "VL3-SigLIP-NaViT patch14 tuned vision encoder"
    llm_type = "Qwen2.5-7B"
    supported_integration_modes = (
        "official_processor",
        "autogaze_frame_selection",
        "autogaze_chop_selection",
        "autogaze_zero_mask",
        "post_encoder_pruning",
    )
    direct_token_injection_supported = "unknown"
    siglip_based = True
    positional_encoding_type = "siglip_navit_positioning_plus_qwen2_5_rope"
    dense_grid_dependency = "unknown"
    variable_visual_token_support = "unknown"
    visual_placeholder_dynamic = "unknown"
    direct_sparse_autogaze_supported = "unknown"
    patch_grid_accessible = "unknown"
    deterministic_position_adaptation = "unknown"
    projector_status = "needs_code_inspection: spatial merge/compression and projector"
    placeholder_status = "needs_code_inspection: token compression and placeholder count"
    positional_encoding_status = "needs_code_inspection: NaViT layout, spatial merge, and token compression must be preserved"
    token_count_compatibility = "needs_code_inspection: mm_max_length/spatial_merge/compression controls token budget"
    recommended_mode = "autogaze_frame_selection"
    recommended_first_smoke_mode = "autogaze_frame_selection"
    risk_level = "high"
    compatibility_status = "siglip_candidate"
    unsupported_mode_reasons = {
        "siglip_sparse_patch": "candidate only; NaViT layout, spatial merge, token compression, and placeholders are not verified",
        "rope_sparse_patch": "VideoLLaMA3 sparse RoPE path is not verified; NaViT packing and compression may require dense layout",
        "direct_visual_token_injection": "VideoLLaMA3 projector, token compression, and placeholder alignment have not been verified",
    }


class VideoChatFlashAdapter(ExternalMLLMAdapter):
    name = "videochat_flash"
    default_model_id = "OpenGVLab/VideoChat-Flash-Qwen2-7B_res448"
    encoder_type = "UMT-L or UMT-HD-L video tower with hierarchical compression"
    llm_type = "Qwen2-7B"
    supported_integration_modes = ("official_processor", "autogaze_frame_selection", "autogaze_chop_selection", "autogaze_zero_mask")
    direct_token_injection_supported = False
    siglip_based = False
    positional_encoding_type = "umt_hierarchical_temporal_spatial_positions_plus_qwen_rope"
    dense_grid_dependency = True
    variable_visual_token_support = False
    visual_placeholder_dynamic = "unknown"
    direct_sparse_autogaze_supported = False
    patch_grid_accessible = False
    deterministic_position_adaptation = False
    projector_status = "blocked_for_direct_sparse: hierarchical compressor produces low fixed token counts"
    placeholder_status = "blocked_until_compressed_token_placeholders_are_verified"
    positional_encoding_status = "blocked_for_direct_sparse: UMT hierarchy and compression do not expose AutoGaze SigLIP grid"
    token_count_compatibility = "blocked_for_direct_sparse: model compresses to configured per-frame tokens"
    recommended_mode = "autogaze_frame_selection"
    recommended_first_smoke_mode = "autogaze_frame_selection"
    risk_level = "high"
    compatibility_status = "input_selection_only"
    unsupported_mode_reasons = {
        "siglip_sparse_patch": "VideoChat-Flash uses UMT-family encoders, not SigLIP",
        "rope_sparse_patch": "VideoChat-Flash does not expose a verified explicit sparse RoPE path before hierarchical compression",
        "post_encoder_pruning": "hierarchical compression already changes token layout; pruning semantics need code inspection",
        "direct_visual_token_injection": "hierarchical token compressor and placeholder contract are not verified",
    }


class InternVL35Adapter(ExternalMLLMAdapter):
    name = "internvl3_5"
    default_model_id = "OpenGVLab/InternVL3_5-8B"
    encoder_type = "InternViT with dynamic tiling and pixel shuffle compression"
    llm_type = "Qwen3-8B"
    supported_integration_modes = ("official_processor", "autogaze_frame_selection", "autogaze_chop_selection", "autogaze_zero_mask")
    direct_token_injection_supported = False
    siglip_based = False
    positional_encoding_type = "internvit_absolute_or_dynamic_tile_positions_plus_qwen3_rope"
    dense_grid_dependency = True
    variable_visual_token_support = "unknown"
    visual_placeholder_dynamic = "unknown"
    direct_sparse_autogaze_supported = False
    patch_grid_accessible = False
    deterministic_position_adaptation = False
    projector_status = "blocked_for_direct_sparse: pixel shuffle/downsample contract is dense-tile specific"
    placeholder_status = "blocked_until_dynamic_tile_placeholders_are_verified"
    positional_encoding_status = "blocked_for_direct_sparse: dynamic tiles and pixel shuffle change token layout"
    token_count_compatibility = "blocked_for_direct_sparse: image_seq_length/downsample contract is model-specific"
    recommended_mode = "autogaze_chop_selection"
    recommended_first_smoke_mode = "autogaze_chop_selection"
    risk_level = "high"
    compatibility_status = "input_selection_only"
    unsupported_mode_reasons = {
        "siglip_sparse_patch": "InternVL3.5 uses InternViT, not SigLIP",
        "rope_sparse_patch": "InternVL3.5 direct sparse RoPE path is not verified and dynamic tiling/pixel shuffle require dense tile layout",
        "post_encoder_pruning": "pixel-shuffle and dynamic patch routing must be mapped before pruning",
        "direct_visual_token_injection": "dynamic tile counts, pixel shuffle, and image placeholder expansion are not verified",
    }


class Qwen25VLAdapter(QwenAdapter):
    name = "qwen2_5_vl"
    default_model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    encoder_type = "Qwen2.5 native dynamic-resolution ViT"
    llm_type = "Qwen2.5-7B"
    supported_integration_modes: tuple[str, ...] = ("official_processor", "autogaze_frame_selection", "autogaze_chop_selection", "autogaze_zero_mask", "post_encoder_pruning")
    direct_token_injection_supported: bool | str = False
    native_sparse_patch_supported: bool | str = False
    light_modified_sparse_supported: bool | str = False
    rope_sparse_patch_supported: bool | str = False
    autogaze_zero_mask_supported: bool | str = True
    post_encoder_zero_mask_supported: bool | str = False
    input_selection_only_supported: bool | str = True
    siglip_based: bool | str = False
    positional_encoding_type = "qwen2_5_vl_mrope_window_attention"
    dense_grid_dependency: bool | str = True
    variable_visual_token_support: bool | str = "processor_dynamic_only"
    visual_placeholder_dynamic: bool | str = "processor_dynamic_only"
    direct_sparse_autogaze_supported: bool | str = False
    patch_grid_accessible: bool | str = "processor_grid_thw_only"
    deterministic_position_adaptation: bool | str = False
    projector_status = "blocked_for_direct_injection: visual merger expects official grid_thw path"
    placeholder_status = "blocked_for_direct_injection: placeholders are processor-generated"
    positional_encoding_status = "blocked_for_direct_injection: M-RoPE, temporal IDs, and window attention depend on official grid_thw"
    token_count_compatibility = "blocked_for_direct_injection: visual merger and placeholder counts are processor-generated"
    requires_official_processor = True
    requires_training_or_trainable_adapter: bool | str = False
    recommended_mode = "autogaze_frame_selection"
    recommended_first_smoke_mode = "autogaze_frame_selection"
    risk_level = "high"
    compatibility_status = "input_selection_only"
    local_asset_status = "local_exists_if_weights/Qwen2.5-VL-7B-Instruct_is_present"
    official_processor_support_status = "implemented_when_local_assets_and_transformers_are_available"
    input_selection_support_status = "supported_as_input_level_selection_only"
    zero_mask_support_status = "supported_as_image_space_or_patch_embed_probe_only_no_encoder_acceleration"
    unsupported_mode_reasons = {
        "siglip_sparse_patch": "Qwen2.5-VL does not use SigLIP",
        "rope_sparse_patch": "Qwen2.5-VL M-RoPE/grid_thw/window attention sparse path is not verified for holes",
        "direct_visual_token_injection": "M-RoPE position IDs, attention masks, visual merger, and placeholder counts are not verified for selected-token injection",
    }

    def load(self, *, allow_real_model_loading: bool = False, device: str = "cpu", dtype: str = "float32") -> AdapterStatus:
        if not allow_real_model_loading:
            self.status = AdapterStatus(
                self.name,
                "stub-only",
                "real Qwen2.5-VL loading disabled",
                metadata=self._adapter_report(integration_mode="load", generation_ran=False),
            )
            return self.status
        return super().load(allow_real_model_loading=allow_real_model_loading, device=device, dtype=dtype)

    def supports_direct_visual_tokens(self) -> bool:
        return False

    def supports_native_sparse_patch(self) -> bool:
        return False

    def supports_light_modified_sparse(self) -> bool:
        return False

    def supports_rope_sparse_patch(self) -> bool:
        return False

    def supports_direct_visual_token_injection(self) -> bool:
        return False

    def supports_input_level_selection(self) -> bool:
        return True

    def supports_autogaze_zero_mask(self) -> bool:
        return True

    def supports_post_encoder_zero_mask(self) -> bool:
        return False

    def supports_post_encoder_pruning(self) -> bool:
        return True

    def run_official_processor_path(self, **kwargs: Any) -> dict[str, Any]:
        if self.model is not None and self.processor is not None:
            return self.generate(**kwargs)
        return self._mode_status("official_processor", **kwargs)

    def run_autogaze_frame_selection_path(self, **_: Any) -> dict[str, Any]:
        if self.model is not None and self.processor is not None:
            result = self.generate(**_)
            metadata = dict(result.get("metadata") or {})
            metadata["integration_mode"] = "autogaze_frame_selection"
            metadata["autogaze_input_selection_mode"] = "autogaze_frame_selection"
            metadata["autogaze_visual_tokens_injected"] = False
            result["metadata"] = metadata
            return result
        return self._mode_status("autogaze_frame_selection", **_)

    def run_autogaze_chop_selection_path(self, **_: Any) -> dict[str, Any]:
        if self.model is not None and self.processor is not None:
            result = self.generate(**_)
            metadata = dict(result.get("metadata") or {})
            metadata["integration_mode"] = "autogaze_chop_selection"
            metadata["autogaze_input_selection_mode"] = "autogaze_chop_selection"
            metadata["autogaze_visual_tokens_injected"] = False
            result["metadata"] = metadata
            return result
        return self._mode_status("autogaze_chop_selection", **_)

    def run_autogaze_zero_mask_path(self, **_: Any) -> dict[str, Any]:
        if self.model is not None and self.processor is not None:
            result = self.generate(**_)
            metadata = dict(result.get("metadata") or {})
            metadata["integration_mode"] = "autogaze_zero_mask"
            metadata["autogaze_zero_mask"] = True
            metadata["autogaze_visual_tokens_injected"] = False
            metadata["qwen_encoder_side_acceleration_claimed"] = False
            result["metadata"] = metadata
            return result
        return self._mode_status("autogaze_zero_mask", **_)

    def run_siglip_sparse_patch_path(self, **_: Any) -> dict[str, Any]:
        return self._mode_status("siglip_sparse_patch", **_)

    def run_native_sparse_patch_path(self, **_: Any) -> dict[str, Any]:
        return self._mode_status("native_sparse_patch", **_)

    def run_light_modified_sparse_path(self, **_: Any) -> dict[str, Any]:
        return self._mode_status("light_modified_sparse", **_)

    def run_rope_sparse_patch_path(self, **_: Any) -> dict[str, Any]:
        return self._mode_status("rope_sparse_patch", **_)

    def run_post_encoder_pruning_path(self, **_: Any) -> dict[str, Any]:
        return self._mode_status("post_encoder_pruning", **_)

    def run_post_encoder_zero_mask_path(self, **_: Any) -> dict[str, Any]:
        return self._mode_status("post_encoder_zero_mask", **_)

    def run_direct_visual_token_injection_path(self, **_: Any) -> dict[str, Any]:
        return self._mode_status("direct_visual_token_injection", **_)

    def _mode_status(self, integration_mode: str, **kwargs: Any) -> dict[str, Any]:
        if integration_mode not in self.supported_integration_modes or integration_mode == "direct_visual_token_injection":
            reason = self.unsupported_mode_reasons.get(
                integration_mode,
                f"{integration_mode} is not verified for {self.name}",
            )
            raise NotImplementedError(self._unsupported_message(integration_mode, reason))
        if self.status.status == "dummy":
            return self._dummy_generation(
                query_text=str(kwargs.get("query_text") or ""),
                max_new_tokens=int(kwargs.get("max_new_tokens") or 32),
                integration_mode=integration_mode,
            )
        return {
            "status": "stub-only",
            "answer": None,
            "reason": (
                f"{self.name} {integration_mode} is registered. "
                "Use real Qwen official processor loading for generation; input-level AutoGaze selection is not yet implemented here."
            ),
            "query_text_used": True,
            "metadata": self._adapter_report(integration_mode=integration_mode, generation_ran=False),
        }

    def _unsupported_message(self, integration_mode: str, reason: str) -> str:
        return (
            f"{self.name} does not support {integration_mode}: {reason}. "
            "No fallback to NVILA, modified SigLIP, or trainable projection adapter is allowed."
        )

    def _adapter_report(
        self,
        *,
        integration_mode: str,
        generation_ran: bool,
        unsupported_reason: str | None = None,
    ) -> dict[str, Any]:
        requested_model = self.config.get("model_id") or self.config.get("checkpoint_path") or self.default_model_id
        return {
            "requested_model": requested_model,
            "actual_model_loaded": requested_model if self.status.status == "real" else None,
            "adapter": self.name,
            "integration_mode": integration_mode,
            "real_checkpoint_loaded": self.status.status == "real",
            "generation_ran": generation_ran,
            "unsupported_reason": unsupported_reason,
            "encoder_type": self.encoder_type,
            "llm_type": self.llm_type,
            "supported_integration_modes": list(self.supported_integration_modes),
            "direct_token_injection_supported": self.direct_token_injection_supported,
            "native_sparse_patch_supported": self.native_sparse_patch_supported,
            "light_modified_sparse_supported": self.light_modified_sparse_supported,
            "rope_sparse_patch_supported": self.rope_sparse_patch_supported,
            "autogaze_zero_mask_supported": self.autogaze_zero_mask_supported,
            "post_encoder_zero_mask_supported": self.post_encoder_zero_mask_supported,
            "input_selection_only_supported": self.input_selection_only_supported,
            "siglip_based": self.siglip_based,
            "positional_encoding_compatibility": self.positional_encoding_status,
            "positional_encoding_type": self.positional_encoding_type,
            "dense_grid_dependency": self.dense_grid_dependency,
            "variable_visual_token_support": self.variable_visual_token_support,
            "visual_placeholder_dynamic": self.visual_placeholder_dynamic,
            "direct_sparse_autogaze_supported": self.direct_sparse_autogaze_supported,
            "patch_grid_accessible": self.patch_grid_accessible,
            "deterministic_position_adaptation": self.deterministic_position_adaptation,
            "projector_status": self.projector_status,
            "placeholder_status": self.placeholder_status,
            "token_count_compatibility": self.token_count_compatibility,
            "requires_official_processor": self.requires_official_processor,
            "requires_training_or_trainable_adapter": self.requires_training_or_trainable_adapter,
            "recommended_mode": self.recommended_mode,
            "recommended_first_smoke_mode": self.recommended_first_smoke_mode,
            "risk_level": self.risk_level,
            "compatibility_status": self.compatibility_status,
            "local_asset_status": self.local_asset_status,
            "official_processor_support_status": self.official_processor_support_status,
            "input_selection_support_status": self.input_selection_support_status,
            "zero_mask_support_status": self.zero_mask_support_status,
        }


def _qwen_patch_mask_from_autogaze(
    autogaze: Mapping[str, Any],
    *,
    qwen_t: int,
    qwen_h: int,
    qwen_w: int,
    min_keep_tokens: int,
    empty_chunk_policy: str,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    per_frame = list(autogaze.get("per_frame") or [])
    if not per_frame:
        return None, {"reason": "Qwen AutoGaze mask cannot be built because per-frame AutoGaze records are empty"}
    mask = torch.zeros((qwen_t, qwen_h, qwen_w), dtype=torch.float32)
    empty_chunks = 0
    boxes_seen = 0
    chunks_with_boxes = 0
    frame_count = len(per_frame)
    min_keep = max(0, int(min_keep_tokens))
    policy = empty_chunk_policy or "keep_center"

    for chunk_idx in range(qwen_t):
        start = int(math.floor(chunk_idx * frame_count / float(qwen_t)))
        end = int(math.ceil((chunk_idx + 1) * frame_count / float(qwen_t)))
        start = max(0, min(frame_count - 1, start))
        end = max(start + 1, min(frame_count, end))
        chunk_mask = mask[chunk_idx]
        chunk_boxes = 0
        for frame_record in per_frame[start:end]:
            for patch_record in list(frame_record.get("selected_patch_records") or []):
                box = patch_record.get("normalized_box") if isinstance(patch_record, Mapping) else None
                if not isinstance(box, (list, tuple)) or len(box) != 4:
                    continue
                if _mark_normalized_box_on_qwen_grid(chunk_mask, box):
                    boxes_seen += 1
                    chunk_boxes += 1
        if chunk_boxes > 0:
            chunks_with_boxes += 1
        if int(chunk_mask.sum().item()) == 0:
            empty_chunks += 1
            if policy == "block":
                return None, {
                    "reason": f"Qwen AutoGaze mask produced an empty temporal chunk at index {chunk_idx}",
                    "qwen_autogaze_empty_chunk_policy": policy,
                }
            if policy == "keep_all":
                chunk_mask.fill_(1.0)
            elif policy == "keep_none":
                pass
            else:
                _mark_center_tokens(chunk_mask, max(1, min_keep))
        if min_keep > 0 and int(chunk_mask.sum().item()) < min_keep:
            _mark_center_tokens(chunk_mask, min_keep)

    flat_mask = mask.reshape(-1)
    selected = int(flat_mask.sum().item())
    original = int(flat_mask.numel())
    if selected <= 0:
        return None, {"reason": "Qwen AutoGaze mask selected zero Qwen visual patches"}
    return flat_mask, {
        "qwen_visual_mask_grid_thw": [qwen_t, qwen_h, qwen_w],
        "qwen_visual_tokens_before": original,
        "qwen_visual_tokens_kept_by_mask": selected,
        "qwen_visual_mask_keep_ratio": selected / float(original),
        "qwen_autogaze_frame_records": frame_count,
        "qwen_autogaze_boxes_mapped": boxes_seen,
        "qwen_autogaze_temporal_chunks_with_boxes": chunks_with_boxes,
        "qwen_autogaze_empty_temporal_chunks": empty_chunks,
        "qwen_autogaze_empty_chunk_policy": policy,
        "qwen_autogaze_min_keep_tokens": min_keep,
        "qwen_visual_mask_source": "autogaze_selected_patch_records_normalized_box_overlap",
    }


def _mark_normalized_box_on_qwen_grid(mask: torch.Tensor, box: Any) -> bool:
    x0, y0, x1, y1 = _normalized_box(box)
    if x1 <= x0 or y1 <= y0:
        return False
    height, width = int(mask.shape[0]), int(mask.shape[1])
    col0 = max(0, min(width - 1, int(math.floor(x0 * width))))
    col1 = max(col0 + 1, min(width, int(math.ceil(x1 * width))))
    row0 = max(0, min(height - 1, int(math.floor(y0 * height))))
    row1 = max(row0 + 1, min(height, int(math.ceil(y1 * height))))
    mask[row0:row1, col0:col1] = 1.0
    return True


def _normalized_box(box: Any) -> tuple[float, float, float, float]:
    values = [float(item) for item in box]
    x0, y0, x1, y1 = values
    x0 = max(0.0, min(1.0, x0))
    y0 = max(0.0, min(1.0, y0))
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def _mark_center_tokens(mask: torch.Tensor, count: int) -> None:
    height, width = int(mask.shape[0]), int(mask.shape[1])
    center_y = (height - 1) / 2.0
    center_x = (width - 1) / 2.0
    candidates: list[tuple[float, int, int]] = []
    for row in range(height):
        for col in range(width):
            if float(mask[row, col].item()) > 0:
                continue
            dist = (row - center_y) ** 2 + (col - center_x) ** 2
            candidates.append((dist, row, col))
    for _dist, row, col in sorted(candidates)[: max(0, int(count) - int(mask.sum().item()))]:
        mask[row, col] = 1.0


def _looks_like_local_path(value: str) -> bool:
    if value.startswith((".", "/", "~")):
        return True
    return value.startswith("weights/") or value.startswith("checkpoints/") or value.endswith(".pt") or value.endswith(".safetensors")


def _resolve_local_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _model_reference_for_loading(value: Any) -> str:
    text = str(value)
    if _looks_like_local_path(text):
        return str(_resolve_local_path(text))
    return text


def _missing_sharded_checkpoint_files(path: Path) -> list[str]:
    index_path = path / "model.safetensors.index.json" if path.is_dir() else None
    if index_path is None or not index_path.exists():
        return []
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    weight_map = data.get("weight_map")
    if not isinstance(weight_map, Mapping):
        return []
    expected = sorted({str(name) for name in weight_map.values()})
    return [name for name in expected if not (path / name).exists()]


def _from_pretrained_kwargs(config: Mapping[str, Any], *, dtype: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if "trust_remote_code" in config:
        kwargs["trust_remote_code"] = bool(config.get("trust_remote_code"))
    if "local_files_only" in config:
        kwargs["local_files_only"] = bool(config.get("local_files_only"))
    model_dtype = _torch_dtype(dtype)
    if model_dtype is not None:
        kwargs["dtype"] = model_dtype
    for key, value in dict(config.get("from_pretrained_kwargs") or {}).items():
        if key == "torch_dtype":
            key = "dtype"
            value = _dtype_from_config_value(value)
        kwargs[key] = value
    return kwargs


def _processor_from_pretrained_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if "trust_remote_code" in config:
        kwargs["trust_remote_code"] = bool(config.get("trust_remote_code"))
    if "local_files_only" in config:
        kwargs["local_files_only"] = bool(config.get("local_files_only"))
    for key, value in dict(config.get("processor_from_pretrained_kwargs") or {}).items():
        kwargs[key] = value
    if "use_fast" not in kwargs:
        kwargs["use_fast"] = False
    return kwargs


def _nvila_processor_from_pretrained_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    kwargs = _processor_from_pretrained_kwargs(config)
    autogaze_model_id = (
        kwargs.get("autogaze_model_id")
        or config.get("autogaze_model_id")
        or config.get("poc_autogaze_processor_path")
        or config.get("poc_autogaze_checkpoint_path")
    )
    if autogaze_model_id is None and _resolve_local_path("weights/AutoGaze").exists():
        autogaze_model_id = "weights/AutoGaze"
    if autogaze_model_id is not None:
        kwargs["autogaze_model_id"] = _model_reference_for_loading(autogaze_model_id)
    max_num_frames = _local_autogaze_max_num_frames(autogaze_model_id)
    if max_num_frames is not None:
        _set_or_validate_nvila_frame_counts(kwargs, max_num_frames=max_num_frames)
    if not bool(config.get("sync_autogaze_controls_from_config", False)):
        return kwargs

    autogaze_enabled = config.get("poc_autogaze_enabled")
    if autogaze_enabled is True:
        gaze_ratio = config.get("poc_gaze_ratio")
        task_loss_requirement = config.get("poc_task_loss_requirement")
        if gaze_ratio is not None:
            kwargs.setdefault("gazing_ratio_tile", gaze_ratio)
            kwargs.setdefault("gazing_ratio_thumbnail", gaze_ratio)
        if task_loss_requirement is not None:
            kwargs.setdefault("task_loss_requirement_tile", task_loss_requirement)
            kwargs.setdefault("task_loss_requirement_thumbnail", task_loss_requirement)
    elif autogaze_enabled is False:
        kwargs.setdefault("gazing_ratio_tile", None)
        kwargs.setdefault("gazing_ratio_thumbnail", None)
        kwargs.setdefault("task_loss_requirement_tile", None)
        kwargs.setdefault("task_loss_requirement_thumbnail", None)
    return kwargs


def _local_autogaze_max_num_frames(autogaze_model_id: Any) -> int | None:
    if autogaze_model_id in {None, ""}:
        return None
    text = str(autogaze_model_id)
    if not _looks_like_local_path(text):
        return None
    path = _resolve_local_path(text)
    config_path = path / "config.json" if path.is_dir() else path
    if not config_path.exists():
        return None
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    value = data.get("max_num_frames")
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _set_or_validate_nvila_frame_counts(kwargs: dict[str, Any], *, max_num_frames: int) -> None:
    value = kwargs.get("num_video_frames")
    if value is None:
        kwargs["num_video_frames"] = max_num_frames
    else:
        num_video_frames = int(value)
        if num_video_frames <= 0 or num_video_frames % max_num_frames != 0:
            raise ValueError(
                "NVILA processor num_video_frames must be a positive multiple of "
                f"AutoGaze max_num_frames ({max_num_frames}); got {num_video_frames}"
            )
        kwargs["num_video_frames"] = num_video_frames

    thumbnail_value = kwargs.get("num_video_frames_thumbnail")
    if thumbnail_value is None:
        kwargs["num_video_frames_thumbnail"] = max_num_frames
    else:
        num_thumbnail_frames = int(thumbnail_value)
        if num_thumbnail_frames <= 0:
            raise ValueError("NVILA processor num_video_frames_thumbnail must be positive")
        kwargs["num_video_frames_thumbnail"] = num_thumbnail_frames


def _positive_int_or_none(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _jsonable_processor_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in kwargs.items():
        if isinstance(value, torch.dtype):
            result[key] = str(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            result[key] = value
        elif isinstance(value, (list, tuple)):
            result[key] = list(value)
        else:
            result[key] = str(value)
    return result


def _torch_dtype(dtype: str) -> torch.dtype | None:
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    return None


def _dtype_from_config_value(value: Any) -> Any:
    if isinstance(value, torch.dtype):
        return value
    text = str(value)
    if text.startswith("torch."):
        text = text.removeprefix("torch.")
    parsed = _torch_dtype(text)
    return parsed if parsed is not None else value


def _official_video_input(
    *,
    video: torch.Tensor,
    video_path: str | None,
    target_frame_count: int | None = None,
    batch_pil_frames: bool = False,
) -> tuple[Any, str]:
    if video_path and video_path != "dummy":
        path = Path(video_path).expanduser()
        if path.exists():
            return str(path if path.is_absolute() else path.resolve()), "video_path"
        repo_path = REPO_ROOT / path
        if repo_path.exists():
            return str(repo_path), "video_path"
        return video_path, "video_reference"
    frames = _video_tensor_to_pil(video, target_frame_count=target_frame_count)
    suffix = f"_to_{len(frames)}" if target_frame_count is not None else ""
    if batch_pil_frames:
        return [frames], f"processed_tensor_pil_video{suffix}"
    return frames, f"processed_tensor_pil_frames{suffix}"


def _qwen_vl_utils_available() -> bool:
    return importlib.util.find_spec("qwen_vl_utils") is not None


def _qwen_message_video_reference(video_input: Any, video_input_kind: str) -> Any:
    if video_input_kind in {"video_path", "video_reference"}:
        return video_input
    return "processed_tensor_frames"


def _qwen_processor_video_payload(video_input: Any) -> Any:
    if isinstance(video_input, list):
        return [video_input]
    return [video_input]


def _move_tensor_mapping(value: Mapping[str, Any], device: str) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, item in value.items():
        moved[key] = item.to(device) if isinstance(item, torch.Tensor) else item
    return moved


def _video_tensor_to_pil(video: torch.Tensor, *, target_frame_count: int | None = None) -> list[Any]:
    from PIL import Image

    if video.ndim != 5:
        raise ValueError(f"expected video shape [B,T,C,H,W], got {tuple(video.shape)}")
    batch, frame_count = int(video.shape[0]), int(video.shape[1])
    frames = video.detach().cpu().float().clamp(0, 1)
    frames = frames.reshape(batch * frame_count, *frames.shape[2:]) if batch > 1 else frames[0]
    if target_frame_count is not None:
        target = int(target_frame_count)
        if target <= 0:
            raise ValueError("target_frame_count must be positive")
        if batch > 1 and int(frames.shape[0]) > target:
            remainder = int(frames.shape[0]) % target
            if remainder:
                pad = frames[-1:].expand(target - remainder, -1, -1, -1)
                frames = torch.cat([frames, pad], dim=0)
        elif int(frames.shape[0]) > target:
            if target == 1:
                indices = torch.tensor([0], dtype=torch.long)
            else:
                indices = torch.linspace(0, int(frames.shape[0]) - 1, steps=target).round().to(torch.long)
            frames = frames.index_select(0, indices)
        elif int(frames.shape[0]) < target:
            if int(frames.shape[0]) == 0:
                raise ValueError("cannot pad an empty video tensor")
            pad = frames[-1:].expand(target - int(frames.shape[0]), -1, -1, -1)
            frames = torch.cat([frames, pad], dim=0)
    images = []
    for frame in frames:
        array = (frame.permute(1, 2, 0).numpy() * 255).astype("uint8")
        images.append(Image.fromarray(array, mode="RGB"))
    return images
