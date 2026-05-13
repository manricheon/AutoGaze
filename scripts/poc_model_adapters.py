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

    def preprocess_or_accept_features(self, video: torch.Tensor, autogaze: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return {"video": video, "autogaze": autogaze}

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


class GenericVitAdapter(VisionEncoderAdapter):
    name = "generic_vit"


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

    def count_visual_tokens(self, visual_tokens: torch.Tensor | None) -> int | None:
        if visual_tokens is None:
            return None
        if visual_tokens.ndim < 2:
            return None
        return int(visual_tokens.shape[-2])

    def supports_direct_visual_tokens(self) -> bool:
        return False

    def supports_official_processor_path(self) -> bool:
        return False

    def default_module_path(self) -> str | None:
        return None

    def default_class_name(self) -> str | None:
        return None

    def _checkpoint(self) -> Any:
        return self.config.get("checkpoint_path") or self.config.get("model_id")


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
            target_frame_count = getattr(self.processor, "num_video_frames", None)
            video_input, video_input_kind = _official_video_input(
                video=video,
                video_path=video_path,
                target_frame_count=target_frame_count,
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
                    "processor_autogaze_controls": _jsonable_processor_kwargs(_nvila_processor_from_pretrained_kwargs(self.config)),
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
            inputs, input_metadata = self._prepare_official_qwen_inputs(
                query_text=query_text,
                video=video,
                video_path=None if use_autogaze_mask else video_path,
                prefer_processed_tensor=use_autogaze_mask,
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
