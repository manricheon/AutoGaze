#!/usr/bin/env python3
from __future__ import annotations

from typing import Mapping

from poc_model_adapters import (
    ApolloAdapter,
    ExternalMLLMAdapter,
    ExternalVisionEncoderAdapter,
    GenericMLLMAdapter,
    GenericVitAdapter,
    InternVL35Adapter,
    LlavaOVAdapter,
    LongVAAdapter,
    LongVILAAdapter,
    MLLMAdapter,
    ModifiedSiglipAdapter,
    NVILAAdapter,
    QwenAdapter,
    Qwen25VLAdapter,
    VJEPA2Adapter,
    VanillaSiglipAdapter,
    VideoChatFlashAdapter,
    VideoLLaMA3Adapter,
    VisionEncoderAdapter,
)


VISION_ENCODERS: dict[str, type[VisionEncoderAdapter]] = {
    "modified_siglip": ModifiedSiglipAdapter,
    "vanilla_siglip": VanillaSiglipAdapter,
    "vjepa2": VJEPA2Adapter,
    "generic_vit": GenericVitAdapter,
    "external": ExternalVisionEncoderAdapter,
}

MLLMS: dict[str, type[MLLMAdapter]] = {
    "nvila": NVILAAdapter,
    "qwen": QwenAdapter,
    "generic_mllm": GenericMLLMAdapter,
    "llava_ov": LlavaOVAdapter,
    "longva": LongVAAdapter,
    "longvila_r1": LongVILAAdapter,
    "apollo": ApolloAdapter,
    "videollama3": VideoLLaMA3Adapter,
    "videochat_flash": VideoChatFlashAdapter,
    "internvl3_5": InternVL35Adapter,
    "qwen2_5_vl": Qwen25VLAdapter,
}


def _external_metadata(cls: type[MLLMAdapter], *, status: str) -> dict[str, object]:
    if not issubclass(cls, ExternalMLLMAdapter) and cls is not Qwen25VLAdapter:
        raise TypeError(f"{cls.__name__} is not an external MLLM metadata adapter")
    return {
        "model_name": cls.name,
        "default_model_id": getattr(cls, "default_model_id", ""),
        "adapter_class": cls.__name__,
        "vision_encoder_type": getattr(cls, "encoder_type", "unknown"),
        "supported_integration_modes": list(getattr(cls, "supported_integration_modes", ())),
        "direct_token_injection_supported": getattr(cls, "direct_token_injection_supported", "unknown"),
        "native_sparse_patch_support": getattr(cls, "native_sparse_patch_supported", False),
        "light_modified_sparse_support": getattr(cls, "light_modified_sparse_supported", False),
        "rope_sparse_patch_support": getattr(cls, "rope_sparse_patch_supported", False),
        "autogaze_zero_mask_support": getattr(cls, "autogaze_zero_mask_supported", "unknown"),
        "post_encoder_zero_mask_support": getattr(cls, "post_encoder_zero_mask_supported", "unknown"),
        "input_selection_only_support": getattr(cls, "input_selection_only_supported", "unknown"),
        "siglip_based": getattr(cls, "siglip_based", "unknown"),
        "positional_encoding_type": getattr(cls, "positional_encoding_type", "unknown"),
        "positional_encoding_status": getattr(cls, "positional_encoding_status", "needs_code_inspection"),
        "dense_grid_dependency": getattr(cls, "dense_grid_dependency", "unknown"),
        "variable_visual_token_support": getattr(cls, "variable_visual_token_support", "unknown"),
        "visual_placeholder_dynamic": getattr(cls, "visual_placeholder_dynamic", "unknown"),
        "direct_sparse_autogaze_supported": getattr(cls, "direct_sparse_autogaze_supported", "unknown"),
        "requires_official_processor": getattr(cls, "requires_official_processor", True),
        "requires_training_or_trainable_adapter": getattr(cls, "requires_training_or_trainable_adapter", "unknown"),
        "recommended_mode": getattr(cls, "recommended_mode", "autogaze_frame_selection"),
        "recommended_first_smoke_mode": getattr(cls, "recommended_first_smoke_mode", getattr(cls, "recommended_mode", "autogaze_frame_selection")),
        "recommended_modes": [getattr(cls, "recommended_mode", "autogaze_frame_selection")],
        "local_asset_status": getattr(cls, "local_asset_status", "not_checked"),
        "official_processor_support_status": getattr(cls, "official_processor_support_status", "stub_until_assets_verified"),
        "input_selection_support_status": getattr(cls, "input_selection_support_status", "supported_as_input_level_selection_only"),
        "zero_mask_support_status": getattr(cls, "zero_mask_support_status", "supported_as_image_space_probe_only"),
        "risk_level": getattr(cls, "risk_level", "high"),
        "compatibility_status": getattr(cls, "compatibility_status", "needs_code_inspection"),
        "status": status,
    }


def _mllm_metadata(
    cls: type[MLLMAdapter],
    *,
    default_model_id: str,
    supported_integration_modes: list[str],
    direct_token_injection_supported: bool | str,
    vision_encoder_type: str,
    siglip_based: bool | str,
    positional_encoding_type: str,
    positional_encoding_status: str,
    dense_grid_dependency: bool | str,
    variable_visual_token_support: bool | str,
    visual_placeholder_dynamic: bool | str,
    direct_sparse_autogaze_supported: bool | str,
    requires_official_processor: bool,
    requires_training_or_trainable_adapter: bool | str,
    recommended_mode: str,
    risk_level: str,
    compatibility_status: str,
    status: str,
) -> dict[str, object]:
    return {
        "model_name": cls.name,
        "default_model_id": default_model_id,
        "adapter_class": cls.__name__,
        "vision_encoder_type": vision_encoder_type,
        "supported_integration_modes": supported_integration_modes,
        "direct_token_injection_supported": direct_token_injection_supported,
        "native_sparse_patch_support": False,
        "light_modified_sparse_support": False,
        "rope_sparse_patch_support": False,
        "autogaze_zero_mask_support": "not_applicable",
        "post_encoder_zero_mask_support": "not_applicable",
        "input_selection_only_support": "not_applicable",
        "siglip_based": siglip_based,
        "positional_encoding_type": positional_encoding_type,
        "positional_encoding_status": positional_encoding_status,
        "dense_grid_dependency": dense_grid_dependency,
        "variable_visual_token_support": variable_visual_token_support,
        "visual_placeholder_dynamic": visual_placeholder_dynamic,
        "direct_sparse_autogaze_supported": direct_sparse_autogaze_supported,
        "requires_official_processor": requires_official_processor,
        "requires_training_or_trainable_adapter": requires_training_or_trainable_adapter,
        "recommended_mode": recommended_mode,
        "recommended_first_smoke_mode": recommended_mode,
        "recommended_modes": [recommended_mode],
        "local_asset_status": "not_checked",
        "official_processor_support_status": "processor_managed" if requires_official_processor else "not_required",
        "input_selection_support_status": "not_applicable",
        "zero_mask_support_status": "not_applicable",
        "risk_level": risk_level,
        "compatibility_status": compatibility_status,
        "status": status,
    }


MLLM_REGISTRY_METADATA: dict[str, dict[str, object]] = {
    "nvila": _mllm_metadata(
        NVILAAdapter,
        default_model_id="weights/NVILA-8B-HD-Video",
        supported_integration_modes=["official_processor"],
        direct_token_injection_supported=False,
        vision_encoder_type="NVILA/VILA SigLIP-family official processor path",
        siglip_based=True,
        positional_encoding_type="canonical_nvila_processor_managed",
        positional_encoding_status="implemented_canonical_processor_path; direct injection not claimed",
        dense_grid_dependency="processor_managed",
        variable_visual_token_support="processor_managed",
        visual_placeholder_dynamic="processor_managed",
        direct_sparse_autogaze_supported=False,
        requires_official_processor=True,
        requires_training_or_trainable_adapter=False,
        recommended_mode="official_processor",
        risk_level="low",
        compatibility_status="native_candidate",
        status="implemented",
    ),
    "qwen": _mllm_metadata(
        QwenAdapter,
        default_model_id="weights/Qwen2.5-VL-7B-Instruct",
        supported_integration_modes=["official_processor", "post_encoder_pruning"],
        direct_token_injection_supported=False,
        vision_encoder_type="Qwen2.5 native dynamic-resolution ViT",
        siglip_based=False,
        positional_encoding_type="mrope_window_attention",
        positional_encoding_status="blocked_for_direct_injection: M-RoPE/grid_thw required",
        dense_grid_dependency=True,
        variable_visual_token_support="processor_dynamic_only",
        visual_placeholder_dynamic="processor_dynamic_only",
        direct_sparse_autogaze_supported=False,
        requires_official_processor=True,
        requires_training_or_trainable_adapter=False,
        recommended_mode="official_processor",
        risk_level="high",
        compatibility_status="input_selection_only",
        status="implemented",
    ),
    "generic_mllm": _mllm_metadata(
        GenericMLLMAdapter,
        default_model_id="",
        supported_integration_modes=[],
        direct_token_injection_supported=False,
        vision_encoder_type="unknown",
        siglip_based="unknown",
        positional_encoding_type="unknown",
        positional_encoding_status="not_applicable",
        dense_grid_dependency="unknown",
        variable_visual_token_support="unknown",
        visual_placeholder_dynamic="unknown",
        direct_sparse_autogaze_supported=False,
        requires_official_processor=False,
        requires_training_or_trainable_adapter="unknown",
        recommended_mode="stub_status_only",
        risk_level="high",
        compatibility_status="unsupported_for_now",
        status="stub-only",
    ),
    "llava_ov": _external_metadata(LlavaOVAdapter, status="stub-only"),
    "longva": _external_metadata(LongVAAdapter, status="stub-only"),
    "longvila_r1": _external_metadata(LongVILAAdapter, status="stub-only"),
    "apollo": _external_metadata(ApolloAdapter, status="stub-only"),
    "videollama3": _external_metadata(VideoLLaMA3Adapter, status="stub-only"),
    "videochat_flash": _external_metadata(VideoChatFlashAdapter, status="stub-only"),
    "internvl3_5": _external_metadata(InternVL35Adapter, status="stub-only"),
    "qwen2_5_vl": _external_metadata(Qwen25VLAdapter, status="stub-only"),
}


VISION_ENCODER_REGISTRY_METADATA: dict[str, dict[str, object]] = {
    "modified_siglip": {
        "model_name": "modified_siglip",
        "default_model_id": "weights/siglip2-base-patch16-224",
        "adapter_class": "ModifiedSiglipAdapter",
        "supported_integration_modes": ["siglip_sparse_patch"],
        "direct_token_injection_supported": False,
        "siglip_based": True,
        "positional_encoding_status": "implemented AutoGaze multi-scale position selection",
        "requires_official_processor": False,
        "requires_training_or_trainable_adapter": False,
        "recommended_mode": "siglip_sparse_patch",
        "risk_level": "low",
        "status": "implemented",
    },
    "vanilla_siglip": {
        "model_name": "vanilla_siglip",
        "default_model_id": "weights/siglip2-base-patch16-224",
        "adapter_class": "VanillaSiglipAdapter",
        "supported_integration_modes": ["official_processor"],
        "direct_token_injection_supported": False,
        "siglip_based": True,
        "positional_encoding_status": "vanilla dense grid; AutoGaze sparse path disabled by default",
        "requires_official_processor": False,
        "requires_training_or_trainable_adapter": False,
        "recommended_mode": "official_processor",
        "risk_level": "medium",
        "status": "implemented",
    },
    "vjepa2": {
        "model_name": "vjepa2",
        "default_model_id": "weights/vjepa2-vitl-fpc64-256",
        "adapter_class": "VJEPA2Adapter",
        "model_type": "video_encoder",
        "supported_integration_modes": [
            "vjepa2_official_dense",
            "vjepa2_video_classification",
            "vjepa2_feature_extraction",
            "autogaze_frame_selection_vjepa2",
            "autogaze_chop_selection_vjepa2",
            "autogaze_zero_mask_vjepa2",
            "vjepa2_context_mask_probe",
            "vjepa2_sparse_tubelet",
        ],
        "direct_token_injection_supported": False,
        "siglip_based": False,
        "positional_encoding_type": "3d_rope",
        "positional_encoding_status": "needs_source_inspection: AutoGaze patch IDs are not mapped to V-JEPA2 tubelet positions",
        "patch_structure": "tubelet",
        "patch_size": 16,
        "crop_size": 256,
        "frames_per_clip": 64,
        "tubelet_size": 2,
        "supports_official_dense": True,
        "supports_autogaze_frame_selection": True,
        "supports_autogaze_chop_selection": True,
        "supports_autogaze_zero_mask": True,
        "autogaze_zero_mask_support": True,
        "native_sparse_patch_support": False,
        "light_modified_sparse_support": False,
        "rope_sparse_patch_support": "needs_source_inspection",
        "post_encoder_zero_mask_support": "not_applicable",
        "input_selection_only_support": True,
        "local_asset_status": "local_exists_if_weights/vjepa2-vitl-fpc64-256_is_present",
        "official_processor_support_status": "registered_dense_feature_path",
        "input_selection_support_status": "supported_as_dense_clip_selection_only",
        "zero_mask_support_status": "supported_as_image_space_probe_only_no_encoder_acceleration",
        "supports_context_mask_probe": "unknown",
        "supports_sparse_tubelet": "unknown",
        "supports_direct_mllm_projection": False,
        "requires_official_processor": True,
        "requires_training_or_trainable_adapter": False,
        "requires_training_for_mllm_projection": "true_or_unknown",
        "recommended_mode": "vjepa2_official_dense",
        "recommended_first_smoke_mode": "vjepa2_feature_extraction",
        "recommended_modes": [
            "vjepa2_official_dense",
            "autogaze_frame_selection_vjepa2",
            "autogaze_chop_selection_vjepa2",
            "autogaze_zero_mask_vjepa2",
        ],
        "risk_level": "medium_high",
        "status": "needs_code_inspection",
    },
    "generic_vit": {
        "model_name": "generic_vit",
        "default_model_id": "",
        "adapter_class": "GenericVitAdapter",
        "supported_integration_modes": [],
        "direct_token_injection_supported": False,
        "siglip_based": "unknown",
        "positional_encoding_status": "not_applicable",
        "requires_official_processor": False,
        "requires_training_or_trainable_adapter": "unknown",
        "recommended_mode": "stub_status_only",
        "risk_level": "high",
        "status": "stub-only",
    },
    "external": {
        "model_name": "external",
        "default_model_id": "",
        "adapter_class": "ExternalVisionEncoderAdapter",
        "supported_integration_modes": ["official_processor"],
        "direct_token_injection_supported": False,
        "siglip_based": "unknown",
        "positional_encoding_status": "requires explicit adapter/config inspection",
        "requires_official_processor": True,
        "requires_training_or_trainable_adapter": "unknown",
        "recommended_mode": "official_processor",
        "risk_level": "high",
        "status": "stub-only",
    },
}


def build_vision_encoder(name: str, config: Mapping | None = None) -> VisionEncoderAdapter:
    try:
        cls = VISION_ENCODERS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown vision encoder {name!r}; valid names: {sorted(VISION_ENCODERS)}") from exc
    return cls(config)


def build_mllm(name: str, config: Mapping | None = None) -> MLLMAdapter:
    try:
        cls = MLLMS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown MLLM {name!r}; valid names: {sorted(MLLMS)}") from exc
    return cls(config)


def get_mllm_registry_metadata(name: str | None = None) -> dict[str, object] | dict[str, dict[str, object]]:
    if name is None:
        return {key: dict(value) for key, value in MLLM_REGISTRY_METADATA.items()}
    try:
        return dict(MLLM_REGISTRY_METADATA[name])
    except KeyError as exc:
        raise ValueError(f"No MLLM metadata for {name!r}; valid names: {sorted(MLLM_REGISTRY_METADATA)}") from exc


def get_vision_encoder_registry_metadata(name: str | None = None) -> dict[str, object] | dict[str, dict[str, object]]:
    if name is None:
        return {key: dict(value) for key, value in VISION_ENCODER_REGISTRY_METADATA.items()}
    try:
        return dict(VISION_ENCODER_REGISTRY_METADATA[name])
    except KeyError as exc:
        raise ValueError(
            f"No vision encoder metadata for {name!r}; valid names: {sorted(VISION_ENCODER_REGISTRY_METADATA)}"
        ) from exc
