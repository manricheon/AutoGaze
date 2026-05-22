from __future__ import annotations

from typing import Any


def build_pre_vit_sparse_contract(model_family: str) -> dict[str, Any]:
    family = _normalize_family(model_family)
    if family in {"qwen2.5-vl", "qwen3-vl", "qwen3-vl-moe", "qwen2-vl"}:
        status = "implemented_pending_cuda" if family in {"qwen2.5-vl", "qwen3-vl"} else "adapter_probe_required"
        return _contract(
            model_family=model_family,
            family_group="qwen_grid_vl",
            priority=1 if family == "qwen2.5-vl" else 2,
            difficulty="low" if family in {"qwen2.5-vl", "qwen3-vl"} else "medium",
            status=status,
            selector_plan_format="SparseSelectionPlan",
            model_grid_fields=["pixel_values_videos", "video_grid_thw", "spatial_merge_size"],
            coordinate_space="qwen_video_grid_thw_merged_visual_tokens",
            required_hooks=[
                "processor pixel_values_videos output",
                "video_grid_thw",
                "model.visual/get_video_features input",
                "visual placeholder insertion order",
            ],
            expected_gain=["vision_encoder", "mllm_prefill", "kv_cache"],
            next_actions=[
                "CUDA smoke full/chunked/sparse modes on the same video",
                "verify visual_tokens_after_prune is lower than visual_tokens_before_prune",
                "verify answer quality and HLVid scoring denominator",
            ],
        )
    if family in {"nvila-video-plugin", "longvila"}:
        return _contract(
            model_family=model_family,
            family_group="vila_family",
            priority=3 if family == "nvila-video-plugin" else 4,
            difficulty="medium_high",
            status="in_process_probe_required",
            selector_plan_format="SparseSelectionPlan",
            model_grid_fields=["video frame tensor", "tile metadata", "vision tower patch order"],
            coordinate_space="siglip_patch_grid_before_vila_projector",
            required_hooks=[
                "processor video tensor/frame contract",
                "vision_tower.forward input tensor",
                "vision_tower.forward output tokens",
                "mm_projector input/output tokens",
                "LLM visual token insertion boundary",
            ],
            expected_gain=["vision_encoder", "mllm_prefill", "kv_cache"],
            next_actions=[
                "load VILA-family model in-process instead of external CLI",
                "capture processor output pixel tensor and frame/tile metadata",
                "map SparseSelectionPlan patch coordinates to vision tower token order",
            ],
        )
    if family == "internvl3":
        return _contract(
            model_family=model_family,
            family_group="internvl_dynamic_tile",
            priority=5,
            difficulty="medium",
            status="dynamic_tile_probe_required",
            selector_plan_format="SparseSelectionPlan",
            model_grid_fields=["pixel_values", "num_patches_list"],
            coordinate_space="dynamic_tile_grid_plus_patch_row_col",
            required_hooks=[
                "dynamic preprocess tile list",
                "num_patches_list",
                "vision_model pixel_values input",
                "language model visual feature packing order",
            ],
            expected_gain=["vision_encoder", "mllm_prefill", "kv_cache"],
            next_actions=[
                "collect dynamic tile config and num_patches_list mapping",
                "apply tile-level prune before patch-level prune",
                "verify thumbnail tile policy",
            ],
        )
    if family == "llava-onevision":
        candidate = build_llava_onevision_pre_vit_candidate()
        return _contract(
            model_family=model_family,
            family_group="llava_onevision",
            priority=6,
            difficulty="high",
            status="candidate_design_required",
            selector_plan_format="SparseSelectionPlan",
            model_grid_fields=["videos", "image_grid_thw/video_grid", "pooled video tokens"],
            coordinate_space="frame_or_tile_before_siglip_pooling",
            required_hooks=[
                "video frame/tile sampling boundary",
                "SigLIP input tensor",
                "video pooling input/output",
                "visual placeholder insertion order",
            ],
            expected_gain=["vision_encoder_if_frame_or_tile_pruned", "mllm_prefill", "kv_cache"],
            next_actions=candidate["next_actions"],
            extra=candidate,
        )
    if family in {"generic-siglip", "generic-clip", "vjepa2"}:
        return _contract(
            model_family=model_family,
            family_group="generic_vit",
            priority=7,
            difficulty="medium",
            status="vit_only_adapter_required",
            selector_plan_format="SparseSelectionPlan",
            model_grid_fields=["pixel_values", "patch_size", "position_embeddings"],
            coordinate_space="patch_embedding_sequence",
            required_hooks=[
                "patch embedding output",
                "absolute/rope position embedding",
                "encoder attention mask",
            ],
            expected_gain=["vision_encoder"],
            next_actions=[
                "build ViT-only sparse benchmark before MLLM integration",
                "define how sparse features are packed for downstream projector",
            ],
        )
    return _contract(
        model_family=model_family,
        family_group="unknown_remote_code",
        priority=99,
        difficulty="high",
        status="probe_required",
        selector_plan_format="SparseSelectionPlan",
        model_grid_fields=[],
        coordinate_space="unknown",
        required_hooks=[
            "processor output tensor",
            "vision encoder input/output",
            "projector input/output",
            "LLM visual token insertion boundary",
        ],
        expected_gain=[],
        next_actions=["collect model-specific feature packing probe"],
    )


def pre_vit_sparse_model_matrix() -> list[dict[str, Any]]:
    families = [
        "qwen2.5-vl",
        "qwen3-vl",
        "nvila-video-plugin",
        "longvila",
        "internvl3",
        "llava-onevision",
        "generic-siglip",
        "vjepa2",
        "unknown-remote-code",
    ]
    rows = [build_pre_vit_sparse_contract(family) for family in families]
    return sorted(rows, key=lambda row: (int(row["priority"]), str(row["model_family"])))


def build_llava_onevision_pre_vit_candidate() -> dict[str, Any]:
    return {
        "model_family": "llava-onevision",
        "recommended_first_pre_vit_unit": "frame_or_tile",
        "patch_level_status": "hard_due_to_video_pooling",
        "candidate_paths": [
            {
                "name": "frame_level_pre_vit",
                "selector_mapping": "aggregate AutoGaze selected patches per frame and keep selected frames before SigLIP",
                "expected_gain": ["vision_encoder_if_frames_are_dropped", "mllm_prefill"],
                "risk": "coarse selection may discard temporal evidence",
            },
            {
                "name": "tile_level_pre_vit",
                "selector_mapping": "map AutoGaze boxes to anyres/video tiles and keep selected tiles before SigLIP",
                "expected_gain": ["vision_encoder", "mllm_prefill"],
                "risk": "must preserve tile order and video pooling metadata",
            },
            {
                "name": "patch_level_pre_pooling",
                "selector_mapping": "sparse SigLIP patch tokens before LLaVA video pooling",
                "expected_gain": ["vision_encoder", "mllm_prefill"],
                "risk": "hardest path because pooled visual tokens may assume dense grids",
            },
        ],
        "guardrails": [
            "do not silently fall back to dense execution when AutoGaze is requested",
            "do not claim ViT patch-level gain until pooling boundary is bypassed",
            "report integration_level as frame_or_tile_pre_vit or post_encoder_token_prune",
        ],
        "next_actions": [
            "inspect processor video tensor shape for local checkpoint",
            "capture SigLIP input tensor and pooled token count",
            "try frame/tile pruning before patch-level sparse pooling",
        ],
    }


def _contract(
    *,
    model_family: str,
    family_group: str,
    priority: int,
    difficulty: str,
    status: str,
    selector_plan_format: str,
    model_grid_fields: list[str],
    coordinate_space: str,
    required_hooks: list[str],
    expected_gain: list[str],
    next_actions: list[str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model_family": model_family,
        "family_group": family_group,
        "priority": int(priority),
        "difficulty": difficulty,
        "status": status,
        "selector_plan_format": selector_plan_format,
        "model_grid_fields": list(model_grid_fields),
        "coordinate_space": coordinate_space,
        "required_hooks": list(required_hooks),
        "expected_gain": list(expected_gain),
        "next_actions": list(next_actions),
    }
    if extra:
        payload.update(extra)
    return payload


def _normalize_family(model_family: str) -> str:
    return str(model_family or "").strip().lower()
