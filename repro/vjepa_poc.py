from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from repro.common import write_json
from repro.plugins.gaze_plan import (
    EncoderMapping,
    MllmMapping,
    PatchSpace,
    PreprocessSpace,
    SelectedPatch,
    SourceVideo,
    SparseSelectionPlan,
    sparse_selection_plan_from_dict,
)
from repro.plugins.vjepa_mapping import VjepaGridConfig, vjepa_token_selection_from_sparse_plan
from repro.plugins.vjepa_mapping import scale_aware_vjepa_selection_from_sparse_plan


def build_synthetic_sparse_selection_plan() -> SparseSelectionPlan:
    patches = [
        SelectedPatch(
            frame_index=0,
            frame_order=0,
            tile_id=0,
            scale_id=1,
            scale_size=224,
            patch_index=0,
            bbox_resized_xyxy=[0, 0, 16, 16],
            bbox_original_xyxy=[0.0, 0.0, 16.0, 16.0],
            autoregressive_order=0,
        ),
        SelectedPatch(
            frame_index=1,
            frame_order=1,
            tile_id=0,
            scale_id=1,
            scale_size=224,
            patch_index=1,
            bbox_resized_xyxy=[16, 0, 32, 16],
            bbox_original_xyxy=[16.0, 0.0, 32.0, 16.0],
            autoregressive_order=1,
        ),
        SelectedPatch(
            frame_index=2,
            frame_order=2,
            tile_id=0,
            scale_id=0,
            scale_size=112,
            patch_index=0,
            bbox_resized_xyxy=[0, 0, 32, 32],
            bbox_original_xyxy=[0.0, 0.0, 32.0, 32.0],
            autoregressive_order=2,
        ),
    ]
    return SparseSelectionPlan(
        selector_name="autogaze-direct-synthetic",
        source_video=SourceVideo(
            path="synthetic://autogaze-vjepa",
            source_width=224,
            source_height=224,
            sampled_frame_indices=[0, 1, 2, 3],
        ),
        preprocess_space=PreprocessSpace(
            resize_policy="synthetic_square_224",
            resized_width=224,
            resized_height=224,
        ),
        patch_space=PatchSpace(
            autogaze_patch_size=16,
            encoder_patch_size=16,
            scale_ids=[0, 1],
            scale_sizes=[112, 224],
        ),
        selected_patches=patches,
        encoder_mapping=EncoderMapping(status="not_mapped"),
        mllm_mapping=MllmMapping(status="not_mapped"),
        raw_patch_tokens=4 * ((112 // 16) ** 2 + (224 // 16) ** 2),
        selected_patch_tokens=len(patches),
        quality_control={
            "purpose": "Kaggle-safe synthetic mapping probe",
            "expected_behavior": "tubelet union and coarse-scale bbox expansion",
        },
    )


def run_mapping_probe(
    *,
    sparse_selection_plan: SparseSelectionPlan,
    frames_per_clip: int,
    tubelet_size: int,
    crop_size: int,
    patch_size: int,
    include_scale_aware: bool = False,
    tiny_encoder_smoke: bool = False,
    qwen_bridge_smoke: bool = False,
    output_json: str | Path | None = None,
    output_md: str | Path | None = None,
) -> dict[str, Any]:
    grid_config = VjepaGridConfig(
        frames_per_clip=int(frames_per_clip),
        tubelet_size=int(tubelet_size),
        crop_size=int(crop_size),
        patch_size=int(patch_size),
    )
    selection = vjepa_token_selection_from_sparse_plan(sparse_selection_plan, grid_config)
    payload = {
        "runner": "repro.vjepa_poc",
        "implementation_status": "mapping_probe_ready" if selection.status == "mapped" else "mapping_failed",
        "integration_level": "pre_vit_sparse_mapping_probe",
        "accuracy_status": "not_claimed",
        "sparse_selection_plan": sparse_selection_plan.to_dict(),
        **selection.to_dict(),
        "next_steps": [
            "Use selected_token_indices after dense V-JEPA Conv3D patch embedding.",
            "Run V-JEPA transformer encoder on selected hidden states only.",
            "Treat Qwen bridge without a trained projector as zero_shot_wiring_probe.",
        ],
    }
    if include_scale_aware:
        scale_selection = scale_aware_vjepa_selection_from_sparse_plan(
            sparse_selection_plan,
            frames_per_clip=frames_per_clip,
            tubelet_size=tubelet_size,
            patch_size=patch_size,
        )
        payload["scale_aware_vjepa"] = scale_selection.to_dict()["vjepa"]
    if tiny_encoder_smoke:
        payload["vjepa_sparse_encoder_smoke"] = _run_tiny_vjepa_encoder_smoke(
            grid_config=grid_config,
            selected_token_indices=selection.selected_token_indices,
        )
    if qwen_bridge_smoke:
        from repro.plugins.vjepa_qwen_bridge import run_fake_qwen_bridge_smoke

        payload["vjepa_qwen_bridge_smoke"] = run_fake_qwen_bridge_smoke(
            selected_token_count=max(1, len(selection.selected_token_indices)),
            vjepa_hidden_size=72,
            qwen_hidden_size=128,
        )
    if output_json is not None:
        write_json(output_json, payload)
    if output_md is not None:
        _write_markdown_report(output_md, payload)
    return payload


def load_sparse_selection_plan(path: str | Path) -> SparseSelectionPlan:
    return sparse_selection_plan_from_dict(json.loads(Path(path).read_text()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AutoGaze + V-JEPA sparse mapping PoC")
    parser.add_argument("--synthetic", action="store_true", help="Run a Kaggle-safe synthetic mapping probe.")
    parser.add_argument("--sparse-selection-plan-json", help="SparseSelectionPlan JSON emitted by AutoGaze selector.")
    parser.add_argument("--frames-per-clip", type=int, default=4)
    parser.add_argument("--tubelet-size", type=int, default=2)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--scale-aware", action="store_true", help="Also run per-AutoGaze-scale V-JEPA packing probe.")
    parser.add_argument(
        "--tiny-encoder-smoke",
        action="store_true",
        help="Run a random-weight tiny V-JEPA encoder on selected embeddings to verify the sparse hook.",
    )
    parser.add_argument(
        "--qwen-bridge-smoke",
        action="store_true",
        help="Run a fake Qwen generate smoke that packs selected V-JEPA tokens as Qwen video embeddings.",
    )
    parser.add_argument("--output-json", default="outputs/autogaze_vjepa/vjepa_mapping_probe.json")
    parser.add_argument("--output-md", default="outputs/autogaze_vjepa/vjepa_mapping_probe.md")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.synthetic:
        plan = build_synthetic_sparse_selection_plan()
    elif args.sparse_selection_plan_json:
        plan = load_sparse_selection_plan(args.sparse_selection_plan_json)
    else:
        raise SystemExit("--synthetic or --sparse-selection-plan-json is required")

    payload = run_mapping_probe(
        sparse_selection_plan=plan,
        frames_per_clip=args.frames_per_clip,
        tubelet_size=args.tubelet_size,
        crop_size=args.crop_size,
        patch_size=args.patch_size,
        include_scale_aware=args.scale_aware,
        tiny_encoder_smoke=args.tiny_encoder_smoke,
        qwen_bridge_smoke=args.qwen_bridge_smoke,
        output_json=args.output_json,
        output_md=args.output_md,
    )
    print(json.dumps(_summary_for_stdout(payload), indent=2, sort_keys=True))
    return 0


def _summary_for_stdout(payload: dict[str, Any]) -> dict[str, Any]:
    vjepa = payload["vjepa"]
    summary = {
        "implementation_status": payload["implementation_status"],
        "accuracy_status": payload["accuracy_status"],
        "grid_thw": vjepa["grid_thw"],
        "raw_token_count": vjepa["raw_token_count"],
        "selected_token_count": vjepa["selected_token_count"],
        "reduction_ratio": vjepa["reduction_ratio"],
        "selected_tokens_by_scale": vjepa["selected_tokens_by_scale"],
    }
    if "scale_aware_vjepa" in payload:
        scale = payload["scale_aware_vjepa"]
        summary["scale_aware"] = {
            "raw_token_count": scale["raw_token_count"],
            "selected_token_count": scale["selected_token_count"],
            "reduction_ratio": scale["reduction_ratio"],
            "selected_tokens_by_scale": scale["selected_tokens_by_scale"],
        }
    if "vjepa_sparse_encoder_smoke" in payload:
        smoke = payload["vjepa_sparse_encoder_smoke"]
        summary["vjepa_sparse_encoder_smoke"] = {
            "status": smoke["status"],
            "metrics": smoke["metrics"],
            "last_hidden_state_shape": smoke["last_hidden_state_shape"],
        }
    if "vjepa_qwen_bridge_smoke" in payload:
        smoke = payload["vjepa_qwen_bridge_smoke"]
        summary["vjepa_qwen_bridge_smoke"] = {
            "status": smoke["status"],
            "visual_tokens_inserted": smoke["bridge_metadata"]["visual_tokens_inserted"],
            "accuracy_status": smoke["bridge_metadata"]["accuracy_status"],
        }
    return summary


def _write_markdown_report(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    vjepa = payload["vjepa"]
    lines = [
        "# AutoGaze + V-JEPA PoC",
        "",
        "## Status",
        "",
        f"- implementation_status: `{payload['implementation_status']}`",
        f"- integration_level: `{payload['integration_level']}`",
        f"- accuracy_status: `{payload['accuracy_status']}`",
        "",
        "## V-JEPA Token Mapping",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| grid_thw | `{vjepa['grid_thw']}` |",
        f"| raw_token_count | {vjepa['raw_token_count']} |",
        f"| selected_token_count | {vjepa['selected_token_count']} |",
        f"| reduction_ratio | {vjepa['reduction_ratio']} |",
        "",
        "## Mapping Policy",
        "",
        "| Policy | Value |",
        "|---|---|",
    ]
    for key, value in vjepa["mapping_policy"].items():
        lines.append(f"| {key} | `{value}` |")
    lines.extend(
        [
            "",
            "## Selected Tokens By AutoGaze Scale",
            "",
            "| scale_id | selected V-JEPA tokens |",
            "|---:|---:|",
        ]
    )
    for scale_id, count in vjepa["selected_tokens_by_scale"].items():
        lines.append(f"| {scale_id} | {count} |")
    if "scale_aware_vjepa" in payload:
        scale = payload["scale_aware_vjepa"]
        lines.extend(
            [
                "",
                "## Scale-Aware V-JEPA Packing",
                "",
                "| Metric | Value |",
                "|---|---:|",
                f"| raw_token_count | {scale['raw_token_count']} |",
                f"| selected_token_count | {scale['selected_token_count']} |",
                f"| reduction_ratio | {scale['reduction_ratio']} |",
                "",
                "| scale_id | grid_thw | selected V-JEPA tokens |",
                "|---:|---|---:|",
            ]
        )
        for scale_id, item in scale["scale_passes"].items():
            lines.append(f"| {scale_id} | `{item['grid_thw']}` | {item['selected_token_count']} |")
    if "vjepa_sparse_encoder_smoke" in payload:
        smoke = payload["vjepa_sparse_encoder_smoke"]
        lines.extend(
            [
                "",
                "## Tiny Sparse Encoder Smoke",
                "",
                f"- status: `{smoke['status']}`",
                f"- last_hidden_state_shape: `{smoke['last_hidden_state_shape']}`",
                f"- position_mask_shape: `{smoke['position_mask_shape']}`",
                f"- encoder_token_reduction_ratio: `{smoke['metrics']['encoder_token_reduction_ratio']}`",
            ]
        )
    if "vjepa_qwen_bridge_smoke" in payload:
        smoke = payload["vjepa_qwen_bridge_smoke"]
        metadata = smoke["bridge_metadata"]
        lines.extend(
            [
                "",
                "## V-JEPA To Qwen Bridge Smoke",
                "",
                f"- status: `{smoke['status']}`",
                f"- visual_tokens_inserted: `{metadata['visual_tokens_inserted']}`",
                f"- qwen_hidden_size: `{metadata['qwen_hidden_size']}`",
                f"- projection: `{metadata['projection']}`",
                f"- accuracy_status: `{metadata['accuracy_status']}`",
                f"- inputs_embeds_shape: `{smoke['generate_kwargs']['inputs_embeds_shape']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `tubelet_size=2` uses union across the two frames in each temporal tubelet.",
            "- Coarse AutoGaze patches expand to all overlapping V-JEPA spatial cells.",
            "- This report does not claim Qwen accuracy without a trained V-JEPA-to-Qwen projector.",
            "",
        ]
    )
    target.write_text("\n".join(lines))


def _run_tiny_vjepa_encoder_smoke(
    *,
    grid_config: VjepaGridConfig,
    selected_token_indices: list[int],
) -> dict[str, Any]:
    import torch
    from transformers.models.vjepa2.configuration_vjepa2 import VJEPA2Config
    from transformers.models.vjepa2.modeling_vjepa2 import VJEPA2Encoder

    from repro.plugins.vjepa_sparse_runtime import run_vjepa_encoder_on_selected_embeddings

    config = VJEPA2Config(
        crop_size=int(grid_config.crop_size),
        patch_size=int(grid_config.patch_size),
        frames_per_clip=int(grid_config.frames_per_clip),
        tubelet_size=int(grid_config.tubelet_size),
        hidden_size=72,
        num_attention_heads=3,
        num_hidden_layers=1,
        mlp_ratio=2.0,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(7)
    encoder = VJEPA2Encoder(config)
    patch_embeddings = torch.randn(1, grid_config.raw_token_count, config.hidden_size)
    result = run_vjepa_encoder_on_selected_embeddings(
        encoder,
        patch_embeddings,
        selected_token_indices=selected_token_indices,
    )
    return {
        "status": "passed",
        "model": "random_weight_tiny_vjepa2_encoder",
        "note": "This verifies the sparse encoder hook and position_mask plumbing; it is not an accuracy run.",
        "last_hidden_state_shape": list(result["last_hidden_state"].shape),
        "position_mask_shape": list(result["position_mask"].shape),
        "metrics": result["metrics"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
