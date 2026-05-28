from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


DEFAULT_TITLE = "Colab Verification"
SUCCESS_STATUSES = {"passed", "executed", "ok", "success", True}

PREFERRED_TOKEN_KEYS = [
    "autogaze_raw_patch_tokens",
    "autogaze_selected_patch_tokens",
    "autogaze_reduction_ratio",
    "raw_patch_tokens_before_vit",
    "visual_tokens_before_prune",
    "visual_tokens_after_prune",
    "vjepa_raw_tokens",
    "vjepa_selected_tokens",
    "vjepa_reduction_ratio",
    "encoder_input_tokens",
    "llm_visual_tokens",
    "qwen_visual_tokens_inserted",
    "qwen_context_tokens",
    "llm_context_tokens",
]

PREFERRED_LATENCY_KEYS = [
    "video_metadata_read",
    "video_decode_ms",
    "video_decode_resize",
    "vjepa_video_decode_resize",
    "preprocess_without_autogaze",
    "autogaze_selector_total",
    "autogaze_total",
    "qwen_vit_prepare",
    "vit_ms",
    "vision_encoder_ms",
    "vjepa_sparse_encode",
    "qwen_bridge_pack",
    "projector_ms",
    "llm_forward_ms",
    "generation_rest_ms",
    "qwen_generate",
    "generate_ms",
    "total",
]

VISUALIZATION_KEYS = [
    ("selected frames", "selected_frames_grid_image"),
    ("AutoGaze overlay", "autogaze_overlay_image"),
    ("V-JEPA token mask", "vjepa_token_mask_image"),
    ("resized selected frames", "resized_selected_frames_grid_image"),
    ("resized AutoGaze overlay", "resized_autogaze_overlay_image"),
    ("NVILA selected frames video", "selected_frames_video"),
    ("NVILA AutoGaze overlay video", "overlay_video"),
    ("NVILA processor frames video", "processor_frames_video"),
    ("NVILA processor overlay video", "processor_overlay_video"),
    ("gazing info JSON", "gazing_info_json"),
    ("AutoGaze sparse plan JSON", "sparse_selection_plan_json"),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build colab_verification.md from direct AutoGaze runner JSON artifacts. "
            "This is independent of scripts/run_colab_autogaze_cuda_smoke.py."
        )
    )
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument("--video", default="")
    parser.add_argument("--query", default="")
    parser.add_argument("--entrypoint-verification-json")
    parser.add_argument("--case", action="append", default=[], help="Case in the form name=/path/to/result.json")
    parser.add_argument("--repo-root", default="")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--weights-root", default="")
    parser.add_argument("--commit", default="")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output_md = Path(args.output_md)
    payload = build_colab_verification_payload(
        title=args.title,
        video=args.video,
        query=args.query,
        cases=[parse_case_arg(item) for item in args.case],
        entrypoint_verification_json=Path(args.entrypoint_verification_json) if args.entrypoint_verification_json else None,
        output_md=output_md,
        repo_root=args.repo_root,
        output_root=args.output_root,
        weights_root=args.weights_root,
        commit=args.commit,
    )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_colab_verification_markdown(payload, output_md=output_md), encoding="utf-8")


def parse_case_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--case must use name=/path/to/result.json")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("--case name must not be empty")
    return name, Path(path)


def build_colab_verification_payload(
    *,
    title: str = DEFAULT_TITLE,
    video: str = "",
    query: str = "",
    cases: Iterable[tuple[str, Path]] = (),
    entrypoint_verification_json: Path | None = None,
    output_md: Path | None = None,
    repo_root: str = "",
    output_root: str = "",
    weights_root: str = "",
    commit: str = "",
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    for name, path in cases:
        case = load_case_result(name, path)
        results[name] = case
        if case.get("status") not in SUCCESS_STATUSES:
            failure = case.get("failure") or {}
            failures.append(
                {
                    "kind": failure.get("kind") or "case_failed",
                    "name": name,
                    "stage": failure.get("stage"),
                    "source_json": case.get("source_json"),
                }
            )

    if entrypoint_verification_json:
        verifier = load_entrypoint_verification(entrypoint_verification_json)
        results["entrypoint_verifier"] = verifier
        if verifier.get("status") not in ("passed", True):
            failures.append(
                {
                    "kind": verifier.get("failure", {}).get("kind") or "verification_failed",
                    "name": "entrypoint_verifier",
                    "source_json": str(entrypoint_verification_json),
                }
            )

    return {
        "title": title or DEFAULT_TITLE,
        "summary": {
            "passed": not failures,
            "case_count": len([name for name in results if name != "entrypoint_verifier"]),
            "failed_count": len(failures),
        },
        "paths": {
            "repo_root": repo_root,
            "output_root": output_root,
            "weights_root": weights_root,
            "video": video,
            "verification_md": str(output_md) if output_md else "",
            "entrypoint_verification_json": str(entrypoint_verification_json) if entrypoint_verification_json else "",
        },
        "commit": commit,
        "prompt": query,
        "results": results,
        "failures": failures,
    }


def load_case_result(name: str, path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "case": name,
            "source_json": str(path),
            "status": "missing_artifact",
            "failure": {"kind": "missing_artifact", "stage": "load_json", "path": str(path)},
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "case": name,
            "source_json": str(path),
            "status": "parse_failed",
            "failure": {"kind": "parse_failed", "stage": "load_json", "path": str(path), "message": str(exc)},
        }
    return normalize_case_result(name, path, payload)


def load_entrypoint_verification(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "status": "missing_artifact",
            "source_json": str(path),
            "failure": {"kind": "missing_artifact", "stage": "load_entrypoint_verification", "path": str(path)},
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "status": "parse_failed",
            "source_json": str(path),
            "failure": {"kind": "parse_failed", "stage": "load_entrypoint_verification", "path": str(path), "message": str(exc)},
        }
    matrix = payload.get("script_matrix") or []
    commands = payload.get("commands") or []
    return {
        "status": normalize_status(payload),
        "source_json": str(path),
        "summary": payload.get("summary") or {},
        "verified_script_ids": [row.get("id") for row in matrix if row.get("id")],
        "verified_entrypoints": [
            {
                "id": row.get("id"),
                "entrypoint": row.get("entrypoint"),
                "selector": row.get("selector"),
                "vit": row.get("vit"),
                "mllm": row.get("mllm"),
            }
            for row in matrix
        ],
        "dry_run_commands": [command.get("name") for command in commands if str(command.get("name", "")).endswith("_dry_run")],
    }


def normalize_case_result(name: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    generation = payload.get("generation") if isinstance(payload.get("generation"), dict) else {}
    return {
        "case": name,
        "source_json": str(path),
        "status": normalize_status(payload),
        "autogaze_mode": first_present(payload, [["autogaze_mode"], ["run_identity", "autogaze_mode"], ["config", "autogaze_mode"]]),
        "generated_text": extract_answer(payload),
        "tokens": extract_mapping(
            payload,
            [
                ["tokens"],
                ["token_metrics"],
                ["token_counts"],
                ["metrics", "tokens"],
                ["summary", "tokens"],
                ["result", "token_metrics"],
                ["generation", "metrics", "tokens"],
            ],
        ),
        "latency_ms": extract_mapping(
            payload,
            [
                ["latency_ms"],
                ["timing", "latency_ms"],
                ["metrics", "latency_ms"],
                ["summary", "latency_ms"],
                ["result", "latency_ms"],
                ["generation", "metrics", "latency_ms"],
            ],
        ),
        "memory_bytes": extract_mapping(
            payload,
            [
                ["memory_bytes"],
                ["memory"],
                ["metrics", "memory_bytes"],
                ["summary", "memory_bytes"],
                ["result", "memory_bytes"],
                ["generation", "metrics", "memory_bytes"],
            ],
        ),
        "visualizations": extract_visualizations(payload),
        "failure": first_present(payload, [["failure"], ["error"], ["summary", "failure"]]),
        "summary": summary or payload.get("summary"),
        "pipeline": infer_pipeline(name, payload),
        "execution_kind": infer_execution_kind(name, payload),
        "implementation_status": payload.get("implementation_status") or generation.get("status") or result.get("status"),
        "artifact_status": artifact_status(extract_visualizations(payload)),
    }


def normalize_status(payload: dict[str, Any]) -> str:
    status = payload.get("status")
    if isinstance(status, bool):
        return "passed" if status else "failed"
    if isinstance(status, str) and status:
        return status
    generation = payload.get("generation")
    if isinstance(generation, dict) and generation.get("status"):
        return str(generation.get("status"))
    summary = payload.get("summary")
    if isinstance(summary, dict) and summary.get("status"):
        return str(summary.get("status"))
    passed = (payload.get("summary") or {}).get("passed")
    if isinstance(passed, bool):
        return "passed" if passed else "failed"
    if payload.get("failure") or payload.get("error"):
        return "failed"
    if isinstance(payload.get("result"), dict) or isinstance(payload.get("generation"), dict):
        return "passed"
    return "unknown"


def extract_answer(payload: dict[str, Any]) -> str:
    value = first_present(
        payload,
        [
            ["generated_text"],
            ["answer"],
            ["prediction"],
            ["response"],
            ["raw_output"],
            ["result", "generated_text"],
            ["result", "answer"],
            ["result", "prediction"],
            ["result", "raw_output"],
            ["summary", "generated_text"],
            ["summary", "answer"],
            ["generation", "text"],
        ],
    )
    return "" if value is None else str(value)


def extract_mapping(payload: dict[str, Any], paths: list[list[str]]) -> dict[str, Any]:
    value = first_present(payload, paths)
    return value if isinstance(value, dict) else {}


def extract_visualizations(payload: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for path in (
        ["visualizations"],
        ["visualization"],
        ["artifacts", "visualizations"],
        ["result", "visualizations"],
        ["result", "visualization"],
        ["generation", "metrics", "visualizations"],
    ):
        value = first_present(payload, [path])
        if isinstance(value, dict):
            merged.update(value)
    sparse_plan = first_present(
        payload,
        [
            ["direct_autogaze_selector", "sparse_selection_plan_json"],
            ["autogaze_selector", "sparse_selection_plan_json"],
            ["generation", "metrics", "sparse_selection_plan_json"],
        ],
    )
    if sparse_plan:
        merged.setdefault("sparse_selection_plan_json", sparse_plan)
    return merged


def first_present(payload: dict[str, Any], paths: list[list[str]]) -> Any:
    for path in paths:
        current: Any = payload
        for key in path:
            if not isinstance(current, dict) or key not in current:
                current = None
                break
            current = current[key]
        if current is not None:
            return current
    return None


def render_colab_verification_markdown(payload: dict[str, Any], *, output_md: str | Path) -> str:
    output_path = Path(output_md)
    results = payload.get("results") or {}
    cases = [(name, result) for name, result in results.items() if name != "entrypoint_verifier"]
    verifier = results.get("entrypoint_verifier") or {}
    summary = payload.get("summary") or {}
    paths = payload.get("paths") or {}
    prompt = payload.get("prompt") or ""
    title = payload.get("title") or DEFAULT_TITLE
    lines = [
        f"# {title}",
        "",
        "## Environment",
        "",
        f"- status: `{summary.get('passed')}`",
        f"- cases: `{summary.get('case_count') or len(cases)}`",
        f"- failed: `{summary.get('failed_count')}`",
        f"- repo_root: `{paths.get('repo_root') or ''}`",
        f"- output_root: `{paths.get('output_root') or ''}`",
        f"- weights_root: `{paths.get('weights_root') or ''}`",
        f"- commit: `{payload.get('commit') or ''}`",
        "",
        "## Query / Video",
        "",
        f"- video: `{paths.get('video') or ''}`",
        f"- text_query: `{prompt}`",
        "",
        "## Case Summary",
        "",
        "| case | status | answer | total ms | selected / raw tokens | visual tokens | peak memory |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    if cases:
        lines.extend(_case_summary_row(name, result) for name, result in cases)
    else:
        lines.append("| _none_ |  |  |  |  |  |  |")
    lines.extend(render_v2_evidence_matrix(cases))
    lines.extend(
        [
            "",
            "## Answers",
            "",
            "| case | answer |",
            "|---|---|",
        ]
    )
    for name, result in cases:
        lines.append(f"| {markdown_escape(name)} | {markdown_escape(str(result.get('generated_text') or ''))} |")
    lines.extend(render_metric_table("Token Comparison", cases, "tokens", PREFERRED_TOKEN_KEYS))
    lines.extend(render_metric_table("Latency Comparison", cases, "latency_ms", PREFERRED_LATENCY_KEYS, prefix="latency_ms."))
    lines.extend(render_metric_table("Memory Comparison", cases, "memory_bytes", [], prefix="memory."))
    lines.extend(render_visualization_sections(cases, output_path))
    lines.extend(render_entrypoint_verification(verifier))
    lines.extend(render_failures(payload))
    lines.extend(
        [
            "## Artifacts",
            "",
            f"- summary_json: `{paths.get('summary_json') or ''}`",
            f"- verification_md: `{paths.get('verification_md') or str(output_path)}`",
            f"- entrypoint_verification_json: `{paths.get('entrypoint_verification_json') or ''}`",
            "",
        ]
    )
    return "\n".join(lines)


def render_v2_evidence_matrix(cases: list[tuple[str, dict[str, Any]]]) -> list[str]:
    lines = [
        "",
        "## V2 Pipeline Evidence Matrix",
        "",
        "| pipeline | case | kind | status | AutoGaze | selected / raw | visual artifacts | source |",
        "|---|---|---|---|---|---:|---|---|",
    ]
    if not cases:
        return lines + ["| _none_ |  |  |  |  |  |  |  |", ""]
    for name, result in cases:
        tokens = result.get("tokens") or {}
        visuals = result.get("visualizations") or {}
        lines.append(
            "| {pipeline} | {case} | {kind} | `{status}` | `{autogaze}` | {selected} / {raw} | {visuals} | `{source}` |".format(
                pipeline=markdown_escape(str(result.get("pipeline") or infer_pipeline(name, result))),
                case=markdown_escape(name),
                kind=markdown_escape(str(result.get("execution_kind") or infer_execution_kind(name, result))),
                status=markdown_escape(str(result.get("status") or "")),
                autogaze=markdown_escape(str(result.get("autogaze_mode") or autogaze_state_from_name(name))),
                selected=fmt(selected_token_value(tokens)),
                raw=fmt(raw_token_value(tokens)),
                visuals=markdown_escape(str(artifact_status(visuals))),
                source=markdown_escape(str(result.get("source_json") or "")),
            )
        )
    return lines + [""]


def _case_summary_row(name: str, result: dict[str, Any]) -> str:
    tokens = result.get("tokens") or {}
    latency = result.get("latency_ms") or {}
    return (
        f"| {markdown_escape(name)} | `{result.get('status')}` | {markdown_escape(str(result.get('generated_text') or ''))} | "
        f"{fmt(latency.get('total'))} | {fmt(selected_token_value(tokens))} / {fmt(raw_token_value(tokens))} | "
        f"{fmt(visual_token_value(tokens))} | {fmt_bytes(peak_memory(result))} |"
    )


def selected_token_value(tokens: dict[str, Any]) -> Any:
    return first_existing_key(
        tokens,
        [
            "autogaze_selected_patch_tokens",
            "visual_tokens_after_prune",
            "vjepa_selected_tokens",
            "encoder_input_tokens",
            "llm_visual_tokens",
            "qwen_visual_tokens_inserted",
        ],
    )


def raw_token_value(tokens: dict[str, Any]) -> Any:
    return first_existing_key(
        tokens,
        [
            "autogaze_raw_patch_tokens",
            "raw_patch_tokens_before_vit",
            "visual_tokens_before_prune",
            "vjepa_raw_tokens",
            "full_patch_tokens",
        ],
    )


def visual_token_value(tokens: dict[str, Any]) -> Any:
    return first_existing_key(tokens, ["llm_visual_tokens", "qwen_visual_tokens_inserted", "visual_tokens_after_prune"])


def infer_pipeline(name: str, payload: dict[str, Any]) -> str:
    text = " ".join(
        str(value)
        for value in [
            name,
            payload.get("runner"),
            payload.get("model_path"),
            payload.get("case"),
            payload.get("integration_level"),
            payload.get("implementation_status"),
        ]
    ).lower()
    generation = payload.get("generation") if isinstance(payload.get("generation"), dict) else {}
    adapter = str(generation.get("adapter") or "").lower()
    if "vjepa" in text:
        return "V-JEPA2 + Qwen"
    if "nvila" in text:
        return "NVILA-HD"
    if "qwen" in text or "qwen" in adapter:
        return "Qwen"
    return "unknown"


def infer_execution_kind(name: str, payload: dict[str, Any]) -> str:
    text = f"{name} {payload.get('source_json') or ''}".lower()
    if "hlvid" in text or "benchmark" in text:
        return "benchmark"
    return "single"


def autogaze_state_from_name(name: str) -> str:
    lower = name.lower()
    if "autogaze" in lower or "sparse" in lower or lower.endswith("_on"):
        return "on"
    if "off" in lower or "keep" in lower or "dense" in lower or "full" in lower:
        return "off"
    return "unknown"


def artifact_status(visualizations: dict[str, Any]) -> str:
    if not isinstance(visualizations, dict) or not visualizations:
        return "missing"
    present = [key for key, value in visualizations.items() if value not in (None, "", [])]
    if not present:
        return "missing"
    frame_count = visualizations.get("frame_count_rendered") or visualizations.get("sampled_frame_count")
    if frame_count:
        return f"recorded:{len(present)} artifacts, frames={frame_count}"
    return f"recorded:{len(present)} artifacts"


def first_existing_key(mapping: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def render_metric_table(
    title: str,
    cases: list[tuple[str, dict[str, Any]]],
    section: str,
    preferred_keys: list[str],
    *,
    prefix: str = "",
) -> list[str]:
    keys = metric_keys(cases, section, preferred_keys)
    lines = ["", f"## {title}", ""]
    if not keys:
        return lines + ["_No data available._", ""]
    lines.append("| metric | " + " | ".join(markdown_escape(name) for name, _ in cases) + " |")
    lines.append("|---" + "|---:" * len(cases) + "|")
    for key in keys:
        cells = [metric(result, section, key) for _, result in cases]
        lines.append(f"| {prefix}{markdown_escape(key)} | " + " | ".join(cells) + " |")
    return lines + [""]


def metric_keys(cases: list[tuple[str, dict[str, Any]]], section: str, preferred_keys: list[str]) -> list[str]:
    seen: set[str] = set()
    dynamic: list[str] = []
    for _, result in cases:
        values = result.get(section) or {}
        if not isinstance(values, dict):
            continue
        for key in values:
            if key not in seen:
                seen.add(key)
                dynamic.append(key)
    if not preferred_keys:
        return dynamic
    ordered = [key for key in preferred_keys if key in seen]
    ordered.extend(key for key in dynamic if key not in ordered)
    return ordered


def metric(result: dict[str, Any], section: str, key: str) -> str:
    return fmt((result.get(section) or {}).get(key))


def render_visualization_sections(cases: list[tuple[str, dict[str, Any]]], output_md: Path) -> list[str]:
    lines = ["", "## Visualizations", ""]
    any_visuals = False
    for name, result in cases:
        visualizations = result.get("visualizations") or {}
        case_lines: list[str] = []
        for label, key in VISUALIZATION_KEYS:
            path = visualizations.get(key)
            if not path:
                continue
            any_visuals = True
            link = markdown_path(str(path), output_md)
            case_lines.extend([f"### {name}: {label}", ""])
            if _looks_like_inline_media(path):
                case_lines.extend([f"![{markdown_escape(label)}]({link})", ""])
            else:
                case_lines.extend([f"[{markdown_escape(label)}]({link})", ""])
            case_lines.extend([f"`{path}`", ""])
        if case_lines:
            metadata = {
                key: value
                for key, value in visualizations.items()
                if key not in {artifact_key for _, artifact_key in VISUALIZATION_KEYS}
            }
            if metadata:
                case_lines.extend(["| visualization metadata | value |", "|---|---|"])
                for key, value in metadata.items():
                    case_lines.append(f"| {markdown_escape(str(key))} | `{markdown_escape(str(value))}` |")
                case_lines.append("")
            lines.extend(case_lines)
    if not any_visuals:
        lines.append("_No visualization artifacts recorded._")
        lines.append("")
    return lines


def _looks_like_inline_media(path: Any) -> bool:
    suffix = Path(str(path)).suffix.lower()
    return suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".mp4", ".mov", ".webm"}


def render_entrypoint_verification(verifier: dict[str, Any]) -> list[str]:
    lines = [
        "",
        "## Entrypoint Verification",
        "",
    ]
    if not verifier:
        return lines + ["_No entrypoint verification artifact recorded._", ""]
    lines.extend(
        [
            f"- status: `{verifier.get('status')}`",
            f"- summary: `{json.dumps(verifier.get('summary') or {}, sort_keys=True)}`",
            f"- verified_script_ids: `{', '.join(str(item) for item in verifier.get('verified_script_ids') or [])}`",
            f"- dry_run_commands: `{', '.join(str(item) for item in verifier.get('dry_run_commands') or [])}`",
            "",
        ]
    )
    entrypoints = verifier.get("verified_entrypoints") or []
    if entrypoints:
        lines.extend(["| id | entrypoint | selector | vit | mllm |", "|---|---|---|---|---|"])
        for row in entrypoints:
            lines.append(
                "| {id} | `{entrypoint}` | {selector} | {vit} | {mllm} |".format(
                    id=markdown_escape(str(row.get("id") or "")),
                    entrypoint=markdown_escape(str(row.get("entrypoint") or "")),
                    selector=markdown_escape(str(row.get("selector") or "")),
                    vit=markdown_escape(str(row.get("vit") or "")),
                    mllm=markdown_escape(str(row.get("mllm") or "")),
                )
            )
        lines.append("")
    return lines


def render_failures(payload: dict[str, Any]) -> list[str]:
    failures = payload.get("failures") or []
    lines = ["", "## Failures", ""]
    if not failures:
        return lines + ["_No failures recorded._", ""]
    lines.extend(["| kind | name | stage | source |", "|---|---|---|---|"])
    for failure in failures:
        lines.append(
            "| {kind} | {name} | {stage} | `{source}` |".format(
                kind=markdown_escape(str(failure.get("kind") or "")),
                name=markdown_escape(str(failure.get("name") or "")),
                stage=markdown_escape(str(failure.get("stage") or "")),
                source=markdown_escape(str(failure.get("source_json") or "")),
            )
        )
    lines.append("")
    return lines


def peak_memory(result: dict[str, Any]) -> Any:
    memory = result.get("memory_bytes") or {}
    if not isinstance(memory, dict):
        return None
    if "cuda_peak_total" in memory:
        return memory.get("cuda_peak_total")
    numeric_values = [value for value in memory.values() if isinstance(value, (int, float))]
    return max(numeric_values, default=None)


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def fmt_bytes(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return ""
    return f"{float(value) / (1024 ** 3):.3f} GiB"


def markdown_path(path: str, output_md: Path) -> str:
    artifact = Path(path)
    try:
        return artifact.relative_to(output_md.parent).as_posix()
    except ValueError:
        return artifact.as_posix()


def markdown_escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    main()
