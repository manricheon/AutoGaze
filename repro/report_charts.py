from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_COLORS = (
    "#3366cc",
    "#dc3912",
    "#ff9900",
    "#109618",
    "#990099",
    "#0099c6",
    "#dd4477",
)

STAGE_COLORS = {
    "Pre(no AG)": "#5b8def",
    "AutoGaze": "#f59f00",
    "ViT": "#2f9e44",
    "LLM": "#7048e8",
    "Other": "#adb5bd",
    "Full patch": "#5b8def",
    "Selected patch": "#f59f00",
    "LLM visual": "#7048e8",
    "Peak": "#495057",
    "Overall": "#495057",
}


@dataclass(frozen=True)
class ChartSegment:
    name: str
    value: float
    color: str | None = None


@dataclass(frozen=True)
class ChartBar:
    label: str
    segments: list[ChartSegment]


@dataclass(frozen=True)
class ChartArtifact:
    title: str
    path: Path


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "chart"


def shorten_label(value: str, *, max_chars: int = 32) -> str:
    text = str(value)
    if len(text) <= max_chars:
        return text
    if "/" in text:
        suffix = text.rsplit("/", 1)[-1]
        if suffix:
            prefix_len = max(8, min(len(text), max(14, max_chars - len(suffix) - 3)))
            return f"{text[:prefix_len]}...{suffix}"
    if max_chars <= 8:
        return text[:max_chars]
    keep = max_chars - 3
    left = max(1, keep // 2)
    right = max(1, keep - left)
    return f"{text[:left]}...{text[-right:]}"


def numeric_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def first_number(*values: Any) -> float | None:
    for value in values:
        if isinstance(value, dict):
            for key in (
                "median",
                "mean",
                "value",
                "autogaze",
                "after_autogaze_actual",
                "after_autogaze",
                "keep_all",
                "before_keep_all_estimated",
            ):
                number = numeric_or_none(value.get(key))
                if number is not None:
                    return number
        number = numeric_or_none(value)
        if number is not None:
            return number
    return None


def write_bar_chart(
    path: str | Path,
    *,
    title: str,
    bars: list[ChartBar],
    unit: str = "",
    width: int = 960,
) -> ChartArtifact:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    clean_bars = [
        ChartBar(
            label=bar.label,
            segments=[segment for segment in bar.segments if numeric_or_none(segment.value) not in {None, 0.0}],
        )
        for bar in bars
    ]
    clean_bars = [bar for bar in clean_bars if bar.segments]
    if not clean_bars:
        clean_bars = [ChartBar(label="no_data", segments=[ChartSegment("no_data", 1.0, "#cccccc")])]
        unit = ""

    label_width = 190
    right_pad = 130
    bar_width = max(width - label_width - right_pad, 160)
    row_height = 42
    top = 54
    bottom = 74
    height = top + bottom + len(clean_bars) * row_height
    max_total = max(sum(max(float(segment.value), 0.0) for segment in bar.segments) for bar in clean_bars) or 1.0

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-size:12px;fill:#222}.title{font-size:18px;font-weight:700}.label{font-weight:600}.legend{font-size:11px}</style>',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text class="title" x="16" y="28">{html.escape(title)}</text>',
    ]
    legend_names: list[tuple[str, str]] = []
    color_index = 0
    for row_index, bar in enumerate(clean_bars):
        y = top + row_index * row_height
        total = sum(max(float(segment.value), 0.0) for segment in bar.segments)
        svg.append(f'<text class="label" x="16" y="{y + 20}">{html.escape(shorten_label(bar.label))}</text>')
        svg.append(f'<rect x="{label_width}" y="{y + 6}" width="{bar_width}" height="22" fill="#f1f3f5"/>')
        x = label_width
        for segment in bar.segments:
            color = segment.color or STAGE_COLORS.get(segment.name) or DEFAULT_COLORS[color_index % len(DEFAULT_COLORS)]
            color_index += 1
            segment_width = bar_width * max(float(segment.value), 0.0) / max_total
            svg.append(
                f'<rect x="{x:.2f}" y="{y + 6}" width="{segment_width:.2f}" height="22" fill="{color}"/>'
            )
            x += segment_width
            if (segment.name, color) not in legend_names:
                legend_names.append((segment.name, color))
        value_label = _format_chart_value(total, unit)
        svg.append(f'<text x="{label_width + bar_width + 10}" y="{y + 21}">{html.escape(value_label)}</text>')

    legend_x = 16
    legend_y = height - 34
    for name, color in legend_names[:8]:
        svg.append(f'<rect x="{legend_x}" y="{legend_y - 10}" width="10" height="10" fill="{color}"/>')
        svg.append(f'<text class="legend" x="{legend_x + 14}" y="{legend_y}">{html.escape(name)}</text>')
        legend_x += min(150, 42 + len(name) * 7)
    svg.append("</svg>")
    target.write_text("\n".join(svg) + "\n", encoding="utf-8")
    return ChartArtifact(title=title, path=target)


def _format_chart_value(value: float, unit: str) -> str:
    if value >= 1000:
        formatted = f"{value:,.0f}"
    elif value >= 10:
        formatted = f"{value:,.1f}".rstrip("0").rstrip(".")
    else:
        formatted = f"{value:.3f}".rstrip("0").rstrip(".")
    return f"{formatted} {unit}".strip()


def build_standard_report_charts(
    *,
    metrics: dict[str, Any],
    output_dir: str | Path,
) -> list[ChartArtifact]:
    output = Path(output_dir)
    artifacts: list[ChartArtifact] = []
    latency = _latency_bars(metrics)
    if latency:
        artifacts.append(
            write_bar_chart(output / "latency_breakdown.svg", title="Latency Breakdown", bars=latency, unit="ms")
        )
    tokens = _token_bars(metrics)
    if tokens:
        artifacts.append(
            write_bar_chart(output / "token_patch_budget.svg", title="Token / Patch Budget", bars=tokens, unit="tokens")
        )
    memory = _memory_bars(metrics)
    if memory:
        artifacts.append(write_bar_chart(output / "memory_peaks.svg", title="Memory Peaks", bars=memory, unit="bytes"))
    return artifacts


def _group(metrics: dict[str, Any], name: str) -> dict[str, Any]:
    value = metrics.get(name)
    return value if isinstance(value, dict) else {}


def _metric(group: dict[str, Any], names: tuple[str, ...], mode: str | None = None) -> float | None:
    for name in names:
        value = group.get(name)
        if mode is not None and isinstance(value, dict):
            number = numeric_or_none(value.get(mode))
            if number is not None:
                return number
        number = first_number(value)
        if mode is None and number is not None:
            return number
    return None


def _has_mode_comparison(group: dict[str, Any]) -> bool:
    return any(isinstance(value, dict) and ("keep_all" in value or "autogaze" in value) for value in group.values())


def _latency_bars(metrics: dict[str, Any]) -> list[ChartBar]:
    latency = _group(metrics, "latency_ms")
    if not latency:
        return []
    modes = ("keep_all", "autogaze") if _has_mode_comparison(latency) else (None,)
    bars: list[ChartBar] = []
    for mode in modes:
        label = mode or "current"
        total = _metric(latency, ("total_ms", "total", "total_median"), mode)
        segments = [
            ("Pre(no AG)", _metric(latency, ("preprocess_without_autogaze_ms", "preprocess_without_autogaze_median", "preprocess_total_ms", "preprocess_total_median"), mode)),
            ("AutoGaze", _metric(latency, ("autogaze_total_ms", "autogaze_ms", "autogaze_total_median", "autogaze_median", "gazing_info_total_ms"), mode)),
            ("ViT", _metric(latency, ("vit_encoder_ms", "vision_encoder_ms", "vit_encoder_median", "qwen_vit_prepare", "qwen_vit_prepare_ms"), mode)),
            ("LLM", _metric(latency, ("llm_ms", "llm_median", "generate", "generate_ms"), mode)),
        ]
        chart_segments = [ChartSegment(name, value) for name, value in segments if value is not None and value > 0]
        known = sum(segment.value for segment in chart_segments)
        if total is not None and total > known:
            chart_segments.append(ChartSegment("Other", total - known))
        if not chart_segments and total is not None:
            chart_segments.append(ChartSegment("total", total))
        if chart_segments:
            bars.append(ChartBar(str(label), chart_segments))
    return bars


def _token_bars(metrics: dict[str, Any]) -> list[ChartBar]:
    tokens = _group(metrics, "tokens")
    if not tokens:
        return []
    specs = [
        (
            "Dense ref",
            (
                "single_scale_dense_siglip_reference_patch_tokens",
                "single_scale_dense_vision_budget.total_patch_tokens",
            ),
        ),
        (
            "Full patch",
            (
                "hd_multiscale_keep_all_patch_tokens",
                "raw_vit_patch_tokens_before_selector",
                "encoder_patch_tokens_before_keep_all_or_raw",
                "visual_tokens_before_prune",
            ),
        ),
        (
            "Selected patch",
            (
                "autogaze_selected_total_patch_tokens",
                "encoder_input_patch_tokens_after_autogaze",
                "encoder_patch_tokens_after_autogaze",
                "visual_tokens_after_prune",
            ),
        ),
        (
            "LLM visual",
            (
                "llm_visual_tokens_actual_from_budget",
                "llm_visual_tokens_after_actual",
                "llm_context_tokens",
            ),
        ),
    ]
    bars: list[ChartBar] = []
    for label, names in specs:
        value = _metric(tokens, names)
        if value is not None and value > 0:
            bars.append(ChartBar(label, [ChartSegment(label, value)]))
    for label, value in tokens.items():
        if not isinstance(value, dict):
            continue
        before = first_number(value.get("before_keep_all_estimated"), value.get("before_keep_all_or_raw"), value.get("keep_all"))
        after = first_number(value.get("after_autogaze_actual"), value.get("after_autogaze"), value.get("autogaze"))
        if before is not None and after is not None:
            bars.append(ChartBar(f"{label}:before", [ChartSegment("Full patch", before)]))
            bars.append(ChartBar(f"{label}:after", [ChartSegment("Selected patch", after)]))
    return bars


def _memory_bars(metrics: dict[str, Any]) -> list[ChartBar]:
    memory = _group(metrics, "memory_bytes")
    if not memory:
        return []
    specs = [
        ("Processor", ("processor_peak", "processor_peak_median")),
        ("AutoGaze", ("autogaze_peak", "autogaze_tile_tensor_peak_per_temporal_chunk")),
        ("ViT", ("siglip_gazed_hidden_peak", "vision_encoder_peak")),
        ("LLM", ("llm_peak", "llm_peak_median", "peak_cuda_allocated")),
        ("Overall", ("overall_peak", "overall_peak_median", "peak_cuda_reserved")),
    ]
    bars: list[ChartBar] = []
    for label, names in specs:
        value = _metric(memory, names)
        if value is not None and value > 0:
            bars.append(ChartBar(label, [ChartSegment(label, value)]))
    return bars
