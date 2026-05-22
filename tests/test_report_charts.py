from repro.report_charts import ChartBar, ChartSegment, write_bar_chart
from repro.report_charts import build_standard_report_charts, latency_attribution_bars, latency_stage_bars, shorten_label


def test_write_bar_chart_creates_self_contained_svg(tmp_path):
    chart_path = tmp_path / "latency.svg"

    artifact = write_bar_chart(
        chart_path,
        title="Latency Stack",
        bars=[
            ChartBar(
                label="keep_all",
                segments=[
                    ChartSegment("preprocess", 1000.0),
                    ChartSegment("generate", 4000.0),
                ],
            ),
            ChartBar(
                label="autogaze",
                segments=[
                    ChartSegment("preprocess", 900.0),
                    ChartSegment("autogaze", 300.0),
                    ChartSegment("generate", 2000.0),
                ],
            ),
        ],
        unit="ms",
    )

    svg = chart_path.read_text()
    assert artifact.path == chart_path
    assert artifact.title == "Latency Stack"
    assert "<svg" in svg
    assert "Latency Stack" in svg
    assert "keep_all" in svg
    assert "autogaze" in svg
    assert "preprocess" in svg


def test_standard_latency_chart_uses_stable_readable_stage_colors(tmp_path):
    artifacts = build_standard_report_charts(
        metrics={
            "latency_ms": {
                "total_ms": {"keep_all": 7000, "autogaze": 5300},
                "preprocess_without_autogaze_ms": {"keep_all": 1000, "autogaze": 1000},
                "video_decode_ms": {"keep_all": 300, "autogaze": 300},
                "preprocess_total_ms": {"keep_all": 1000, "autogaze": 1800},
                "autogaze_total_ms": {"keep_all": 0, "autogaze": 800},
                "vit_encoder_ms": {"keep_all": 2000, "autogaze": 500},
                "llm_ms": {"keep_all": 4000, "autogaze": 3000},
            }
        },
        output_dir=tmp_path,
    )

    latency_svg = next(artifact.path for artifact in artifacts if artifact.path.name == "latency_breakdown.svg")
    svg = latency_svg.read_text()
    assert "Decode/read" in svg
    assert "Prep rest" in svg
    assert "Pre(no AG)" not in svg
    assert "AutoGaze" in svg
    assert "ViT" in svg
    assert "LLM" in svg
    assert "#5b8def" in svg
    assert "#15aabf" in svg
    assert "#f59f00" in svg
    assert "#2f9e44" in svg
    assert "#7048e8" in svg
    assert "preprocess_total_ms" not in svg


def test_standard_charts_include_wall_clock_and_attribution_latency_views(tmp_path):
    metrics = {
        "latency_ms": {
            "total_ms": {"keep_all": 7000, "autogaze": 5300},
            "video_decode_read_ms": {"keep_all": 300, "autogaze": 300},
            "video_frame_resize_ms": {"keep_all": 100, "autogaze": 100},
            "video_tiling_ms": {"keep_all": 600, "autogaze": 600},
            "selector_input_build_ms": {"keep_all": 0, "autogaze": 50},
            "autogaze_total_ms": {"keep_all": 0, "autogaze": 800},
            "vit_encoder_ms": {"keep_all": 2000, "autogaze": 500},
            "mm_projector_ms": {"keep_all": 200, "autogaze": 80},
            "llm_ms": {"keep_all": 3800, "autogaze": 2870},
        }
    }

    stage = latency_stage_bars(metrics)
    attribution = latency_attribution_bars(metrics)
    artifacts = build_standard_report_charts(metrics=metrics, output_dir=tmp_path)

    assert stage[1].segments == [
        ChartSegment("Decode/read", 300.0),
        ChartSegment("Frame resize", 100.0),
        ChartSegment("Tile/tensor prep", 600.0),
        ChartSegment("Selector input", 50.0),
        ChartSegment("AutoGaze", 800.0),
        ChartSegment("ViT", 500.0),
        ChartSegment("Projector", 80.0),
        ChartSegment("LLM", 2870.0),
    ]
    assert attribution[1].segments == [
        ChartSegment("Video I/O", 300.0),
        ChartSegment("Pre-model prep", 700.0),
        ChartSegment("AutoGaze pipeline", 850.0),
        ChartSegment("Vision pipeline", 580.0),
        ChartSegment("MLLM pipeline", 2870.0),
    ]
    assert (tmp_path / "latency_breakdown.svg").is_file()
    assert (tmp_path / "latency_attribution.svg").is_file()
    assert any(artifact.title == "Latency Breakdown (Wall Clock)" for artifact in artifacts)
    assert any(artifact.title == "Latency Attribution" for artifact in artifacts)
    attribution_svg = (tmp_path / "latency_attribution.svg").read_text()
    assert "AutoGaze pipeline" in attribution_svg
    assert "Vision pipeline" in attribution_svg


def test_standard_latency_chart_does_not_show_inclusive_preprocess_as_no_ag(tmp_path):
    artifacts = build_standard_report_charts(
        metrics={
            "latency_ms": {
                "total_median": 9000,
                "preprocess_total_median": 3000,
                "autogaze_total_median": 800,
                "vit_encoder_median": 1200,
                "llm_median": 4000,
            }
        },
        output_dir=tmp_path,
    )

    latency_svg = next(artifact.path for artifact in artifacts if artifact.path.name == "latency_breakdown.svg")
    svg = latency_svg.read_text()
    assert "Pre(no AG)" not in svg
    assert "AutoGaze" in svg


def test_shorten_label_keeps_svg_labels_compact():
    label = shorten_label("qwen_chunked_vit_autogaze_sparse/128f/3840x2160", max_chars=24)

    assert label == "qwen_chunked_v...3840x2160"
