from repro.report_charts import ChartBar, ChartSegment, write_bar_chart


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
