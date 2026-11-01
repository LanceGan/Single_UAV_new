from pathlib import Path


def test_build_framework_figure_exports_publication_artifacts(tmp_path: Path):
    from plot_algorithm_framework import build_framework_figure

    artifacts = build_framework_figure(tmp_path)

    assert set(artifacts) == {"svg", "pdf", "png"}
    assert artifacts["svg"].read_text(encoding="utf-8").startswith("<?xml")
    assert artifacts["pdf"].read_bytes().startswith(b"%PDF")
    assert artifacts["png"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert all(path.stat().st_size > 10_000 for path in artifacts.values())
    assert not any(line.endswith(" ") for line in artifacts["svg"].read_text(encoding="utf-8").splitlines())


def test_framework_figure_contains_required_method_semantics(tmp_path: Path):
    from plot_algorithm_framework import build_framework_figure

    artifacts = build_framework_figure(tmp_path)
    svg = artifacts["svg"].read_text(encoding="utf-8")

    required_labels = (
        "G2A outage",
        "A2G uplink-SNR",
        "Heterogeneous payloads",
        "Dual-map-aware IGA",
        "Optimized inspection order",
        "Twin critics",
        "Training only",
        "Guide-free Actor",
        "T +",
        "T_out",
    )
    assert all(label in svg for label in required_labels)


def test_framework_uses_compact_labels_for_ieee_column_width(tmp_path: Path):
    from plot_algorithm_framework import build_framework_figure

    svg = build_framework_figure(tmp_path)["svg"].read_text(encoding="utf-8")

    assert "System inputs" in svg
    assert "Map-aware TD3" in svg
    assert "Mission output" in svg
    assert "System and task inputs" not in svg
    assert "Map-aware TD3 trajectory optimization" not in svg
    assert "Optimized mission" not in svg
    assert "Guide-free Actor (evaluation)" in svg
    assert "Guide-free Actor at evaluation" not in svg
