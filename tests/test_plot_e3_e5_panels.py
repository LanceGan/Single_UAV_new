from __future__ import annotations

import importlib.util
from pathlib import Path


P3_FIGURES = (
    Path(__file__).resolve().parents[1] / ".." / "P3" / "figs" / "v5"
).resolve()


def _load(name: str):
    path = P3_FIGURES / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _assert_panel_bundle(stem: Path, panel: str) -> None:
    for suffix in (".pdf", ".svg", ".png", ".tiff"):
        path = Path(f"{stem}_panel_{panel}{suffix}")
        assert path.is_file()
        assert path.stat().st_size > 0
        if suffix == ".pdf":
            assert path.read_bytes().startswith(b"%PDF")


def test_e3_exports_three_standalone_panels():
    module = _load("plot_e3_a2g_paired_v5.py")
    module.save_panel_files(module.load_rows())
    stem = module.OUTPUT
    for panel in ("a", "b", "c"):
        _assert_panel_bundle(stem, panel)


def test_e5_exports_matrix_and_decomposition_as_standalone_panels():
    module = _load("plot_e5_method_payload_matrix.py")
    module.save_panel_files(module.load_rows())
    stem = module.OUTPUT
    for panel in ("a", "b"):
        _assert_panel_bundle(stem, panel)
