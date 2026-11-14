from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / ".."
    / "P3"
    / "figs"
    / "v5"
    / "plot_e4_realtime_comm.py"
)
spec = importlib.util.spec_from_file_location("plot_e4_realtime_comm", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_qos_realtime_series_cover_all_tiers_and_settings():
    series = module._load_series("qos")
    assert len(series) == 15
    assert {(item["level"], item["setting"]) for item in series} == {
        (level, f"{setting} s")
        for level in ("Low", "Medium", "High")
        for setting in (3, 4, 5, 6, 8)
    }
    assert all(item["outage_probability"].shape == item["time_s"].shape for item in series)
    assert all(
        item["contiguous_outage_s"].shape == item["time_s"].shape
        for item in series
    )


def test_weight_realtime_series_cover_all_tiers_and_settings():
    series = module._load_series("weight")
    assert len(series) == 9
    assert {(item["level"], item["setting"]) for item in series} == {
        (level, weight)
        for level in ("Low", "Medium", "High")
        for weight in ("0.8/0.2", "0.5/0.5", "0.2/0.8")
    }


def test_realtime_figures_have_separate_panel_assets():
    root = module.HERE
    for stem in (
        "e4_qos_realtime_comm_v5",
        "e4_weight_realtime_comm_v5",
    ):
        for suffix in ("a", "b", "c", "d", "e", "f"):
            path = root / f"{stem}_panel_{suffix}.pdf"
            assert path.is_file()
        assert (root / f"{stem}_legend.pdf").is_file()
