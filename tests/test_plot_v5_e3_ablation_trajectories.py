from __future__ import annotations

import json
from pathlib import Path

import numpy as np


P3_FIGURES = (
    Path(__file__).resolve().parents[1] / ".." / "P3" / "figs" / "v5"
).resolve()
SOURCE = P3_FIGURES / "source_data" / "e3_a2g_trajectories_v5"


def test_e3_trajectory_source_contains_all_six_replays():
    payload = json.loads(SOURCE.with_suffix(".json").read_text(encoding="utf-8"))
    records = payload["records"]

    assert len(records) == 6
    assert {
        (record["level"], record["method"]) for record in records
    } == {
        (level, method)
        for level in ("Low", "Medium", "High")
        for method in ("A2G-blind", "A2G-visible")
    }
    high_blind = next(
        record
        for record in records
        if record["level"] == "High" and record["method"] == "A2G-blind"
    )
    assert high_blind["complete"] is False
    assert high_blind["visited_points"] == 2
    assert high_blind["contiguous_outage_s"] == 585.0


def test_e3_panels_are_independent_assets():
    for panel in "abc":
        for suffix in (".pdf", ".svg", ".png", ".tiff"):
            path = P3_FIGURES / f"e3_a2g_trajectories_v5_panel_{panel}{suffix}"
            assert path.is_file()
            assert path.stat().st_size > 0


def test_e3_current_outage_trace_resets_in_normal_regions():
    with np.load(SOURCE.with_suffix(".npz")) as payload:
        current = np.asarray(payload["high_blind_current_outage_s"], dtype=float)
        maximum = np.asarray(payload["high_blind_outage_s"], dtype=float)

    assert np.any(current == 0.0)
    assert np.any(np.diff(current) < 0.0)
    assert np.isclose(np.max(maximum), 585.0)
