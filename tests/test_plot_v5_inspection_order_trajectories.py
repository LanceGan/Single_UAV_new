from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIGURES = (ROOT.parent / "P3" / "figs" / "v5").resolve()
SOURCE = FIGURES / "source_data" / "e2_inspection_order_trajectories_v5"


def test_inspection_order_has_six_final_methods_and_three_levels():
    payload = json.loads(SOURCE.with_suffix(".json").read_text(encoding="utf-8"))
    records = payload["records"]

    assert len(records) == 18
    assert {
        record["method"] for record in records
    } == {
        "Proposed PRIGA",
        "Yang-style",
        "EEGA-style GA",
        "DRL-SA (G2A-aware SA)",
        "SA-D3QN (distance-only SA)",
        "EEAP-PSO",
    }
    assert {
        record["level"] for record in records
    } == {"Low", "Medium", "High"}
    assert all(
        sorted(record["inspection_order"]) == [1, 2, 3, 4, 5, 6]
        for record in records
    )


def test_inspection_order_trajectories_are_independent_panels():
    with np.load(SOURCE.with_suffix(".npz")) as payload:
        assert len(
            [key for key in payload.files if key.endswith("_positions_m")]
        ) == 18
    for panel in "abcdef":
        assert (
            FIGURES / f"e2_inspection_order_trajectories_v5_panel_{panel}.pdf"
        ).is_file()


def test_display_panels_replace_drl_sa_with_distance_tsp_reference():
    display = json.loads(
        (
            FIGURES
            / "source_data"
            / "e2_trajectory_panels_v5.json"
        ).read_text(encoding="utf-8")
    )
    assert display["protocol"]["display_methods"] == [
        "Proposed PRIGA",
        "Yang-style",
        "EEGA-style GA",
        "Distance-TSP+CA-TD3",
        "SA-D3QN (distance-only SA)",
        "EEAP-PSO",
    ]
    assert "DRL-SA (G2A-aware SA)" in display["protocol"]["table_methods"]
    assert "DRL-SA (G2A-aware SA)" not in display["protocol"]["display_methods"]
    assert {
        record["method"] for record in display["records"]
    } == set(display["protocol"]["display_methods"])
    distance = [
        record
        for record in display["records"]
        if record["method"] == "Distance-TSP+CA-TD3"
    ]
    assert {record["level"] for record in distance} == {
        "Low",
        "Medium",
        "High",
    }
    assert any(not record["complete"] for record in distance)
