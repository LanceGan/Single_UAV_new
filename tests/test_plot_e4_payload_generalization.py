from __future__ import annotations

import csv
import json
from pathlib import Path


SOURCE = (
    Path(__file__).resolve().parents[1]
    / ".."
    / "P3"
    / "figs"
    / "v5"
    / "source_data"
)
FIGURE = SOURCE.parent


def _csv_rows(name):
    with (SOURCE / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_payload_qos_table_covers_three_levels_and_five_budgets():
    rows = _csv_rows("e4_qos_table_v5.csv")

    assert len(rows) == 15
    assert {
        (row["level"], int(float(row["qos_budget_s"])))
        for row in rows
    } == {
        (level, budget)
        for level in ("Low", "Medium", "High")
        for budget in (3, 4, 5, 6, 8)
    }
    assert all(
        row["contiguous_outage_s"]
        and row["complete"] in {"True", "False"}
        for row in rows
    )


def test_payload_weight_table_covers_three_levels_and_three_weights():
    rows = _csv_rows("e4_weight_table_v5.csv")

    assert len(rows) == 9
    assert {
        (row["level"], row["w1"], row["w2"])
        for row in rows
    } == {
        (level, first, second)
        for level in ("Low", "Medium", "High")
        for first, second in (
            ("0.8", "0.2"),
            ("0.5", "0.5"),
            ("0.2", "0.8"),
        )
    }


def test_low_qos_and_weight_trajectory_panels_are_independent_artifacts():
    stems = (
        FIGURE / "e4_low_qos_trajectories_v5_panel_a",
        FIGURE / "e4_low_weight_trajectories_v5_panel_b",
    )
    for stem in stems:
        for suffix in (".pdf", ".svg", ".png", ".tiff"):
            path = stem.with_suffix(suffix)
            assert path.is_file() and path.stat().st_size > 0


def test_spatial_panels_use_only_low_tier_and_matching_maps():
    payload = json.loads(
        (
            SOURCE / "e4_low_generalization_trajectories_v5.json"
        ).read_text(encoding="utf-8")
    )
    rows = payload["records"]
    assert len(rows) == 8
    assert {row["level"] for row in rows} == {"Low"}
    assert {
        (row["panel"], row["map_background"]) for row in rows
    } == {
        ("a", "G2A coverage probability"),
        ("b", "A2G SNR"),
    }


def test_payload_generalization_sources_keep_all_rollout_statuses():
    payload = json.loads(
        (SOURCE / "e4_qos_table_v5.json").read_text(encoding="utf-8")
    )
    assert payload["protocol"]["levels"] == ["Low", "Medium", "High"]
    assert payload["protocol"]["qos_budget_s"] == [3, 4, 5, 6, 8]
    assert any(not row["complete"] for row in payload["records"])
    assert any(
        row["complete"] and not row["feasible"]
        for row in payload["records"]
    )


def test_incomplete_e4_runs_record_horizon_vs_qos_failure_mode():
    qos = json.loads(
        (SOURCE / "e4_qos_table_v5.json").read_text(encoding="utf-8")
    )
    medium_qos4 = next(
        row
        for row in qos["records"]
        if row["level"] == "Medium" and row["qos_budget_s"] == 4.0
    )
    assert medium_qos4["termination_reason"] == "timeout"
    assert medium_qos4["horizon_exhausted"] is True
    assert (
        medium_qos4["failure_mode"]
        == "horizon_exhausted_constraint_satisfied"
    )

    weights = json.loads(
        (SOURCE / "e4_weight_table_v5.json").read_text(encoding="utf-8")
    )
    incomplete = [
        row for row in weights["records"] if not row["complete"]
    ]
    assert len(incomplete) == 3
    assert all(row["horizon_exhausted"] for row in incomplete)
    assert all(
        row["failure_mode"] == "horizon_exhausted_with_qos_violation"
        for row in incomplete
    )
