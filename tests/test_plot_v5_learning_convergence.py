from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / ".."
    / "P3"
    / "figs"
    / "v5"
    / "plot_v5_learning_convergence.py"
).resolve()
SPEC = importlib.util.spec_from_file_location("plot_v5_learning_convergence", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_e1_histories_cover_five_algorithms_and_three_tiers():
    data, rows, selected = MODULE.build_e1_data()
    assert len(data) == 5
    assert all(set(levels) == set(MODULE.LEVELS) for levels in data.values())
    assert rows
    assert len(selected) == 15


def test_e5_histories_cover_six_learned_pipelines_and_three_tiers():
    data, rows, selected = MODULE.build_e5_data()
    assert len(data) == 6
    assert all(set(levels) == set(MODULE.LEVELS) for levels in data.values())
    assert rows
    assert len(selected) == 18


def test_invalid_task_times_are_not_treated_as_objectives():
    data, _, _ = MODULE.build_e1_data()
    for levels in data.values():
        for records in levels.values():
            for record in records:
                if not record["complete"]:
                    assert record["task_time_s"] is None
                    assert record["objective_s"] is None


def test_e1_and_e5_have_independent_panel_files():
    figure_dir = SCRIPT.parent
    stems = [
        *(f"e1_trajectory_convergence_v5_panel_{panel}" for panel in "abcdef"),
        "e1_trajectory_convergence_v5_legend",
        *(f"e5_end_to_end_convergence_v5_panel_{panel}" for panel in "abc"),
        "e5_end_to_end_convergence_v5_legend",
    ]
    for stem in stems:
        for suffix in (".pdf", ".svg", ".png", ".tiff"):
            path = figure_dir / f"{stem}{suffix}"
            assert path.is_file() and path.stat().st_size > 0


def test_e1_protocol_records_connected_raw_observations():
    payload = json.loads(
        (
            SCRIPT.parent
            / "source_data"
            / "e1_trajectory_convergence_v5.json"
        ).read_text(encoding="utf-8")
    )
    protocol = payload["protocol"]

    assert "observed task time" in protocol["panel_left"]
    assert "connected in episode order" in protocol["panel_left"]
    assert protocol["lines_break_at_missing_evaluations"] is True
