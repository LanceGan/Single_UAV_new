from __future__ import annotations

import importlib.util
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
