import json
from pathlib import Path

import numpy as np

from evaluate_qos_robustness import (
    _point_from_reports,
    actor_snapshots,
    load_astar_points,
    parse_budgets,
    parse_run,
    plot_robustness,
)


def test_parse_and_discover(tmp_path):
    label, run_dir = parse_run(f"w= {tmp_path}".replace("= ", "="))
    assert label == "w"
    assert run_dir == tmp_path
    assert parse_budgets(["30,20", "25"]) == [20.0, 25.0, 30.0]
    models = tmp_path / "models"
    models.mkdir()
    for episode in (50, 100):
        (models / f"td3_bc_td3_actor{episode}.pth").touch()
    (models / "td3_bc_td3_actoreval_best.pth").touch()
    assert [episode for episode, _ in actor_snapshots(tmp_path)] == [50, 100]


def _report(success, task=150.0, delay=80.0, outage=20.0):
    return {
        "success": success,
        "termination_reason": "success" if success else "qos_violation",
        "visited_inspection_points": 6 if success else 5,
        "uploaded_mb": 450.0 if success else 300.0,
        "task_time_s": task,
        "transmission_delay_s": delay,
        "outage_time_s": outage,
    }


def test_point_does_not_rank_infeasible_as_feasible():
    reports = [
        {"episode": 50, "checkpoint": Path("a"), "report": _report(False, 80, 40, 25.5)},
        {"episode": 100, "checkpoint": Path("b"), "report": _report(True, 140, 70, 24.5)},
    ]
    point = _point_from_reports(reports, 25, 0.8, 0.2)
    assert point["success"] is True
    assert point["best_feasible_episode"] == 100
    assert point["objective_s"] == 252.0

    point = _point_from_reports(reports[:1], 25, 0.8, 0.2)
    assert point["success"] is False
    assert point["objective_s"] is None
    assert point["snapshot_success_rate"] == 0.0


def test_load_astar_and_plot(tmp_path):
    metrics_dir = tmp_path / "astar" / "low"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "metrics.json").write_text(json.dumps({
        "planner": {"success": True, "status": "success"},
        "world_replay": {"success": True, "objective_s": 214.0, "outage_time_s": 27.5},
    }))
    astar = load_astar_points([f"30={tmp_path / 'astar'}"])
    assert astar[0]["objective_s"] == 214.0
    runs = [{
        "objective_w1": 0.5,
        "objective_w2": 0.5,
        "points": [{
            "budget_s": 30.0,
            "objective_s": 235.5,
            "outage_s": 21.5,
            "snapshot_success_rate": 0.5,
        }],
    }]
    plot_robustness(runs, astar, tmp_path / "figure", "test")
    assert (tmp_path / "figure.png").is_file()
    assert (tmp_path / "figure.pdf").is_file()
