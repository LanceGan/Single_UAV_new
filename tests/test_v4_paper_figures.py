import json
from pathlib import Path

import pytest

import plot_v4_paper_figures as figures


def test_v4_dataset_uses_all_qos_runs_and_keeps_ppo_as_failure():
    dataset = figures.build_dataset()

    assert len(dataset["qos"]) == 15
    assert {row["budget_s"] for row in dataset["qos"]} == {3.0, 4.0, 5.0, 6.0, 8.0}
    assert all(row["best_feasible"] is not None for row in dataset["qos"])

    ppo = next(row for row in dataset["layer1"] if row["label"] == "PPO")
    assert ppo["best_feasible"] is None
    assert ppo["feasible_evaluations"] == 0
    assert ppo["selected_evaluation"]["success"] is False


def test_v4_dataset_reports_expected_main_budget_results():
    dataset = figures.build_dataset()
    weights = {row["weight_id"]: row for row in dataset["weights_at_qos6"]}
    pure_astar = next(row for row in dataset["layer1"] if row["label"] == "A*")

    assert weights["w80-20"]["best_feasible"]["objective_s"] == pytest.approx(244.5)
    assert weights["w50-50"]["best_feasible"]["objective_s"] == pytest.approx(244.5)
    assert weights["w20-80"]["best_feasible"]["objective_s"] == pytest.approx(250.0)
    assert all(row["best_feasible"]["outage_s"] <= 6.0 for row in weights.values())
    assert pure_astar["deterministic"] is True
    assert pure_astar["best_feasible"] is None
    assert pure_astar["selected_evaluation"]["outage_time_s"] == pytest.approx(6.5)


def test_v4_dataset_keeps_pure_heuristic_failures_out_of_rankings():
    dataset = figures.build_dataset()
    pure_pso = next(row for row in dataset["layer1"] if row["label"] == "PSO")

    assert pure_pso["best_feasible"] is None
    assert pure_pso["feasible_evaluations"] == 0
    assert pure_pso["selected_evaluation"]["termination_reason"] == "qos_violation"
    assert pure_pso["selected_evaluation"]["visited_inspection_points"] == 1


def test_v4_dataset_includes_adapted_mco_astar_and_high_timeout():
    dataset = figures.build_dataset()
    mco = {row["level"]: row for row in dataset["load"]["mco_astar"]}

    assert mco["low"]["best_feasible"]["objective_s"] == pytest.approx(457.5)
    assert mco["medium"]["best_feasible"]["objective_s"] == pytest.approx(789.0)
    assert mco["high"]["best_feasible"] is None
    assert mco["high"]["selected_evaluation"]["visited_inspection_points"] == 5


def test_load_schema_validation_rejects_legacy_cumulative_outage(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "state_schema": "legacy-cumulative",
                "qos_constraint": "cumulative",
                "outage_max_s": 6,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "eval_history.json").write_text(
        json.dumps({"eval_best": {"episode": 1}, "history": [{"success": True}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="v4-max-contiguous"):
        figures.load_run(run_dir, "legacy", 6)
