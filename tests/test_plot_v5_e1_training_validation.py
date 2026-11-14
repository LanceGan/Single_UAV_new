from plot_v5_e1_training_validation import (
    LEVELS,
    METHODS,
    TRAJECTORY_METHODS,
    load_histories,
    smooth_series,
)
import csv
from pathlib import Path


P3_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "P3"
    / "figs"
    / "v5"
    / "source_data"
)


def test_moving_average_preserves_leading_missing_values():
    result = smooth_series([1, 2, 3, 4], 2)
    assert result[0] != result[0]
    assert result[1:].tolist() == [1.5, 2.5, 3.5]


def test_histories_have_three_learned_executors_and_four_final_rows():
    rewards, evaluations, selected = load_histories()
    assert set(METHODS) == {"CA-TD3", "D3QN", "PPO"}
    assert set(TRAJECTORY_METHODS) == {
        "CA-TD3",
        "D3QN",
        "PPO",
        "Dual-map A*",
    }
    assert set(LEVELS) == {"Low", "Medium", "High"}
    assert len(rewards) == 3 * 3 * 3000
    assert len(evaluations) == 3 * 3 * 120
    assert len(selected) == 3 * 4
    assert "DDPG" not in {row["method"] for row in rewards}
    assert "SAC" not in {row["method"] for row in rewards}
    assert "Dual-map A*" in {row["method"] for row in selected}


def test_final_e1_table_uses_dual_map_astar_instead_of_sac():
    with (P3_SOURCE / "e1_trajectory_results_v5.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))

    assert len(rows) == 12
    assert {row["method"] for row in rows} == {
        "CA-TD3",
        "D3QN",
        "PPO",
        "Dual-map A*",
    }
    astar_medium = next(
        row
        for row in rows
        if row["method"] == "Dual-map A*" and row["level"] == "Medium"
    )
    assert float(astar_medium["task_time_s"]) == 189.5
    assert float(astar_medium["transmission_delay_s"]) == 127.0
    assert float(astar_medium["objective_s"]) == 316.5
    assert float(astar_medium["contiguous_outage_s"]) == 4.0
    assert astar_medium["complete"] == "True"
    assert astar_medium["feasible"] == "True"
