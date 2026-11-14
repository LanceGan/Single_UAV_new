from plot_v5_e1_training_validation import (
    LEVELS,
    METHODS,
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


def test_histories_have_four_executors_at_all_levels():
    rewards, evaluations, selected = load_histories()
    assert set(METHODS) == {"CA-TD3", "D3QN", "PPO", "SAC"}
    assert set(LEVELS) == {"Low", "Medium", "High"}
    assert len(rewards) == 3 * 4 * 3000
    assert len(evaluations) == 3 * 4 * 120
    assert len(selected) == 3 * 4
    assert "DDPG" not in {row["method"] for row in rewards}


def test_final_e1_table_uses_actor_replay_for_sac_medium():
    with (P3_SOURCE / "e1_trajectory_results_v5.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))

    assert len(rows) == 12
    assert {row["method"] for row in rows} == {
        "CA-TD3",
        "D3QN",
        "PPO",
        "SAC",
    }
    sac_medium = next(
        row
        for row in rows
        if row["method"] == "SAC" and row["level"] == "Medium"
    )
    assert float(sac_medium["transmission_delay_s"]) == 328.0
    assert float(sac_medium["objective_s"]) == 1328.0
    assert float(sac_medium["contiguous_outage_s"]) == 20.5
    assert sac_medium["complete"] == "True"
    assert sac_medium["feasible"] == "False"
