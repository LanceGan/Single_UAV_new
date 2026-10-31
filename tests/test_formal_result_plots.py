import numpy as np


def test_evaluation_series_masks_failed_objectives():
    from plot_formal_results import evaluation_series

    history = {
        "history": [
            {"episode": 25, "success": False, "task_time_s": 40.0,
             "transmission_delay_s": 5.0, "total_return": -10.0,
             "visited_inspection_points": 2},
            {"episode": 50, "success": True, "task_time_s": 100.0,
             "transmission_delay_s": 30.0, "total_return": 5.0,
             "visited_inspection_points": 6},
        ]
    }

    series = evaluation_series(history)

    np.testing.assert_allclose(series["objective_raw"], [45.0, 130.0])
    assert np.isnan(series["objective_feasible"][0])
    assert series["objective_feasible"][1] == 130.0
    np.testing.assert_array_equal(series["success"], [False, True])


def test_moving_average_uses_window_ending_episode():
    from plot_formal_results import moving_average

    episodes, averages = moving_average(np.array([1.0, 2.0, 3.0, 4.0]), window=2)

    np.testing.assert_array_equal(episodes, [2, 3, 4])
    np.testing.assert_allclose(averages, [1.5, 2.5, 3.5])


def test_moving_average_handles_short_series():
    from plot_formal_results import moving_average

    episodes, averages = moving_average(np.array([2.0, 4.0]), window=5)

    assert episodes.size == 0
    assert averages.size == 0


def test_trajectory_title_exposes_failed_rollout_status():
    from plot_formal_results import trajectory_title

    title = trajectory_title("DDPG-BC, Low load", 900, {
        "success": False, "termination_reason": "timeout"})

    assert "episode 900" in title
    assert "infeasible: timeout" in title
