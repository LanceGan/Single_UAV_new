from plot_v5_e1_trajectories import evaluate_all


def test_all_e1_low_actor_replays_match_eval_best():
    results = evaluate_all()
    assert set(results) == {"CA-TD3", "D3QN", "PPO", "DDPG", "SAC"}
    assert results["CA-TD3"]["report"]["task_time_s"] == 169.0
    assert results["D3QN"]["report"]["task_time_s"] == 175.0
    assert results["PPO"]["report"]["task_time_s"] == 190.0
    assert results["SAC"]["report"]["task_time_s"] == 170.0
    assert results["DDPG"]["report"]["visited_inspection_points"] == 4
