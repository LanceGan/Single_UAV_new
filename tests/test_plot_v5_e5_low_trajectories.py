from plot_v5_e5_low_trajectories import evaluate_all


def test_all_e5_low_actor_replays_match_recorded_metrics():
    results = evaluate_all()
    assert set(results) == {
        "DRM-HO",
        "MCO-A* adaptation",
        "EEGA+TD3",
        "Distance-TSP+TD3",
        "Yang-style+TD3",
        "DRL-SA",
        "SA-D3QN",
    }
    assert results["DRM-HO"]["report"]["task_time_s"] == 169.0
    assert results["MCO-A* adaptation"]["report"]["task_time_s"] == 287.5
    assert results["DRL-SA"]["report"]["task_time_s"] == 219.5
    assert results["SA-D3QN"]["report"]["task_time_s"] == 200.0
