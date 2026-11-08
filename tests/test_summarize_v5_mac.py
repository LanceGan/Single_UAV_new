import json

from summarize_v5_mac import summarize_run


def _write_run(root, *, episodes, history, best, config=None):
    run = root / "E3" / "visible" / "baseline" / "medium" / "td3_bc"
    run.mkdir(parents=True)
    payload = {
        "episodes": episodes,
        "algorithm": "td3_bc",
        "data_level": "medium",
        "sequence_method": "distance_tsp",
        "a2g_blind": False,
        "qos_enforcement": "continue",
        "outage_max_s": 6.0,
        "payloads_mb": [180.0] * 5,
        "order": [1, 6, 5, 2, 3, 4],
    }
    payload.update(config or {})
    (run / "run_config.json").write_text(json.dumps(payload), encoding="utf-8")
    (run / "eval_history.json").write_text(
        json.dumps({"eval_best": best, "history": history}),
        encoding="utf-8",
    )
    return run


def test_completed_feasible_run(tmp_path):
    best = {
        "episode": 50,
        "success": True,
        "visited_inspection_points": 6,
        "uploaded_mb": 900.0,
        "task_time_s": 200.0,
        "transmission_delay_s": 100.0,
        "outage_time_s": 4.0,
        "outage_limit_s": 6.0,
    }
    run = _write_run(
        tmp_path,
        episodes=100,
        history=[best, {**best, "episode": 100}],
        best=best,
    )
    record = summarize_run(run, tmp_path)
    assert record["status"] == "completed"
    assert record["completed_evaluations"] == 2
    assert record["feasible_evaluations"] == 2
    assert record["objective_s"] == 300.0
    assert record["eval_best_qos_feasible"] is True


def test_running_run_does_not_claim_completion(tmp_path):
    best = {
        "episode": 25,
        "success": False,
        "visited_inspection_points": 6,
        "uploaded_mb": 900.0,
        "task_time_s": 250.0,
        "transmission_delay_s": 125.0,
        "outage_time_s": 7.0,
        "outage_limit_s": 6.0,
    }
    run = _write_run(tmp_path, episodes=100, history=[best], best=best)
    record = summarize_run(run, tmp_path)
    assert record["status"] == "running"
    assert record["completed_evaluations"] == 1
    assert record["feasible_evaluations"] == 0
    assert record["eval_best_qos_feasible"] is False
