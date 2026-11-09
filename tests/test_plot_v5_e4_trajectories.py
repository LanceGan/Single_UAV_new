from __future__ import annotations

from pathlib import Path

from plot_v5_e4_trajectories import (
    QOS_VALUES_S,
    WEIGHT_VALUES,
    _actor_for_record,
    _record_lookup,
    _select_records,
    evaluate_all,
)


def test_e4_actor_paths_exist_for_every_selected_panel():
    qos_records, weight_records, unavailable = _select_records()
    assert len(qos_records) == 2 * len(QOS_VALUES_S)
    assert len(weight_records) == 3 * len(WEIGHT_VALUES)
    assert all(Path(item["actor"]).is_file() for item in qos_records)
    assert all(Path(item["actor"]).is_file() for item in weight_records)
    assert any("Low QoS 3 s" in item for item in unavailable)


def test_qos6_balanced_actors_use_verified_v4_bundles():
    lookup = _record_lookup(
        Path(__file__).resolve().parents[1]
        / ".."
        / "P3"
        / "figs"
        / "v5"
        / "source_data"
        / "e4_weight_sensitivity_v5.json"
    )
    for level in ("Medium", "High"):
        actor = _actor_for_record(lookup[(level, 6.0, 50, 50)])
        assert actor is not None
        assert actor.name == f"{level.lower()}_qos6_actor.pth"


def test_weight_replays_match_all_nine_recorded_outcomes():
    _, weight_records, _ = _select_records()
    results = evaluate_all(weight_records)
    assert len(results) == 9
    assert sum(result["status"] == "feasible" for result in results.values()) == 6
    assert sum(result["status"] == "incomplete" for result in results.values()) == 3
