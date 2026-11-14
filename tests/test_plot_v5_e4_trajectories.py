from __future__ import annotations

import json
from pathlib import Path

from plot_v5_e4_trajectories import (
    QOS_VALUES_S,
    WEIGHT_VALUES,
    _actor_for_record,
    _record_lookup,
    _select_records,
    evaluate_all,
)


def test_e4_sources_exist_for_every_selected_panel():
    qos_records, weight_records, unavailable = _select_records()
    assert len(qos_records) == 3 * len(QOS_VALUES_S)
    assert len(weight_records) == 3 * len(WEIGHT_VALUES)
    assert unavailable == []
    assert all(
        (
            item["actor"] is not None
            and Path(item["actor"]).is_file()
        )
        or (
            item["win_trajectory"] is not None
            and Path(item["win_trajectory"]).is_file()
        )
        for item in qos_records
    )
    assert all(Path(item["actor"]).is_file() for item in weight_records)


def test_recovered_win_low_qos_trajectories_match_eval_history():
    qos_records, _, _ = _select_records()
    recovered = [
        item
        for item in qos_records
        if item["source_kind"] == "win_trajectory"
    ]
    assert {item["column"] for item in recovered} == {3.0, 4.0, 5.0, 8.0}
    results = evaluate_all(recovered)
    assert len(results) == 4
    expected_time = {3.0: 169.0, 4.0: 173.5, 5.0: 176.5, 8.0: 164.0}
    for item in recovered:
        result = results[item["panel_id"] + "_qos"]
        assert result["report"]["task_time_s"] == expected_time[item["column"]]
        assert result["report"]["visited_inspection_points"] == 6
        assert result["report"]["uploaded_mb"] == 450.0


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


def test_e4_spatial_panel_artifacts_are_independent_and_complete():
    root = (
        Path(__file__).resolve().parents[2]
        / "P3"
        / "figs"
        / "v5"
        / "panels"
    )
    expected = {
        "e4_qos_trajectories_v5": list("abcdefghijklmno"),
        "e4_weight_trajectories_v5": list("abcdefghi"),
    }
    for stem, panel_ids in expected.items():
        folder = root / stem
        metadata = json.loads(
            (folder / "metadata.json").read_text(encoding="utf-8")
        )
        assert [item["panel"] for item in metadata["panels"]] == panel_ids
        for item in metadata["panels"]:
            assert all(Path(path).is_file() for path in item["files"].values())
        for path in metadata["shared_assets"].values():
            assert Path(path).is_file()
