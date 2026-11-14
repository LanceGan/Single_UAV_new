import json
from pathlib import Path

from plot_v5_e1_trajectories import evaluate_all


def test_all_e1_low_actor_replays_match_eval_best():
    results = evaluate_all()
    assert set(results) == {"CA-TD3", "D3QN", "PPO", "DDPG", "SAC"}
    assert results["CA-TD3"]["report"]["task_time_s"] == 169.0
    assert results["D3QN"]["report"]["task_time_s"] == 175.0
    assert results["PPO"]["report"]["task_time_s"] == 190.0
    assert results["SAC"]["report"]["task_time_s"] == 170.0
    assert results["DDPG"]["report"]["visited_inspection_points"] == 4


def test_retired_e1_low_spatial_panels_are_not_reintroduced():
    panel_root = (
        Path(__file__).resolve().parents[2]
        / "P3"
        / "figs"
        / "v5"
        / "panels"
        / "e1_low_trajectories_v5"
    )
    metadata = json.loads(
        (panel_root / "metadata.json").read_text(encoding="utf-8")
    )

    assert [item["panel"] for item in metadata["panels"]] == list(
        "abcdefghij"
    )
    for item in metadata["panels"]:
        assert all(not Path(path).exists() for path in item["files"].values())
    for path in metadata["shared_assets"].values():
        assert not Path(path).exists()
