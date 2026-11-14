from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "journal_results/v5-deterministic/dual-map-astar"


def test_v5_dual_map_astar_uses_both_radio_maps_and_satisfies_protocol():
    config = json.loads((RUN / "run_config.json").read_text(encoding="utf-8"))
    assert config["total_steps"] == 2000
    assert config["outage_max_s"] == 6.0
    assert "A2G" in config["information_use"]
    assert "G2A" in config["information_use"]

    for level in ("low", "medium", "high"):
        metrics = json.loads(
            (RUN / level / "metrics.json").read_text(encoding="utf-8")
        )
        assert metrics["information_use"].keys() == {"A2G", "G2A"}
        assert metrics["total_steps"] == 2000
        assert metrics["planner"]["status"] == "success"
        assert metrics["world_replay"]["success"] is True
        assert metrics["world_replay"]["visited_inspection_points"] == 6
        assert np.isclose(
            metrics["world_replay"]["uploaded_mb"],
            metrics["world_replay"]["expected_upload_mb"],
            atol=1e-6,
        )
        assert (
            metrics["world_replay"]["outage_time_s"]
            <= metrics["world_replay"]["outage_limit_s"]
        )


def test_v5_dual_map_astar_payload_tiers_have_distinct_search_routes():
    with np.load(RUN / "medium" / "trajectory.npz") as medium:
        medium_positions = np.asarray(medium["positions"], dtype=float)
    with np.load(RUN / "high" / "trajectory.npz") as high:
        high_positions = np.asarray(high["positions"], dtype=float)

    assert medium_positions.shape != high_positions.shape
    assert not (
        medium_positions.shape == high_positions.shape
        and np.array_equal(medium_positions, high_positions)
    )
