from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "sync_bundles" / "v5_e1_medium_high_actors"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_sync_manifest_has_six_matching_actor_hashes():
    manifest = json.loads((BUNDLE / "MANIFEST.json").read_text())
    assert len(manifest["items"]) == 6
    assert {item["label"] for item in manifest["items"]} == {
        "d3qn_medium",
        "d3qn_high",
        "ppo_medium",
        "ppo_high",
        "sac_medium",
        "sac_high",
    }
    for item in manifest["items"]:
        actor = BUNDLE / item["actor"]
        assert actor.is_file()
        assert _sha256(actor) == item["actor_sha256"]


def test_sync_trajectories_are_complete_and_internally_consistent():
    for label in (
        "d3qn_medium",
        "d3qn_high",
        "ppo_medium",
        "ppo_high",
        "sac_medium",
        "sac_high",
    ):
        metrics = json.loads(
            (BUNDLE / "metrics" / f"{label}_metrics.json").read_text()
        )
        with np.load(BUNDLE / "trajectory" / f"{label}_trajectory.npz") as trace:
            required = {
                "positions",
                "time_s",
                "pending_mb",
                "visited_points",
                "outage_s",
                "current_contiguous_outage_s",
            }
            assert required <= set(trace.files)
            lengths = {len(trace[key]) for key in required}
            assert len(lengths) == 1
            assert np.isfinite(trace["positions"]).all()
            assert np.all(np.diff(trace["time_s"]) >= -1e-12)
            assert int(trace["visited_points"][-1]) == 6
            assert float(trace["pending_mb"][-1]) == 0.0
            assert np.isclose(
                float(trace["current_contiguous_outage_s"].max()),
                float(metrics["outage_time_s"]),
            )
            assert np.isclose(
                float(trace["time_s"][-1]),
                float(metrics["task_time_s"]),
            )


def test_sac_medium_uses_the_accepted_actor_replay_values():
    metrics = json.loads(
        (BUNDLE / "metrics" / "sac_medium_metrics.json").read_text()
    )
    assert metrics["task_time_s"] == 1000.0
    assert metrics["transmission_delay_s"] == 328.0
    assert metrics["T_plus_tau_s"] == 1328.0
    assert metrics["outage_time_s"] == 20.5
    assert metrics["visited_inspection_points"] == 6
    assert metrics["uploaded_mb"] == 900.0
    assert metrics["mission_complete"] is False
    assert metrics["qos_feasible"] is False
