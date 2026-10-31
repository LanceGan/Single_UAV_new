"""A2G-blind baselines: same task and reward, no uplink knowledge in the stack."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from World import World
from map_guidance import MapGuide


def _world(variant="A", **overrides):
    kwargs = dict(length=20, width=20, uav_num=1, dist_max=0.25, delta_t=0.5, t=100,
                  ini_loc=[2.62, 2.65], end_loc=[18.95, 11.12],
                  users_name="results/datas/Users_6.txt", BS_loc=np.zeros((4, 3)),
                  traverse_sequence=[1, 2, 3, 4, 5, 6], variant=variant)
    kwargs.update(overrides)
    return World(**kwargs)


def test_blind_state_drops_exactly_the_a2g_snr_feature():
    sighted = _world()
    blind = _world(a2g_blind=True)
    sighted_state, _ = sighted.reset()
    blind_state, _ = blind.reset()

    assert sighted.state_dim == 8 and blind.state_dim == 7
    np.testing.assert_allclose(blind_state, np.delete(sighted_state, 6))


def test_blind_rejects_map_neighborhood_variant():
    with pytest.raises(ValueError, match="a2g_blind"):
        _world(variant="AB", a2g_blind=True)


def _guide_world():
    shape = (21, 21)
    return SimpleNamespace(
        radio_maps=SimpleNamespace(a2g_snr=np.zeros(shape), g2a_outage=np.zeros(shape)),
        BandWidth=1.0, dist_max=0.25, delta_T=0.5,
        target_loc=np.array([1.5, 1.0]), distance=0.3, transmit=False,
        data_size=10.0, Traverse_order=1,
        UAVs=[SimpleNamespace(x=0.3, y=1.0)], length=2.0, width=2.0,
    )


def test_blind_mapguide_plans_uploads_without_any_rate_knowledge():
    sighted = MapGuide(_guide_world())
    sighted.rate = np.zeros_like(sighted.rate)
    with pytest.raises(RuntimeError, match="upload"):
        sighted._plan(np.array([0.3, 1.0]))

    blind = MapGuide(_guide_world(), use_a2g=False)
    blind.rate = np.zeros_like(blind.rate)  # must be irrelevant to a blind teacher
    action = blind.action()
    assert np.all(np.isfinite(action)) and 0.0 <= action[1] <= 0.25


def test_run_context_round_trips_blind_configuration(tmp_path):
    from evaluate_policy import (
        A2G_BLIND_STATE_SCHEMA,
        STATE_SCHEMA,
        _map_hashes,
        load_run_context,
    )

    config = {
        "algorithm": "td3_bc", "data_level": "low", "episodes": 3000,
        "map_hashes": _map_hashes(), "net_width": 256,
        "order": [1, 6, 5, 2, 3, 4], "outage_max_s": 100.0,
        "payloads_mb": [90.0, 90.0, 60.0, 60.0, 90.0, 60.0],
        "state_dim": 7, "uav_h": 1.0, "variant": "A",
        "a2g_blind": True, "state_schema": A2G_BLIND_STATE_SCHEMA,
    }
    (tmp_path / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
    context = load_run_context(tmp_path)
    assert context.a2g_blind is True and context.state_dim == 7

    config["state_schema"] = STATE_SCHEMA  # wrong schema for a blind run
    (tmp_path / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="state schema"):
        load_run_context(tmp_path)
