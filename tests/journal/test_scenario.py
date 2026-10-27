import json

import numpy as np

from journal.scenario import DATA_VECTORS_PATH, load_scenario


def test_fixed_vector_is_loaded_verbatim():
    stored = json.loads(DATA_VECTORS_PATH.read_text(encoding="utf-8"))
    scenario = load_scenario("low")

    assert scenario.payload_mb.tolist() == stored["vectors_mb"]["low"]
    assert len(scenario.payload_mb) == len(scenario.inspection_xy) == 6


def test_payload_vectors_stay_in_declared_intervals():
    low = load_scenario("low").payload_mb
    medium = load_scenario("medium").payload_mb
    high = load_scenario("high").payload_mb

    assert np.all((low >= 50) & (low < 100))
    assert np.all((medium >= 100) & (medium < 200))
    assert np.all((high >= 200) & (high <= 300))


def test_legacy_spatial_inputs_are_reused():
    scenario = load_scenario("medium")

    assert scenario.start_xy == (2.62, 2.65)
    assert scenario.end_xy == (18.95, 11.12)
    assert scenario.gbs_xyz.shape == (4, 3)
