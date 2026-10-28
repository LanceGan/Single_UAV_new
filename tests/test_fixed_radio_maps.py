import numpy as np

from fixed_radio_maps import FixedRadioMaps


def test_fixed_map_lookup_matches_saved_grid_origin():
    maps = FixedRadioMaps("results/datas")
    a2g = np.load("results/datas/Radio_datas_A2G.npz")
    g2a = np.load("results/datas/Radio_datas.npz")
    assert maps.a2g_snr_db(np.array([0.0, 0.0])) == a2g["arr_1"][0, 0]
    assert maps.g2a_outage_probability(np.array([0.0, 0.0])) == g2a["arr_0"][0, 0]
