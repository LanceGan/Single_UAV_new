"""Deterministic access to the preconstructed A2G and G2A radio maps."""

from pathlib import Path

import numpy as np


class FixedRadioMaps:
    def __init__(self, data_dir):
        data_dir = Path(data_dir)
        a2g = np.load(data_dir / "Radio_datas_A2G.npz")
        g2a = np.load(data_dir / "Radio_datas.npz")
        self.a2g_snr = a2g["arr_1"]
        self.g2a_outage = g2a["arr_0"]
        self.g2a_snr = g2a["arr_1"]

    @staticmethod
    def _index(location_km):
        point = np.asarray(location_km, dtype=float)
        return tuple(np.clip(np.rint(point * 100).astype(int), 0, 200))

    def a2g_snr_db(self, location_km):
        x, y = self._index(location_km)
        return float(self.a2g_snr[y, x])

    def g2a_outage_probability(self, location_km):
        x, y = self._index(location_km)
        return float(self.g2a_outage[y, x])

    def g2a_snr_db(self, location_km):
        x, y = self._index(location_km)
        return float(self.g2a_snr[y, x])
