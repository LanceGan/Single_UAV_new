"""Seed-controlled access to the repository's original dual radio maps."""

from __future__ import annotations

import importlib

import numpy as np

from journal.scenario import Scenario


class RadioAdapter:
    """Convert environment coordinates once before querying legacy maps."""

    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario
        self.a2g_module = importlib.import_module("radio_map_A2G")
        self.g2a_module = importlib.import_module("radio_map_G2A")

    def seed(self, seed: int) -> None:
        np.random.seed(seed)

    @staticmethod
    def _legacy_location(xy: np.ndarray) -> np.ndarray:
        point = np.asarray(xy, dtype=float).reshape(2)
        location = np.zeros((1, 3), dtype=float)
        location[0, :2] = point / 10.0
        location[0, 2] = 0.1
        return location

    def a2g_rate_mb_s(self, xy: np.ndarray) -> float:
        snr_db = float(self.a2g_module.getPointDateRate(self._legacy_location(xy)))
        return float(np.log2(1.0 + 10.0 ** (snr_db / 10.0)))

    def g2a_outage(self, xy: np.ndarray) -> bool:
        outage_prob, _ = self.g2a_module.getPointMiniOutage(self._legacy_location(xy))
        return bool(float(outage_prob[0]) >= 0.5)
