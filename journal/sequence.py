from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EdgeEstimate:
    distance_m: float
    upload_rate_mb_s: float
    outage_s: float
    payload_mb: float
    feasible: bool
    cost: float


class EdgeEstimator:
    def __init__(self, scenario, radio) -> None:
        self.scenario, self.radio = scenario, radio

    def estimate(self, i: int, j: int, payload_mb: float) -> EdgeEstimate:
        a, b = self.scenario.inspection_xy[i], self.scenario.inspection_xy[j]
        distance = float(np.linalg.norm(b - a))
        slots = max(1, math.ceil(distance / (self.scenario.speed_max * self.scenario.delta_t)))
        points = np.linspace(a, b, slots + 1)[1:]
        rates = [self.radio.a2g_rate_mb_s(point) for point in points]
        outages = sum(bool(self.radio.g2a_outage(point)) for point in points) * self.scenario.delta_t
        rate = float(np.mean(rates))
        service = sum(rates) * self.scenario.delta_t
        feasible = outages <= self.scenario.outage_max_s and service >= payload_mb and rate > 0
        cost = distance * outages * payload_mb / rate if feasible else math.inf
        return EdgeEstimate(distance, rate, outages, payload_mb, feasible, cost)
