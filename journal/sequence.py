from __future__ import annotations

import math
import itertools
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


class DataAwareIGA:
    """Exact small-instance oracle for the six-node IGA objective.

    The current conference scenario has six inspection nodes, so enumerating
    720 orders provides a deterministic reference before replacing this oracle
    with the legacy GA operators for larger instances.
    """

    def __init__(self, scenario, radio) -> None:
        self.scenario, self.radio = scenario, radio
        self.estimator = EdgeEstimator(scenario, radio)

    def solve(self) -> list[int]:
        nodes = tuple(range(len(self.scenario.inspection_xy)))
        best_order, best_cost = list(nodes), math.inf
        for order in itertools.permutations(nodes):
            cost = 0.0
            for source, target in zip(order, order[1:]):
                edge = self.estimator.estimate(source, target, float(self.scenario.payload_mb[source]))
                cost += edge.cost
            if cost < best_cost:
                best_order, best_cost = list(order), cost
        return best_order
