"""Deterministic sequence baselines: distance TSP and G2A-aware simulated annealing.

The proposed sequence is the data-aware IGA (distance + G2A outage + A2G upload +
payload). These baselines isolate what each extra term buys:

- ``distance_tsp_order``: flight distance only (no radio maps).
- ``g2a_sa_order``: flight distance + G2A outage, via simulated annealing
  (the conference "DRL-SA" sequence recipe; ignores A2G upload and payload).
"""

from __future__ import annotations

import itertools
import math
import random
from pathlib import Path

import numpy as np

from fixed_radio_maps import FixedRadioMaps

ROOT = Path(__file__).resolve().parent
REFERENCE_SPEED_M_S = 50.0
DELTA_T_S = 0.5
START = np.array([2.62, 2.65])
END = np.array([18.95, 11.12])


def load_planning_nodes() -> np.ndarray:
    users = np.loadtxt(ROOT / "results/datas/Users_6.txt")[:, :2]
    return np.vstack((START, users, END)) / 10.0  # km, node 0 = start, 7 = end


def _leg_outage_s(a: np.ndarray, b: np.ndarray, maps: FixedRadioMaps) -> float:
    """Accumulated G2A outage time (seconds) along the straight leg a -> b."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    distance_km = float(np.linalg.norm(b - a))
    duration_s = distance_km * 1000.0 / REFERENCE_SPEED_M_S
    count = max(1, int(math.ceil(duration_s / DELTA_T_S)))
    times = np.minimum(np.arange(1, count + 1) * DELTA_T_S, duration_s)
    durations = np.diff(np.concatenate(([0.0], times)))
    points = a + (b - a) * (times / duration_s)[:, None]
    return float(sum(dt for point, dt in zip(points, durations)
                     if maps.g2a_outage_probability(point) >= 0.5))


def _sequence_cost(order: list[int], nodes: np.ndarray, maps: FixedRadioMaps) -> float:
    sequence = [0] + [int(node) for node in order] + [len(nodes) - 1]
    cost = 0.0
    for i in range(len(sequence) - 1):
        a, b = nodes[sequence[i]], nodes[sequence[i + 1]]
        flight_s = float(np.linalg.norm(b - a)) * 1000.0 / REFERENCE_SPEED_M_S
        cost += flight_s + _leg_outage_s(a, b, maps)
    return cost


def distance_tsp_order() -> list[int]:
    """Exact shortest-flight order over the six inspection points (1..6)."""
    nodes = load_planning_nodes()
    maps = FixedRadioMaps(ROOT / "results" / "datas")
    del maps  # distance-only baseline does not read the radio maps
    best_order, best_cost = None, math.inf
    for order in itertools.permutations(range(1, 7)):
        cost = _flight_only_cost(list(order), nodes)
        if cost < best_cost:
            best_order, best_cost = list(order), cost
    return best_order


def _flight_only_cost(order: list[int], nodes: np.ndarray) -> float:
    sequence = [0] + list(order) + [len(nodes) - 1]
    return float(sum(np.linalg.norm(nodes[sequence[i + 1]] - nodes[sequence[i]]) * 1000.0 / REFERENCE_SPEED_M_S
                     for i in range(len(sequence) - 1)))


def g2a_sa_order(seed: int = 1, iterations: int = 4000,
                 init_temperature: float = 5.0) -> list[int]:
    """Simulated annealing over distance + G2A outage, start/end fixed."""
    rng = random.Random(seed)
    nodes = load_planning_nodes()
    maps = FixedRadioMaps(ROOT / "results" / "datas")
    current = list(range(1, 7))
    rng.shuffle(current)
    current_cost = _sequence_cost(current, nodes, maps)
    best_order, best_cost = current, current_cost
    temperature = init_temperature
    for iteration in range(iterations):
        temperature = init_temperature * (1.0 - (iteration + 1) / iterations)
        i, j = rng.sample(range(6), 2)
        candidate = current.copy()
        candidate[i], candidate[j] = candidate[j], candidate[i]
        cost = _sequence_cost(candidate, nodes, maps)
        if cost < current_cost or rng.random() < math.exp((current_cost - cost) / max(temperature, 1e-9)):
            current, current_cost = candidate, cost
            if cost < best_cost:
                best_order, best_cost = candidate, cost
    return best_order


if __name__ == "__main__":
    print("distance TSP order:", distance_tsp_order())
    print("G2A-SA order (seed 1):", g2a_sa_order(seed=1))
