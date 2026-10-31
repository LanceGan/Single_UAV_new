"""Baseline inspection-sequence optimizers: exact distance TSP and G2A-aware SA.

Both share the fixed start/end and six inspection nodes with the GA pipeline.
Distance TSP ignores every map and payload (pure geometry, exact over all 720
permutations). SA mirrors the conference DRL-SA baseline: its edge cost sees
only the downlink G2A map (flight seconds plus outage seconds), never the A2G
map or the payload vector.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np

from fixed_radio_maps import FixedRadioMaps
from sequence_runner import load_planning_nodes

ROOT = Path(__file__).resolve().parent
SPEED_MPS = 50.0  # 25 m per 0.5 s slot, the World kinematic bound


def _path_cost(cost_matrix, order):
    nodes = (0, *order, cost_matrix.shape[0] - 1)
    return float(sum(cost_matrix[a, b] for a, b in zip(nodes, nodes[1:])))


def distance_matrix_m(nodes_km):
    delta = nodes_km[:, None, :] - nodes_km[None, :, :]
    return np.linalg.norm(delta, axis=-1) * 1000.0


def distance_tsp_order(nodes_km=None):
    """Exact minimum-distance visiting order over all permutations.

    No radio map and no payload enters the objective; this is the pure
    geometric baseline the comparison plan calls Distance-TSP.
    """
    nodes = load_planning_nodes() if nodes_km is None else np.asarray(nodes_km, dtype=float)
    metres = distance_matrix_m(nodes)
    inner = tuple(range(1, len(nodes) - 1))
    best = min(itertools.permutations(inner), key=lambda order: _path_cost(metres, order))
    meta = {
        "method": "exact distance TSP over all permutations",
        "distance_m": _path_cost(metres, best),
        "uses_g2a": False,
        "uses_a2g": False,
        "uses_payloads": False,
    }
    return list(best), meta


def g2a_edge_seconds(a_km, b_km, maps, sample_spacing_m=25.0):
    """Flight seconds and estimated outage seconds along a straight segment."""
    a_km = np.asarray(a_km, dtype=float)
    b_km = np.asarray(b_km, dtype=float)
    distance_m = float(np.linalg.norm(b_km - a_km)) * 1000.0
    flight_s = distance_m / SPEED_MPS
    samples = max(2, int(math.ceil(distance_m / sample_spacing_m)) + 1)
    points = np.linspace(a_km, b_km, samples)
    outage_fraction = float(np.mean([maps.g2a_outage_probability(p) >= 0.5 for p in points]))
    return flight_s, flight_s * outage_fraction


def g2a_cost_matrix_s(nodes_km, maps=None):
    maps = maps if maps is not None else FixedRadioMaps(ROOT / "results" / "datas")
    count = len(nodes_km)
    cost = np.zeros((count, count), dtype=float)
    for i in range(count):
        for j in range(count):
            if i == j:
                continue
            flight_s, outage_s = g2a_edge_seconds(nodes_km[i], nodes_km[j], maps)
            cost[i, j] = flight_s + outage_s
    return cost


def sa_order(nodes_km=None, maps=None, seed=1, initial_temperature=None,
             cooling=0.995, sweeps=4000):
    """Simulated annealing on G2A-only edge seconds (conference DRL-SA style).

    Swap neighbourhood, Metropolis acceptance, exponential cooling. The
    returned meta records the seed and the achieved objective so runs are
    reproducible and auditable.
    """
    nodes = load_planning_nodes() if nodes_km is None else np.asarray(nodes_km, dtype=float)
    cost_matrix = g2a_cost_matrix_s(nodes, maps)
    rng = np.random.default_rng(seed)
    current = list(range(1, len(nodes) - 1))
    rng.shuffle(current)
    current_cost = _path_cost(cost_matrix, current)
    best, best_cost = current.copy(), current_cost
    temperature = float(initial_temperature) if initial_temperature else max(0.3 * current_cost, 1e-9)
    for _ in range(int(sweeps)):
        i, j = sorted(rng.choice(len(current), size=2, replace=False))
        candidate = current.copy()
        candidate[i], candidate[j] = candidate[j], candidate[i]
        candidate_cost = _path_cost(cost_matrix, candidate)
        delta = candidate_cost - current_cost
        if delta <= 0 or rng.random() < math.exp(-delta / max(temperature, 1e-12)):
            current, current_cost = candidate, candidate_cost
            if current_cost < best_cost:
                best, best_cost = current.copy(), current_cost
        temperature *= float(cooling)
    meta = {
        "method": "simulated annealing on G2A flight+outage seconds",
        "seed": int(seed),
        "objective_s": best_cost,
        "sweeps": int(sweeps),
        "cooling": float(cooling),
        "uses_g2a": True,
        "uses_a2g": False,
        "uses_payloads": False,
    }
    return best, meta
