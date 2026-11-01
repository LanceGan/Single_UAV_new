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


def _ordered_crossover(parent_a, parent_b, rng):
    size = len(parent_a)
    i, j = sorted(rng.choice(size, size=2, replace=False))
    child = [None] * size
    child[i:j + 1] = parent_a[i:j + 1]
    fill = [gene for gene in parent_b if gene not in child]
    for index in range(size):
        if child[index] is None:
            child[index] = fill.pop(0)
    return child


def ga_proxy_order(nodes_km=None, seed=1, population=24, generations=200,
                   crossover_rate=0.9, mutation_rate=0.2, elite=2):
    """Ordinary GA stand-in for EEGA ("EEGA-TD3 (GA proxy)").

    Fitness is total geometric distance only; EEGA's energy and throughput
    mechanisms are NOT implemented, so results must carry the proxy label
    until the original paper's formulas are available.
    """
    nodes = load_planning_nodes() if nodes_km is None else np.asarray(nodes_km, dtype=float)
    metres = distance_matrix_m(nodes)
    rng = np.random.default_rng(seed)
    inner = list(range(1, len(nodes) - 1))

    def fitness(order):
        return _path_cost(metres, order)

    population_list = [list(rng.permutation(inner)) for _ in range(int(population))]
    for _ in range(int(generations)):
        ranked = sorted(population_list, key=fitness)
        next_population = [order.copy() for order in ranked[:int(elite)]]
        while len(next_population) < len(population_list):
            contenders = [ranked[index] for index in rng.choice(len(ranked), size=3, replace=False)]
            parent_a = min(contenders, key=fitness)
            contenders = [ranked[index] for index in rng.choice(len(ranked), size=3, replace=False)]
            parent_b = min(contenders, key=fitness)
            child = (_ordered_crossover(parent_a, parent_b, rng)
                     if rng.random() < crossover_rate else parent_a.copy())
            if rng.random() < mutation_rate:
                i, j = sorted(rng.choice(len(child), size=2, replace=False))
                child[i], child[j] = child[j], child[i]
            next_population.append(child)
        population_list = next_population
    best = min(population_list, key=fitness)
    best = [int(gene) for gene in best]
    meta = {
        "method": "ordinary GA proxy for EEGA (distance-only fitness)",
        "seed": int(seed),
        "distance_m": fitness(best),
        "population": int(population),
        "generations": int(generations),
        "uses_g2a": False,
        "uses_a2g": False,
        "uses_payloads": False,
        "proxy_disclaimer": "EEGA energy/throughput mechanisms not implemented",
    }
    return list(best), meta


def g2a_reciprocal_rate(a_km, b_km, maps, sample_spacing_m=25.0):
    """Mean uplink-rate estimate along a leg from the G2A map (channel reciprocity).

    Layer-3 baselines may not read the project's A2G map; their originals used
    their own channel models, which this reciprocity proxy stands in for. The
    constant bandwidth factor is dropped because only ratios affect ordering.
    """
    a_km = np.asarray(a_km, dtype=float)
    b_km = np.asarray(b_km, dtype=float)
    distance_m = float(np.linalg.norm(b_km - a_km)) * 1000.0
    samples = max(2, int(math.ceil(distance_m / sample_spacing_m)) + 1)
    points = np.linspace(a_km, b_km, samples)
    rates = [math.log2(1.0 + 10.0 ** (maps.g2a_snr_db(point) / 10.0)) for point in points]
    return max(float(np.mean(rates)), 1e-9)


def _leg_tables(nodes_km, payloads_mb, maps):
    """Flight seconds, reciprocal-rate and departure payload for every leg."""
    maps = maps if maps is not None else FixedRadioMaps(ROOT / "results" / "datas")
    node_payloads = np.concatenate(([0.0], np.asarray(payloads_mb, dtype=float), [0.0]))
    count = len(nodes_km)
    flight_s = np.zeros((count, count))
    rate = np.ones((count, count))
    for i in range(count):
        for j in range(count):
            if i == j:
                continue
            distance_m = float(np.linalg.norm(nodes_km[j] - nodes_km[i])) * 1000.0
            flight_s[i, j] = distance_m / SPEED_MPS
            rate[i, j] = g2a_reciprocal_rate(nodes_km[i], nodes_km[j], maps)
    return flight_s, rate, node_payloads


def eega_cost_matrix(nodes_km, payloads_mb, maps=None):
    """EEGA edge weight L_ij = (I_i / Q_ij) * E_ij with E_ij ∝ flight seconds.

    I_i is the payload collected at the departure node (uploaded on this leg),
    Q_ij the reciprocity-estimated throughput. Start legs carry no payload and
    are literally zero-cost under the published formula; this is recorded, not
    silently patched.
    """
    flight_s, rate, node_payloads = _leg_tables(np.asarray(nodes_km, dtype=float), payloads_mb, maps)
    cost = (node_payloads[:, None] / rate) * flight_s
    np.fill_diagonal(cost, 0.0)
    return cost


def eega_order(payloads_mb, nodes_km=None, maps=None, seed=1, population=24,
               generations=200, crossover_rate=0.9, mutation_rate=0.2, elite=2):
    """EEGA-style GA on the weighted edge matrix ("EEGA-TD3" sequence stage).

    The published edge weight is implemented exactly; the adaptive operator
    schedule of the original is unpublished, so standard tournament/OX/swap
    operators are used and disclosed.
    """
    nodes = load_planning_nodes() if nodes_km is None else np.asarray(nodes_km, dtype=float)
    cost = eega_cost_matrix(nodes, payloads_mb, maps)
    rng = np.random.default_rng(seed)
    inner = list(range(1, len(nodes) - 1))

    def fitness(order):
        return _path_cost(cost, order)

    population_list = [list(rng.permutation(inner)) for _ in range(int(population))]
    for _ in range(int(generations)):
        ranked = sorted(population_list, key=fitness)
        next_population = [order.copy() for order in ranked[:int(elite)]]
        while len(next_population) < len(population_list):
            contenders = [ranked[index] for index in rng.choice(len(ranked), size=3, replace=False)]
            parent_a = min(contenders, key=fitness)
            contenders = [ranked[index] for index in rng.choice(len(ranked), size=3, replace=False)]
            parent_b = min(contenders, key=fitness)
            child = (_ordered_crossover(parent_a, parent_b, rng)
                     if rng.random() < crossover_rate else parent_a.copy())
            if rng.random() < mutation_rate:
                i, j = sorted(rng.choice(len(child), size=2, replace=False))
                child[i], child[j] = child[j], child[i]
            next_population.append(child)
        population_list = next_population
    best = [int(gene) for gene in min(population_list, key=fitness)]
    meta = {
        "method": "EEGA-style GA on L_ij=(I_i/Q_ij)*E_ij (G2A reciprocity throughput)",
        "seed": int(seed),
        "objective": float(_path_cost(cost, best)),
        "uses_g2a": True,
        "uses_a2g": False,
        "uses_payloads": True,
        "adaptation_notes": "Q_ij from G2A reciprocity; adaptive GA operator schedule unpublished, "
                            "standard tournament/OX/swap used; start legs zero-cost per the literal formula",
    }
    return best, meta


def yang_order(payloads_mb, nodes_km=None, maps=None):
    """Yang-style weighted traversal order, solved exactly over all permutations.

    Edge cost = flight seconds + I_i/Q_ij (estimated offload seconds), so data
    offloading dominates exactly when payloads are large, per the described
    weighting principle. The original solves trajectories with SCA/BCD; the
    "+TD3" pairing is this project's adaptation and must be declared as such.
    """
    nodes = load_planning_nodes() if nodes_km is None else np.asarray(nodes_km, dtype=float)
    flight_s, rate, node_payloads = _leg_tables(nodes, payloads_mb, maps)
    cost = flight_s + node_payloads[:, None] / rate
    np.fill_diagonal(cost, 0.0)
    inner = tuple(range(1, len(nodes) - 1))
    best = min(itertools.permutations(inner), key=lambda order: _path_cost(cost, order))
    meta = {
        "method": "Yang-style weighted order: flight seconds + I_i/Q_ij, exact over permutations",
        "objective_s": _path_cost(cost, list(best)),
        "uses_g2a": True,
        "uses_a2g": False,
        "uses_payloads": True,
        "adaptation_notes": "original trajectory method is SCA/BCD; TD3 pairing is an adaptation; "
                            "Q_ij from G2A reciprocity",
    }
    return [int(node) for node in best], meta


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
