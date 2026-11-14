"""Baseline inspection-sequence optimizers: exact distance TSP and G2A-aware SA.

Both share the fixed start/end and six inspection nodes with the GA pipeline.
Distance TSP ignores every map and payload (pure geometry, exact over all 720
permutations). SA mirrors the conference DRL-SA baseline: its edge cost sees
only the downlink G2A map (flight seconds plus outage seconds), never the A2G
map or the payload vector.
"""

from __future__ import annotations

import itertools
import heapq
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


def distance_astar_order(nodes_km=None):
    """Exact sequence-only A* on the pure geometric fixed-endpoint TSP.

    The search state is ``(last_inspection_node, visited_mask)``.  The
    heuristic is the straight-line distance from the current node to the fixed
    end point; because every remaining inspection point still has to be
    visited, this is an admissible lower bound on the completion cost.
    """
    nodes = load_planning_nodes() if nodes_km is None else np.asarray(nodes_km, dtype=float)
    metres = distance_matrix_m(nodes)
    inspection = tuple(range(1, len(nodes) - 1))
    end = len(nodes) - 1
    count = len(inspection)
    full_mask = (1 << count) - 1

    def heuristic(node):
        return float(metres[node, end])

    initial = (0, 0)
    frontier = [(heuristic(0), 0.0, initial)]
    best_cost = {initial: 0.0}
    parents = {initial: None}
    expansions = 0
    while frontier:
        _, cost, state = heapq.heappop(frontier)
        if cost != best_cost.get(state):
            continue
        node, mask = state
        if mask == full_mask:
            total = cost + float(metres[node, end])
            order = []
            cursor = state
            while cursor is not None:
                current_node, _ = cursor
                if current_node != 0:
                    order.append(int(current_node))
                cursor = parents[cursor]
            order.reverse()
            return order, {
                "method": "A* on standard distance objective",
                "distance_m": float(total),
                "expansions": int(expansions),
                "optimality_proven": True,
                "uses_g2a": False,
                "uses_a2g": False,
                "uses_payloads": False,
            }

        expansions += 1
        for bit, node_id in enumerate(inspection):
            if mask & (1 << bit):
                continue
            next_state = (int(node_id), mask | (1 << bit))
            next_cost = cost + float(metres[node, node_id])
            if next_cost >= best_cost.get(next_state, float("inf")):
                continue
            best_cost[next_state] = next_cost
            parents[next_state] = state
            heapq.heappush(
                frontier,
                (next_cost + heuristic(node_id), next_cost, next_state),
            )
    raise RuntimeError("A* exhausted the sequence state space without a path")


def distance_pso_order(nodes_km=None, seed=1, particles=80, iterations=250,
                       inertia=0.7, cognitive=1.5, social=1.5):
    """Discrete random-key PSO for the standard distance objective."""
    nodes = load_planning_nodes() if nodes_km is None else np.asarray(nodes_km, dtype=float)
    metres = distance_matrix_m(nodes)
    inner = tuple(range(1, len(nodes) - 1))
    rng = np.random.default_rng(seed)

    def decode(position):
        return [int(inner[index]) for index in np.argsort(position)]

    def fitness(position):
        return _path_cost(metres, decode(position))

    positions = rng.uniform(-1.0, 1.0, size=(int(particles), len(inner)))
    velocities = rng.normal(0.0, 0.1, size=positions.shape)
    personal_best = positions.copy()
    personal_cost = np.asarray([fitness(position) for position in positions])
    best_index = int(np.argmin(personal_cost))
    global_best = personal_best[best_index].copy()
    global_cost = float(personal_cost[best_index])

    for _ in range(int(iterations)):
        r1 = rng.random(positions.shape)
        r2 = rng.random(positions.shape)
        velocities = (
            float(inertia) * velocities
            + float(cognitive) * r1 * (personal_best - positions)
            + float(social) * r2 * (global_best[None, :] - positions)
        )
        positions = np.clip(positions + velocities, -4.0, 4.0)
        for index, position in enumerate(positions):
            cost = fitness(position)
            if cost < personal_cost[index]:
                personal_cost[index] = cost
                personal_best[index] = position.copy()
                if cost < global_cost:
                    global_cost = float(cost)
                    global_best = position.copy()
    order = decode(global_best)
    return order, {
        "method": "random-key PSO on standard distance objective",
        "seed": int(seed),
        "distance_m": float(global_cost),
        "particles": int(particles),
        "iterations": int(iterations),
        "uses_g2a": False,
        "uses_a2g": False,
        "uses_payloads": False,
    }


def _eeap_particle_update(current, personal_best, global_best, inertia,
                          cognitive, social, rng):
    """Update one permutation particle with the EEAP-PSO parent rule."""
    total = float(inertia) + float(cognitive) + float(social)
    draw = float(rng.random()) * total
    if draw <= float(inertia):
        second_parent = list(reversed(current))
    elif draw <= float(inertia) + float(cognitive):
        second_parent = list(personal_best)
    else:
        second_parent = list(global_best)

    start, end = sorted(rng.choice(len(current), size=2, replace=False))
    child = [None] * len(current)
    child[int(start):int(end) + 1] = current[int(start):int(end) + 1]
    fill = [gene for gene in second_parent if gene not in child]
    for index in range(len(child)):
        if child[index] is None:
            child[index] = fill.pop(0)
    return [int(gene) for gene in child]


def _eeap_two_opt(order, fitness):
    """Return the best reverse-segment neighbour until no improvement remains."""
    incumbent = [int(node) for node in order]
    incumbent_cost = float(fitness(incumbent))
    improved = True
    while improved:
        improved = False
        best_order = incumbent
        best_cost = incumbent_cost
        for start in range(len(incumbent) - 1):
            for end in range(start + 2, len(incumbent) + 1):
                candidate = (
                    incumbent[:start]
                    + list(reversed(incumbent[start:end]))
                    + incumbent[end:]
                )
                candidate_cost = float(fitness(candidate))
                if candidate_cost < best_cost - 1e-12:
                    best_order = candidate
                    best_cost = candidate_cost
        if best_cost < incumbent_cost - 1e-12:
            incumbent = [int(node) for node in best_order]
            incumbent_cost = best_cost
            improved = True
    return incumbent, incumbent_cost


def eeap_pso_order(
    payloads_mb,
    nodes_km=None,
    maps=None,
    seed=1,
    particles=80,
    iterations=200,
    inertia_start=0.9,
    inertia_end=0.4,
    cognitive_start=0.8,
    social_start=2.2,
    two_opt_interval=20,
):
    """EEAP-PSO sequence adaptation with cargo-side constraints removed.

    The source algorithm is the energy-oriented PSO layer in the MCO-A*
    study. For the single-mission inspection problem, payload-capacity limits,
    onboard-energy limits, warehouse returns, and the multi-trip dynamic
    program are removed. The retained particle operators optimize one
    permutation on the source paper's rotary-wing energy model, using the V5
    payload vector as relative load in the absence of a mass estimate. The
    start and end remain the fixed inspection endpoints. This is an
    adaptation, not an exact reproduction of the multi-trip cargo model.
    """
    nodes = (
        load_planning_nodes()
        if nodes_km is None
        else np.asarray(nodes_km, dtype=float)
    )
    if len(nodes) < 3:
        raise ValueError("EEAP-PSO needs start, inspection, and end nodes")

    distance_matrix = distance_matrix_m(nodes)
    inspection = tuple(range(1, len(nodes) - 1))
    payloads = np.asarray(payloads_mb, dtype=float)
    if len(payloads) != len(inspection):
        raise ValueError("payload vector must align with inspection nodes")
    rng = np.random.default_rng(seed)

    def leg_energy(distance_m, current_load):
        rho, rotor_radius, rotor_area = 1.225, 0.4, 0.503
        rotor_speed, tip_speed = 300.0, 120.0
        solidity, drag_ratio, induced_factor = 0.05, 0.6, 0.1
        profile_coefficient, reference_speed = 0.012, 50.0
        base_weight = 200.0
        profile_power = (
            profile_coefficient
            / 8.0
            * rho
            * solidity
            * rotor_area
            * rotor_speed**3
            * rotor_radius**3
        )
        weight = base_weight + float(current_load)
        induced_base = (
            (1.0 + induced_factor)
            * weight**1.5
            / math.sqrt(2.0 * rho * rotor_area)
        )
        effective_speed = math.sqrt(weight / (2.0 * rho * rotor_area))
        parasite_power = (
            drag_ratio
            / 2.0
            * rho
            * solidity
            * rotor_area
            * reference_speed**3
        )
        blade_power = profile_power * (
            1.0 + 3.0 * reference_speed**2 / tip_speed**2
        )
        induced_term = max(
            0.0,
            math.sqrt(
                1.0
                + reference_speed**4 / (4.0 * effective_speed**4)
            )
            - reference_speed**2 / (2.0 * effective_speed**2),
        )
        induced_power = induced_base * math.sqrt(induced_term)
        return (
            (parasite_power + induced_power + blade_power)
            / reference_speed
            * float(distance_m)
            + weight * 100.0
        )

    def fitness(order):
        current_load = 0.0
        total = 0.0
        previous = 0
        for node in [int(value) for value in order]:
            total += leg_energy(distance_matrix[previous, node], current_load)
            current_load += float(payloads[node - 1])
            previous = node
        total += leg_energy(
            distance_matrix[previous, len(nodes) - 1], current_load
        )
        return total

    nearest = []
    remaining = set(inspection)
    current = 0
    while remaining:
        current = min(remaining, key=lambda node: distance_matrix[current, node])
        nearest.append(int(current))
        remaining.remove(current)

    population = [nearest]
    population.extend(
        [int(node) for node in rng.permutation(inspection)]
        for _ in range(max(0, int(particles) - 1))
    )
    personal_best = [list(particle) for particle in population]
    personal_cost = np.asarray(
        [fitness(particle) for particle in population], dtype=float
    )
    best_index = int(np.argmin(personal_cost))
    global_best = list(personal_best[best_index])
    global_cost = float(personal_cost[best_index])
    incumbent_orders = [
        (0, list(global_best), global_cost),
    ]

    for iteration in range(1, int(iterations) + 1):
        fraction = iteration / max(int(iterations), 1)
        inertia = float(inertia_start) + fraction * (
            float(inertia_end) - float(inertia_start)
        )
        population = [
            _eeap_particle_update(
                particle,
                personal_best[index],
                global_best,
                inertia,
                cognitive_start,
                social_start,
                rng,
            )
            for index, particle in enumerate(population)
        ]
        costs = np.asarray([fitness(particle) for particle in population])
        for index, (particle, cost) in enumerate(zip(population, costs)):
            if cost < personal_cost[index] - 1e-12:
                personal_cost[index] = float(cost)
                personal_best[index] = list(particle)

        current_index = int(np.argmin(costs))
        if costs[current_index] < global_cost - 1e-12:
            global_cost = float(costs[current_index])
            global_best = list(population[current_index])
            global_best, global_cost = _eeap_two_opt(global_best, fitness)
        if (
            int(two_opt_interval) > 0
            and iteration % int(two_opt_interval) == 0
        ):
            global_best, global_cost = _eeap_two_opt(global_best, fitness)
        incumbent_orders.append(
            (iteration, list(global_best), float(global_cost))
        )

    return [int(node) for node in global_best], {
        "method": "EEAP-PSO sequence adaptation",
        "source_algorithm": "EEAP-PSO from the MCO-A* study",
        "seed": int(seed),
        "objective": float(global_cost),
        "particles": int(particles),
        "iterations": int(iterations),
        "inertia_start": float(inertia_start),
        "inertia_end": float(inertia_end),
        "cognitive": float(cognitive_start),
        "social": float(social_start),
        "two_opt_interval": int(two_opt_interval),
        "uses_g2a": False,
        "uses_a2g": False,
        "uses_payloads": True,
        "removed_constraints": [
            "payload-capacity limit",
            "onboard-energy limit",
            "warehouse returns and multi-trip dynamic programming",
        ],
        "objective_note": (
            "single-tour rotary-wing energy; V5 payload values used as relative "
            "load because the scenario provides data volume rather than mass"
        ),
        "incumbent_orders": incumbent_orders,
    }


def priga_cost_matrix(
    nodes_km,
    payloads_mb,
    maps=None,
    outage_probability_threshold=0.5,
    outage_total_weight=0.25,
    outage_max_weight=1.0,
):
    """Payload/radio-informed edge matrix for the V5 PRIGA surrogate.

    The surrogate aligns the sequence objective with the realized mission
    quantities by combining flight time, estimated upload delay, total outage
    exposure, and an explicit penalty on the longest contiguous outage of each
    straight-line leg.
    """
    nodes = np.asarray(nodes_km, dtype=float)
    maps = maps if maps is not None else FixedRadioMaps(ROOT / "results" / "datas")
    count = len(nodes)
    payloads = np.concatenate(([0.0], np.asarray(payloads_mb, dtype=float), [0.0]))
    cost = np.zeros((count, count), dtype=float)
    details = np.zeros((count, count, 4), dtype=float)
    for source in range(count):
        for destination in range(count):
            if source == destination:
                continue
            distance_m = float(np.linalg.norm(nodes[destination] - nodes[source])) * 1000.0
            samples = max(
                2,
                int(math.ceil(distance_m / 25.0)) + 1,
            )
            points = np.linspace(nodes[source], nodes[destination], samples)
            rates = np.asarray([
                math.log2(1.0 + 10.0 ** (maps.a2g_snr_db(point) / 10.0))
                for point in points
            ])
            outages = np.asarray([
                maps.g2a_outage_probability(point) >= outage_probability_threshold
                for point in points
            ], dtype=bool)
            slot_s = distance_m / SPEED_MPS / max(samples - 1, 1)
            current_run = max_run = 0
            for outage in outages:
                current_run = current_run + 1 if outage else 0
                max_run = max(max_run, current_run)
            flight_s = distance_m / SPEED_MPS
            mean_rate = float(np.mean(rates))
            upload_s = float(payloads[source]) / max(mean_rate, 1e-9)
            outage_total_s = float(np.sum(outages) * slot_s)
            outage_max_s = float(max_run * slot_s)
            cost[source, destination] = (
                flight_s
                + upload_s
                + float(outage_total_weight) * outage_total_s
                + float(outage_max_weight) * outage_max_s
            )
            details[source, destination] = (
                flight_s,
                upload_s,
                outage_max_s,
                outage_total_s,
            )
    return cost, details


def priga_order(
    payloads_mb,
    nodes_km=None,
    maps=None,
    seed=1,
    population=25,
    generations=500,
    outage_probability_threshold=0.5,
):
    """Restore the original PRIGA order used by the formal experiments.

    The canonical edge cost remains
    ``flight_time + payload/rate + total_outage``.  The experimental V5
    maximum-contiguous-outage surrogate is intentionally not used.
    """
    from sequence_runner import solve_ga_order

    order = solve_ga_order(
        payloads_mb,
        population=population,
        iterations=generations,
        outage_probability_threshold=outage_probability_threshold,
    )
    return [int(node) for node in order], {
        "method": "PRIGA (original payload/radio edge cost)",
        "seed": int(seed),
        "uses_g2a": True,
        "uses_a2g": True,
        "uses_payloads": True,
        "outage_probability_threshold": float(outage_probability_threshold),
        "formula": "flight_time + payload/mean_A2G_rate + total_outage",
    }


def g2a_edge_seconds(a_km, b_km, maps, sample_spacing_m=25.0,
                     outage_probability_threshold=0.5):
    """Flight seconds and estimated outage seconds along a straight segment."""
    a_km = np.asarray(a_km, dtype=float)
    b_km = np.asarray(b_km, dtype=float)
    distance_m = float(np.linalg.norm(b_km - a_km)) * 1000.0
    flight_s = distance_m / SPEED_MPS
    samples = max(2, int(math.ceil(distance_m / sample_spacing_m)) + 1)
    points = np.linspace(a_km, b_km, samples)
    outage_fraction = float(np.mean([
        maps.g2a_outage_probability(p) >= outage_probability_threshold
        for p in points
    ]))
    return flight_s, flight_s * outage_fraction


def g2a_cost_matrix_s(nodes_km, maps=None, outage_probability_threshold=0.5):
    maps = maps if maps is not None else FixedRadioMaps(ROOT / "results" / "datas")
    count = len(nodes_km)
    cost = np.zeros((count, count), dtype=float)
    for i in range(count):
        for j in range(count):
            if i == j:
                continue
            flight_s, outage_s = g2a_edge_seconds(
                nodes_km[i],
                nodes_km[j],
                maps,
                outage_probability_threshold=outage_probability_threshold,
            )
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
             cooling=0.995, sweeps=4000, outage_probability_threshold=0.5):
    """Simulated annealing on G2A-only edge seconds (conference DRL-SA style).

    Swap neighbourhood, Metropolis acceptance, exponential cooling. The
    returned meta records the seed and the achieved objective so runs are
    reproducible and auditable.
    """
    nodes = load_planning_nodes() if nodes_km is None else np.asarray(nodes_km, dtype=float)
    cost_matrix = g2a_cost_matrix_s(
        nodes,
        maps,
        outage_probability_threshold=outage_probability_threshold,
    )
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
        "outage_probability_threshold": float(outage_probability_threshold),
    }
    return best, meta


def distance_sa_order(nodes_km=None, seed=1, initial_temperature=None,
                      cooling=0.995, sweeps=4000):
    """Simulated annealing on pure flight distance for the SA-D3QN baseline."""
    nodes = load_planning_nodes() if nodes_km is None else np.asarray(nodes_km, dtype=float)
    cost_matrix = distance_matrix_m(nodes)
    rng = np.random.default_rng(seed)
    current = list(range(1, len(nodes) - 1))
    rng.shuffle(current)
    current_cost = _path_cost(cost_matrix, current)
    best, best_cost = current.copy(), current_cost
    temperature = (
        float(initial_temperature)
        if initial_temperature is not None
        else max(0.3 * current_cost, 1e-9)
    )
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
    return best, {
        "method": "simulated annealing on pure flight distance",
        "seed": int(seed),
        "distance_m": float(best_cost),
        "sweeps": int(sweeps),
        "cooling": float(cooling),
        "uses_g2a": False,
        "uses_a2g": False,
        "uses_payloads": False,
    }


def service_sa_order(payloads_mb, nodes_km=None, maps=None, seed=1,
                     initial_temperature=None, cooling=0.995, sweeps=4000):
    """Simulated annealing on flight plus payload offload time (DRL-SA).

    The baseline is A2G-map-unaware, so the throughput estimate comes from G2A
    reciprocity. Upload physics remains active in the real environment.
    """
    nodes = load_planning_nodes() if nodes_km is None else np.asarray(nodes_km, dtype=float)
    flight_s, rate, node_payloads = _leg_tables(nodes, payloads_mb, maps)
    cost_matrix = flight_s + node_payloads[:, None] / np.maximum(rate, 1e-9)
    rng = np.random.default_rng(seed)
    current = list(range(1, len(nodes) - 1))
    rng.shuffle(current)
    current_cost = _path_cost(cost_matrix, current)
    best, best_cost = current.copy(), current_cost
    temperature = (
        float(initial_temperature)
        if initial_temperature is not None
        else max(0.3 * current_cost, 1e-9)
    )
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
    return best, {
        "method": "simulated annealing on flight plus payload offload time",
        "seed": int(seed),
        "objective_s": float(best_cost),
        "sweeps": int(sweeps),
        "cooling": float(cooling),
        "uses_g2a": True,
        "uses_a2g": False,
        "uses_payloads": True,
        "throughput_estimate": "G2A reciprocity",
    }
