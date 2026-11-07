"""Radio-map-free A* and PSO baselines for end-to-end trajectories.

Both baselines receive only the start, inspection-point, and destination
coordinates.  They do not read G2A outage, A2G rate, payload, or the proposed
sequence.  The resulting trajectory is replayed in the common World with a
radio-map-free stop-and-upload protocol: after collecting a payload, the UAV
hovers at that point until the buffer clears before continuing.
"""

from __future__ import annotations

from dataclasses import dataclass
import contextlib
import heapq
import io
import itertools
import math
import time
from typing import Sequence

import numpy as np

from astar_planner import SpatialGraph
from pso_trajectory import make_context
from evaluate_policy import rollout


@dataclass(frozen=True)
class PureAStarResult:
    status: str
    node_path: tuple[int, ...]
    inspection_order: tuple[int, ...]
    path_distance: float
    expansions: int
    generated_states: int
    runtime_s: float
    optimality_proven: bool
    message: str

    @property
    def success(self) -> bool:
        return self.status == "success"


class PureAStarTrajectoryPlanner:
    """A* over ``(grid node, visited inspection set)`` using distance only."""

    def __init__(
        self,
        *,
        graph: SpatialGraph,
        start_node: int,
        inspection_nodes: Sequence[int],
        end_node: int,
        arrival_radius: float = 0.5,
        max_expansions: int = 3_000_000,
        max_runtime_s: float = 300.0,
    ):
        self.graph = graph
        self.start_node = int(start_node)
        self.inspection_nodes = tuple(int(node) for node in inspection_nodes)
        self.end_node = int(end_node)
        self.arrival_radius = float(arrival_radius)
        self.max_expansions = int(max_expansions)
        self.max_runtime_s = float(max_runtime_s)
        self.point_count = len(self.inspection_nodes)
        self.full_mask = (1 << self.point_count) - 1

    def _mst_heuristic(self, node: int, visited_mask: int) -> float:
        vertices = [self.graph.positions[node]]
        vertices.extend(
            self.graph.positions[self.inspection_nodes[index]]
            for index in range(self.point_count)
            if not visited_mask & (1 << index)
        )
        vertices.append(self.graph.positions[self.end_node])
        if len(vertices) <= 1:
            return 0.0
        in_tree = np.zeros(len(vertices), dtype=bool)
        distance = np.full(len(vertices), np.inf)
        distance[0] = 0.0
        total = 0.0
        for _ in range(len(vertices)):
            candidates = np.where(~in_tree, distance, np.inf)
            index = int(np.argmin(candidates))
            if not np.isfinite(candidates[index]):
                break
            in_tree[index] = True
            total += float(candidates[index])
            delta = np.linalg.norm(
                np.asarray(vertices) - np.asarray(vertices[index]),
                axis=1,
            )
            distance = np.minimum(distance, delta)
        return total

    def _near_visited_points(self, node: int, visited_mask: int) -> int:
        point = self.graph.positions[node]
        mask = visited_mask
        for index, inspection_node in enumerate(self.inspection_nodes):
            if mask & (1 << index):
                continue
            if np.linalg.norm(point - self.graph.positions[inspection_node]) <= self.arrival_radius + 1e-12:
                mask |= 1 << index
        return mask

    def _inspection_order(self, states: Sequence[tuple[int, int]]) -> tuple[int, ...]:
        order = []
        previous = 0
        for _, mask in states:
            changed = mask ^ previous
            for index in range(self.point_count):
                if changed & (1 << index):
                    order.append(index + 1)
            previous = mask
        return tuple(order)

    def plan(self) -> PureAStarResult:
        started = time.monotonic()
        initial = (int(self.start_node), self._near_visited_points(self.start_node, 0))
        frontier = []
        tie = itertools.count()
        heapq.heappush(
            frontier,
            (self._mst_heuristic(initial[0], initial[1]), 0.0, next(tie), initial),
        )
        best = {initial: 0.0}
        parent = {initial: None}
        expansions = 0
        generated = 1

        while frontier:
            if expansions >= self.max_expansions:
                return self._failure("expansion_limit", expansions, generated, started)
            if time.monotonic() - started >= self.max_runtime_s:
                return self._failure("time_limit", expansions, generated, started)
            _, cost, _, state = heapq.heappop(frontier)
            if cost != best.get(state):
                continue
            node, mask = state
            if (
                mask == self.full_mask
                and np.linalg.norm(
                    self.graph.positions[node] - self.graph.positions[self.end_node]
                ) <= self.arrival_radius + 1e-12
            ):
                states = []
                cursor = state
                while cursor is not None:
                    states.append(cursor)
                    cursor = parent[cursor]
                states.reverse()
                return PureAStarResult(
                    "success",
                    tuple(item[0] for item in states),
                    self._inspection_order(states),
                    float(cost),
                    expansions,
                    generated,
                    time.monotonic() - started,
                    True,
                    "shortest distance-only trajectory on the 25 m graph",
                )

            expansions += 1
            for destination in self.graph.neighbors[node]:
                if destination == node:
                    continue
                edge = float(np.linalg.norm(
                    self.graph.positions[destination] - self.graph.positions[node]
                ))
                next_state = (
                    int(destination),
                    self._near_visited_points(destination, mask),
                )
                next_cost = cost + edge
                if next_cost >= best.get(next_state, float("inf")):
                    continue
                best[next_state] = next_cost
                parent[next_state] = state
                generated += 1
                heapq.heappush(
                    frontier,
                    (
                        next_cost + self._mst_heuristic(*next_state),
                        next_cost,
                        next(tie),
                        next_state,
                    ),
                )
        return self._failure("exhausted", expansions, generated, started)

    def _failure(self, status: str, expansions: int, generated: int, started: float):
        return PureAStarResult(
            status, (), (), 0.0, expansions, generated,
            time.monotonic() - started, False,
            "distance-only A* exhausted without a complete trajectory",
        )


@dataclass(frozen=True)
class PurePSOResult:
    status: str
    inspection_order: tuple[int, ...]
    path_distance: float
    iterations_run: int
    best_iteration: int
    swarm_size: int
    seed: int
    convergence: tuple[float, ...]
    message: str

    @property
    def success(self) -> bool:
        return self.status == "success"


def _route_distance(order: Sequence[int], points: np.ndarray) -> float:
    route = [points[0]] + [points[int(node)] for node in order] + [points[-1]]
    return float(sum(np.linalg.norm(route[index + 1] - route[index]) for index in range(len(route) - 1)))


def solve_pure_pso(
    *,
    points: np.ndarray,
    seed: int = 1,
    swarm: int = 50,
    iterations: int = 100,
    inertia: float = 0.72,
    cognitive: float = 1.49,
    social: float = 1.49,
    patience: int = 20,
) -> PurePSOResult:
    """Standard continuous PSO whose decoded permutation defines the path."""
    points = np.asarray(points, dtype=float)
    point_count = len(points) - 2
    rng = np.random.default_rng(seed)
    position = rng.uniform(-1.0, 1.0, size=(swarm, point_count))
    velocity = np.zeros_like(position)
    personal_best = position.copy()
    personal_cost = np.full(swarm, np.inf)
    best_position = None
    best_cost = np.inf
    best_iteration = -1
    stale = 0
    convergence = []

    def decode(row):
        return tuple(int(index) + 1 for index in np.argsort(row, kind="stable"))

    for iteration in range(int(iterations)):
        for index in range(swarm):
            order = decode(position[index])
            cost = _route_distance(order, points)
            if cost < personal_cost[index]:
                personal_cost[index] = cost
                personal_best[index] = position[index].copy()
            if cost < best_cost:
                best_cost = cost
                best_position = position[index].copy()
                best_iteration = iteration
                stale = -1
        stale += 1
        convergence.append(float(best_cost))
        if stale >= patience:
            break
        r1 = rng.random((swarm, point_count))
        r2 = rng.random((swarm, point_count))
        velocity = (
            inertia * velocity
            + cognitive * r1 * (personal_best - position)
            + social * r2 * (best_position - position)
        )
        velocity = np.clip(velocity, -0.5, 0.5)
        position = np.clip(position + velocity, -1.0, 1.0)

    return PurePSOResult(
        "success",
        decode(best_position),
        float(best_cost),
        len(convergence),
        int(best_iteration),
        int(swarm),
        int(seed),
        tuple(convergence),
        "PSO minimized a radio-map-free piecewise-linear path length",
    )


def _path_policy(world, route_points: Sequence[np.ndarray]):
    route = [np.asarray(point, dtype=float) for point in route_points]
    cursor = 0

    def act(_state):
        nonlocal cursor
        while True:
            if not bool(world.transmit):
                return np.zeros(2)
            if cursor >= len(route):
                return np.zeros(2)
            here = np.array([world.UAVs[0].x, world.UAVs[0].y], dtype=float)
            if np.linalg.norm(route[cursor] - here) <= 1e-8:
                cursor += 1
                continue
            delta = route[cursor] - here
            return np.array(
                [math.atan2(delta[1], delta[0]), min(float(np.linalg.norm(delta)), world.dist_max)],
                dtype=float,
            )

    return act


def replay_route(
    *,
    world,
    context,
    route_points: Sequence[np.ndarray],
) -> tuple[dict, dict]:
    with contextlib.redirect_stdout(io.StringIO()):
        return rollout(context, world, _path_policy(world, route_points))


def points_with_start_end(users: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    return np.vstack([np.asarray(start, dtype=float), users, np.asarray(end, dtype=float)])


def pso_route_points(
    *,
    start: np.ndarray,
    users: np.ndarray,
    end: np.ndarray,
    order: Sequence[int],
) -> tuple[np.ndarray, ...]:
    points = [np.asarray(start, dtype=float)]
    points.extend(np.asarray(users[int(node) - 1], dtype=float) for node in order)
    points.append(np.asarray(end, dtype=float))
    return tuple(points)
