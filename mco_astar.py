"""MCO-A* trajectory baseline adapted to the World simulation.

The implemented search follows the load-bearing idea of the MCO-A* baseline
from Cao et al.: the state contains both position and the current contiguous
G2A outage duration, and transitions exceeding the outage budget are pruned.
The original 10 m continuous-time grid is mapped to the repository's 25 m,
0.5 s control lattice so that planner and World replay use identical moves.

The source method does not model uplink offloading.  For integration into the
inspection task, this module therefore pairs its trajectory search with an
A2G-blind upload-hub rule: choose the nearest safe hover location by geographic
detour only.  This is an explicitly disclosed adaptation, not a claim to
reproduce the complete RM-HO energy-oriented cargo workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import itertools
import math
import time
from typing import Sequence

import numpy as np

from astar_planner import SpatialGraph


@dataclass(frozen=True)
class MCOPathResult:
    status: str
    node_path: tuple[int, ...]
    outage_streak_slots_path: tuple[int, ...]
    cost_slots: int
    expansions: int
    generated_states: int
    runtime_s: float
    optimality_proven: bool
    message: str

    @property
    def success(self) -> bool:
        return self.status == "success"


class MCOAStarPlanner:
    """Shortest path subject to a hard cap on contiguous outage slots."""

    def __init__(
        self,
        *,
        graph: SpatialGraph,
        outage_by_node: Sequence[bool],
        qos_slots: int,
        max_expansions: int = 500_000,
        max_runtime_s: float = 120.0,
    ):
        self.graph = graph
        self.outage_by_node = np.asarray(outage_by_node, dtype=bool)
        self.qos_slots = int(qos_slots)
        self.max_expansions = int(max_expansions)
        self.max_runtime_s = float(max_runtime_s)
        if len(self.outage_by_node) != len(graph.positions):
            raise ValueError("outage_by_node must align with graph positions")
        if self.qos_slots < 0:
            raise ValueError("qos_slots must be non-negative")

    def _heuristic(self, node: int, goal_node: int) -> int:
        distance = float(np.linalg.norm(
            self.graph.positions[goal_node] - self.graph.positions[node]
        ))
        return int(math.ceil(max(0.0, distance) / self.graph.spacing - 1e-12))

    def plan(
        self,
        *,
        start_node: int,
        goal_node: int,
        blocked_nodes: Sequence[int] = (),
        initial_outage_streak_slots: int = 0,
    ) -> MCOPathResult:
        started = time.monotonic()
        blocked = set(int(node) for node in blocked_nodes)
        if not 0 <= start_node < len(self.graph.positions):
            raise IndexError("start_node is outside graph")
        if not 0 <= goal_node < len(self.graph.positions):
            raise IndexError("goal_node is outside graph")
        if goal_node in blocked:
            return MCOPathResult(
                "blocked_goal", (), (), 0, 0, 0, 0.0, False,
                "goal node is blocked by the target protection disk",
            )

        initial_streak = max(0, int(initial_outage_streak_slots))
        if initial_streak > self.qos_slots:
            return MCOPathResult(
                "infeasible_start", (), (), 0, 0, 0, 0.0, False,
                "initial contiguous outage already exceeds the budget",
            )

        initial_state = (int(start_node), initial_streak)
        frontier: list[tuple[int, int, int, int, int]] = []
        tie_breaker = itertools.count()
        heapq.heappush(
            frontier,
            (self._heuristic(start_node, goal_node), 0, next(tie_breaker), start_node, initial_streak),
        )
        best_cost = {initial_state: 0}
        parents: dict[tuple[int, int], tuple[int, int] | None] = {initial_state: None}
        expansions = 0
        generated = 1

        while frontier:
            if expansions >= self.max_expansions:
                return MCOPathResult(
                    "expansion_limit", (), (), 0, expansions, generated,
                    time.monotonic() - started, False,
                    "maximum expansions reached before the goal",
                )
            if time.monotonic() - started > self.max_runtime_s:
                return MCOPathResult(
                    "time_limit", (), (), 0, expansions, generated,
                    time.monotonic() - started, False,
                    "wall-clock limit reached before the goal",
                )

            _, cost, _, node, streak = heapq.heappop(frontier)
            state = (node, streak)
            if cost != best_cost.get(state):
                continue
            if node == goal_node:
                states = []
                cursor: tuple[int, int] | None = state
                while cursor is not None:
                    states.append(cursor)
                    cursor = parents[cursor]
                states.reverse()
                return MCOPathResult(
                    "success",
                    tuple(item[0] for item in states),
                    tuple(item[1] for item in states),
                    cost,
                    expansions,
                    generated,
                    time.monotonic() - started,
                    True,
                    "shortest feasible trajectory on the adapted 25 m lattice",
                )

            expansions += 1
            for destination in self.graph.neighbors[node]:
                if destination in blocked and destination != start_node:
                    continue
                next_streak = streak + 1 if self.outage_by_node[destination] else 0
                if next_streak > self.qos_slots:
                    continue
                next_state = (int(destination), next_streak)
                next_cost = cost + 1
                if next_cost >= best_cost.get(next_state, 1 << 60):
                    continue
                best_cost[next_state] = next_cost
                parents[next_state] = state
                generated += 1
                heapq.heappush(
                    frontier,
                    (
                        next_cost + self._heuristic(destination, goal_node),
                        next_cost,
                        next(tie_breaker),
                        destination,
                        next_streak,
                    ),
                )

        return MCOPathResult(
            "exhausted", (), (), 0, expansions, generated,
            time.monotonic() - started, False,
            "all feasible states were exhausted",
        )


class MCOAStarGuide:
    """Deterministic fixed-order guide using adapted MCO-A* trajectories."""

    def __init__(
        self,
        *,
        world,
        graph: SpatialGraph,
        arrival_radius: float = 0.5,
        guard_margin: float = 0.15,
        max_expansions_per_search: int = 500_000,
        max_runtime_s_per_search: float = 120.0,
        qos_slots_override: int | None = None,
        profile: str = "hard-qos",
    ):
        self.world = world
        self.graph = graph
        self.arrival_radius = float(arrival_radius)
        self.guard_margin = float(guard_margin)
        positions_km = graph.positions / 10.0
        self.outage_by_node = np.asarray([
            world.radio_maps.g2a_outage_probability(point) >= 0.5
            for point in positions_km
        ])
        qos_slots = (
            int(round(world.outage_max_s / world.delta_T))
            if qos_slots_override is None
            else int(qos_slots_override)
        )
        self.planner = MCOAStarPlanner(
            graph=graph,
            outage_by_node=self.outage_by_node,
            qos_slots=qos_slots,
            max_expansions=max_expansions_per_search,
            max_runtime_s=max_runtime_s_per_search,
        )
        self.profile = str(profile)
        self.route: list[np.ndarray] = []
        self.index = 0
        self.key = None
        self.search_count = 0
        self.search_expansions = 0
        self.search_generated_states = 0
        self.search_runtime_s = 0.0
        self.search_failures: list[dict[str, object]] = []
        self.plan_history: list[dict[str, object]] = []

    def _record_search(self, result: MCOPathResult, *, purpose: str) -> None:
        self.search_count += 1
        self.search_expansions += result.expansions
        self.search_generated_states += result.generated_states
        self.search_runtime_s += result.runtime_s
        if not result.success:
            self.search_failures.append({
                "purpose": purpose,
                "status": result.status,
                "message": result.message,
            })

    def planner_audit(self) -> dict[str, object]:
        return {
            "algorithm": "MCO-A* adapted to 25 m / 0.5 s lattice",
            "profile": self.profile,
            "qos_constraint": "max_contiguous_outage",
            "qos_slots": self.planner.qos_slots,
            "searches": self.search_count,
            "expansions": self.search_expansions,
            "generated_states": self.search_generated_states,
            "runtime_s": self.search_runtime_s,
            "failures": list(self.search_failures),
            "plans": list(self.plan_history),
            "optimality_scope": "shortest feasible trajectory per planned segment on the adapted lattice",
        }

    def _nearest_node(self, point: Sequence[float]) -> int:
        distances = np.linalg.norm(self.graph.positions - np.asarray(point, dtype=float), axis=1)
        return int(np.argmin(distances))

    def _current_outage_slots(self) -> int:
        return int(round(
            float(self.world.current_contiguous_outage_s) / self.world.delta_T
        ))

    def _target_blocked_nodes(self, target_node: int) -> set[int]:
        target = self.graph.positions[target_node]
        radius = self.arrival_radius + self.guard_margin
        inside = np.linalg.norm(self.graph.positions - target, axis=1) <= radius + 1e-12
        blocked = set(int(node) for node in np.flatnonzero(inside))
        blocked.discard(int(target_node))
        return blocked

    def _plan_upload_hub(
        self,
        *,
        source_node: int,
        target_node: int,
        blocked: set[int],
    ) -> MCOPathResult:
        target = self.graph.positions[target_node]
        source = self.graph.positions[source_node]
        blocked_mask = np.zeros(len(self.graph.positions), dtype=bool)
        blocked_mask[list(blocked)] = True
        candidates = np.flatnonzero(~blocked_mask & ~self.outage_by_node)
        if len(candidates) == 0:
            raise RuntimeError("MCO-A* guide found no safe upload hub")

        detour = (
            np.linalg.norm(self.graph.positions[candidates] - source, axis=1)
            + np.linalg.norm(self.graph.positions[candidates] - target, axis=1)
        )
        order = candidates[np.argsort(detour, kind="stable")]
        failures = []
        for hub in order:
            result = self.planner.plan(
                start_node=source_node,
                goal_node=int(hub),
                blocked_nodes=tuple(blocked),
                initial_outage_streak_slots=self._current_outage_slots(),
            )
            self._record_search(result, purpose="upload_hub")
            if result.success:
                return result
            failures.append(result.status)
        unique = sorted(set(failures))
        raise RuntimeError(
            "MCO-A* guide could not reach any safe upload hub; "
            f"planner statuses={unique}"
        )

    def _plan(self, position: np.ndarray) -> None:
        source_node = self._nearest_node(position)
        target_node = self._nearest_node(self.world.target_loc)
        if self.world.transmit:
            result = self.planner.plan(
                start_node=source_node,
                goal_node=target_node,
                initial_outage_streak_slots=self._current_outage_slots(),
            )
            purpose = "target"
            self._record_search(result, purpose=purpose)
        else:
            blocked = self._target_blocked_nodes(target_node)
            result = self._plan_upload_hub(
                source_node=source_node,
                target_node=target_node,
                blocked=blocked,
            )
            purpose = "upload_hub"
        if not result.success:
            raise RuntimeError(f"MCO-A* route planning failed: {result.message}")
        self.route = [self.graph.positions[node].copy() for node in result.node_path]
        self.index = 0
        self.key = (self.world.Traverse_order, self.world.transmit)
        self.plan_history.append({
            "traverse_order": int(self.world.Traverse_order),
            "purpose": purpose,
            "transmit": bool(self.world.transmit),
            "start_node": source_node,
            "target_node": target_node,
            "goal_node": int(result.node_path[-1]),
            "path_nodes": len(result.node_path),
            "path_steps": len(result.node_path) - 1,
            "outage_streak_end_slots": result.outage_streak_slots_path[-1],
            "status": result.status,
        })

    def action(self) -> np.ndarray:
        world = self.world
        position = np.array([world.UAVs[0].x, world.UAVs[0].y], dtype=float)
        off_route = (
            self.index < len(self.route)
            and np.linalg.norm(self.route[self.index] - position) > 1e-8
        )
        if self.key != (world.Traverse_order, world.transmit) or off_route:
            self._plan(position)

        while (
            self.index < len(self.route)
            and np.linalg.norm(self.route[self.index] - position) <= 1e-8
        ):
            self.index += 1
        if self.index >= len(self.route):
            return np.zeros(2)

        delta = self.route[self.index] - position
        distance = float(np.linalg.norm(delta))
        if distance > world.dist_max + 1e-8:
            raise RuntimeError(
                f"MCO-A* route edge is {distance:.6f}, above World limit {world.dist_max:.6f}"
            )
        self.index += 1
        if distance <= 1e-12:
            return np.zeros(2)
        return np.array([math.atan2(delta[1], delta[0]), distance], dtype=float)
