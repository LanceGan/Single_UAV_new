"""Joint sequence-and-trajectory A* on the discrete UAV planning graph.

The search state carries the visited inspection set, pending payload, continuous
outage streak and elapsed slots. No inspection order is supplied: entering an
unvisited inspection disk while the previous payload is clear collects that
point, so the A* search determines both the visit order and the flight path.
Optimality claims are limited to this discretized graph and accounting model.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from collections import deque
import heapq
import itertools
import math
import time

import numpy as np

from astar_planner import EPSILON, SpatialGraph, _data_rate_mb_s


@dataclass(frozen=True)
class JointPlannerState:
    node: int
    visited_mask: int
    remaining_mb: float
    outage_streak_slots: int
    elapsed_slots: int
    target_index: int


@dataclass(frozen=True)
class JointSlotTransition:
    state: JointPlannerState
    cost_s: float
    uploaded_mb: float
    pending_at_start: bool
    newly_visited: int | None
    success: bool


@dataclass(frozen=True)
class JointPlannerResult:
    status: str
    node_path: tuple[int, ...]
    state_path: tuple[JointPlannerState, ...]
    inspection_order: tuple[int, ...]
    planner_cost_s: float
    expansions: int
    generated_labels: int
    runtime_s: float
    optimality_proven: bool
    message: str

    @property
    def success(self) -> bool:
        return self.status == "success"


@dataclass(frozen=True)
class _JointLabel:
    state: JointPlannerState
    cost_s: float


class JointDominanceTable:
    """Pareto labels at each ``(node, visited_mask, selected_target)``."""

    def __init__(self):
        self._labels: dict[tuple[int, int, int], list[_JointLabel]] = {}

    @staticmethod
    def dominates(left: _JointLabel, right: _JointLabel) -> bool:
        a, b = left.state, right.state
        return (
            a.remaining_mb <= b.remaining_mb + EPSILON
            and left.cost_s <= right.cost_s + EPSILON
            and a.outage_streak_slots <= b.outage_streak_slots
            and a.elapsed_slots <= b.elapsed_slots
        )

    def accept(self, label: _JointLabel) -> bool:
        key = label.state.node, label.state.visited_mask, label.state.target_index
        current = self._labels.get(key, [])
        if any(self.dominates(existing, label) for existing in current):
            return False
        self._labels[key] = [
            existing for existing in current if not self.dominates(label, existing)
        ] + [label]
        return True

    def contains(self, label: _JointLabel) -> bool:
        key = label.state.node, label.state.visited_mask, label.state.target_index
        return label in self._labels.get(key, ())


def joint_transition(
    state: JointPlannerState,
    *,
    destination: int,
    positions: np.ndarray,
    inspection_points: np.ndarray,
    payloads_mb: np.ndarray,
    end_position: np.ndarray,
    snr_db_by_node: np.ndarray,
    outage_by_node: np.ndarray,
    delta_t: float = 0.5,
    arrival_radius: float = 0.5,
    qos_slots: int = 12,
    horizon_slots: int = 1000,
) -> JointSlotTransition | None:
    """Advance one slot while allowing any unvisited point to be collected."""
    if state.elapsed_slots >= horizon_slots or state.target_index < 0:
        return None
    if not 0 <= destination < len(positions):
        raise IndexError("destination node is outside positions")

    elapsed = state.elapsed_slots + 1
    if elapsed > horizon_slots:
        return None
    outage_streak_slots = (
        state.outage_streak_slots + 1
        if float(outage_by_node[destination]) >= 0.5
        else 0
    )
    if outage_streak_slots > qos_slots:
        return None

    pending = state.remaining_mb > EPSILON
    remaining = max(0.0, float(state.remaining_mb))
    uploaded = 0.0
    if pending:
        uploaded = min(
            remaining,
            _data_rate_mb_s(float(snr_db_by_node[destination])) * delta_t,
        )
        remaining = max(0.0, remaining - uploaded)
        if remaining <= EPSILON:
            remaining = 0.0
    cost = delta_t + (delta_t if pending else 0.0)

    visited_mask = int(state.visited_mask)
    full_mask = (1 << len(payloads_mb)) - 1
    newly_visited = None
    next_target = int(state.target_index)
    point = np.asarray(positions[destination], dtype=float)
    inspection_count = len(payloads_mb)
    if next_target < inspection_count:
        distance = float(np.linalg.norm(
            point - np.asarray(inspection_points[next_target], dtype=float)
        ))
        if distance <= arrival_radius + 1e-12:
            if remaining > EPSILON:
                return None
            newly_visited = next_target
            visited_mask |= 1 << next_target
            remaining = max(0.0, float(payloads_mb[next_target]))
            next_target = -1

    success = bool(
        next_target == inspection_count
        and visited_mask == full_mask
        and remaining <= EPSILON
        and float(np.linalg.norm(
            point - np.asarray(end_position, dtype=float)
        )) <= arrival_radius + 1e-12
    )
    next_state = JointPlannerState(
        node=int(destination),
        visited_mask=visited_mask,
        remaining_mb=remaining,
        outage_streak_slots=outage_streak_slots,
        elapsed_slots=elapsed,
        target_index=next_target,
    )
    return JointSlotTransition(
        state=next_state,
        cost_s=cost,
        uploaded_mb=uploaded,
        pending_at_start=pending,
        newly_visited=newly_visited,
        success=success,
    )


@dataclass(frozen=True)
class _Record:
    label: _JointLabel
    parent: int | None


class JointAStarPlanner:
    """A* whose state lets the inspection order emerge from the trajectory."""

    def __init__(
        self,
        *,
        graph: SpatialGraph,
        start_node: int,
        inspection_points: np.ndarray,
        payloads_mb: np.ndarray,
        end_position: np.ndarray,
        snr_db_by_node: np.ndarray,
        outage_by_node: np.ndarray,
        delta_t: float = 0.5,
        arrival_radius: float = 0.5,
        qos_slots: int = 12,
        horizon_slots: int = 1000,
        max_expansions: int = 2_000_000,
        max_labels: int = 4_000_000,
        max_runtime_s: float = 900.0,
        step_distance: float = 0.25,
        progress_every: int = 0,
        progress_callback=None,
    ):
        self.graph = graph
        self.start_node = int(start_node)
        self.inspection_points = np.asarray(inspection_points, dtype=float)
        self.payloads_mb = np.asarray(payloads_mb, dtype=float)
        self.end_position = np.asarray(end_position, dtype=float)
        self.snr_db_by_node = np.asarray(snr_db_by_node, dtype=float)
        self.outage_by_node = np.asarray(outage_by_node)
        self.delta_t = float(delta_t)
        self.arrival_radius = float(arrival_radius)
        self.qos_slots = int(qos_slots)
        self.horizon_slots = int(horizon_slots)
        self.max_expansions = int(max_expansions)
        self.max_labels = int(max_labels)
        self.max_runtime_s = float(max_runtime_s)
        self.step_distance = float(step_distance)
        self.progress_every = int(progress_every)
        self.progress_callback = progress_callback
        if self.inspection_points.ndim != 2 or self.inspection_points.shape[1] != 2:
            raise ValueError("inspection_points must be an Nx2 array")
        if len(self.payloads_mb) != len(self.inspection_points):
            raise ValueError("payloads_mb must align with inspection_points")
        if len(self.snr_db_by_node) != len(graph.positions):
            raise ValueError("SNR samples must align with graph positions")
        if len(self.outage_by_node) != len(graph.positions):
            raise ValueError("outage samples must align with graph positions")
        self.full_mask = (1 << len(self.inspection_points)) - 1
        self.max_upload_mb_per_slot = max(
            _data_rate_mb_s(float(np.max(self.snr_db_by_node))) * self.delta_t,
            EPSILON,
        )
        self._distance_to_inspection: list[np.ndarray] = []
        for point in self.inspection_points:
            seeds = [
                int(node) for node, position in enumerate(graph.positions)
                if float(np.linalg.norm(position - point)) <= self.arrival_radius + 1e-12
            ]
            self._distance_to_inspection.append(self._multi_source_steps(seeds))
        end_seeds = [
            int(node) for node, position in enumerate(graph.positions)
            if float(np.linalg.norm(position - self.end_position)) <= self.arrival_radius + 1e-12
        ]
        self._distance_to_end = self._multi_source_steps(end_seeds)
        self._target_route_lower_bounds = self._build_target_route_lower_bounds()

    def _multi_source_steps(self, seeds: list[int]) -> np.ndarray:
        distances = np.full(len(self.graph.positions), np.inf, dtype=float)
        queue = deque()
        for node in seeds:
            distances[node] = 0.0
            queue.append(node)
        while queue:
            node = queue.popleft()
            next_distance = distances[node] + 1.0
            for destination in self.graph.neighbors[node]:
                if next_distance < distances[destination]:
                    distances[destination] = next_distance
                    queue.append(destination)
        return distances

    def _build_target_route_lower_bounds(self) -> np.ndarray:
        """Shortest graph route after first reaching each selected inspection disk."""
        count = len(self.inspection_points)
        full_mask = self.full_mask
        table = np.full((count, full_mask + 1), np.inf, dtype=float)
        target_to_target = np.empty((count, count), dtype=float)
        target_to_end = np.empty(count, dtype=float)
        for first in range(count):
            distances = self._distance_to_inspection[first]
            for second in range(count):
                target_to_target[first, second] = float(
                    np.min(distances[self._distance_to_inspection[second] == 0.0])
                )
            target_to_end[first] = float(
                np.min(distances[self._distance_to_end == 0.0])
            )

        for first in range(count):
            others_mask = full_mask ^ (1 << first)
            dp = np.full((full_mask + 1, count), np.inf, dtype=float)
            for target in range(count):
                if others_mask & (1 << target):
                    dp[1 << target, target] = target_to_target[first, target]
            for mask in range(1, full_mask + 1):
                if mask & (1 << first):
                    continue
                for last in range(count):
                    bit = 1 << last
                    if not mask & bit:
                        continue
                    previous_mask = mask ^ bit
                    if previous_mask == 0:
                        continue
                    dp[mask, last] = min(
                        dp[previous_mask, previous]
                        + target_to_target[previous, last]
                        for previous in range(count)
                        if previous_mask & (1 << previous)
                    )
            table[first, 0] = target_to_end[first]
            for mask in range(1, others_mask + 1):
                if mask & (1 << first):
                    continue
                table[first, mask] = min(
                    dp[mask, last] + target_to_end[last]
                    for last in range(count)
                    if mask & (1 << last)
                )
        return table

    def _heuristic(self, state: JointPlannerState) -> float:
        remaining_mask = self.full_mask ^ int(state.visited_mask)
        count = len(self.inspection_points)
        target = int(state.target_index)
        if target == count:
            flight_distance_steps = self._distance_to_end[state.node]
        elif target >= 0:
            flight_distance_steps = (
                self._distance_to_inspection[target][state.node]
                + self._target_route_lower_bounds[
                    target, remaining_mask ^ (1 << target)
                ]
            )
        else:
            candidates = [
                self._distance_to_inspection[index][state.node]
                + self._target_route_lower_bounds[
                    index, remaining_mask ^ (1 << index)
                ]
                for index in range(count)
                if remaining_mask & (1 << index)
            ]
            if not candidates:
                candidates = [self._distance_to_end[state.node]]
            flight_distance_steps = min(candidates)
        flight_slots = int(math.ceil(flight_distance_steps - 1e-12))
        unvisited = [
            index for index in range(len(self.inspection_points))
            if remaining_mask & (1 << index)
        ]
        remaining_data = float(state.remaining_mb) + sum(
            float(self.payloads_mb[index]) for index in unvisited
        )
        upload_slots = int(math.ceil(remaining_data / self.max_upload_mb_per_slot - 1e-12))
        return (max(flight_slots, upload_slots) + upload_slots) * self.delta_t

    @staticmethod
    def _path(records: list[_Record], record_id: int):
        states = []
        while record_id is not None:
            states.append(records[record_id].label.state)
            record_id = records[record_id].parent
        states.reverse()
        return tuple(states)

    @staticmethod
    def _inspection_order(states: tuple[JointPlannerState, ...]) -> tuple[int, ...]:
        order = []
        previous_mask = 0
        for state in states:
            changed = state.visited_mask ^ previous_mask
            for index in range(32):
                if changed & (1 << index):
                    order.append(index + 1)
            previous_mask = state.visited_mask
        return tuple(order)

    def plan(self) -> JointPlannerResult:
        started = time.monotonic()
        initial_state = JointPlannerState(self.start_node, 0, 0.0, 0, 0, -1)
        initial_label = _JointLabel(initial_state, 0.0)
        dominance = JointDominanceTable()
        dominance.accept(initial_label)
        records = [_Record(initial_label, None)]
        tie_breaker = itertools.count()
        heap: list[tuple[float, float, int, int, int, int, int, int]] = []

        def push(record_id: int) -> None:
            state = records[record_id].label.state
            cost = records[record_id].label.cost_s
            heapq.heappush(
                heap,
                (
                    cost + self._heuristic(state),
                    cost,
                    state.elapsed_slots,
                    state.outage_streak_slots,
                    state.visited_mask,
                    state.node,
                    next(tie_breaker),
                    record_id,
                ),
            )

        def finish(status: str, message: str, record_id: int = 0, optimal: bool = False):
            states = self._path(records, record_id)
            return JointPlannerResult(
                status=status,
                node_path=tuple(state.node for state in states),
                state_path=states,
                inspection_order=self._inspection_order(states),
                planner_cost_s=records[record_id].label.cost_s,
                expansions=expansions,
                generated_labels=generated,
                runtime_s=time.monotonic() - started,
                optimality_proven=optimal,
                message=message,
            )

        expansions = 0
        generated = 1
        best_record = 0

        def best_key(record_id: int):
            state = records[record_id].label.state
            return (
                -state.visited_mask.bit_count(),
                state.remaining_mb,
                records[record_id].label.cost_s,
                state.target_index,
                state.node,
            )

        push(0)
        while heap:
            if expansions >= self.max_expansions:
                return finish(
                    "expansion_limit",
                    "maximum expansions reached before the joint goal",
                    record_id=best_record,
                )
            if time.monotonic() - started >= self.max_runtime_s:
                return finish(
                    "time_limit",
                    "wall-clock limit reached before the joint goal",
                    record_id=best_record,
                )
            *_, record_id = heapq.heappop(heap)
            record = records[record_id]
            if not dominance.contains(record.label):
                continue
            state = record.label.state
            if (
                self.progress_callback is not None
                and self.progress_every > 0
                and expansions % self.progress_every == 0
            ):
                best_state = records[best_record].label.state
                self.progress_callback(
                    {
                        "expansions": expansions,
                        "generated_labels": generated,
                        "visited_points": best_state.visited_mask.bit_count(),
                        "best_partial_cost_s": records[best_record].label.cost_s,
                        "partial_order": list(self._inspection_order(
                            self._path(records, best_record)
                        )),
                        "runtime_s": time.monotonic() - started,
                    }
                )
            if (
                state.target_index == len(self.inspection_points)
                and state.visited_mask == self.full_mask
                and state.remaining_mb <= EPSILON
            ):
                distance_to_end = float(
                    np.linalg.norm(self.graph.positions[state.node] - self.end_position)
                )
                if distance_to_end <= self.arrival_radius + 1e-12:
                    return finish(
                        "success",
                        "joint visit order, trajectory and upload completion found",
                        record_id=record_id,
                        optimal=True,
                    )
            if state.elapsed_slots >= self.horizon_slots:
                continue
            expansions += 1
            if state.target_index == -1:
                remaining_mask = self.full_mask ^ state.visited_mask
                candidates = [
                    index for index in range(len(self.inspection_points))
                    if remaining_mask & (1 << index)
                ]
                if not candidates:
                    candidates = [len(self.inspection_points)]
                for target in candidates:
                    selected_state = replace(state, target_index=target)
                    label = _JointLabel(selected_state, record.label.cost_s)
                    if dominance.accept(label):
                        records.append(_Record(label, record_id))
                        generated += 1
                        new_id = len(records) - 1
                        if best_key(new_id) < best_key(best_record):
                            best_record = new_id
                        push(new_id)
                continue
            for destination in self.graph.neighbors[state.node]:
                transition = joint_transition(
                    state,
                    destination=destination,
                    positions=self.graph.positions,
                    inspection_points=self.inspection_points,
                    payloads_mb=self.payloads_mb,
                    end_position=self.end_position,
                    snr_db_by_node=self.snr_db_by_node,
                    outage_by_node=self.outage_by_node,
                    delta_t=self.delta_t,
                    arrival_radius=self.arrival_radius,
                    qos_slots=self.qos_slots,
                    horizon_slots=self.horizon_slots,
                )
                if transition is None:
                    continue
                if len(records) >= self.max_labels:
                    return finish(
                        "memory_limit",
                        "joint search label cap reached",
                        record_id=best_record,
                    )
                label = _JointLabel(
                    transition.state,
                    record.label.cost_s + transition.cost_s,
                )
                if not dominance.accept(label):
                    continue
                records.append(_Record(label, record_id))
                generated += 1
                new_id = len(records) - 1
                if best_key(new_id) < best_key(best_record):
                    best_record = new_id
                push(new_id)
        return finish("exhausted", "all reachable joint states were exhausted", record_id=best_record)
