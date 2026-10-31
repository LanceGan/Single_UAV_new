"""QoS-constrained A* on the 25 m lattice used by :mod:`World`.

Coordinates use the legacy World unit (100 m), so a 25 m move is 0.25 and
the 50 m arrival radius is 0.5.  Payload values are always retained as real
numbers; quantization is used only to index conservative dominance sets.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import itertools
import math
import time
from typing import Mapping, Sequence

import numpy as np


EPSILON = 1e-9


@dataclass(frozen=True)
class PlannerState:
    node: int
    leg: int
    remaining_mb: float
    outage_slots: int
    elapsed_slots: int


@dataclass(frozen=True)
class SlotTransition:
    state: PlannerState
    cost_s: float
    uploaded_mb: float
    pending_at_start: bool
    arrived: bool
    success: bool


@dataclass(frozen=True)
class DominanceLabel:
    state: PlannerState
    cost_s: float


@dataclass(frozen=True)
class PlannerResult:
    status: str
    node_path: tuple[int, ...]
    state_path: tuple[PlannerState, ...]
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
class SpatialGraph:
    positions: np.ndarray
    neighbors: tuple[tuple[int, ...], ...]
    grid_shape: tuple[int, int]
    spacing: float
    special_nodes: Mapping[str, int]
    _grid_lookup: Mapping[tuple[float, float], int]

    @classmethod
    def build(
        cls,
        *,
        length: float,
        width: float,
        spacing: float = 0.25,
        special_points: Mapping[str, Sequence[float]] | None = None,
    ) -> "SpatialGraph":
        if length <= 0 or width <= 0 or spacing <= 0:
            raise ValueError("length, width and spacing must be positive")
        nx = int(round(length / spacing)) + 1
        ny = int(round(width / spacing)) + 1
        if not math.isclose((nx - 1) * spacing, length, abs_tol=1e-10):
            raise ValueError("length must be an integer multiple of spacing")
        if not math.isclose((ny - 1) * spacing, width, abs_tol=1e-10):
            raise ValueError("width must be an integer multiple of spacing")

        positions: list[tuple[float, float]] = []
        grid_lookup: dict[tuple[float, float], int] = {}
        for y_index in range(ny):
            for x_index in range(nx):
                point = (round(x_index * spacing, 12), round(y_index * spacing, 12))
                grid_lookup[point] = len(positions)
                positions.append(point)

        adjacency = [set((node,)) for node in range(len(positions))]
        for y_index in range(ny):
            for x_index in range(nx):
                node = y_index * nx + x_index
                if x_index > 0:
                    adjacency[node].add(node - 1)
                if x_index + 1 < nx:
                    adjacency[node].add(node + 1)
                if y_index > 0:
                    adjacency[node].add(node - nx)
                if y_index + 1 < ny:
                    adjacency[node].add(node + nx)

        specials: dict[str, int] = {}
        for name, raw_point in (special_points or {}).items():
            point = tuple(float(value) for value in raw_point)
            if len(point) != 2:
                raise ValueError(f"special point {name!r} must have two coordinates")
            if not (0.0 <= point[0] <= length and 0.0 <= point[1] <= width):
                raise ValueError(f"special point {name!r} lies outside the map")
            special_node = len(positions)
            positions.append(point)
            adjacency.append({special_node})
            specials[name] = special_node
            for grid_node in range(nx * ny):
                distance = math.dist(point, positions[grid_node])
                if distance <= spacing + 1e-12:
                    adjacency[special_node].add(grid_node)
                    adjacency[grid_node].add(special_node)
            if len(adjacency[special_node]) == 1:
                raise ValueError(f"special point {name!r} has no grid node within one step")

        return cls(
            positions=np.asarray(positions, dtype=float),
            neighbors=tuple(tuple(sorted(nodes)) for nodes in adjacency),
            grid_shape=(ny, nx),
            spacing=float(spacing),
            special_nodes=specials,
            _grid_lookup=grid_lookup,
        )

    def grid_node_at(self, point: Sequence[float]) -> int:
        key = tuple(round(float(value), 12) for value in point)
        return self._grid_lookup[key]


def arrival_disk_heuristic(
    position: Sequence[float],
    target: Sequence[float],
    *,
    arrival_radius: float = 0.5,
    world_unit_m: float = 100.0,
    speed_m_s: float = 50.0,
) -> float:
    """Admissible travel-time lower bound to the target's arrival disk."""
    center_distance = math.dist(position, target)
    remaining_m = max(0.0, center_distance - arrival_radius) * world_unit_m
    return remaining_m / speed_m_s


def _data_rate_mb_s(snr_db: float, bandwidth_mhz: float = 1.0) -> float:
    return bandwidth_mhz * math.log2(1.0 + 10.0 ** (float(snr_db) / 10.0))


def transition_slot(
    state: PlannerState,
    *,
    destination: int,
    positions: np.ndarray,
    targets: np.ndarray,
    payloads_mb: np.ndarray,
    snr_db_by_node: np.ndarray,
    outage_by_node: np.ndarray,
    delta_t: float = 0.5,
    arrival_radius: float = 0.5,
    qos_slots: int = 200,
    horizon_slots: int = 1000,
) -> SlotTransition | None:
    """Apply one slot in exactly the order used by ``World.step_inside``.

    Movement is represented by ``destination``.  The destination map sample
    is used for outage and upload.  Upload/delay depend on payload pending at
    slot start, QoS is checked next, and arrival/new payload generation last.
    ``None`` denotes a transition that World would terminate as infeasible.
    """
    if state.leg >= len(targets) or state.elapsed_slots >= horizon_slots:
        return None
    if not 0 <= destination < len(positions):
        raise IndexError("destination node is outside positions")

    elapsed = state.elapsed_slots + 1
    if elapsed > horizon_slots:
        return None
    outage_value = float(outage_by_node[destination])
    outage_slots = state.outage_slots + int(outage_value >= 0.5)

    pending = state.remaining_mb > EPSILON
    remaining = max(0.0, float(state.remaining_mb))
    uploaded = 0.0
    if pending:
        uploaded = min(remaining, _data_rate_mb_s(snr_db_by_node[destination]) * delta_t)
        remaining = max(0.0, remaining - uploaded)
        if remaining <= EPSILON:
            remaining = 0.0
    cost = delta_t + (delta_t if pending else 0.0)

    # World rejects outage strictly greater than 100 s, after uploading but
    # before accepting an arrival in the same slot.
    if outage_slots > qos_slots:
        return None

    arrived = math.dist(positions[destination], targets[state.leg]) <= arrival_radius + 1e-12
    leg = state.leg
    success = False
    if arrived:
        # Entering a target with data still pending is an immediate failed
        # episode in World and therefore never belongs to a feasible path.
        if remaining > EPSILON:
            return None
        if leg == len(targets) - 1:
            leg += 1
            success = True
        else:
            if leg >= len(payloads_mb):
                raise ValueError("payloads_mb must provide one payload for each inspection target")
            remaining = max(0.0, float(payloads_mb[leg]))
            leg += 1

    next_state = PlannerState(
        node=int(destination),
        leg=leg,
        remaining_mb=remaining,
        outage_slots=outage_slots,
        elapsed_slots=elapsed,
    )
    return SlotTransition(next_state, cost, uploaded, pending, arrived, success)


class DominanceTable:
    """Sparse Pareto sets of labels at the same (node, leg).

    A label is dropped only when another label at the same node and leg is no worse
    in every resource: remaining payload, cost, QoS usage and elapsed slots. Payload
    stays a real number; ``bucket_mb`` is kept for configuration compatibility.
    """

    def __init__(self, bucket_mb: float = 1.0):
        if bucket_mb <= 0:
            raise ValueError("bucket_mb must be positive")
        self.bucket_mb = float(bucket_mb)
        self._labels: dict[tuple[int, int], list[DominanceLabel]] = {}

    @staticmethod
    def _key(label: DominanceLabel) -> tuple[int, int]:
        return label.state.node, label.state.leg

    @staticmethod
    def dominates(left: DominanceLabel, right: DominanceLabel) -> bool:
        a, b = left.state, right.state
        return (
            a.remaining_mb <= b.remaining_mb + EPSILON
            and left.cost_s <= right.cost_s + EPSILON
            and a.outage_slots <= b.outage_slots
            and a.elapsed_slots <= b.elapsed_slots
        )

    def accept(self, label: DominanceLabel) -> bool:
        key = self._key(label)
        current = self._labels.get(key, [])
        if any(self.dominates(existing, label) for existing in current):
            return False
        self._labels[key] = [existing for existing in current if not self.dominates(label, existing)] + [label]
        return True

    def contains(self, label: DominanceLabel) -> bool:
        return label in self._labels.get(self._key(label), ())


@dataclass(frozen=True)
class _Record:
    label: DominanceLabel
    parent: int | None


class AStarPlanner:
    def __init__(
        self,
        *,
        graph: SpatialGraph,
        start_node: int,
        targets: np.ndarray,
        payloads_mb: np.ndarray,
        snr_db_by_node: np.ndarray,
        outage_by_node: np.ndarray,
        bucket_mb: float = 1.0,
        delta_t: float = 0.5,
        arrival_radius: float = 0.5,
        qos_slots: int = 200,
        horizon_slots: int = 1000,
        max_expansions: int = 500_000,
        max_labels: int = 1_000_000,
        max_runtime_s: float = 600.0,
        heuristic: str = "mission",
        step_distance: float = 0.25,
    ):
        if heuristic not in ("mission", "leg", "zero"):
            raise ValueError("heuristic must be 'mission', 'leg' or 'zero'")
        self.heuristic = heuristic
        self.step_distance = float(step_distance)
        self.graph = graph
        self.start_node = int(start_node)
        self.targets = np.asarray(targets, dtype=float)
        self.payloads_mb = np.asarray(payloads_mb, dtype=float)
        self.snr_db_by_node = np.asarray(snr_db_by_node, dtype=float)
        self.outage_by_node = np.asarray(outage_by_node)
        self.bucket_mb = float(bucket_mb)
        self.delta_t = float(delta_t)
        self.arrival_radius = float(arrival_radius)
        self.qos_slots = int(qos_slots)
        self.horizon_slots = int(horizon_slots)
        self.max_expansions = int(max_expansions)
        self.max_labels = int(max_labels)
        self.max_runtime_s = float(max_runtime_s)
        if not 0 <= self.start_node < len(graph.positions):
            raise ValueError("start_node is outside graph")
        if self.targets.ndim != 2 or self.targets.shape[1] != 2 or len(self.targets) == 0:
            raise ValueError("targets must be a non-empty Nx2 array")
        if len(self.payloads_mb) != len(self.targets) - 1:
            raise ValueError("payloads_mb must align with all non-endpoint targets")
        if len(self.snr_db_by_node) != len(graph.positions) or len(self.outage_by_node) != len(graph.positions):
            raise ValueError("radio arrays must align with graph positions")
        # Upload depends only on the destination node, so the best node bounds every slot.
        self.max_upload_mb_per_slot = max(
            _data_rate_mb_s(float(np.max(self.snr_db_by_node))) * self.delta_t, EPSILON)
        # Each later leg starts inside the previous arrival disk and ends inside the next one.
        self._goal_leg = len(self.targets)
        self._later_leg_flight_slots = [
            self._flight_slots(float(np.linalg.norm(self.targets[k + 1] - self.targets[k])) - 2 * self.arrival_radius)
            for k in range(len(self.targets) - 1)
        ]

    def _flight_slots(self, distance: float) -> int:
        # Arrival is only checked after a slot, so every leg takes at least one slot.
        return max(1, math.ceil(max(0.0, distance) / self.step_distance - 1e-9))

    def _upload_slots(self, payload_mb: float) -> int:
        if payload_mb <= EPSILON:
            return 0
        return math.ceil(payload_mb / self.max_upload_mb_per_slot - 1e-9)

    def _heuristic(self, state: PlannerState) -> float:
        """Admissible lower bound on the remaining T + tau cost.

        Per leg, the slots spent are at least max(flight slots, slots that start with
        pending data), and every pending slot adds one more delta_t of delay.
        """
        if state.leg >= self._goal_leg or self.heuristic == "zero":
            return 0.0
        if self.heuristic == "leg":
            return arrival_disk_heuristic(
                self.graph.positions[state.node],
                self.targets[state.leg],
                arrival_radius=self.arrival_radius,
            )
        distance = float(np.linalg.norm(self.graph.positions[state.node] - self.targets[state.leg]))
        upload = self._upload_slots(state.remaining_mb)
        slots = max(self._flight_slots(distance - self.arrival_radius), upload) + upload
        for leg in range(state.leg, self._goal_leg - 1):
            upload = self._upload_slots(float(self.payloads_mb[leg]))
            slots += max(self._later_leg_flight_slots[leg], upload) + upload
        return slots * self.delta_t

    @staticmethod
    def _path(records: list[_Record], record_id: int) -> tuple[tuple[int, ...], tuple[PlannerState, ...]]:
        states: list[PlannerState] = []
        while record_id is not None:
            record = records[record_id]
            states.append(record.label.state)
            record_id = record.parent
        states.reverse()
        return tuple(state.node for state in states), tuple(states)

    def plan(self, initial_state: PlannerState | None = None, stop_leg: int | None = None) -> PlannerResult:
        """Search from ``initial_state`` until leg ``stop_leg`` is reached (default: mission end)."""
        started = time.monotonic()
        self._goal_leg = len(self.targets) if stop_leg is None else int(stop_leg)
        if initial_state is None:
            initial_state = PlannerState(self.start_node, 0, 0.0, 0, 0)
        initial_label = DominanceLabel(initial_state, 0.0)
        dominance = DominanceTable(self.bucket_mb)
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
                    state.outage_slots,
                    state.leg,
                    state.node,
                    next(tie_breaker),
                    record_id,
                ),
            )

        push(0)
        expansions = 0
        generated = 1
        best_record = 0

        def best_key(record_id: int) -> tuple[float, ...]:
            state = records[record_id].label.state
            return (
                -float(state.leg),
                self._heuristic(state),
                state.remaining_mb,
                records[record_id].label.cost_s,
                float(state.node),
            )

        def finish(status: str, message: str, *, record_id: int | None = None, optimal: bool = False) -> PlannerResult:
            # Read best_record at call time; a default argument would freeze it at the start node.
            if record_id is None:
                record_id = best_record
            node_path, state_path = self._path(records, record_id)
            return PlannerResult(
                status=status,
                node_path=node_path,
                state_path=state_path,
                planner_cost_s=records[record_id].label.cost_s,
                expansions=expansions,
                generated_labels=generated,
                runtime_s=time.monotonic() - started,
                optimality_proven=optimal,
                message=message,
            )

        while heap:
            if expansions >= self.max_expansions:
                return finish("expansion_limit", "maximum expansions reached before a feasible path was found")
            if time.monotonic() - started >= self.max_runtime_s:
                return finish("time_limit", "wall-clock limit reached before a feasible path was found")

            *_, record_id = heapq.heappop(heap)
            record = records[record_id]
            if not dominance.contains(record.label):
                continue
            state = record.label.state
            if state.leg >= self._goal_leg:
                return finish(
                    "success",
                    "feasible path found and popped from the A* queue",
                    record_id=record_id,
                    optimal=True,
                )
            if state.elapsed_slots >= self.horizon_slots:
                continue
            expansions += 1

            for destination in self.graph.neighbors[state.node]:
                transition = transition_slot(
                    state,
                    destination=destination,
                    positions=self.graph.positions,
                    targets=self.targets,
                    payloads_mb=self.payloads_mb,
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
                    return finish("memory_limit", "label memory cap reached before a feasible path was found")
                label = DominanceLabel(transition.state, record.label.cost_s + transition.cost_s)
                if not dominance.accept(label):
                    continue
                records.append(_Record(label, record_id))
                generated += 1
                new_id = len(records) - 1
                if best_key(new_id) < best_key(best_record):
                    best_record = new_id
                push(new_id)

        return finish("exhausted", "all reachable labels were exhausted without a feasible path")

    def plan_sequential(self) -> PlannerResult:
        """Solve one leg at a time, each optimally from where the previous leg arrived.

        Faster than the joint search but greedy across legs, so the result is a
        feasible upper bound and never claims global optimality.
        """
        state = PlannerState(self.start_node, 0, 0.0, 0, 0)
        node_path: list[int] = [state.node]
        state_path: list[PlannerState] = [state]
        cost = runtime = 0.0
        expansions = generated = 0
        for leg in range(1, len(self.targets) + 1):
            segment = self.plan(initial_state=state, stop_leg=leg)
            node_path.extend(segment.node_path[1:])
            state_path.extend(segment.state_path[1:])
            cost += segment.planner_cost_s
            expansions += segment.expansions
            generated += segment.generated_labels
            runtime += segment.runtime_s
            if not segment.success:
                return PlannerResult(segment.status, tuple(node_path), tuple(state_path), cost,
                                     expansions, generated, runtime, False,
                                     f"leg {leg}: {segment.message}")
            state = segment.state_path[-1]
        self._goal_leg = len(self.targets)
        return PlannerResult("success", tuple(node_path), tuple(state_path), cost, expansions, generated,
                             runtime, False, "feasible path found leg by leg; each leg optimal given its start")


def sample_radio_maps(graph: SpatialGraph, radio_maps) -> tuple[np.ndarray, np.ndarray]:
    """Precompute destination radio values once for all search transitions."""
    snr = np.empty(len(graph.positions), dtype=float)
    outage = np.empty(len(graph.positions), dtype=bool)
    for node, point in enumerate(graph.positions):
        point_km = point / 10.0
        snr[node] = radio_maps.a2g_snr_db(point_km)
        outage[node] = radio_maps.g2a_outage_probability(point_km) >= 0.5
    return snr, outage

