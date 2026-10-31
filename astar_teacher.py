"""Use the QoS-constrained A* solution as a demonstration teacher.

The A* path for a fixed ``(level, order)`` pair is planned once and cached, then
replayed one edge at a time through the real :class:`World`. Continuous agents
receive the edge as a Variant-A action; discrete agents snap it to the nearest
codec action. This replaces MapGuide as the demonstration source for the *_bc
agents, so the learned policy is pulled toward the near-optimal offline plan.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from astar_planner import AStarPlanner, SpatialGraph, sample_radio_maps
from fixed_radio_maps import FixedRadioMaps

ROOT = Path(__file__).resolve().parent
START = np.array([2.62, 2.65], dtype=float)
END = np.array([18.95, 11.12], dtype=float)
DEFAULT_CACHE_DIR = ROOT / "journal_results" / "astar-teacher"


def _map_hashes() -> dict[str, str]:
    data_dir = ROOT / "results" / "datas"
    return {
        name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
        for name in ("Radio_datas.npz", "Radio_datas_A2G.npz")
    }


def _plan_sequential_positions(level: str, order: Sequence[int]) -> tuple[np.ndarray, dict]:
    from training_config import load_fixed_payloads

    payloads = load_fixed_payloads(level)
    order = tuple(int(node) for node in order)
    users = np.loadtxt(ROOT / "results" / "datas" / "Users_6.txt")[:, :2]
    specials = {"start": START, "end": END}
    specials.update({f"inspection_{i}": point for i, point in enumerate(users, start=1)})
    graph = SpatialGraph.build(length=20.0, width=20.0, spacing=0.25, special_points=specials)
    targets = np.vstack([users[node - 1] for node in order] + [END])
    leg_payloads = np.asarray([payloads[node - 1] for node in order], dtype=float)
    maps = FixedRadioMaps(ROOT / "results" / "datas")
    snr, outage = sample_radio_maps(graph, maps)
    planner = AStarPlanner(
        graph=graph,
        start_node=graph.special_nodes["start"],
        targets=targets,
        payloads_mb=leg_payloads,
        snr_db_by_node=snr,
        outage_by_node=outage,
    )
    result = planner.plan_sequential()
    if not result.success:
        raise RuntimeError(f"A* teacher planning failed for {level}: {result.message}")
    positions = graph.positions[list(result.node_path)]
    meta = {
        "level": level,
        "order": list(order),
        "objective_s": float(result.planner_cost_s),
        "expansions": int(result.expansions),
        "runtime_s": float(result.runtime_s),
        "map_hashes": _map_hashes(),
    }
    return positions, meta


def load_or_plan_teacher_path(level: str, order: Sequence[int], *, cache_dir=None,
                              plan_if_missing: bool = True) -> tuple[np.ndarray, dict]:
    """Return ``(positions, meta)`` for the A* teacher of one ``(level, order)``.

    A JSON cache is keyed by level; its stored order must match the request. When
    ``plan_if_missing`` is false and the cached order differs, a ValueError is
    raised so callers never silently mix teachers across experiments.
    """
    order = tuple(int(node) for node in order)
    cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cache_path = cache_dir / f"teacher_{level}.json"
    if cache_path.is_file():
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if tuple(data.get("order", ())) == order:
            positions = np.asarray(data["positions"], dtype=float)
            meta = dict(data)
            meta["cache_dir"] = str(cache_dir)
            meta["from_cache"] = True
            return positions, meta
        if not plan_if_missing:
            raise ValueError(f"cached teacher order {data.get('order')} != requested {order}")

    positions, meta = _plan_sequential_positions(level, order)
    meta["positions"] = positions.tolist()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    meta["cache_dir"] = str(cache_dir)
    meta["from_cache"] = False
    return positions, meta


class AStarTeacher:
    """Replay a fixed A* node path through ``world`` in the policy's action frame."""

    def __init__(self, world, positions: np.ndarray):
        self.world = world
        self.positions = np.asarray(positions, dtype=float)
        if len(self.positions) < 2:
            raise ValueError("teacher path must contain at least two nodes")
        self.index = 0

    def _next_edge(self) -> tuple[float, float]:
        world = self.world
        if self.index >= len(self.positions) - 1:
            return 0.0, 0.0
        current = np.array([world.UAVs[0].x, world.UAVs[0].y], dtype=float)
        desired = self.positions[self.index + 1]
        displacement = desired - current
        distance = float(np.linalg.norm(displacement))
        if distance > world.dist_max + 1e-8:
            raise RuntimeError(f"teacher edge {distance * 100:.1f} m exceeds the 25 m action bound")
        heading = math.atan2(displacement[1], displacement[0]) if distance > 1e-12 else 0.0
        self.index += 1
        return heading, distance

    def action(self) -> np.ndarray:
        """Variant-A ``[relative heading, distance]`` action for the next edge."""
        return self.world.absolute_to_action(np.array(self._next_edge(), dtype=float))

    def discrete_action(self, codec) -> int:
        """Nearest codec action index for the next edge; advances the replay."""
        return self.nearest_discrete(codec, self.action())

    @staticmethod
    def nearest_discrete(codec, action: np.ndarray) -> int:
        """Snap a continuous ``[relative heading, distance]`` to the nearest codec index."""
        heading, distance = float(action[0]), float(action[1])
        if distance <= 1e-9:
            return 0  # hover
        best, best_cost = 0, math.inf
        for index in range(len(codec)):
            h, d = codec.decode(index)
            dh = abs(math.atan2(math.sin(h - heading), math.cos(h - heading)))
            cost = dh + abs(d - distance) / 0.25
            if cost < best_cost:
                best, best_cost = index, cost
        return best


def d3qn_demo_pairs(level, order, codec, cache_dir=None):
    """Return ``(states, actions)`` demonstration pairs for D3QN.

    The A* path is replayed with its exact continuous actions so the episode stays
    on the optimal trajectory; each visited state is paired with the nearest
    discrete codec action. These pairs feed only the imitation/margin loss.
    """
    import contextlib
    import io

    from World import World
    from training_config import load_fixed_payloads

    positions, _ = load_or_plan_teacher_path(level, order, cache_dir=cache_dir)
    order = tuple(int(node) for node in order)
    payloads = load_fixed_payloads(level)
    with contextlib.redirect_stdout(io.StringIO()):
        world = World(
            length=20, width=20, uav_num=1, dist_max=0.25, delta_t=0.5, t=1000,
            data_sizes=payloads, outage_max_s=100.0,
            ini_loc=START.tolist(), end_loc=END.tolist(),
            users_name=str(ROOT / "results" / "datas" / "Users_6.txt"),
            BS_loc=np.array([[4.77, 4.56, 0.25], [14.89, 4.89, 0.25],
                             [4.63, 13.78, 0.25], [13.48, 14.89, 0.25]]),
            traverse_sequence=list(order), variant="A",
        )
    state, t = world.reset()
    teacher = AStarTeacher(world, positions)
    states, actions = [], []
    done = False
    while not done:
        rel = teacher.action()
        states.append(np.asarray(state, dtype=np.float32))
        actions.append(teacher.nearest_discrete(codec, rel))
        with contextlib.redirect_stdout(io.StringIO()):
            state, _, done, t, _ = world.step_inside(rel, state, t)
    return np.asarray(states, dtype=np.float32), np.asarray(actions, dtype=np.int64)
