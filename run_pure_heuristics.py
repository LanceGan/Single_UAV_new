"""Run radio-map-free A* and PSO end-to-end trajectory baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from astar_planner import SpatialGraph
from pure_heuristic_baselines import (
    PureAStarTrajectoryPlanner,
    pso_route_points,
    replay_route,
    solve_pure_pso,
)
from pso_trajectory import build_world, make_context
from training_config import load_fixed_payloads


ROOT = Path(__file__).resolve().parent
START = np.array([2.62, 2.65], dtype=float)
END = np.array([18.95, 11.12], dtype=float)
LEVELS = ("low", "medium", "high")


def _map_hashes() -> dict[str, str]:
    data_dir = ROOT / "results" / "datas"
    return {
        name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
        for name in ("Radio_datas.npz", "Radio_datas_A2G.npz")
    }


def _build_graph(users: np.ndarray) -> SpatialGraph:
    specials = {"start": START, "end": END}
    specials.update({
        f"inspection_{index}": point
        for index, point in enumerate(users, start=1)
    })
    return SpatialGraph.build(
        length=20.0,
        width=20.0,
        spacing=0.25,
        special_points=specials,
    )


def _replay(
    *,
    method: str,
    level: str,
    order,
    route_points,
    planner_record: dict,
    outage_max_s: float,
) -> tuple[dict, dict]:
    payloads = load_fixed_payloads(level)
    world = build_world(payloads, order, outage_max_s)
    context = make_context(payloads, order, outage_max_s)
    report, trace = replay_route(world=world, context=context, route_points=route_points)
    report["algorithm"] = method
    report["policy_mode"] = (
        "radio-map-free geometric trajectory with stop-and-upload at each inspection point"
    )
    report["inspection_order"] = list(order)
    report["payloads_mb"] = np.asarray(payloads, dtype=float).tolist()
    report["objective_s"] = (
        float(report["task_time_s"]) + float(report["transmission_delay_s"])
        if report["success"]
        else None
    )
    report["metrics_source"] = "real World replay"
    report["planner"] = planner_record
    report["radio_maps_used_for_planning"] = False
    report["payload_used_for_planning"] = False
    report["upload_protocol"] = (
        "After collecting a point's payload, hover at that point until the buffer "
        "clears; then continue along the precomputed geometric trajectory."
    )
    report["map_hashes_replay_only"] = _map_hashes()
    return report, trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", nargs="+", choices=LEVELS, default=LEVELS)
    parser.add_argument("--outage-max-s", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "journal_results" / "v4-deterministic" / "pure-heuristics-6",
    )
    args = parser.parse_args()
    args.output = args.output.resolve()

    users = np.loadtxt(ROOT / "results" / "datas" / "Users_6.txt")[:, :2]
    graph = _build_graph(users)
    astar = PureAStarTrajectoryPlanner(
        graph=graph,
        start_node=graph.special_nodes["start"],
        inspection_nodes=[graph.special_nodes[f"inspection_{index}"] for index in range(1, 7)],
        end_node=graph.special_nodes["end"],
    ).plan()
    if not astar.success:
        raise RuntimeError(f"pure A* failed: {astar.message}")
    astar_route = tuple(graph.positions[node] for node in astar.node_path)
    astar_record = {
        "status": astar.status,
        "inspection_order": list(astar.inspection_order),
        "path_distance_world_units": astar.path_distance,
        "expansions": astar.expansions,
        "generated_states": astar.generated_states,
        "runtime_s": astar.runtime_s,
        "optimality_scope": "shortest distance-only trajectory on the 25 m graph",
    }

    pso = solve_pure_pso(
        points=np.vstack([START, users, END]),
        seed=args.seed,
    )
    pso_route = pso_route_points(
        start=START,
        users=users,
        end=END,
        order=pso.inspection_order,
    )
    pso_record = {
        "status": pso.status,
        "inspection_order": list(pso.inspection_order),
        "path_distance_world_units": pso.path_distance,
        "iterations_run": pso.iterations_run,
        "best_iteration": pso.best_iteration,
        "swarm_size": pso.swarm_size,
        "seed": pso.seed,
        "convergence": list(pso.convergence),
        "optimality_scope": "best sampled permutation under a geometry-only objective",
    }

    summary = {}
    for level in args.levels:
        for method, order, route, planner_record in (
            ("Pure A*", astar.inspection_order, astar_route, astar_record),
            ("Pure PSO", pso.inspection_order, pso_route, pso_record),
        ):
            report, trace = _replay(
                method=method,
                level=level,
                order=order,
                route_points=route,
                planner_record=planner_record,
                outage_max_s=args.outage_max_s,
            )
            slug = "astar" if method == "Pure A*" else "pso"
            target = args.output / slug / level
            target.mkdir(parents=True, exist_ok=True)
            (target / "metrics.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            np.savez(target / "trajectory.npz", **trace)
            summary.setdefault(level, {})[method] = {
                "success": report["success"],
                "termination_reason": report["termination_reason"],
                "inspection_order": list(order),
                "path_distance_world_units": planner_record["path_distance_world_units"],
                "task_time_s": report["task_time_s"],
                "transmission_delay_s": report["transmission_delay_s"],
                "objective_s": report["objective_s"],
                "outage_s": report["outage_time_s"],
                "visited_inspection_points": report["visited_inspection_points"],
                "uploaded_mb": report["uploaded_mb"],
            }
            print(json.dumps({level: {method: summary[level][method]}}, ensure_ascii=False))

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
