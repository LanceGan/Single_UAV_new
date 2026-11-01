"""Run the QoS-constrained A* baseline and replay it through the real World."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from astar_planner import AStarPlanner, PlannerResult, SpatialGraph, sample_radio_maps
from fixed_radio_maps import FixedRadioMaps
from sequence_runner import solve_ga_order
from training_config import load_fixed_payloads
from World import World


ROOT = Path(__file__).resolve().parent
START = np.array([2.62, 2.65], dtype=float)
END = np.array([18.95, 11.12], dtype=float)
BASE_STATIONS = np.array(
    [[4.77, 4.56, 0.25], [14.89, 4.89, 0.25], [4.63, 13.78, 0.25], [13.48, 14.89, 0.25]],
    dtype=float,
)


def _map_hashes() -> dict[str, str]:
    data_dir = ROOT / "results" / "datas"
    return {
        name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
        for name in ("Radio_datas.npz", "Radio_datas_A2G.npz")
    }


def load_users() -> np.ndarray:
    return np.loadtxt(ROOT / "results" / "datas" / "Users_6.txt")[:, :2]


def build_world(order: Sequence[int], payloads_mb: np.ndarray, outage_max_s: float = 100.0) -> World:
    with contextlib.redirect_stdout(io.StringIO()):
        return World(
            length=20,
            width=20,
            uav_num=1,
            dist_max=0.25,
            delta_t=0.5,
            t=1000,
            uav_h=1.0,
            data_size=300,
            data_sizes=np.asarray(payloads_mb, dtype=float),
            outage_max_s=float(outage_max_s),
            ini_loc=START.tolist(),
            end_loc=END.tolist(),
            users_name=str(ROOT / "results" / "datas" / "Users_6.txt"),
            BS_loc=BASE_STATIONS,
            traverse_sequence=list(order),
            variant="A",
        )


def replay_node_path(world: World, graph: SpatialGraph, node_path: Sequence[int]):
    """Convert graph edges to variant-A actions and replay them in ``World``."""
    if not node_path:
        raise ValueError("node_path must include at least the initial node")
    with contextlib.redirect_stdout(io.StringIO()):
        state, step = world.reset()

    initial = np.array([world.UAVs[0].x, world.UAVs[0].y], dtype=float)
    if not np.allclose(initial, graph.positions[node_path[0]], atol=1e-8):
        raise ValueError("the graph path does not start at World's initial location")
    positions = [initial.tolist()]
    times = [0.0]
    outages = [0.0]
    pending = [float(world.data_size)]
    visited = [int(world.Traverse_order)]
    actions = [[0.0, 0.0]]
    total_uploaded = 0.0
    flight_distance_m = 0.0
    done = False
    terminal = False

    for destination in node_path[1:]:
        current = np.array([world.UAVs[0].x, world.UAVs[0].y], dtype=float)
        desired = np.asarray(graph.positions[destination], dtype=float)
        displacement = desired - current
        distance = float(np.linalg.norm(displacement))
        if distance > world.dist_max + 1e-8:
            raise ValueError(f"graph edge is {distance * 100:.3f} m, above World's 25 m limit")
        absolute_heading = math.atan2(displacement[1], displacement[0]) if distance > 1e-12 else 0.0
        action = world.absolute_to_action(np.array([absolute_heading, distance], dtype=float))
        with contextlib.redirect_stdout(io.StringIO()):
            state, _, done, step, terminal = world.step_inside(action, state, step)
        actual = np.array([world.UAVs[0].x, world.UAVs[0].y], dtype=float)
        flight_distance_m += float(np.linalg.norm(actual - current) * 100.0)
        total_uploaded += float(world.last_uploaded_data)
        positions.append(actual.tolist())
        times.append(float(step * world.delta_T))
        outages.append(float(world.out_time))
        pending.append(float(world.data_size))
        visited.append(int(world.Traverse_order))
        actions.append(action.tolist())
        if done:
            break

    expected_upload = float(sum(world.payload_for_user(user) for user in world.Traverse))
    success = bool(terminal and total_uploaded >= expected_upload - 1e-7 and world.out_time <= world.outage_max_s)
    if success:
        reason = "success"
    elif world.out_time > world.outage_max_s:
        reason = "qos_violation"
    elif done and step >= world.T:
        reason = "timeout"
    elif done:
        reason = "world_rejected_candidate"
    else:
        reason = "incomplete_candidate_path"
    task_time = float(step * world.delta_T)
    transmission_delay = float(world.trans_delay * world.delta_T)
    trace = {
        "positions": np.asarray(positions, dtype=float),
        "time_s": np.asarray(times, dtype=float),
        "outage_s": np.asarray(outages, dtype=float),
        "pending_mb": np.asarray(pending, dtype=float),
        "visited_points": np.asarray(visited, dtype=int),
        "actions": np.asarray(actions, dtype=float),
    }
    report = {
        "success": success,
        "termination_reason": reason,
        "task_time_s": task_time,
        "transmission_delay_s": transmission_delay,
        "objective_s": task_time + transmission_delay,
        "outage_time_s": float(world.out_time),
        "outage_limit_s": float(world.outage_max_s),
        "flight_distance_m": flight_distance_m,
        "uploaded_mb": total_uploaded,
        "expected_upload_mb": expected_upload,
        "trajectory_steps": int(step),
        "visited_inspection_points": int(min(world.Traverse_order, len(world.Traverse))),
        "expected_inspection_points": len(world.Traverse),
        "variant": world.variant,
        "metrics_source": "real World replay",
    }
    return report, trace


def _jsonable_trace(trace: dict[str, np.ndarray]) -> dict[str, list]:
    return {key: np.asarray(value).tolist() for key, value in trace.items()}


def _plot_trace(trace: dict[str, np.ndarray], output_stem: Path, title: str, draw_background: bool) -> None:
    figure, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
    if draw_background:
        data = np.load(ROOT / "results" / "datas" / "Radio_datas_A2G.npz")
        contour = axis.contourf(
            np.asarray(data["arr_2"]) * 10.0,
            np.asarray(data["arr_3"]) * 10.0,
            np.asarray(data["arr_1"]),
            levels=30,
        )
        figure.colorbar(contour, ax=axis, label="A2G SNR (dB)")
    trajectory = np.asarray(trace["positions"], dtype=float) * 100.0
    axis.plot(trajectory[:, 0], trajectory[:, 1], color="tab:red", linewidth=1.8, label="A* replay")
    axis.scatter(trajectory[0, 0], trajectory[0, 1], c="black", s=55, marker="o", label="Start")
    axis.scatter(trajectory[-1, 0], trajectory[-1, 1], c="gold", s=65, marker="*", label="Replay end")
    axis.set_xlabel("X (m)")
    axis.set_ylabel("Y (m)")
    axis.set_aspect("equal")
    axis.set_title(title)
    axis.legend(loc="best")
    figure.savefig(output_stem.with_suffix(".png"), dpi=300)
    figure.savefig(output_stem.with_suffix(".pdf"))
    plt.close(figure)


def save_level_artifacts(
    *,
    output_dir: Path,
    level: str,
    order: Sequence[int],
    payloads_mb: np.ndarray,
    graph: SpatialGraph,
    planner_result: PlannerResult,
    planner_mode: str = "joint",
    replay_report: dict,
    trace: dict[str, np.ndarray],
    draw_background: bool = True,
) -> dict:
    target = Path(output_dir) / level
    target.mkdir(parents=True, exist_ok=True)
    planner_metrics = {
        "status": planner_result.status,
        "message": planner_result.message,
        "success": planner_result.success,
        "optimality_proven": planner_result.optimality_proven,
        "quantized_objective_s": planner_result.planner_cost_s,
        "expansions": planner_result.expansions,
        "generated_labels": planner_result.generated_labels,
        "runtime_s": planner_result.runtime_s,
        "path_slots": max(0, len(planner_result.node_path) - 1),
    }
    metrics = {
        "algorithm": "qos_constrained_astar_25m_4neighbor_hover",
        "search_mode": planner_mode,
        "data_level": level,
        "inspection_order": list(order),
        "payloads_mb": np.asarray(payloads_mb, dtype=float).tolist(),
        "map_hashes": _map_hashes(),
        "planner": planner_metrics,
        "world_replay": replay_report,
        "comparison_metrics_source": "world_replay",
    }
    (target / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    trace_json = {
        **_jsonable_trace(trace),
        "planner_node_path": list(planner_result.node_path),
        "planner_positions": graph.positions[list(planner_result.node_path)].tolist(),
    }
    (target / "trajectory.json").write_text(json.dumps(trace_json, indent=2), encoding="utf-8")
    np.savez(
        target / "trajectory.npz",
        **trace,
        planner_node_path=np.asarray(planner_result.node_path, dtype=int),
        planner_positions=graph.positions[list(planner_result.node_path)],
    )
    _plot_trace(
        trace,
        target / "trajectory",
        f"A* {level}: planner={planner_result.status}, replay={replay_report['termination_reason']}",
        draw_background,
    )
    return metrics


def _fair_order(payloads_mb: np.ndarray) -> tuple[int, ...]:
    # This recomputes the same data-aware IGA/exact-fallback sequence used by
    # training.  It reads no TD3 path, replay, actor, or checkpoint artifact.
    with contextlib.redirect_stdout(io.StringIO()):
        return tuple(int(node) for node in solve_ga_order(payloads_mb))


def run_level(level: str, args) -> dict:
    payloads = load_fixed_payloads(level)
    order = tuple(args.order) if args.order else _fair_order(payloads)
    users = load_users()
    specials = {"start": START, "end": END}
    specials.update({f"inspection_{index}": point for index, point in enumerate(users, start=1)})
    graph = SpatialGraph.build(length=20.0, width=20.0, spacing=0.25, special_points=specials)
    targets = np.vstack([users[user - 1] for user in order] + [END])
    leg_payloads = np.asarray([payloads[user - 1] for user in order], dtype=float)
    maps = FixedRadioMaps(ROOT / "results" / "datas")
    snr, outage = sample_radio_maps(graph, maps)
    planner = AStarPlanner(
        graph=graph,
        start_node=graph.special_nodes["start"],
        targets=targets,
        payloads_mb=leg_payloads,
        snr_db_by_node=snr,
        outage_by_node=outage,
        bucket_mb=args.bucket_mb,
        qos_slots=int(round(args.outage_max_s / 0.5)),
        max_expansions=args.max_expansions,
        max_labels=args.max_labels,
        max_runtime_s=args.max_runtime_s,
    )
    result = planner.plan_sequential() if args.mode == "sequential" else planner.plan()
    world = build_world(order, payloads, outage_max_s=args.outage_max_s)
    replay_report, trace = replay_node_path(world, graph, result.node_path)
    replay_report["termination_reason"] = (
        replay_report["termination_reason"]
        if result.success
        else f"planner_{result.status}:{replay_report['termination_reason']}"
    )
    return save_level_artifacts(
        output_dir=args.output_dir,
        level=level,
        order=order,
        payloads_mb=payloads,
        graph=graph,
        planner_result=result,
        planner_mode=args.mode,
        replay_report=replay_report,
        trace=trace,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", nargs="+", choices=("low", "medium", "high"), default=("low", "medium", "high"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "journal_results" / "astar-baseline")
    parser.add_argument("--order", nargs="+", type=int, help="explicit inspection order, mainly for reproducible smoke runs")
    parser.add_argument("--mode", choices=("sequential", "joint"), default="sequential",
                        help="sequential: optimal per leg given the previous arrival (fast, upper bound); "
                             "joint: whole-mission A* (optimal on this graph, may hit search limits)")
    parser.add_argument("--outage-max-s", type=float, default=100.0,
                        help="hard G2A outage budget for both the planner and the World replay")
    parser.add_argument("--bucket-mb", type=float, default=1.0)
    parser.add_argument("--max-expansions", type=int, default=500_000)
    parser.add_argument("--max-labels", type=int, default=1_000_000)
    parser.add_argument("--max-runtime-s", type=float, default=600.0)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.order and sorted(args.order) != [1, 2, 3, 4, 5, 6]:
        parser.error("--order must be a permutation of inspection users 1..6")

    summary = {}
    for level in args.levels:
        metrics = run_level(level, args)
        summary[level] = {
            "planner_status": metrics["planner"]["status"],
            "planner_optimality_proven": metrics["planner"]["optimality_proven"],
            "replay_success": metrics["world_replay"]["success"],
            "objective_s": metrics["world_replay"]["objective_s"],
        }
        print(json.dumps({level: summary[level]}, ensure_ascii=False))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

