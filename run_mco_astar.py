"""Run the adapted MCO-A* trajectory baseline through the real World.

The visit order is fixed to the PRIGA order used by the proposed method.
MCO-A* plans the trajectory with a hard contiguous-outage state, while upload
locations are selected by an explicitly A2G-blind detour heuristic.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from astar_planner import SpatialGraph
from mco_astar import MCOAStarGuide
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


def _build_world(
    order,
    payloads_mb: np.ndarray,
    outage_max_s: float,
    total_steps: int,
    qos_enforcement: str,
) -> World:
    return World(
        length=20,
        width=20,
        uav_num=1,
        dist_max=0.25,
        delta_t=0.5,
        t=int(total_steps),
        data_sizes=np.asarray(payloads_mb, dtype=float),
        outage_max_s=float(outage_max_s),
        ini_loc=START.tolist(),
        end_loc=END.tolist(),
        users_name=str(ROOT / "results" / "datas" / "Users_6.txt"),
        BS_loc=BASE_STATIONS,
        traverse_sequence=list(order),
        qos_enforcement=qos_enforcement,
    )


def _run_level_profile(
    *,
    level: str,
    graph: SpatialGraph,
    outage_max_s: float = 6.0,
    order_override=None,
    total_steps: int = 1000,
    qos_enforcement: str = "hard",
    qos_slots_override: int | None = None,
    planner_profile: str = "hard-qos",
) -> dict:
    payloads = load_fixed_payloads(level)
    if order_override is None:
        with contextlib.redirect_stdout(io.StringIO()):
            order = tuple(int(node) for node in solve_ga_order(payloads, iterations=50))
    else:
        order = tuple(int(node) for node in order_override)
    if sorted(order) != [1, 2, 3, 4, 5, 6]:
        raise ValueError("inspection order must be a permutation of users 1..6")

    world = _build_world(
        order,
        payloads,
        outage_max_s,
        total_steps=total_steps,
        qos_enforcement=qos_enforcement,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        state, step = world.reset()
    guide = MCOAStarGuide(
        world=world,
        graph=graph,
        qos_slots_override=qos_slots_override,
        profile=planner_profile,
    )

    positions = [[world.UAVs[0].x, world.UAVs[0].y]]
    times = [0.0]
    outage_times = [0.0]
    total_outage_times = [0.0]
    current_outages = [0.0]
    pending = [float(world.data_size)]
    visited_points = [int(world.Traverse_order)]
    actions = [[0.0, 0.0]]
    visits = []
    uploaded_mb = 0.0
    total_reward = 0.0
    done = False
    terminal = False
    failure = None

    while not done:
        previous_visit = int(world.Traverse_order)
        previous_pending = float(world.data_size)
        try:
            action = guide.action()
        except Exception as exc:  # preserve a traceable failure artifact
            failure = {"type": type(exc).__name__, "message": str(exc)}
            break
        with contextlib.redirect_stdout(io.StringIO()):
            state, reward, done, step, terminal = world.step_inside(action, state, step)
        total_reward += float(reward)
        uploaded_mb += float(world.last_uploaded_data)
        positions.append([world.UAVs[0].x, world.UAVs[0].y])
        times.append(float(step * world.delta_T))
        outage_times.append(float(world.max_contiguous_outage_s))
        total_outage_times.append(float(world.out_time))
        current_outages.append(float(world.current_contiguous_outage_s))
        pending.append(float(world.data_size))
        visited_points.append(int(world.Traverse_order))
        actions.append(np.asarray(action, dtype=float).tolist())
        if world.Traverse_order != previous_visit or terminal:
            visits.append({
                "node": order[previous_visit] if previous_visit < len(order) else "end",
                "time_s": float(step * world.delta_T),
                "outage_s": float(world.max_contiguous_outage_s),
                "total_outage_s": float(world.out_time),
                "pending_before_collection": max(
                    0.0,
                    previous_pending - float(world.last_uploaded_data),
                ),
            })
        if not np.all(np.isfinite(state)):
            failure = {"type": "NonFiniteState", "message": "World returned a non-finite state"}
            break

    task_time = float(step * world.delta_T)
    transmission_delay = float(world.trans_delay * world.delta_T)
    expected_upload = float(payloads.sum())
    mission_complete = bool(
        failure is None
        and terminal
        and uploaded_mb >= expected_upload - 1e-7
        and world.Traverse_order >= len(order)
    )
    max_outage = float(world.max_contiguous_outage_s)
    qos_feasible = bool(mission_complete and max_outage <= world.outage_max_s)
    if qos_feasible:
        termination_reason = "success"
    elif mission_complete:
        termination_reason = "constraint_violation"
    elif failure is not None:
        termination_reason = f"planner_failure:{failure['type']}"
    elif done and step >= world.T:
        termination_reason = "horizon_timeout"
    elif done:
        termination_reason = "world_rejected_candidate"
    else:
        termination_reason = "incomplete_mission"

    return {
        "algorithm": "mco_astar_fixed_priga_a2g_blind_upload_hub",
        "source_method": "MCO-A* from the supplied ICCT-2025 source archive",
        "qos_constraint": "max_contiguous_outage",
        "budget_s": float(outage_max_s),
        "level": level,
        "success": qos_feasible,
        "mission_complete": mission_complete,
        "qos_feasible": qos_feasible,
        "constraint_satisfied": qos_feasible,
        "outage_budget_exceeded": bool(max_outage > world.outage_max_s),
        "qos_enforcement": qos_enforcement,
        "total_steps": int(total_steps),
        "termination_reason": termination_reason,
        "failure": failure,
        "inspection_order": list(order),
        "payloads_mb": payloads.tolist(),
        "visited_inspection_points": int(min(world.Traverse_order, len(order))),
        "expected_inspection_points": len(order),
        "task_time_s": task_time,
        "transmission_delay_s": transmission_delay,
        "objective_s": task_time + transmission_delay if mission_complete else None,
        "outage_s": max_outage,
        "total_outage_s": float(world.out_time),
        "uploaded_mb": uploaded_mb,
        "expected_upload_mb": expected_upload,
        "trajectory_steps": int(step),
        "return": total_reward,
        "reward_terms": world.reward_summary(),
        "reward_accounting_error": float(
            abs(sum(world.reward_summary().values()) - total_reward)
        ),
        "max_pending_data_at_visit": max(
            (item["pending_before_collection"] for item in visits),
            default=0.0,
        ),
        "visits": visits,
        "planner": guide.planner_audit(),
        "planner_profile": planner_profile,
        "adaptation": {
            "visit_order": "fixed PRIGA order",
            "trajectory": "MCO-A* state (node, contiguous G2A outage slots)",
            "lattice": "25 m cardinal lattice, 0.5 s edge cost",
            "upload_hub": "A2G-blind nearest safe geographic detour",
            "original_vs_adapted": (
                "The source algorithm optimizes energy for multi-trip cargo UAVs and "
                "does not model uplink offloading. The adapted baseline retains the "
                "outage-state A* but uses the unified inspection-upload environment."
            ),
        },
        "map_hashes": _map_hashes(),
        "metrics_source": "real World replay",
        "trace": {
            "positions": np.asarray(positions, dtype=float),
            "time_s": np.asarray(times, dtype=float),
            "outage_s": np.asarray(outage_times, dtype=float),
            "total_outage_s": np.asarray(total_outage_times, dtype=float),
            "current_contiguous_outage_s": np.asarray(current_outages, dtype=float),
            "pending_mb": np.asarray(pending, dtype=float),
            "visited_points": np.asarray(visited_points, dtype=int),
            "actions": np.asarray(actions, dtype=float),
        },
    }


def run_level(
    *,
    level: str,
    graph: SpatialGraph,
    outage_max_s: float = 6.0,
    order_override=None,
    total_steps: int = 1000,
    qos_enforcement: str = "continue",
    allow_relaxed_fallback: bool = True,
) -> dict:
    """Run the hard-QoS MCO baseline and an explicit relaxed completion fallback.

    The strict planner is always attempted first.  If it cannot generate a
    complete mission, the same search is rerun without the six-second pruning
    cap solely to obtain a complete diagnostic trajectory and objective.  The
    fallback is labelled and never counted as a QoS-feasible result.
    """
    strict = _run_level_profile(
        level=level,
        graph=graph,
        outage_max_s=outage_max_s,
        order_override=order_override,
        total_steps=total_steps,
        qos_enforcement=qos_enforcement,
        qos_slots_override=None,
        planner_profile="hard-qos",
    )
    if strict["mission_complete"] or not allow_relaxed_fallback:
        strict["relaxed_fallback_used"] = False
        return strict

    relaxed = _run_level_profile(
        level=level,
        graph=graph,
        outage_max_s=outage_max_s,
        order_override=order_override,
        total_steps=total_steps,
        qos_enforcement=qos_enforcement,
        qos_slots_override=int(total_steps),
        planner_profile="relaxed-completion-fallback",
    )
    relaxed["relaxed_fallback_used"] = True
    relaxed["strict_attempt"] = {
        key: strict[key]
        for key in (
            "mission_complete",
            "qos_feasible",
            "termination_reason",
            "task_time_s",
            "transmission_delay_s",
            "outage_s",
            "visited_inspection_points",
            "uploaded_mb",
        )
    }
    return relaxed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--levels",
        nargs="+",
        choices=("low", "medium", "high"),
        default=("low", "medium", "high"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "journal_results" / "v4-deterministic" / "mco-astar-6",
    )
    parser.add_argument("--outage-max-s", type=float, default=6.0)
    parser.add_argument("--total-steps", type=int, default=1000)
    parser.add_argument(
        "--qos-enforcement",
        choices=World.QOS_ENFORCEMENT_MODES,
        default="continue",
    )
    parser.add_argument("--no-relaxed-fallback", action="store_true")
    parser.add_argument("--order", nargs="+", type=int)
    args = parser.parse_args()
    args.output = args.output.resolve()

    users = np.loadtxt(ROOT / "results" / "datas" / "Users_6.txt")[:, :2]
    graph = _build_graph(users)
    summary = {}
    for level in args.levels:
        result = run_level(
            level=level,
            graph=graph,
            outage_max_s=args.outage_max_s,
            order_override=args.order,
            total_steps=args.total_steps,
            qos_enforcement=args.qos_enforcement,
            allow_relaxed_fallback=not args.no_relaxed_fallback,
        )
        trace = result.pop("trace")
        target = args.output / level
        target.mkdir(parents=True, exist_ok=True)
        (target / "metrics.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        np.savez(target / "trajectory.npz", **trace)
        summary[level] = {
            "success": result["success"],
            "mission_complete": result["mission_complete"],
            "qos_feasible": result["qos_feasible"],
            "termination_reason": result["termination_reason"],
            "inspection_order": result["inspection_order"],
            "task_time_s": result["task_time_s"],
            "transmission_delay_s": result["transmission_delay_s"],
            "objective_s": result["objective_s"],
            "outage_s": result["outage_s"],
            "relaxed_fallback_used": result["relaxed_fallback_used"],
            "uploaded_mb": result["uploaded_mb"],
            "visited_inspection_points": result["visited_inspection_points"],
        }
        print(json.dumps({level: summary[level]}, ensure_ascii=False), flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.output / "run_config.json").write_text(
        json.dumps(
            {
                "algorithm": "mco_astar_adapted",
                "levels": list(args.levels),
                "total_steps": int(args.total_steps),
                "task_horizon_s": float(args.total_steps * 0.5),
                "qos_constraint": "max_contiguous_outage",
                "outage_max_s": float(args.outage_max_s),
                "qos_enforcement": args.qos_enforcement,
                "relaxed_completion_fallback": not args.no_relaxed_fallback,
                "order_override": args.order,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    raise SystemExit(
        0
        if all(item["mission_complete"] for item in summary.values())
        else 1
    )


if __name__ == "__main__":
    main()
