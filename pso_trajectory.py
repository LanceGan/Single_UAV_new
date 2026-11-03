"""PSO end-to-end trajectory baseline (offline metaheuristic planner).

Decision vector: for each leg, three perpendicular control-point offsets plus a
pre-arrival hover duration. Each candidate is executed slot-by-slot through the
shared World via the standard evaluation ledger, so its metrics are produced by
exactly the same accounting as every other method. Feasible candidates score
their T + tau; infeasible ones a progress-shaped penalty that always ranks
below any feasible one. Like A* and MapGuide this is a single-instance offline
planner: full dual-map knowledge, no generalisation, re-solve per instance.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from World import World
from evaluate_policy import rollout as ledger_rollout
from sequence_runner import solve_ga_order
from training_config import load_fixed_payloads

ROOT = Path(__file__).resolve().parent
START = np.array([2.62, 2.65])
END = np.array([18.95, 11.12])
BASE_STATIONS = np.array([[4.77, 4.56, 0.25], [14.89, 4.89, 0.25],
                          [4.63, 13.78, 0.25], [13.48, 14.89, 0.25]])
CONTROL_FRACTIONS = (0.25, 0.5, 0.75)
OFFSET_BOUND = 4.0        # world units (one unit = 100 m)
HOVER_BOUND = 150.0       # slots (75 s) of pre-arrival dwell
GENES_PER_LEG = len(CONTROL_FRACTIONS) + 1
FAIL_BASE = 5000.0


def build_world(payloads, order, outage_max_s=100.0):
    with contextlib.redirect_stdout(io.StringIO()):
        return World(length=20, width=20, uav_num=1, dist_max=0.25, delta_t=0.5, t=1000,
                     uav_h=1.0, data_size=300, data_sizes=np.asarray(payloads, dtype=float),
                     outage_max_s=outage_max_s, ini_loc=START.tolist(), end_loc=END.tolist(),
                     users_name=str(ROOT / "results" / "datas" / "Users_6.txt"),
                     BS_loc=BASE_STATIONS, traverse_sequence=list(order), variant="baseline")


def make_context(payloads, order, outage_max_s=100.0):
    return SimpleNamespace(
        run_dir=None, config={}, algorithm="PSO", state_dim=8, net_width=0,
        variant="baseline", order=tuple(int(node) for node in order),
        payloads_mb=np.asarray(payloads, dtype=float), outage_max_s=float(outage_max_s),
        total_steps=1000, length=20.0, width=20.0, delta_t=0.5, dist_max=0.25,
    )


def leg_endpoints(order):
    users = np.loadtxt(ROOT / "results" / "datas" / "Users_6.txt")[:, :2]
    points = [START] + [users[node - 1] for node in order] + [END]
    return [(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
            for a, b in zip(points, points[1:])]


def build_leg_plans(order, genes):
    """Per-leg vertex lists (with the hover vertex marked) and hover budgets."""
    genes = np.asarray(genes, dtype=float).reshape(len(order) + 1, GENES_PER_LEG)
    plans = []
    for (a, b), row in zip(leg_endpoints(order), genes):
        direction = b - a
        unit = direction / max(float(np.linalg.norm(direction)), 1e-9)
        normal = np.array([-unit[1], unit[0]])
        vertices = [a + fraction * direction + offset * normal
                    for fraction, offset in zip(CONTROL_FRACTIONS, row[:3])]
        hover_vertex = b - unit * 0.8  # just outside the 0.5 arrival disk
        plans.append({"vertices": vertices + [hover_vertex, b],
                      "hover_index": len(vertices),
                      "hover_slots": max(0, int(round(row[3])))})
    return plans


def trajectory_policy(world, plans):
    """Stateful absolute-action policy following the encoded vertex plans."""
    cursor = {"leg": 0, "vertex": 0, "hover_left": None}

    def act(_state):
        while True:
            if cursor["leg"] >= len(plans):
                return np.array([0.0, 0.0])
            plan = plans[cursor["leg"]]
            # Leg advances when the world has moved on to the next target.
            expected_leg = int(getattr(world, "Traverse_order", cursor["leg"]))
            if expected_leg > cursor["leg"]:
                cursor.update(leg=expected_leg, vertex=0, hover_left=None)
                continue
            here = np.array([world.UAVs[0].x, world.UAVs[0].y])
            if cursor["vertex"] >= len(plan["vertices"]):
                return np.array([0.0, 0.0])  # at the final vertex; let arrival land
            target = plan["vertices"][cursor["vertex"]]
            distance = float(np.linalg.norm(target - here))
            at_hover_vertex = cursor["vertex"] == plan["hover_index"]
            if distance < 0.05:
                if at_hover_vertex:
                    if cursor["hover_left"] is None:
                        cursor["hover_left"] = plan["hover_slots"]
                    buffer_clear = bool(getattr(world, "transmit", True))
                    if cursor["hover_left"] > 0 and not buffer_clear:
                        cursor["hover_left"] -= 1
                        return np.array([0.0, 0.0])
                cursor["vertex"] += 1
                cursor["hover_left"] = None
                continue
            delta = target - here
            heading = float(np.arctan2(delta[1], delta[0]))
            return np.array([heading, min(distance, world.dist_max)])

    return act


def evaluate_vector(payloads, order, genes, outage_max_s=100.0):
    world = build_world(payloads, order, outage_max_s)
    context = make_context(payloads, order, outage_max_s)
    plans = build_leg_plans(order, genes)
    with contextlib.redirect_stdout(io.StringIO()):
        report, trace = ledger_rollout(context, world, trajectory_policy(world, plans))
    return report, trace


def fitness_from_report(report):
    if report["success"]:
        return report["task_time_s"] + report["transmission_delay_s"]
    return (FAIL_BASE - 400.0 * report["visited_inspection_points"]
            - report["uploaded_mb"])


def solve(payloads, order, seed=1, swarm=40, iterations=60, inertia=0.72,
          cognitive=1.49, social=1.49, patience=15, outage_max_s=100.0, log=print):
    rng = np.random.default_rng(seed)
    legs = len(order) + 1
    dims = legs * GENES_PER_LEG
    low = np.tile([-OFFSET_BOUND] * 3 + [0.0], legs)
    high = np.tile([OFFSET_BOUND] * 3 + [HOVER_BOUND], legs)
    pos = np.clip(rng.normal(0.0, 1.0, size=(swarm, dims)) * (high - low) * 0.1
                  + (low + high) / 2.0 * np.tile([0.0, 0.0, 0.0, 1.0], legs),
                  low, high)
    vel = np.zeros_like(pos)
    personal_best = pos.copy()
    personal_cost = np.full(swarm, np.inf)
    best_cost, best_vector, best_report, best_trace = np.inf, None, None, None
    stale = 0
    for iteration in range(int(iterations)):
        for index in range(swarm):
            report, trace = evaluate_vector(payloads, order, pos[index], outage_max_s)
            cost = fitness_from_report(report)
            if cost < personal_cost[index]:
                personal_cost[index], personal_best[index] = cost, pos[index].copy()
            if cost < best_cost:
                best_cost, best_vector = cost, pos[index].copy()
                best_report, best_trace = report, trace
                stale = -1
        stale += 1
        if stale >= patience:
            log(f"early stop at iteration {iteration + 1} (no improvement for {patience})")
            break
        r1 = rng.random((swarm, dims))
        r2 = rng.random((swarm, dims))
        vel = (inertia * vel + cognitive * r1 * (personal_best - pos)
               + social * r2 * (best_vector - pos))
        vel = np.clip(vel, -(high - low) * 0.2, (high - low) * 0.2)
        pos = np.clip(pos + vel, low, high)
        log(f"iter {iteration + 1}/{iterations}: best={best_cost:.1f} "
            f"success={bool(best_report and best_report['success'])}", flush=True)
    return {
        "best_cost": float(best_cost),
        "best_vector": best_vector.tolist() if best_vector is not None else None,
        "report": best_report,
        "trace": best_trace,
        "seed": int(seed), "swarm": int(swarm), "iterations": int(iterations),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_level", choices=("low", "medium", "high"), default="low")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--swarm", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--output", default="journal_results/pso-trajectory")
    args = parser.parse_args()
    np.random.seed(args.seed)
    payloads = load_fixed_payloads(args.data_level)
    order = list(solve_ga_order(payloads))
    print("order:", order, flush=True)
    result = solve(payloads, order, seed=args.seed, swarm=args.swarm,
                   iterations=args.iterations)
    output = Path(args.output) / args.data_level
    output.mkdir(parents=True, exist_ok=True)
    trace = result.pop("trace")
    if trace is not None:
        np.savez(output / "trajectory.npz", **trace)
    (output / "metrics.json").write_text(json.dumps(
        {**{k: v for k, v in result.items() if k != "trace"},
         "order": order, "payloads_mb": [float(p) for p in payloads],
         "method": "PSO end-to-end trajectory (offline, full dual-map knowledge)"},
        indent=2, ensure_ascii=False), encoding="utf-8")
    if trace is not None:
        from evaluate_policy import plot_trajectory
        from plot_formal_results import _style
        _style()
        context = SimpleNamespace(order=tuple(order), algorithm="PSO")
        title = (f"PSO-trajectory, {args.data_level.capitalize()} load: "
                 f"{'success' if result['report']['success'] else 'infeasible'}, "
                 f"objective={result['best_cost']:.1f}s")
        plot_trajectory(context, dict(trace), output / "trajectory_dual_maps", title)
    print(json.dumps({"best_cost": result["best_cost"], "report": result["report"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
