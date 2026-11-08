"""Run complete, non-learning inspection missions before expensive training."""

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from World import World
from map_guidance import MapGuide
from sequence_runner import solve_ga_order
from training_config import load_fixed_payloads


def run_preflight(level, output_dir=None, order=None, outage_max_s=100,
                  total_steps=1000, a2g_blind=False,
                  outage_probability_threshold=0.5,
                  qos_enforcement="hard"):
    payloads = load_fixed_payloads(level)
    with contextlib.redirect_stdout(io.StringIO()):
        order = (
            solve_ga_order(
                payloads,
                iterations=50,
                outage_probability_threshold=outage_probability_threshold,
            )
            if order is None else order
        )
        world = World(length=20, width=20, uav_num=1, dist_max=0.25, delta_t=0.5, t=total_steps,
                      data_sizes=payloads, outage_max_s=outage_max_s, ini_loc=[2.62, 2.65],
                      end_loc=[18.95, 11.12], users_name="results/datas/Users_6.txt",
                      BS_loc=np.array([[4.77,4.56,0.25],[14.89,4.89,0.25],[4.63,13.78,0.25],[13.48,14.89,0.25]]),
                      traverse_sequence=order,
                      outage_probability_threshold=outage_probability_threshold,
                      qos_enforcement=qos_enforcement)
    state, t = world.reset()
    guide = MapGuide(world, use_a2g=not bool(a2g_blind))
    trace, visits = [], []
    total_reward = uploaded = 0.0
    done = terminal = False
    while not done:
        action = guide.action()
        previous_visit = world.Traverse_order
        old_pending = world.data_size
        with contextlib.redirect_stdout(io.StringIO()):
            state, reward, done, t, terminal = world.step_inside(action, state, t)
        total_reward += reward
        uploaded += world.last_uploaded_data
        trace.append([world.UAVs[0].x, world.UAVs[0].y, t * world.delta_T,
                      world.max_contiguous_outage_s, world.out_time,
                      world.current_contiguous_outage_s, world.data_size,
                      world.Traverse_order, *action])
        if world.Traverse_order != previous_visit or terminal:
            visits.append({"node": order[previous_visit] if previous_visit < len(order) else "end",
                           "time_s": t * world.delta_T,
                           "outage_s": world.max_contiguous_outage_s,
                           "total_outage_s": world.out_time,
                           "pending_before_collection": max(0.0, old_pending - world.last_uploaded_data)})
        if not np.all(np.isfinite(state)):
            raise RuntimeError("Non-finite policy observation")
    mission_complete = bool(
        terminal
        and uploaded >= payloads.sum() - 1e-7
        and world.Traverse_order >= len(order)
    )
    qos_feasible = bool(
        mission_complete
        and world.max_contiguous_outage_s <= world.outage_max_s
    )
    report = {"level": level, "qos_constraint": "max_contiguous_outage",
              "outage_probability_threshold": float(outage_probability_threshold),
              "qos_enforcement": qos_enforcement,
              "success": qos_feasible,
              "mission_complete": mission_complete,
              "qos_feasible": qos_feasible,
              "constraint_satisfied": qos_feasible,
              "outage_budget_exceeded": bool(
                  world.max_contiguous_outage_s > world.outage_max_s
              ),
              "order": order, "payloads_mb": payloads.tolist(), "time_s": t * world.delta_T,
              "transmission_delay_s": float(world.trans_delay * world.delta_T),
              "outage_s": world.max_contiguous_outage_s,
              "total_outage_s": world.out_time,
              "uploaded_mb": float(round(uploaded, 8)),
              "expected_upload_mb": float(payloads.sum()), "visited_points": world.Traverse_order,
              "return": float(total_reward), "reward_terms": world.reward_summary(), "visits": visits,
              "reward_accounting_error": float(abs(sum(world.reward_summary().values()) - total_reward)),
              "max_pending_data_at_visit": max((v["pending_before_collection"] for v in visits), default=0.0),
              "map_hashes": {name: hashlib.sha256(Path("results/datas", name).read_bytes()).hexdigest()
                             for name in ("Radio_datas.npz", "Radio_datas_A2G.npz")},
              "method": "map-aware reference guidance (no learned policy)"}
    if output_dir is not None:
        path = Path(output_dir) / level
        path.mkdir(parents=True, exist_ok=True)
        (path / "preflight.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        np.savez(path / "preflight_trace.npz", trace=np.array(trace))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", choices=("all", "low", "medium", "high"), default="all")
    parser.add_argument("--output", default="journal_results/preflight")
    parser.add_argument("--outage-max-s", type=float, default=8.0,
                        help="maximum-continuous G2A outage budget")
    parser.add_argument("--outage-probability-threshold", type=float, default=0.5,
                        help="G2A outage event threshold kappa_out")
    args = parser.parse_args()
    levels = ("low", "medium", "high") if args.level == "all" else (args.level,)
    passed = True
    for level in levels:
        result = run_preflight(
            level,
            args.output,
            outage_max_s=args.outage_max_s,
            outage_probability_threshold=args.outage_probability_threshold,
        )
        passed = passed and result["success"]
        print(json.dumps({k: result[k] for k in ("level", "success", "order", "time_s", "outage_s", "uploaded_mb", "visited_points")}), flush=True)
    raise SystemExit(0 if passed else 1)
