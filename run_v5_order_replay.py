"""Replay existing CA-TD3 actors on E2 order candidates without retraining."""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from evaluate_policy import (
    STATE_SCHEMA,
    _build_world,
    load_actor,
    rollout,
)
from map_guidance import MapGuide
from mission_preflight import run_preflight
from training_config import load_fixed_payloads


ROOT = Path(__file__).resolve().parent
SEQUENCE_RESULTS = ROOT / "journal_results" / "v5-sequence" / "sequence-study.json"
ACTOR_BY_LEVEL = {
    "low": ROOT / "sync_bundles" / "v4_eval_best_actors" / "w50-50_qos6_actor.pth",
    "medium": ROOT / "sync_bundles" / "v4_eval_best_actors" / "medium_qos6_actor.pth",
    "high": ROOT / "sync_bundles" / "v4_eval_best_actors" / "high_qos6_actor.pth",
}
NEW_METHODS = ("ga", "pso", "sa", "astar", "eega", "yang")


class ActorMapGuideController:
    """Use the learned actor first and hand control to MapGuide after stagnation."""

    def __init__(self, world, actor, *, stall_steps=100):
        self.world = world
        self.actor = actor
        self.guide = MapGuide(world, use_a2g=True)
        self.stall_steps = int(stall_steps)
        self.last_target = int(world.Traverse_order)
        self.best_distance = float(world.target_dis)
        self.last_progress_step = 0
        self.takeover_step = None
        self.actor_steps = 0
        self.guide_steps = 0

    @property
    def message_guide_active(self):
        return self.takeover_step is not None

    def __call__(self, state):
        world = self.world
        target = int(world.Traverse_order)
        distance = float(world.target_dis)
        if target != self.last_target or distance < self.best_distance - 1e-4:
            self.last_target = target
            self.best_distance = distance
            self.last_progress_step = int(world.t)
        elif (
            self.takeover_step is None
            and int(world.t) - self.last_progress_step >= self.stall_steps
        ):
            self.takeover_step = int(world.t)

        if self.takeover_step is None:
            self.actor_steps += 1
            with torch.no_grad():
                return self.actor(
                    torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
                )[0].numpy()
        self.guide_steps += 1
        return self.guide.action()


def _context(level, order, run_dir):
    payloads = load_fixed_payloads(level)
    config = {
        "data_level": level,
        "variant": "A",
        "a2g_blind": False,
        "state_schema": STATE_SCHEMA,
        "state_dim": 8,
        "net_width": 256,
        "outage_max_s": 6.0,
        "outage_probability_threshold": 0.5,
        "uav_h": 1.0,
        "total_steps": 2000,
        "qos_constraint": "max_contiguous_outage",
        "algorithm": "td3_bc",
    }
    return SimpleNamespace(
        run_dir=run_dir,
        config=config,
        algorithm="td3_bc",
        state_dim=8,
        net_width=256,
        variant="A",
        order=tuple(int(node) for node in order),
        payloads_mb=np.asarray(payloads, dtype=float),
        outage_max_s=6.0,
        outage_probability_threshold=0.5,
        a2g_blind=False,
        total_steps=2000,
        qos_enforcement="continue",
        length=20.0,
        width=20.0,
        delta_t=0.5,
        dist_max=0.25,
    )


def _replay_task(task):
    level, method_id, order, output_root, stall_steps = task
    checkpoint = ACTOR_BY_LEVEL[level]
    run_dir = Path(output_root) / method_id / level
    run_dir.mkdir(parents=True, exist_ok=True)
    context = _context(level, order, run_dir)
    actor = load_actor(checkpoint, context.state_dim, context.net_width, algorithm="td3_bc")

    with contextlib.redirect_stdout(io.StringIO()):
        world = _build_world(context)
    controller = ActorMapGuideController(
        world,
        actor,
        stall_steps=stall_steps,
    )
    report, trace = rollout(context, world, controller)
    hybrid_report = report
    full_restart_fallback = False
    if not report["mission_complete"]:
        full_restart_fallback = True
        fallback = run_preflight(
            level,
            order=context.order,
            outage_max_s=context.outage_max_s,
            total_steps=context.total_steps,
            a2g_blind=False,
            outage_probability_threshold=context.outage_probability_threshold,
            qos_enforcement=context.qos_enforcement,
        )
        fallback["task_time_s"] = float(fallback.pop("time_s"))
        fallback["outage_time_s"] = float(fallback.pop("outage_s"))
        fallback["total_outage_time_s"] = float(fallback.pop("total_outage_s"))
        fallback["visited_inspection_points"] = int(fallback.pop("visited_points"))
        fallback["expected_inspection_points"] = len(context.order)
        fallback["objective_s"] = (
            fallback["task_time_s"] + fallback["transmission_delay_s"]
            if fallback["mission_complete"]
            else None
        )
        fallback["termination_reason"] = (
            "success"
            if fallback["qos_feasible"]
            else "constraint_violation"
            if fallback["mission_complete"]
            else "incomplete"
        )
        report = fallback
        trace = None
    report.update(
        {
            "level": level,
            "method_id": method_id,
            "inspection_order": list(context.order),
            "shared_actor": str(checkpoint.relative_to(ROOT)),
            "trajectory_actor_shared_across_orders": True,
            "evaluation_protocol": (
                "existing CA-TD3 actor followed by MapGuide completion fallback "
                "after equal stagnation threshold; 1000 s; QoS records but continues"
            ),
            "control_mode": "CA-TD3 actor with MapGuide assisted completion",
            "stall_threshold_steps": int(stall_steps),
            "takeover_step": controller.takeover_step,
            "takeover_time_s": (
                controller.takeover_step * 0.5
                if controller.takeover_step is not None
                else None
            ),
            "actor_controlled_steps": int(controller.actor_steps),
            "mapguide_controlled_steps": int(controller.guide_steps),
            "final_trajectory_actor_steps": (
                0 if full_restart_fallback else int(controller.actor_steps)
            ),
            "final_trajectory_mapguide_steps": (
                int(report.get("trajectory_steps", 0))
                if full_restart_fallback
                else int(controller.guide_steps)
            ),
            "full_restart_fallback": full_restart_fallback,
        }
    )
    if full_restart_fallback:
        report["failed_hybrid_attempt"] = {
            key: hybrid_report.get(key)
            for key in (
                "mission_complete",
                "qos_feasible",
                "termination_reason",
                "task_time_s",
                "transmission_delay_s",
                "outage_time_s",
                "visited_inspection_points",
                "uploaded_mb",
            )
        }
    (run_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    trajectory_path = run_dir / "trajectory.npz"
    if trace is not None:
        np.savez(trajectory_path, **trace)
    elif trajectory_path.exists():
        trajectory_path.unlink()
    return {"level": level, "method_id": method_id, "report": report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", nargs="+", choices=("medium", "high"), default=("medium", "high"))
    parser.add_argument("--methods", nargs="+", choices=NEW_METHODS, default=NEW_METHODS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--stall-steps", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "journal_results" / "v5-mac" / "E2",
    )
    args = parser.parse_args()
    sequence_data = json.loads(SEQUENCE_RESULTS.read_text(encoding="utf-8"))
    lookup = {
        (record["level"], record["method_id"]): record
        for record in sequence_data["records"]
    }
    tasks = []
    for level in args.levels:
        for method_id in args.methods:
            record = lookup[(level, method_id)]
            tasks.append(
                (
                    level,
                    method_id,
                    record["inspection_order"],
                    args.output.resolve(),
                    args.stall_steps,
                )
            )
    if int(args.workers) <= 1:
        results = [_replay_task(task) for task in tasks]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
            results = list(executor.map(_replay_task, tasks))
    summary = {"records": results}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "records": len(results),
                "levels": args.levels,
                "methods": args.methods,
                "output": str(args.output),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
