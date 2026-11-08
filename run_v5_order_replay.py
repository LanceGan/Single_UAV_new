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
from training_config import load_fixed_payloads


ROOT = Path(__file__).resolve().parent
SEQUENCE_RESULTS = ROOT / "journal_results" / "v5-sequence" / "sequence-study.json"
ACTOR_BY_LEVEL = {
    "low": ROOT / "sync_bundles" / "v4_eval_best_actors" / "w50-50_qos6_actor.pth",
    "medium": ROOT / "sync_bundles" / "v4_eval_best_actors" / "medium_qos6_actor.pth",
    "high": ROOT / "sync_bundles" / "v4_eval_best_actors" / "high_qos6_actor.pth",
}
NEW_METHODS = ("ga", "pso", "sa", "astar", "eega", "yang")


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
    level, method_id, order, output_root = task
    checkpoint = ACTOR_BY_LEVEL[level]
    run_dir = Path(output_root) / method_id / level
    run_dir.mkdir(parents=True, exist_ok=True)
    context = _context(level, order, run_dir)
    actor = load_actor(checkpoint, context.state_dim, context.net_width, algorithm="td3_bc")

    def act(state):
        with torch.no_grad():
            return actor(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))[0].numpy()

    with contextlib.redirect_stdout(io.StringIO()):
        world = _build_world(context)
    report, trace = rollout(context, world, act)
    report.update(
        {
            "level": level,
            "method_id": method_id,
            "inspection_order": list(context.order),
            "shared_actor": str(checkpoint.relative_to(ROOT)),
            "trajectory_actor_shared_across_orders": True,
            "evaluation_protocol": "existing CA-TD3 actor; 1000 s; QoS records but continues",
        }
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    np.savez(run_dir / "trajectory.npz", **trace)
    return {"level": level, "method_id": method_id, "report": report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", nargs="+", choices=("medium", "high"), default=("medium", "high"))
    parser.add_argument("--methods", nargs="+", choices=NEW_METHODS, default=NEW_METHODS)
    parser.add_argument("--workers", type=int, default=4)
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
            tasks.append((level, method_id, record["inspection_order"], args.output.resolve()))
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
