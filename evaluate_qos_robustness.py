"""Post-hoc QoS robustness evaluation for previously trained TD3 checkpoints.

The actors are trained once with the maximum-contiguous-outage budget recorded in each run's
``run_config.json``.  This script changes only the evaluation budget, replays
the saved periodic actor snapshots, and selects the best feasible snapshot for
each weight setting.  It therefore measures the robustness of existing
policies to a tighter QoS constraint; it does not measure the effect of
retraining under each budget.
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluate_policy import evaluate_checkpoint, load_run_context, selection_key
from plot_formal_results import _save_pair, _style


ROOT = Path(__file__).resolve().parent
ACTOR_RE = re.compile(r".*_actor(?P<episode>\d+)\.pth$")
DEFAULT_BUDGETS = (3.0, 4.0, 5.0, 6.0, 8.0)
DEFAULT_RUNS = (
    "w80/20=journal_results/weight-sweep/w80-20/A/low/td3_bc",
    "w50/50=journal_results/formal-v7/A/low/td3_bc",
    "w20/80=journal_results/weight-sweep/w20-80/A/low/td3_bc",
)


def parse_run(raw):
    label, separator, run_dir = raw.partition("=")
    if not separator or not label or not run_dir:
        raise ValueError(f"invalid run specification: {raw!r}; expected LABEL=RUN_DIR")
    return label, Path(run_dir)


def parse_budgets(values):
    budgets = []
    for value in values:
        budgets.extend(float(item) for item in value.split(",") if item.strip())
    return sorted(set(budgets))


def actor_snapshots(run_dir):
    snapshots = []
    for path in (Path(run_dir) / "models").glob("*_actor*.pth"):
        match = ACTOR_RE.fullmatch(path.name)
        if match:
            snapshots.append((int(match.group("episode")), path))
    if not snapshots:
        raise FileNotFoundError(f"no periodic actor snapshots in {Path(run_dir) / 'models'}")
    return sorted(snapshots)


def _point_from_reports(reports, budget, w1, w2):
    feasible = [
        row for row in reports
        if row["report"]["success"] and row["report"]["outage_time_s"] <= budget + 1e-9
    ]
    best_feasible = min(feasible, key=lambda row: selection_key(row["report"], w1, w2)) if feasible else None
    best_progress = min(reports, key=lambda row: selection_key(row["report"], w1, w2))
    report = best_feasible["report"] if best_feasible else None
    successes = len(feasible)
    return {
        "budget_s": float(budget),
        "objective_s": (float(2.0 * (w1 * report["task_time_s"]
                                    + w2 * report["transmission_delay_s"]))
                        if report else None),
        "unweighted_objective_s": (float(report["task_time_s"]
                                         + report["transmission_delay_s"])
                                   if report else None),
        "task_time_s": float(report["task_time_s"]) if report else None,
        "transmission_delay_s": (float(report["transmission_delay_s"])
                                  if report else None),
        "outage_s": float(report["outage_time_s"]) if report else None,
        "total_outage_s": float(report["total_outage_time_s"]) if report else None,
        "success": bool(report),
        "best_feasible_episode": int(best_feasible["episode"]) if best_feasible else None,
        "best_feasible_checkpoint": (str(best_feasible["checkpoint"])
                                     if best_feasible else None),
        "best_progress_episode": int(best_progress["episode"]),
        "best_progress_termination_reason": best_progress["report"]["termination_reason"],
        "best_progress_outage_s": float(best_progress["report"]["outage_time_s"]),
        "snapshot_evaluations": len(reports),
        "snapshot_successes": successes,
        "snapshot_success_rate": successes / len(reports),
    }


def evaluate_run(label, run_dir, budgets, progress=print):
    context = load_run_context(run_dir)
    config = context.config
    w1 = float(config.get("objective_w1", 0.5))
    w2 = float(config.get("objective_w2", 0.5))
    snapshots = actor_snapshots(context.run_dir)
    reports = []
    for index, (episode, checkpoint) in enumerate(snapshots, start=1):
        report, _ = evaluate_checkpoint(context, checkpoint)
        reports.append({"episode": episode, "checkpoint": checkpoint, "report": report})
        if index % 10 == 0 or index == len(snapshots):
            progress(f"{label} reference replay snapshot {index}/{len(snapshots)}", flush=True)
    points = [_point_from_reports(reports, budget, w1, w2) for budget in budgets]
    return {
        "label": label,
        "run_dir": str(context.run_dir),
        "training_budget_s": context.outage_max_s,
        "objective_w1": w1,
        "objective_w2": w2,
        "snapshot_count": len(snapshots),
        "points": points,
    }


def load_astar_points(paths):
    points = []
    for raw in paths:
        budget, separator, directory = raw.partition("=")
        if not separator:
            raise ValueError(f"invalid A* point: {raw!r}; expected BUDGET=DIR")
        metrics_path = Path(directory)
        if metrics_path.is_dir():
            metrics_path = metrics_path / "low" / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        replay = metrics["world_replay"]
        points.append({
            "budget_s": float(budget),
            "success": bool(metrics["planner"]["success"] and replay["success"]),
            "objective_s": float(replay["objective_s"]),
            "outage_s": float(replay["outage_time_s"]),
            "planner_status": metrics["planner"]["status"],
        })
    return sorted(points, key=lambda point: point["budget_s"])


def plot_robustness(runs, astar_points, stem, title):
    colors = ("#D55E00", "#0072B2", "#009E73")
    markers = ("s", "o", "^")
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), constrained_layout=True)
    for index, run in enumerate(runs):
        points = run["points"]
        budgets = [point["budget_s"] for point in points]
        objectives = np.asarray([
            np.nan if point["objective_s"] is None else point["objective_s"]
            for point in points
        ])
        outages = np.asarray([
            np.nan if point["outage_s"] is None else point["outage_s"]
            for point in points
        ])
        color = colors[index % len(colors)]
        marker = markers[index % len(markers)]
        label = rf"$w_1={run['objective_w1']:.1f}, w_2={run['objective_w2']:.1f}$"
        axes[0].plot(budgets, objectives,
                     color=color, marker=marker, linewidth=1.8, label=label)
        axes[1].plot(budgets, outages,
                     color=color, marker=marker, linewidth=1.8, label=label)
        axes[2].plot(budgets, [100.0 * point["snapshot_success_rate"] for point in points],
                     color=color, marker=marker, linewidth=1.8, label=label)

    if astar_points:
        feasible = [point for point in astar_points if point["success"]]
        if feasible:
            axes[0].plot([point["budget_s"] for point in feasible],
                         [point["objective_s"] for point in feasible],
                         color="0.25", linestyle="--", marker="x", linewidth=1.2,
                         label="QoS-constrained A*")
            axes[1].plot([point["budget_s"] for point in feasible],
                         [point["outage_s"] for point in feasible],
                         color="0.25", linestyle="--", marker="x", linewidth=1.2,
                         label="QoS-constrained A*")

    all_budgets = sorted({
        point["budget_s"] for run in runs for point in run["points"]
    } | {point["budget_s"] for point in astar_points})
    axes[0].set_ylabel(r"$2(w_1T+w_2\tau_{\mathrm{total}})$ (s)")
    axes[0].set_title("Weighted objective")
    axes[1].plot(all_budgets, all_budgets, color="0.55", linestyle=":",
                 linewidth=1.1, label="QoS budget")
    axes[1].set_ylabel(r"Realized $T_{\mathrm{out}}^{\mathrm{cont}}$ (s)")
    axes[1].set_title("Longest continuous outage")
    axes[2].set_ylabel("Feasible saved actors (%)")
    axes[2].set_title("Snapshot feasibility")
    axes[2].set_ylim(0, 105)
    for axis in axes:
        axis.set_xlabel(r"Evaluation budget $T_{\mathrm{out}}^{\max}$ (s)")
        axis.grid(alpha=0.25)
        axis.legend(loc="best", fontsize=8)
    figure.suptitle(title)
    _save_pair(figure, stem)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", default=None,
                        metavar="LABEL=RUN_DIR",
                        help="trained run; defaults to the three objective weightings")
    parser.add_argument("--budget", action="append", default=None,
                        help="one or more comma-separated evaluation budgets")
    parser.add_argument("--astar", action="append", default=None,
                        metavar="BUDGET=METRICS_OR_DIR")
    parser.add_argument("--output", default="journal_results/qos-sweep-reweighted")
    parser.add_argument("--title", default="QoS robustness of existing TD3 policies")
    args = parser.parse_args()

    runs = [parse_run(raw) for raw in (args.run or DEFAULT_RUNS)]
    budgets = parse_budgets(args.budget or [",".join(map(str, DEFAULT_BUDGETS))])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    _style()

    astar_points = load_astar_points(args.astar or [])
    results = [
        evaluate_run(label, run_dir, budgets, progress=lambda *values, **kwargs:
                     print(*values, **kwargs))
        for label, run_dir in runs
    ]
    summary = {
        "protocol": (
            "Post-hoc deterministic replay of existing periodic actor checkpoints. "
            "Actor training used each run's recorded state scale and maximum-contiguous "
            "outage budget. A trajectory is feasible under each tighter test budget when "
            "its realized maximum continuous outage does not exceed that budget; the "
            "policy input scale stays fixed. "
            "No retraining."
        ),
        "selection": "feasible first, then minimum 2*(objective_w1*T + objective_w2*tau)",
        "budgets_s": budgets,
        "astar": astar_points,
        "runs": results,
    }
    (output / "qos_reweighted.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_robustness(results, astar_points, output / "qos_reweighted", args.title)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
