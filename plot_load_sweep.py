"""Journal load-sensitivity figure: objective vs payload level with references.

TD3 points come from the paper's allocated runs (one directory per level); the
A* and MapGuide planner values are per-level references. A second panel
decomposes TD3's objective into T and tau and marks the outage usage against
the fixed 100 s budget.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_formal_results import _save_pair, _style


def load_point(run_dir):
    best = json.loads((Path(run_dir) / "eval_history.json").read_text(encoding="utf-8"))["eval_best"]
    history = json.loads((Path(run_dir) / "eval_history.json").read_text(encoding="utf-8"))["history"]
    successes = sum(1 for item in history if item.get("success"))
    return {
        "T": float(best["task_time_s"]),
        "tau": float(best["transmission_delay_s"]),
        "outage": float(best["outage_time_s"]),
        "successes": successes,
        "evaluations": len(history),
    }


def plot_load_sweep(levels, td3, astar, mapguide, outage_budget, stem, title):
    objectives = [point["T"] + point["tau"] for point in td3]
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), constrained_layout=True)

    axes[0].plot(levels, astar, marker="^", color="0.3", linewidth=1.6,
                 linestyle=(0, (6, 3)), label="A* (offline reference)")
    axes[0].plot(levels, objectives, marker="o", color="#0072B2", linewidth=2.0, label="TD3")
    axes[0].plot(levels, mapguide, marker="s", color="#D55E00", linewidth=1.6,
                 linestyle=(0, (2, 2)), label="MapGuide planner")
    axes[0].set_ylabel(r"$T+\tau_{\mathrm{total}}$ (s)")
    axes[0].set_xlabel("Payload level")
    axes[0].set_title("Objective vs data volume")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper left")

    x = np.arange(len(levels))
    times = [point["T"] for point in td3]
    taus = [point["tau"] for point in td3]
    axes[1].bar(x, times, width=0.55, color="#0072B2", label=r"$T$")
    axes[1].bar(x, taus, width=0.55, bottom=times, color="#56B4E9", label=r"$\tau_{\mathrm{total}}$")
    for index, point in enumerate(td3):
        axes[1].annotate(f"$T_{{out}}$={point['outage']:.1f}s",
                         xy=(x[index], times[index] + taus[index]),
                         xytext=(0, 5), textcoords="offset points",
                         ha="center", fontsize=8, color="0.25")
    axes[1].axhline(outage_budget, color="0.5", linewidth=0.0)  # keep ylim sane only
    axes[1].set_xticks(x, levels)
    axes[1].set_xlabel("Payload level")
    axes[1].set_ylabel("Seconds")
    axes[1].set_title(r"TD3 decomposition ($T_{out}$ budget 100 s, never binding)")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(loc="upper left")
    figure.suptitle(title)
    _save_pair(figure, stem)


def parse_point(raw):
    label, _, run_dir = raw.rpartition("=")
    if not label or not run_dir:
        raise argparse.ArgumentTypeError(f"--point must look like LEVEL=DIR, got {raw!r}")
    return label, run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point", action="append", required=True, metavar="LEVEL=RUN_DIR",
                        help="TD3 run per payload level, in display order")
    parser.add_argument("--astar", type=float, nargs="+", required=True,
                        help="A* reference per level, same order as --point")
    parser.add_argument("--mapguide", type=float, nargs="+", required=True,
                        help="MapGuide planner reference per level")
    parser.add_argument("--outage-budget", type=float, default=100.0)
    parser.add_argument("--output", default="journal_results/load-sweep")
    parser.add_argument("--title", default="Data-volume sensitivity (Variant A)")
    args = parser.parse_args()
    points = [parse_point(raw) for raw in args.point]
    if not (len(points) == len(args.astar) == len(args.mapguide)):
        raise SystemExit("--point/--astar/--mapguide must have matching lengths")
    _style()
    levels = [label for label, _ in points]
    td3 = [load_point(run_dir) for _, run_dir in points]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    plot_load_sweep(levels, td3, args.astar, args.mapguide, args.outage_budget,
                    output / "load_sensitivity", args.title)
    summary = {
        "levels": levels,
        "td3": [{"level": level, **point} for level, point in zip(levels, td3)],
        "astar": dict(zip(levels, args.astar)),
        "mapguide": dict(zip(levels, args.mapguide)),
    }
    (output / "load_sensitivity.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
