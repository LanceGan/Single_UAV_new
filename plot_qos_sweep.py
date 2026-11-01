"""QoS-sensitivity figure: objective, success rate and used outage vs the budget.

Each point is one full training run evaluated by its guide-free eval_best; the
A* planner value is drawn as the offline reference. A flat curve with used
outage far below every budget is itself the finding: the constraint is not
binding for map-aware policies.
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
    data = json.loads((Path(run_dir) / "eval_history.json").read_text(encoding="utf-8"))
    history = data["history"]
    best = data["eval_best"]
    successes = sum(1 for item in history if item.get("success"))
    return {
        "objective_s": float(best["task_time_s"] + best["transmission_delay_s"]),
        "outage_s": float(best["outage_time_s"]),
        "success_rate": successes / len(history),
        "evaluations": len(history),
        "successes": successes,
        "best_episode": int(best["episode"]),
    }


def parse_point(raw):
    budget, _, run_dir = raw.partition("=")
    return float(budget), run_dir


def plot_sweep(points, astar_value, stem, title):
    budgets = [budget for budget, _ in points]
    objectives = [point["objective_s"] for _, point in points]
    outages = [point["outage_s"] for _, point in points]
    rates = [100.0 * point["success_rate"] for _, point in points]

    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), constrained_layout=True)
    axes[0].plot(budgets, objectives, marker="o", color="#0072B2", linewidth=1.8,
                 label=r"TD3 eval\_best $T+\tau_{\mathrm{total}}$")
    if astar_value is not None:
        axes[0].axhline(astar_value, color="0.3", linestyle=(0, (6, 3)), linewidth=1.2)
        axes[0].annotate(f"A*: {astar_value:.1f} s (all budgets)",
                         xy=(budgets[-1], astar_value), xytext=(-4, 4),
                         textcoords="offset points", ha="right", fontsize=9, color="0.25")
    axes[0].set_xlabel(r"Outage budget $T_{\mathrm{out}}^{\max}$ (s)")
    axes[0].set_ylabel(r"$T+\tau_{\mathrm{total}}$ (s)")
    axes[0].set_title("Objective vs QoS budget")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].plot(budgets, outages, marker="s", color="#D55E00", linewidth=1.8,
                 label=r"Used outage $T_{\mathrm{out}}$ (eval\_best)")
    axes[1].plot(budgets, budgets, color="0.4", linestyle=":", linewidth=1.2,
                 label="Budget (binding line)")
    axis2 = axes[1].twinx()
    axis2.bar(budgets, rates, width=4.5, color="#0072B2", alpha=0.25,
              label="Feasible evaluation rate")
    axis2.set_ylabel("Feasible evaluation rate (%)")
    axis2.set_ylim(0, 100)
    axes[1].set_xlabel(r"Outage budget $T_{\mathrm{out}}^{\max}$ (s)")
    axes[1].set_ylabel(r"$T_{\mathrm{out}}$ (s)")
    axes[1].set_title("Outage usage stays far below every budget")
    axes[1].grid(alpha=0.25)
    handles1, labels1 = axes[1].get_legend_handles_labels()
    handles2, labels2 = axis2.get_legend_handles_labels()
    axes[1].legend(handles1 + handles2, labels1 + labels2, loc="upper left", fontsize=8)
    figure.suptitle(title)
    _save_pair(figure, stem)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point", action="append", required=True,
                        metavar="BUDGET=RUN_DIR", help="e.g. 60=journal_results/qos-sweep/qos60/A/low/td3_bc")
    parser.add_argument("--astar", type=float, default=None, help="flat A* reference value")
    parser.add_argument("--output", default="journal_results/qos-sweep")
    parser.add_argument("--title", default="QoS sensitivity (Low load, Variant A)")
    args = parser.parse_args()
    _style()
    points = sorted((parse_point(raw) for raw in args.point), key=lambda pair: pair[0])
    loaded = [(budget, load_point(run_dir)) for budget, run_dir in points]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    plot_sweep(loaded, args.astar, output / "qos_sensitivity", args.title)
    summary = {
        "astar_reference_s": args.astar,
        "points": [{"budget_s": budget, **point} for budget, point in loaded],
    }
    (output / "qos_sensitivity.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
