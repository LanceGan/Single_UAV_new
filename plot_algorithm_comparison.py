"""Overlay several runs' evaluation results into paper-ready comparison figures.

Each run contributes its feasible T+tau trajectory (line) and infeasible
evaluations (crosses). Reference planners (A*, MapGuide) appear as horizontal
lines, so one figure answers which learners work and how close they get.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_formal_results import _save_pair, _style, evaluation_series

METHOD_COLORS = {"TD3": "#0072B2", "DDPG": "#D55E00", "D3QN": "#009E73"}


def load_series(run_dir):
    history = json.loads((Path(run_dir) / "eval_history.json").read_text(encoding="utf-8"))
    return evaluation_series(history)


def summarize(label, series):
    feasible = series["objective_feasible"]
    has_success = bool(np.any(series["success"]))
    best = None
    if has_success:
        index = int(np.nanargmin(feasible))
        best = {
            "episode": int(series["episodes"][index]),
            "objective_s": float(feasible[index]),
            "task_time_s": float(series["task_time"][index]),
            "transmission_delay_s": float(series["delay"][index]),
        }
    return {
        "label": label,
        "evaluations": int(series["episodes"].size),
        "successes": int(series["success"].sum()),
        "success_rate": float(series["success"].mean()) if series["episodes"].size else 0.0,
        "best_feasible": best,
    }


def _color(label, index):
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return METHOD_COLORS.get(label, cycle[index % len(cycle)])


def plot_objective_comparison(entries, references, stem, title):
    """Feasible objectives carry values; infeasible evaluations only mark presence.

    A failed episode's raw termination value is not an objective (early QoS
    deaths look deceptively small), so failures are pinned to per-method rug
    lanes above the value axis instead of being placed at their raw numbers.
    """
    figure, axis = plt.subplots(figsize=(8.6, 5.2), constrained_layout=True)
    tops = [value for _, value in references]
    for index, (label, series) in enumerate(entries):
        color = _color(label, index)
        if np.any(series["success"]):
            axis.plot(series["episodes"], series["objective_feasible"], color=color,
                      marker="o", markersize=3.0, linewidth=1.6,
                      label=f"{label} (feasible evaluations)")
            tops.append(float(np.nanmax(series["objective_feasible"])))

    base = max(tops) if tops else 500.0
    lane_step = 0.055 * base
    x_left = min(series["episodes"].min() for _, series in entries)
    x_right = max(series["episodes"].max() for _, series in entries)
    for index, (label, series) in enumerate(entries):
        color = _color(label, index)
        lane_y = base * 1.10 + index * lane_step
        failed_episodes = series["episodes"][~series["success"]]
        axis.scatter(failed_episodes, np.full(failed_episodes.shape, lane_y),
                     marker="|", s=52, color=color, alpha=0.85, linewidths=1.1)
        axis.text(x_left, lane_y + lane_step * 0.18,
                  f"{label}: infeasible evaluations ({failed_episodes.size}/{series['episodes'].size})",
                  fontsize=8, color=color, ha="left", va="bottom")

    for index, (name, value) in enumerate(references):
        style = (0, (6, 3)) if index % 2 == 0 else (0, (2, 2))
        axis.axhline(value, color="0.3", linewidth=1.1, linestyle=style)
        axis.annotate(f"{name}: {value:.1f} s", xy=(x_right, value),
                      xytext=(-4, 4), textcoords="offset points",
                      ha="right", va="bottom", fontsize=9, color="0.25")
    axis.set_ylim(0, base * 1.10 + len(entries) * lane_step + lane_step)
    axis.set_xlabel("Training episode")
    axis.set_ylabel(r"$T+\tau_{\mathrm{total}}$ (s)")
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend(loc="center right", framealpha=0.92)
    _save_pair(figure, stem)


def plot_success_comparison(entries, stem, title):
    labels = [label for label, _ in entries]
    rates = [100.0 * float(series["success"].mean()) for _, series in entries]
    colors = [_color(label, index) for index, label in enumerate(labels)]
    figure, axis = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)
    bars = axis.bar(labels, rates, color=colors, width=0.55)
    for bar, (label, series) in zip(bars, entries):
        successes = int(series["success"].sum())
        total = int(series["episodes"].size)
        note = f"{successes}/{total}"
        if successes:
            note += f"\nbest {np.nanmin(series['objective_feasible']):.1f} s"
        axis.annotate(note, xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                      xytext=(0, 4), textcoords="offset points",
                      ha="center", va="bottom", fontsize=9)
    axis.set_ylabel("Feasible evaluation rate (%)")
    axis.set_ylim(0, 100)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    _save_pair(figure, stem)


def parse_pair(raw, kind, cast=str):
    name, _, value = raw.partition("=")
    if not name or not value:
        raise argparse.ArgumentTypeError(f"{kind} must look like name=value, got {raw!r}")
    return name, cast(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True,
                        metavar="LABEL=RUN_DIR",
                        help="evaluation history source, e.g. TD3=journal_results/.../td3_bc")
    parser.add_argument("--reference", action="append", default=[],
                        metavar="NAME=SECONDS", help="horizontal reference line")
    parser.add_argument("--output", default="journal_results/comparison-low")
    parser.add_argument("--title", default="Algorithm comparison")
    args = parser.parse_args()
    _style()

    entries = []
    for raw in args.run:
        label, run_dir = parse_pair(raw, "--run")
        entries.append((label, load_series(run_dir)))
    references = [parse_pair(raw, "--reference", float) for raw in args.reference]

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    plot_objective_comparison(entries, references, output / "comparison_objective", args.title)
    plot_success_comparison(entries, output / "comparison_success", args.title)
    summary = {
        "title": args.title,
        "references": {name: value for name, value in references},
        "methods": [summarize(label, series) for label, series in entries],
    }
    (output / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
