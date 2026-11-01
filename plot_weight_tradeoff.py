"""Objective-weight tradeoff: trajectory overlay on the dual maps plus a T-tau scatter.

Each run contributes its guide-free eval_best trajectory; weights enter training
through the per-slot objective terms, so trajectory divergence (straighter and
faster for time-heavy weights, deeper high-SNR detours for delay-heavy weights)
is the behavioural evidence that the weighting works.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_formal_results import _save_pair, _style

ROOT = Path(__file__).resolve().parent
COLORS = ("#D55E00", "#FFFFFF", "#56B4E9")


def load_run(run_dir):
    run_dir = Path(run_dir)
    best = json.loads((run_dir / "eval_history.json").read_text(encoding="utf-8"))["eval_best"]
    trace = np.load(run_dir / "paper_figures" / "trajectory_eval_best.npz")
    return {
        "T": float(best["task_time_s"]),
        "tau": float(best["transmission_delay_s"]),
        "outage": float(best["outage_time_s"]),
        "positions": np.asarray(trace["positions"], dtype=float) * 100.0,
    }


def plot_overlay(entries, stem, title):
    data_dir = ROOT / "results" / "datas"
    g2a = np.load(data_dir / "Radio_datas.npz")
    a2g = np.load(data_dir / "Radio_datas_A2G.npz")
    users = np.loadtxt(data_dir / "Users_6.txt")[:, :2] * 100.0
    start = np.array([2.62, 2.65]) * 100.0
    end = np.array([18.95, 11.12]) * 100.0

    figure, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    panels = (
        (axes[0], 1.0 - g2a["arr_0"], g2a, "G2A coverage probability"),
        (axes[1], a2g["arr_1"], a2g, "A2G SNR"),
    )
    for axis, background, map_data, panel_title in panels:
        axis.contourf(np.asarray(map_data["arr_2"]) * 10.0,
                      np.asarray(map_data["arr_3"]) * 10.0, background, levels=30)
        for index, (label, run) in enumerate(entries):
            color = COLORS[index % len(COLORS)]
            axis.plot(run["positions"][:, 0], run["positions"][:, 1], color=color,
                      linewidth=2.0, label=f"{label} (T={run['T']:.0f}, " + r"$\tau$" + f"={run['tau']:.0f})")
        axis.scatter(users[:, 0], users[:, 1], c="black", marker="^", s=55, zorder=3)
        axis.scatter(*start, c="red", s=70, zorder=4)
        axis.scatter(*end, c="orange", s=70, zorder=4)
        axis.set_title(panel_title)
        axis.set_xlabel("X (m)")
        axis.set_ylabel("Y (m)")
        axis.legend(loc="upper right", fontsize=8, framealpha=0.9)
    figure.suptitle(title)
    _save_pair(figure, stem)


def plot_tradeoff(entries, stem, title):
    figure, axis = plt.subplots(figsize=(5.6, 4.6), constrained_layout=True)
    taus = [run["tau"] for _, run in entries]
    times = [run["T"] for _, run in entries]
    axis.plot(taus, times, color="0.6", linewidth=1.0, linestyle="--", zorder=1)
    for index, (label, run) in enumerate(entries):
        color = COLORS[index % len(COLORS)] if index != 1 else "#0072B2"
        axis.scatter(run["tau"], run["T"], s=70, color=color, zorder=3)
        axis.annotate(label, (run["tau"], run["T"]), xytext=(6, 6),
                      textcoords="offset points", fontsize=9)
    axis.set_xlabel(r"$\tau_{\mathrm{total}}$ (s)")
    axis.set_ylabel(r"$T$ (s)")
    axis.set_title(title)
    axis.grid(alpha=0.25)
    _save_pair(figure, stem)


def parse_pair(raw):
    label, _, run_dir = raw.rpartition("=")
    if not label or not run_dir:
        raise argparse.ArgumentTypeError(f"--run must look like LABEL=DIR, got {raw!r}")
    return label, run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=RUN_DIR")
    parser.add_argument("--output", default="journal_results/weight-sweep")
    parser.add_argument("--title", default="Objective-weight tradeoff (Low load)")
    args = parser.parse_args()
    _style()
    entries = [(label, load_run(run_dir)) for label, run_dir in (parse_pair(raw) for raw in args.run)]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    plot_overlay(entries, output / "weight_trajectories_dual_maps", args.title)
    plot_tradeoff(entries, output / "weight_tradeoff", args.title)
    summary = {label: {key: run[key] for key in ("T", "tau", "outage")} for label, run in entries}
    (output / "weight_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
