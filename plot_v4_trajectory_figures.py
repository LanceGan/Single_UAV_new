#!/usr/bin/env python3
"""Render the v4 trajectory figures from formal actor checkpoints."""

from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from matplotlib.lines import Line2D

from evaluate_policy import checkpoint_paths, evaluate_checkpoint, load_run_context
from plot_v4_paper_figures import (
    D3QN_COLOR,
    FAILURE_COLOR,
    GRID_COLOR,
    PROPOSED_COLOR,
    REFERENCE_COLOR,
    _panel_label,
    _style,
)

try:
    from audit_panel_alignment import require_matplotlib_panel_alignment
except ImportError:  # pragma: no cover - QA dependency is external
    require_matplotlib_panel_alignment = None


ROOT = Path(__file__).resolve().parent
width_mm = 183.0
WIDTH_DOUBLE_IN = width_mm / 25.4
DEFAULT_OUTPUT = ROOT / "paper_figures" / "v4-next"
BUNDLES = ROOT / "sync_bundles" / "v4_eval_best_actors"
DATA_DIR = ROOT / "results" / "datas"

PROPOSED_RUNS = {
    "Low": ROOT / "journal_results" / "v4-qos" / "w50-50" / "qos6" / "A" / "low" / "td3_bc",
    "Medium": ROOT / "journal_results" / "v4-models" / "load" / "medium" / "A" / "medium" / "td3_bc",
    "High": ROOT / "journal_results" / "v4-models" / "load" / "high" / "A" / "high" / "td3_bc",
}
WEIGHT_RUNS = {
    "(0.8, 0.2)": ROOT / "journal_results" / "v4-qos" / "w80-20" / "qos6" / "A" / "low" / "td3_bc",
    "(0.5, 0.5)": ROOT / "journal_results" / "v4-qos" / "w50-50" / "qos6" / "A" / "low" / "td3_bc",
    "(0.2, 0.8)": ROOT / "journal_results" / "v4-qos" / "w20-80" / "qos6" / "A" / "low" / "td3_bc",
}
WEIGHT_CHECKPOINTS = {
    "(0.8, 0.2)": BUNDLES / "w80-20_qos6_actor.pth",
    "(0.5, 0.5)": BUNDLES / "w50-50_qos6_actor.pth",
    "(0.2, 0.8)": BUNDLES / "w20-80_qos6_actor.pth",
}
PROPOSED_CHECKPOINTS = {
    "Low": BUNDLES / "w50-50_qos6_actor.pth",
    "Medium": BUNDLES / "medium_qos6_actor.pth",
    "High": BUNDLES / "high_qos6_actor.pth",
}
BASELINE_RUNS = {
    "Yang-style+TD3": ROOT / "journal_results" / "v4-baselines" / "yang" / "A" / "low" / "td3_bc",
    "EEGA-TD3": ROOT / "journal_results" / "v4-baselines" / "eega" / "A" / "low" / "td3_bc",
    "Distance-TSP+TD3": ROOT / "journal_results" / "v4-baselines" / "distance-tsp" / "A" / "low" / "td3_bc",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def _evaluate(run_dir: Path, checkpoint: Path) -> tuple[dict, dict]:
    context = load_run_context(run_dir)
    with contextlib.redirect_stdout(io.StringIO()):
        report, trace = evaluate_checkpoint(context, checkpoint)
    return report, trace


def _baseline_checkpoint(run_dir: Path) -> Path:
    context = load_run_context(run_dir)
    paths = checkpoint_paths(run_dir, context.algorithm, int(context.config["episodes"]))
    return paths["eval_best"]


def _map_data() -> tuple[dict, dict]:
    return np.load(DATA_DIR / "Radio_datas.npz"), np.load(DATA_DIR / "Radio_datas_A2G.npz")


def _map_layers(g2a: dict, a2g: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray(g2a["arr_2"]) * 10.0,
        np.asarray(g2a["arr_3"]) * 10.0,
        1.0 - np.asarray(g2a["arr_0"]),
        np.asarray(a2g["arr_1"]),
    )


def _draw_map(
    axis,
    x_grid,
    y_grid,
    values,
    *,
    cmap,
    vmin,
    vmax,
    title,
    users,
    show_numbering=False,
):
    mesh = axis.contourf(x_grid, y_grid, values, levels=np.linspace(vmin, vmax, 28),
                         cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_xlim(0, 2000)
    axis.set_ylim(0, 2000)
    axis.set_aspect("equal")
    axis.set_xticks([0, 500, 1000, 1500, 2000])
    axis.set_yticks([0, 500, 1000, 1500, 2000])
    axis.set_xlabel("X (m)")
    axis.set_ylabel("Y (m)")
    axis.text(
        0.015,
        0.985,
        title,
        transform=axis.transAxes,
        ha="left",
        va="top",
        color="white",
        fontsize=7.0,
        path_effects=[path_effects.withStroke(linewidth=1.5, foreground="#202020")],
        zorder=8,
    )
    axis.scatter(users[:, 0], users[:, 1], s=19, marker="^", color="#202020",
                 edgecolor="white", linewidth=0.35, zorder=5)
    axis.scatter(262, 265, s=30, color="#D94B3D", edgecolor="white",
                 linewidth=0.5, zorder=6)
    axis.scatter(1895, 1112, s=32, color="#E39A35", edgecolor="white",
                 linewidth=0.5, zorder=6)
    if show_numbering:
        for number, point in enumerate(users, start=1):
            axis.annotate(str(number), point, xytext=(3, 4), textcoords="offset points",
                          color="white", fontsize=6.2, fontweight="bold", zorder=7)
    return mesh


def _draw_path(axis, trace, color, label=None, linewidth=1.35):
    positions = np.asarray(trace["positions"], dtype=float) * 100.0
    axis.plot(positions[:, 0], positions[:, 1], color="#202020",
              linewidth=linewidth + 1.0, alpha=0.25, zorder=2)
    axis.plot(positions[:, 0], positions[:, 1], color=color,
              linewidth=linewidth, label=label, zorder=3)


def _save(figure, stem: Path, *, exclude_axes=None):
    if require_matplotlib_panel_alignment is None:
        raise RuntimeError("nature-figure panel alignment helper is unavailable")
    figure.canvas.draw()
    require_matplotlib_panel_alignment(
        figure,
        json_out=stem.with_suffix(".alignment.json"),
        overlay_svg=stem.with_suffix(".alignment.svg"),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        exclude_axes=exclude_axes or [],
        strict=True,
    )
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches=None, metadata={"Title": stem.name})
    figure.savefig(stem.with_suffix(".svg"), bbox_inches=None, metadata={"Title": stem.name})
    svg_path = stem.with_suffix(".svg")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n",
        encoding="utf-8",
    )
    figure.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches=None)
    figure.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches=None)
    plt.close(figure)


def _tier_trajectories(output: Path) -> None:
    g2a, a2g = _map_data()
    x_grid, y_grid, coverage, snr = _map_layers(g2a, a2g)
    users = np.loadtxt(DATA_DIR / "Users_6.txt")[:, :2] * 100.0
    figure, axes = plt.subplots(3, 2, figsize=(WIDTH_DOUBLE_IN, 5.9))
    figure.subplots_adjust(left=0.075, right=0.9, bottom=0.06, top=0.93, wspace=0.12, hspace=0.23)
    payload_totals = {"Low": 450, "Medium": 900, "High": 1500}
    traces = {}
    coverage_mesh = None
    snr_mesh = None
    for row_index, (level, run_dir) in enumerate(PROPOSED_RUNS.items()):
        report, trace = _evaluate(run_dir, PROPOSED_CHECKPOINTS[level])
        traces[level] = trace
        objective = report["task_time_s"] + report["transmission_delay_s"]
        title = f"{level}, {payload_totals[level]} MB, {objective:.1f} s"
        coverage_mesh = _draw_map(
            axes[row_index, 0], x_grid, y_grid, coverage, cmap="viridis",
            vmin=0.0, vmax=1.0, title=f"{title} | G2A", users=users,
        )
        snr_mesh = _draw_map(
            axes[row_index, 1], x_grid, y_grid, snr, cmap="viridis",
            vmin=float(snr.min()), vmax=float(snr.max()), title="A2G SNR",
            users=users,
        )
        _draw_path(axes[row_index, 0], trace, "white")
        _draw_path(axes[row_index, 1], trace, "white")
    for index, axis in enumerate(axes.ravel()):
        _panel_label(axis, "abcdef"[index])
    coverage_cax = figure.add_axes([0.92, 0.55, 0.012, 0.34])
    snr_cax = figure.add_axes([0.92, 0.12, 0.012, 0.34])
    coverage_bar = figure.colorbar(coverage_mesh, cax=coverage_cax)
    coverage_bar.set_label("G2A coverage probability")
    coverage_bar.ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
    snr_bar = figure.colorbar(snr_mesh, cax=snr_cax)
    snr_bar.set_label("A2G SNR (dB)")
    snr_bar.ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
    np.savez(output / "trajectory_tiers_v4_data.npz",
             **{f"{level}_positions": trace["positions"] for level, trace in traces.items()})
    _save(figure, output / "trajectory_tiers_v4", exclude_axes=[coverage_cax, snr_cax])


def _weight_trajectories(output: Path) -> None:
    g2a, a2g = _map_data()
    x_grid, y_grid, _, snr = _map_layers(g2a, a2g)
    users = np.loadtxt(DATA_DIR / "Users_6.txt")[:, :2] * 100.0
    figure, axes = plt.subplots(1, 2, figsize=(WIDTH_DOUBLE_IN, 3.05))
    figure.subplots_adjust(left=0.07, right=0.87, bottom=0.21, top=0.82, wspace=0.16)
    colors = ["#B56B45", "#236D86", "#7A6F9B"]
    traces = {}
    records = []
    mesh = None
    for color, (label, run_dir) in zip(colors, WEIGHT_RUNS.items()):
        report, trace = _evaluate(run_dir, WEIGHT_CHECKPOINTS[label])
        traces[label] = trace
        records.append((label, report, color))
        mesh = _draw_map(
            axes[0], x_grid, y_grid, snr, cmap="viridis",
            vmin=float(snr.min()), vmax=float(snr.max()),
            title="Weighted trajectories | A2G SNR", users=users,
        )
        _draw_path(axes[0], trace, color, linewidth=1.25)

    axes[0].legend(
        handles=[Line2D([0], [0], color=color, linewidth=1.5, label=label)
                 for label, _, color in records],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=3,
        fontsize=6.5,
    )
    for label, report, color in records:
        task = report["task_time_s"]
        delay = report["transmission_delay_s"]
        axes[1].scatter(task, delay, s=42, color=color, edgecolor="white",
                        linewidth=0.6, zorder=3)
        if label == "(0.5, 0.5)":
            offset, align, vertical = (-5, -9), "right", "top"
        else:
            offset, align, vertical = (5, 5), "left", "bottom"
        axes[1].annotate(
            f"{label}\n{task + delay:.1f} s",
            xy=(task, delay),
            xytext=offset,
            textcoords="offset points",
            color=color,
            ha=align,
            va=vertical,
        )
    axes[1].set_xlim(145, 182)
    axes[1].set_ylim(70, 93)
    axes[1].set_xticks([150, 160, 170, 180])
    axes[1].set_yticks([70, 75, 80, 85, 90])
    axes[1].set_xlabel("Mission time T (s)")
    axes[1].set_ylabel("Transmission delay τ (s)")
    axes[1].set_title("Realized T–τ trade-off", loc="left", pad=3)
    _panel_label(axes[0], "a")
    _panel_label(axes[1], "b")
    color_cax = figure.add_axes([0.895, 0.23, 0.012, 0.62])
    color_bar = figure.colorbar(mesh, cax=color_cax)
    color_bar.set_label("A2G SNR (dB)")
    color_bar.ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
    np.savez(output / "weight_trajectories_v4_data.npz",
             **{f"{weight}_positions": trace["positions"] for weight, trace in traces.items()})
    _save(figure, output / "weight_trajectories_and_tradeoff_v4", exclude_axes=[color_cax])


def _baseline_trajectories(output: Path) -> None:
    g2a, a2g = _map_data()
    x_grid, y_grid, _, snr = _map_layers(g2a, a2g)
    users = np.loadtxt(DATA_DIR / "Users_6.txt")[:, :2] * 100.0
    rows = [("DRM-HO", PROPOSED_RUNS["Low"], PROPOSED_CHECKPOINTS["Low"], PROPOSED_COLOR)]
    for label, run_dir in BASELINE_RUNS.items():
        rows.append((label, run_dir, _baseline_checkpoint(run_dir), "#6F747A"))
    figure, axes = plt.subplots(2, 2, figsize=(WIDTH_DOUBLE_IN, 4.15))
    figure.subplots_adjust(left=0.07, right=0.86, bottom=0.12, top=0.92, wspace=0.12, hspace=0.23)
    traces = {}
    mesh = None
    for index, (label, run_dir, checkpoint, color) in enumerate(rows):
        report, trace = _evaluate(run_dir, checkpoint)
        traces[label] = trace
        objective = report["task_time_s"] + report["transmission_delay_s"]
        axis = axes.ravel()[index]
        mesh = _draw_map(
            axis, x_grid, y_grid, snr, cmap="viridis",
            vmin=float(snr.min()), vmax=float(snr.max()),
            title=f"{label}\n{objective:.1f} s", users=users,
        )
        _draw_path(axis, trace, color)
        _panel_label(axis, "abcd"[index])
    color_cax = figure.add_axes([0.885, 0.19, 0.012, 0.66])
    color_bar = figure.colorbar(mesh, cax=color_cax)
    color_bar.set_label("A2G SNR (dB)")
    color_bar.ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
    np.savez(output / "traj_baselines_low_data.npz",
             **{f"{label}_positions": trace["positions"] for label, trace in traces.items()})
    _save(figure, output / "traj_baselines_low", exclude_axes=[color_cax])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    _style()
    _tier_trajectories(args.output)
    _weight_trajectories(args.output)
    _baseline_trajectories(args.output)
    print(f"Wrote trajectory figures to {args.output}")


if __name__ == "__main__":
    main()
