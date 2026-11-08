"""Plot the completed V5 A2G and QoS/objective-weight sensitivity results."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
FIGURE_DIR = ROOT.parent / "P3" / "figs" / "v5"
SOURCE_DIR = FIGURE_DIR / "source_data"
QA_DIR = FIGURE_DIR / "qa"
SKILL_SCRIPTS = Path("/Users/ganruijie.1/.codex/skills/nature-figure/scripts")

LEVELS = ("Low", "Medium", "High")
LEVEL_COLORS = {
    "Low": "#0F4D92",
    "Medium": "#B8781A",
    "High": "#6A519A",
}
A2G_COLORS = {
    "blind": "#8C96A8",
    "visible": "#0F4D92",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)


def _number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _load_run(path, *, level, method, setting):
    path = ROOT / path
    config = json.loads((path / "run_config.json").read_text(encoding="utf-8"))
    payload = json.loads((path / "eval_history.json").read_text(encoding="utf-8"))
    history = payload["history"]
    best = payload.get("eval_best") or payload.get("best") or {}
    expected_upload = float(sum(config.get("payloads_mb", [])))
    expected_points = 6

    def complete(item):
        points = int(item.get("visited_inspection_points", 0) or 0)
        uploaded = float(item.get("uploaded_mb", 0.0) or 0.0)
        return points >= expected_points and uploaded >= expected_upload - 1e-7

    completed = [item for item in history if complete(item)]
    feasible = [item for item in completed if item.get("success", False)]
    best_complete = complete(best)
    task_time = _number(best.get("task_time_s")) if best_complete else None
    transmission_delay = (
        _number(best.get("transmission_delay_s")) if best_complete else None
    )
    outage = _number(best.get("outage_time_s"))
    outage_limit = _number(best.get("outage_limit_s")) or _number(
        config.get("outage_max_s")
    )
    successful = (
        best_complete
        and bool(best.get("success", False))
        and outage is not None
        and outage_limit is not None
        and outage <= outage_limit + 1e-9
    )
    visited = int(best.get("visited_inspection_points", 0) or 0)
    return {
        "level": level,
        "method": method,
        "setting": setting,
        "complete": best_complete,
        "qos_feasible": successful,
        "task_time_s": task_time,
        "transmission_delay_s": transmission_delay,
        "objective_s": (
            task_time + transmission_delay
            if task_time is not None and transmission_delay is not None
            else None
        ),
        "contiguous_outage_s": outage,
        "outage_limit_s": outage_limit,
        "visited_points": visited,
        "completion_fraction": visited / expected_points,
        "uploaded_mb": _number(best.get("uploaded_mb")),
        "expected_upload_mb": expected_upload,
        "evaluations": len(history),
        "completed_evaluations": len(completed),
        "feasible_evaluations": len(feasible),
        "feasible_rate_pct": 100.0 * len(feasible) / max(1, len(history)),
        "eval_best_episode": int(best.get("episode", -1) or -1),
        "diagnostic_task_time_s": _number(best.get("task_time_s")),
        "diagnostic_transmission_delay_s": _number(
            best.get("transmission_delay_s")
        ),
        "source_path": str(path.resolve()),
    }


def load_e3():
    rows = [
        _load_run(
            "journal_results/v4-baselines/distance-tsp/A/low/td3_bc",
            level="Low",
            method="A2G-blind",
            setting="reused v4",
        ),
        _load_run(
            "journal_results/v4-baselines/distance-tsp-a2g-visible/A/low/td3_bc",
            level="Low",
            method="A2G-visible",
            setting="reused v4",
        ),
        _load_run(
            "journal_results/v5-mac/E3/distance_tsp_blind/baseline/medium/td3_bc",
            level="Medium",
            method="A2G-blind",
            setting="V5",
        ),
        _load_run(
            "journal_results/v5-mac/E3/distance_tsp_visible/baseline/medium/td3_bc",
            level="Medium",
            method="A2G-visible",
            setting="V5",
        ),
        _load_run(
            "journal_results/v5-mac/E3/distance_tsp_blind/baseline/high/td3_bc",
            level="High",
            method="A2G-blind",
            setting="V5",
        ),
        _load_run(
            "journal_results/v5-mac/E3/distance_tsp_visible/baseline/high/td3_bc",
            level="High",
            method="A2G-visible",
            setting="V5",
        ),
    ]
    return rows


def load_e4_qos():
    rows = []
    for budget in (3, 4, 5, 6, 8):
        rows.append(
            _load_run(
                f"journal_results/v4-qos/w50-50/qos{budget}/A/low/td3_bc",
                level="Low",
                method="PRIGA+CA-TD3",
                setting=f"qos={budget}s, w50-50",
            )
        )
        for level in ("medium", "high"):
            if budget == 6:
                path = (
                    f"journal_results/v4-models/load/{level}/"
                    f"A/{level}/td3_bc"
                )
            else:
                path = (
                    f"journal_results/v5-mac/E4/qos{budget}/w50-50/"
                    f"baseline/{level}/td3_bc"
                )
            rows.append(
                _load_run(
                    path,
                    level=level.capitalize(),
                    method="PRIGA+CA-TD3",
                    setting=f"qos={budget}s, w50-50",
                )
            )
    return rows


def load_e4_weights():
    rows = []
    weight_specs = (
        ("80/20", "w80-20", (0.8, 0.2)),
        ("50/50", "w50-50", (0.5, 0.5)),
        ("20/80", "w20-80", (0.2, 0.8)),
    )
    for label, slug, weights in weight_specs:
        rows.append(
            _load_run(
                f"journal_results/v4-qos/{slug}/qos6/A/low/td3_bc",
                level="Low",
                method="PRIGA+CA-TD3",
                setting=f"qos=6s, w={label}",
            )
        )
        for level in ("medium", "high"):
            if label == "50/50":
                path = (
                    f"journal_results/v4-models/load/{level}/A/{level}/td3_bc"
                )
            else:
                path = (
                    f"journal_results/v5-mac/E4/qos6/{slug}/"
                    f"baseline/{level}/td3_bc"
                )
            rows.append(
                _load_run(
                    path,
                    level=level.capitalize(),
                    method="PRIGA+CA-TD3",
                    setting=f"qos=6s, w={label}",
                )
            )
    return rows


def write_source(rows, stem):
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = SOURCE_DIR / f"{stem}.csv"
    json_path = SOURCE_DIR / f"{stem}.json"
    columns = list(rows[0]) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(
        json.dumps(
            {
                "protocol": {
                    "seed": 1,
                    "uncertainty": "none (single seed)",
                    "valid_objective": "6/6 visits and full upload",
                },
                "records": rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _panel_label(ax, label):
    ax.text(
        -0.13,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def _style_axes(ax):
    ax.grid(axis="y", color="#D9D9D9", lw=0.45, zorder=0)
    ax.set_axisbelow(True)


def _save(fig, axes, output, panel_ids):
    sys.path.insert(0, str(SKILL_SCRIPTS))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    require_matplotlib_panel_alignment(
        fig,
        axes=list(axes),
        panel_ids=panel_ids,
        row_groups=[panel_ids],
        json_out=f"{output}.alignment.json",
        overlay_svg=f"{output}.alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_e3(rows):
    levels = list(LEVELS)
    x = np.arange(len(levels))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.65))
    for ax, field, ylabel in (
        (axes[0], "objective_s", "Objective T+τ (s)"),
        (axes[1], "completion_fraction", "Mission completion fraction"),
    ):
        for offset, method, label in (
            (-width / 2, "A2G-blind", "A2G-blind"),
            (width / 2, "A2G-visible", "A2G-visible"),
        ):
            values = []
            for level in levels:
                row = next(
                    item
                    for item in rows
                    if item["level"] == level and item["method"] == method
                )
                values.append(row[field] if row[field] is not None else np.nan)
            ax.bar(
                x + offset,
                values,
                width=width,
                color=A2G_COLORS["blind" if method == "A2G-blind" else "visible"],
                edgecolor="white",
                linewidth=0.4,
                zorder=2,
                label=label,
            )
        ax.set_xticks(x, levels)
        ax.set_ylabel(ylabel)
        _style_axes(ax)
    axes[0].set_ylim(0, 800)
    axes[1].set_ylim(0, 1.08)
    axes[1].set_yticks([0, 1 / 3, 2 / 3, 1], ["0", "2/6", "4/6", "6/6"])

    high_blind = next(
        item
        for item in rows
        if item["level"] == "High" and item["method"] == "A2G-blind"
    )
    axes[0].text(
        x[2] - width / 2,
        20,
        "no valid\nobjective",
        ha="center",
        va="bottom",
        fontsize=6.2,
        color="#5D5D5D",
    )
    axes[1].text(
        x[2] - width / 2,
        high_blind["completion_fraction"] + 0.04,
        "2/6",
        ha="center",
        va="bottom",
        fontsize=6.2,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=2)
    fig.text(
        0.5,
        -0.005,
        "Single seed; no uncertainty interval. A2G-blind High has no valid "
        "6/6/full-upload objective, so no bar is drawn in panel a.",
        ha="center",
        va="top",
        fontsize=6.4,
    )
    fig.tight_layout(rect=(0, 0.075, 1, 0.92), pad=0.5, w_pad=1.4)
    _save(fig, axes, FIGURE_DIR / "e3_a2g_ablation_v5", ["a", "b"])


def _plot_sensitivity(rows, settings, tick_labels, xlabel, output_stem):
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.7))
    fields = (
        ("objective_s", "Objective T+τ (s)"),
        ("completion_fraction", "Mission completion fraction"),
        ("feasible_rate_pct", "Feasible evaluations (%)"),
    )
    x_by_setting = {setting: index for index, setting in enumerate(settings)}
    handles = []
    for ax, (field, ylabel) in zip(axes, fields):
        for level in LEVELS:
            level_rows = [row for row in rows if row["level"] == level]
            x = np.asarray([x_by_setting[row["setting"]] for row in level_rows])
            y = np.asarray(
                [row[field] if row[field] is not None else np.nan for row in level_rows],
                dtype=float,
            )
            color = LEVEL_COLORS[level]
            line, = ax.plot(x, y, color=color, lw=1.15, alpha=0.8, label=level)
            handles = handles or [line]
            for row in level_rows:
                if row[field] is None:
                    continue
                marker = "o" if row["complete"] and row["qos_feasible"] else (
                    "s" if row["complete"] else "x"
                )
                if marker == "x":
                    ax.scatter(
                        x_by_setting[row["setting"]],
                        row[field],
                        s=18,
                        marker=marker,
                        color=color,
                        linewidths=0.8,
                        zorder=3,
                    )
                else:
                    ax.scatter(
                        x_by_setting[row["setting"]],
                        row[field],
                        s=18,
                        marker=marker,
                        facecolor=color,
                        edgecolor="white",
                        linewidths=0.4,
                        zorder=3,
                    )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(tick_labels)), tick_labels)
        _style_axes(ax)
    axes[1].set_ylim(0, 1.08)
    axes[1].set_yticks([0, 1 / 3, 2 / 3, 1], ["0", "2/6", "4/6", "6/6"])
    axes[2].set_ylim(0, 100)
    axes[2].set_yticks([0, 25, 50, 75, 100])
    for index, ax in enumerate(axes):
        _panel_label(ax, "abc"[index])

    level_handles = [
        plt.Line2D([0], [0], color=LEVEL_COLORS[level], lw=1.2, label=level)
        for level in LEVELS
    ]
    fig.legend(level_handles, LEVELS, loc="upper center", bbox_to_anchor=(0.5, 1.035), ncol=3)
    fig.text(
        0.5,
        -0.015,
        "Single seed. Circles, complete and QoS-feasible; squares, complete but "
        "QoS-infeasible; crosses, incomplete. Missing objective values are not imputed.",
        ha="center",
        va="top",
        fontsize=6.25,
    )
    fig.tight_layout(rect=(0, 0.095, 1, 0.92), pad=0.45, w_pad=1.0)
    _save(fig, axes, FIGURE_DIR / output_stem, ["a", "b", "c"])


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    QA_DIR.mkdir(parents=True, exist_ok=True)
    e3 = load_e3()
    write_source(e3, "e3_a2g_ablation_v5")
    plot_e3(e3)

    qos = load_e4_qos()
    write_source(qos, "e4_qos_sensitivity_v5")
    _plot_sensitivity(
        qos,
        [f"qos={budget}s, w50-50" for budget in (3, 4, 5, 6, 8)],
        ["3", "4", "5", "6", "8"],
        "Contiguous-outage budget (s)",
        "e4_qos_sensitivity_v5",
    )

    weights = load_e4_weights()
    write_source(weights, "e4_weight_sensitivity_v5")
    _plot_sensitivity(
        weights,
        [f"qos=6s, w={label}" for label in ("80/20", "50/50", "20/80")],
        ["80/20", "50/50", "20/80"],
        "Objective weight w1/w2",
        "e4_weight_sensitivity_v5",
    )
    print(FIGURE_DIR)


if __name__ == "__main__":
    main()
