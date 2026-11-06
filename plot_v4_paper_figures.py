#!/usr/bin/env python3
"""Build and audit the paper figures for the v4 maximum-contiguous-outage study.

The script reads the raw run directories, validates the common system contract,
and writes source data plus publication-oriented figures. It intentionally
separates failed evaluations from feasible objectives: a PPO run with 0/120
feasible evaluations is displayed as a failure, never as the minimum objective.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from audit_panel_alignment import require_matplotlib_panel_alignment
except ImportError:  # pragma: no cover - exercised only without the QA helper
    require_matplotlib_panel_alignment = None


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "paper_figures" / "v4"
width_mm = 183.0
WIDTH_DOUBLE_IN = width_mm / 25.4
WIDTH_SINGLE_IN = 89.0 / 25.4
QOS_BUDGETS = (3, 4, 5, 6, 8)

QOS_WEIGHTS = (
    {
        "id": "w80-20",
        "w1": 0.8,
        "w2": 0.2,
        "label": "80/20",
        "long_label": "(0.8, 0.2)",
        "color": "#D55E00",
        "marker": "o",
    },
    {
        "id": "w50-50",
        "w1": 0.5,
        "w2": 0.5,
        "label": "50/50",
        "long_label": "(0.5, 0.5)",
        "color": "#0B6E99",
        "marker": "s",
    },
    {
        "id": "w20-80",
        "w1": 0.2,
        "w2": 0.8,
        "label": "20/80",
        "long_label": "(0.2, 0.8)",
        "color": "#4C78A8",
        "marker": "^",
    },
)

PROPOSED_COLOR = "#0B6E99"
PPO_COLOR = "#6A51A3"
D3QN_COLOR = "#2A9D8F"
BLIND_COLOR = "#B07C2B"
VISIBLE_COLOR = "#2A9D8F"
REFERENCE_COLOR = "#5F6368"
METHOD_COLORS = {
    "Proposed (IGA+TD3)": PROPOSED_COLOR,
    "Yang-style+TD3": "#8C6D31",
    "EEGA-TD3": "#6B8E23",
    "Distance-TSP+TD3": BLIND_COLOR,
    "SA-D3QN": "#A05195",
    "DRL-SA": "#C44E52",
    "D3QN": D3QN_COLOR,
    "PPO": PPO_COLOR,
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7.5,
            "legend.fontsize": 6.7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_qos_contract(config: dict, source: Path, budget: float | None = None) -> None:
    schema = str(config.get("state_schema", ""))
    if not schema.startswith("v4-max-contiguous"):
        raise ValueError(f"{source}: expected a v4-max-contiguous state schema, got {schema!r}")
    if config.get("qos_constraint") != "max_contiguous_outage":
        raise ValueError(f"{source}: expected max_contiguous_outage QoS constraint")
    if budget is not None and float(config.get("outage_max_s")) != float(budget):
        raise ValueError(
            f"{source}: configured budget {config.get('outage_max_s')} does not equal {budget}"
        )


def load_run(run_dir: Path, label: str, budget: float | None = None) -> dict:
    """Load one run and separate its selected result from feasible evidence."""
    run_dir = Path(run_dir)
    config = _read_json(run_dir / "run_config.json")
    data = _read_json(run_dir / "eval_history.json")
    _validate_qos_contract(config, run_dir / "run_config.json", budget)
    history = data.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError(f"{run_dir / 'eval_history.json'}: missing non-empty evaluation history")
    selected = data.get("eval_best", data.get("best"))
    if not isinstance(selected, dict):
        raise ValueError(f"{run_dir / 'eval_history.json'}: missing eval_best/best selection")

    budget_s = float(config.get("outage_max_s", budget or 0))
    feasible = [
        row
        for row in history
        if bool(row.get("success"))
        and int(row.get("visited_inspection_points", 0)) == 6
        and float(row.get("outage_time_s", np.inf)) <= budget_s
    ]
    w1 = float(config.get("objective_w1", 0.5) or 0.5)
    w2 = float(config.get("objective_w2", 0.5) or 0.5)

    def selection_key(row: dict) -> tuple[float, float]:
        return (
            w1 * float(row["task_time_s"]) + w2 * float(row["transmission_delay_s"]),
            float(row["task_time_s"]) + float(row["transmission_delay_s"]),
        )

    selected_is_feasible = any(row.get("episode") == selected.get("episode") for row in feasible)
    best = selected if selected_is_feasible else (
        min(feasible, key=selection_key) if feasible else None
    )
    record = {
        "label": label,
        "source": str(run_dir.relative_to(ROOT)),
        "algorithm": config.get("algorithm"),
        "budget_s": float(config.get("outage_max_s", budget or 0)),
        "evaluations": len(history),
        "feasible_evaluations": len(feasible),
        "feasible_rate": len(feasible) / len(history),
        "best_feasible": None,
        "selected_evaluation": {
            key: selected.get(key)
            for key in (
                "episode",
                "success",
                "termination_reason",
                "task_time_s",
                "transmission_delay_s",
                "outage_time_s",
                "visited_inspection_points",
            )
        },
    }
    if best is not None:
        record["best_feasible"] = {
            "episode": int(best["episode"]),
            "task_time_s": float(best["task_time_s"]),
            "transmission_delay_s": float(best["transmission_delay_s"]),
            "objective_s": float(best["task_time_s"]) + float(best["transmission_delay_s"]),
            "outage_s": float(best["outage_time_s"]),
            "visited_inspection_points": int(best["visited_inspection_points"]),
        }
    return record


def load_mapguide_preflight(level: str) -> dict:
    source = ROOT / "journal_results" / "v4-deterministic" / "mapguide-preflight-6" / level / "preflight.json"
    data = _read_json(source)
    if not data.get("success"):
        raise ValueError(f"{source}: MapGuide preflight is not successful")
    delay_s = -100.0 * float(data["reward_terms"]["delay"])
    task_time_s = float(data["time_s"])
    return {
        "level": level,
        "task_time_s": task_time_s,
        "transmission_delay_s": delay_s,
        "objective_s": task_time_s + delay_s,
        "outage_s": float(data["outage_s"]),
        "source": str(source.relative_to(ROOT)),
    }


def load_astar_main6(level: str) -> dict:
    source = ROOT / "journal_results" / "v4-deterministic" / "astar-main-6" / level / "metrics.json"
    data = _read_json(source)
    if data.get("qos_constraint") != "max_contiguous_outage":
        raise ValueError(f"{source}: unexpected QoS constraint")
    planner = data["planner"]
    replay = data["world_replay"]
    if not planner.get("success") or not replay.get("success"):
        raise ValueError(f"{source}: A* result is not a successful replay")
    if float(replay["outage_time_s"]) > 6.0:
        raise ValueError(f"{source}: A* replay violates the 6 s constraint")
    return {
        "level": level,
        "task_time_s": float(replay["task_time_s"]),
        "transmission_delay_s": float(replay["transmission_delay_s"]),
        "objective_s": float(replay["objective_s"]),
        "outage_s": float(replay["outage_time_s"]),
        "planner_optimality_proven": bool(planner["optimality_proven"]),
        "source": str(source.relative_to(ROOT)),
    }


def build_dataset() -> dict:
    qos = []
    qos_lookup = {}
    for weight in QOS_WEIGHTS:
        for budget in QOS_BUDGETS:
            run_dir = (
                ROOT
                / "journal_results"
                / "v4-qos"
                / weight["id"]
                / f"qos{budget}"
                / "A"
                / "low"
                / "td3_bc"
            )
            record = load_run(run_dir, f"TD3 {weight['long_label']}", budget)
            record.update({"weight_id": weight["id"], "w1": weight["w1"], "w2": weight["w2"]})
            qos.append(record)
            qos_lookup[(weight["id"], budget)] = record

    missing_best = [row["label"] + f" qos{row['budget_s']:g}" for row in qos if row["best_feasible"] is None]
    if missing_best:
        raise ValueError("QoS runs without a feasible eval_best: " + ", ".join(missing_best))

    layer1 = [
        qos_lookup[("w50-50", 6)],
        load_run(
            ROOT / "journal_results" / "v4-models" / "layer1" / "d3qn" / "low" / "d3qn",
            "D3QN",
            6,
        ),
        load_run(
            ROOT / "journal_results" / "v4-models" / "ppo" / "A" / "low" / "ppo",
            "PPO",
            6,
        ),
    ]

    layer3 = [
        qos_lookup[("w50-50", 6)],
        load_run(
            ROOT / "journal_results" / "v4-baselines" / "yang" / "A" / "low" / "td3_bc",
            "Yang-style+TD3",
            6,
        ),
        load_run(
            ROOT / "journal_results" / "v4-baselines" / "eega" / "A" / "low" / "td3_bc",
            "EEGA-TD3",
            6,
        ),
        load_run(
            ROOT / "journal_results" / "v4-baselines" / "distance-tsp" / "A" / "low" / "td3_bc",
            "Distance-TSP+TD3",
            6,
        ),
        load_run(
            ROOT / "journal_results" / "v4-baselines" / "sa-d3qn" / "low" / "d3qn",
            "SA-D3QN",
            6,
        ),
        load_run(
            ROOT / "journal_results" / "v4-baselines" / "drl-sa" / "low" / "d3qn",
            "DRL-SA",
            6,
        ),
    ]

    load_paths = (
        ("Low", qos_lookup[("w50-50", 6)]),
        (
            "Medium",
            load_run(
                ROOT
                / "journal_results"
                / "v4-models"
                / "load"
                / "medium"
                / "A"
                / "medium"
                / "td3_bc",
                "Proposed (IGA+TD3)",
                6,
            ),
        ),
        (
            "High",
            load_run(
                ROOT
                / "journal_results"
                / "v4-models"
                / "load"
                / "high"
                / "A"
                / "high"
                / "td3_bc",
                "Proposed (IGA+TD3)",
                6,
            ),
        ),
    )

    ablation = [
        load_run(
            ROOT
            / "journal_results"
            / "v4-baselines"
            / "distance-tsp"
            / "A"
            / "low"
            / "td3_bc",
            "A2G-blind",
            6,
        ),
        load_run(
            ROOT
            / "journal_results"
            / "v4-baselines"
            / "distance-tsp-a2g-visible"
            / "A"
            / "low"
            / "td3_bc",
            "A2G-visible",
            6,
        ),
    ]

    return {
        "protocol": {
            "qos_constraint": "max_contiguous_outage",
            "state_schema_prefix": "v4-max-contiguous",
            "qos_budgets_s": list(QOS_BUDGETS),
            "main_budget_s": 6,
            "seed": 1,
            "feasible_rate_definition": "successful deterministic evaluations / periodic deterministic evaluations",
            "objective_definition": "T + transmission_delay_s",
            "uncertainty": "not estimated: one training seed per reported run",
        },
        "qos": qos,
        "weights_at_qos6": [qos_lookup[(weight["id"], 6)] for weight in QOS_WEIGHTS],
        "load": {
            "proposed": [{"level": level, **record} for level, record in load_paths],
            "mapguide": [load_mapguide_preflight(level) for level in ("low", "medium", "high")],
            "astar": [load_astar_main6(level) for level in ("low", "medium", "high")],
        },
        "layer1": layer1,
        "layer3": layer3,
        "a2g_ablation": ablation,
    }


def _panel_label(axis, label: str, x: float = -0.12) -> None:
    axis.text(
        x,
        1.04,
        label,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        fontweight="bold",
    )


def _objective(record: dict) -> float:
    if record.get("best_feasible") is None:
        raise ValueError(f"{record['label']} has no feasible objective")
    return float(record["best_feasible"]["objective_s"])


def _task_time(record: dict) -> float:
    if record.get("best_feasible") is None:
        raise ValueError(f"{record['label']} has no feasible task time")
    return float(record["best_feasible"]["task_time_s"])


def _delay(record: dict) -> float:
    if record.get("best_feasible") is None:
        raise ValueError(f"{record['label']} has no feasible transmission delay")
    return float(record["best_feasible"]["transmission_delay_s"])


def _outage(record: dict) -> float:
    if record.get("best_feasible") is None:
        return float(record["selected_evaluation"]["outage_time_s"])
    return float(record["best_feasible"]["outage_s"])


def _method_label(label: str) -> str:
    if label.startswith("TD3 (0.5, 0.5)"):
        return "Proposed (IGA+TD3)"
    return label


def _save_figure(figure, stem: Path, multi_panel: bool) -> None:
    if require_matplotlib_panel_alignment is None:
        raise RuntimeError(
            "Panel-alignment helper unavailable; add nature-figure/scripts to PYTHONPATH"
        )
    figure.canvas.draw()
    require_matplotlib_panel_alignment(
        figure,
        json_out=stem.with_suffix(".alignment.json"),
        overlay_svg=stem.with_suffix(".alignment.svg"),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=multi_panel,
        strict=True,
    )
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches=None, metadata={"Title": stem.name})
    figure.savefig(stem.with_suffix(".svg"), bbox_inches=None, metadata={"Title": stem.name})
    figure.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches=None)
    figure.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches=None)
    plt.close(figure)


def plot_qos_sensitivity(dataset: dict, output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(WIDTH_DOUBLE_IN, 2.8))
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.22, top=0.84, wspace=0.36)
    for weight in QOS_WEIGHTS:
        rows = [row for row in dataset["qos"] if row["weight_id"] == weight["id"]]
        rows.sort(key=lambda row: row["budget_s"])
        budgets = [row["budget_s"] for row in rows]
        axes[0].plot(
            budgets,
            [_objective(row) for row in rows],
            color=weight["color"],
            marker=weight["marker"],
            markersize=3.8,
            linewidth=1.4,
            label=weight["long_label"],
        )
        axes[1].plot(
            budgets,
            [_outage(row) for row in rows],
            color=weight["color"],
            marker=weight["marker"],
            markersize=3.8,
            linewidth=1.4,
        )
        axes[2].plot(
            budgets,
            100.0 * np.asarray([row["feasible_rate"] for row in rows]),
            color=weight["color"],
            marker=weight["marker"],
            markersize=3.8,
            linewidth=1.4,
        )
    for index, axis in enumerate(axes):
        _panel_label(axis, "abc"[index])
        axis.set_xticks(QOS_BUDGETS)
        axis.set_xlabel("Longest-outage budget (s)")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.55)
        axis.axvline(6, color="#7F7F7F", linestyle=(0, (2, 2)), linewidth=0.8)
    axes[0].set_ylabel("Mission objective T + τ (s)")
    axes[0].set_ylim(180, 265)
    axes[0].set_title("Objective")
    axes[1].plot(QOS_BUDGETS, QOS_BUDGETS, color=REFERENCE_COLOR, linewidth=0.9, linestyle=":")
    axes[1].set_ylabel("Longest continuous outage (s)")
    axes[1].set_ylim(0, 8.7)
    axes[1].set_title("Constraint use")
    axes[2].set_ylabel("Feasible evaluations (%)")
    axes[2].set_ylim(0, 100)
    axes[2].set_title("Selection reliability")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=3)
    _save_figure(figure, output / "qos_sensitivity_v4", multi_panel=True)


def plot_weight_sensitivity(dataset: dict, output: Path) -> None:
    rows = dataset["weights_at_qos6"]
    labels = [next(w["label"] for w in QOS_WEIGHTS if w["id"] == row["weight_id"]) for row in rows]
    x = np.arange(len(rows))
    figure, axes = plt.subplots(1, 3, figsize=(WIDTH_DOUBLE_IN, 2.65))
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.23, top=0.84, wspace=0.36)
    colors = [next(w["color"] for w in QOS_WEIGHTS if w["id"] == row["weight_id"]) for row in rows]
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.set_xlabel("w1 / w2")
    axes[0].bar(x, [_task_time(row) for row in rows], color=colors, width=0.62)
    axes[0].set_ylabel("Mission time T (s)")
    axes[0].set_ylim(0, 190)
    axes[0].set_title("Flight time")
    axes[1].bar(x, [_delay(row) for row in rows], color=colors, width=0.62)
    axes[1].set_ylabel("Transmission delay τ (s)")
    axes[1].set_ylim(0, 100)
    axes[1].set_title("Uplink delay")
    axes[2].bar(x, [100.0 * row["feasible_rate"] for row in rows], color=colors, width=0.62)
    axes[2].set_ylabel("Feasible evaluations (%)")
    axes[2].set_ylim(0, 100)
    axes[2].set_title("Feasibility")
    for index, row in enumerate(rows):
        axes[2].text(
            index,
            100.0 * row["feasible_rate"] - 7,
            f"{row['feasible_evaluations']}/{row['evaluations']}",
            ha="center",
            va="top",
            color="white",
        )
    for index, axis in enumerate(axes):
        _panel_label(axis, "abc"[index])
    _save_figure(figure, output / "objective_weight_sensitivity_v4", multi_panel=True)


def plot_load_sensitivity(dataset: dict, output: Path) -> None:
    proposed = dataset["load"]["proposed"]
    mapguide = dataset["load"]["mapguide"]
    astar = dataset["load"]["astar"]
    levels = [row["level"] for row in proposed]
    x = np.arange(len(levels))
    figure, axes = plt.subplots(1, 2, figsize=(WIDTH_DOUBLE_IN, 2.65))
    figure.subplots_adjust(left=0.08, right=0.985, bottom=0.31, top=0.83, wspace=0.34)

    times = np.asarray([_task_time(row) for row in proposed])
    delays = np.asarray([_delay(row) for row in proposed])
    axes[0].bar(x, times, width=0.58, color=PROPOSED_COLOR, label="T")
    axes[0].bar(x, delays, width=0.58, bottom=times, color="#79A9C1", label="τ")
    axes[0].scatter(
        x,
        [row["objective_s"] for row in mapguide],
        marker="_",
        s=95,
        linewidth=1.6,
        color=REFERENCE_COLOR,
        label="MapGuide",
        zorder=4,
    )
    axes[0].scatter(
        x,
        [row["objective_s"] for row in astar],
        marker="^",
        s=34,
        facecolors="white",
        edgecolors=REFERENCE_COLOR,
        linewidth=1.0,
        label="A*",
        zorder=4,
    )
    for index, row in enumerate(proposed):
        total = _objective(row)
        axes[0].annotate(
            f"{total:.1f}",
            xy=(index, total),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )
    axes[0].set_xticks(x, levels)
    axes[0].set_ylabel("Mission objective (s)")
    axes[0].set_ylim(0, 570)
    axes[0].set_title("Objective decomposition")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.08, 0.01), ncol=4)

    rates = [100.0 * row["feasible_rate"] for row in proposed]
    bars = axes[1].bar(x, rates, color=PROPOSED_COLOR, width=0.58)
    for bar, row in zip(bars, proposed):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() - 6,
            f"{row['feasible_evaluations']}/{row['evaluations']}\n"
            f"outage {_outage(row):.1f}s",
            ha="center",
            va="top",
            color="white",
        )
    axes[1].set_xticks(x, levels)
    axes[1].set_ylabel("Feasible evaluations (%)")
    axes[1].set_ylim(0, 100)
    axes[1].set_title("Reliability under the 6 s constraint")
    for index, axis in enumerate(axes):
        _panel_label(axis, "ab"[index])
    _save_figure(figure, output / "load_sensitivity_v4", multi_panel=True)


def plot_layer1(dataset: dict, output: Path) -> None:
    rows = dataset["layer1"]
    labels = [_method_label(row["label"]) for row in rows]
    x = np.arange(len(rows))
    colors = [PROPOSED_COLOR, D3QN_COLOR, PPO_COLOR]
    feasible = [row["best_feasible"] is not None for row in rows]
    objectives = [_objective(row) if has else np.nan for row, has in zip(rows, feasible)]

    figure, axes = plt.subplots(1, 3, figsize=(WIDTH_DOUBLE_IN, 2.65))
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.25, top=0.84, wspace=0.38)
    for index, row in enumerate(rows):
        if feasible[index]:
            axes[0].bar(index, objectives[index], color=colors[index], width=0.58)
            axes[0].text(
                index,
                objectives[index] - 20,
                f"{objectives[index]:.1f}",
                ha="center",
                va="top",
                color="white",
            )
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Feasible T + τ (s)")
    axes[0].set_ylim(0, 310)
    axes[0].set_title("Selected objective")

    width = 0.25
    for index, row in enumerate(rows):
        task = (_task_time(row) if feasible[index]
                else float(row["selected_evaluation"]["task_time_s"]))
        delay = (_delay(row) if feasible[index]
                 else float(row["selected_evaluation"]["transmission_delay_s"]))
        axes[1].bar(index - width / 2, task, width=width, color=colors[index])
        axes[1].bar(index + width / 2, delay, width=width, color=colors[index], alpha=0.48)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Seconds")
    axes[1].set_ylim(0, 290)
    axes[1].set_title("T (solid) and τ (light)")

    rates = [100.0 * row["feasible_rate"] for row in rows]
    axes[2].bar(x, rates, color=colors, width=0.58)
    for index, row in enumerate(rows):
        axes[2].text(index, rates[index] + 2, f"{row['feasible_evaluations']}/{row['evaluations']}",
                     ha="center", va="bottom")
    axes[2].set_xticks(x, labels)
    axes[2].set_ylabel("Feasible evaluations (%)")
    axes[2].set_ylim(0, 100)
    axes[2].set_title("Reliability")
    for index, axis in enumerate(axes):
        _panel_label(axis, "abc"[index])
    _save_figure(figure, output / "layer1_methods_v4", multi_panel=True)


def plot_layer3(dataset: dict, output: Path) -> None:
    rows = dataset["layer3"]
    labels = [_method_label(row["label"]) for row in rows]
    y = np.arange(len(rows))
    objectives = [_objective(row) for row in rows]
    rates = [100.0 * row["feasible_rate"] for row in rows]
    colors = [METHOD_COLORS.get(label, "#777777") for label in labels]
    figure, axes = plt.subplots(1, 2, figsize=(WIDTH_DOUBLE_IN, 2.85), sharey=True)
    figure.subplots_adjust(left=0.22, right=0.985, bottom=0.22, top=0.84, wspace=0.12)
    axes[0].barh(y, objectives, color=colors, height=0.62)
    for index, value in enumerate(objectives):
        axes[0].annotate(
            f"{value:.1f}",
            xy=(value, index),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            ha="left",
        )
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Feasible T + τ (s)")
    axes[0].set_xlim(0, 380)
    axes[0].set_title("Selected objective")
    axes[1].barh(y, rates, color=colors, height=0.62)
    for index, row in enumerate(rows):
        axes[1].annotate(
            f"{row['feasible_evaluations']}/{row['evaluations']}",
            xy=(rates[index], index),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            ha="left",
        )
    axes[1].set_xlabel("Feasible evaluations (%)")
    axes[1].set_xlim(0, 110)
    axes[1].set_title("Reliability")
    for index, axis in enumerate(axes):
        _panel_label(axis, "ab"[index])
    _save_figure(figure, output / "layer3_baselines_v4", multi_panel=True)


def plot_a2g_ablation(dataset: dict, output: Path) -> None:
    rows = dataset["a2g_ablation"]
    labels = [row["label"] for row in rows]
    x = np.arange(len(rows))
    times = [_task_time(row) for row in rows]
    delays = [_delay(row) for row in rows]
    colors = [BLIND_COLOR, VISIBLE_COLOR]
    figure, axis = plt.subplots(figsize=(WIDTH_SINGLE_IN, 2.65))
    figure.subplots_adjust(left=0.19, right=0.98, bottom=0.23, top=0.87)
    axis.bar(x, times, width=0.56, color=colors, label="T")
    axis.bar(x, delays, width=0.56, bottom=times, color=colors, alpha=0.45, label="τ")
    for index, row in enumerate(rows):
        axis.annotate(
            f"{_objective(row):.1f} s",
            xy=(index, _objective(row)),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )
        axis.text(index, times[index] / 2, f"T {times[index]:.1f}", ha="center",
                  va="center", color="white")
        axis.text(index, times[index] + delays[index] / 2,
                  f"τ {delays[index]:.1f}", ha="center", va="center", color="black")
    axis.set_xticks(x, labels)
    axis.set_ylabel("Seconds")
    axis.set_ylim(0, 330)
    _panel_label(axis, "a")
    _save_figure(figure, output / "a2g_ablation_v4", multi_panel=False)


def load_rewards(record: dict) -> tuple[np.ndarray, np.ndarray]:
    run_dir = ROOT / record["source"]
    if (run_dir / "UAV_TD3_improved.npz").is_file():
        archive = np.load(run_dir / "UAV_TD3_improved.npz")
        rewards = np.asarray(archive["ep_rewards"], dtype=float) * 100.0
        guided = np.asarray(archive["ep_guided"], dtype=bool)
        return rewards, guided
    archive = np.load(run_dir / "training_history.npz")
    rewards = np.asarray(archive["episode_rewards"], dtype=float) * 100.0
    guided = (np.asarray(archive["episode_guided"], dtype=bool)
              if "episode_guided" in archive
              else np.zeros(rewards.size, dtype=bool))
    return rewards, guided


def plot_training_returns(dataset: dict, output: Path) -> None:
    proposed = dataset["load"]["proposed"]
    network_rows = [
        ("Low", proposed[0], PROPOSED_COLOR),
        ("Medium", proposed[1], "#4C78A8"),
        ("High", proposed[2], "#8C6D31"),
        ("D3QN", dataset["layer1"][1], D3QN_COLOR),
        ("PPO", dataset["layer1"][2], PPO_COLOR),
    ]
    figure, axis = plt.subplots(figsize=(WIDTH_DOUBLE_IN, 2.85))
    figure.subplots_adjust(left=0.085, right=0.985, bottom=0.22, top=0.86)
    for label, record, color in network_rows:
        rewards, _ = load_rewards(record)
        episodes = np.arange(1, rewards.size + 1)
        axis.plot(episodes, rewards, color=color, linewidth=0.35, alpha=0.16, rasterized=True)
        if rewards.size >= 50:
            averages = np.convolve(rewards, np.ones(50) / 50.0, mode="valid")
            axis.plot(episodes[49:], averages, color=color, linewidth=1.4, label=label)
        else:
            axis.plot([], [], color=color, label=label)
    axis.axhline(0, color="#666666", linewidth=0.6)
    axis.set_xlim(0, 2000)
    axis.set_xlabel("Training episode")
    axis.set_ylabel("Episode return (objective seconds)")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.55)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, 1.09), ncol=5)
    _panel_label(axis, "a", x=0.0)
    _save_figure(figure, output / "training_returns_v4", multi_panel=False)


def write_source_data(dataset: dict, output: Path) -> None:
    (output / "figure_source_data_v4.json").write_text(
        json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    rows = []
    for group in ("qos", "layer1", "layer3", "a2g_ablation"):
        for record in dataset[group]:
            best = record.get("best_feasible")
            rows.append(
                {
                    "group": group,
                    "label": record["label"],
                    "budget_s": record["budget_s"],
                    "evaluations": record["evaluations"],
                    "feasible_evaluations": record["feasible_evaluations"],
                    "feasible_rate": record["feasible_rate"],
                    "best_episode": best["episode"] if best else "",
                    "task_time_s": best["task_time_s"] if best else "",
                    "transmission_delay_s": best["transmission_delay_s"] if best else "",
                    "objective_s": best["objective_s"] if best else "",
                    "outage_s": best["outage_s"] if best else _outage(record),
                    "source": record["source"],
                }
            )
    for record in dataset["load"]["proposed"]:
        best = record["best_feasible"]
        rows.append(
            {
                "group": "load",
                "label": record["level"],
                "budget_s": record["budget_s"],
                "evaluations": record["evaluations"],
                "feasible_evaluations": record["feasible_evaluations"],
                "feasible_rate": record["feasible_rate"],
                "best_episode": best["episode"] if best else "",
                "task_time_s": best["task_time_s"] if best else "",
                "transmission_delay_s": best["transmission_delay_s"] if best else "",
                "objective_s": best["objective_s"] if best else "",
                "outage_s": best["outage_s"] if best else "",
                "source": record["source"],
            }
        )
    for record in dataset["load"]["astar"]:
        rows.append(
            {
                "group": "astar",
                "label": record["level"],
                "budget_s": 6,
                "evaluations": 1,
                "feasible_evaluations": 1,
                "feasible_rate": 1.0,
                "best_episode": "",
                "task_time_s": record["task_time_s"],
                "transmission_delay_s": record["transmission_delay_s"],
                "objective_s": record["objective_s"],
                "outage_s": record["outage_s"],
                "source": record["source"],
            }
        )
    fieldnames = list(rows[0])
    with (output / "figure_source_data_v4.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_figures(dataset: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    write_source_data(dataset, output)
    plot_qos_sensitivity(dataset, output)
    plot_weight_sensitivity(dataset, output)
    plot_load_sensitivity(dataset, output)
    plot_layer1(dataset, output)
    plot_layer3(dataset, output)
    plot_a2g_ablation(dataset, output)
    plot_training_returns(dataset, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    _style()
    dataset = build_dataset()
    build_figures(dataset, args.output)
    print(f"Wrote v4 figures and source data to {args.output}")


if __name__ == "__main__":
    main()
