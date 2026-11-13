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
        "color": "#B56B45",
        "marker": "o",
    },
    {
        "id": "w50-50",
        "w1": 0.5,
        "w2": 0.5,
        "label": "50/50",
        "long_label": "(0.5, 0.5)",
        "color": "#236D86",
        "marker": "s",
    },
    {
        "id": "w20-80",
        "w1": 0.2,
        "w2": 0.8,
        "label": "20/80",
        "long_label": "(0.2, 0.8)",
        "color": "#7A6F9B",
        "marker": "^",
    },
)

PROPOSED_COLOR = "#236D86"
PROPOSED_LIGHT = "#A6CAD4"
PPO_COLOR = "#7A6F9B"
D3QN_COLOR = "#648875"
PURE_ASTAR_COLOR = "#9A7B56"
PURE_PSO_COLOR = "#8A6F88"
BLIND_COLOR = "#9A7B56"
VISIBLE_COLOR = "#236D86"
REFERENCE_COLOR = "#686868"
FAILURE_COLOR = "#B14B3D"
GRID_COLOR = "#E1E4E7"
METHOD_COLORS = {
    "DRM-HO": PROPOSED_COLOR,
    "A*": PURE_ASTAR_COLOR,
    "PSO": PURE_PSO_COLOR,
    "Yang-style+TD3": "#8B7657",
    "EEGA-TD3": "#708A68",
    "Distance-TSP+TD3": BLIND_COLOR,
    "SA-D3QN": "#8A6F88",
    "DRL-SA": "#A9655D",
    "D3QN": D3QN_COLOR,
    "PPO": PPO_COLOR,
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "font.size": 7.2,
            "axes.labelsize": 7.2,
            "axes.titlesize": 7.4,
            "legend.fontsize": 6.7,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "legend.frameon": False,
            "lines.solid_capstyle": "round",
            "patch.linewidth": 0.0,
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


def load_pure_heuristic(level: str, method: str) -> dict:
    slugs = {"A*": "astar", "PSO": "pso"}
    if method not in slugs:
        raise ValueError(f"unknown pure heuristic baseline: {method}")
    source = (
        ROOT
        / "journal_results"
        / "v4-deterministic"
        / "pure-heuristics-6"
        / slugs[method]
        / level
        / "metrics.json"
    )
    data = _read_json(source)
    if float(data["outage_limit_s"]) != 6.0:
        raise ValueError(f"{source}: expected the 6 s main outage budget")
    if data.get("qos_constraint") != "max_contiguous_outage":
        raise ValueError(f"{source}: expected max_contiguous_outage QoS constraint")
    success = bool(data.get("success"))
    record = {
        "label": method,
        "source": str(source.relative_to(ROOT)),
        "algorithm": data.get("algorithm"),
        "budget_s": 6.0,
        "evaluations": 1,
        "feasible_evaluations": int(success),
        "feasible_rate": float(success),
        "best_feasible": None,
        "selected_evaluation": {
            "episode": None,
            "success": success,
            "termination_reason": data.get("termination_reason"),
            "task_time_s": data.get("task_time_s"),
            "transmission_delay_s": data.get("transmission_delay_s"),
            "outage_time_s": data.get("outage_time_s"),
            "visited_inspection_points": data.get("visited_inspection_points"),
        },
        "level": level,
        "deterministic": True,
    }
    if success:
        record["best_feasible"] = {
            "episode": None,
            "task_time_s": float(data["task_time_s"]),
            "transmission_delay_s": float(data["transmission_delay_s"]),
            "objective_s": float(data["task_time_s"]) + float(data["transmission_delay_s"]),
            "outage_s": float(data["outage_time_s"]),
            "visited_inspection_points": int(data["visited_inspection_points"]),
        }
    return record


def load_mco_astar(level: str) -> dict:
    source = (
        ROOT
        / "journal_results"
        / "v4-deterministic"
        / "mco-astar-6"
        / level
        / "metrics.json"
    )
    data = _read_json(source)
    if float(data.get("budget_s", -1)) != 6.0:
        raise ValueError(f"{source}: expected the 6 s main outage budget")
    success = bool(data.get("success"))
    return {
        "label": "MCO-A* (adapted)",
        "level": level,
        "source": str(source.relative_to(ROOT)),
        "algorithm": "mco_astar",
        "budget_s": 6.0,
        "evaluations": 1,
        "feasible_evaluations": int(success),
        "feasible_rate": float(success),
        "deterministic": True,
        "best_feasible": {
            "episode": None,
            "task_time_s": float(data["task_time_s"]),
            "transmission_delay_s": float(data["transmission_delay_s"]),
            "objective_s": float(data["task_time_s"]) + float(data["transmission_delay_s"]),
            "outage_s": float(data["outage_s"]),
            "visited_inspection_points": int(data["visited_inspection_points"]),
        } if success else None,
        "selected_evaluation": {
            "episode": None,
            "success": success,
            "termination_reason": data.get("termination_reason"),
            "task_time_s": data.get("task_time_s"),
            "transmission_delay_s": data.get("transmission_delay_s"),
            "outage_time_s": data.get("outage_s"),
            "visited_inspection_points": data.get("visited_inspection_points"),
            "uploaded_mb": data.get("uploaded_mb"),
        },
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
            record = load_run(run_dir, f"CA-TD3 {weight['long_label']}", budget)
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
        load_pure_heuristic("low", "A*"),
        load_pure_heuristic("low", "PSO"),
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
                "DRM-HO",
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
                "DRM-HO",
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
            "mco_astar": [load_mco_astar(level) for level in ("low", "medium", "high")],
            "pure_heuristics": [
                load_pure_heuristic(level, method)
                for level in ("low", "medium", "high")
                for method in ("A*", "PSO")
            ],
        },
        "layer1": layer1,
        "layer3": layer3,
        "a2g_ablation": ablation,
    }


def _panel_label(axis, label: str) -> None:
    axis.annotate(
        label,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(-10, 6),
        textcoords="offset points",
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
    if label.startswith("CA-TD3 (0.5, 0.5)"):
        return "DRM-HO"
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
    svg_path = stem.with_suffix(".svg")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches=None, metadata={"Title": stem.name})
    figure.savefig(svg_path, bbox_inches=None, metadata={"Title": stem.name})
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    figure.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches=None)
    figure.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches=None)
    plt.close(figure)


def plot_qos_sensitivity(dataset: dict, output: Path) -> None:
    figure = plt.figure(figsize=(WIDTH_DOUBLE_IN, 3.15))
    grid = figure.add_gridspec(2, 2, height_ratios=(1.28, 1.0))
    axes = [
        figure.add_subplot(grid[0, :]),
        figure.add_subplot(grid[1, 0]),
        figure.add_subplot(grid[1, 1]),
    ]
    figure.subplots_adjust(left=0.085, right=0.985, bottom=0.16, top=0.82, hspace=0.62, wspace=0.34)

    for weight in QOS_WEIGHTS:
        rows = sorted(
            (row for row in dataset["qos"] if row["weight_id"] == weight["id"]),
            key=lambda row: row["budget_s"],
        )
        budgets = np.asarray([row["budget_s"] for row in rows])
        common = {
            "color": weight["color"],
            "marker": weight["marker"],
            "markersize": 3.7,
            "linewidth": 1.35,
            "markeredgecolor": "white",
            "markeredgewidth": 0.45,
        }
        axes[0].plot(
            budgets,
            [_objective(row) for row in rows],
            label=weight["long_label"],
            **common,
        )
        axes[1].plot(
            budgets,
            budgets - np.asarray([_outage(row) for row in rows]),
            **common,
        )
        axes[2].plot(
            budgets,
            100.0 * np.asarray([row["feasible_rate"] for row in rows]),
            **common,
        )

    for axis in axes:
        axis.set_xticks(QOS_BUDGETS)
        axis.set_xlim(2.7, 8.3)
        axis.grid(axis="y", color=GRID_COLOR, linewidth=0.55)
        axis.axvline(6, color=REFERENCE_COLOR, linestyle=(0, (2, 2)), linewidth=0.8)
    for axis in axes[1:]:
        axis.set_xlabel("Longest-outage budget (s)")

    axes[0].set_ylabel("Mission objective, T + τ (s)")
    axes[0].set_ylim(238, 258)
    axes[1].axhline(0, color=REFERENCE_COLOR, linewidth=0.8)
    axes[1].set_ylabel("QoS headroom (s)")
    axes[1].set_ylim(-0.2, 5.8)
    axes[1].set_xlabel("Outage budget (s)")
    axes[2].set_ylabel("Feasible evaluations (%)")
    axes[2].set_ylim(0, 100)

    for index, axis in enumerate(axes):
        _panel_label(axis, "abc"[index])
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=3,
        title="Objective weights",
        title_fontsize=6.7,
    )
    _save_figure(figure, output / "qos_sensitivity_v4", multi_panel=True)


def plot_weight_sensitivity(dataset: dict, output: Path) -> None:
    rows = dataset["weights_at_qos6"]
    figure, axis = plt.subplots(figsize=(WIDTH_SINGLE_IN, 2.25))
    figure.subplots_adjust(left=0.19, right=0.97, bottom=0.22, top=0.96)

    x_values = np.asarray([_task_time(row) for row in rows])
    y_values = np.asarray([_delay(row) for row in rows])
    objectives = x_values + y_values
    for level in (245, 250, 255, 260):
        xs = np.asarray([145.0, 180.0])
        axis.plot(xs, level - xs, color=GRID_COLOR, linewidth=0.7, linestyle=(0, (2, 2)), zorder=0)

    for row, x_value, y_value, objective in zip(rows, x_values, y_values, objectives):
        style = next(weight for weight in QOS_WEIGHTS if weight["id"] == row["weight_id"])
        axis.scatter(
            x_value,
            y_value,
            s=42,
            color=style["color"],
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        axis.annotate(
            f"{style['long_label']}\n{objective:.1f} s",
            xy=(x_value, y_value),
            xytext=(5, 5),
            textcoords="offset points",
            ha="left",
            va="bottom",
            color=style["color"],
        )

    axis.set_xlim(145, 180)
    axis.set_ylim(70, 93)
    axis.set_xticks([150, 160, 170, 180])
    axis.set_yticks([70, 75, 80, 85, 90])
    axis.set_xlabel("Mission time T (s)")
    axis.set_ylabel("Transmission delay τ (s)")
    axis.grid(color=GRID_COLOR, linewidth=0.55)
    _save_figure(figure, output / "objective_weight_sensitivity_v4", multi_panel=False)


def plot_load_sensitivity(dataset: dict, output: Path) -> None:
    proposed = dataset["load"]["proposed"]
    load_totals = {"Low": 450, "Medium": 900, "High": 1500}
    levels = [f"{row['level']}\n{load_totals[row['level']]} MB" for row in proposed]
    x = np.arange(len(levels))
    figure, axes = plt.subplots(1, 2, figsize=(WIDTH_DOUBLE_IN, 2.7))
    figure.subplots_adjust(left=0.085, right=0.985, bottom=0.29, top=0.86, wspace=0.32)

    times = np.asarray([_task_time(row) for row in proposed])
    delays = np.asarray([_delay(row) for row in proposed])
    axes[0].bar(x, times, width=0.56, color=PROPOSED_COLOR, label="Flight time T")
    axes[0].bar(x, delays, width=0.56, bottom=times, color=PROPOSED_LIGHT, label="Transmission delay τ")
    for index, row in enumerate(proposed):
        total = _objective(row)
        axes[0].annotate(
            f"{total:.1f}",
            xy=(index, total),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    axes[0].set_xticks(x, levels)
    axes[0].set_ylabel("Mission objective (s)")
    axes[0].set_ylim(0, 570)

    rates = [100.0 * row["feasible_rate"] for row in proposed]
    bars = axes[1].bar(x, rates, color=PROPOSED_COLOR, width=0.56, alpha=0.9)
    for bar, row in zip(bars, proposed):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 3,
            f"{row['feasible_evaluations']}/{row['evaluations']}\n"
            f"outage {_outage(row):.1f}s",
            ha="center",
            va="bottom",
            color=PROPOSED_COLOR,
        )
    axes[1].set_xticks(x, levels)
    axes[1].set_ylabel("Feasible evaluations (%)")
    axes[1].set_ylim(0, 115)
    for index, axis in enumerate(axes):
        _panel_label(axis, "ab"[index])
    _save_figure(figure, output / "load_sensitivity_v4", multi_panel=True)


def plot_layer1(dataset: dict, output: Path) -> None:
    rows = dataset["layer1"]
    labels = [_method_label(row["label"]) for row in rows]
    colors = [METHOD_COLORS.get(label, "#777777") for label in labels]
    feasible = [row["best_feasible"] is not None for row in rows]
    objectives = [_objective(row) if has else np.nan for row, has in zip(rows, feasible)]
    y = np.arange(len(rows))

    figure, axes = plt.subplots(1, 2, figsize=(WIDTH_DOUBLE_IN, 2.65), sharey=True)
    figure.subplots_adjust(left=0.2, right=0.985, bottom=0.2, top=0.89, wspace=0.18)

    for index, (label, color, has, objective) in enumerate(zip(labels, colors, feasible, objectives)):
        if has:
            axes[0].hlines(index, 0, objective, color=color, linewidth=1.4, alpha=0.45, zorder=1)
            axes[0].scatter(objective, index, s=38, color=color, edgecolor="white",
                            linewidth=0.7, zorder=3)
            axes[0].annotate(
                f"{objective:.1f}",
                xy=(objective, index),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                color=color,
            )
        else:
            axes[0].scatter(4, index, marker="x", s=30, linewidth=1.5,
                            color=FAILURE_COLOR, zorder=3)
            axes[0].text(12, index, "No feasible policy", va="center",
                         color=FAILURE_COLOR)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 390)
    axes[0].set_xlabel("Feasible objective, T + τ (s)")

    rates = np.asarray([100.0 * row["feasible_rate"] for row in rows])
    for index, (color, rate, row) in enumerate(zip(colors, rates, rows)):
        axes[1].hlines(index, 0, rate, color=color, linewidth=1.4, alpha=0.45, zorder=1)
        axes[1].scatter(rate, index, s=38, color=color, edgecolor="white",
                        linewidth=0.7, zorder=3)
        denominator = row["evaluations"]
        if rate == 0:
            annotation = (
                f"0/{denominator}, {_outage(row):.1f} s"
            )
        else:
            annotation = (
                f"{row['feasible_evaluations']}/{denominator}\n"
                f"outage {_outage(row):.1f} s"
            )
        axes[1].annotate(
            annotation,
            xy=(rate, index),
            xytext=(16 if rate == 0 else 6, 0),
            textcoords="offset points",
            va="center",
            color=color if rate > 0 else FAILURE_COLOR,
        )
    axes[1].set_xlim(0, 132)
    axes[1].set_xlabel("Feasible evaluations (%)")
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
    figure, axes = plt.subplots(1, 2, figsize=(WIDTH_DOUBLE_IN, 2.7), sharey=True)
    figure.subplots_adjust(left=0.23, right=0.985, bottom=0.2, top=0.89, wspace=0.12)
    for index, value in enumerate(objectives):
        axes[0].hlines(index, 0, value, color=colors[index], linewidth=1.35,
                       alpha=0.42, zorder=1)
        axes[0].scatter(value, index, s=38, color=colors[index],
                        edgecolor="white", linewidth=0.7, zorder=3)
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
    axes[0].set_xlabel("Feasible objective, T + τ (s)")
    axes[0].set_xlim(0, 390)
    axes[0].axvline(objectives[0], color=REFERENCE_COLOR, linewidth=0.75,
                    linestyle=(0, (2, 2)), zorder=0)
    for index, row in enumerate(rows):
        axes[1].hlines(index, 0, rates[index], color=colors[index],
                       linewidth=1.35, alpha=0.42, zorder=1)
        axes[1].scatter(rates[index], index, s=38, color=colors[index],
                        edgecolor="white", linewidth=0.7, zorder=3)
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
    for index, axis in enumerate(axes):
        _panel_label(axis, "ab"[index])
    _save_figure(figure, output / "layer3_baselines_v4", multi_panel=True)


def plot_a2g_ablation(dataset: dict, output: Path) -> None:
    rows = dataset["a2g_ablation"]
    blind, visible = rows
    metric_labels = ["T", "τ", "T + τ"]
    blind_values = np.asarray([_task_time(blind), _delay(blind), _objective(blind)])
    visible_values = np.asarray([_task_time(visible), _delay(visible), _objective(visible)])
    y = np.arange(len(metric_labels))

    figure, axis = plt.subplots(figsize=(WIDTH_SINGLE_IN, 2.2))
    figure.subplots_adjust(left=0.2, right=0.97, bottom=0.2, top=0.72)
    height = 0.32
    visible_y = y - height / 2
    blind_y = y + height / 2
    axis.barh(visible_y, visible_values, height=height, color=VISIBLE_COLOR,
              label="A2G-visible")
    axis.barh(blind_y, blind_values, height=height, color=BLIND_COLOR,
              label="A2G-blind")
    for values, positions, color in (
        (visible_values, visible_y, VISIBLE_COLOR),
        (blind_values, blind_y, BLIND_COLOR),
    ):
        for value, position in zip(values, positions):
            axis.annotate(
                f"{value:.1f}",
                xy=(value, position),
                xytext=(5, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                color=color,
            )
    axis.set_yticks(y, metric_labels)
    axis.invert_yaxis()
    axis.set_xlim(0, 330)
    axis.set_xlabel("Seconds")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2)
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


def load_eval_series(record: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    history = _read_json(ROOT / record["source"] / "eval_history.json")["history"]
    episodes = np.asarray([row["episode"] for row in history], dtype=int)
    success = np.asarray([bool(row["success"]) for row in history], dtype=float)
    objective = np.asarray(
        [
            float(row["task_time_s"]) + float(row["transmission_delay_s"])
            if row["success"] else np.nan
            for row in history
        ],
        dtype=float,
    )
    return episodes, success, objective


def plot_training_returns(dataset: dict, output: Path) -> None:
    proposed = dataset["load"]["proposed"]
    d3qn = next(row for row in dataset["layer1"] if row["label"] == "D3QN")
    ppo = next(row for row in dataset["layer1"] if row["label"] == "PPO")
    network_rows = [
        ("Low", proposed[0], PROPOSED_COLOR),
        ("Medium", proposed[1], "#4C78A8"),
        ("High", proposed[2], "#8C6D31"),
        ("D3QN", d3qn, D3QN_COLOR),
        ("PPO", ppo, PPO_COLOR),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(WIDTH_DOUBLE_IN, 2.8), sharex=True)
    figure.subplots_adjust(left=0.085, right=0.985, bottom=0.2, top=0.8, wspace=0.3)
    for label, record, color in network_rows:
        rewards, guided = load_rewards(record)
        training_episodes = np.arange(1, rewards.size + 1)
        axes[0].plot(
            training_episodes,
            rewards,
            color=color,
            linewidth=0.3,
            alpha=0.1,
            rasterized=True,
            label="_nolegend_",
        )
        reward_window = min(50, rewards.size)
        reward_average = np.convolve(
            rewards, np.ones(reward_window) / reward_window, mode="valid"
        )
        axes[0].plot(
            training_episodes[reward_window - 1:],
            reward_average,
            color=color,
            linewidth=1.35,
            label=label,
        )

        episodes, success, objective = load_eval_series(record)
        window = 10
        rolling_count = np.convolve(success, np.ones(window), mode="full")[: success.size]
        rolling_rate = rolling_count / np.minimum(np.arange(1, success.size + 1), window)
        axes[1].plot(
            episodes,
            100.0 * success,
            color=color,
            marker="o",
            markersize=1.5,
            alpha=0.28,
            linestyle="none",
            label="_nolegend_",
        )
        axes[1].plot(
            episodes,
            100.0 * rolling_rate,
            color=color,
            linewidth=1.35,
            label="_nolegend_",
        )

    axes[0].set_xlim(0, 3000)
    axes[0].set_xlabel("Training episode")
    axes[0].set_ylabel("Episode reward (objective seconds)")
    axes[0].grid(axis="y", color=GRID_COLOR, linewidth=0.55)
    axes[0].axhline(0, color=REFERENCE_COLOR, linewidth=0.65)
    axes[1].set_xlim(0, 3000)
    axes[1].set_ylim(-2, 102)
    axes[1].set_xlabel("Training episode")
    axes[1].set_ylabel("Feasible evaluations (%)")
    axes[1].grid(axis="y", color=GRID_COLOR, linewidth=0.55)
    for index, axis in enumerate(axes):
        _panel_label(axis, "ab"[index])
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=5)
    _save_figure(figure, output / "training_validation_v4", multi_panel=True)


def write_source_data(dataset: dict, output: Path) -> None:
    source_dataset = dict(dataset)
    proposed = dataset["load"]["proposed"]
    d3qn = next(row for row in dataset["layer1"] if row["label"] == "D3QN")
    ppo = next(row for row in dataset["layer1"] if row["label"] == "PPO")
    training_rows = {
        "Low": proposed[0],
        "Medium": proposed[1],
        "High": proposed[2],
        "D3QN": d3qn,
        "PPO": ppo,
    }
    source_dataset["training_validation"] = {}
    for label, record in training_rows.items():
        episodes, success, objective = load_eval_series(record)
        rewards, guided = load_rewards(record)
        source_dataset["training_validation"][label] = {
            "episode": episodes.tolist(),
            "success": success.astype(int).tolist(),
            "feasible_objective_s": [
                None if not np.isfinite(value) else float(value) for value in objective
            ],
            "training_episode_reward": rewards.tolist(),
            "training_episode_guided": guided.astype(int).tolist(),
            "source": record["source"],
        }
    (output / "figure_source_data_v4.json").write_text(
        json.dumps(source_dataset, indent=2, ensure_ascii=False), encoding="utf-8"
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
    for record in dataset["load"]["mco_astar"]:
        best = record["best_feasible"]
        rows.append(
            {
                "group": "mco_astar",
                "label": record["level"],
                "budget_s": record["budget_s"],
                "evaluations": record["evaluations"],
                "feasible_evaluations": record["feasible_evaluations"],
                "feasible_rate": record["feasible_rate"],
                "best_episode": "",
                "task_time_s": best["task_time_s"] if best else record["selected_evaluation"]["task_time_s"],
                "transmission_delay_s": (
                    best["transmission_delay_s"] if best
                    else record["selected_evaluation"]["transmission_delay_s"]
                ),
                "objective_s": best["objective_s"] if best else "",
                "outage_s": _outage(record),
                "source": record["source"],
            }
        )
    fieldnames = list(rows[0])
    with (output / "figure_source_data_v4.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_figures(dataset: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    write_source_data(dataset, output)
    plot_qos_sensitivity(dataset, output)
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
