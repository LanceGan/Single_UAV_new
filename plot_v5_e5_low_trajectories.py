"""Render the completed V5 Low-tier seven-method trajectory comparison."""

from __future__ import annotations

import csv
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluate_d3qn import (
    evaluate_checkpoint as evaluate_d3qn_checkpoint,
    load_d3qn_run_context,
)
from evaluate_policy import evaluate_checkpoint, load_run_context
from plot_v4_trajectory_figures import (
    _draw_map,
    _draw_path,
    _map_data,
    _map_layers,
    _panel_label,
)


ROOT = Path(__file__).resolve().parent
P3 = ROOT.parent / "P3"
FIGURE_DIR = P3 / "figs" / "v5"
SOURCE_DIR = FIGURE_DIR / "source_data"
SKILL_SCRIPTS = Path("/Users/ganruijie.1/.codex/skills/nature-figure/scripts")
ACTOR_BUNDLE = ROOT / "sync_bundles" / "v4_eval_best_actors"

METHOD_SPECS = {
    "DRM-HO": {
        "run": ROOT / "journal_results/v4-qos/w50-50/qos6/A/low/td3_bc",
        "actor": ACTOR_BUNDLE / "w50-50_qos6_actor.pth",
        "expected": (169.0, 75.5, 3.0, 6),
        "role": "proposed",
    },
    "MCO-A* adaptation": {
        "deterministic": ROOT / "journal_results/v5-deterministic/mco-astar/low",
        "expected": (287.5, 170.0, 6.0, 6),
        "role": "baseline",
    },
    "EEGA+TD3": {
        "run": ROOT / "journal_results/v4-baselines/eega/A/low/td3_bc",
        "actor": ROOT
        / "journal_results/v4-baselines/eega/A/low/td3_bc/models/"
        "td3_bc_td3_actoreval_best.pth",
        "expected": (169.0, 109.5, 3.0, 6),
        "role": "baseline",
    },
    "Distance-TSP+TD3": {
        "run": ROOT / "journal_results/v4-baselines/distance-tsp/A/low/td3_bc",
        "actor": ROOT
        / "journal_results/v4-baselines/distance-tsp/A/low/td3_bc/models/"
        "td3_bc_td3_actoreval_best.pth",
        "expected": (167.0, 124.5, 3.5, 6),
        "role": "baseline",
    },
    "Yang-style+TD3": {
        "run": ROOT / "journal_results/v4-baselines/yang/A/low/td3_bc",
        "actor": ROOT
        / "journal_results/v4-baselines/yang/A/low/td3_bc/models/"
        "td3_bc_td3_actoreval_best.pth",
        "expected": (190.5, 104.5, 3.5, 6),
        "role": "baseline",
    },
    "DRL-SA": {
        "run": ROOT / "journal_results/v4-baselines/drl-sa/low/d3qn",
        "actor": ROOT
        / "journal_results/v4-baselines/drl-sa/low/d3qn/models/d3qn_eval_best.pt",
        "expected": (219.5, 128.0, 3.5, 6),
        "role": "baseline",
        "d3qn": True,
    },
    "SA-D3QN": {
        "run": ROOT / "journal_results/v4-baselines/sa-d3qn/low/d3qn",
        "actor": ROOT
        / "journal_results/v4-baselines/sa-d3qn/low/d3qn/models/d3qn_eval_best.pt",
        "expected": (200.0, 146.5, 4.5, 6),
        "role": "baseline",
        "d3qn": True,
    },
}
METHOD_ORDER = tuple(METHOD_SPECS)

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


def _vals(report):
    return (
        float(report["task_time_s"]),
        float(report["transmission_delay_s"]),
        float(report["outage_time_s"]),
        int(report["visited_inspection_points"]),
    )


def evaluate_all():
    results = {}
    for method, spec in METHOD_SPECS.items():
        if "deterministic" in spec:
            metrics = json.loads(
                (spec["deterministic"] / "metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            with np.load(spec["deterministic"] / "trajectory.npz") as payload:
                trace = {key: payload[key] for key in payload.files}
            values = (
                metrics["task_time_s"],
                metrics["transmission_delay_s"],
                metrics["outage_s"],
                metrics["visited_inspection_points"],
            )
            report = {
                **metrics,
                "success": bool(metrics["success"]),
            }
        elif spec.get("d3qn"):
            context = load_d3qn_run_context(spec["run"])
            with redirect_stdout(io.StringIO()):
                report, trace = evaluate_d3qn_checkpoint(
                    context, spec["actor"]
                )
            values = _vals(report)
        else:
            context = load_run_context(spec["run"])
            with redirect_stdout(io.StringIO()):
                report, trace = evaluate_checkpoint(context, spec["actor"])
            values = _vals(report)
        if not np.allclose(values, spec["expected"], atol=1e-9):
            raise ValueError(
                f"{method} replay mismatch: expected {spec['expected']}, got {values}"
            )
        results[method] = {
            "report": report,
            "trace": trace,
            "run": spec.get("run", spec.get("deterministic")),
            "actor": spec.get("actor"),
            "role": spec["role"],
        }
    return results


def write_source(results):
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    arrays = {}
    metrics = {}
    for method in METHOD_ORDER:
        result = results[method]
        trace = result["trace"]
        positions = np.asarray(trace["positions"], dtype=float)
        times = np.asarray(trace["time_s"], dtype=float)
        outage = np.asarray(trace["outage_s"], dtype=float)
        visited = np.asarray(trace["visited_points"], dtype=int)
        key = method.replace("*", "star").replace("+", "_").replace("-", "_")
        arrays[f"{key}_x_km"] = positions[:, 0]
        arrays[f"{key}_y_km"] = positions[:, 1]
        arrays[f"{key}_time_s"] = times
        arrays[f"{key}_outage_s"] = outage
        arrays[f"{key}_visited_points"] = visited
        for x, y, time_s, outage_s, visited_count in zip(
            positions[:, 0], positions[:, 1], times, outage, visited
        ):
            rows.append(
                {
                    "method": method,
                    "time_s": float(time_s),
                    "x_m": float(x * 100.0),
                    "y_m": float(y * 100.0),
                    "outage_s": float(outage_s),
                    "visited_points": int(visited_count),
                }
            )
        report = result["report"]
        metrics[method] = {
            "role": result["role"],
            "run": str(result["run"].resolve()),
            "actor": str(result["actor"].resolve()) if result["actor"] else None,
            "success": bool(report.get("success", False)),
            "task_time_s": report.get("task_time_s"),
            "transmission_delay_s": report.get("transmission_delay_s"),
            "objective_s": (
                report.get("task_time_s") + report.get("transmission_delay_s")
            ),
            "outage_time_s": report.get(
                "outage_time_s", report.get("outage_s")
            ),
            "visited_inspection_points": report.get(
                "visited_inspection_points"
            ),
            "uploaded_mb": report.get("uploaded_mb"),
            "inspection_order": report.get("inspection_order"),
        }
    np.savez(SOURCE_DIR / "e5_low_trajectories_v5.npz", **arrays)
    csv_path = SOURCE_DIR / "e5_low_trajectories_v5.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "method",
                "time_s",
                "x_m",
                "y_m",
                "outage_s",
                "visited_points",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    (SOURCE_DIR / "e5_low_trajectory_metrics_v5.json").write_text(
        json.dumps(
            {
                "protocol": {
                    "level": "Low",
                    "qos_constraint": "max_contiguous_outage <= 6 s",
                    "seed": 1,
                    "policy_mode": "deterministic, no exploration noise or MapGuide",
                },
                "methods": metrics,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def plot(results):
    g2a, a2g = _map_data()
    x_grid, y_grid, _, snr = _map_layers(g2a, a2g)
    users = np.loadtxt(
        ROOT / "results" / "datas" / "Users_6.txt"
    )[:, :2] * 100.0
    figure, axes = plt.subplots(2, 4, figsize=(7.15, 4.4))
    figure.subplots_adjust(
        left=0.065,
        right=0.90,
        bottom=0.09,
        top=0.91,
        wspace=0.14,
        hspace=0.22,
    )
    panel_axes = []
    panel_ids = []
    mesh = None
    for index, method in enumerate(METHOD_ORDER):
        row, column = divmod(index, 4)
        axis = axes[row, column]
        instance = results[method]
        report = instance["report"]
        objective = report["task_time_s"] + report["transmission_delay_s"]
        title = f"{method}\n{objective:.1f} s"
        mesh = _draw_map(
            axis,
            x_grid,
            y_grid,
            snr,
            cmap="viridis",
            vmin=float(snr.min()),
            vmax=float(snr.max()),
            title=title,
            users=users,
            show_numbering=False,
        )
        mesh.set_rasterized(True)
        color = "#FF5A5F" if instance["role"] == "proposed" else "#FFFFFF"
        _draw_path(axis, instance["trace"], color)
        panel_axes.append(axis)
        panel_ids.append("abcdefg"[index])
    for column in range(3):
        axes[0, column].set_xlabel("")
    axes[1, 3].axis("off")
    for row in range(2):
        for column in range(1, 4):
            axes[row, column].tick_params(axis="y", labelleft=False)
            axes[row, column].set_ylabel("")

    for panel_id, axis in zip(panel_ids, panel_axes):
        _panel_label(axis, panel_id)
    cax = figure.add_axes([0.915, 0.18, 0.008, 0.6])
    colorbar = figure.colorbar(mesh, cax=cax)
    colorbar.set_label("A2G SNR (dB)")
    figure.text(
        0.5,
        0.012,
        "All methods complete 6/6 inspections and full upload. Coral marks the "
        "proposed DRM-HO path; white marks reference methods. Results are "
        "single-run, deterministic and guide-free.",
        ha="center",
        va="bottom",
        fontsize=6.3,
    )

    output = FIGURE_DIR / "e5_low_trajectories_v5"
    sys.path.insert(0, str(SKILL_SCRIPTS))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    require_matplotlib_panel_alignment(
        figure,
        axes=panel_axes,
        panel_ids=panel_ids,
        row_groups=[panel_ids[:4], panel_ids[4:]],
        column_groups=[
            ["a", "e"],
            ["b", "f"],
            ["c", "g"],
        ],
        exclude_axes=[cax, axes[1, 3]],
        json_out=f"{output}.alignment.json",
        overlay_svg=f"{output}.alignment.svg",
        strict=True,
    )
    figure.savefig(output.with_suffix(".pdf"), dpi=600, bbox_inches=None)
    figure.savefig(output.with_suffix(".svg"), bbox_inches=None)
    figure.savefig(output.with_suffix(".png"), dpi=600, bbox_inches=None)
    figure.savefig(
        output.with_suffix(".tiff"),
        dpi=600,
        bbox_inches=None,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(figure)


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    results = evaluate_all()
    write_source(results)
    plot(results)
    print(FIGURE_DIR / "e5_low_trajectories_v5.pdf")


if __name__ == "__main__":
    main()
