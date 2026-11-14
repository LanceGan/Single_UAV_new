"""Render elapsed-time objective and current-outage curves for E5 pipelines."""

from __future__ import annotations

import csv
import io
import json
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from evaluate_d3qn import (
    _build_world as build_d3qn_world,
    load_d3qn_checkpoint,
    load_d3qn_run_context,
)
from evaluate_policy import evaluate_checkpoint, load_run_context
from discrete_actions import DiscreteActionCodec
from plot_v5_e4_trajectories import _evaluate_on_recorded_device
import torch


ROOT = Path(__file__).resolve().parent
FIGURE_DIR = ROOT.parent / "P3" / "figs" / "v5"
SOURCE_DIR = FIGURE_DIR / "source_data"

LEVELS = ("Low", "Medium", "High")
METHODS = (
    "DRM-HO",
    "MCO-A* adaptation",
    "EEGA+TD3",
    "Distance-TSP+TD3",
    "Yang-style+TD3",
    "DRL-SA",
    "SA-D3QN",
)
COLORS = {
    "DRM-HO": "#C23B3B",
    "MCO-A* adaptation": "#4B5563",
    "EEGA+TD3": "#0F4D92",
    "Distance-TSP+TD3": "#B8781A",
    "Yang-style+TD3": "#3F7F6F",
    "DRL-SA": "#8365A8",
    "SA-D3QN": "#A75E87",
}
LINE_STYLES = {
    "DRM-HO": "-",
    "MCO-A* adaptation": (0, (4, 2)),
    "EEGA+TD3": "-",
    "Distance-TSP+TD3": "-",
    "Yang-style+TD3": "-",
    "DRL-SA": "-",
    "SA-D3QN": "-",
}

RUNS = {
    "DRM-HO": {
        "Low": "journal_results/v4-qos/w50-50/qos6/A/low/td3_bc",
        "Medium": "journal_results/v4-models/load/medium/A/medium/td3_bc",
        "High": "journal_results/v4-models/load/high/A/high/td3_bc",
    },
    "EEGA+TD3": {
        "Low": "journal_results/v4-baselines/eega/A/low/td3_bc",
        "Medium": "journal_results/v5-mac/E5/eega_td3/baseline/medium/td3_bc",
        "High": "journal_results/v5-mac/E5/eega_td3/baseline/high/td3_bc",
    },
    "Distance-TSP+TD3": {
        "Low": "journal_results/v4-baselines/distance-tsp/A/low/td3_bc",
        "Medium": "journal_results/v5-mac/E5/distance_tsp_td3/baseline/medium/td3_bc",
        "High": "journal_results/v5-mac/E5/distance_tsp_td3/baseline/high/td3_bc",
    },
    "Yang-style+TD3": {
        "Low": "journal_results/v4-baselines/yang/A/low/td3_bc",
        "Medium": "journal_results/v5-mac/E5/yang_td3/baseline/medium/td3_bc",
        "High": "journal_results/v5-mac/E5/yang_td3/baseline/high/td3_bc",
    },
    "DRL-SA": {
        "Low": "journal_results/v4-baselines/drl-sa/low/d3qn",
        "Medium": "journal_results/v5-mac/E5/drl_sa/medium/d3qn",
        "High": "journal_results/v5-mac/E5/drl_sa/high/d3qn",
    },
    "SA-D3QN": {
        "Low": "journal_results/v4-baselines/sa-d3qn/low/d3qn",
        "Medium": "journal_results/v5-mac/E5/sa_d3qn/medium/d3qn",
        "High": "journal_results/v5-mac/E5/sa_d3qn/high/d3qn",
    },
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


def _load_expected():
    with (SOURCE_DIR / "e5_method_payload_matrix_v5.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    return {(row["method"], row["level"]): row for row in rows}


def _actor_for(run: Path, d3qn: bool):
    pattern = "d3qn_eval_best.pt" if d3qn else "*actoreval_best.pth"
    matches = sorted((run / "models").glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one eval-best actor for {run}, got {matches}"
        )
    return matches[0]


def _evaluate_run(method, level, d3qn):
    run = ROOT / RUNS[method][level]
    if method == "DRM-HO":
        actor = {
            "Low": (
                ROOT
                / "sync_bundles/v4_eval_best_actors/w50-50_qos6_actor.pth"
            ),
            "Medium": (
                ROOT
                / "sync_bundles/v4_eval_best_actors/medium_qos6_actor.pth"
            ),
            "High": (
                ROOT
                / "sync_bundles/v4_eval_best_actors/high_qos6_actor.pth"
            ),
        }[level]
    else:
        actor = _actor_for(run, d3qn)
    if d3qn:
        context = load_d3qn_run_context(run)
        context = replace(
            context,
            total_steps=int(context.config.get("total_steps", 1000)),
        )
        network, _ = load_d3qn_checkpoint(
            actor,
            context.state_dim,
            context.action_count,
            context.net_width,
        )
        device_name = str(context.config.get("device", "cpu")).lower()
        if device_name == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError(f"{run}: MPS replay device is unavailable")
        device = torch.device(device_name)
        network = network.to(device).eval()
        codec = DiscreteActionCodec()

        def act(state):
            with torch.no_grad():
                tensor = torch.as_tensor(
                    state, dtype=torch.float32, device=device
                ).unsqueeze(0)
                action_index = int(network(tensor).argmax(dim=1).item())
            return codec.decode(action_index)

        with redirect_stdout(io.StringIO()):
            from evaluate_policy import rollout

            report, trace = rollout(
                context,
                build_d3qn_world(context),
                act,
            )
        report["replay_device"] = device_name
    else:
        context = load_run_context(run)
        with redirect_stdout(io.StringIO()):
            report, trace = _evaluate_on_recorded_device(context, actor)
    return report, trace, run, actor


def _evaluate_mco(level):
    run = ROOT / "journal_results" / "v5-deterministic" / "mco-astar" / level.lower()
    report = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    with np.load(run / "trajectory.npz") as payload:
        trace = {key: payload[key] for key in payload.files}
    return report, trace, run, None


def _prefix_series(trace):
    times = np.asarray(trace["time_s"], dtype=float)
    pending = np.asarray(trace["pending_mb"], dtype=float)
    outage = np.asarray(
        trace["current_contiguous_outage_s"], dtype=float
    )
    if not (len(times) == len(pending) == len(outage)):
        raise ValueError("elapsed-time arrays have inconsistent lengths")
    if np.any(np.diff(times) < -1e-12):
        raise ValueError("flight time is not monotone")
    delay_prefix = np.cumsum((pending[:-1] > 0.0) * np.diff(times))
    objective_prefix = times[1:] + delay_prefix
    return times, objective_prefix, outage


def evaluate_all():
    expected = _load_expected()
    results = {}
    for method in METHODS:
        results[method] = {}
        for level in LEVELS:
            if method == "MCO-A* adaptation":
                report, trace, run, actor = _evaluate_mco(level)
            else:
                report, trace, run, actor = _evaluate_run(
                    method,
                    level,
                    method in {"DRL-SA", "SA-D3QN"},
                )
            expected_row = expected[(method, level)]
            complete = int(expected_row["visited_points"]) >= 6
            actual_outage = float(
                report.get("outage_time_s", report.get("outage_s", 0.0))
            )
            if abs(actual_outage - float(expected_row["contiguous_outage_s"])) > 1e-9:
                raise ValueError(
                    f"{method} {level} outage mismatch: "
                    f"{actual_outage} vs {expected_row['contiguous_outage_s']}"
                )
            actual_visits = int(report.get("visited_inspection_points", 0))
            if actual_visits != int(expected_row["visited_points"]):
                raise ValueError(
                    f"{method} {level} visit mismatch: "
                    f"{actual_visits} vs {expected_row['visited_points']}"
                )
            if complete:
                for field in ("task_time_s", "transmission_delay_s"):
                    actual = float(report[field])
                    expected_value = float(expected_row[field])
                    if abs(actual - expected_value) > 1e-9:
                        raise ValueError(
                            f"{method} {level} {field} mismatch: "
                            f"{actual} vs {expected_value}"
                        )
            results[method][level] = {
                "report": report,
                "trace": trace,
                "complete": complete,
                "run": run,
                "actor": actor,
                "source_status": expected_row["status"],
            }
    return results


def write_source(results):
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    arrays = {}
    summaries = []
    for method in METHODS:
        for level in LEVELS:
            item = results[method][level]
            times, objective, outage = _prefix_series(item["trace"])
            if item["complete"]:
                expected_final = float(
                    item["report"]["task_time_s"]
                    + item["report"]["transmission_delay_s"]
                )
                if abs(float(objective[-1]) - expected_final) > 1e-9:
                    raise ValueError(
                        f"{method} {level} objective-prefix mismatch: "
                        f"{objective[-1]} vs {expected_final}"
                    )
            key = (
                method.replace("*", "star")
                .replace("+", "_")
                .replace("-", "_")
                .replace(" ", "_")
                + f"_{level.lower()}"
            )
            arrays[f"{key}_time_s"] = times
            arrays[f"{key}_objective_s"] = objective
            arrays[f"{key}_current_contiguous_outage_s"] = outage
            for index, time_s in enumerate(times, start=1):
                objective_s = (
                    float(objective[index - 1])
                    if index <= len(objective)
                    else None
                )
                rows.append(
                    {
                        "method": method,
                        "level": level,
                        "step": index,
                        "flight_time_s": float(time_s),
                        "objective_s": objective_s,
                        "current_contiguous_outage_s": float(
                            outage[index - 1]
                        ),
                        "complete": item["complete"],
                    }
                )
            summaries.append(
                {
                    "method": method,
                    "level": level,
                    "complete": item["complete"],
                    "status": item["source_status"],
                    "run": str(item["run"].resolve()),
                    "actor": (
                        str(item["actor"].resolve())
                        if item["actor"] is not None
                        else None
                    ),
                    "final_flight_time_s": float(times[-1]),
                    "final_objective_s": float(objective[-1]),
                    "final_current_contiguous_outage_s": float(outage[-1]),
                }
            )
    np.savez_compressed(SOURCE_DIR / "e5_time_series_v5.npz", **arrays)
    with (SOURCE_DIR / "e5_time_series_v5.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    (SOURCE_DIR / "e5_time_series_v5.json").write_text(
        json.dumps(
            {
                "protocol": {
                    "x_axis": "elapsed flight time within the deterministic rollout",
                    "objective": (
                        "J(t) = elapsed T(t) + cumulative transmission delay "
                        "through time t"
                    ),
                    "contiguous_outage": (
                        "current consecutive outage-run duration; resets to "
                        "zero in the first non-outage slot"
                    ),
                    "seed": 1,
                },
                "series": summaries,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _save_formats(figure, output):
    figure.savefig(output.with_suffix(".pdf"), dpi=600, bbox_inches="tight")
    figure.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=600, bbox_inches="tight")
    figure.savefig(
        output.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(figure)


def _draw_panel(axis, results, level, metric):
    for method in METHODS:
        item = results[method][level]
        times, objective, outage = _prefix_series(item["trace"])
        if metric == "objective":
            x = times[1:]
            y = objective
        else:
            x = times
            y = outage
        axis.plot(
            x,
            y,
            color=COLORS[method],
            linestyle=LINE_STYLES[method],
            linewidth=1.35 if method == "DRM-HO" else 0.95,
            label=method,
        )
        if not item["complete"]:
            axis.plot(
                x[-1],
                y[-1],
                marker="x",
                markersize=4,
                markeredgewidth=1.0,
                color=COLORS[method],
                linestyle="none",
            )

    axis.set_xlabel("Flight time (s)")
    axis.set_ylabel(
        "Cumulative objective $J(t)$ (s)"
        if metric == "objective"
        else "Current contiguous outage (s)"
    )
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.45)
    axis.set_axisbelow(True)


def plot(results):
    for panel, level in zip("abcdef", [l for l in LEVELS for _ in range(2)]):
        metric = "objective" if "ace".find(panel) >= 0 else "outage"
        figure, axis = plt.subplots(1, 1, figsize=(3.45, 2.3))
        _draw_panel(axis, results, level, metric)
        figure.tight_layout(pad=0.6)
        _save_formats(
            figure,
            FIGURE_DIR / f"e5_time_series_v5_panel_{panel}",
        )

    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[method],
            linestyle=LINE_STYLES[method],
            linewidth=1.2,
            label=method,
        )
        for method in METHODS
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            marker="x",
            linestyle="none",
            color="#5C5C5C",
            markersize=4.5,
            markeredgewidth=1.0,
            label="Incomplete rollout",
        )
    )
    legend = plt.figure(figsize=(7.15, 0.62))
    legend.legend(
        handles=handles,
        loc="center",
        ncol=2,
        frameon=False,
        fontsize=6.2,
        handlelength=2.1,
    )
    _save_formats(legend, FIGURE_DIR / "e5_time_series_v5_legend")
    plt.close(legend)


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    results = evaluate_all()
    write_source(results)
    plot(results)
    print(SOURCE_DIR / "e5_time_series_v5.csv")
    print(FIGURE_DIR / "e5_time_series_v5_panel_a.pdf")


if __name__ == "__main__":
    main()
