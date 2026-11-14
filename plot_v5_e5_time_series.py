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
from fixed_radio_maps import FixedRadioMaps
from plot_v5_e4_trajectories import _evaluate_on_recorded_device
import torch


ROOT = Path(__file__).resolve().parent
FIGURE_DIR = ROOT.parent / "P3" / "figs" / "v5"
SOURCE_DIR = FIGURE_DIR / "source_data"
RADIO_MAPS = FixedRadioMaps(ROOT / "results" / "datas")

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


def _time_series(trace):
    times = np.asarray(trace["time_s"], dtype=float)
    pending = np.asarray(trace["pending_mb"], dtype=float)
    outage = np.asarray(
        trace["current_contiguous_outage_s"], dtype=float
    )
    positions = np.asarray(trace["positions"], dtype=float)
    if not (
        len(times)
        == len(pending)
        == len(outage)
        == len(positions)
    ):
        raise ValueError("elapsed-time arrays have inconsistent lengths")
    if np.any(np.diff(times) < -1e-12):
        raise ValueError("flight time is not monotone")
    outage_probability = np.asarray(
        [
            # World stores positions in 100 m units before converting to the
            # FixedRadioMaps coordinate scale used by its radio lookup.
            RADIO_MAPS.g2a_outage_probability(position / 10.0)
            for position in positions
        ],
        dtype=float,
    )
    if "g2a_outage_probability" in trace:
        recorded_probability = np.asarray(
            trace["g2a_outage_probability"], dtype=float
        )
        if recorded_probability.shape != outage_probability.shape:
            raise ValueError(
                "recorded G2A outage-probability series has inconsistent length"
            )
        if not np.allclose(
            recorded_probability,
            outage_probability,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "recorded G2A outage probability does not match the radio map"
            )
    delay_prefix = np.cumsum((pending[:-1] > 0.0) * np.diff(times))
    objective_prefix = times[1:] + delay_prefix
    return (
        times,
        objective_prefix,
        outage_probability,
        outage,
        positions,
    )


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
        item = results[method]["Low"]
        times, objective, outage_probability, outage, positions = _time_series(
            item["trace"]
        )
        if item["complete"]:
            expected_final = float(
                item["report"]["task_time_s"]
                + item["report"]["transmission_delay_s"]
            )
            if abs(float(objective[-1]) - expected_final) > 1e-9:
                raise ValueError(
                    f"{method} Low objective-prefix mismatch: "
                    f"{objective[-1]} vs {expected_final}"
                )
        key = (
            method.replace("*", "star")
            .replace("+", "_")
            .replace("-", "_")
            .replace(" ", "_")
        )
        arrays[f"{key}_time_s"] = times
        arrays[f"{key}_g2a_outage_probability"] = outage_probability
        arrays[f"{key}_current_contiguous_outage_s"] = outage
        arrays[f"{key}_positions_km"] = positions
        for index, time_s in enumerate(times, start=1):
            rows.append(
                {
                    "method": method,
                    "level": "Low",
                    "step": index,
                    "flight_time_s": float(time_s),
                    "g2a_outage_probability": float(
                        outage_probability[index - 1]
                    ),
                    "current_contiguous_outage_s": float(outage[index - 1]),
                    "complete": item["complete"],
                }
            )
        summaries.append(
            {
                "method": method,
                "level": "Low",
                "complete": item["complete"],
                "status": item["source_status"],
                "run": str(item["run"].resolve()),
                "actor": (
                    str(item["actor"].resolve())
                    if item["actor"] is not None
                    else None
                ),
                "final_flight_time_s": float(times[-1]),
                "max_g2a_outage_probability": float(
                    np.max(outage_probability)
                ),
                "max_current_contiguous_outage_s": float(np.max(outage)),
            }
        )
    output_stem = SOURCE_DIR / "e5_low_realtime_v5"
    np.savez_compressed(output_stem.with_suffix(".npz"), **arrays)
    with output_stem.with_suffix(".csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    output_stem.with_suffix(".json").write_text(
        json.dumps(
            {
                "protocol": {
                    "scope": (
                        "Low payload tier only for visualization; the complete "
                        "21-row method-by-payload table is retained separately"
                    ),
                    "x_axis": "elapsed flight time within the deterministic rollout",
                    "g2a_outage_probability": (
                        "Radio_datas.npz arr_0 sampled at each recorded UAV "
                        "position through the World radio-map convention; "
                        "values are not complemented"
                    ),
                    "outage_probability_threshold": 0.5,
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
        times, _, outage_probability, outage, _ = _time_series(item["trace"])
        if metric == "probability":
            y = outage_probability
        else:
            y = outage
        axis.plot(
            times,
            y,
            color=COLORS[method],
            linestyle=LINE_STYLES[method],
            linewidth=1.35 if method == "DRM-HO" else 0.95,
            label=method,
        )

    axis.set_xlabel("Flight time (s)")
    axis.set_xlim(0.0, 220.0)
    axis.set_xticks([0, 50, 100, 150, 200])
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.45)
    axis.set_axisbelow(True)
    if metric == "probability":
        axis.axhline(
            0.5,
            color="#6B6B6B",
            linewidth=0.75,
            linestyle=(0, (2.0, 2.0)),
            zorder=0,
        )
        axis.set_ylim(0.0, 1.0)
        axis.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        axis.set_ylabel("G2A outage probability")
    else:
        axis.set_ylim(0.0, 6.5)
        axis.set_yticks([0, 1, 2, 3, 4, 5, 6])
        axis.set_ylabel("Current contiguous outage (s)")


def plot(results):
    for panel, metric in (("a", "probability"), ("b", "outage")):
        figure, axis = plt.subplots(1, 1, figsize=(3.45, 2.3))
        _draw_panel(axis, results, "Low", metric)
        figure.tight_layout(pad=0.6)
        _save_formats(
            figure,
            FIGURE_DIR / f"e5_low_realtime_v5_panel_{panel}",
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
    handles.extend(
        [
            Line2D(
                [0],
                [0],
                color="#6B6B6B",
                linewidth=0.75,
                linestyle=(0, (2.0, 2.0)),
                label="0.5 outage threshold",
            ),
        ]
    )
    legend = plt.figure(figsize=(7.15, 0.62))
    legend.legend(
        handles=handles,
        loc="center",
        ncol=4,
        frameon=False,
        fontsize=6.2,
        handlelength=2.1,
    )
    _save_formats(legend, FIGURE_DIR / "e5_low_realtime_v5_legend")
    plt.close(legend)


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    results = evaluate_all()
    write_source(results)
    plot(results)
    print(SOURCE_DIR / "e5_low_realtime_v5.csv")
    print(FIGURE_DIR / "e5_low_realtime_v5_panel_a.pdf")
    print(FIGURE_DIR / "e5_low_realtime_v5_panel_b.pdf")


if __name__ == "__main__":
    main()
