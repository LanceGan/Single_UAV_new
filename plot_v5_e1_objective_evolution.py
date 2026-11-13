"""Plot v4 E1 objective-component evolution versus deterministic evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT.parent / "P3" / "figs" / "v5" / "e1_objective_evolution_v4"
SKILL_SCRIPTS = Path("/Users/ganruijie.1/.codex/skills/nature-figure/scripts")
RUNS = {
    "CA-TD3": ROOT / "journal_results/v4-qos/w50-50/qos6/A/low/td3_bc/eval_history.json",
    "D3QN": ROOT / "journal_results/v4-models/layer1/d3qn/low/d3qn/eval_history.json",
    "PPO": ROOT / "journal_results/v4-models/ppo/A/low/ppo/eval_history.json",
}
PLOT_METHODS = ("CA-TD3", "D3QN")
COLORS = {
    "CA-TD3": "#0F4D92",
    "D3QN": "#B8781A",
    "PPO": "#9A4D8E",
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


def _complete(item):
    points = int(item.get("visited_inspection_points", 0))
    expected_points = int(item.get("expected_inspection_points", 6))
    uploaded = float(item.get("uploaded_mb", 0.0))
    expected_upload = float(item.get("expected_upload_mb", 0.0) or 0.0)
    return points >= expected_points and uploaded >= expected_upload - 1e-7


def load_records():
    records = []
    method_counts = {}
    for method, path in RUNS.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        best = payload.get("eval_best") or payload.get("best") or {}
        best_episode = int(best.get("episode", -1))
        counts = {
            "evaluations": len(payload["history"]),
            "completed": 0,
            "qos_feasible": 0,
            "complete_infeasible": 0,
            "incomplete": 0,
            "eval_best_episode": best_episode,
        }
        for item in payload["history"]:
            completed = _complete(item)
            qos_feasible = completed and bool(item.get("success", False))
            objective_valid = (
                completed
                and item.get("task_time_s") is not None
                and item.get("transmission_delay_s") is not None
            )
            if completed:
                counts["completed"] += 1
                counts["qos_feasible" if qos_feasible else "complete_infeasible"] += 1
            else:
                counts["incomplete"] += 1
            records.append(
                {
                    "method": method,
                    "episode": int(item["episode"]),
                    "completed": completed,
                    "objective_valid": objective_valid,
                    "qos_feasible": qos_feasible if completed else None,
                    "task_time_s": float(item["task_time_s"])
                    if objective_valid
                    else None,
                    "transmission_delay_s": float(item["transmission_delay_s"])
                    if objective_valid
                    else None,
                    "reported_task_time_s": item.get("task_time_s"),
                    "reported_transmission_delay_s": item.get(
                        "transmission_delay_s"
                    ),
                    "outage_time_s": float(
                        item.get("outage_time_s", item.get("outage_s", np.nan))
                    ),
                    "is_eval_best": int(item["episode"]) == best_episode,
                    "termination_reason": item.get("termination_reason", ""),
                }
            )
        method_counts[method] = counts
    return records, method_counts


def write_source_data(records, method_counts, output):
    source_dir = output.parent / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    csv_path = source_dir / f"{output.name}.csv"
    json_path = source_dir / f"{output.name}.json"
    columns = [
        "method",
        "episode",
        "completed",
        "objective_valid",
        "qos_feasible",
        "task_time_s",
        "transmission_delay_s",
        "reported_task_time_s",
        "reported_transmission_delay_s",
        "outage_time_s",
        "is_eval_best",
        "termination_reason",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    json_path.write_text(
        json.dumps(
            {
                "protocol": {
                    "level": "Low",
                    "evaluation_interval_episodes": 25,
                    "first_recorded_evaluation_episode": 25,
                    "episode_zero_available": False,
                    "methods_plotted": list(PLOT_METHODS),
                    "methods_in_source_data": list(RUNS),
                    "included_points": "6/6 visits and complete upload",
                    "excluded_points": "timeout and premature termination",
                    "note": (
                        "All evaluations are retained here. Incomplete rollouts have "
                        "objective_valid=false because no end-to-end T or tau objective "
                        "was completed. The original runs did not evaluate the "
                        "untrained policy at episode 0. PPO is retained in the source "
                        "data but omitted from the plotted curves."
                    ),
                },
                "method_counts": method_counts,
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return csv_path, json_path


def plot(records, method_counts, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.65), constrained_layout=False)
    fields = (("task_time_s", r"Mission time $T$ (s)"),
              ("transmission_delay_s", r"Transmission delay $\tau$ (s)"))
    handles, labels = [], []
    legend_methods = set()
    for ax, (field, ylabel) in zip(axes, fields):
        for method in PLOT_METHODS:
            rows = [record for record in records if record["method"] == method]
            valid = [row for row in rows if row["objective_valid"]]
            if not valid:
                continue
            valid_episodes = np.asarray([row["episode"] for row in valid])
            valid_values = np.asarray([row[field] for row in valid])
            color = COLORS[method]
            line, = ax.plot(
                valid_episodes,
                valid_values,
                color=color,
                lw=1.25,
                alpha=0.90,
                label=(
                    f"{method} "
                    f"({method_counts[method]['completed']}/"
                    f"{method_counts[method]['evaluations']})"
                ),
                zorder=2,
            )
            best = next((row for row in valid if row["is_eval_best"]), None)
            if best is not None:
                ax.axvline(
                    best["episode"],
                    color=color,
                    lw=0.75,
                    linestyle=(0, (2, 2)),
                    alpha=0.65,
                    zorder=1,
                )
            if method not in legend_methods:
                handles.append(line)
                labels.append(line.get_label())
                legend_methods.add(method)
        ax.set_xlabel("Evaluation episode")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 3050)
        ax.set_xticks([0, 500, 1000, 1500, 2000, 2500, 3000])
        ax.grid(axis="y", color="#D9D9D9", lw=0.45, zorder=0)
        ax.text(
            -0.14,
            1.03,
            "ab"[list(axes).index(ax)],
            transform=ax.transAxes,
            fontsize=8,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=len(PLOT_METHODS),
    )
    fig.text(
        0.5,
        -0.01,
        "Curves connect completed 6/6 and full-upload evaluations; "
        "incomplete or timed-out evaluations are omitted.\n"
        "PPO is omitted (one completed evaluation). "
        "Vertical dotted lines mark eval-best; legend counts are "
        "completed/evaluations.",
        ha="center",
        va="top",
        fontsize=6.4,
        linespacing=1.2,
    )
    fig.tight_layout(rect=(0, 0.105, 1, 0.92), pad=0.5, w_pad=1.4)
    sys.path.insert(0, str(SKILL_SCRIPTS))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    require_matplotlib_panel_alignment(
        fig,
        axes=list(axes),
        panel_ids=["a", "b"],
        row_groups=[["a", "b"]],
        json_out=f"{output}.alignment.json",
        overlay_svg=f"{output}.alignment.svg",
        strict=True,
    )
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    records, method_counts = load_records()
    write_source_data(records, method_counts, args.output)
    plot(records, method_counts, args.output)
    print(args.output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
