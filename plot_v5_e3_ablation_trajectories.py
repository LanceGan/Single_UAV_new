"""Replay and plot the V5 A2G-observability ablation trajectories."""

from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
import torch
from matplotlib.lines import Line2D

from evaluate_policy import (
    _build_world,
    checkpoint_paths,
    load_actor,
    load_run_context,
    rollout,
)
from plot_v4_trajectory_figures import _draw_map, _map_data, _map_layers


ROOT = Path(__file__).resolve().parent
P3_FIGURE_DIR = ROOT.parent / "P3" / "figs" / "v5"
SOURCE_DIR = P3_FIGURE_DIR / "source_data"
QA_DIR = P3_FIGURE_DIR / "qa"
SKILL_SCRIPTS = Path.home() / ".codex" / "skills" / "nature-figure" / "scripts"
RECORD_PATH = SOURCE_DIR / "e3_a2g_ablation_v5.json"
OUTPUT_STEM = P3_FIGURE_DIR / "e3_a2g_trajectories_v5"
SOURCE_STEM = SOURCE_DIR / "e3_a2g_trajectories_v5"

LEVELS = ("Low", "Medium", "High")
METHODS = ("A2G-blind", "A2G-visible")
RUNS = {
    ("Low", "A2G-blind"): (
        ROOT
        / "journal_results"
        / "v4-baselines"
        / "distance-tsp"
        / "A"
        / "low"
        / "td3_bc"
    ),
    ("Low", "A2G-visible"): (
        ROOT
        / "journal_results"
        / "v4-baselines"
        / "distance-tsp-a2g-visible"
        / "A"
        / "low"
        / "td3_bc"
    ),
    ("Medium", "A2G-blind"): (
        ROOT
        / "journal_results"
        / "v5-mac"
        / "E3"
        / "distance_tsp_blind"
        / "baseline"
        / "medium"
        / "td3_bc"
    ),
    ("Medium", "A2G-visible"): (
        ROOT
        / "journal_results"
        / "v5-mac"
        / "E3"
        / "distance_tsp_visible"
        / "baseline"
        / "medium"
        / "td3_bc"
    ),
    ("High", "A2G-blind"): (
        ROOT
        / "journal_results"
        / "v5-mac"
        / "E3"
        / "distance_tsp_blind"
        / "baseline"
        / "high"
        / "td3_bc"
    ),
    ("High", "A2G-visible"): (
        ROOT
        / "journal_results"
        / "v5-mac"
        / "E3"
        / "distance_tsp_visible"
        / "baseline"
        / "high"
        / "td3_bc"
    ),
}

COLORS = {
    "A2G-visible": "#175A9C",
    "A2G-blind": "#D95F4B",
}
LINESTYLES = {
    "A2G-visible": "-",
    "A2G-blind": (0, (4.0, 2.0)),
}
PAYLOAD_TOTAL_MB = {"Low": 450.0, "Medium": 900.0, "High": 1500.0}

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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evaluate_on_recorded_device(context, checkpoint: Path):
    recorded = str(context.config.get("device", "cpu")).lower()
    device_name = "cpu" if recorded in {"", "none"} else recorded
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            f"{context.run_dir} requires MPS replay precision, but MPS is unavailable"
        )
    if device_name not in {"cpu", "mps"}:
        raise ValueError(
            f"{context.run_dir} has unsupported replay device {device_name!r}"
        )
    device = torch.device(device_name)
    actor = load_actor(
        checkpoint,
        context.state_dim,
        context.net_width,
        algorithm=context.algorithm,
    ).to(device)
    actor.eval()
    world = _build_world(context)

    def act(state):
        with torch.no_grad():
            action = actor(
                torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            )[0]
        return action.cpu().numpy()

    report, trace = rollout(context, world, act)
    report["replay_device"] = device_name
    report["checkpoint"] = str(checkpoint.resolve())
    return report, trace


def _check_expected(record: dict, report: dict) -> None:
    complete = bool(record["complete"])
    expected = {
        "visited_inspection_points": int(record["visited_points"]),
        "uploaded_mb": float(record["uploaded_mb"]),
        "outage_time_s": float(record["contiguous_outage_s"]),
        "task_time_s": float(
            record["task_time_s"]
            if complete
            else record["diagnostic_task_time_s"]
        ),
        "transmission_delay_s": float(
            record["transmission_delay_s"]
            if complete
            else record["diagnostic_transmission_delay_s"]
        ),
    }
    for field, value in expected.items():
        actual = float(report[field])
        if not np.isclose(actual, value, atol=1e-9):
            raise ValueError(
                f"{record['level']} {record['method']} mismatch for {field}: "
                f"{actual} vs {value}"
            )
    if bool(report["mission_complete"]) != complete:
        raise ValueError(
            f"{record['level']} {record['method']} completion mismatch"
        )


def replay_all() -> tuple[list[dict], dict[str, np.ndarray]]:
    payload = json.loads(RECORD_PATH.read_text(encoding="utf-8"))
    records = {
        (record["level"], record["method"]): record
        for record in payload["records"]
    }
    results = []
    arrays = {}
    for level in LEVELS:
        for method in METHODS:
            source_record = records[(level, method)]
            run_dir = RUNS[(level, method)]
            context = load_run_context(run_dir)
            checkpoint = checkpoint_paths(
                run_dir,
                context.algorithm,
                int(context.config["episodes"]),
            )["eval_best"]
            with contextlib.redirect_stdout(io.StringIO()):
                report, trace = _evaluate_on_recorded_device(context, checkpoint)
            _check_expected(source_record, report)
            results.append(
                {
                    "level": level,
                    "method": method,
                    "complete": bool(report["mission_complete"]),
                    "report": report,
                    "trace": trace,
                    "source_record": source_record,
                    "run_dir": run_dir,
                    "checkpoint": checkpoint,
                }
            )
            prefix = f"{level.lower()}_{'aware' if method == 'A2G-visible' else 'blind'}"
            arrays[f"{prefix}_positions_m"] = np.asarray(trace["positions"]) * 100.0
            arrays[f"{prefix}_time_s"] = np.asarray(trace["time_s"], dtype=float)
            arrays[f"{prefix}_outage_s"] = np.asarray(trace["outage_s"], dtype=float)
            arrays[f"{prefix}_current_outage_s"] = np.asarray(
                trace["current_contiguous_outage_s"], dtype=float
            )
            arrays[f"{prefix}_visited_points"] = np.asarray(
                trace["visited_points"], dtype=int
            )
    return results, arrays


def write_source(results: list[dict], arrays: dict[str, np.ndarray]) -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(SOURCE_STEM.with_suffix(".npz"), **arrays)
    rows = []
    summaries = []
    for item in results:
        trace = item["trace"]
        prefix = (
            f"{item['level'].lower()}_"
            f"{'aware' if item['method'] == 'A2G-visible' else 'blind'}"
        )
        positions = arrays[f"{prefix}_positions_m"]
        for step, (position, time_s, outage_s, current_outage, visited) in enumerate(
            zip(
                positions,
                arrays[f"{prefix}_time_s"],
                arrays[f"{prefix}_outage_s"],
                arrays[f"{prefix}_current_outage_s"],
                arrays[f"{prefix}_visited_points"],
            )
        ):
            rows.append(
                {
                    "level": item["level"],
                    "method": item["method"],
                    "step": step,
                    "time_s": float(time_s),
                    "x_m": float(position[0]),
                    "y_m": float(position[1]),
                    "max_contiguous_outage_s": float(outage_s),
                    "current_contiguous_outage_s": float(current_outage),
                    "visited_points": int(visited),
                }
            )
        report = item["report"]
        summaries.append(
            {
                "level": item["level"],
                "method": item["method"],
                "complete": bool(report["mission_complete"]),
                "qos_feasible": bool(report["qos_feasible"]),
                "task_time_s": report["task_time_s"],
                "transmission_delay_s": report["transmission_delay_s"],
                "contiguous_outage_s": report["outage_time_s"],
                "visited_points": report["visited_inspection_points"],
                "uploaded_mb": report["uploaded_mb"],
                "expected_upload_mb": report["expected_upload_mb"],
                "replay_device": report["replay_device"],
                "source_path": str(item["run_dir"].resolve()),
                "actor_path": str(item["checkpoint"].resolve()),
                "actor_sha256": _sha256(item["checkpoint"]),
            }
        )
    with SOURCE_STEM.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    SOURCE_STEM.with_suffix(".json").write_text(
        json.dumps(
            {
                "protocol": {
                    "fixed_order": [1, 6, 5, 2, 3, 4],
                    "outage_threshold": 0.5,
                    "current_contiguous_outage": (
                        "increments in consecutive outage slots and resets to zero "
                        "in the first non-outage slot"
                    ),
                    "map_background": "A2G SNR",
                    "trajectory_units": "m",
                },
                "records": summaries,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _legend_handles():
    return [
        Line2D(
            [0], [0], marker="o", linestyle="none", markerfacecolor="#D94B3D",
            markeredgecolor="white", markersize=5.2, label="Start",
        ),
        Line2D(
            [0], [0], marker="^", linestyle="none", markerfacecolor="#202020",
            markeredgecolor="white", markersize=5.2, label="Inspection point",
        ),
        Line2D(
            [0], [0], marker="o", linestyle="none", markerfacecolor="#E39A35",
            markeredgecolor="white", markersize=5.2, label="End",
        ),
        Line2D(
            [0], [0], color=COLORS["A2G-visible"], linewidth=1.5,
            label="A2G-aware",
        ),
        Line2D(
            [0], [0], color=COLORS["A2G-blind"], linewidth=1.5,
            linestyle=LINESTYLES["A2G-blind"], label="A2G-blind",
        ),
        Line2D(
            [0], [0], marker="x", linestyle="none", color="#6B6B6B",
            markersize=5.0, markeredgewidth=1.1,
            label="Incomplete rollout",
        ),
    ]


def _save_formats(figure, stem: Path, axis, color_axis) -> None:
    sys.path.insert(0, str(SKILL_SCRIPTS))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    figure.canvas.draw()
    require_matplotlib_panel_alignment(
        figure,
        axes=[axis],
        panel_ids=["a"],
        exclude_axes=[color_axis],
        json_out=f"{stem}.alignment.json",
        overlay_svg=f"{stem}.alignment.svg",
        require_panel_labels=False,
        strict=True,
    )
    figure.savefig(stem.with_suffix(".pdf"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    figure.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    figure.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.02,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(figure)


def write_panels(results: list[dict]) -> None:
    g2a, a2g = _map_data()
    x_grid, y_grid, _, snr = _map_layers(g2a, a2g)
    users = np.loadtxt(ROOT / "results" / "datas" / "Users_6.txt")[:, :2] * 100.0
    for panel_id, level in zip("abc", LEVELS):
        figure = plt.figure(figsize=(2.55, 2.75))
        axis = figure.add_axes((0.16, 0.25, 0.62, 0.63))
        mesh = _draw_map(
            axis,
            x_grid,
            y_grid,
            snr,
            cmap="viridis",
            vmin=float(snr.min()),
            vmax=float(snr.max()),
            title="",
            users=users,
            show_numbering=False,
        )
        axis.set_title(
            f"{level} ({PAYLOAD_TOTAL_MB[level]:.0f} MB)",
            fontsize=7.5,
            fontweight="bold",
            pad=4,
        )
        for item in results:
            if item["level"] != level:
                continue
            positions = np.asarray(item["trace"]["positions"], dtype=float) * 100.0
            method = item["method"]
            line = axis.plot(
                positions[:, 0],
                positions[:, 1],
                color=COLORS[method],
                linestyle=LINESTYLES[method],
                linewidth=1.55,
                zorder=6 if method == "A2G-visible" else 5,
            )[0]
            line.set_path_effects(
                [path_effects.withStroke(linewidth=2.8, foreground="#202020")]
            )
            if not item["complete"]:
                axis.plot(
                    positions[-1, 0],
                    positions[-1, 1],
                    marker="x",
                    markersize=6.0,
                    markeredgewidth=1.3,
                    color=COLORS[method],
                    linestyle="none",
                    zorder=8,
                )
                axis.text(
                    positions[-1, 0] - 18,
                    positions[-1, 1] + 28,
                    f"{item['report']['visited_inspection_points']}/6",
                    color="white",
                    fontsize=6.2,
                    ha="right",
                    va="bottom",
                    path_effects=[
                        path_effects.withStroke(linewidth=1.5, foreground="#202020")
                    ],
                    zorder=9,
                )
        axis.tick_params(axis="both", labelsize=6.8)
        axis.xaxis.label.set_size(7.0)
        axis.yaxis.label.set_size(7.0)
        color_axis = figure.add_axes((0.84, 0.25, 0.035, 0.63))
        colorbar = figure.colorbar(mesh, cax=color_axis)
        colorbar.set_label("A2G SNR (dB)", fontsize=7.0)
        colorbar.set_ticks([-5, 0, 5, 10, 15, 20, 25])
        colorbar.ax.tick_params(labelsize=6.5)
        figure.legend(
            handles=_legend_handles(),
            loc="lower center",
            bbox_to_anchor=(0.51, 0.015),
            ncol=3,
            fontsize=6.4,
            columnspacing=0.8,
            handlelength=1.8,
            handletextpad=0.45,
            borderpad=0.3,
            labelspacing=0.28,
            frameon=False,
        )
        _save_formats(
            figure,
            P3_FIGURE_DIR / f"{OUTPUT_STEM.name}_panel_{panel_id}",
            axis,
            color_axis,
        )


def main() -> None:
    QA_DIR.mkdir(parents=True, exist_ok=True)
    results, arrays = replay_all()
    write_source(results, arrays)
    write_panels(results)
    print(SOURCE_STEM.with_suffix(".json"))
    for panel_id in "abc":
        print(P3_FIGURE_DIR / f"{OUTPUT_STEM.name}_panel_{panel_id}.pdf")


if __name__ == "__main__":
    main()
