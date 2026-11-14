"""Replay four trajectory executors across all payload levels and draw maps."""

from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from discrete_actions import DiscreteActionCodec
from evaluate_d3qn import (
    _build_world as build_d3qn_world,
    load_d3qn_checkpoint,
    load_d3qn_run_context,
)
from evaluate_policy import load_run_context, rollout
from plot_v4_trajectory_figures import (
    _draw_map,
    _draw_path,
    _map_data,
    _map_layers,
)
from plot_v5_e4_trajectories import _evaluate_on_recorded_device
from trajectory_legend import (
    add_internal_legend,
    align_colorbar_to_axis,
    simplify_colorbar_ticks,
)


ROOT = Path(__file__).resolve().parent
FIGURE_DIR = ROOT.parent / "P3" / "figs" / "v5"
SOURCE_DIR = FIGURE_DIR / "source_data"
QA_DIR = FIGURE_DIR / "qa"
SKILL_SCRIPTS = Path.home() / ".codex" / "skills" / "nature-figure" / "scripts"
ACTOR_BUNDLE = ROOT / "sync_bundles" / "v4_eval_best_actors"
MEDIUM_HIGH_BUNDLE = ROOT / "sync_bundles" / "v5_e1_medium_high_actors"
LOW_TRAJECTORY_SOURCE = FIGURE_DIR / "source_data" / "e1_low_trajectories_v5.npz"
LOW_METRICS_SOURCE = FIGURE_DIR / "source_data" / "e1_low_trajectory_metrics_v5.json"

LEVELS = ("Low", "Medium", "High")
A_STAR_METHOD = "Dual-map A*"
METHODS = ("CA-TD3", "D3QN", "PPO", A_STAR_METHOD)
LEVEL_COLORS = {
    "Low": "#202020",
    "Medium": "#C2185B",
    "High": "#1565C0",
}
LEVEL_LINESTYLES = {
    "Low": "-",
    "Medium": (0, (4.0, 2.0)),
    "High": (0, (1.5, 1.4)),
}
PAYLOAD_TOTAL_MB = {"Low": 450, "Medium": 900, "High": 1500}
RUNS = {
    "CA-TD3": {
        "Low": "journal_results/v4-qos/w50-50/qos6/A/low/td3_bc",
        "Medium": "journal_results/v4-models/load/medium/A/medium/td3_bc",
        "High": "journal_results/v4-models/load/high/A/high/td3_bc",
    },
    "D3QN": {
        "Low": "journal_results/v4-models/layer1/d3qn/low/d3qn",
        "Medium": "journal_results/v5-win/E1/d3qn/medium/d3qn",
        "High": "journal_results/v5-win/E1/d3qn/high/d3qn",
    },
    "PPO": {
        "Low": "journal_results/v5-win/E1/ppo/A/low/ppo",
        "Medium": "journal_results/v5-win/E1/ppo/A/medium/ppo",
        "High": "journal_results/v5-win/E1/ppo/A/high/ppo",
    },
    A_STAR_METHOD: {
        "Low": "journal_results/v5-deterministic/dual-map-astar/low",
        "Medium": "journal_results/v5-deterministic/dual-map-astar/medium",
        "High": "journal_results/v5-deterministic/dual-map-astar/high",
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


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _actor_for(method, level, run):
    if method == "CA-TD3":
        return ACTOR_BUNDLE / {
            "Low": "w50-50_qos6_actor.pth",
            "Medium": "medium_qos6_actor.pth",
            "High": "high_qos6_actor.pth",
        }[level]
    if method == "D3QN":
        if level in {"Medium", "High"}:
            return (
                MEDIUM_HIGH_BUNDLE
                / f"d3qn_{level.lower()}_actor_eval_best.pt"
            )
        return run / "models" / "d3qn_eval_best.pt"
    if method == "PPO":
        if level in {"Medium", "High"}:
            return (
                MEDIUM_HIGH_BUNDLE
                / f"ppo_{level.lower()}_actor_eval_best.pth"
            )
        local = run / "models" / "ppo_ppo_actoreval_best.pth"
        if local.is_file():
            return local
        if level == "Low":
            return ROOT / "sync_bundles/v5_e1_low_trajectories/ppo_actor_eval_best.pth"
        raise FileNotFoundError(
            f"missing {level} PPO eval-best actor: {local}"
        )
    if method == A_STAR_METHOD:
        return None
    raise ValueError(method)


def _selected_eval(run):
    payload = json.loads(
        (run / "eval_history.json").read_text(encoding="utf-8")
    )
    selected = payload.get("eval_best") or payload.get("best") or {}
    if not selected:
        raise ValueError(f"{run}: no eval_best/best record")
    return selected


def _load_archive_trace(method, level):
    source = (
        LOW_TRAJECTORY_SOURCE
        if level == "Low"
        else MEDIUM_HIGH_BUNDLE
        / "trajectory"
        / f"{method.lower().replace('-', '_')}_{level.lower()}_trajectory.npz"
    )
    with np.load(source, allow_pickle=True) as payload:
        if "positions" in payload:
            positions = np.asarray(payload["positions"], dtype=float)
        else:
            positions = np.column_stack(
                (
                    np.asarray(payload[f"{method}_x_km"], dtype=float),
                    np.asarray(payload[f"{method}_y_km"], dtype=float),
                )
            )
        key_prefix = f"{method}_" if level == "Low" else ""
        trace = {
            "positions": positions,
            "time_s": np.asarray(
                payload[
                    f"{key_prefix}time_s" if key_prefix else "time_s"
                ],
                dtype=float,
            ),
            "outage_s": np.asarray(
                payload[
                    f"{key_prefix}outage_s" if key_prefix else "outage_s"
                ],
                dtype=float,
            ),
            "visited_points": np.asarray(
                payload[
                    f"{key_prefix}visited_points"
                    if key_prefix
                    else "visited_points"
                ],
                dtype=int,
            ),
        }
        for key in (
            "pending_mb",
            "total_outage_s",
            "current_contiguous_outage_s",
            "actions",
        ):
            if key in payload:
                trace[key] = np.asarray(payload[key])
    return source, trace


def _archive_metrics(method, level):
    if level == "Low":
        payload = json.loads(
            LOW_METRICS_SOURCE.read_text(encoding="utf-8")
        )
        metrics = dict(payload["methods"][method])
        metrics["source"] = LOW_METRICS_SOURCE
    else:
        source = (
            MEDIUM_HIGH_BUNDLE
            / "metrics"
            / f"{method.lower().replace('-', '_')}_{level.lower()}_metrics.json"
        )
        metrics = json.loads(source.read_text(encoding="utf-8"))
        metrics["source"] = source
    metrics.setdefault(
        "mission_complete",
        metrics.get("success", False),
    )
    metrics.setdefault(
        "qos_feasible",
        metrics.get("status") == "feasible"
        or (
            metrics.get("mission_complete", False)
            and not metrics.get("qos_violation", False)
        ),
    )
    return metrics


def _evaluate_d3qn_recorded(run, actor):
    context = load_d3qn_run_context(run)
    context = replace(
        context,
        total_steps=int(context.config.get("total_steps", 1000)),
    )
    network, metadata = load_d3qn_checkpoint(
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
            action = int(network(tensor).argmax(dim=1).item())
        return codec.decode(action)

    report, trace = rollout(context, build_d3qn_world(context), act)
    report["replay_device"] = device_name
    report["checkpoint_metadata"] = metadata
    return report, trace


def evaluate_all():
    results = {}
    for method in METHODS:
        results[method] = {}
        for level in LEVELS:
            run = ROOT / RUNS[method][level]
            if method == A_STAR_METHOD:
                metrics = json.loads(
                    (run / "metrics.json").read_text(encoding="utf-8")
                )
                report = dict(metrics["world_replay"])
                if not report.get("success", False):
                    raise ValueError(
                        f"{level} {method} did not complete the V5 protocol"
                    )
                with np.load(run / "trajectory.npz") as payload:
                    trace = {key: payload[key] for key in payload.files}
                actor = None
                trajectory_source = run / "trajectory.npz"
                metrics_source = run / "metrics.json"
                replay_device = "deterministic search"
                matches_eval_history = True
                selected = {
                    "episode": 0,
                    "task_time_s": report["task_time_s"],
                    "transmission_delay_s": report["transmission_delay_s"],
                    "outage_time_s": report["outage_time_s"],
                    "visited_inspection_points": report[
                        "visited_inspection_points"
                    ],
                    "uploaded_mb": report["uploaded_mb"],
                }
                results[method][level] = {
                    "run": run,
                    "actor": actor,
                    "report": report,
                    "trace": trace,
                    "selected": selected,
                    "trajectory_source": trajectory_source,
                    "metrics_source": metrics_source,
                    "replay_device": replay_device,
                    "matches_eval_history": matches_eval_history,
                }
                continue
            actor = _actor_for(method, level, run)
            if not actor.is_file():
                raise FileNotFoundError(
                    f"missing {level} {method} eval-best actor: {actor}"
                )
            selected = _selected_eval(run)
            if method == "CA-TD3":
                context = load_run_context(run)
                with contextlib.redirect_stdout(io.StringIO()):
                    report, trace = _evaluate_on_recorded_device(
                        context, actor
                    )
                trajectory_source = None
                metrics_source = run / "eval_history.json"
                replay_device = report.get("replay_device", "cpu")
                matches_eval_history = True
            else:
                trajectory_source, trace = _load_archive_trace(method, level)
                report = _archive_metrics(method, level)
                metrics_source = report.pop("source")
                replay_device = "source_archive"
                expected = {
                    "task_time_s": selected.get("task_time_s"),
                    "transmission_delay_s": selected.get(
                        "transmission_delay_s"
                    ),
                    "outage_time_s": selected.get("outage_time_s"),
                    "visited_inspection_points": selected.get(
                        "visited_inspection_points"
                    ),
                    "uploaded_mb": selected.get("uploaded_mb"),
                }
                mismatches = {
                    key: (report.get(key), value)
                    for key, value in expected.items()
                    if value is not None
                    and (
                        report.get(key) is None
                        or not np.isclose(
                            float(report[key]), float(value), atol=1e-9
                        )
                    )
                }
                matches_eval_history = not mismatches
                if mismatches:
                    raise ValueError(
                        f"{method} {level} archive mismatch: {mismatches}"
                    )
            results[method][level] = {
                "run": run,
                "actor": actor,
                "report": report,
                "trace": trace,
                "selected": selected,
                "trajectory_source": trajectory_source,
                "metrics_source": metrics_source,
                "replay_device": replay_device,
                "matches_eval_history": matches_eval_history,
            }
    return results


def write_source(results):
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    arrays = {}
    rows = []
    summaries = []
    selected_rows = []
    for method in METHODS:
        for level in LEVELS:
            item = results[method][level]
            trace = item["trace"]
            key = (
                method.lower()
                .replace("*", "star")
                .replace("-", "_")
                .replace(" ", "_")
                + f"_{level.lower()}"
            )
            positions = np.asarray(trace["positions"], dtype=float) * 100.0
            times = np.asarray(trace["time_s"], dtype=float)
            outage = np.asarray(trace["outage_s"], dtype=float)
            visited = np.asarray(trace["visited_points"], dtype=int)
            current = (
                np.asarray(
                    trace["current_contiguous_outage_s"], dtype=float
                )
                if "current_contiguous_outage_s" in trace
                else None
            )
            pending = (
                np.asarray(trace["pending_mb"], dtype=float)
                if "pending_mb" in trace
                else None
            )
            total_outage = (
                np.asarray(trace["total_outage_s"], dtype=float)
                if "total_outage_s" in trace
                else None
            )
            arrays[f"{key}_positions_m"] = positions
            arrays[f"{key}_time_s"] = times
            arrays[f"{key}_outage_s"] = outage
            arrays[f"{key}_visited_points"] = visited
            if current is not None:
                arrays[f"{key}_current_contiguous_outage_s"] = current
            if pending is not None:
                arrays[f"{key}_pending_mb"] = pending
            if total_outage is not None:
                arrays[f"{key}_total_outage_s"] = total_outage
            for step, (point, time_s, outage_s, visits) in enumerate(
                zip(positions, times, outage, visited)
            ):
                rows.append(
                    {
                        "method": method,
                        "level": level,
                        "step": step,
                        "time_s": float(time_s),
                        "x_m": float(point[0]),
                        "y_m": float(point[1]),
                        "max_contiguous_outage_s": float(outage_s),
                        "current_contiguous_outage_s": (
                            float(current[step])
                            if current is not None
                            else None
                        ),
                        "visited_points": int(visits),
                        "pending_mb": (
                            float(pending[step])
                            if pending is not None
                            else None
                        ),
                    }
                )
            report = item["report"]
            selected = item["selected"]
            task_time = float(report["task_time_s"])
            delay = float(report["transmission_delay_s"])
            expected_upload = float(
                sum(
                    {
                        "Low": [90, 90, 60, 60, 90, 60],
                        "Medium": [180, 180, 180, 120, 120, 120],
                        "High": [210, 290, 290, 290, 210, 210],
                    }[level]
                )
            )
            complete = (
                int(report["visited_inspection_points"])
                >= int(report.get("expected_inspection_points", 6))
                and float(report["uploaded_mb"])
                >= expected_upload - 1e-7
            )
            objective = task_time + delay if complete else None
            feasible = bool(complete and float(report["outage_time_s"]) <= 6.0)
            actor = item["actor"]
            actor_sha = _sha256(actor) if actor is not None else None
            trajectory_source = item["trajectory_source"]
            trajectory_sha = (
                _sha256(trajectory_source)
                if trajectory_source is not None
                else None
            )
            summaries.append(
                {
                    "method": method,
                    "level": level,
                    "task_time_s": task_time,
                    "transmission_delay_s": delay,
                    "objective_s": objective,
                    "contiguous_outage_s": float(report["outage_time_s"]),
                    "visited_points": int(
                        report["visited_inspection_points"]
                    ),
                    "uploaded_mb": float(report["uploaded_mb"]),
                    "complete": complete,
                    "feasible": feasible,
                    "mission_complete_record": report.get(
                        "mission_complete"
                    ),
                    "eval_best_episode": int(selected.get("episode", 0)),
                    "run": str(item["run"].resolve()),
                    "actor": str(actor.resolve()) if actor is not None else None,
                    "actor_sha256": actor_sha,
                    "trajectory_source": (
                        str(trajectory_source.resolve())
                        if trajectory_source is not None
                        else "Mac actor replay"
                    ),
                    "trajectory_sha256": trajectory_sha,
                    "metrics_source": str(
                        item["metrics_source"].resolve()
                    ),
                    "replay_device": item["replay_device"],
                    "matches_eval_history": item["matches_eval_history"],
                }
            )
            selected_rows.append(
                {
                    "level": level,
                    "method": method,
                    "eval_best_episode": int(selected.get("episode", 0)),
                    "task_time_s": task_time,
                    "transmission_delay_s": delay,
                    "objective_s": objective,
                    "contiguous_outage_s": float(report["outage_time_s"]),
                    "outage_limit_s": 6.0,
                    "visited_points": int(
                        report["visited_inspection_points"]
                    ),
                    "expected_visits": 6,
                    "uploaded_mb": float(report["uploaded_mb"]),
                    "expected_upload_mb": expected_upload,
                    "complete": complete,
                    "feasible": feasible,
                    "source_path": str(item["run"].resolve()),
                    "trajectory_source": (
                        str(trajectory_source.resolve())
                        if trajectory_source is not None
                        else "Mac actor replay"
                    ),
                    "actor_sha256": actor_sha,
                    "trajectory_sha256": trajectory_sha,
                    "matches_eval_history": item["matches_eval_history"],
                }
            )
    np.savez_compressed(SOURCE_DIR / "e1_payload_trajectories_v5.npz", **arrays)
    with (SOURCE_DIR / "e1_payload_trajectories_v5.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    with (SOURCE_DIR / "e1_trajectory_results_v5.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(selected_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(selected_rows)
    (SOURCE_DIR / "e1_payload_trajectories_v5.json").write_text(
        json.dumps(
            {
                "protocol": {
                    "methods": list(METHODS),
                    "levels": list(LEVELS),
                    "policy_mode": (
                        "deterministic eval-best policies, no MapGuide; "
                        "CA-TD3 is replayed locally, D3QN/PPO use validated "
                        "archived trajectories, and Dual-map A* uses its "
                        "deterministic World-replayed search artifacts"
                    ),
                    "map_backgrounds": {
                        "A2G": "SNR (dB)",
                        "G2A": "coverage probability",
                    },
                },
                "records": summaries,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _legend_handles(method):
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="#D94B3D",
            markeredgewidth=0.0,
            markersize=4.8,
            label="Start",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            linestyle="none",
            markerfacecolor="#202020",
            markeredgewidth=0.0,
            markersize=4.8,
            label="Inspection point",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="#E39A35",
            markeredgewidth=0.0,
            markersize=4.8,
            label="End",
        ),
    ]
    handles.extend(
        Line2D(
            [0],
            [0],
            color=LEVEL_COLORS[level],
            linestyle=LEVEL_LINESTYLES[level],
            linewidth=1.4,
            label=f"{level} ({PAYLOAD_TOTAL_MB[level]} MB)",
        )
        for level in LEVELS
    )
    return handles


def _draw_level_path(axis, trace, method, level):
    _draw_path(
        axis,
        trace,
        LEVEL_COLORS[level],
        label=level,
        linewidth=1.25,
    )


def _save_formats(figure, output, axis, color_axis):
    figure.canvas.draw()
    sys.path.insert(0, str(SKILL_SCRIPTS))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    require_matplotlib_panel_alignment(
        figure,
        axes=[axis],
        panel_ids=["a"],
        exclude_axes=[color_axis],
        json_out=f"{output}.alignment.json",
        overlay_svg=f"{output}.alignment.svg",
        require_panel_labels=False,
        strict=True,
    )
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=600, bbox_inches="tight")
    figure.savefig(
        output.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(figure)


def plot(results):
    g2a, a2g = _map_data()
    x_grid, y_grid, coverage, snr = _map_layers(g2a, a2g)
    users = np.loadtxt(ROOT / "results/datas/Users_6.txt")[:, :2] * 100.0
    for map_name, values, color_label in (
        ("a2g", snr, "A2G SNR (dB)"),
        ("g2a", coverage, "G2A coverage probability"),
    ):
        for panel, method in zip("abcd", METHODS):
            figure = plt.figure(figsize=(2.35, 2.85))
            grid = figure.add_gridspec(
                1, 2, width_ratios=[1.0, 0.055], wspace=0.12
            )
            axis = figure.add_subplot(grid[0, 0])
            color_axis = figure.add_subplot(grid[0, 1])
            mesh = _draw_map(
                axis,
                x_grid,
                y_grid,
                values,
                cmap="viridis",
                vmin=0.0 if map_name == "g2a" else float(np.min(snr)),
                vmax=1.0 if map_name == "g2a" else float(np.max(snr)),
                title="",
                users=users,
                show_numbering=False,
            )
            axis.set_xlim(0.0, 2000.0)
            axis.set_ylim(0.0, 2000.0)
            axis.set_aspect("equal")
            axis.set_title(
                method,
                fontsize=7.3,
                fontweight="bold",
                pad=4,
            )
            axis.set_xticks([0, 1000, 2000])
            axis.set_yticks([0, 1000, 2000])
            axis.tick_params(axis="both", labelsize=5.4)
            axis.xaxis.label.set_size(6.0)
            axis.yaxis.label.set_size(6.0)
            for level in LEVELS:
                trace = results[method][level]["trace"]
                _draw_level_path(axis, trace, method, level)
            colorbar = figure.colorbar(mesh, cax=color_axis)
            colorbar.set_label(color_label)
            colorbar.ax.tick_params(labelsize=5.4)
            colorbar.ax.yaxis.label.set_size(6.0)
            simplify_colorbar_ticks(
                colorbar,
                format_string="%.0f" if map_name == "a2g" else "%.1f",
                nbins=6,
            )
            figure.subplots_adjust(
                left=0.17,
                right=0.90,
                top=0.90,
                bottom=0.12,
            )
            add_internal_legend(
                axis,
                _legend_handles(method),
                ncol=1,
                fontsize=5.0,
            )
            align_colorbar_to_axis(axis, color_axis)
            _save_formats(
                figure,
                FIGURE_DIR
                / f"e1_{map_name}_trajectories_v5_panel_{panel}",
                axis,
                color_axis,
            )


def main():
    QA_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    results = evaluate_all()
    write_source(results)
    plot(results)
    print(SOURCE_DIR / "e1_payload_trajectories_v5.json")
    print(FIGURE_DIR / "e1_a2g_trajectories_v5_panel_a.pdf")
    print(FIGURE_DIR / "e1_g2a_trajectories_v5_panel_a.pdf")


if __name__ == "__main__":
    main()
