"""Replay six final inspection orders with a shared CA-TD3 actor."""

from __future__ import annotations

import concurrent.futures
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
import numpy as np
import torch
from matplotlib.lines import Line2D

from evaluate_policy import _build_world, load_actor, load_run_context, rollout
from fixed_radio_maps import FixedRadioMaps
from mission_preflight import run_preflight
from plot_v5_e4_trajectories import _evaluate_on_recorded_device
from run_v5_order_replay import ActorMapGuideController, _context
from sequence_optimizers import yang_order
from sequence_runner import load_planning_nodes
from training_config import load_fixed_payloads


ROOT = Path(__file__).resolve().parent
P3 = ROOT.parent / "P3"
FIGURE_DIR = P3 / "figs" / "v5"
SOURCE_DIR = FIGURE_DIR / "source_data"
QA_DIR = FIGURE_DIR / "qa"
SKILL_SCRIPTS = Path.home() / ".codex" / "skills" / "nature-figure" / "scripts"
ACTOR_BY_LEVEL = {
    "low": ROOT / "sync_bundles" / "v4_eval_best_actors" / "w50-50_qos6_actor.pth",
    "medium": ROOT / "sync_bundles" / "v4_eval_best_actors" / "medium_qos6_actor.pth",
    "high": ROOT / "sync_bundles" / "v4_eval_best_actors" / "high_qos6_actor.pth",
}
OUTPUT_ROOT = ROOT / "journal_results" / "v5-mac" / "E2-final"
ORDER_SOURCE = SOURCE_DIR / "e2_sequence_convergence_v5.json"

LEVELS = ("Low", "Medium", "High")
METHODS = (
    "Proposed PRIGA",
    "Yang-style",
    "EEGA-style GA",
    "DRL-SA (G2A-aware SA)",
    "SA-D3QN (distance-only SA)",
    "EEAP-PSO",
)
DISPLAY_METHODS = (
    "Proposed PRIGA",
    "Yang-style",
    "EEGA-style GA",
    "Distance-TSP+CA-TD3",
    "SA-D3QN (distance-only SA)",
    "EEAP-PSO",
)
METHOD_TO_SOURCE = {
    "Proposed PRIGA": "PRIGA (proposed)",
    "EEGA-style GA": "EEGA-style GA (Algorithm in EEGA study)",
    "DRL-SA (G2A-aware SA)": (
        "DRL-SA sequence (Algorithm in DRL-SA study)"
    ),
    "SA-D3QN (distance-only SA)": (
        "SA-D3QN sequence (Algorithm in SA-D3QN study)"
    ),
    "EEAP-PSO": (
        "EEAP-PSO (Algorithm in MCO-A* study; constraints removed)"
    ),
}
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
DISTANCE_TSP_RUNS = {
    "Low": ROOT / "journal_results/v4-baselines/distance-tsp/A/low/td3_bc",
    "Medium": (
        ROOT
        / "journal_results/v5-mac/E5/distance_tsp_td3/baseline/medium/td3_bc"
    ),
    "High": (
        ROOT
        / "journal_results/v5-mac/E5/distance_tsp_td3/baseline/high/td3_bc"
    ),
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


def load_orders():
    payload = json.loads(ORDER_SOURCE.read_text(encoding="utf-8"))
    selected = {
        (row["level"], row["method"]): row["incumbent_order"]
        for row in payload["selected"]
    }
    nodes = load_planning_nodes()
    maps = FixedRadioMaps(ROOT / "results" / "datas")
    orders = {}
    for level in LEVELS:
        orders[level] = {}
        for method in METHODS:
            if method == "Yang-style":
                order, _ = yang_order(
                    load_fixed_payloads(level.lower()),
                    nodes_km=nodes,
                    maps=maps,
                )
            else:
                order = [
                    int(node)
                    for node in selected[
                        (level, METHOD_TO_SOURCE[method])
                    ].split("-")
                ]
            if sorted(order) != [1, 2, 3, 4, 5, 6]:
                raise ValueError(f"invalid order for {method} {level}")
            orders[level][method] = order
    return orders


def _fallback(run_dir, level, order):
    report = run_preflight(
        level.lower(),
        output_dir=run_dir,
        order=order,
        outage_max_s=6.0,
        total_steps=2000,
        a2g_blind=False,
        outage_probability_threshold=0.5,
        qos_enforcement="continue",
    )
    path = run_dir / level.lower() / "preflight_trace.npz"
    with np.load(path) as payload:
        raw = np.asarray(payload["trace"], dtype=float)
    trace = {
        "positions": raw[:, :2],
        "time_s": raw[:, 2],
        "outage_s": raw[:, 3],
        "total_outage_s": raw[:, 4],
        "current_contiguous_outage_s": raw[:, 5],
        "pending_mb": raw[:, 6],
        "visited_points": raw[:, 7].astype(int),
        "actions": raw[:, 8:10],
    }
    return report, trace


def _replay(task):
    level, method, order, output_root = task
    level_key = level.lower()
    method_id = (
        method.lower()
        .replace(" ", "-")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "-")
    )
    run_dir = Path(output_root) / method_id / level_key
    run_dir.mkdir(parents=True, exist_ok=True)
    context = _context(level_key, order, run_dir)
    actor = load_actor(
        ACTOR_BY_LEVEL[level_key],
        context.state_dim,
        context.net_width,
        algorithm="td3_bc",
    )
    recorded_device = context.config.get("device")
    device_name = (
        "cpu"
        if recorded_device in {None, "", "none"}
        else str(recorded_device).lower()
    )
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(f"{level} {method}: MPS replay is unavailable")
    device = torch.device(device_name)
    actor = actor.to(device).eval()
    with contextlib.redirect_stdout(io.StringIO()):
        world = _build_world(context)
    if method == "Proposed PRIGA":
        def act(state):
            with torch.no_grad():
                action = actor(
                    torch.as_tensor(
                        state, dtype=torch.float32, device=device
                    ).unsqueeze(0)
                )[0]
            return action.cpu().numpy()

        report, trace = rollout(context, world, act)
        control = "Actor only"
    else:
        controller = ActorMapGuideController(
            world, actor, stall_steps=100
        )
        report, trace = rollout(context, world, controller)
        control = (
            "Actor + MapGuide assistance"
            if controller.message_guide_active
            else "Actor only"
        )
        if not report["mission_complete"]:
            report, trace = _fallback(run_dir, level, order)
            report["task_time_s"] = float(report.pop("time_s"))
            report["outage_time_s"] = float(report.pop("outage_s"))
            report["total_outage_time_s"] = float(
                report.pop("total_outage_s")
            )
            report["visited_inspection_points"] = int(
                report.pop("visited_points")
            )
            report["expected_inspection_points"] = len(order)
            control = "Full MapGuide restart"
    complete = bool(
        report["mission_complete"]
        and int(report["visited_inspection_points"]) >= 6
        and float(report["uploaded_mb"])
        >= float(report["expected_upload_mb"]) - 1e-7
    )
    feasible = bool(complete and float(report["outage_time_s"]) <= 6.0)
    report["complete"] = complete
    report["feasible"] = feasible
    report["objective_s"] = (
        float(report["task_time_s"]) + float(report["transmission_delay_s"])
        if complete
        else None
    )
    np.savez(run_dir / "trajectory.npz", **trace)
    metrics = {
        **report,
        "level": level,
        "method": method,
        "inspection_order": list(order),
        "control": control,
        "complete": complete,
        "feasible": feasible,
        "source_run": str(run_dir),
        "shared_actor": str(ACTOR_BY_LEVEL[level_key].resolve()),
        "shared_actor_sha256": _sha256(ACTOR_BY_LEVEL[level_key]),
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return metrics


def _distance_tsp_records():
    records = []
    for level in LEVELS:
        run_dir = DISTANCE_TSP_RUNS[level]
        actor = run_dir / "models" / "td3_bc_td3_actoreval_best.pth"
        context = load_run_context(run_dir)
        with contextlib.redirect_stdout(io.StringIO()):
            report, trace = _evaluate_on_recorded_device(context, actor)
        eval_payload = json.loads(
            (run_dir / "eval_history.json").read_text(encoding="utf-8")
        )
        selected = (
            eval_payload.get("eval_best")
            or eval_payload.get("best")
            or {}
        )
        expected = {
            "task_time_s": selected.get("task_time_s"),
            "transmission_delay_s": selected.get("transmission_delay_s"),
            "outage_time_s": selected.get("outage_time_s"),
            "visited_inspection_points": selected.get(
                "visited_inspection_points"
            ),
            "uploaded_mb": selected.get("uploaded_mb"),
        }
        for key, value in expected.items():
            if value is None:
                continue
            if not np.isclose(float(report[key]), float(value), atol=1e-9):
                raise ValueError(
                    f"Distance-TSP+CA-TD3 {level} {key} mismatch: "
                    f"{report[key]} vs {value}"
                )
        complete = bool(
            int(report["visited_inspection_points"]) >= 6
            and float(report["uploaded_mb"])
            >= float(report["expected_upload_mb"]) - 1e-7
        )
        feasible = bool(
            complete and float(report["outage_time_s"]) <= 6.0
        )
        output_dir = (
            OUTPUT_ROOT
            / "distance-tsp-ca-td3"
            / level.lower()
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez(output_dir / "trajectory.npz", **trace)
        metrics = {
            **report,
            "level": level,
            "method": "Distance-TSP+CA-TD3",
            "inspection_order": list(context.order),
            "control": "Actor only",
            "complete": complete,
            "feasible": feasible,
            "objective_s": (
                float(report["task_time_s"])
                + float(report["transmission_delay_s"])
                if complete
                else None
            ),
            "source_run": str(output_dir),
            "source_actor": str(actor.resolve()),
            "source_actor_sha256": _sha256(actor),
            "source_pipeline": str(run_dir.resolve()),
            "source_eval_history": str(
                (run_dir / "eval_history.json").resolve()
            ),
            "source_commit": "v4-baselines" if level == "Low" else "v5-mac/E5",
            "display_note": (
                "Additional complete-pipeline spatial reference; not added "
                "to the 18-row order comparison table."
            ),
        }
        (output_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        records.append(metrics)
    return records


def _save_source(records):
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    arrays = {}
    summaries = []
    for record in records:
        method_id = record["method"]
        level = record["level"]
        run_dir = Path(record["source_run"])
        with np.load(run_dir / "trajectory.npz") as payload:
            trace = {key: payload[key] for key in payload.files}
        key = (
            method_id.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_")
            .replace("-", "_")
            + f"_{level.lower()}"
        )
        arrays[f"{key}_positions_m"] = np.asarray(trace["positions"]) * 100.0
        arrays[f"{key}_time_s"] = np.asarray(trace["time_s"])
        arrays[f"{key}_current_contiguous_outage_s"] = np.asarray(
            trace["current_contiguous_outage_s"]
        )
        positions = np.asarray(trace["positions"], dtype=float) * 100.0
        for step, point in enumerate(positions):
            rows.append(
                {
                    "method": method_id,
                    "level": level,
                    "step": step,
                    "x_m": float(point[0]),
                    "y_m": float(point[1]),
                    "time_s": float(trace["time_s"][step]),
                    "current_contiguous_outage_s": float(
                        trace["current_contiguous_outage_s"][step]
                    ),
                }
            )
        summaries.append(record)
    np.savez_compressed(
        SOURCE_DIR / "e2_inspection_order_trajectories_v5.npz",
        **arrays,
    )
    with (SOURCE_DIR / "e2_inspection_order_trajectories_v5.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    (SOURCE_DIR / "e2_inspection_order_trajectories_v5.json").write_text(
        json.dumps(
            {
                "protocol": {
                    "protocol": (
                        "shared CA-TD3 actor; MapGuide assistance after "
                        "stagnation; full restart fallback if still incomplete"
                    ),
                    "qos_limit_s": 6.0,
                    "seed": 1,
                },
                "records": summaries,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _save_display_source(records):
    rows = []
    arrays = {}
    for record in records:
        method = record["method"]
        level = record["level"]
        run_dir = Path(record["source_run"])
        with np.load(run_dir / "trajectory.npz") as payload:
            positions = np.asarray(payload["positions"], dtype=float) * 100.0
            times = np.asarray(payload["time_s"], dtype=float)
            current = np.asarray(
                payload["current_contiguous_outage_s"], dtype=float
            )
        key = (
            method.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_")
            .replace("-", "_")
            + f"_{level.lower()}"
        )
        arrays[f"{key}_positions_m"] = positions
        arrays[f"{key}_time_s"] = times
        arrays[f"{key}_current_contiguous_outage_s"] = current
        for step, point in enumerate(positions):
            rows.append(
                {
                    "panel": "abcdef"[DISPLAY_METHODS.index(method)],
                    "method": method,
                    "level": level,
                    "step": step,
                    "x_m": float(point[0]),
                    "y_m": float(point[1]),
                    "time_s": float(times[step]),
                    "current_contiguous_outage_s": float(current[step]),
                    "complete": record["complete"],
                    "feasible": record["feasible"],
                }
            )
    files = (
        "e2_trajectory_panels_v5.csv",
        "e2_trajectory_panels_v5.json",
        "e2_trajectory_panels_v5.npz",
    )
    np.savez_compressed(SOURCE_DIR / files[2], **arrays)
    with (SOURCE_DIR / files[0]).open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    (SOURCE_DIR / files[1]).write_text(
        json.dumps(
            {
                "protocol": {
                    "display_methods": list(DISPLAY_METHODS),
                    "table_methods": list(METHODS),
                    "trajectory_set_note": (
                        "DRL-SA is retained in the 18-row table but omitted "
                        "from the spatial panels because it selected the same "
                        "order and shared replay trajectory as SA-D3QN. "
                        "Distance-TSP+CA-TD3 is an additional A2G-blind "
                        "complete-pipeline spatial reference and is not added "
                        "to the order table."
                    ),
                    "qos_limit_s": 6.0,
                },
                "records": [
                    {
                        "method": record["method"],
                        "level": record["level"],
                        "complete": record["complete"],
                        "feasible": record["feasible"],
                        "inspection_order": record["inspection_order"],
                        "control": record.get("control"),
                        "source_run": record["source_run"],
                        "source_pipeline": record.get("source_pipeline"),
                        "source_actor_sha256": record.get(
                            "source_actor_sha256",
                            record.get("shared_actor_sha256"),
                        ),
                    }
                    for record in records
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _legend_handles():
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="#D94B3D",
            markeredgecolor="white",
            markersize=4.8,
            label="Start",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            linestyle="none",
            markerfacecolor="#202020",
            markeredgecolor="white",
            markersize=4.8,
            label="Inspection point",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="#E39A35",
            markeredgecolor="white",
            markersize=4.8,
            label="End",
        ),
        *[
            Line2D(
                [0],
                [0],
                color=LEVEL_COLORS[level],
                linestyle=LEVEL_LINESTYLES[level],
                linewidth=1.25,
                label=f"{level} ({PAYLOAD_TOTAL_MB[level]} MB)",
            )
            for level in LEVELS
        ],
    ]


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


def _plot(records):
    users = np.loadtxt(ROOT / "results/datas/Users_6.txt")[:, :2] * 100.0
    by_method = {method: {} for method in DISPLAY_METHODS}
    for record in records:
        if record["method"] not in by_method:
            continue
        by_method[record["method"]][record["level"]] = record
    for panel, method in zip("abcdef", DISPLAY_METHODS):
        figure, axis = plt.subplots(1, 1, figsize=(2.65, 2.95))
        axis.scatter(
            users[:, 0],
            users[:, 1],
            marker="^",
            s=22,
            color="#202020",
            edgecolor="white",
            linewidth=0.4,
            zorder=5,
        )
        axis.scatter(
            [262],
            [265],
            marker="o",
            s=38,
            color="#D94B3D",
            edgecolor="white",
            zorder=6,
        )
        axis.scatter(
            [1895],
            [1112],
            marker="o",
            s=40,
            color="#E39A35",
            edgecolor="white",
            zorder=6,
        )
        for level in LEVELS:
            record = by_method[method][level]
            with np.load(
                Path(record["source_run"]) / "trajectory.npz"
            ) as payload:
                positions = np.asarray(payload["positions"], dtype=float) * 100.0
            axis.plot(
                positions[:, 0],
                positions[:, 1],
                color=LEVEL_COLORS[level],
                linestyle=LEVEL_LINESTYLES[level],
                linewidth=1.25,
                zorder=4,
            )
            if not record["complete"]:
                axis.plot(
                    positions[-1, 0],
                    positions[-1, 1],
                    marker="x",
                    markersize=5,
                    markeredgewidth=1.1,
                    color=LEVEL_COLORS[level],
                    linestyle="none",
                    zorder=7,
                )
        axis.set_xlim(0, 2000)
        axis.set_ylim(0, 2000)
        axis.set_aspect("equal")
        axis.set_xticks([0, 1000, 2000])
        axis.set_yticks([0, 1000, 2000])
        axis.set_xlabel("X (m)")
        axis.set_ylabel("Y (m)")
        axis.grid(color="#D9D9D9", linewidth=0.35, alpha=0.7)
        axis.set_title(method, fontsize=7.0, pad=4)
        figure.legend(
            handles=_legend_handles(),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=2,
            fontsize=6.8,
            handlelength=1.8,
            columnspacing=0.9,
            handletextpad=0.45,
            borderpad=0.2,
            labelspacing=0.3,
            frameon=False,
        )
        figure.tight_layout(rect=(0, 0.20, 1, 1), pad=0.5)
        _save_formats(
            figure,
            FIGURE_DIR / f"e2_inspection_order_trajectories_v5_panel_{panel}",
        )


def main():
    QA_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    orders = load_orders()
    tasks = [
        (
            level,
            method,
            orders[level][method],
            OUTPUT_ROOT.resolve(),
        )
        for level in LEVELS
        for method in METHODS
    ]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        records = list(executor.map(_replay, tasks))
    records.sort(
        key=lambda item: (LEVELS.index(item["level"]), METHODS.index(item["method"]))
    )
    _save_source(records)
    distance_records = _distance_tsp_records()
    display_records = [
        record
        for record in records
        if record["method"] != "DRL-SA (G2A-aware SA)"
    ] + distance_records
    _save_display_source(display_records)
    _plot(display_records)
    print(SOURCE_DIR / "e2_inspection_order_trajectories_v5.json")
    print(SOURCE_DIR / "e2_trajectory_panels_v5.json")
    print(FIGURE_DIR / "e2_inspection_order_trajectories_v5_panel_a.pdf")


if __name__ == "__main__":
    main()
