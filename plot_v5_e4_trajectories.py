"""Render verified V5 E4 trajectory panels from archived eval-best actors."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

from evaluate_policy import (
    _build_world,
    load_actor,
    load_run_context,
    rollout,
)
from plot_v4_trajectory_figures import (
    _draw_map,
    _map_data,
    _map_layers,
)


ROOT = Path(__file__).resolve().parent
P3 = ROOT.parent / "P3"
FIGURE_DIR = P3 / "figs" / "v5"
SOURCE_DIR = FIGURE_DIR / "source_data"
QA_DIR = FIGURE_DIR / "qa"
SKILL_SCRIPTS = Path("/Users/ganruijie.1/.codex/skills/nature-figure/scripts")
ACTOR_BUNDLE = ROOT / "sync_bundles" / "v4_eval_best_actors"
QOS_RECORD_PATH = SOURCE_DIR / "e4_qos_sensitivity_v5.json"
WEIGHT_RECORD_PATH = SOURCE_DIR / "e4_weight_sensitivity_v5.json"
WIN_RECOVERY_COMMIT = "5517943a6d13660f0d1a3d3b177400e08fd667bf"

QOS_VALUES_S = (3.0, 4.0, 5.0, 6.0, 8.0)
WEIGHT_VALUES = ((0.8, 0.2), (0.5, 0.5), (0.2, 0.8))
QOS_LEVELS = ("Low", "Medium", "High")
WEIGHT_LEVELS = ("Low", "Medium", "High")
PANEL_IDS = "abcdefghijklmno"

WIN_LOW_QOS_ACTOR_SHA256 = {
    3.0: "0a70b15dd4c7822eb7d1f023af8a32c38a6446e95ba8f80047795c8f3898e3e8",
    4.0: "0444cdc686bc704ea82f245ecf5a9bcddbda995ba743b07205ed43a36246a782",
    5.0: "822422addb9a93f8564e92c969e7806ee7a5a0279484d68cb5f8c7fc2c89a07e",
    8.0: "12f7620755e230eb67ae8e77d49be06779f1ca57303374ba26e17368494a6c2e",
}

STATUS_STYLE = {
    "feasible": {"color": "#FFFFFF", "linestyle": "-", "label": "Complete, feasible"},
    "constraint": {
        "color": "#FF665C",
        "linestyle": "-",
        "label": "Complete, QoS-infeasible",
    },
    "incomplete": {
        "color": "#F4C542",
        "linestyle": (0, (2.2, 1.4)),
        "label": "Incomplete",
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _weight_label(weights) -> str:
    return f"{float(weights[0]):.1f}/{float(weights[1]):.1f}"


def _status(report: dict) -> str:
    complete = (
        int(report["visited_inspection_points"])
        >= int(report["expected_inspection_points"])
        and float(report["uploaded_mb"])
        >= float(report["expected_upload_mb"]) - 1e-7
    )
    if not complete:
        return "incomplete"
    if bool(report["outage_budget_exceeded"]):
        return "constraint"
    return "feasible"


def _status_lines(report: dict) -> tuple[str, str]:
    visited = int(report["visited_inspection_points"])
    uploaded = float(report["uploaded_mb"])
    expected_upload = float(report["expected_upload_mb"])
    outage = float(report["outage_time_s"])
    status = _status(report)
    if status == "incomplete":
        return (
            f"{visited}/6 · {uploaded:.0f}/{expected_upload:.0f} MB",
            f"outage {outage:.1f} s · incomplete",
        )
    objective = float(report["task_time_s"]) + float(
        report["transmission_delay_s"]
    )
    suffix = " · QoS-infeasible" if status == "constraint" else ""
    return (
        f"6/6 · T+τ {objective:.1f} s",
        f"outage {outage:.1f} s{suffix}",
    )


def _record_lookup(path: Path) -> dict[tuple, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    lookup = {}
    for source_record in payload["records"]:
        record = dict(source_record)
        setting = record["setting"]
        qos_match = re.search(r"qos=(\d+(?:\.\d+)?)s", setting)
        weight_match = re.search(r"w=(\d+)/(\d+)", setting)
        if weight_match is None:
            weight_match = re.search(r"w(\d+)-(\d+)", setting)
        if qos_match is None or weight_match is None:
            raise ValueError(f"cannot parse E4 setting: {setting!r}")
        key = (
            record["level"],
            float(qos_match.group(1)),
            int(weight_match.group(1)),
            int(weight_match.group(2)),
        )
        record["objective_weights"] = [
            int(weight_match.group(1)) / 100.0,
            int(weight_match.group(2)) / 100.0,
        ]
        lookup[key] = record
    return lookup


def _weight_key(weights) -> tuple[int, int]:
    return int(round(100 * float(weights[0]))), int(
        round(100 * float(weights[1]))
    )


def _actor_for_record(record: dict) -> Path | None:
    run = Path(record["source_path"])
    archived = run / "models" / "td3_bc_td3_actoreval_best.pth"
    if archived.is_file():
        return archived

    level = record["level"].lower()
    weights = _weight_key(record["objective_weights"])
    if run.parts[-6:] == (
        "v4-models",
        "load",
        level,
        "A",
        level,
        "td3_bc",
    ):
        if weights == (50, 50):
            return ACTOR_BUNDLE / f"{level}_qos6_actor.pth"

    if "v4-qos" in run.parts and float(record["outage_limit_s"]) == 6.0:
        bundle_names = {
            (80, 20): "w80-20_qos6_actor.pth",
            (50, 50): "w50-50_qos6_actor.pth",
            (20, 80): "w20-80_qos6_actor.pth",
        }
        if weights in bundle_names:
            return ACTOR_BUNDLE / bundle_names[weights]
    return None


def _win_trajectory_for_record(record: dict) -> Path | None:
    if record["level"] != "Low":
        return None
    if _weight_key(record["objective_weights"]) != (50, 50):
        return None
    qos_s = float(record["outage_limit_s"])
    if qos_s not in WIN_LOW_QOS_ACTOR_SHA256:
        return None
    path = ACTOR_BUNDLE / f"low_qos{qos_s:g}_trajectory.npz"
    return path if path.is_file() else None


def _select_records() -> tuple[list[dict], list[dict], list[str]]:
    qos_lookup = _record_lookup(QOS_RECORD_PATH)
    weight_lookup = _record_lookup(WEIGHT_RECORD_PATH)
    qos_records = []
    weight_records = []
    unavailable = []

    for level in QOS_LEVELS:
        for qos_s in QOS_VALUES_S:
            record = qos_lookup[(level, qos_s, 50, 50)]
            actor = _actor_for_record(record)
            win_trajectory = _win_trajectory_for_record(record)
            if actor is None and win_trajectory is None:
                unavailable.append(
                    f"QoS {level} {qos_s:g} s: eval-best actor not archived"
                )
                continue
            source_kind = "actor_replay" if actor is not None else "win_trajectory"
            qos_records.append(
                {
                    "figure": "qos",
                    "panel_id": PANEL_IDS[len(qos_records)],
                    "row": level,
                    "column": qos_s,
                    "setting": f"QoS {qos_s:g} s",
                    "record": record,
                    "run": Path(record["source_path"]),
                    "actor": actor,
                    "win_trajectory": win_trajectory,
                    "source_kind": source_kind,
                }
            )

    for level in WEIGHT_LEVELS:
        for weights in WEIGHT_VALUES:
            key = (level, 6.0, *_weight_key(weights))
            record = weight_lookup[key]
            actor = _actor_for_record(record)
            if actor is None:
                unavailable.append(
                    f"weight {level} {_weight_label(weights)}: "
                    "eval-best actor not archived"
                )
                continue
            weight_records.append(
                {
                    "figure": "weight",
                    "panel_id": PANEL_IDS[len(weight_records)],
                    "row": level,
                    "column": weights,
                    "setting": f"({_weight_label(weights)})",
                    "record": record,
                    "run": Path(record["source_path"]),
                    "actor": actor,
                    "win_trajectory": None,
                    "source_kind": "actor_replay",
                }
            )

    expected_qos = len(QOS_LEVELS) * len(QOS_VALUES_S)
    expected_weight = len(WEIGHT_LEVELS) * len(WEIGHT_VALUES)
    if len(qos_records) != expected_qos:
        raise FileNotFoundError(
            "unexpected QoS trajectory actor availability: "
            + "; ".join(unavailable)
        )
    if len(weight_records) != expected_weight:
        raise FileNotFoundError(
            "unexpected weight trajectory actor availability: "
            + "; ".join(unavailable)
        )
    return qos_records, weight_records, unavailable


def _validate_report(record: dict, report: dict) -> None:
    complete = bool(record["complete"])
    feasible = bool(record["qos_feasible"])
    actual_complete = (
        int(report["visited_inspection_points"])
        >= int(report["expected_inspection_points"])
        and float(report["uploaded_mb"])
        >= float(report["expected_upload_mb"]) - 1e-7
    )
    actual_feasible = actual_complete and not bool(
        report["outage_budget_exceeded"]
    )
    if actual_complete != complete:
        raise ValueError(
            f"{record['setting']} completeness mismatch: "
            f"{actual_complete} vs {complete}"
        )
    if actual_feasible != feasible:
        raise ValueError(
            f"{record['setting']} QoS mismatch: "
            f"{actual_feasible} vs {feasible}"
        )
    expected_values = {
        "visited_inspection_points": int(record["visited_points"]),
        "uploaded_mb": float(record["uploaded_mb"]),
        "outage_time_s": float(record["contiguous_outage_s"]),
    }
    for field, expected in expected_values.items():
        actual = float(report[field])
        if not np.isclose(actual, expected, atol=1e-9):
            raise ValueError(
                f"{record['setting']} {field} mismatch: "
                f"{actual} vs {expected}"
            )

    if complete:
        expected_values = {
            "task_time_s": float(record["task_time_s"]),
            "transmission_delay_s": float(record["transmission_delay_s"]),
        }
    else:
        expected_values = {
            "task_time_s": float(record["diagnostic_task_time_s"]),
            "transmission_delay_s": float(
                record["diagnostic_transmission_delay_s"]
            ),
        }
    for field, expected in expected_values.items():
        actual = float(report[field])
        if not np.isclose(actual, expected, atol=1e-9):
            raise ValueError(
                f"{record['setting']} {field} mismatch: "
                f"{actual} vs {expected}"
            )


def _evaluate_on_recorded_device(context, checkpoint):
    recorded_device = context.config.get("device")
    device_name = (
        "cpu"
        if recorded_device in {None, "", "none"}
        else str(recorded_device).lower()
    )
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            f"{context.run_dir}: run requires MPS replay precision, "
            "but MPS is unavailable"
        )
    if device_name not in {"cpu", "mps"}:
        raise ValueError(
            f"{context.run_dir}: unsupported replay device {device_name!r}"
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
                torch.as_tensor(state, dtype=torch.float32, device=device)
                .unsqueeze(0)
            )[0]
        return action.cpu().numpy()

    report, trace = rollout(context, world, act)
    report["checkpoint"] = str(Path(checkpoint).resolve())
    report["replay_device"] = device_name
    return report, trace


def _load_win_trajectory(item: dict) -> tuple[dict, dict]:
    record = item["record"]
    trajectory_path = item["win_trajectory"]
    metrics_path = (
        ACTOR_BUNDLE / f"low_qos{float(record['outage_limit_s']):g}_metrics.json"
    )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    with np.load(trajectory_path) as payload:
        trace = {key: payload[key] for key in payload.files}
    report = {
        "success": bool(metrics["success"]),
        "mission_complete": bool(record["complete"]),
        "qos_feasible": bool(record["qos_feasible"]),
        "outage_budget_exceeded": (
            float(metrics["outage_time_s"])
            > float(record["outage_limit_s"])
        ),
        "termination_reason": metrics["termination_reason"],
        "visited_inspection_points": int(
            metrics["visited_inspection_points"]
        ),
        "expected_inspection_points": 6,
        "uploaded_mb": float(metrics["uploaded_mb"]),
        "expected_upload_mb": float(record["expected_upload_mb"]),
        "task_time_s": float(metrics["task_time_s"]),
        "transmission_delay_s": float(metrics["transmission_delay_s"]),
        "outage_time_s": float(metrics["outage_time_s"]),
        "outage_limit_s": float(record["outage_limit_s"]),
        "replay_device": "original-win-run",
    }
    _validate_report(record, report)

    positions = np.asarray(trace["positions"], dtype=float)
    times = np.asarray(trace["time_s"], dtype=float)
    outage = np.asarray(trace["outage_s"], dtype=float)
    visited = np.asarray(trace["visited_points"], dtype=int)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(f"{trajectory_path}: invalid positions shape")
    if not (
        len(positions) == len(times) == len(outage) == len(visited)
    ):
        raise ValueError(f"{trajectory_path}: trajectory arrays differ in length")
    if not np.isclose(times[-1], report["task_time_s"], atol=1e-9):
        raise ValueError(f"{trajectory_path}: final time mismatch")
    if not np.isclose(outage.max(), report["outage_time_s"], atol=1e-9):
        raise ValueError(f"{trajectory_path}: maximum outage mismatch")
    if int(visited.max()) != report["visited_inspection_points"]:
        raise ValueError(f"{trajectory_path}: visited-point mismatch")
    if "x" in trace and "y" in trace:
        if not np.allclose(
            positions,
            np.column_stack((trace["x"], trace["y"])),
            atol=1e-12,
        ):
            raise ValueError(f"{trajectory_path}: x/y differ from positions")
    return report, trace


def evaluate_all(records: list[dict]) -> dict[str, dict]:
    results = {}
    for item in records:
        if item["source_kind"] == "win_trajectory":
            report, trace = _load_win_trajectory(item)
            actor_sha256 = WIN_LOW_QOS_ACTOR_SHA256[
                float(item["record"]["outage_limit_s"])
            ]
        else:
            context = load_run_context(item["run"])
            with redirect_stdout(io.StringIO()):
                report, trace = _evaluate_on_recorded_device(
                    context, item["actor"]
                )
            _validate_report(item["record"], report)
            actor_sha256 = _sha256(item["actor"])
        result = dict(item)
        result["report"] = report
        result["trace"] = trace
        result["actor_sha256"] = actor_sha256
        result["trajectory_sha256"] = (
            _sha256(item["win_trajectory"])
            if item["win_trajectory"] is not None
            else None
        )
        result["status"] = _status(report)
        results[item["panel_id"] + "_" + item["figure"]] = result
    return results


def _safe_key(item: dict) -> str:
    if item["figure"] == "qos":
        return f"qos_{item['row'].lower()}_{item['column']:g}s"
    label = _weight_label(item["column"]).replace(".", "")
    return f"weight_{item['row'].lower()}_{label.replace('/', '_')}"


def write_source(qos_records, weight_records, results, unavailable):
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    for figure_name, records in (
        ("qos", qos_records),
        ("weight", weight_records),
    ):
        rows = []
        arrays = {}
        metrics = []
        for item in records:
            result = results[item["panel_id"] + "_" + figure_name]
            trace = result["trace"]
            positions = np.asarray(trace["positions"], dtype=float) * 100.0
            times = np.asarray(trace["time_s"], dtype=float)
            outage = np.asarray(trace["outage_s"], dtype=float)
            visited = np.asarray(trace["visited_points"], dtype=int)
            key = _safe_key(item)
            arrays[f"{key}_positions_m"] = positions
            arrays[f"{key}_time_s"] = times
            arrays[f"{key}_outage_s"] = outage
            arrays[f"{key}_visited_points"] = visited
            for x_m, y_m, time_s, outage_s, visited_count in zip(
                positions[:, 0],
                positions[:, 1],
                times,
                outage,
                visited,
            ):
                rows.append(
                    {
                        "panel": item["panel_id"],
                        "figure": figure_name,
                        "level": item["row"],
                        "setting": item["setting"],
                        "status": result["status"],
                        "time_s": float(time_s),
                        "x_m": float(x_m),
                        "y_m": float(y_m),
                        "outage_s": float(outage_s),
                        "visited_points": int(visited_count),
                    }
                )
            report = result["report"]
            objective = report.get("task_time_s")
            if objective is not None:
                objective = float(objective) + float(
                    report["transmission_delay_s"]
                )
            metrics.append(
                {
                    "panel": item["panel_id"],
                    "level": item["row"],
                    "setting": item["setting"],
                    "status": result["status"],
                    "source_kind": result["source_kind"],
                    "run": str(item["run"].resolve()),
                    "actor": (
                        str(item["actor"].resolve())
                        if item["actor"] is not None
                        else None
                    ),
                    "actor_sha256": result["actor_sha256"],
                    "trajectory_source": (
                        str(item["win_trajectory"].resolve())
                        if item["win_trajectory"] is not None
                        else str(item["run"].resolve())
                    ),
                    "trajectory_sha256": result["trajectory_sha256"],
                    "replay_device": result["report"].get("replay_device"),
                    "win_recovery_commit": (
                        WIN_RECOVERY_COMMIT
                        if result["source_kind"] == "win_trajectory"
                        else None
                    ),
                    "eval_best_episode": item["record"]["eval_best_episode"],
                    "mission_complete": bool(report["mission_complete"]),
                    "qos_feasible": bool(report["qos_feasible"]),
                    "task_time_s": report.get("task_time_s"),
                    "transmission_delay_s": report.get(
                        "transmission_delay_s"
                    ),
                    "objective_s": objective,
                    "contiguous_outage_s": report["outage_time_s"],
                    "outage_limit_s": report["outage_limit_s"],
                    "visited_inspection_points": report[
                        "visited_inspection_points"
                    ],
                    "uploaded_mb": report["uploaded_mb"],
                    "expected_upload_mb": report["expected_upload_mb"],
                    "termination_reason": report["termination_reason"],
                }
            )
        stem = f"e4_{figure_name}_trajectories_v5"
        np.savez_compressed(SOURCE_DIR / f"{stem}.npz", **arrays)
        csv_path = SOURCE_DIR / f"{stem}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "panel",
                    "figure",
                    "level",
                    "setting",
                    "status",
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
        payload = {
            "protocol": {
                "suite": "V5 E4",
                "seed": 1,
                "evaluation_steps": 2000,
                "evaluation_mode": (
                    "deterministic actor only; no exploration noise or MapGuide"
                ),
                "valid_objective": "6/6 visits and full upload",
                "single_seed_uncertainty": "none",
            },
            "unavailable_trajectory_actors": unavailable,
            "records": metrics,
        }
        (SOURCE_DIR / f"{stem}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def _draw_status_path(axis, trace, status):
    positions = np.asarray(trace["positions"], dtype=float) * 100.0
    style = STATUS_STYLE[status]
    axis.plot(
        positions[:, 0],
        positions[:, 1],
        color="#202020",
        linewidth=2.1,
        alpha=0.28,
        linestyle=style["linestyle"],
        zorder=2,
    )
    axis.plot(
        positions[:, 0],
        positions[:, 1],
        color=style["color"],
        linewidth=1.35,
        linestyle=style["linestyle"],
        zorder=3,
    )


def _panel_label_external(axis, panel_id):
    axis.annotate(
        panel_id,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(-10, 8),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=8,
        fontweight="bold",
    )


def _prepare_common(
    figure,
    axes,
    *,
    levels,
    columns,
    records,
    results,
    figure_name,
    bottom,
    top,
    wspace,
    hspace,
):
    g2a, a2g = _map_data()
    x_grid, y_grid, _, snr = _map_layers(g2a, a2g)
    users = np.loadtxt(ROOT / "results" / "datas" / "Users_6.txt")[:, :2] * 100.0
    mesh = None
    for item in records:
        row = levels.index(item["row"])
        column = columns.index(item["column"])
        axis = axes[row, column]
        result = results[item["panel_id"] + "_" + figure_name]
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
        mesh.set_rasterized(True)
        _draw_status_path(axis, result["trace"], result["status"])
        first_status, second_status = _status_lines(result["report"])
        axis.set_title(
            f"{item['row']}\n{item['setting']}\n"
            f"{first_status}\n{second_status}",
            fontsize=5.9,
            pad=2.2,
            linespacing=1.15,
        )
        _panel_label_external(axis, item["panel_id"])

    for row in range(len(levels)):
        for column in range(len(columns)):
            axis = axes[row, column]
            if column > 0:
                axis.tick_params(axis="y", labelleft=False)
                axis.set_ylabel("")
            if row < len(levels) - 1:
                axis.set_xlabel("")
                axis.tick_params(axis="x", labelbottom=False)

    cax = figure.add_axes([0.918, max(0.24, bottom + 0.04), 0.008, 0.54])
    colorbar = figure.colorbar(mesh, cax=cax)
    colorbar.set_label("A2G SNR (dB)")

    handles = [
        Line2D(
            [0],
            [0],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.35,
            label=style["label"],
            path_effects=[
                path_effects.withStroke(linewidth=2.2, foreground="#202020")
            ],
        )
        for style in STATUS_STYLE.values()
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.47, 0.035),
        ncol=3,
        handlelength=2.0,
        columnspacing=1.6,
        fontsize=6.3,
    )
    figure.text(
        0.47,
        0.006,
        "Titles show completion and QoS status. T+τ is omitted for incomplete "
        "rollouts; all paths are deterministic, guide-free single-seed replays.",
        ha="center",
        va="bottom",
        fontsize=6.1,
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.90,
        bottom=bottom,
        top=top,
        wspace=wspace,
        hspace=hspace,
    )
    return cax


def _save(figure, output: Path, panel_axes, panel_ids, row_groups, column_groups, cax):
    figure.canvas.draw()
    sys.path.insert(0, str(SKILL_SCRIPTS))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    require_matplotlib_panel_alignment(
        figure,
        axes=panel_axes,
        panel_ids=panel_ids,
        row_groups=row_groups,
        column_groups=column_groups,
        exclude_axes=[cax],
        json_out=f"{output}.alignment.json",
        overlay_svg=f"{output}.alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
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


def plot_qos(records, results):
    figure, axes = plt.subplots(
        len(QOS_LEVELS),
        len(QOS_VALUES_S),
        figsize=(7.15, 6.15),
    )
    cax = _prepare_common(
        figure,
        axes,
        levels=QOS_LEVELS,
        columns=QOS_VALUES_S,
        records=records,
        results=results,
        figure_name="qos",
        bottom=0.13,
        top=0.80,
        wspace=0.22,
        hspace=0.48,
    )
    panel_axes = axes.ravel().tolist()
    panel_ids = [item["panel_id"] for item in records]
    output = FIGURE_DIR / "e4_qos_trajectories_v5"
    _save(
        figure,
        output,
        panel_axes,
        panel_ids,
        [panel_ids[:5], panel_ids[5:10], panel_ids[10:]],
        [panel_ids[index::5] for index in range(5)],
        cax,
    )


def plot_weight(records, results):
    figure, axes = plt.subplots(
        len(WEIGHT_LEVELS),
        len(WEIGHT_VALUES),
        figsize=(7.15, 6.15),
    )
    cax = _prepare_common(
        figure,
        axes,
        levels=WEIGHT_LEVELS,
        columns=WEIGHT_VALUES,
        records=records,
        results=results,
        figure_name="weight",
        bottom=0.13,
        top=0.80,
        wspace=0.25,
        hspace=0.48,
    )
    panel_axes = axes.ravel().tolist()
    panel_ids = [item["panel_id"] for item in records]
    output = FIGURE_DIR / "e4_weight_trajectories_v5"
    _save(
        figure,
        output,
        panel_axes,
        panel_ids,
        [panel_ids[:3], panel_ids[3:6], panel_ids[6:]],
        [panel_ids[index::3] for index in range(3)],
        cax,
    )


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    QA_DIR.mkdir(parents=True, exist_ok=True)
    qos_records, weight_records, unavailable = _select_records()
    results = evaluate_all(qos_records + weight_records)
    write_source(qos_records, weight_records, results, unavailable)
    plot_qos(qos_records, results)
    plot_weight(weight_records, results)
    print(FIGURE_DIR / "e4_qos_trajectories_v5.pdf")
    print(FIGURE_DIR / "e4_weight_trajectories_v5.pdf")


if __name__ == "__main__":
    main()
