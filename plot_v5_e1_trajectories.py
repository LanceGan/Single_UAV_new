"""Render the finalized V5 E1 Low-tier dual-map trajectory comparison."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
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
QA_DIR = FIGURE_DIR / "qa"
SKILL_SCRIPTS = Path("/Users/ganruijie.1/.codex/skills/nature-figure/scripts")

CONTINUOUS_RUNS = {
    "CA-TD3": {
        "run": ROOT / "journal_results/v4-qos/w50-50/qos6/A/low/td3_bc",
        "actor": ROOT / "sync_bundles/v4_eval_best_actors/w50-50_qos6_actor.pth",
        "expected": (169.0, 75.5, 3.0, 6),
        "status": "feasible",
    },
    "PPO": {
        "run": ROOT / "journal_results/v5-win/E1/ppo/A/low/ppo",
        "actor": ROOT / "sync_bundles/v5_e1_low_trajectories/ppo_actor_eval_best.pth",
        "expected": (190.0, 75.5, 3.0, 6),
        "status": "feasible",
    },
    "DDPG": {
        "run": ROOT / "journal_results/v5-win/E1/ddpg/baseline/low/ddpg_bc",
        "actor": ROOT / "sync_bundles/v5_e1_low_trajectories/ddpg_actor_eval_best.pth",
        "expected": (1000.0, 140.5, 20.0, 4),
        "status": "incomplete",
    },
    "SAC": {
        "run": ROOT / "journal_results/v5-win/E1/sac/A/low/sac",
        "actor": ROOT / "sync_bundles/v5_e1_low_trajectories/sac_actor_eval_best.pth",
        "expected": (170.0, 81.5, 4.0, 6),
        "status": "feasible",
    },
}
D3QN_RUN = ROOT / "journal_results/v4-models/layer1/d3qn/low/d3qn"
D3QN_ACTOR = D3QN_RUN / "models/d3qn_eval_best.pt"
D3QN_EXPECTED = (175.0, 105.5, 3.0, 6)
METHOD_ORDER = ("CA-TD3", "D3QN", "PPO", "DDPG", "SAC")

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


def _report_values(report):
    return (
        float(report["task_time_s"]),
        float(report["transmission_delay_s"]),
        float(report["outage_time_s"]),
        int(report["visited_inspection_points"]),
    )


def evaluate_all():
    results = {}
    for method, spec in CONTINUOUS_RUNS.items():
        context = load_run_context(spec["run"])
        with redirect_stdout(io.StringIO()):
            report, trace = evaluate_checkpoint(context, spec["actor"])
        values = _report_values(report)
        if not np.allclose(values, spec["expected"], atol=1e-9):
            raise ValueError(
                f"{method} replay mismatch: expected {spec['expected']}, got {values}"
            )
        results[method] = {
            "report": report,
            "trace": trace,
            "actor": spec["actor"],
            "run": spec["run"],
            "status": spec["status"],
        }

    d3qn_context = load_d3qn_run_context(D3QN_RUN)
    with redirect_stdout(io.StringIO()):
        d3qn_report, d3qn_trace = evaluate_d3qn_checkpoint(
            d3qn_context, D3QN_ACTOR
        )
    d3qn_values = _report_values(d3qn_report)
    if not np.allclose(d3qn_values, D3QN_EXPECTED, atol=1e-9):
        raise ValueError(
            f"D3QN replay mismatch: expected {D3QN_EXPECTED}, got {d3qn_values}"
        )
    results["D3QN"] = {
        "report": d3qn_report,
        "trace": d3qn_trace,
        "actor": D3QN_ACTOR,
        "run": D3QN_RUN,
        "status": "feasible",
    }
    return results


def write_source(results):
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    trajectory_path = SOURCE_DIR / "e1_low_trajectories_v5.npz"
    metrics_path = SOURCE_DIR / "e1_low_trajectory_metrics_v5.json"
    trajectory_csv = SOURCE_DIR / "e1_low_trajectories_v5.csv"

    arrays = {}
    rows = []
    metrics = {}
    for method in METHOD_ORDER:
        result = results[method]
        trace = result["trace"]
        positions = np.asarray(trace["positions"], dtype=float)
        times = np.asarray(trace["time_s"], dtype=float)
        outage = np.asarray(trace["outage_s"], dtype=float)
        visited = np.asarray(trace["visited_points"], dtype=int)
        arrays[f"{method}_x_km"] = positions[:, 0]
        arrays[f"{method}_y_km"] = positions[:, 1]
        arrays[f"{method}_time_s"] = times
        arrays[f"{method}_outage_s"] = outage
        arrays[f"{method}_visited_points"] = visited
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
            "status": result["status"],
            "run": str(result["run"].resolve()),
            "actor": str(result["actor"].resolve()),
            "success": bool(report.get("success", False)),
            "termination_reason": report.get("termination_reason"),
            "task_time_s": report.get("task_time_s"),
            "transmission_delay_s": report.get("transmission_delay_s"),
            "objective_s": (
                report.get("task_time_s") + report.get("transmission_delay_s")
                if report.get("success")
                else None
            ),
            "outage_time_s": report.get("outage_time_s"),
            "visited_inspection_points": report.get("visited_inspection_points"),
            "uploaded_mb": report.get("uploaded_mb"),
        }
    np.savez(trajectory_path, **arrays)
    with trajectory_csv.open("w", newline="", encoding="utf-8") as stream:
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
    metrics_path.write_text(
        json.dumps(
            {
                "protocol": {
                    "level": "Low",
                    "qos_constraint": "max_contiguous_outage <= 6 s",
                    "seed": 1,
                    "policy_mode": "deterministic, no exploration noise or MapGuide",
                    "uncertainty": "none (single seed)",
                },
                "methods": metrics,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def write_bundle_validation(results):
    bundle_dir = ROOT / "sync_bundles" / "v5_e1_low_trajectories"
    manifest = json.loads((bundle_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    manifest_by_label = {item["label"].upper(): item for item in manifest["items"]}
    validation = {
        "source_commit": "c9f74d9",
        "items": {},
    }
    for method in ("PPO", "DDPG", "SAC"):
        result = results[method]
        item = manifest_by_label[method]
        digest = hashlib.sha256(Path(result["actor"]).read_bytes()).hexdigest()
        report = result["report"]
        expected = {
            "task_time_s": item["task_time_s"],
            "transmission_delay_s": item["transmission_delay_s"],
            "outage_time_s": item["outage_time_s"],
            "visited_inspection_points": item["visited_inspection_points"],
            "uploaded_mb": item["uploaded_mb"],
        }
        history_payload = json.loads(
            (result["run"] / "eval_history.json").read_text(encoding="utf-8")
        )
        history_best = (
            history_payload.get("eval_best")
            or history_payload.get("best")
            or {}
        )
        expected_history = {
            key: history_best.get(key) for key in expected
        }
        actual = {
            key: report.get(key) for key in expected
        }
        validation["items"][method] = {
            "actor_sha256_match": digest == item["actor_sha256"],
            "actor_sha256": digest,
            "manifest_metrics": expected,
            "manifest_matches_eval_history": all(
                expected[key] == value
                for key, value in expected_history.items()
            ),
            "eval_history_best": expected_history,
            "actual": actual,
            "actual_matches_eval_history": all(
                actual[key] == value
                for key, value in expected_history.items()
            ),
        }

    # The supplied DDPG trajectory does not match its own eval-history entry.
    # Preserve that result in the validation record and emit the locally
    # reproducible trace from the hash-verified actor.
    validation["items"]["DDPG"]["supplied_trajectory_matches"] = False
    validation["items"]["DDPG"]["note"] = (
        "Supplied trajectory reports 1/6 visits and 90 MB, while eval-history "
        "episode 1750 reports 4/6 visits and 300 MB. The local actor replay "
        "reproduces the eval-history record and is used for the figure."
    )
    ddpg_trace = results["DDPG"]["trace"]
    np.savez(
        bundle_dir / "e1_low_ddpg_trajectory_macverified.npz",
        x=np.asarray(ddpg_trace["positions"])[:, 0],
        y=np.asarray(ddpg_trace["positions"])[:, 1],
        time_s=np.asarray(ddpg_trace["time_s"]),
        step=np.arange(len(ddpg_trace["time_s"]), dtype=int),
        outage_s=np.asarray(ddpg_trace["outage_s"]),
        pending_mb=np.asarray(ddpg_trace["pending_mb"]),
        visited_points=np.asarray(ddpg_trace["visited_points"]),
    )
    (bundle_dir / "e1_low_ddpg_metrics_macverified.json").write_text(
        json.dumps(results["DDPG"]["report"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle_dir / "MAC_VALIDATION.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def plot(results):
    g2a, a2g = _map_data()
    x_grid, y_grid, coverage, snr = _map_layers(g2a, a2g)
    users = np.loadtxt(
        ROOT / "results" / "datas" / "Users_6.txt"
    )[:, :2] * 100.0

    figure, axes = plt.subplots(2, 5, figsize=(7.2, 5.2))
    figure.subplots_adjust(
        left=0.065,
        right=0.88,
        bottom=0.10,
        top=0.94,
        wspace=0.16,
        hspace=0.18,
    )
    coverage_mesh = None
    snr_mesh = None
    for column, method in enumerate(METHOD_ORDER):
        report = results[method]["report"]
        trace = results[method]["trace"]
        if report.get("success"):
            title = (
                f"{method}: {report['task_time_s'] + report['transmission_delay_s']:.1f} s"
            )
        else:
            title = (
                f"{method}: {report['visited_inspection_points']}/6\n"
                f"outage={report['outage_time_s']:.1f} s"
            )
        snr_mesh = _draw_map(
            axes[0, column],
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
        coverage_mesh = _draw_map(
            axes[1, column],
            x_grid,
            y_grid,
            coverage,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            title="G2A coverage",
            users=users,
            show_numbering=False,
        )
        path_color = "#FFFFFF" if report.get("success") else "#FF7F50"
        _draw_path(axes[0, column], trace, path_color)
        _draw_path(axes[1, column], trace, path_color)

    for row_index, row_axes in enumerate(axes):
        for column, axis in enumerate(row_axes):
            if column:
                axis.set_ylabel("")
                axis.tick_params(labelleft=False, left=False)
            if row_index == 0:
                axis.set_xlabel("")
                axis.tick_params(labelbottom=False, bottom=False)

    panel_ids = list("abcdefghij")
    for panel_id, axis in zip(panel_ids, axes.ravel()):
        _panel_label(axis, panel_id)

    snr_cax = figure.add_axes([0.895, 0.55, 0.008, 0.35])
    coverage_cax = figure.add_axes([0.895, 0.11, 0.008, 0.35])
    snr_bar = figure.colorbar(snr_mesh, cax=snr_cax)
    snr_bar.set_label("A2G SNR (dB)")
    coverage_bar = figure.colorbar(coverage_mesh, cax=coverage_cax)
    coverage_bar.set_label("G2A coverage probability")
    coverage_bar.ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
    figure.text(
        0.5,
        0.012,
        "Top: A2G SNR map; bottom: G2A coverage probability. White paths are "
        "complete feasible runs; the coral path is the DDPG incomplete diagnostic.\n"
        "Results are deterministic and guide-free.",
        ha="center",
        va="bottom",
        fontsize=6.3,
        linespacing=1.2,
    )

    output = FIGURE_DIR / "e1_low_trajectories_v5"
    sys.path.insert(0, str(SKILL_SCRIPTS))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    require_matplotlib_panel_alignment(
        figure,
        axes=list(axes.ravel()),
        panel_ids=panel_ids,
        row_groups=[panel_ids[:5], panel_ids[5:]],
        column_groups=[[panel_ids[i], panel_ids[i + 5]] for i in range(5)],
        exclude_axes=[snr_cax, coverage_cax],
        json_out=f"{output}.alignment.json",
        overlay_svg=f"{output}.alignment.svg",
        strict=True,
    )
    figure.savefig(output.with_suffix(".pdf"), bbox_inches=None)
    figure.savefig(output.with_suffix(".svg"), bbox_inches=None)
    figure.savefig(output.with_suffix(".png"), dpi=600, bbox_inches=None)
    figure.savefig(output.with_suffix(".tiff"), dpi=600, bbox_inches=None)
    plt.close(figure)


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    QA_DIR.mkdir(parents=True, exist_ok=True)
    results = evaluate_all()
    write_source(results)
    write_bundle_validation(results)
    plot(results)
    print(FIGURE_DIR / "e1_low_trajectories_v5.pdf")


if __name__ == "__main__":
    main()
