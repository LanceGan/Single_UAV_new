"""Render V5 reward/feasibility histories and the E1 result table."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
FIGURE_DIR = ROOT.parent / "P3" / "figs" / "v5"
SOURCE_DIR = FIGURE_DIR / "source_data"
TRAJECTORY_SUMMARY = SOURCE_DIR / "e1_payload_trajectories_v5.json"

LEVELS = ("Low", "Medium", "High")
METHODS = ("CA-TD3", "D3QN", "PPO", "SAC")
COLORS = {
    "CA-TD3": "#C23B3B",
    "D3QN": "#6A519A",
    "PPO": "#D08B28",
    "SAC": "#2C8C76",
}
RUNS = {
    "Low": {
        "CA-TD3": "journal_results/v4-qos/w50-50/qos6/A/low/td3_bc",
        "D3QN": "journal_results/v4-models/layer1/d3qn/low/d3qn",
        "PPO": "journal_results/v5-win/E1/ppo/A/low/ppo",
        "SAC": "journal_results/v5-win/E1/sac/A/low/sac",
    },
    "Medium": {
        "CA-TD3": "journal_results/v4-models/load/medium/A/medium/td3_bc",
        "D3QN": "journal_results/v5-win/E1/d3qn/medium/d3qn",
        "PPO": "journal_results/v5-win/E1/ppo/A/medium/ppo",
        "SAC": "journal_results/v5-win/E1/sac/A/medium/sac",
    },
    "High": {
        "CA-TD3": "journal_results/v4-models/load/high/A/high/td3_bc",
        "D3QN": "journal_results/v5-win/E1/d3qn/high/d3qn",
        "PPO": "journal_results/v5-win/E1/ppo/A/high/ppo",
        "SAC": "journal_results/v5-win/E1/sac/A/high/sac",
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


def smooth_series(values, window):
    values = np.asarray(values, dtype=float)
    result = np.full(values.shape, np.nan, dtype=float)
    if len(values) < int(window):
        return result
    kernel = np.ones(int(window), dtype=float) / float(window)
    result[int(window) - 1 :] = np.convolve(values, kernel, mode="valid")
    return result


def _reward_payload(run: Path):
    candidates = (
        (run / "UAV_TD3_improved.npz", "ep_rewards"),
        (run / "training_history.npz", "episode_rewards"),
    )
    for path, key in candidates:
        if path.is_file():
            with np.load(path, allow_pickle=True) as payload:
                if key in payload:
                    return path, key, np.asarray(payload[key], dtype=float)
    raise FileNotFoundError(f"no reward history found under {run}")


def load_histories():
    rewards = []
    evaluations = []
    selected = []
    for level in LEVELS:
        for method in METHODS:
            run = ROOT / RUNS[level][method]
            reward_path, reward_key, values = _reward_payload(run)
            for episode, reward in enumerate(values, start=1):
                rewards.append(
                    {
                        "level": level,
                        "method": method,
                        "episode": int(episode),
                        "reward": float(reward),
                        "source_path": str(reward_path.resolve()),
                        "source_key": reward_key,
                    }
                )

            payload = json.loads(
                (run / "eval_history.json").read_text(encoding="utf-8")
            )
            for record in payload["history"]:
                evaluations.append(
                    {
                        "level": level,
                        "method": method,
                        "episode": int(record["episode"]),
                        "success": bool(record.get("success", False)),
                        "source_path": str(
                            (run / "eval_history.json").resolve()
                        ),
                    }
                )

            config = json.loads(
                (run / "run_config.json").read_text(encoding="utf-8")
            )
            best = payload.get("eval_best") or payload.get("best") or {}
            visited = int(best.get("visited_inspection_points", 0) or 0)
            expected_upload = float(sum(config.get("payloads_mb", [])))
            uploaded = float(best.get("uploaded_mb", 0.0) or 0.0)
            complete = (
                visited >= len(config.get("payloads_mb", []))
                and uploaded >= expected_upload - 1e-7
            )
            outage = float(best.get("outage_time_s", 0.0) or 0.0)
            outage_max = float(config.get("outage_max_s", 6.0))
            feasible = bool(complete and outage <= outage_max)
            task_time = (
                float(best["task_time_s"])
                if complete and best.get("task_time_s") is not None
                else None
            )
            delay = (
                float(best["transmission_delay_s"])
                if complete and best.get("transmission_delay_s") is not None
                else None
            )
            selected.append(
                {
                    "level": level,
                    "method": method,
                    "eval_best_episode": int(best.get("episode", 0)),
                    "task_time_s": task_time,
                    "transmission_delay_s": delay,
                    "objective_s": (
                        task_time + delay
                        if task_time is not None and delay is not None
                        else None
                    ),
                    "contiguous_outage_s": outage,
                    "outage_limit_s": outage_max,
                    "visited_points": visited,
                    "expected_visits": len(config.get("payloads_mb", [])),
                    "uploaded_mb": uploaded,
                    "expected_upload_mb": expected_upload,
                    "complete": complete,
                    "feasible": feasible,
                    "source_path": str(run.resolve()),
                }
            )
    if TRAJECTORY_SUMMARY.is_file():
        payload = json.loads(TRAJECTORY_SUMMARY.read_text(encoding="utf-8"))
        validated = {
            (record["method"], record["level"]): record
            for record in payload["records"]
        }
        selected = []
        for level in LEVELS:
            for method in METHODS:
                record = validated[(method, level)]
                selected.append(
                    {
                        "level": level,
                        "method": method,
                        "eval_best_episode": record["eval_best_episode"],
                        "task_time_s": record["task_time_s"],
                        "transmission_delay_s": record[
                            "transmission_delay_s"
                        ],
                        "objective_s": record["objective_s"],
                        "contiguous_outage_s": record[
                            "contiguous_outage_s"
                        ],
                        "outage_limit_s": 6.0,
                        "visited_points": record["visited_points"],
                        "expected_visits": 6,
                        "uploaded_mb": record["uploaded_mb"],
                        "expected_upload_mb": sum(
                            {
                                "Low": [90, 90, 60, 60, 90, 60],
                                "Medium": [180, 180, 180, 120, 120, 120],
                                "High": [210, 290, 290, 290, 210, 210],
                            }[level]
                        ),
                        "complete": record["complete"],
                        "feasible": record["feasible"],
                        "source_path": record["run"],
                        "trajectory_source": record["trajectory_source"],
                        "actor_sha256": record["actor_sha256"],
                        "trajectory_sha256": record[
                            "trajectory_sha256"
                        ],
                        "matches_eval_history": record[
                            "matches_eval_history"
                        ],
                    }
                )
    return rewards, evaluations, selected


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_source(rewards, evaluations, selected):
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(SOURCE_DIR / "e1_training_rewards_v5.csv", rewards)
    _write_csv(SOURCE_DIR / "e1_feasibility_v5.csv", evaluations)
    _write_csv(SOURCE_DIR / "e1_trajectory_results_v5.csv", selected)
    (SOURCE_DIR / "e1_training_validation_v5.json").write_text(
        json.dumps(
            {
                "protocol": {
                    "levels": list(LEVELS),
                    "methods": list(METHODS),
                    "training_episodes": 3000,
                    "evaluation_interval_episodes": 25,
                    "uncertainty": "none (one deterministic training run)",
                    "reward_rendering": (
                        "raw reward and trailing 50-episode mean"
                    ),
                    "feasibility_rendering": (
                        "raw success indicator and trailing 10-evaluation mean"
                    ),
                },
                "runs": {
                    level: {
                        method: str((ROOT / relative).resolve())
                        for method, relative in methods.items()
                    }
                    for level, methods in RUNS.items()
                },
                "selected": selected,
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


def _style_axis(axis, *, ylabel, xlabel):
    axis.set_xlim(0, 3050)
    axis.set_xticks([0, 500, 1000, 1500, 2000, 2500, 3000])
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.45)
    axis.set_axisbelow(True)


def _plot_reward_panel(axis, level, rewards):
    for method in METHODS:
        values = np.asarray(
            [
                row["reward"]
                for row in rewards
                if row["level"] == level and row["method"] == method
            ],
            dtype=float,
        )
        episodes = np.arange(1, len(values) + 1)
        axis.plot(
            episodes,
            values,
            color=COLORS[method],
            linewidth=0.35,
            alpha=0.13,
        )
        axis.plot(
            episodes,
            smooth_series(values, 50),
            color=COLORS[method],
            linewidth=1.2,
            label=method,
        )
    _style_axis(
        axis,
        ylabel="Training reward",
        xlabel="Training episode",
    )


def _plot_feasibility_panel(axis, level, evaluations):
    for method in METHODS:
        records = [
            row
            for row in evaluations
            if row["level"] == level and row["method"] == method
        ]
        episodes = np.asarray(
            [row["episode"] for row in records], dtype=float
        )
        success = np.asarray(
            [row["success"] for row in records], dtype=float
        )
        axis.plot(
            episodes,
            success,
            color=COLORS[method],
            linewidth=0.55,
            alpha=0.18,
        )
        axis.plot(
            episodes,
            smooth_series(success, 10),
            color=COLORS[method],
            linewidth=1.2,
            label=method,
        )
    _style_axis(
        axis,
        ylabel="Feasible evaluations (%)",
        xlabel="Evaluation episode",
    )
    axis.set_ylim(-0.03, 1.03)
    axis.set_yticks(
        [0, 0.25, 0.5, 0.75, 1],
        ["0", "25", "50", "75", "100"],
    )


def _save_metric_panels(kind, plotter, values):
    for panel, level in zip("abc", LEVELS):
        figure, axis = plt.subplots(1, 1, figsize=(3.45, 2.65))
        plotter(axis, level, values)
        axis.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.20),
            ncol=2,
            fontsize=6.2,
            handlelength=2.1,
        )
        figure.subplots_adjust(
            left=0.18,
            right=0.98,
            bottom=0.27,
            top=0.96,
        )
        _save_formats(
            figure,
            FIGURE_DIR / f"{kind}_panel_{panel}",
        )

    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[method],
            linewidth=1.2,
            label=method,
        )
        for method in METHODS
    ]
    legend = plt.figure(figsize=(7.15, 0.55))
    legend.legend(
        handles=handles,
        loc="center",
        ncol=4,
        frameon=False,
        fontsize=6.2,
        handlelength=2.1,
    )
    _save_formats(legend, FIGURE_DIR / f"{kind}_legend")
    plt.close(legend)


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    rewards, evaluations, selected = load_histories()
    write_source(rewards, evaluations, selected)
    _save_metric_panels(
        "e1_reward_curves_v5",
        _plot_reward_panel,
        rewards,
    )
    _save_metric_panels(
        "e1_feasibility_curves_v5",
        _plot_feasibility_panel,
        evaluations,
    )
    print(SOURCE_DIR / "e1_trajectory_results_v5.csv")
    print(FIGURE_DIR / "e1_reward_curves_v5_panel_a.pdf")
    print(FIGURE_DIR / "e1_feasibility_curves_v5_panel_a.pdf")


if __name__ == "__main__":
    main()
