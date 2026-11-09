"""Plot completed V5 Low-tier E1 reward and feasibility histories."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
FIGURE_DIR = ROOT.parent / "P3" / "figs" / "v5"
SOURCE_DIR = FIGURE_DIR / "source_data"
SKILL_SCRIPTS = Path("/Users/ganruijie.1/.codex/skills/nature-figure/scripts")
METHODS = {
    "CA-TD3": {
        "reward_path": ROOT
        / "journal_results/v4-qos/w50-50/qos6/A/low/td3_bc/UAV_TD3_improved.npz",
        "reward_key": "ep_rewards",
        "eval_path": ROOT
        / "journal_results/v4-qos/w50-50/qos6/A/low/td3_bc/eval_history.json",
        "color": "#0F4D92",
    },
    "D3QN": {
        "reward_path": ROOT
        / "journal_results/v4-models/layer1/d3qn/low/d3qn/training_history.npz",
        "reward_key": "episode_rewards",
        "eval_path": ROOT
        / "journal_results/v4-models/layer1/d3qn/low/d3qn/eval_history.json",
        "color": "#B8781A",
    },
    "PPO": {
        "reward_path": ROOT
        / "journal_results/v5-win/E1/ppo/A/low/ppo/training_history.npz",
        "reward_key": "episode_rewards",
        "eval_path": ROOT
        / "journal_results/v5-win/E1/ppo/A/low/ppo/eval_history.json",
        "color": "#9A4D8E",
    },
    "DDPG": {
        "reward_path": ROOT
        / "journal_results/v5-win/E1/ddpg/baseline/low/ddpg_bc/UAV_TD3_improved.npz",
        "reward_key": "ep_rewards",
        "eval_path": ROOT
        / "journal_results/v5-win/E1/ddpg/baseline/low/ddpg_bc/eval_history.json",
        "color": "#4D7F60",
    },
    "SAC": {
        "reward_path": ROOT
        / "journal_results/v5-win/E1/sac/A/low/sac/training_history.npz",
        "reward_key": "episode_rewards",
        "eval_path": ROOT
        / "journal_results/v5-win/E1/sac/A/low/sac/eval_history.json",
        "color": "#3A8EA5",
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
    if len(values) < window:
        return result
    kernel = np.ones(window, dtype=float) / window
    result[window - 1 :] = np.convolve(values, kernel, mode="valid")
    return result


def load_low_histories():
    reward_rows = []
    eval_rows = []
    for method, spec in METHODS.items():
        with np.load(spec["reward_path"], allow_pickle=True) as payload:
            rewards = np.asarray(payload[spec["reward_key"]], dtype=float)
        for episode, reward in enumerate(rewards, start=1):
            reward_rows.append(
                {
                    "method": method,
                    "episode": episode,
                    "reward": float(reward),
                }
            )
        payload = json.loads(
            spec["eval_path"].read_text(encoding="utf-8")
        )
        for item in payload["history"]:
            eval_rows.append(
                {
                    "method": method,
                    "episode": int(item["episode"]),
                    "success": bool(item.get("success", False)),
                }
            )
    return reward_rows, eval_rows


def write_source(reward_rows, eval_rows):
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    reward_csv = SOURCE_DIR / "e1_training_rewards_v5.csv"
    eval_csv = SOURCE_DIR / "e1_feasibility_v5.csv"
    metadata_path = SOURCE_DIR / "e1_training_validation_v5.json"
    with reward_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["method", "episode", "reward"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(reward_rows)
    with eval_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["method", "episode", "success"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(eval_rows)
    metadata_path.write_text(
        json.dumps(
            {
                "protocol": {
                    "level": "Low",
                    "episodes": 3000,
                    "evaluation_interval_episodes": 25,
                    "run_index": 1,
                    "uncertainty": "none (one run)",
                },
                "methods": {
                    method: {
                        "reward_source": str(spec["reward_path"].resolve()),
                        "reward_key": spec["reward_key"],
                        "eval_source": str(spec["eval_path"].resolve()),
                    }
                    for method, spec in METHODS.items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def plot(reward_rows, eval_rows):
    labels = list(METHODS)
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.7))
    for method in labels:
        color = METHODS[method]["color"]
        rewards = np.asarray(
            [
                row["reward"]
                for row in reward_rows
                if row["method"] == method
            ],
            dtype=float,
        )
        episodes = np.arange(1, len(rewards) + 1)
        smooth = smooth_series(rewards, 50)
        axes[0].plot(episodes, rewards, color=color, lw=0.35, alpha=0.13)
        axes[0].plot(episodes, smooth, color=color, lw=1.2, label=method)

        evaluations = [
            row for row in eval_rows if row["method"] == method
        ]
        eval_episodes = np.asarray(
            [row["episode"] for row in evaluations], dtype=int
        )
        success = np.asarray(
            [row["success"] for row in evaluations], dtype=float
        )
        trailing = smooth_series(success, 10)
        axes[1].plot(
            eval_episodes,
            success,
            color=color,
            lw=0.55,
            alpha=0.18,
        )
        axes[1].plot(eval_episodes, trailing, color=color, lw=1.2)

    axes[0].set_xlim(0, 3050)
    axes[0].set_ylim(-35, 8)
    axes[0].set_yticks([-30, -20, -10, 0])
    axes[0].set_ylabel("Training reward")
    axes[0].set_xlabel("Training episode")
    axes[1].set_xlim(0, 3050)
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_yticks([0, 0.25, 0.5, 0.75, 1], ["0", "25", "50", "75", "100"])
    axes[1].set_ylabel("Feasible evaluations (%)")
    axes[1].set_xlabel("Evaluation episode")
    for ax in axes:
        ax.set_xticks([0, 500, 1000, 1500, 2000, 2500, 3000])
        ax.grid(axis="y", color="#D9D9D9", lw=0.45)
        ax.set_axisbelow(True)
    axes[0].text(
        -0.14,
        1.03,
        "a",
        transform=axes[0].transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    axes[1].text(
        -0.14,
        1.03,
        "b",
        transform=axes[1].transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=5,
    )
    fig.text(
        0.5,
        -0.005,
        "Panel a: raw reward (faint) and 50-episode smoothed trace.\n"
        "Panel b: deterministic feasibility (faint) and trailing 10-evaluation "
        "smoothed trace. Single deterministic run; no uncertainty interval.",
        ha="center",
        va="top",
        fontsize=6.3,
    )
    fig.tight_layout(rect=(0, 0.095, 1, 0.92), pad=0.5, w_pad=1.4)

    output = FIGURE_DIR / "e1_training_validation_v5"
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
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    reward_rows, eval_rows = load_low_histories()
    write_source(reward_rows, eval_rows)
    plot(reward_rows, eval_rows)
    print(FIGURE_DIR)


if __name__ == "__main__":
    main()
