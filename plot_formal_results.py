"""Generate paper-ready figures for completed formal training runs."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluate_d3qn import (
    d3qn_checkpoint_paths,
    evaluate_checkpoint as evaluate_d3qn_checkpoint,
    load_d3qn_run_context,
)
from evaluate_policy import (
    checkpoint_paths,
    evaluate_checkpoint,
    load_run_context,
    method_display_name,
    plot_trajectory,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_RUNS = (
    Path("A/low/td3_bc"),
    Path("A/medium/td3_bc"),
    Path("A/high/td3_bc"),
    Path("A/low/ddpg_bc"),
)


def evaluation_series(history):
    rows = history["history"]
    episodes = np.asarray([row["episode"] for row in rows], dtype=int)
    task_time = np.asarray([row["task_time_s"] for row in rows], dtype=float)
    delay = np.asarray([row["transmission_delay_s"] for row in rows], dtype=float)
    returns = np.asarray([row["total_return"] for row in rows], dtype=float)
    visited = np.asarray([row["visited_inspection_points"] for row in rows], dtype=int)
    success = np.asarray([row["success"] for row in rows], dtype=bool)
    objective = task_time + delay
    return {
        "episodes": episodes,
        "task_time": task_time,
        "delay": delay,
        "returns": returns,
        "visited": visited,
        "success": success,
        "objective_raw": objective,
        "objective_feasible": np.where(success, objective, np.nan),
    }


def moving_average(values, window=200):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.array([], dtype=int), values
    if values.size < window:
        return np.array([], dtype=int), np.array([], dtype=float)
    kernel = np.ones(window, dtype=float) / window
    averages = np.convolve(values, kernel, mode="valid")
    episodes = np.arange(window, values.size + 1)
    return episodes, averages


DISPLAY_EPISODE_LIMIT = 2000


def actor_reward_series(rewards, guided, limit=DISPLAY_EPISODE_LIMIT):
    """Actor-only returns within the displayed learning phase.

    Demonstration (teacher) episodes are excluded so the curve reflects the
    agent's own behaviour; failures within the window are kept as they are.
    """
    rewards = np.asarray(rewards, dtype=float)
    guided = np.asarray(guided, dtype=bool)
    episodes = np.arange(1, rewards.size + 1)
    keep = (~guided) & (episodes <= int(limit))
    return episodes[keep], rewards[keep]


def plot_rewards_display(rewards, guided, stem, title, window=200, limit=DISPLAY_EPISODE_LIMIT):
    """Course-report reward curve: first ``limit`` episodes, actor episodes only."""
    episodes, actor = actor_reward_series(rewards, guided, limit=limit)
    figure, axis = plt.subplots(figsize=(8.2, 4.6), constrained_layout=True)
    axis.plot(episodes, actor, color="0.72", linewidth=0.6, alpha=0.6,
              label="Actor episode return")
    if actor.size >= window:
        kernel = np.ones(window, dtype=float) / window
        averages = np.convolve(actor, kernel, mode="valid")
        axis.plot(episodes[window - 1:], averages, color="#0072B2", linewidth=2.0,
                  label=f"{window}-episode moving average")
    axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.6)
    axis.set_xlim(0, limit)
    axis.set_xlabel("Training episode")
    axis.set_ylabel("Episode return")
    axis.set_title(f"{title} (first {limit} episodes, actor episodes only)")
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    _save_pair(figure, stem)


def _style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "figure.dpi": 120,
    })


def _save_pair(figure, stem):
    stem = Path(stem)
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def trajectory_title(title, episode, report):
    status = "success" if report["success"] else f"infeasible: {report['termination_reason']}"
    return f"{title}: eval_best episode {episode} ({status})"


def plot_objectives(series, stem, title):
    episodes = series["episodes"]
    success = series["success"]
    failed = ~success
    figure, axes = plt.subplots(2, 1, figsize=(8.2, 7.0), sharex=True, constrained_layout=True)

    axes[0].plot(episodes, series["task_time"], color="#0072B2", linewidth=1.5, label=r"$T$")
    axes[0].plot(episodes, series["delay"], color="#D55E00", linewidth=1.5, label=r"$\tau_{\mathrm{total}}$")
    axes[0].scatter(episodes[failed], series["task_time"][failed], marker="x", color="#0072B2", s=24, alpha=0.55)
    axes[0].scatter(episodes[failed], series["delay"][failed], marker="x", color="#D55E00", s=24, alpha=0.55)
    axes[0].set_ylabel("Time (s)")
    axes[0].set_title("Mission time and transmission delay")
    axes[0].grid(alpha=0.25)
    axes[0].legend(ncol=2)

    axes[1].plot(episodes, series["objective_raw"], color="0.65", linewidth=1.0,
                 linestyle="--", label="Raw termination value")
    axes[1].scatter(episodes[failed], series["objective_raw"][failed], marker="x",
                    color="#CC3311", s=30, label="Infeasible evaluation")
    if np.any(success):
        axes[1].plot(episodes, series["objective_feasible"], color="#009E73", marker="o",
                     markersize=3.5, linewidth=1.8, label=r"Feasible $T+\tau_{\mathrm{total}}$")
    else:
        axes[1].text(0.03, 0.94, "No feasible evaluation", transform=axes[1].transAxes,
                     ha="left", va="top", color="#CC3311", weight="bold",
                     bbox={"facecolor": "white", "edgecolor": "#CC3311", "alpha": 0.9,
                           "boxstyle": "round,pad=0.25"})
    axes[1].set_xlabel("Training episode")
    axes[1].set_ylabel(r"$T+\tau_{\mathrm{total}}$ (s)")
    axes[1].set_title("Optimization objective (feasible evaluations emphasized)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(ncol=2, loc="best")
    figure.suptitle(title)
    _save_pair(figure, stem)


def plot_rewards(rewards, series, stem, title, window=200):
    train_episodes = np.arange(1, len(rewards) + 1)
    average_episodes, averages = moving_average(rewards, window=window)
    success = series["success"]
    failed = ~success
    figure, axis = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    axis.plot(train_episodes, rewards, color="0.72", linewidth=0.55, alpha=0.5,
              label="Training return")
    if averages.size:
        axis.plot(average_episodes, averages, color="#0072B2", linewidth=2.0,
                  label=f"{window}-episode moving average")
    axis.scatter(series["episodes"][failed], series["returns"][failed], marker="x",
                 color="#CC3311", s=28, alpha=0.75, label="Actor evaluation (infeasible)")
    if np.any(success):
        axis.scatter(series["episodes"][success], series["returns"][success], marker="*",
                     color="#009E73", edgecolor="black", linewidth=0.25, s=65,
                     label="Actor evaluation (feasible)", zorder=4)
    axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.6)
    axis.set_xlabel("Training episode")
    axis.set_ylabel("Episode return")
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, loc="best")
    _save_pair(figure, stem)


REPORT_KEYS = (
    "success", "termination_reason", "visited_inspection_points",
    "uploaded_mb", "task_time_s", "transmission_delay_s",
    "outage_time_s", "flight_distance_m", "total_return",
)


def _generate_d3qn_figures(run_dir, config):
    """Figures for a discrete D3QN run; its eval history keeps the best entry under ``best``."""
    history = json.loads((run_dir / "eval_history.json").read_text(encoding="utf-8"))
    best = history.get("best")
    series = evaluation_series(history)
    archive = np.load(run_dir / "training_history.npz")
    rewards = np.asarray(archive["episode_rewards"], dtype=float)
    guided = (np.asarray(archive["episode_guided"], dtype=bool)
              if "episode_guided" in archive else np.zeros(rewards.size, dtype=bool))
    output = run_dir / "paper_figures"
    output.mkdir(parents=True, exist_ok=True)

    load_label = str(config["data_level"]).capitalize()
    title = f"D3QN, {load_label} load, Variant {config['variant']}"
    plot_objectives(series, output / "objective_vs_episode", title)
    plot_rewards(rewards, series, output / "reward_vs_episode", title)
    plot_rewards_display(rewards, guided, output / "reward_vs_episode_0_2000", title)

    eval_best_report = None
    checkpoint = d3qn_checkpoint_paths(run_dir)["eval_best"]
    if best is not None and checkpoint.is_file():
        context = load_d3qn_run_context(run_dir)
        report, trace = evaluate_d3qn_checkpoint(context, checkpoint)
        plot_trajectory(
            context,
            trace,
            output / "trajectory_eval_best_dual_maps",
            trajectory_title(title, best["episode"], report),
        )
        np.savez(output / "trajectory_eval_best.npz", **trace)
        eval_best_report = {key: report[key] for key in REPORT_KEYS}

    summary = {
        "run_dir": str(run_dir),
        "algorithm": "d3qn",
        "data_level": config["data_level"],
        "variant": config["variant"],
        "evaluation_count": int(len(series["episodes"])),
        "successful_evaluations": int(series["success"].sum()),
        "eval_best_episode": int(best["episode"]) if best else None,
        "eval_best": eval_best_report,
    }
    (output / "figure_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def method_title(algorithm, config):
    """Figure label: paper method name, qualified by a non-IGA sequence baseline."""
    base = method_display_name(algorithm)
    sequence = str(config.get("sequence_method", ""))
    if "distance TSP" in sequence:
        return f"Distance-TSP+{base}"
    if "annealing" in sequence:
        return f"DRL-SA ({base} trajectory)"
    return base


def generate_run_figures(run_dir):
    run_dir = Path(run_dir).resolve()
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    if config.get("algorithm") == "d3qn":
        return _generate_d3qn_figures(run_dir, config)
    context = load_run_context(run_dir)
    history = json.loads((run_dir / "eval_history.json").read_text(encoding="utf-8"))
    series = evaluation_series(history)
    reward_archive = np.load(run_dir / "UAV_TD3_improved.npz")
    rewards = np.asarray(reward_archive["ep_rewards"], dtype=float)
    output = run_dir / "paper_figures"
    output.mkdir(parents=True, exist_ok=True)

    load_label = str(context.config["data_level"]).capitalize()
    method_label = method_title(context.algorithm, context.config)
    title = f"{method_label}, {load_label} load, Variant {context.variant}"
    plot_objectives(series, output / "objective_vs_episode", title)
    plot_rewards(rewards, series, output / "reward_vs_episode", title)
    guided = (np.asarray(reward_archive["ep_guided"], dtype=bool)
              if "ep_guided" in reward_archive else np.zeros(rewards.size, dtype=bool))
    plot_rewards_display(rewards, guided, output / "reward_vs_episode_0_2000", title)

    paths = checkpoint_paths(run_dir, context.algorithm, context.config["episodes"])
    if "eval_best" not in paths or not paths["eval_best"].is_file():
        raise FileNotFoundError(f"eval_best actor not found in {run_dir / 'models'}")
    report, trace = evaluate_checkpoint(context, paths["eval_best"])
    plot_trajectory(
        context,
        trace,
        output / "trajectory_eval_best_dual_maps",
        trajectory_title(title, history["eval_best"]["episode"], report),
    )
    np.savez(output / "trajectory_eval_best.npz", **trace)

    summary = {
        "run_dir": str(run_dir),
        "algorithm": context.algorithm,
        "data_level": context.config["data_level"],
        "variant": context.variant,
        "evaluation_count": int(len(series["episodes"])),
        "successful_evaluations": int(series["success"].sum()),
        "eval_best_episode": int(history["eval_best"]["episode"]),
        "eval_best": {
            key: report[key]
            for key in (
                "success", "termination_reason", "visited_inspection_points",
                "uploaded_mb", "task_time_s", "transmission_delay_s",
                "outage_time_s", "flight_distance_m", "total_return",
            )
        },
    }
    (output / "figure_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="journal_results/formal-v5-restart")
    parser.add_argument("--run_dir", action="append", default=[])
    args = parser.parse_args()
    _style()
    root = Path(args.root)
    if not root.is_absolute():
        root = ROOT / root
    root.mkdir(parents=True, exist_ok=True)
    run_dirs = [Path(path) for path in args.run_dir] if args.run_dir else [root / suffix for suffix in DEFAULT_RUNS]
    summaries = []
    for run_dir in run_dirs:
        summary = generate_run_figures(run_dir)
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False))
    (root / "paper_figure_summaries.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
