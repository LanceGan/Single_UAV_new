"""Deterministic, guide-free evaluation for trained continuous-control policies."""

import argparse
import contextlib
import hashlib
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from World import World
from td3_actor import Actor


ROOT = Path(__file__).resolve().parent
REQUIRED_CONFIG_FIELDS = {
    "algorithm",
    "data_level",
    "episodes",
    "map_hashes",
    "net_width",
    "order",
    "outage_max_s",
    "payloads_mb",
    "state_dim",
    "uav_h",
}


@dataclass(frozen=True)
class RunContext:
    run_dir: Path
    config: dict
    algorithm: str
    state_dim: int
    net_width: int
    order: tuple
    payloads_mb: np.ndarray
    outage_max_s: float
    total_steps: int = 1000
    length: float = 20.0
    width: float = 20.0
    delta_t: float = 0.5
    dist_max: float = 0.25


def _map_hashes():
    data_dir = ROOT / "results" / "datas"
    return {
        name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
        for name in ("Radio_datas.npz", "Radio_datas_A2G.npz")
    }


def load_run_context(run_dir):
    run_dir = Path(run_dir).resolve()
    config_path = run_dir / "run_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"run_config.json not found in {run_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    missing = sorted(REQUIRED_CONFIG_FIELDS - config.keys())
    if missing:
        raise ValueError(f"run_config.json missing required fields: {', '.join(missing)}")
    if config["algorithm"] not in {"td3", "ddpg"}:
        raise ValueError(f"unsupported algorithm: {config['algorithm']!r}")
    if int(config["state_dim"]) != 8:
        raise ValueError(f"expected the current 8-dimensional state, got {config['state_dim']}")
    order = tuple(int(value) for value in config["order"])
    payloads = np.asarray(config["payloads_mb"], dtype=float)
    if len(order) != len(payloads) or sorted(order) != list(range(1, len(payloads) + 1)):
        raise ValueError("run_config.json has inconsistent order and payloads_mb")
    actual_hashes = _map_hashes()
    if config["map_hashes"] != actual_hashes:
        raise ValueError("fixed radio maps do not match the maps recorded during training")
    return RunContext(
        run_dir=run_dir,
        config=config,
        algorithm=config["algorithm"],
        state_dim=int(config["state_dim"]),
        net_width=int(config["net_width"]),
        order=order,
        payloads_mb=payloads,
        outage_max_s=float(config["outage_max_s"]),
    )


def checkpoint_paths(run_dir, algorithm, episodes):
    if algorithm not in {"td3", "ddpg"}:
        raise ValueError(f"unsupported algorithm: {algorithm!r}")
    model_dir = Path(run_dir) / "models"
    prefix = f"{algorithm}_{algorithm}_actor"
    return {
        "final": model_dir / f"{prefix}{episodes}.pth",
        "training_best": model_dir / f"{prefix}best.pth",
    }


def load_actor(checkpoint, state_dim, net_width):
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    max_action = torch.tensor([math.pi, 0.25], dtype=torch.float32)
    actor = Actor(state_dim, 2, net_width, max_action).cpu()
    try:
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
        actor.load_state_dict(state_dict, strict=True)
    except (RuntimeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"checkpoint is incompatible with run_config.json: {checkpoint}") from error
    actor.eval()
    return actor


def classify_termination(
    *,
    terminal,
    step,
    max_steps,
    outage_s,
    outage_max_s,
    visited_points,
    expected_points,
    uploaded_mb,
    expected_upload_mb,
):
    if outage_s > outage_max_s:
        return False, "qos_violation"
    if step >= max_steps and not terminal:
        return False, "timeout"
    full_upload = uploaded_mb >= expected_upload_mb - 1e-7
    all_points = visited_points >= expected_points
    if terminal and full_upload and all_points:
        return True, "success"
    if not full_upload and all_points:
        return False, "incomplete_upload_at_target"
    if not all_points:
        return False, "premature_termination"
    return False, "invalid_terminal_state"


def _build_world(context):
    return World(
        length=context.length,
        width=context.width,
        uav_num=1,
        dist_max=context.dist_max,
        delta_t=context.delta_t,
        t=context.total_steps,
        uav_h=float(context.config["uav_h"]),
        data_size=300,
        data_sizes=context.payloads_mb,
        outage_max_s=context.outage_max_s,
        ini_loc=[2.62, 2.65],
        end_loc=[18.95, 11.12],
        users_name=str(ROOT / "results" / "datas" / "Users_6.txt"),
        BS_loc=np.array(
            [[4.77, 4.56, 0.25], [14.89, 4.89, 0.25], [4.63, 13.78, 0.25], [13.48, 14.89, 0.25]]
        ),
        traverse_sequence=list(context.order),
    )


def evaluate_checkpoint(context, checkpoint):
    actor = load_actor(checkpoint, context.state_dim, context.net_width)
    with contextlib.redirect_stdout(io.StringIO()):
        world = _build_world(context)
        state, step = world.reset()
    positions = [[world.UAVs[0].x, world.UAVs[0].y]]
    times = [0.0]
    outage_times = [0.0]
    pending = [float(world.data_size)]
    visited = [int(world.Traverse_order)]
    a2g_snr = [float(world.current_a2g_snr)]
    g2a_outage = [float(world.current_g2a_outage_probability)]
    actions = [[0.0, 0.0]]
    visits = []
    total_reward = 0.0
    uploaded_mb = 0.0
    flight_distance_m = 0.0
    done = False
    terminal = False

    while not done:
        with torch.no_grad():
            action = actor(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))[0].numpy()
        action = np.clip(action, [-math.pi, 0.0], [math.pi, context.dist_max])
        previous_position = np.array([world.UAVs[0].x, world.UAVs[0].y])
        previous_visit = world.Traverse_order
        with contextlib.redirect_stdout(io.StringIO()):
            state, reward, done, step, terminal = world.step_inside(action, state, step)
        position = np.array([world.UAVs[0].x, world.UAVs[0].y])
        flight_distance_m += float(np.linalg.norm(position - previous_position) * 100.0)
        total_reward += float(reward)
        uploaded_mb += float(world.last_uploaded_data)
        positions.append(position.tolist())
        times.append(float(step * context.delta_t))
        outage_times.append(float(world.out_time))
        pending.append(float(world.data_size))
        visited.append(int(world.Traverse_order))
        a2g_snr.append(float(world.current_a2g_snr))
        g2a_outage.append(float(world.current_g2a_outage_probability))
        actions.append(action.tolist())
        if world.Traverse_order != previous_visit or terminal:
            node = context.order[previous_visit] if previous_visit < len(context.order) else "end"
            visits.append(
                {
                    "node": node,
                    "time_s": float(step * context.delta_t),
                    "outage_s": float(world.out_time),
                    "uploaded_total_mb": float(uploaded_mb),
                }
            )

    success, termination_reason = classify_termination(
        terminal=terminal,
        step=step,
        max_steps=context.total_steps,
        outage_s=world.out_time,
        outage_max_s=context.outage_max_s,
        visited_points=world.Traverse_order,
        expected_points=len(context.order),
        uploaded_mb=uploaded_mb,
        expected_upload_mb=float(context.payloads_mb.sum()),
    )
    time_s = float(step * context.delta_t)
    reward_terms = {key: float(value) for key, value in world.reward_summary().items()}
    trace = {
        "positions": np.asarray(positions, dtype=float),
        "time_s": np.asarray(times, dtype=float),
        "outage_s": np.asarray(outage_times, dtype=float),
        "pending_mb": np.asarray(pending, dtype=float),
        "visited_points": np.asarray(visited, dtype=int),
        "a2g_snr_db": np.asarray(a2g_snr, dtype=float),
        "g2a_outage_probability": np.asarray(g2a_outage, dtype=float),
        "actions": np.asarray(actions, dtype=float),
    }
    report = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "algorithm": context.algorithm,
        "policy_mode": "deterministic actor only (no MapGuide, no exploration noise)",
        "success": success,
        "termination_reason": termination_reason,
        "visited_inspection_points": int(min(world.Traverse_order, len(context.order))),
        "expected_inspection_points": len(context.order),
        "uploaded_mb": float(round(uploaded_mb, 8)),
        "expected_upload_mb": float(context.payloads_mb.sum()),
        "task_time_s": time_s,
        "trajectory_steps": int(step),
        "flight_distance_m": flight_distance_m,
        "average_speed_mps": flight_distance_m / time_s if time_s else 0.0,
        "outage_time_s": float(world.out_time),
        "outage_limit_s": context.outage_max_s,
        "outage_fraction": float(world.out_time / time_s) if time_s else 0.0,
        "transmission_delay_s": float(world.trans_delay * context.delta_t),
        "boundary_violations": int(world.fa),
        "mean_a2g_snr_db": float(np.mean(a2g_snr)),
        "min_a2g_snr_db": float(np.min(a2g_snr)),
        "mean_g2a_outage_probability": float(np.mean(g2a_outage)),
        "total_return": total_reward,
        "reward_terms": reward_terms,
        "reward_accounting_error": float(abs(sum(reward_terms.values()) - total_reward)),
        "inspection_order": list(context.order),
        "payloads_mb": context.payloads_mb.tolist(),
        "visits": visits,
        "map_hashes": _map_hashes(),
    }
    return report, trace


def plot_trajectory(context, trace, output_stem, title):
    data_dir = ROOT / "results" / "datas"
    g2a = np.load(data_dir / "Radio_datas.npz")
    a2g = np.load(data_dir / "Radio_datas_A2G.npz")
    users = np.loadtxt(data_dir / "Users_6.txt")[:, :2] * 100.0
    trajectory = trace["positions"] * 100.0
    start = np.array([2.62, 2.65]) * 100.0
    end = np.array([18.95, 11.12]) * 100.0

    figure, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    panels = (
        (axes[0], 1.0 - g2a["arr_0"], g2a, "G2A coverage probability", "Coverage probability"),
        (axes[1], a2g["arr_1"], a2g, "A2G SNR", "SNR (dB)"),
    )
    for axis, background, map_data, panel_title, color_label in panels:
        contour = axis.contourf(
            np.asarray(map_data["arr_2"]) * 10.0,
            np.asarray(map_data["arr_3"]) * 10.0,
            background,
            levels=30,
        )
        axis.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color="white",
            linewidth=2.0,
            label=f"{context.algorithm.upper()} trajectory",
        )
        axis.scatter(users[:, 0], users[:, 1], c="black", marker="^", s=55, label="Inspection point", zorder=3)
        axis.scatter(*start, c="red", s=70, label="Start", zorder=4)
        axis.scatter(*end, c="orange", s=70, label="End", zorder=4)
        for visit_number, node_index in enumerate(context.order, start=1):
            axis.annotate(str(visit_number), users[node_index - 1], xytext=(5, 5), textcoords="offset points", color="white", weight="bold")
        axis.set_title(panel_title)
        axis.set_xlabel("X (m)")
        axis.set_ylabel("Y (m)")
        axis.set_aspect("equal")
        axis.legend(loc="upper right", fontsize=8)
        figure.colorbar(contour, ax=axis, label=color_label)
    figure.suptitle(title)
    output_stem = Path(output_stem)
    figure.savefig(output_stem.with_suffix(".png"), dpi=300)
    figure.savefig(output_stem.with_suffix(".pdf"))
    plt.close(figure)


def _save_evaluation(context, checkpoint, label, output_dir):
    report, trace = evaluate_checkpoint(context, checkpoint)
    target = output_dir / label
    target.mkdir(parents=True, exist_ok=True)
    (target / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez(target / "trajectory.npz", **trace)
    plot_trajectory(
        context,
        trace,
        target / "trajectory_dual_maps",
        f"{context.algorithm.upper()} {label}: {report['termination_reason']}",
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", default="journal_results/formal-v2/low/td3")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    context = load_run_context(args.run_dir)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else context.run_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = checkpoint_paths(context.run_dir, context.algorithm, context.config["episodes"])
    reports = {label: _save_evaluation(context, path, label, output_dir) for label, path in checkpoints.items()}
    history_path = context.run_dir / "UAV_TD3_improved.npz"
    training_best_episode = None
    if history_path.is_file():
        history = np.load(history_path)
        if "ep_rewards" in history:
            training_best_episode = int(np.argmax(history["ep_rewards"]) + 1)
    summary = {
        "run_dir": str(context.run_dir),
        "algorithm": context.algorithm,
        "training_best_episode": training_best_episode,
        "evaluation": {
            label: {
                key: report[key]
                for key in (
                    "checkpoint",
                    "success",
                    "termination_reason",
                    "visited_inspection_points",
                    "uploaded_mb",
                    "task_time_s",
                    "outage_time_s",
                    "total_return",
                )
            }
            for label, report in reports.items()
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
