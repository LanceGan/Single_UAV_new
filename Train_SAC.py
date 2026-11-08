"""Train SACC on the shared UAV inspection MDP."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from ReplayBuffer import ReplayBuffer
from SAC import SAC
from World import World
from demonstration import linear_decay
from evaluate_policy import (
    A2G_BLIND_STATE_SCHEMA,
    STATE_SCHEMA,
    _build_world,
    load_run_context,
    rollout,
    selection_key,
)
from exploration import exploration_scales
from map_guidance import MapGuide
from mission_preflight import run_preflight
from sequence_optimizers import (
    distance_astar_order,
    distance_pso_order,
    distance_sa_order,
    distance_tsp_order,
    eega_order,
    ga_proxy_order,
    priga_order,
    yang_order,
)
from training_config import build_output_paths, load_fixed_payloads


ROOT = Path(__file__).resolve().parent
START = [2.62, 2.65]
END = [18.95, 11.12]
BASE_STATIONS = np.asarray(
    [
        [4.77, 4.56, 0.25],
        [14.89, 4.89, 0.25],
        [4.63, 13.78, 0.25],
        [13.48, 14.89, 0.25],
    ],
    dtype=float,
)


def create_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_level", choices=("low", "medium", "high"), default="low")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--output_root", default="journal_results/v5-models/sac")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eval_interval", type=int, default=25)
    parser.add_argument("--checkpoint_interval", type=int, default=50)
    parser.add_argument("--uav_h", type=float, default=1.0)
    parser.add_argument("--outage_max_s", type=float, default=6.0)
    parser.add_argument("--outage_probability_threshold", type=float, default=0.5)
    parser.add_argument("--total_steps", type=int, default=2000)
    parser.add_argument(
        "--qos_enforcement",
        choices=World.QOS_ENFORCEMENT_MODES,
        default="continue",
    )
    parser.add_argument("--objective_w1", type=float, default=0.5)
    parser.add_argument("--objective_w2", type=float, default=0.5)
    parser.add_argument("--variant", choices=World.VARIANTS, default="A")
    parser.add_argument("--a2g_blind", action="store_true")
    parser.add_argument(
        "--sequence_method",
        choices=(
            "priga", "iga", "ga", "pso", "sa_distance", "astar",
            "eega", "yang", "distance_tsp",
        ),
        default="priga",
    )
    parser.add_argument("--ga_iterations", type=int, default=500)
    parser.add_argument("--teacher", choices=("mapguide", "none"), default="mapguide")
    parser.add_argument("--pretrain_demo_episodes", type=int, default=40)
    parser.add_argument("--pretrain_updates", type=int, default=20_000)
    parser.add_argument("--guided_warmup_episodes", type=int, default=400)
    parser.add_argument("--guide_prob", type=float, default=0.5)
    parser.add_argument("--guide_min_prob", type=float, default=0.5)
    parser.add_argument("--bc_weight", type=float, default=1.0)
    parser.add_argument("--bc_min_weight", type=float, default=0.25)
    parser.add_argument("--bc_decay_episodes", type=int, default=600)
    parser.add_argument("--demo_fraction", type=float, default=0.25)
    parser.add_argument("--demo_noise_scale", type=float, default=0.25)
    parser.add_argument("--net_width", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--critic_learning_rate", type=float, default=3e-4)
    parser.add_argument("--alpha_learning_rate", type=float, default=3e-4)
    parser.add_argument("--init_alpha", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--buffer", type=int, default=200_000)
    parser.add_argument("--train_start", type=int, default=1000)
    parser.add_argument("--train_frequency", type=int, default=2)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--preflight_only", action="store_true")
    parser.add_argument("--allow_preflight_failure", action="store_true")
    parser.add_argument("--device", default=None)
    return parser


def build_sequence(args, payloads):
    if args.sequence_method in ("priga", "iga"):
        return priga_order(
            payloads,
            seed=args.seed,
            generations=args.ga_iterations,
        )
    if args.sequence_method in ("ga",):
        return ga_proxy_order(seed=args.seed)
    if args.sequence_method == "pso":
        return distance_pso_order(seed=args.seed)
    if args.sequence_method == "sa_distance":
        return distance_sa_order(seed=args.seed)
    if args.sequence_method == "astar":
        return distance_astar_order()
    if args.sequence_method == "eega":
        return eega_order(payloads, seed=args.seed)
    if args.sequence_method == "yang":
        return yang_order(payloads)
    if args.sequence_method == "distance_tsp":
        return distance_tsp_order()
    raise ValueError(f"unsupported sequence method: {args.sequence_method}")


def make_world(args, order, payloads):
    return World(
        length=20,
        width=20,
        uav_num=1,
        dist_max=0.25,
        delta_t=0.5,
        t=int(args.total_steps),
        uav_h=args.uav_h,
        data_size=300,
        data_sizes=payloads,
        outage_max_s=args.outage_max_s,
        ini_loc=START,
        end_loc=END,
        users_name=str(ROOT / "results" / "datas" / "Users_6.txt"),
        BS_loc=BASE_STATIONS,
        traverse_sequence=list(order),
        variant=args.variant,
        a2g_blind=args.a2g_blind,
        objective_w1=args.objective_w1,
        objective_w2=args.objective_w2,
        outage_probability_threshold=args.outage_probability_threshold,
        qos_enforcement=args.qos_enforcement,
    )


def _guided_action(world, guide, episode, episodes, noise_scale):
    uav = world.UAVs[0]
    target_distance = float(np.linalg.norm(
        np.asarray([uav.x, uav.y]) - world.target_loc
    ))
    angle_noise, distance_noise = exploration_scales(
        episode, max(int(episodes), 1), target_distance
    )
    absolute = guide.action(
        angle_noise * noise_scale,
        distance_noise * noise_scale,
        np.random,
    )
    return np.clip(
        world.absolute_to_action(absolute),
        [-np.pi, 0.0],
        [np.pi, world.dist_max],
    )


def save_actor(agent, model_dir, label):
    agent.save_actor(model_dir / f"sac_sac_actor{label}.pth")


def main():
    args = create_parser().parse_args()
    if args.episodes <= 0 or args.eval_interval <= 0:
        raise ValueError("episodes and eval_interval must be positive")
    if not 0.0 <= args.guide_min_prob <= args.guide_prob <= 1.0:
        raise ValueError("MapGuide probabilities must satisfy 0 <= min <= max <= 1")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    payloads = load_fixed_payloads(args.data_level)
    order, sequence_meta = build_sequence(args, payloads)
    order = [int(node) for node in order]
    output_paths = build_output_paths(
        args.output_root, args.data_level, "sac", variant=args.variant
    )
    run_dir = output_paths["root"]
    model_dir = run_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(output_paths["logs"])

    preflight = run_preflight(
        args.data_level,
        run_dir / "preflight",
        order=order,
        outage_max_s=args.outage_max_s,
        total_steps=args.total_steps,
        a2g_blind=args.a2g_blind,
        outage_probability_threshold=args.outage_probability_threshold,
        qos_enforcement=args.qos_enforcement,
    )
    if not preflight["success"] and not args.allow_preflight_failure:
        writer.close()
        raise RuntimeError(
            "Full mission preflight failed; SAC training was not started. "
            "Pass --allow_preflight_failure for a completion-oriented run."
        )
    if args.preflight_only:
        writer.close()
        return

    world = make_world(args, order, payloads)
    agent = SAC(
        env_with_Dead=True,
        state_dim=world.state_dim,
        action_dim=2,
        max_action=np.asarray([np.pi, world.dist_max], dtype=np.float32),
        train_path=output_paths["logs"],
        gamma=args.gamma,
        net_width=args.net_width,
        a_lr=args.learning_rate,
        c_lr=args.critic_learning_rate,
        alpha_lr=args.alpha_learning_rate,
        Q_batchsize=args.batch_size,
        tau=args.tau,
        init_alpha=args.init_alpha,
        device=args.device,
    )
    replay = ReplayBuffer(
        world.state_dim,
        2,
        max_size=args.buffer,
        demo_max_size=args.buffer,
        demo_fraction=args.demo_fraction,
    )
    config = {
        **vars(args),
        "algorithm": "sac",
        "base_agent": "sac",
        "state_dim": int(world.state_dim),
        "state_schema": A2G_BLIND_STATE_SCHEMA if args.a2g_blind else STATE_SCHEMA,
        "qos_constraint": "max_contiguous_outage",
        "order": order,
        "payloads_mb": payloads.tolist(),
        "sequence_method": sequence_meta["method"],
        "sequence_meta": sequence_meta,
        "map_hashes": preflight["map_hashes"],
        "preflight_success": bool(preflight["success"]),
        "preflight_allowed_failure": bool(args.allow_preflight_failure and not preflight["success"]),
        "model_selection": "eval_best: feasible completion, then complete-infeasible, then progress",
        "evaluation_control": "deterministic policy mean; no MapGuide or sampling",
        "reward_schema": "shared v4-seconds communication-aware reward",
        "demonstration": "persistent MapGuide demo replay and BC auxiliary loss",
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    def collect_episode(episode, guided):
        state, step = world.reset()
        guide = MapGuide(world, use_a2g=not args.a2g_blind) if guided else None
        done = False
        episode_reward = 0.0
        episode_length = 0
        while not done:
            if guided:
                action = _guided_action(
                    world,
                    guide,
                    episode,
                    args.episodes,
                    args.demo_noise_scale,
                )
            else:
                action = agent.select_action(state, deterministic=False)
            previous_state = state
            with contextlib.redirect_stdout(io.StringIO()):
                state, reward, done, step, _ = world.step_inside(action, state, step)
            replay.add(
                previous_state,
                action,
                float(reward),
                state,
                float(done),
                demo=guided,
            )
            episode_reward += float(reward)
            episode_length += 1
        return episode_reward, episode_length

    # Demo-only pretraining.
    for episode in range(1, int(args.pretrain_demo_episodes) + 1):
        collect_episode(episode, guided=True)
    for _ in range(int(args.pretrain_updates)):
        if replay.total_size:
            agent.train(replay)

    eval_context = load_run_context(run_dir)
    eval_world = _build_world(eval_context)
    eval_history = []
    eval_best = None

    def evaluate(episode):
        nonlocal eval_best
        report, _ = rollout(
            eval_context,
            eval_world,
            lambda state: agent.select_action(state, deterministic=True),
        )
        item = {"episode": episode, **report}
        eval_history.append(item)
        if eval_best is None or selection_key(
            report, args.objective_w1, args.objective_w2
        ) < selection_key(eval_best, args.objective_w1, args.objective_w2):
            eval_best = item
            save_actor(agent, model_dir, "eval_best")
        public_best = {
            key: value for key, value in eval_best.items()
            if key not in ("visits", "reward_terms", "map_hashes")
        }
        (run_dir / "eval_history.json").write_text(
            json.dumps(
                {"eval_best": public_best, "history": eval_history},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        writer.add_scalar("eval/success", float(report["success"]), episode)
        writer.add_scalar("eval/mission_complete", float(report["mission_complete"]), episode)
        writer.add_scalar(
            "eval/objective_s",
            report["task_time_s"] + report["transmission_delay_s"],
            episode,
        )

    rewards, lengths, guided_flags, episodes = [], [], [], []
    global_step = 0
    for episode in range(1, args.episodes + 1):
        guide_prob = linear_decay(
            args.guide_prob,
            episode - 1,
            args.guided_warmup_episodes,
            floor=args.guide_min_prob,
        )
        guided = args.teacher == "mapguide" and np.random.random() < guide_prob
        agent.bc_weight = linear_decay(
            args.bc_weight,
            episode - 1,
            args.bc_decay_episodes,
            floor=args.bc_min_weight,
        )
        reward, length = collect_episode(episode, guided)
        global_step += length
        rewards.append(reward)
        lengths.append(length)
        guided_flags.append(guided)
        episodes.append(episode)
        writer.add_scalar("episode/reward", reward, episode)
        if replay.size >= args.train_start:
            for _ in range(max(1, length // max(args.train_frequency, 1))):
                if replay.total_size >= args.batch_size:
                    agent.train(replay)
        if episode % args.checkpoint_interval == 0:
            save_actor(agent, model_dir, episode)
        if episode % args.eval_interval == 0 or episode == args.episodes:
            evaluate(episode)

    save_actor(agent, model_dir, args.episodes)
    np.savez(
        run_dir / "training_history.npz",
        episode_rewards=np.asarray(rewards),
        episode_lengths=np.asarray(lengths, dtype=int),
        episode_guided=np.asarray(guided_flags, dtype=bool),
        episodes=np.asarray(episodes, dtype=int),
        global_steps=global_step,
    )
    np.savez(
        run_dir / "UAV_TD3_improved.npz",
        ep_rewards=np.asarray(rewards),
        ep_lengths=np.asarray(lengths, dtype=int),
        ep_guided=np.asarray(guided_flags, dtype=bool),
    )
    writer.close()
    print(run_dir, flush=True)


if __name__ == "__main__":
    main()
