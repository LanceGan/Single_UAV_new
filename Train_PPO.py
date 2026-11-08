"""Train a continuous-action PPO baseline on the shared UAV inspection task."""

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

from PPO import PPOAgent, generalized_advantage_estimate
from World import World
from evaluate_policy import (
    A2G_BLIND_STATE_SCHEMA,
    STATE_SCHEMA,
    _build_world,
    load_run_context,
    rollout,
    selection_key,
)
from map_guidance import MapGuide
from mission_preflight import run_preflight
from exploration import exploration_scales
from demonstration import linear_decay
from sequence_optimizers import (
    distance_sa_order,
    distance_tsp_order,
    eega_order,
    sa_order,
    service_sa_order,
    yang_order,
)
from sequence_runner import solve_ga_order
from training_config import build_output_paths, load_fixed_payloads


ROOT = Path(__file__).resolve().parent
START = [2.62, 2.65]
END = [18.95, 11.12]
BASE_STATIONS = np.array([
    [4.77, 4.56, 0.25],
    [14.89, 4.89, 0.25],
    [4.63, 13.78, 0.25],
    [13.48, 14.89, 0.25],
])


def create_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_level", choices=("low", "medium", "high"), default="low")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--output_root", default="journal_results/v4-models/ppo")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eval_interval", type=int, default=25)
    parser.add_argument("--uav_h", type=float, default=1.0)
    parser.add_argument("--outage_max_s", type=float, default=6.0)
    parser.add_argument("--objective_w1", type=float, default=0.5)
    parser.add_argument("--objective_w2", type=float, default=0.5)
    parser.add_argument("--variant", choices=World.VARIANTS, default="A")
    parser.add_argument("--a2g_blind", action="store_true")
    parser.add_argument("--sequence_method",
                        choices=("iga", "distance_tsp", "sa", "sa_service",
                                 "sa_distance", "eega", "yang"),
                        default="iga")
    parser.add_argument("--ga_iterations", type=int, default=500)
    parser.add_argument("--teacher", choices=("mapguide", "none"), default="mapguide")
    parser.add_argument("--pretrain_demo_episodes", type=int, default=40,
                        help="MapGuide demonstration episodes collected before online training")
    parser.add_argument("--pretrain_updates", type=int, default=20000,
                        help="supervised actor updates on demonstrations before online training")
    parser.add_argument("--demo_noise_scale", type=float, default=0.25)
    parser.add_argument("--guide_prob", type=float, default=0.5,
                        help="initial whole-episode MapGuide probability during online training")
    parser.add_argument("--guide_min_prob", type=float, default=0.2,
                        help="MapGuide probability kept for the rest of training")
    parser.add_argument("--guided_warmup_episodes", type=int, default=400)
    parser.add_argument("--demo_fraction", type=float, default=0.25,
                        help="demonstration share of each BC auxiliary batch")
    parser.add_argument("--bc_weight", type=float, default=1.0,
                        help="initial behavior-cloning auxiliary weight")
    parser.add_argument("--bc_min_weight", type=float, default=0.25)
    parser.add_argument("--bc_decay_episodes", type=int, default=600)
    parser.add_argument("--net_width", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--critic_learning_rate", type=float, default=1e-3)
    parser.add_argument("--n_steps", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--update_epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_ratio", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--checkpoint_interval", type=int, default=50)
    parser.add_argument("--preflight_only", action="store_true")
    parser.add_argument("--device", default=None)
    return parser


def build_sequence(args, payloads):
    if args.sequence_method == "distance_tsp":
        return distance_tsp_order()
    if args.sequence_method == "sa":
        return sa_order(seed=args.seed)
    if args.sequence_method == "sa_service":
        return service_sa_order(payloads, seed=args.seed)
    if args.sequence_method == "sa_distance":
        return distance_sa_order(seed=args.seed)
    if args.sequence_method == "eega":
        return eega_order(payloads, seed=args.seed)
    if args.sequence_method == "yang":
        return yang_order(payloads)
    order = solve_ga_order(payloads, iterations=args.ga_iterations)
    return order, {
        "method": "GA with exact verification/fallback",
        "uses_g2a": True,
        "uses_a2g": True,
        "uses_payloads": True,
    }


def make_world(args, order, payloads):
    return World(
        length=20,
        width=20,
        uav_num=1,
        dist_max=0.25,
        delta_t=0.5,
        t=1000,
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
    )


def collect_mapguide_demonstrations(args, world, episodes):
    states, actions = [], []
    for episode in range(1, int(episodes) + 1):
        state, step = world.reset()
        guide = MapGuide(world, use_a2g=not args.a2g_blind)
        done = False
        while not done:
            uav = world.UAVs[0]
            target_distance = float(np.linalg.norm(
                np.array([uav.x, uav.y]) - world.target_loc
            ))
            angle_noise, distance_noise = exploration_scales(
                episode, max(int(args.episodes), 1), target_distance
            )
            absolute = guide.action(
                angle_noise * args.demo_noise_scale,
                distance_noise * args.demo_noise_scale,
                np.random,
            )
            action = world.absolute_to_action(absolute)
            action = np.clip(action, [-np.pi, 0.0], [np.pi, world.dist_max])
            states.append(np.asarray(state, dtype=np.float32))
            actions.append(np.asarray(action, dtype=np.float32))
            with contextlib.redirect_stdout(io.StringIO()):
                state, _, done, step, _ = world.step_inside(action, state, step)
    return np.asarray(states), np.asarray(actions)


def train_from_buffer(agent, transitions, args, demo=None, bc_weight=0.0):
    values = np.asarray([row["value"] for row in transitions], dtype=np.float32)
    next_values = np.asarray([row["next_value"] for row in transitions], dtype=np.float32)
    rewards = np.asarray([row["reward"] for row in transitions], dtype=np.float32)
    dones = np.asarray([row["done"] for row in transitions], dtype=np.float32)
    returns, advantages = generalized_advantage_estimate(
        rewards, values, next_values, dones,
        gamma=args.gamma, lam=args.gae_lambda,
    )
    metrics = agent.update(
        states=np.asarray([row["state"] for row in transitions]),
        actions=np.asarray([row["action"] for row in transitions]),
        old_log_probs=np.asarray([row["log_prob"] for row in transitions]),
        returns=returns,
        advantages=advantages,
        epochs=args.update_epochs,
        batch_size=args.batch_size,
        demo=demo,
        bc_weight=bc_weight,
        demo_fraction=args.demo_fraction,
    )
    return metrics


def main():
    args = create_parser().parse_args()
    if args.episodes <= 0 or args.eval_interval <= 0 or args.n_steps <= 0:
        raise ValueError("episodes, eval_interval and n_steps must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    payloads = load_fixed_payloads(args.data_level)
    order, sequence_meta = build_sequence(args, payloads)
    order = [int(node) for node in order]
    if sorted(order) != list(range(1, len(payloads) + 1)):
        raise ValueError("sequence must contain every inspection point exactly once")

    output_paths = build_output_paths(
        args.output_root, args.data_level, "ppo", variant=args.variant
    )
    run_dir = output_paths["root"]
    run_dir.mkdir(parents=True, exist_ok=True)
    model_dir = run_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(output_paths["logs"])

    preflight = run_preflight(
        args.data_level,
        run_dir / "preflight",
        order=order,
        outage_max_s=args.outage_max_s,
        a2g_blind=args.a2g_blind,
    )
    if not preflight["success"]:
        raise RuntimeError("Full mission preflight failed; PPO training was not started.")
    if args.preflight_only:
        writer.close()
        return

    world = make_world(args, order, payloads)
    state_dim = int(world.state_dim)
    state_schema = A2G_BLIND_STATE_SCHEMA if args.a2g_blind else STATE_SCHEMA
    config = {
        **vars(args),
        "algorithm": "ppo",
        "base_agent": "ppo",
        "state_dim": state_dim,
        "state_schema": state_schema,
        "qos_constraint": "max_contiguous_outage",
        "order": order,
        "payloads_mb": payloads.tolist(),
        "sequence_method": sequence_meta["method"],
        "sequence_meta": sequence_meta,
        "map_hashes": preflight["map_hashes"],
        "model_selection": "eval_best: feasible first, then minimum weighted T + tau",
        "evaluation_control": "deterministic policy mean (no sampling, no guidance)",
        "reward_schema": "v4-seconds: -(T + tau) + 2x progress + milestones; max contiguous outage hard constraint; x0.01",
        "demonstration": (
            f"{args.pretrain_demo_episodes} MapGuide episodes followed by "
            f"{args.pretrain_updates} supervised actor updates, then whole-episode "
            f"MapGuide guidance with probability {args.guide_prob} -> {args.guide_min_prob} "
            f"over {args.guided_warmup_episodes} episodes and a persistent demonstration "
            f"BC auxiliary loss with weight {args.bc_weight} -> {args.bc_min_weight}"
            if args.teacher == "mapguide" else "none"
        ),
        "guide_schedule": (f"{args.guide_prob} -> {args.guide_min_prob} over {args.guided_warmup_episodes} episodes, then constant"
                           if args.teacher == "mapguide" else "none"),
        "bc_schedule": (f"{args.bc_weight} -> {args.bc_min_weight} over {args.bc_decay_episodes} episodes, then constant"
                        if args.teacher == "mapguide" else "none"),
        "demo_sampling_fraction": args.demo_fraction if args.teacher == "mapguide" else 0.0,
        "training_objective": "clipped PPO surrogate with GAE",
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=2,
        max_action=np.array([np.pi, world.dist_max], dtype=np.float32),
        net_width=args.net_width,
        actor_lr=args.learning_rate,
        critic_lr=args.critic_learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        device=args.device,
    )
    pretrain_loss = None
    if args.teacher == "mapguide" and args.pretrain_demo_episodes > 0:
        demo_world = make_world(args, order, payloads)
        demo_states, demo_actions = collect_mapguide_demonstrations(
            args, demo_world, args.pretrain_demo_episodes
        )
        pretrain_loss = agent.pretrain_actor(
            demo_states, demo_actions, updates=args.pretrain_updates,
            batch_size=args.batch_size,
        )
        (run_dir / "demonstration.json").write_text(json.dumps({
            "episodes": args.pretrain_demo_episodes,
            "transitions": int(len(demo_states)),
            "updates": args.pretrain_updates,
            "loss": pretrain_loss,
        }, indent=2), encoding="utf-8")

    eval_context = load_run_context(run_dir)
    eval_world = _build_world(eval_context)
    eval_history = []
    eval_best = None
    transitions = []
    rewards, lengths, guided, global_steps = [], [], [], []
    global_step = 0

    def save_actor(label):
        agent.save_actor(model_dir / f"ppo_ppo_actor{label}.pth")

    def evaluate(episode):
        nonlocal eval_best
        report, _ = rollout(eval_context, eval_world, agent.deterministic_action)
        item = {"episode": episode, **report}
        eval_history.append(item)
        if eval_best is None or selection_key(
            report, args.objective_w1, args.objective_w2
        ) < selection_key(eval_best, args.objective_w1, args.objective_w2):
            eval_best = item
            save_actor("eval_best")
        public_best = {
            key: value for key, value in eval_best.items()
            if key not in ("visits", "reward_terms", "map_hashes")
        }
        (run_dir / "eval_history.json").write_text(json.dumps(
            {"eval_best": public_best, "history": eval_history},
            indent=2, ensure_ascii=False,
        ), encoding="utf-8")
        writer.add_scalar("eval/success", float(report["success"]), episode)
        writer.add_scalar(
            "eval/objective_s",
            report["task_time_s"] + report["transmission_delay_s"],
            episode,
        )
        writer.add_scalar("eval/max_contiguous_outage_s", report["outage_time_s"], episode)

    demo_states, demo_actions = [], []
    episode = 1
    while episode <= args.episodes:
        guide_probability = (
            linear_decay(args.guide_prob, episode - 1, args.guided_warmup_episodes, floor=args.guide_min_prob)
            if args.teacher == "mapguide" else 0.0
        )
        guided_run = np.random.random() < guide_probability
        bc_weight = (
            linear_decay(args.bc_weight, episode - 1, args.bc_decay_episodes, floor=args.bc_min_weight)
            if args.teacher == "mapguide" else 0.0
        )
        state, step = world.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False
        guide = MapGuide(world, use_a2g=not args.a2g_blind) if guided_run else None

        while not done:
            if guided_run:
                uav = world.UAVs[0]
                target_distance = float(np.linalg.norm(
                    np.array([uav.x, uav.y]) - world.target_loc
                ))
                angle_noise, distance_noise = exploration_scales(
                    episode, max(int(args.episodes), 1), target_distance
                )
                absolute = guide.action(
                    angle_noise * args.demo_noise_scale,
                    distance_noise * args.demo_noise_scale,
                    np.random,
                )
                action = world.absolute_to_action(absolute)
                action = np.clip(action, [-np.pi, 0.0], [np.pi, world.dist_max])
                demo_states.append(np.asarray(state, dtype=np.float32))
                demo_actions.append(np.asarray(action, dtype=np.float32))
                with contextlib.redirect_stdout(io.StringIO()):
                    state, reward, done, step, _ = world.step_inside(action, state, step)
                episode_reward += float(reward)
                episode_length += 1
            else:
                action, log_prob, value = agent.sample_action(state)
                previous_state = state
                with contextlib.redirect_stdout(io.StringIO()):
                    state, reward, done, step, _ = world.step_inside(action, state, step)
                global_step += 1
                episode_reward += float(reward)
                episode_length += 1
                transitions.append({
                    "state": previous_state,
                    "action": action,
                    "log_prob": log_prob,
                    "value": value,
                    "next_value": 0.0 if done else agent.value_of(state),
                    "reward": float(reward),
                    "done": bool(done),
                })
                writer.add_scalar("rollout/action_heading", float(action[0]), global_step)
                writer.add_scalar("rollout/action_distance", float(action[1]), global_step)

                if len(transitions) >= args.n_steps:
                    demo_pair = (
                        (np.asarray(demo_states, dtype=np.float32), np.asarray(demo_actions, dtype=np.float32))
                        if demo_states else None
                    )
                    metrics = train_from_buffer(agent, transitions, args, demo=demo_pair, bc_weight=bc_weight)
                    transitions.clear()
                    writer.add_scalar("train/policy_loss", metrics["policy_loss"], global_step)
                    writer.add_scalar("train/value_loss", metrics["value_loss"], global_step)
                    writer.add_scalar("train/entropy", metrics["entropy"], global_step)

        rewards.append(episode_reward)
        lengths.append(episode_length)
        guided.append(bool(guided_run))
        global_steps.append(global_step)
        writer.add_scalar("episode/reward", episode_reward, episode)
        writer.add_scalar("episode/steps", episode_length, episode)
        writer.add_scalar(
            "episode/max_contiguous_outage_s",
            world.max_contiguous_outage_s,
            episode,
        )
        if episode % args.checkpoint_interval == 0:
            save_actor(episode)
        if episode % args.eval_interval == 0 or episode == args.episodes:
            evaluate(episode)
        episode += 1

    if len(transitions) >= max(args.batch_size, 2):
        demo_pair = (
            (np.asarray(demo_states, dtype=np.float32), np.asarray(demo_actions, dtype=np.float32))
            if demo_states else None
        )
        train_from_buffer(agent, transitions, args, demo=demo_pair, bc_weight=bc_weight)
    save_actor(args.episodes)
    np.savez(
        run_dir / "training_history.npz",
        episode_rewards=np.asarray(rewards, dtype=float),
        episode_lengths=np.asarray(lengths, dtype=int),
        episode_guided=np.asarray(guided, dtype=bool),
        global_steps=np.asarray(global_steps, dtype=int),
    )
    np.savez(
        run_dir / "UAV_TD3_improved.npz",
        ep_rewards=np.asarray(rewards, dtype=float),
        ep_guided=np.asarray(guided, dtype=bool),
        episode_lengths=np.asarray(lengths, dtype=int),
        global_steps=np.asarray(global_steps, dtype=int),
    )
    writer.close()
    print(run_dir)


if __name__ == "__main__":
    main()
