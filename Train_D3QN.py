"""Train the D3QN baseline: guide-free, A* demo pairs, or MapGuide teacher episodes."""

import argparse
import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from D3QN import D3QNAgent, DiscreteReplayBuffer
from World import World
from demonstration import linear_decay
from discrete_actions import DiscreteActionCodec
from evaluate_policy import rollout, selection_key
from exploration import exploration_scales
from astar_teacher import d3qn_demo_pairs
from map_guidance import MapGuide
from sequence_runner import solve_ga_order
from training_config import load_fixed_payloads


ROOT = Path(__file__).resolve().parent
D3QN_STATE_SCHEMA = (
    "v3: uav xy, target-minus-uav xy, pending payload, outage usage, "
    "A2G SNR, G2A outage probability"
)
ACTION_SCHEMA = "hover + 16 relative headings at 0.125 + 16 relative headings at 0.25"


def create_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_level", choices=("low", "medium", "high"), default="low")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--output_root", default="journal_results/d3qn-baseline")
    parser.add_argument("--ga_iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eval_interval", type=int, default=25)
    parser.add_argument("--uav_h", type=float, default=1.0)
    parser.add_argument("--outage_max_s", type=float, default=100.0)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--buffer", type=int, default=200_000)
    parser.add_argument("--net_width", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--train_start", type=int, default=2_000)
    parser.add_argument("--train_frequency", type=int, default=2)
    parser.add_argument("--target_update_interval", type=int, default=1_000)
    parser.add_argument("--gradient_clip", type=float, default=10.0)
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_end", type=float, default=0.05)
    parser.add_argument("--epsilon_decay_steps", type=int, default=150_000)
    parser.add_argument("--checkpoint_interval", type=int, default=50)
    parser.add_argument("--teacher", choices=("astar", "mapguide", "none"), default="astar",
                        help="demonstration source: A* pairs for the margin loss, whole-episode "
                             "MapGuide replayed into the demo partition, or none")
    parser.add_argument("--demo_weight", type=float, default=1.0, help="weight on the DQfD margin loss")
    parser.add_argument("--demo_margin", type=float, default=0.8, help="Q margin required around demonstrated actions")
    parser.add_argument("--guided_warmup_episodes", type=int, default=400,
                        help="episodes over which the whole-episode MapGuide probability decays to guide_min_prob")
    parser.add_argument("--guide_prob", type=float, default=0.5,
                        help="initial whole-episode MapGuide probability (mapguide teacher only)")
    parser.add_argument("--guide_min_prob", type=float, default=0.2,
                        help="MapGuide probability kept for the rest of training (mapguide teacher only)")
    parser.add_argument("--bc_weight", type=float, default=1.0,
                        help="initial DQfD margin weight on replayed demonstrations (mapguide teacher only)")
    parser.add_argument("--bc_min_weight", type=float, default=0.25,
                        help="persistent imitation weight floor (mapguide teacher only)")
    parser.add_argument("--demo_fraction", type=float, default=0.25,
                        help="demonstration share of each replay batch (mapguide teacher only)")
    parser.add_argument("--bc_decay_episodes", type=int, default=600,
                        help="episodes over which bc_weight decays to bc_min_weight")
    parser.add_argument("--margin", type=float, default=1.0,
                        help="DQfD margin used for replayed demonstrations")
    parser.add_argument("--device", default=None)
    return parser


def epsilon_at_step(step, start, end, decay_steps):
    if decay_steps <= 0:
        return float(end)
    fraction = min(max(float(step) / float(decay_steps), 0.0), 1.0)
    return float(start + fraction * (end - start))


def checkpoint_paths(run_dir):
    model_dir = Path(run_dir) / "models"
    return {
        "final": model_dir / "d3qn_final.pt",
        "training_best": model_dir / "d3qn_training_best.pt",
        "eval_best": model_dir / "d3qn_eval_best.pt",
    }


def evaluation_action(agent, codec):
    """Adapt epsilon-free discrete argmax to World's Variant-A action vector."""
    def act(state):
        return codec.decode(agent.select_action(state, epsilon=0.0))
    return act


def _map_hashes():
    data_dir = ROOT / "results" / "datas"
    return {
        name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
        for name in ("Radio_datas.npz", "Radio_datas_A2G.npz")
    }


def _make_world(*, payloads, order, uav_h, outage_max_s):
    return World(
        length=20,
        width=20,
        uav_num=1,
        dist_max=0.25,
        delta_t=0.5,
        t=1000,
        uav_h=uav_h,
        data_size=300,
        data_sizes=payloads,
        outage_max_s=outage_max_s,
        ini_loc=[2.62, 2.65],
        end_loc=[18.95, 11.12],
        users_name=str(ROOT / "results" / "datas" / "Users_6.txt"),
        BS_loc=np.array(
            [[4.77, 4.56, 0.25], [14.89, 4.89, 0.25],
             [4.63, 13.78, 0.25], [13.48, 14.89, 0.25]]
        ),
        traverse_sequence=list(order),
        variant="A",
    )


def _rollout_context(run_dir, config):
    return SimpleNamespace(
        run_dir=Path(run_dir),
        config=config,
        algorithm="d3qn",
        state_dim=8,
        net_width=int(config["net_width"]),
        variant="A",
        order=tuple(config["order"]),
        payloads_mb=np.asarray(config["payloads_mb"], dtype=float),
        outage_max_s=float(config["outage_max_s"]),
        total_steps=1000,
        length=20.0,
        width=20.0,
        delta_t=0.5,
        dist_max=0.25,
    )


def _evaluate_agent(agent, codec, context, world):
    report, _ = rollout(context, world, evaluation_action(agent, codec))
    return report


def _write_training_plot(run_dir, rewards, losses, epsilons):
    episodes = np.arange(1, len(rewards) + 1)
    figure, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True, constrained_layout=True)
    axes[0].plot(episodes, rewards)
    axes[0].set_ylabel("Return")
    axes[1].plot(episodes, losses)
    axes[1].set_ylabel("D3QN loss")
    axes[2].plot(episodes, epsilons)
    axes[2].set_ylabel("Epsilon")
    axes[2].set_xlabel("Episode")
    figure.savefig(Path(run_dir) / "training_metrics.png", dpi=200)
    figure.savefig(Path(run_dir) / "training_metrics.pdf")
    plt.close(figure)


def run_training(
    args,
    *,
    order_solver=solve_ga_order,
    world_factory=_make_world,
    evaluator=_evaluate_agent,
    demo_factory=None,
    guide_factory=MapGuide,
):
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    if args.eval_interval <= 0 or args.checkpoint_interval <= 0:
        raise ValueError("evaluation and checkpoint intervals must be positive")
    guided_enabled = getattr(args, "teacher", "astar") == "mapguide"
    guide_prob = float(getattr(args, "guide_prob", 0.5))
    guide_min_prob = float(getattr(args, "guide_min_prob", 0.2))
    if not 0.0 <= guide_min_prob <= guide_prob:
        raise ValueError("guide_min_prob must be between zero and guide_prob")
    bc_weight = float(getattr(args, "bc_weight", 1.0))
    bc_min_weight = float(getattr(args, "bc_min_weight", 0.25))
    demo_fraction = float(getattr(args, "demo_fraction", 0.25))
    guided_warmup = int(getattr(args, "guided_warmup_episodes", 400))
    bc_decay_episodes = int(getattr(args, "bc_decay_episodes", 600))
    margin = float(getattr(args, "margin", 1.0))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    payloads = load_fixed_payloads(args.data_level)
    order = list(order_solver(payloads, iterations=args.ga_iterations))
    if sorted(order) != list(range(1, len(payloads) + 1)):
        raise ValueError("IGA order must contain every inspection point exactly once")
    run_dir = Path(args.output_root).resolve() / args.data_level / "d3qn"
    run_dir.mkdir(parents=True, exist_ok=True)

    codec = DiscreteActionCodec()
    demo = demo_factory(order, codec) if demo_factory is not None else None
    if guided_enabled:
        guidance = ("whole-episode MapGuide demonstrations replayed into a persistent demo "
                    "partition; DQfD margin with weight decaying to a persistent floor")
    elif demo is not None:
        guidance = "astar-demonstration-margin"
    else:
        guidance = "none"
    config = {
        **vars(args),
        "algorithm": "d3qn",
        "variant": "A",
        "state_dim": 8,
        "state_schema": D3QN_STATE_SCHEMA,
        "action_count": 33,
        "action_schema": ACTION_SCHEMA,
        "order": order,
        "payloads_mb": payloads.tolist(),
        "map_hashes": _map_hashes(),
        "sequence_method": "GA with exact verification/fallback",
        "guidance": guidance,
        "demo_pairs": int(len(demo[1])) if demo is not None else 0,
        "guide_schedule": (f"{guide_prob} -> {guide_min_prob} over {guided_warmup} episodes, "
                           "then constant" if guided_enabled else "none"),
        "bc_schedule": (f"{bc_weight} -> {bc_min_weight} over {bc_decay_episodes} episodes, "
                        "then constant" if guided_enabled else "none"),
        "demo_sampling_fraction": demo_fraction if guided_enabled else 0.0,
        "dqfd_margin": margin if guided_enabled else 0.0,
        "policy_mode": ("epsilon-greedy with whole-episode MapGuide demonstrations; "
                        "deterministic argmax evaluation" if guided_enabled
                        else "epsilon-greedy training; deterministic argmax evaluation"),
        "model_selection": "eval_best: feasible first, then minimum T + tau",
        "checkpoint_schema": D3QNAgent.CHECKPOINT_SCHEMA,
        "done_mask": "all episode endings disable bootstrap",
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    world_arguments = {
        "payloads": payloads,
        "order": order,
        "uav_h": args.uav_h,
        "outage_max_s": args.outage_max_s,
    }
    training_world = world_factory(**world_arguments)
    evaluation_world = world_factory(**world_arguments)
    if int(training_world.state_dim) != 8:
        raise ValueError("D3QN baseline requires the Variant-A 8D observation")
    # World construction may alter NumPy's global RNG through legacy terrain code.
    np.random.seed(args.seed)
    random.seed(args.seed)

    agent = D3QNAgent(
        state_dim=8,
        action_dim=len(codec),
        hidden_dim=args.net_width,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        batch_size=args.batch_size,
        target_update_interval=args.target_update_interval,
        gradient_clip=args.gradient_clip,
        seed=args.seed,
        device=getattr(args, "device", None),
        margin=margin,
    )
    agent.demo_weight = float(getattr(args, "demo_weight", 1.0))
    agent.demo_margin = float(getattr(args, "demo_margin", 0.8))
    replay = DiscreteReplayBuffer(
        8, args.buffer, args.seed,
        demo_capacity=args.buffer, demo_fraction=demo_fraction,
    )
    context = _rollout_context(run_dir, config)
    paths = checkpoint_paths(run_dir)
    paths["final"].parent.mkdir(parents=True, exist_ok=True)

    rewards = []
    episode_losses = []
    epsilons = []
    episode_guided = []
    eval_history = []
    eval_best = None
    best_reward = -float("inf")
    global_step = 0

    for episode in range(1, args.episodes + 1):
        state, step = training_world.reset()
        # A fresh guide per episode mirrors Train_SINR: whole demonstration
        # episodes avoid the guide silently repairing the agent's mistakes.
        guide = guide_factory(training_world) if guided_enabled else None
        guide_probability = (
            linear_decay(guide_prob, episode - 1, guided_warmup, floor=guide_min_prob)
            if guided_enabled else 0.0
        )
        guided_run = bool(np.random.random() < guide_probability) if guided_enabled else False
        episode_guided.append(guided_run)
        agent.bc_weight = (
            linear_decay(bc_weight, episode - 1, bc_decay_episodes, floor=bc_min_weight)
            if guided_enabled else 0.0
        )
        done = False
        episode_reward = 0.0
        losses = []
        epsilon = epsilon_at_step(
            global_step, args.epsilon_start, args.epsilon_end, args.epsilon_decay_steps
        )
        while not done:
            epsilon = epsilon_at_step(
                global_step, args.epsilon_start, args.epsilon_end, args.epsilon_decay_steps
            )
            if guided_run:
                uav = training_world.UAVs[0]
                target_distance = float(np.linalg.norm(
                    np.array([uav.x, uav.y]) - training_world.target_loc
                ))
                angle_noise, distance_noise = exploration_scales(
                    episode, args.episodes, target_distance
                )
                # MapGuide plans absolute headings; snap them onto the discrete
                # codec so the stored index is exactly what the world executes.
                action_index = codec.encode(training_world.absolute_to_action(
                    guide.action(angle_noise, distance_noise, np.random)
                ))
            else:
                action_index = agent.select_action(state, epsilon)
            next_state, reward, done, step, _ = training_world.step_inside(
                codec.decode(action_index), state, step
            )
            # Every episode ending is terminal for bootstrap, including QoS and timeout.
            replay.add(state, action_index, reward, next_state, done, demo=guided_run)
            state = next_state
            episode_reward += float(reward)
            global_step += 1
            if len(replay) >= max(args.train_start, args.batch_size) and global_step % args.train_frequency == 0:
                losses.append(agent.learn(replay, demo=demo)["loss"])

        rewards.append(episode_reward)
        episode_losses.append(float(np.mean(losses)) if losses else np.nan)
        epsilons.append(epsilon)
        metadata = {"episode": episode, "global_step": global_step, "data_level": args.data_level}
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(paths["training_best"], metadata)
        if episode % args.checkpoint_interval == 0:
            agent.save(paths["final"].parent / f"d3qn_episode_{episode}.pt", metadata)

        if episode % args.eval_interval == 0 or episode == args.episodes:
            report = evaluator(agent, codec, context, evaluation_world)
            item = {"episode": episode, **report}
            eval_history.append(item)
            if eval_best is None or selection_key(report) < selection_key(eval_best):
                eval_best = item
                agent.save(paths["eval_best"], metadata)
            (run_dir / "eval_history.json").write_text(
                json.dumps({"selection": config["model_selection"], "best": eval_best,
                            "history": eval_history}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    agent.save(paths["final"], {"episode": args.episodes, "global_step": global_step,
                                "data_level": args.data_level})
    eval_episodes = np.asarray([item["episode"] for item in eval_history], dtype=int)
    eval_success = np.asarray([item["success"] for item in eval_history], dtype=bool)
    eval_objective = np.asarray(
        [item["task_time_s"] + item["transmission_delay_s"] for item in eval_history], dtype=float
    )
    np.savez(
        run_dir / "training_history.npz",
        episode_rewards=np.asarray(rewards, dtype=float),
        episode_losses=np.asarray(episode_losses, dtype=float),
        epsilons=np.asarray(epsilons, dtype=float),
        episode_guided=np.asarray(episode_guided, dtype=bool),
        eval_episodes=eval_episodes,
        eval_success=eval_success,
        eval_objective_s=eval_objective,
    )
    _write_training_plot(run_dir, rewards, episode_losses, epsilons)
    return run_dir


def main():
    args = create_parser().parse_args()

    def make_demo(order, codec):
        if args.teacher != "astar":
            return None
        return d3qn_demo_pairs(args.data_level, order, codec)

    run_dir = run_training(args, demo_factory=make_demo)
    print(run_dir)


if __name__ == "__main__":
    main()
