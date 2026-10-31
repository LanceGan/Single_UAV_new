import hashlib
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from D3QN import D3QNAgent
from Train_D3QN import (
    D3QN_STATE_SCHEMA,
    checkpoint_paths,
    create_parser,
    epsilon_at_step,
    evaluation_action,
    run_training,
)
from discrete_actions import DiscreteActionCodec
from evaluate_d3qn import load_d3qn_checkpoint, load_d3qn_run_context


def test_d3qn_defaults_to_mapguide_and_lower_learning_rate():
    args = create_parser().parse_args([])

    assert args.teacher == "mapguide"
    assert args.learning_rate == pytest.approx(1e-4)
    assert args.guide_prob == pytest.approx(0.5)
    assert args.guide_min_prob == pytest.approx(0.2)
    assert args.demo_fraction == pytest.approx(0.25)


def _map_hashes():
    data_dir = Path(__file__).resolve().parents[1] / "results" / "datas"
    return {
        name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
        for name in ("Radio_datas.npz", "Radio_datas_A2G.npz")
    }


def _config(**overrides):
    config = {
        "algorithm": "d3qn",
        "data_level": "low",
        "episodes": 3000,
        "map_hashes": _map_hashes(),
        "net_width": 64,
        "order": [1, 6, 2, 3, 5, 4],
        "outage_max_s": 100.0,
        "payloads_mb": [90.0, 90.0, 60.0, 60.0, 90.0, 60.0],
        "state_dim": 8,
        "state_schema": D3QN_STATE_SCHEMA,
        "uav_h": 1.0,
        "variant": "A",
        "action_count": 33,
        "action_schema": "hover + 16 relative headings at 0.125 + 16 relative headings at 0.25",
        "gamma": 0.995,
        "learning_rate": 0.001,
        "batch_size": 64,
        "gradient_clip": 10.0,
        "target_update_interval": 1000,
        "seed": 1,
        "eval_interval": 25,
    }
    config.update(overrides)
    return config


def test_epsilon_schedule_reaches_floor_without_undershooting():
    assert epsilon_at_step(0, 1.0, 0.05, 100) == pytest.approx(1.0)
    assert epsilon_at_step(50, 1.0, 0.05, 100) == pytest.approx(0.525)
    assert epsilon_at_step(100, 1.0, 0.05, 100) == pytest.approx(0.05)
    assert epsilon_at_step(1000, 1.0, 0.05, 100) == pytest.approx(0.05)


def test_periodic_evaluation_action_is_argmax_only_and_guide_free():
    calls = []

    class Agent:
        def select_action(self, state, epsilon):
            calls.append(epsilon)
            return 2

    act = evaluation_action(Agent(), DiscreteActionCodec())

    np.testing.assert_allclose(act(np.zeros(8)), [np.pi / 8.0, 0.125])
    assert calls == [0.0]


def test_run_context_rejects_continuous_or_incompatible_configuration(tmp_path):
    (tmp_path / "run_config.json").write_text(json.dumps(_config()), encoding="utf-8")
    context = load_d3qn_run_context(tmp_path)
    assert context.algorithm == "d3qn"
    assert context.variant == "A"
    assert context.action_count == 33

    (tmp_path / "run_config.json").write_text(
        json.dumps(_config(algorithm="td3")), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="D3QN"):
        load_d3qn_run_context(tmp_path)


def test_d3qn_checkpoint_round_trip_does_not_use_continuous_actor_loader(tmp_path):
    agent = D3QNAgent(8, 33, hidden_dim=64, seed=2, device="cpu")
    checkpoint = tmp_path / "checkpoint.pt"
    agent.save(checkpoint, metadata={"episode": 7})

    network, metadata = load_d3qn_checkpoint(checkpoint, state_dim=8, action_count=33, net_width=64)

    assert metadata["episode"] == 7
    assert network(np.zeros((1, 8), dtype=np.float32)).shape == (1, 33)
    with pytest.raises(ValueError, match="incompatible"):
        load_d3qn_checkpoint(checkpoint, state_dim=7, action_count=33, net_width=64)


def test_checkpoint_names_and_eval_best_are_d3qn_specific(tmp_path):
    paths = checkpoint_paths(tmp_path)
    assert paths["final"].name == "d3qn_final.pt"
    assert paths["training_best"].name == "d3qn_training_best.pt"
    assert paths["eval_best"].name == "d3qn_eval_best.pt"


class _TinyWorld:
    state_dim = 8

    def __init__(self):
        self.reset()

    def reset(self):
        self.step_number = 0
        return np.zeros(8, dtype=float), 0

    def step_inside(self, action, state, step):
        self.step_number += 1
        return np.full(8, self.step_number, dtype=float), 1.0, self.step_number >= 2, self.step_number, False


def test_tiny_training_writes_reproducible_artifacts_and_uses_fixed_order(tmp_path):
    reports = iter(
        [
            {"success": False, "visited_inspection_points": 5, "uploaded_mb": 300.0,
             "task_time_s": 500.0, "transmission_delay_s": 100.0},
            {"success": True, "visited_inspection_points": 6, "uploaded_mb": 450.0,
             "task_time_s": 200.0, "transmission_delay_s": 80.0},
        ]
    )
    args = Namespace(
        data_level="low", episodes=2, output_root=str(tmp_path), ga_iterations=1,
        seed=1, eval_interval=1, uav_h=1.0, outage_max_s=100.0,
        gamma=0.995, buffer=32, net_width=16, learning_rate=0.001,
        batch_size=2, train_start=2, train_frequency=1, target_update_interval=2,
        gradient_clip=1.0, epsilon_start=1.0, epsilon_end=0.05,
        epsilon_decay_steps=10, checkpoint_interval=1,
    )

    run_dir = run_training(
        args,
        order_solver=lambda payloads, iterations: [1, 6, 2, 3, 5, 4],
        world_factory=lambda **kwargs: _TinyWorld(),
        evaluator=lambda *unused: next(reports),
    )

    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    evaluation = json.loads((run_dir / "eval_history.json").read_text(encoding="utf-8"))
    history = np.load(run_dir / "training_history.npz")
    assert config["variant"] == "A" and config["state_dim"] == 8
    assert config["order"] == [1, 6, 2, 3, 5, 4]
    np.testing.assert_allclose(config["payloads_mb"], [90, 90, 60, 60, 90, 60])
    assert config["guidance"] == "none"
    assert [item["episode"] for item in evaluation["history"]] == [1, 2]
    assert evaluation["best"]["episode"] == 2
    assert history["episode_rewards"].shape == (2,)
    assert history["episode_losses"].shape == (2,)
    for path in checkpoint_paths(run_dir).values():
        assert path.is_file()


def test_demonstration_margin_loss_is_finite_and_nonnegative():
    torch = pytest.importorskip("torch")
    from D3QN import D3QNAgent

    agent = D3QNAgent(8, 33, hidden_dim=32, seed=3, device="cpu")
    states = torch.randn(16, 8)
    actions = torch.randint(0, 33, (16, 1))
    loss = agent.demonstration_loss(states, actions)
    assert torch.isfinite(loss) and loss.item() >= 0.0


def test_demonstration_weight_zero_matches_no_demonstration():
    import torch
    from D3QN import D3QNAgent, DiscreteReplayBuffer

    def buffer():
        b = DiscreteReplayBuffer(8, 64, seed=7)
        rng = np.random.default_rng(0)
        for _ in range(64):
            b.add(rng.random(8), int(rng.integers(33)), -0.1, rng.random(8), False)
        return b

    def build():
        return D3QNAgent(8, 33, hidden_dim=32, seed=7, device="cpu")

    demo = (np.random.rand(64, 8).astype("float32"), np.random.randint(33, size=64))

    with_demo = build(); with_demo.demo_weight = 0.0
    with_demo.learn(buffer(), demo=demo)
    without = build()
    without.learn(buffer())

    for left, right in zip(with_demo.online_network.parameters(), without.online_network.parameters()):
        assert torch.equal(left, right)


def test_mixed_discrete_replay_uses_one_quarter_demonstrations():
    import torch
    from D3QN import DiscreteReplayBuffer, sample_mixed_replay

    online = DiscreteReplayBuffer(8, 64, seed=1)
    demo = DiscreteReplayBuffer(8, 64, seed=2)
    for _ in range(64):
        online.add(np.zeros(8), 0, 0.0, np.zeros(8), False)
        demo.add(np.ones(8), 1, 1.0, np.ones(8), False)

    states, actions, rewards, next_states, dones, demo_states, demo_actions = sample_mixed_replay(
        online, demo, batch_size=32, demo_fraction=0.25, device="cpu")

    assert states.shape == (32, 8)
    assert int((states[:, 0] == 1).sum()) == 8
    assert demo_states.shape == (8, 8)
    assert torch.all(demo_actions == 1)


def test_mapguide_discrete_demonstration_completes_low_mission():
    from d3qn_mapguide import collect_mapguide_demonstration

    transitions, report = collect_mapguide_demonstration(
        level="low", order=[1, 6, 2, 3, 5, 4])

    assert report["success"]
    assert report["visited_inspection_points"] == 6
    assert report["uploaded_mb"] == pytest.approx(450.0)
    assert transitions["states"].shape[1] == 8
    assert len(transitions["actions"]) == report["steps"]


def test_training_with_injected_demo_records_guidance(tmp_path):
    reports = iter([{"success": True, "visited_inspection_points": 6, "uploaded_mb": 450.0,
                     "task_time_s": 200.0, "transmission_delay_s": 80.0}])
    args = Namespace(
        data_level="low", episodes=1, output_root=str(tmp_path), ga_iterations=1,
        seed=1, eval_interval=1, uav_h=1.0, outage_max_s=100.0,
        gamma=0.995, buffer=32, net_width=16, learning_rate=0.001,
        batch_size=2, train_start=2, train_frequency=1, target_update_interval=2,
        gradient_clip=1.0, epsilon_start=1.0, epsilon_end=0.05,
        epsilon_decay_steps=10, checkpoint_interval=1, teacher="astar",
        demo_weight=1.0, demo_margin=0.8,
    )

    def demo_factory(order, codec):
        return (np.zeros((4, 8), dtype="float32"), np.zeros(4, dtype="int64"))

    run_dir = run_training(
        args,
        order_solver=lambda payloads, iterations: [1, 6, 2, 3, 5, 4],
        world_factory=lambda **kwargs: _TinyWorld(),
        evaluator=lambda *unused: next(reports),
        demo_factory=demo_factory,
    )
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    assert config["guidance"] == "astar-demonstration-margin"
    assert config["demo_pairs"] == 4
