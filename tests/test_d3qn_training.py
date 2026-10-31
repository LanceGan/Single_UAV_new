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
    epsilon_at_step,
    evaluation_action,
    run_training,
)
from discrete_actions import DiscreteActionCodec
from evaluate_d3qn import load_d3qn_checkpoint, load_d3qn_run_context


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
    assert history["episode_guided"].tolist() == [False, False]
    for path in checkpoint_paths(run_dir).values():
        assert path.is_file()


class _GuidedTinyWorld(_TinyWorld):
    dist_max = 0.25

    class _StubUAV:
        x = 0.0
        y = 0.0

    def __init__(self):
        self.UAVs = [self._StubUAV()]
        self.target_loc = np.array([1.0, 1.0])
        self.executed = []
        super().__init__()

    def absolute_to_action(self, absolute):
        return np.asarray(absolute, dtype=float)

    def step_inside(self, action, state, step):
        self.executed.append(np.asarray(action, dtype=float).copy())
        return super().step_inside(action, state, step)


class _StraightGuide:
    """Stands in for MapGuide: always proposes the same absolute action."""

    def __init__(self, world):
        self.world = world

    def action(self, angle_noise=0.0, distance_noise=0.0, rng=None):
        return np.array([0.3, 0.2])


def test_guided_episodes_execute_snapped_discrete_actions_and_are_marked(tmp_path):
    worlds = []

    def factory(**kwargs):
        worlds.append(_GuidedTinyWorld())
        return worlds[-1]

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
        algorithm="d3qn_bc", guide_prob=1.0, guide_min_prob=1.0,
        bc_weight=None, bc_min_weight=None, demo_fraction=None,
        guided_warmup_episodes=1, bc_decay_episodes=1, margin=1.0,
    )

    run_dir = run_training(
        args,
        order_solver=lambda payloads, iterations: [1, 6, 2, 3, 5, 4],
        world_factory=factory,
        evaluator=lambda *unused: next(reports),
        guide_factory=_StraightGuide,
    )

    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    history = np.load(run_dir / "training_history.npz")
    assert run_dir.name == "d3qn_bc"
    assert config["algorithm"] == "d3qn_bc"
    assert config["guidance"].startswith("whole-episode MapGuide")
    assert history["episode_guided"].tolist() == [True, True]
    # [0.3, 0.2] snaps to heading pi/8 at step 0.25, stored and executed as index 18.
    expected = DiscreteActionCodec().decode(18)
    training_world = worlds[0]
    assert len(training_world.executed) == 4  # two episodes of two slots each
    for action in training_world.executed:
        np.testing.assert_allclose(action, expected)
