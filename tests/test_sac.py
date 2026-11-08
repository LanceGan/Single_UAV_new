import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ReplayBuffer import ReplayBuffer
from SAC import SAC, SACActor


def test_sac_actor_actions_respect_bounds():
    actor = SACActor(7, 32, np.asarray([np.pi, 0.25], dtype=np.float32))
    states = torch.zeros((4, 7), dtype=torch.float32)

    action, log_prob, deterministic = actor(states)

    assert action.shape == (4, 2)
    assert log_prob.shape == (4, 1)
    assert deterministic.shape == (4, 2)
    assert torch.all(action[:, 0] >= -np.pi - 1e-6)
    assert torch.all(action[:, 0] <= np.pi + 1e-6)
    assert torch.all(action[:, 1] >= -1e-6)
    assert torch.all(action[:, 1] <= 0.25 + 1e-6)


def test_sac_update_is_finite_and_changes_parameters(tmp_path):
    torch.manual_seed(3)
    replay = ReplayBuffer(7, 2, max_size=64, demo_max_size=64, demo_fraction=0.25)
    rng = np.random.default_rng(3)
    for index in range(24):
        state = rng.normal(size=7).astype(np.float32)
        action = np.asarray(
            [rng.uniform(-np.pi, np.pi), rng.uniform(0.0, 0.25)],
            dtype=np.float32,
        )
        replay.add(
            state,
            action,
            float(rng.normal()),
            rng.normal(size=7).astype(np.float32),
            float(index % 5 == 0),
            demo=index < 8,
        )
    agent = SAC(
        env_with_Dead=True,
        state_dim=7,
        action_dim=2,
        max_action=np.asarray([np.pi, 0.25], dtype=np.float32),
        train_path=str(tmp_path / "logs"),
        net_width=32,
        Q_batchsize=8,
        device="cpu",
    )
    before = agent.actor.mean.weight.detach().clone()
    metrics = agent.train(replay)
    after = agent.actor.mean.weight.detach().clone()

    assert np.isfinite(metrics["critic_loss"])
    assert np.isfinite(metrics["actor_loss"])
    assert not torch.equal(before, after)


def test_sac_deterministic_action_is_reproducible(tmp_path):
    torch.manual_seed(7)
    agent = SAC(
        env_with_Dead=True,
        state_dim=7,
        action_dim=2,
        max_action=np.asarray([np.pi, 0.25], dtype=np.float32),
        train_path=str(tmp_path / "logs"),
        net_width=16,
        device="cpu",
    )
    state = np.zeros(7, dtype=np.float32)
    first = agent.select_action(state, deterministic=True)
    second = agent.select_action(state, deterministic=True)

    np.testing.assert_allclose(first, second)
