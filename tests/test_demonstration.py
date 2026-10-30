import math

import numpy as np
import pytest
import torch

from DDPG import DDPG
from TD3 import TD3
from ReplayBuffer import ReplayBuffer
from demonstration import behavior_cloning_loss, linear_decay


def test_replay_buffer_keeps_demonstration_flags():
    buffer = ReplayBuffer(3, 2, max_size=4)
    buffer.add(np.zeros(3), [0.0, 0.1], 0.0, np.zeros(3), False)
    buffer.add(np.ones(3), [1.0, 0.2], 0.0, np.ones(3), False, demo=True)
    *_, demo = buffer.sample(64, with_demo=True)
    assert set(demo.cpu().flatten().tolist()) == {0.0, 1.0}
    assert len(buffer.sample(8)) == 5


def test_behavior_cloning_loss_wraps_heading_and_ignores_agent_samples():
    pred = torch.tensor([[math.pi - 1e-4, 0.25], [0.0, 0.0]])
    demo = torch.tensor([[-math.pi + 1e-4, 0.25], [2.0, 0.25]])
    mask = torch.tensor([[1.0], [0.0]])
    assert behavior_cloning_loss(pred, demo, mask, 0.25).item() == pytest.approx(0.0, abs=1e-6)
    assert behavior_cloning_loss(pred, demo, torch.zeros(2, 1), 0.25).item() == 0.0


def test_behavior_cloning_loss_penalizes_heading_and_distance_errors():
    demo = torch.tensor([[0.0, 0.25]])
    mask = torch.ones(1, 1)
    near = behavior_cloning_loss(torch.tensor([[0.1, 0.25]]), demo, mask, 0.25)
    far = behavior_cloning_loss(torch.tensor([[1.5, 0.0]]), demo, mask, 0.25)
    assert 0.0 < near.item() < far.item()


def test_linear_decay_reaches_zero_and_stays_there():
    assert linear_decay(1.0, 0, 400) == 1.0
    assert linear_decay(1.0, 200, 400) == pytest.approx(0.5)
    assert linear_decay(1.0, 400, 400) == 0.0
    assert linear_decay(1.0, 900, 400) == 0.0
    assert linear_decay(1.0, 5, 0) == 0.0


@pytest.mark.parametrize("agent_type", [TD3, DDPG])
def test_behavior_cloning_pulls_actor_toward_demonstrated_action(agent_type, tmp_path):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    buffer = ReplayBuffer(8, 2, max_size=256)
    target = np.array([1.0, 0.2])
    for _ in range(256):
        buffer.add(rng.random(8), target, 0.0, rng.random(8), False, demo=True)
    agent = agent_type(env_with_Dead=True, state_dim=8, action_dim=2,
                       max_action=np.array([math.pi, 0.25]), train_path=str(tmp_path),
                       net_width=32, Q_batchsize=64)
    agent.bc_weight = 1.0
    try:
        probe = rng.random((32, 8))
        def error():
            with torch.no_grad():
                actions = agent.actor(torch.as_tensor(probe, dtype=torch.float32, device=agent.max_action.device))
            return float(np.abs(actions.cpu().numpy() - target).mean())
        before = error()
        for _ in range(300):
            agent.train(buffer)
        assert error() < 0.5 * before
    finally:
        agent.writer.close()
