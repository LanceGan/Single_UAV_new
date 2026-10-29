import math

import numpy as np
import pytest
import torch

from DDPG import DDPG
from TD3 import TD3
from ReplayBuffer import ReplayBuffer


@pytest.mark.parametrize("agent_type", [TD3, DDPG])
def test_real_agent_updates_are_finite_and_targets_have_no_gradients(agent_type, tmp_path):
    rng = np.random.default_rng(7)
    buffer = ReplayBuffer(8, 2, max_size=64)
    for index in range(64):
        buffer.add(rng.random(8), [rng.uniform(-math.pi, math.pi), rng.uniform(0, 0.25)],
                   -1.0, rng.random(8), index % 7 == 0)
    agent = agent_type(env_with_Dead=True, state_dim=8, action_dim=2,
                       max_action=np.array([math.pi, 0.25]), train_path=str(tmp_path),
                       net_width=32, Q_batchsize=16)
    critic = agent.q_critic if agent_type is TD3 else agent.critic
    target = agent.q_critic_target if agent_type is TD3 else agent.critic_target
    before = [p.detach().clone() for p in critic.parameters()]
    actor_before = [p.detach().clone() for p in agent.actor.parameters()]
    try:
        for _ in range(4):
            agent.train(buffer)
        assert any(not torch.equal(old, now) for old, now in zip(before, critic.parameters()))
        assert any(not torch.equal(old, now) for old, now in zip(actor_before, agent.actor.parameters()))
        assert all(torch.isfinite(p).all() for p in list(critic.parameters()) + list(agent.actor.parameters()))
        assert all(p.grad is None for p in target.parameters())
        assert all(p.grad is None for p in agent.actor_target.parameters())
        assert agent.a_iteration > 0 and agent.q_iteration == 4
    finally:
        agent.writer.close()
