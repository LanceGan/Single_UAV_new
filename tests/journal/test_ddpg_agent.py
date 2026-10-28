import numpy as np

from journal.agents.ddpg_agent import DDPGAgent


def test_ddpg_action_respects_continuous_bounds():
    agent = DDPGAgent(state_dim=4, action_bounds=(-np.pi, 0.25), seed=1)
    action = agent.select_action(np.zeros(4))
    assert -np.pi <= action[0] <= np.pi
    assert 0.0 <= action[1] <= 0.25
