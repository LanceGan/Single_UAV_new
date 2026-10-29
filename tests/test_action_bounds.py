import math

import torch

from DDPG import DDPG_Actor
from TD3 import Actor


def test_continuous_actors_emit_nonnegative_distance_actions():
    max_action = torch.tensor([math.pi, 0.25])
    state = torch.randn(32, 8)
    for actor_type in (Actor, DDPG_Actor):
        actions = actor_type(8, 2, 16, max_action)(state)
        assert torch.all(actions[:, 0] >= -math.pi)
        assert torch.all(actions[:, 0] <= math.pi)
        assert torch.all(actions[:, 1] >= 0.0)
        assert torch.all(actions[:, 1] <= 0.25)
