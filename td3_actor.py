"""TD3 actor network without device initialization side effects."""

import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width, maxaction):
        super().__init__()
        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.l3 = nn.Linear(net_width, net_width)
        self.l4 = nn.Linear(net_width, action_dim)
        self.maxaction = maxaction

    def forward(self, state):
        action = torch.tanh(self.l1(state))
        action = torch.tanh(self.l2(action))
        action = torch.tanh(self.l3(action))
        raw = torch.tanh(self.l4(action))
        angle = raw[:, :1] * self.maxaction[0]
        distance = (raw[:, 1:2] + 1.0) * 0.5 * self.maxaction[1]
        return torch.cat([angle, distance], dim=1)
