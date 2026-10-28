from __future__ import annotations

import numpy as np
import torch
from torch import nn


class _Actor(nn.Module):
    def __init__(self, state_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, 2), nn.Tanh())

    def forward(self, state):
        return self.network(state)


class DDPGAgent:
    def __init__(self, state_dim: int, action_bounds: tuple[float, float], seed: int) -> None:
        torch.manual_seed(seed)
        self.actor = _Actor(state_dim).cpu().eval()
        self.action_bounds = action_bounds

    def select_action(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            raw = self.actor(torch.as_tensor(state, dtype=torch.float32)).numpy()
        return np.array([raw[0] * np.pi, (raw[1] + 1.0) * self.action_bounds[1] / 2.0])
