"""Dueling Double-DQN network, replay buffer, and agent."""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )
        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, action_dim)
        )

    @staticmethod
    def aggregate(value, advantage):
        return value + advantage - advantage.mean(dim=1, keepdim=True)

    def forward(self, state):
        if not torch.is_tensor(state):
            state = torch.as_tensor(state, dtype=torch.float32, device=next(self.parameters()).device)
        features = self.feature(state)
        return self.aggregate(self.value(features), self.advantage(features))


class DiscreteReplayBuffer:
    def __init__(self, state_dim, capacity=200_000, seed=1):
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        self.states = np.empty((self.capacity, state_dim), dtype=np.float32)
        self.actions = np.empty((self.capacity, 1), dtype=np.int64)
        self.rewards = np.empty((self.capacity, 1), dtype=np.float32)
        self.next_states = np.empty((self.capacity, state_dim), dtype=np.float32)
        self.dones = np.empty((self.capacity, 1), dtype=np.float32)
        self.position = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.size

    def add(self, state, action, reward, next_state, done):
        self.states[self.position] = state
        self.actions[self.position, 0] = int(action)
        self.rewards[self.position, 0] = float(reward)
        self.next_states[self.position] = next_state
        self.dones[self.position, 0] = float(bool(done))
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device="cpu"):
        if self.size == 0:
            raise ValueError("cannot sample an empty replay buffer")
        indices = self.rng.integers(0, self.size, size=int(batch_size))
        return (
            torch.as_tensor(self.states[indices], device=device),
            torch.as_tensor(self.actions[indices], device=device),
            torch.as_tensor(self.rewards[indices], device=device),
            torch.as_tensor(self.next_states[indices], device=device),
            torch.as_tensor(self.dones[indices], device=device),
        )


@torch.no_grad()
def double_dqn_targets(rewards, dones, next_states, online_network, target_network, gamma):
    """Online network selects the action; target network evaluates that action."""
    next_actions = online_network(next_states).argmax(dim=1, keepdim=True)
    next_values = target_network(next_states).gather(1, next_actions)
    return rewards + float(gamma) * (1.0 - dones) * next_values


class D3QNAgent:
    CHECKPOINT_SCHEMA = "d3qn-v1"

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        learning_rate=1e-3,
        gamma=0.995,
        batch_size=256,
        target_update_interval=1000,
        gradient_clip=10.0,
        seed=1,
        device=None,
    ):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.target_update_interval = int(target_update_interval)
        self.gradient_clip = float(gradient_clip)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.online_network = DuelingQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network = DuelingQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()
        for parameter in self.target_network.parameters():
            parameter.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=learning_rate)
        self.update_count = 0

    def select_action(self, state, epsilon=0.0):
        if self.rng.random() < float(epsilon):
            return int(self.rng.integers(self.action_dim))
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.online_network(state_tensor).argmax(dim=1).item())

    def learn(self, replay_buffer):
        states, actions, rewards, next_states, dones = replay_buffer.sample(
            self.batch_size, self.device
        )
        targets = double_dqn_targets(
            rewards, dones, next_states, self.online_network, self.target_network, self.gamma
        )
        predictions = self.online_network(states).gather(1, actions)
        loss = F.smooth_l1_loss(predictions, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            self.online_network.parameters(), self.gradient_clip
        )
        self.optimizer.step()
        self.update_count += 1
        if self.update_count % self.target_update_interval == 0:
            self.sync_target()
        return {
            "loss": float(loss.detach().cpu()),
            "gradient_norm": min(float(norm), self.gradient_clip),
        }

    def sync_target(self):
        self.target_network.load_state_dict(self.online_network.state_dict())

    def save(self, path, metadata=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema": self.CHECKPOINT_SCHEMA,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "hidden_dim": self.hidden_dim,
                "online_state_dict": self.online_network.state_dict(),
                "target_state_dict": self.target_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "update_count": self.update_count,
                "metadata": dict(metadata or {}),
            },
            path,
        )

    def load(self, path, load_optimizer=True):
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)
        if (
            checkpoint.get("schema") != self.CHECKPOINT_SCHEMA
            or int(checkpoint.get("state_dim", -1)) != self.state_dim
            or int(checkpoint.get("action_dim", -1)) != self.action_dim
            or int(checkpoint.get("hidden_dim", -1)) != self.hidden_dim
        ):
            raise ValueError(f"checkpoint is incompatible with this D3QN agent: {path}")
        self.online_network.load_state_dict(checkpoint["online_state_dict"], strict=True)
        self.target_network.load_state_dict(checkpoint["target_state_dict"], strict=True)
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.update_count = int(checkpoint.get("update_count", 0))
        return dict(checkpoint.get("metadata", {}))
