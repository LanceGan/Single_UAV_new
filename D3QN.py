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


class NStepAssembler:
    """Turn a per-decision transition stream into (1-step, n-step) records.

    Emits, for the oldest buffered decision, both its 1-step fields and the
    n-step aggregate (discounted reward sum, state after n decisions, done).
    Truncated horizons only happen at episode end, where done masks the
    bootstrap anyway, so a fixed per-decision gamma is exact.
    """

    def __init__(self, n, gamma):
        self.n = int(n)
        if self.n < 1:
            raise ValueError("n must be at least 1")
        self.gamma = float(gamma)
        self.pending = []

    def _record(self, horizon):
        state, action, reward, next_state, done = self.pending[0]
        nstep_reward = 0.0
        for index in range(horizon):
            nstep_reward += (self.gamma ** index) * self.pending[index][2]
        _, _, _, nstep_next, nstep_done = self.pending[horizon - 1]
        return (state, action, reward, next_state, done,
                nstep_reward, nstep_next, nstep_done, self.gamma ** horizon)

    def push(self, state, action, reward, next_state, done):
        self.pending.append((state, action, reward, next_state, done))
        records = []
        if done:
            while self.pending:
                records.append(self._record(len(self.pending)))
                self.pending.pop(0)
        elif len(self.pending) == self.n:
            records.append(self._record(self.n))
            self.pending.pop(0)
        return records


class DiscreteReplayBuffer:
    """Online transitions plus a demonstration partition that is never evicted by them."""

    def __init__(self, state_dim, capacity=200_000, seed=1, demo_capacity=None, demo_fraction=0.25):
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        self.demo_fraction = float(demo_fraction)
        if not 0.0 <= self.demo_fraction <= 1.0:
            raise ValueError("demo_fraction must be between 0 and 1")
        self.demo_capacity = int(demo_capacity if demo_capacity is not None else capacity)
        if self.demo_capacity <= 0:
            raise ValueError("demo_capacity must be positive")
        self.states = np.empty((self.capacity, state_dim), dtype=np.float32)
        self.actions = np.empty((self.capacity, 1), dtype=np.int64)
        self.rewards = np.empty((self.capacity, 1), dtype=np.float32)
        self.next_states = np.empty((self.capacity, state_dim), dtype=np.float32)
        self.dones = np.empty((self.capacity, 1), dtype=np.float32)
        self.nstep_rewards = np.empty((self.capacity, 1), dtype=np.float32)
        self.nstep_next_states = np.empty((self.capacity, state_dim), dtype=np.float32)
        self.nstep_dones = np.empty((self.capacity, 1), dtype=np.float32)
        self.nstep_gammas = np.empty((self.capacity, 1), dtype=np.float32)
        self.demo_states = np.empty((self.demo_capacity, state_dim), dtype=np.float32)
        self.demo_actions = np.empty((self.demo_capacity, 1), dtype=np.int64)
        self.demo_rewards = np.empty((self.demo_capacity, 1), dtype=np.float32)
        self.demo_next_states = np.empty((self.demo_capacity, state_dim), dtype=np.float32)
        self.demo_dones = np.empty((self.demo_capacity, 1), dtype=np.float32)
        self.demo_nstep_rewards = np.empty((self.demo_capacity, 1), dtype=np.float32)
        self.demo_nstep_next_states = np.empty((self.demo_capacity, state_dim), dtype=np.float32)
        self.demo_nstep_dones = np.empty((self.demo_capacity, 1), dtype=np.float32)
        self.demo_nstep_gammas = np.empty((self.demo_capacity, 1), dtype=np.float32)
        self.position = 0
        self.size = 0
        self.demo_position = 0
        self.demo_size = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.size + self.demo_size

    DEFAULT_NSTEP_GAMMA = 0.995

    def add(self, state, action, reward, next_state, done, demo=False, nstep=None):
        if nstep is None:  # degenerate n=1 record keeps single-step callers valid
            nstep = (reward, next_state, done, self.DEFAULT_NSTEP_GAMMA)
        nstep_reward, nstep_next, nstep_done, nstep_gamma = nstep
        if demo:
            index = self.demo_position
            self.demo_states[index] = state
            self.demo_actions[index, 0] = int(action)
            self.demo_rewards[index, 0] = float(reward)
            self.demo_next_states[index] = next_state
            self.demo_dones[index, 0] = float(bool(done))
            self.demo_nstep_rewards[index, 0] = float(nstep_reward)
            self.demo_nstep_next_states[index] = nstep_next
            self.demo_nstep_dones[index, 0] = float(bool(nstep_done))
            self.demo_nstep_gammas[index, 0] = float(nstep_gamma)
            self.demo_position = (self.demo_position + 1) % self.demo_capacity
            self.demo_size = min(self.demo_size + 1, self.demo_capacity)
            return
        self.states[self.position] = state
        self.actions[self.position, 0] = int(action)
        self.rewards[self.position, 0] = float(reward)
        self.next_states[self.position] = next_state
        self.dones[self.position, 0] = float(bool(done))
        self.nstep_rewards[self.position, 0] = float(nstep_reward)
        self.nstep_next_states[self.position] = nstep_next
        self.nstep_dones[self.position, 0] = float(bool(nstep_done))
        self.nstep_gammas[self.position, 0] = float(nstep_gamma)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device="cpu"):
        if len(self) == 0:
            raise ValueError("cannot sample an empty replay buffer")
        batch_size = int(batch_size)
        if self.demo_size == 0:
            demo_count = 0
        elif self.size == 0:
            demo_count = batch_size
        else:
            demo_count = int(round(batch_size * self.demo_fraction))
        online_count = batch_size - demo_count
        online = (self.rng.integers(0, self.size, size=online_count)
                  if online_count else np.empty(0, dtype=int))
        demo = (self.rng.integers(0, self.demo_size, size=demo_count)
                if demo_count else np.empty(0, dtype=int))
        demo_mask = np.concatenate(
            (np.zeros((online_count, 1), dtype=np.float32),
             np.ones((demo_count, 1), dtype=np.float32))
        )
        arrays = [
            np.concatenate((online_array[online], demo_array[demo]), axis=0)
            for online_array, demo_array in (
                (self.states, self.demo_states),
                (self.actions, self.demo_actions),
                (self.rewards, self.demo_rewards),
                (self.next_states, self.demo_next_states),
                (self.dones, self.demo_dones),
                (self.nstep_rewards, self.demo_nstep_rewards),
                (self.nstep_next_states, self.demo_nstep_next_states),
                (self.nstep_dones, self.demo_nstep_dones),
                (self.nstep_gammas, self.demo_nstep_gammas),
            )
        ] + [demo_mask]
        if demo_count:
            permutation = self.rng.permutation(batch_size)
            arrays = [array[permutation] for array in arrays]
        return tuple(torch.as_tensor(array, device=device) for array in arrays)


@torch.no_grad()
def double_dqn_targets(rewards, dones, next_states, online_network, target_network, gamma):
    """Online network selects the action; target network evaluates that action.

    ``gamma`` may be a scalar (1-step) or a per-row tensor (n-step horizons).
    """
    next_actions = online_network(next_states).argmax(dim=1, keepdim=True)
    next_values = target_network(next_states).gather(1, next_actions)
    gamma = gamma if torch.is_tensor(gamma) else float(gamma)
    return rewards + gamma * (1.0 - dones) * next_values


def dqfd_margin_loss(q_values, demo_actions, demo_mask, margin):
    """DQfD large-margin term over demonstration-flagged batch rows only.

    Zero exactly when every demonstrated action's Q exceeds all alternatives
    by at least ``margin``; otherwise it pays the shortfall.
    """
    count = demo_mask.sum()
    if count.item() == 0:
        return q_values.sum() * 0.0
    penalty = torch.full_like(q_values, float(margin))
    penalty.scatter_(1, demo_actions, 0.0)
    augmented_max = (q_values + penalty).max(dim=1, keepdim=True).values
    shortfall = augmented_max - q_values.gather(1, demo_actions)
    return (shortfall * demo_mask).sum() / count


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
        margin=1.0,
    ):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.target_update_interval = int(target_update_interval)
        self.gradient_clip = float(gradient_clip)
        self.margin = float(margin)
        self.bc_weight = 0.0  # weight on replayed-demo margin; scheduled per episode
        self.n_step_weight = 1.0  # DQfD keeps the auxiliary n-step TD term at unit weight
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
        self.demo_margin = 0.8
        self.demo_weight = 0.0  # set by the training loop; 0 disables demonstration imitation

    def select_action(self, state, epsilon=0.0):
        if self.rng.random() < float(epsilon):
            return int(self.rng.integers(self.action_dim))
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.online_network(state_tensor).argmax(dim=1).item())

    def demonstration_loss(self, states, actions, margin=None):
        """Large-margin DQfD loss on demonstrated (state, action) pairs.

        Penalizes any non-teacher action whose Q value comes within ``margin`` of
        the teacher action, so argmax is pulled toward the demonstration.
        """
        margin = self.demo_margin if margin is None else float(margin)
        q = self.online_network(states)
        q_teacher = q.gather(1, actions)
        penalties = torch.full_like(q, margin)
        penalties.scatter_(1, actions, 0.0)
        return (penalties + q - q_teacher).clamp(min=0.0).mean()

    def learn(self, replay_buffer, demo=None, demo_weight=None):
        (states, actions, rewards, next_states, dones,
         nstep_rewards, nstep_next_states, nstep_dones, nstep_gammas,
         demo_mask) = replay_buffer.sample(self.batch_size, self.device)
        targets = double_dqn_targets(
            rewards, dones, next_states, self.online_network, self.target_network, self.gamma
        )
        q_values = self.online_network(states)
        predictions = q_values.gather(1, actions)
        td_loss = F.smooth_l1_loss(predictions, targets)
        # DQfD-style auxiliary n-step return (per-row horizon gamma).
        nstep_targets = double_dqn_targets(
            nstep_rewards, nstep_dones, nstep_next_states,
            self.online_network, self.target_network, nstep_gammas,
        )
        nstep_loss = F.smooth_l1_loss(predictions, nstep_targets)
        # Replayed teacher episodes carry a mask; bc_weight is scheduled per episode.
        margin_loss = dqfd_margin_loss(q_values, actions, demo_mask, self.margin)
        loss = td_loss + float(self.n_step_weight) * nstep_loss + float(self.bc_weight) * margin_loss
        # Fixed demonstration (state, action) pairs feed only this auxiliary term.
        if demo is not None:
            demo_states, demo_actions = demo
            weight = self.demo_weight if demo_weight is None else float(demo_weight)
            if weight > 0 and len(demo_actions) > 0:
                indices = self.rng.integers(0, len(demo_actions), size=self.batch_size)
                demo_states_t = torch.as_tensor(demo_states[indices], device=self.device)
                demo_actions_t = torch.as_tensor(demo_actions[indices], device=self.device).long().unsqueeze(1)
                loss = loss + weight * self.demonstration_loss(demo_states_t, demo_actions_t)
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
            "td_loss": float(td_loss.detach().cpu()),
            "nstep_loss": float(nstep_loss.detach().cpu()),
            "margin_loss": float(margin_loss.detach().cpu()),
            "bc_weight": float(self.bc_weight),
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
