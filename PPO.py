"""Minimal PPO agent for the bounded continuous UAV control problem."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from td3_actor import Actor


class ValueNetwork(nn.Module):
    def __init__(self, state_dim: int, net_width: int = 256):
        super().__init__()
        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.l3 = nn.Linear(net_width, net_width)
        self.l4 = nn.Linear(net_width, 1)

    def forward(self, state):
        value = F.relu(self.l1(state))
        value = F.relu(self.l2(value))
        value = F.relu(self.l3(value))
        return self.l4(value).squeeze(-1)


class PPOAgent:
    """Clipped-objective PPO with Gaussian exploration around a bounded actor."""

    def __init__(
        self,
        *,
        state_dim,
        action_dim,
        max_action,
        net_width=256,
        actor_lr=3e-4,
        critic_lr=1e-3,
        gamma=0.995,
        gae_lambda=0.95,
        clip_ratio=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        seed=1,
        device=None,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        torch.manual_seed(seed)
        max_action = torch.as_tensor(max_action, dtype=torch.float32, device=self.device)
        self.max_action = max_action
        self.min_action = torch.tensor(
            [-max_action[0].item(), 0.0], dtype=torch.float32, device=self.device
        )
        self.actor = Actor(state_dim, action_dim, net_width, max_action).to(self.device)
        self.value = ValueNetwork(state_dim, net_width).to(self.device)
        self.log_std = nn.Parameter(
            torch.tensor([-1.4, -1.5], dtype=torch.float32, device=self.device)
        )
        # Heading uses radians directly; distance is scaled to avoid huge noise.
        self.std_scale = torch.tensor([1.0, 0.1], dtype=torch.float32, device=self.device)
        self.actor_optimizer = torch.optim.Adam(
            [*self.actor.parameters(), self.log_std], lr=actor_lr
        )
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=critic_lr)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_ratio = float(clip_ratio)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.action_dim = int(action_dim)

    def _tensor_state(self, state):
        return torch.as_tensor(state, dtype=torch.float32, device=self.device).reshape(1, -1)

    def _distribution(self, states):
        mean = self.actor(states)
        std = torch.exp(torch.clamp(self.log_std, -5.0, 1.0)).expand_as(mean) * self.std_scale
        return mean, torch.distributions.Normal(mean, std)

    def deterministic_action(self, state):
        with torch.no_grad():
            return self.actor(self._tensor_state(state))[0].cpu().numpy()

    def value_of(self, state):
        with torch.no_grad():
            return float(self.value(self._tensor_state(state)).item())

    def sample_action(self, state):
        with torch.no_grad():
            states = self._tensor_state(state)
            mean, distribution = self._distribution(states)
            action = distribution.sample()
            action = torch.maximum(self.min_action, torch.minimum(action, self.max_action))
            log_prob = distribution.log_prob(action).sum(dim=-1)
            value = self.value(states)
        return (
            action[0].cpu().numpy(),
            float(log_prob.item()),
            float(value.item()),
        )

    def evaluate_state_action(self, state, action):
        """Return the log probability and value of a supplied action."""
        with torch.no_grad():
            states = self._tensor_state(state)
            actions = torch.as_tensor(
                np.asarray(action, dtype=np.float32),
                device=self.device,
            ).reshape(1, -1)
            log_prob, _ = self.evaluate_actions(states, actions)
            value = self.value(states)
        return float(log_prob.item()), float(value.item())

    def evaluate_actions(self, states, actions):
        _, distribution = self._distribution(states)
        log_prob = distribution.log_prob(actions).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        return log_prob, entropy

    def update(
        self,
        *,
        states,
        actions,
        old_log_probs,
        returns,
        advantages,
        epochs=10,
        batch_size=256,
    ):
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        count = states.shape[0]
        indices = np.arange(count)
        policy_losses, value_losses, entropies = [], [], []

        for _ in range(int(epochs)):
            np.random.shuffle(indices)
            for start in range(0, count, int(batch_size)):
                batch = indices[start:start + int(batch_size)]
                batch_states = states[batch]
                batch_actions = actions[batch]
                batch_old = old_log_probs[batch]
                batch_returns = returns[batch]
                batch_advantages = advantages[batch]

                log_probs, entropy = self.evaluate_actions(batch_states, batch_actions)
                ratio = torch.exp(log_probs - batch_old)
                unclipped = ratio * batch_advantages
                clipped = torch.clamp(
                    ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio
                ) * batch_advantages
                policy_loss = -torch.minimum(unclipped, clipped).mean()

                values = self.value(batch_states)
                value_loss = F.mse_loss(values, batch_returns)
                entropy_mean = entropy.mean()
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy_mean
                )

                self.actor_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [*self.actor.parameters(), self.log_std], self.max_grad_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    self.value.parameters(), self.max_grad_norm
                )
                self.actor_optimizer.step()
                self.value_optimizer.step()

                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy_mean.item()))

        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
        }

    def pretrain_actor(self, states, actions, updates=5000, batch_size=256):
        if len(states) == 0 or int(updates) <= 0:
            return 0.0
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        losses = []
        for _ in range(int(updates)):
            indices = torch.randint(0, states.shape[0], (min(int(batch_size), states.shape[0]),))
            predicted = self.actor(states[indices])
            target = actions[indices]
            heading_loss = (1.0 - torch.cos(predicted[:, :1] - target[:, :1])).mean()
            distance_loss = (
                (predicted[:, 1:2] - target[:, 1:2]) / self.max_action[1]
            ).pow(2).mean()
            loss = heading_loss + distance_loss
            self.actor_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [*self.actor.parameters(), self.log_std], self.max_grad_norm
            )
            self.actor_optimizer.step()
            losses.append(float(loss.item()))
        return float(np.mean(losses))

    def save_actor(self, checkpoint):
        torch.save(self.actor.state_dict(), checkpoint)


def generalized_advantage_estimate(rewards, values, next_values, dones, gamma=0.995, lam=0.95):
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    next_values = np.asarray(next_values, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    advantages = np.zeros_like(rewards)
    last = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        not_done = 1.0 - dones[index]
        delta = rewards[index] + gamma * next_values[index] * not_done - values[index]
        last = delta + gamma * lam * not_done * last
        advantages[index] = last
    returns = advantages + values
    return returns, advantages
