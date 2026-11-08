"""Soft Actor-Critic for the repository's bounded continuous UAV action."""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from demonstration import behavior_cloning_loss


def _default_device():
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        return torch.device("cuda")
    return torch.device("cpu")


class SACActor(nn.Module):
    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(self, state_dim, net_width, max_action):
        super().__init__()
        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.mean = nn.Linear(net_width, 2)
        self.log_std = nn.Linear(net_width, 2)
        self.register_buffer("max_action", torch.as_tensor(max_action, dtype=torch.float32))

    def _bounded(self, raw):
        tanh_raw = torch.tanh(raw)
        heading = tanh_raw[:, :1] * self.max_action[0]
        distance = (tanh_raw[:, 1:2] + 1.0) * 0.5 * self.max_action[1]
        return torch.cat((heading, distance), dim=1)

    def forward(self, state, deterministic=False, with_log_prob=True):
        hidden = F.relu(self.l1(state))
        hidden = F.relu(self.l2(hidden))
        mean = self.mean(hidden)
        if deterministic:
            action = self._bounded(mean)
            log_prob = torch.zeros((state.shape[0], 1), device=state.device)
            return action, log_prob, action
        log_std = torch.clamp(self.log_std(hidden), self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()
        distribution = torch.distributions.Normal(mean, std)
        raw = distribution.rsample()
        tanh_raw = torch.tanh(raw)
        action = self._bounded(raw)
        if with_log_prob:
            correction = torch.log(
                torch.clamp(1.0 - tanh_raw.pow(2), min=1e-6)
            )
            # Account for the different heading and distance action scales.
            scale = torch.stack(
                (
                    self.max_action[0],
                    0.5 * self.max_action[1],
                )
            ).to(state.device)
            log_prob = distribution.log_prob(raw) - correction - torch.log(scale)
            log_prob = log_prob.sum(dim=1, keepdim=True)
        else:
            log_prob = torch.zeros((state.shape[0], 1), device=state.device)
        deterministic_action = self._bounded(mean)
        return action, log_prob, deterministic_action


class SACCritic(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, net_width),
            nn.ReLU(),
            nn.Linear(net_width, net_width),
            nn.ReLU(),
            nn.Linear(net_width, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, net_width),
            nn.ReLU(),
            nn.Linear(net_width, net_width),
            nn.ReLU(),
            nn.Linear(net_width, 1),
        )

    def forward(self, state, action):
        inputs = torch.cat((state, action), dim=1)
        return self.q1(inputs), self.q2(inputs)


class SACDeploymentActor(nn.Module):
    """Evaluation wrapper returning the deterministic mean action as a tensor."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, state):
        action, _, _ = self.policy(state, deterministic=True)
        return action


class SAC:
    def __init__(
        self,
        env_with_Dead,
        state_dim,
        action_dim,
        max_action,
        train_path,
        gamma=0.995,
        net_width=256,
        a_lr=3e-4,
        c_lr=3e-4,
        alpha_lr=3e-4,
        Q_batchsize=256,
        tau=0.005,
        init_alpha=0.2,
        device=None,
    ):
        self.device = torch.device(device) if device is not None else _default_device()
        self.writer = SummaryWriter(train_path)
        max_action = torch.as_tensor(max_action, dtype=torch.float32, device=self.device)
        self.actor = SACActor(state_dim, net_width, max_action).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=a_lr)
        self.critic = SACCritic(state_dim, action_dim, net_width).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=c_lr)
        self.log_alpha = torch.tensor(
            math.log(float(init_alpha)),
            dtype=torch.float32,
            requires_grad=True,
            device=self.device,
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = -float(action_dim)
        self.env_with_Dead = env_with_Dead
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.Q_batchsize = Q_batchsize
        self.bc_weight = 0.0
        self.update_iteration = 0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, noise=0.0, deterministic=False):
        with torch.no_grad():
            state_tensor = torch.as_tensor(
                np.asarray(state, dtype=np.float32).reshape(1, -1),
                device=self.device,
            )
            sampled, _, mean = self.actor(state_tensor, deterministic=False)
            action = mean if deterministic else sampled
            action = action.cpu().numpy().reshape(-1)
        if noise > 0.0:
            action = action + np.random.normal(0.0, float(noise), size=action.shape)
        lower = np.asarray([-float(self.max_action[0]), 0.0])
        upper = self.max_action.cpu().numpy()
        return np.clip(action, lower, upper)

    def train(self, replay_buffer):
        states, actions, rewards, next_states, dead_mask, demo_mask = replay_buffer.sample(
            self.Q_batchsize, with_demo=True
        )
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor(next_states)
            next_q1, next_q2 = self.critic_target(next_states, next_action)
            next_q = torch.minimum(next_q1, next_q2) - self.alpha * next_log_prob
            target_q = rewards
            if self.env_with_Dead:
                target_q = rewards + (1.0 - dead_mask) * self.gamma * next_q
            else:
                target_q = rewards + self.gamma * next_q

        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        new_action, log_prob, deterministic_action = self.actor(states)
        q1_pi, q2_pi = self.critic(states, new_action)
        q_pi = torch.minimum(q1_pi, q2_pi)
        bc_term = behavior_cloning_loss(
            new_action, actions, demo_mask, float(self.max_action[1].item())
        )
        actor_loss = (self.alpha.detach() * log_prob - q_pi).mean()
        actor_loss = actor_loss + self.bc_weight * bc_term
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(
            self.log_alpha * (log_prob + self.target_entropy).detach()
        ).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        with torch.no_grad():
            for parameter, target_parameter in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_parameter.data.mul_(1.0 - self.tau)
                target_parameter.data.add_(self.tau * parameter.data)

        self.writer.add_scalar("critic_loss", critic_loss, self.update_iteration)
        self.writer.add_scalar("actor_loss", actor_loss, self.update_iteration)
        self.writer.add_scalar("bc_loss", bc_term, self.update_iteration)
        self.writer.add_scalar("alpha", self.alpha, self.update_iteration)
        self.update_iteration += 1
        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self.alpha.item()),
        }

    def save_actor(self, path):
        torch.save(self.actor.state_dict(), path)

    def save(self, episode, model_prefix):
        torch.save(self.actor.state_dict(), f"{model_prefix}sac_actor{episode}.pth")
        torch.save(self.critic.state_dict(), f"{model_prefix}sac_critic{episode}.pth")

    def load_actor(self, path):
        self.actor.load_state_dict(torch.load(path, map_location=self.device))
        self.actor.eval()

    def load(self, episode, model_prefix=""):
        self.actor.load_state_dict(
            torch.load(f"{model_prefix}sac_actor{episode}.pth", map_location=self.device)
        )
        self.critic.load_state_dict(
            torch.load(f"{model_prefix}sac_critic{episode}.pth", map_location=self.device)
        )
