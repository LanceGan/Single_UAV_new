import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DDPG_Actor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width, maxaction):
        super(DDPG_Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.l3 = nn.Linear(net_width, net_width)
        self.l4 = nn.Linear(net_width, action_dim)

        # keep same API as your TD3 actor: maxaction is a tensor on device
        self.maxaction = maxaction

    def forward(self, state):
        a = torch.tanh(self.l1(state))
        a = torch.tanh(self.l2(a))
        a = torch.tanh(self.l3(a))
        raw = torch.tanh(self.l4(a))
        angle = raw[:, :1] * self.maxaction[0]
        distance = (raw[:, 1:2] + 1.0) * 0.5 * self.maxaction[1]
        return torch.cat([angle, distance], dim=1)


class DDPG_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super(DDPG_Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.l3 = nn.Linear(net_width, net_width)
        self.l4 = nn.Linear(net_width, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        q = F.relu(self.l1(sa))
        q = F.relu(self.l2(q))
        q = F.relu(self.l3(q))
        q = self.l4(q)
        return q


class DDPG(object):
    def __init__(
        self,
        env_with_Dead,
        state_dim,
        action_dim,
        max_action,
        train_path,
        gamma=0.99,
        net_width=128,
        a_lr=1e-4,
        c_lr=1e-3,
        Q_batchsize=256,
        tau=0.005
    ):
        self.writer = SummaryWriter(train_path)
        max_action = torch.tensor(max_action, dtype=torch.float32).to(device)
        self.actor = DDPG_Actor(state_dim, action_dim, net_width, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=a_lr)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic = DDPG_Critic(state_dim, action_dim, net_width).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=c_lr)
        self.critic_target = copy.deepcopy(self.critic)

        self.env_with_Dead = env_with_Dead
        self.action_dim = action_dim
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.Q_batchsize = Q_batchsize

        # bookkeeping for tensorboard steps
        self.q_iteration = 0
        self.a_iteration = 0

    def select_action(self, state, noise=0.0):
        """
        state: numpy array (state_dim,)
        noise: gaussian std for exploration in action space (scalar)
        """
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            a = self.actor(state)
            a = a.cpu().numpy().flatten()
            if noise > 0:
                a = a + np.random.normal(0, noise, size=a.shape)
            # clip to action bounds
            lower = np.array([-self.max_action.cpu().numpy()[0], 0.0])
            a = np.clip(a, lower, self.max_action.cpu().numpy())
        return a

    def train(self, replay_buffer):
        # Sample
        s, a, r, s_prime, dead_mask = replay_buffer.sample(self.Q_batchsize)
        # Compute target Q
        with torch.no_grad():
            next_a = self.actor_target(s_prime)
            target_Q = self.critic_target(s_prime, next_a)
            if self.env_with_Dead:
                target_Q = r + (1 - dead_mask) * self.gamma * target_Q
            else:
                target_Q = r + self.gamma * target_Q

        # Current Q
        current_Q = self.critic(s, a)

        # Critic loss and update
        q_loss = F.mse_loss(current_Q, target_Q)
        self.writer.add_scalar('q_loss', q_loss, self.q_iteration)
        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()
        self.q_iteration += 1

        # Actor loss and update (maximize Q -> minimize -Q)
        a_pred = self.actor(s)
        a_loss = -self.critic(s, a_pred).mean()
        self.writer.add_scalar('a_loss', a_loss, self.a_iteration)
        self.actor_optimizer.zero_grad()
        a_loss.backward()
        self.actor_optimizer.step()
        self.a_iteration += 1

        # Soft update targets
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, episode, model_path):
        torch.save(self.actor.state_dict(), model_path + "ddpg_actor{}.pth".format(episode))
        torch.save(self.critic.state_dict(), model_path + "ddpg_critic{}.pth".format(episode))

    def load(self, episode, model_path=None):
        # If model_path is given, use it; otherwise expect files in cwd
        if model_path is None:
            actor_path = "ddpg_actor{}.pth".format(episode)
            critic_path = "ddpg_critic{}.pth".format(episode)
        else:
            actor_path = model_path + "ddpg_actor{}.pth".format(episode)
            critic_path = model_path + "ddpg_critic{}.pth".format(episode)

        self.actor.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path))
