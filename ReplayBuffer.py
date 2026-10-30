import numpy as np
import torch


class ReplayBuffer(object):
    """Separate online and demonstration replay with fixed-ratio mixing."""

    def __init__(self, state_dim, action_dim, max_size=int(1e6),
                 demo_max_size=None, demo_fraction=0.25):
        if not 0.0 <= demo_fraction <= 1.0:
            raise ValueError("demo_fraction must be between 0 and 1")
        self.max_size = int(max_size)
        self.demo_max_size = int(demo_max_size or max_size)
        self.demo_fraction = float(demo_fraction)
        self.ptr = 0
        self.size = 0
        self.demo_ptr = 0
        self.demo_size = 0

        self.state = np.zeros((self.max_size, state_dim))
        self.action = np.zeros((self.max_size, action_dim))
        self.reward = np.zeros((self.max_size, 1))
        self.next_state = np.zeros((self.max_size, state_dim))
        self.dead = np.zeros((self.max_size, 1))

        self.demo_state = np.zeros((self.demo_max_size, state_dim))
        self.demo_action = np.zeros((self.demo_max_size, action_dim))
        self.demo_reward = np.zeros((self.demo_max_size, 1))
        self.demo_next_state = np.zeros((self.demo_max_size, state_dim))
        self.demo_dead = np.zeros((self.demo_max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def total_size(self):
        return self.size + self.demo_size

    def add(self, state, action, reward, next_state, dead, demo=False):
        if demo:
            index = self.demo_ptr
            self.demo_state[index] = state
            self.demo_action[index] = action
            self.demo_reward[index] = reward
            self.demo_next_state[index] = next_state
            self.demo_dead[index] = dead
            self.demo_ptr = (self.demo_ptr + 1) % self.demo_max_size
            self.demo_size = min(self.demo_size + 1, self.demo_max_size)
            return

        index = self.ptr
        self.state[index] = state
        self.action[index] = action
        self.reward[index] = reward
        self.next_state[index] = next_state
        self.dead[index] = dead
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    @staticmethod
    def _indices(size, count):
        return np.random.randint(0, size, size=count) if count else np.empty(0, dtype=int)

    def _arrays(self, online_count, demo_count):
        online = self._indices(self.size, online_count)
        demo = self._indices(self.demo_size, demo_count)
        arrays = []
        for online_array, demo_array in (
            (self.state, self.demo_state),
            (self.action, self.demo_action),
            (self.reward, self.demo_reward),
            (self.next_state, self.demo_next_state),
            (self.dead, self.demo_dead),
        ):
            arrays.append(np.concatenate((online_array[online], demo_array[demo]), axis=0))
        mask = np.concatenate((np.zeros((online_count, 1)), np.ones((demo_count, 1))), axis=0)
        permutation = np.random.permutation(online_count + demo_count)
        return [array[permutation] for array in arrays], mask[permutation]

    def sample(self, batch_size, with_demo=False):
        if self.total_size == 0:
            raise ValueError("cannot sample an empty replay buffer")
        batch_size = int(batch_size)
        if with_demo:
            if self.demo_size == 0:
                demo_count = 0
            elif self.size == 0:
                demo_count = batch_size
            else:
                demo_count = int(round(batch_size * self.demo_fraction))
            online_count = batch_size - demo_count
        else:
            demo_count = batch_size if self.size == 0 else 0
            online_count = batch_size - demo_count

        arrays, demo_mask = self._arrays(online_count, demo_count)
        tensors = tuple(torch.FloatTensor(array).to(self.device) for array in arrays)
        if with_demo:
            tensors += (torch.FloatTensor(demo_mask).to(self.device),)
        return tensors
