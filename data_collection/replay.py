import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e6), track_reward=True, track_terminal=True):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.key_order = ['state', 'action', 'next_state']
        self.buffer = {
            'state': np.zeros((max_size, state_dim)),
            'action': np.zeros((max_size, action_dim)),
            'next_state': np.zeros((max_size, state_dim)),
        }
        if track_reward:
            self.key_order.append('reward')
            self.buffer['reward'] = np.zeros((max_size, 1))
        if track_terminal:
            self.key_order.append('terminal')
            self.buffer['terminal'] = np.zeros((max_size, 1))


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def add(self, state, action, next_state, reward=None, terminal=None):
        self.buffer['state'][self.ptr] = state
        self.buffer['action'][self.ptr] = action
        self.buffer['next_state'][self.ptr] = next_state
        if reward is not None:
            self.buffer['reward'][self.ptr] = reward
        if terminal is not None:
            self.buffer['terminal'][self.ptr] = terminal

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_last_n(self, n):
        if self.ptr < n:
            last_seg_idx = (self.ptr - n) % self.max_size
            return tuple(torch.FloatTensor(
                np.concatenate((self.buffer[k][:self.ptr], self.buffer[k][last_seg_idx:]))).to(self.device)
                for k in self.key_order)
        return tuple(torch.FloatTensor(self.buffer[k][self.ptr-n:self.ptr]).to(self.device)
            for k in self.key_order)
    

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return tuple(torch.FloatTensor(self.buffer[k][ind]).to(self.device)
            for k in self.key_order)
