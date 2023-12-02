import numpy as np
import torch

class ReplayBuffer:
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_size=int(1e6),
                 track_reward=True,
                 track_terminal=True,
                 track_truncate=False,
        ):
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
        if track_truncate:
            self.key_order.append('truncate')
            self.buffer['truncate'] = np.zeros((max_size, 1))


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def add(self, state, action, next_state, reward=None, terminal=None, truncate=None):
        self.buffer['state'][self.ptr] = state
        self.buffer['action'][self.ptr] = action
        self.buffer['next_state'][self.ptr] = next_state
        if reward is not None:
            self.buffer['reward'][self.ptr] = reward
        if terminal is not None:
            self.buffer['terminal'][self.ptr] = terminal
        if truncate is not None:
            self.buffer['truncate'][self.ptr] = truncate

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


    def save(self, path):
        np.savez(path,
                 key_order=self.key_order,
                 max_size=self.max_size,
                 size=self.size,
                 ptr=self.ptr,
                 **self.buffer)

    def __getitem__(self, x):
        return self.buffer[x]

    @staticmethod
    def load(path):
        d = np.load(path)
        dummy = ReplayBuffer(1, 1, 1,
            track_reward=('reward' in d['key_order']),
            track_terminal=('terminal' in d['key_order']),
            track_truncate=('truncate' in d['key_order']),
        )
        dummy.key_order = list(d['key_order'])
        dummy.max_size = int(d['max_size'])
        dummy.size = int(d['size'])
        dummy.ptr = int(d['ptr'])
        dummy.buffer['state'] = d['state']
        dummy.buffer['action'] = d['action']
        dummy.buffer['next_state'] = d['next_state']
        if 'reward' in d['key_order']:
            dummy.buffer['reward'] = d['reward']
        if 'terminal' in d['key_order']:
            dummy.buffer['terminal'] = d['terminal']
        if 'truncate' in d['key_order']:
            dummy.buffer['truncate'] = d['truncate']
        return dummy
