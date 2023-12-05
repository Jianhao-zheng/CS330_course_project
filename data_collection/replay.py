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
                 track_goal=False,
                 goal_dim=None,
                 aux_cols=[],
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
        if track_goal:
            self.key_order.append('goal')
            if goal_dim is None:
                self.buffer['goal'] = np.zeros((max_size, state_dim))
            else:
                self.buffer['goal'] = np.zeros((max_size, goal_dim))
        self.aux_cols = aux_cols
        for a in aux_cols:
            self.buffer[a] = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def add(self, state, action, next_state, reward=None, terminal=None, truncate=None, goal=None, **aux):
        self.buffer['state'][self.ptr] = state
        self.buffer['action'][self.ptr] = action
        self.buffer['next_state'][self.ptr] = next_state
        if reward is not None:
            self.buffer['reward'][self.ptr] = reward
        if terminal is not None:
            self.buffer['terminal'][self.ptr] = terminal
        if truncate is not None:
            self.buffer['truncate'][self.ptr] = truncate
        if goal is not None:
            self.buffer['goal'][self.ptr] = goal
        for k in aux:
            self.buffer[k][self.ptr] = aux[k]

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_last_n(self, n, aux=False):
        cols = (self.key_order + self.aux_cols) if aux else self.key_order
        if self.ptr < n:
            last_seg_idx = (self.ptr - n) % self.max_size
            return tuple(torch.FloatTensor(
                np.concatenate((self.buffer[k][:self.ptr], self.buffer[k][last_seg_idx:]))).to(self.device)
                for k in cols)
        return tuple(torch.FloatTensor(self.buffer[k][self.ptr-n:self.ptr]).to(self.device)
            for k in cols)
    

    def sample(self, batch_size, aux=False):
        cols = (self.key_order + self.aux_cols) if aux else self.key_order
        ind = np.random.randint(0, self.size, size=batch_size)
        return tuple(torch.FloatTensor(self.buffer[k][ind]).to(self.device)
            for k in cols)


    def save(self, path):
        buf = { k: self.buffer[k] for k in self.key_order }
        np.savez(path,
                 key_order=self.key_order,
                 max_size=self.max_size,
                 size=self.size,
                 ptr=self.ptr,
                 **buf)

    def __getitem__(self, x):
        return self.buffer[x]

    @staticmethod
    def load(path):
        d = np.load(path)
        dummy = ReplayBuffer(1, 1, 1,
            track_reward=('reward' in d['key_order']),
            track_terminal=('terminal' in d['key_order']),
            track_truncate=('truncate' in d['key_order']),
            track_goal=('goal' in d['key_order']),
        )
        dummy.key_order = list(d['key_order'])
        dummy.aux_cols = []
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
        if 'goal' in d['key_order']:
            dummy.buffer['goal'] = d['goal']
        return dummy
