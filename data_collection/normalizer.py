
import torch


class RunningNormalizer:

    def __init__(self, device=None, clamp=None):
        '''
        Clamp should be tuple (min, max)
        '''
        self.clamp = clamp
        self.device = device
        self.reset()

    def update(self, value, batched=False):
        '''
        Set batched=True if value has a batch dimension.
        '''
        if not torch.is_tensor(value):
            value = torch.tensor(value).float().to(self.device)
        value = value.detach()
        if not batched:
            value = value.unsqueeze(0)
        if self.n == 0:
            self.mean = value.mean(0)
        else:
            self.mean = (self.mean * self.n + value.sum(0)) / (self.n + value.shape[0])
            self.var = (self.var * self.n + \
                torch.square(value - self.mean.unsqueeze(0)).sum(0)) / (self.n + value.shape[0])
        self.n += value.shape[0]

    def normalize(self, x, do_center=True, do_scale=True, do_clamp=True, batched=False):
        if not torch.is_tensor(x):
            x = torch.tensor(x).float().to(self.device)
        if not batched:
            x = x.unsqueeze(0)
        if do_center:
            x -= self.mean.unsqueeze(0)
        if do_scale:
            x /= torch.sqrt(self.var).unsqueeze(0)
        if do_clamp and self.clamp is not None:
            x = torch.clamp(x, min=self.clamp[0], max=self.clamp[1])
        if not batched:
            x = x.squeeze(0)
        return x
    
    def __call__(self, x, do_center=True, do_scale=True, do_clamp=True, batched=False):
        return self.normalize(x, do_center, do_scale, do_clamp, batched)
    
    def reset(self):
        self.mean = 0.0
        self.var = 1.0
        self.n = 0
