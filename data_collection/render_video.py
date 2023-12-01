
import torch
import torchvision

class MetaWorldVideo:
    def __init__(self):
        self.frames = []

    def add_frame(self, env):
        self.frames.append(env.render())

    def save(self, path, fps=25):
        frames = [torch.tensor(f.copy()) for f in self.frames]
        torchvision.io.write_video(path, torch.stack(frames), fps=fps)
