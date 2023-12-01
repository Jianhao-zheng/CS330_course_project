
import torch
import torchvision
import cv2

class MetaWorldVideo:
    def __init__(self):
        self.frames = []

    def add_frame(self, env, text=''):
        frame = env.render().copy()
        font = cv2.FONT_ITALIC
        font_scale = 0.4
        font_color = (1, 1, 1) # White color
        for i, s in enumerate(text.split('\n')):
            cv2.putText(frame, s, (8, 15+i*16), font, font_scale, font_color, 1, cv2.LINE_AA)
        self.frames.append(frame)

    def save(self, path, fps=25):
        frames = [torch.tensor(f) for f in self.frames]
        torchvision.io.write_video(path, torch.stack(frames), fps=fps)
