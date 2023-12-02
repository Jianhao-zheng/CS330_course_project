from abc import ABC, abstractmethod, abstractproperty

from data_collection.replay import ReplayBuffer
from data_collection.render_video import MetaWorldVideo

class DataCollectionAlgorithm(ABC):

    def __init__(
        self,
        env,
        out_path: str,
        state_dim: int,
        action_dim: int,
        num_cycles: int,
        episodes_per_cycle: int,
        buffer_size: int = int(1e6),
        use_video: bool = False,
        video_freq: int = 1,
        video_out_dir: str = 'video_tmp',
    ) -> None:
        self.env = env
        self.out_path = out_path
        self.data = ReplayBuffer(state_dim, action_dim, max_size=buffer_size)
        self.num_cycles = num_cycles
        self.episodes_per_cycle = episodes_per_cycle
        self.use_video = use_video
        self.video_freq = video_freq
        self.video_out_dir = video_out_dir

    def on_cycle_start(self, cycle_idx):
        pass
    
    @abstractmethod
    def run_episode(self, episode_idx, cycle_idx):
        raise NotImplementedError

    def on_cycle_end(self, cycle_idx):
        pass
    
    def run(self) -> ReplayBuffer:
        if self.use_video:
            self.vid = MetaWorldVideo()
        for c in range(self.num_cycles):
            self.on_cycle_start(c)
            for e in range(self.episodes_per_cycle):
                self.run_episode(e, c)
            self.on_cycle_end(c)
            if self.use_video and c % self.video_freq == 0:
                self.vid.save(f'{self.video_out_dir}/vid_{c}.mp4')
                self.vid = MetaWorldVideo()
        self.data.save(self.out_path)
        return self.data
                

