import uuid
from copy import deepcopy

import torch
import metaworld
from tqdm import tqdm
import numpy as np

import wandb

from data_collection.replay import ReplayBuffer
from data_collection.rnd import RNDModel
from data_collection.td3 import TD3
from data_collection.base import DataCollectionAlgorithm

class GCRLRNDDataCollection(DataCollectionAlgorithm):

    def __init__(
        self,
        env,
        out_path: str,
        rl_alg,
        rnd,
        state_dim: int,
        action_dim: int,
        num_cycles: int,
        episodes_per_cycle: int,
        warm_up_cycles: int = 2,
        max_steps_per_episode: int = 500,
        rnd_lr: float = 0.001,
        buffer_size: int = int(1e6),
        device: torch.device = torch.device('cuda'),
        use_video: bool = False,
        video_freq: int = 1,
        video_out_dir: str = 'video_tmp',
        use_wandb: bool = False,
     ) -> None:
        # initializes data, num_cycles, episodes_per_cycle, + video stuff
        super().__init__(env=env, out_path=out_path, state_dim=state_dim, action_dim=action_dim, num_cycles=num_cycles,
                         episodes_per_cycle=episodes_per_cycle, buffer_size=buffer_size,
                         use_video=use_video, video_freq=video_freq, video_out_dir=video_out_dir)

        self.env = env
        self.rl_alg = rl_alg

        self.warm_up_cycles = warm_up_cycles
        self.max_steps_per_episode = max_steps_per_episode

        self.rnd = rnd
        self.rnd_optimizer = torch.optim.AdamW(rnd.predictor.parameters(), lr=rnd_lr)
        self.live_buffer = ReplayBuffer(state_dim, action_dim, track_reward=False, track_terminal=False)
        
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project='cs330-gcrl-rnd', name=uuid.uuid4().hex)
        self.device = device
        self.step = 0

    def on_cycle_start(self, cycle_idx):
        print(f'Train cycle {cycle_idx}')
        self.cycle_step = 0
    
    def run_episode(self, episode_idx, cycle_idx):
        state, _ = self.env.reset()

        if cycle_idx > self.warm_up_cycles:
            goal_candidates = self.live_buffer.sample(
                self.episodes_per_cycle*self.max_steps_per_episode)[2]
            pred_out, target_out = self.rnd(goal_candidates) # batch this if needed
            goal_idx = torch.argmax(torch.sum((pred_out - target_out)**2), dim=-1)
            goal = goal_candidates[goal_idx]
            if self.use_wandb:
                wandb.log({
                    'step': self.step,
                    'episode': episode_idx + self.episodes_per_cycle * cycle_idx,
                    'cycle': cycle_idx + episode_idx / self.episodes_per_cycle,
                    'goal_rnd': torch.max(torch.sum((pred_out - target_out)**2), dim=-1).item(),
                    'goal': goal.numpy().tolist(),
                })
        else:
            goal = torch.tensor(self.env.observation_space.sample(), dtype=torch.float32, device=self.device)
        
        for i in range(self.max_steps_per_episode):
            self.cycle_step += 1
            self.step += 1
            action = self.rl_alg.select_action(
                torch.cat((torch.tensor(state, device=device, dtype=torch.float32), goal), dim=0))
            next_state, reward, terminal, truncate, info = env.step(action)
            self.live_buffer.add(state, action, next_state)
            self.data.add(state, action, next_state, reward, terminal)
            if self.use_video and cycle_idx % self.video_freq == 0:
                self.vid.add_frame(env, f'Cycle {cycle_idx} Ep {episode_idx}\nExtrinsic reward (unused): {reward:.4f}')
            if self.use_wandb:
                wandb.log({
                    'step': self.step,
                    'episode': self.episodes_per_cycle * cycle_idx + episode_idx + i / self.max_steps_per_episode,
                    'cycle': cycle_idx + episode_idx / self.episodes_per_cycle + i / self.max_steps_per_episode / self.episodes_per_cycle,
                    'reward': reward,
                    'success': info['success'],
                    'near_object': info['near_object'],
                    'obj_to_target': info['obj_to_target'],
                    'grasp_reward': info['grasp_reward'],
                    'in_place_reward': info['in_place_reward']
                })
            if terminal or truncate:
                break
            state = next_state

        for i in range(self.max_steps_per_episode):
            states, actions, next_states = self.live_buffer.sample(128) # batch size?
            goal_samples = states[torch.randint(high=len(states), size=(len(states),)), :]
            rewards = -(~(torch.all(torch.isclose(states, goal_samples), dim=-1))).float() # isclose or strict equality?
            terminals = torch.logical_or(
                torch.all(torch.isclose(states, goal_samples), dim=-1),
                torch.all(torch.isclose(next_states, goal_samples), dim=-1)).float()
            train_states = torch.cat((states, goal_samples), dim=-1)
            train_next_states = torch.cat((states, goal_samples), dim=-1)
            critic_loss, actor_loss = self.rl_alg.train_step((train_states, actions, train_next_states, rewards, terminals))
            if self.use_wandb:
                log_dict = {
                    'step': self.step,
                    'episode': self.episodes_per_cycle * cycle_idx + episode_idx + i / self.max_steps_per_episode,
                    'critic_loss': critic_loss,
                }
                if actor_loss is not None:
                    log_dict['actor_loss'] = actor_loss
                wandb.log(log_dict)
    
    def on_cycle_end(self, cycle_idx):
        cycle_state, _, cycle_next_state = self.live_buffer.sample_last_n(self.cycle_step)
        for k in range(4):
            idx_order = torch.randperm(self.cycle_step).to(device)
            for i in range(0, self.cycle_step, 128): # batch size?
                batch_state = cycle_state[idx_order[i:i+128]]
                pred_out, target_out = self.rnd(batch_state)
                loss = torch.mean((pred_out - target_out)**2)

                loss.backward()
                self.rnd_optimizer.step()
                self.rnd_optimizer.zero_grad()

                if self.use_wandb:
                    wandb.log({
                        'cycle': cycle_idx + k/4 + i/self.cycle_step/4,
                        'rnd_loss': loss,
                    })

if __name__ == '__main__':
    PATH_LENGTH = 2000
    STATE_DIM = 39
    ACTION_DIM = 4

    ml1 = metaworld.ML1("pick-place-v2", seed=0)
    env = ml1.train_classes["pick-place-v2"]()
    env.set_task(ml1.train_tasks[0])
    env.max_path_length = PATH_LENGTH
    env._partially_observable = False,
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    gcrl_rnd_col = GCRLRNDDataCollection(
        env=env,
        out_path='data/gcrl.npz',
        rl_alg=TD3(
            state_dim=STATE_DIM*2,
            action_dim=ACTION_DIM,
            max_action=1,
            device=device,
        ),
        rnd=RNDModel(
            input_size=STATE_DIM,
            output_size=512,
        ).to(device),
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        num_cycles=2, # dummy values for now
        episodes_per_cycle=2,
        warm_up_cycles=1,
        max_steps_per_episode=PATH_LENGTH,
        device=device,
        use_video=True,
        use_wandb=True,
    )
    gcrl_rnd_col.run()

