import uuid
from copy import deepcopy
import logging
import sys
import random
import argparse

import torch
import metaworld
from tqdm import tqdm
import numpy as np

import wandb

from data_collection.replay import ReplayBuffer
from data_collection.aps import APSAgent
from data_collection import url_utils
from data_collection.base import DataCollectionAlgorithm

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class APSDataCollection(DataCollectionAlgorithm):

    def __init__(
        self,
        ml,
        env,
        out_path: str,
        state_dim: int,
        sf_dim: int,
        action_dim: int,
        hidden_dim: int,
        update_every: int,
        update_task_every: int,
        bs: int = 64,
        lr: float = 1e-4,
        num_cycles: int = 200,
        episodes_per_cycle: int = 10,
        warm_up_cycles: int = 0,
        max_steps_per_episode: int = 500,
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

        self.ml = ml
        self.env = env
        self.aps = APSAgent(
            update_task_every_step=update_task_every,
            sf_dim=sf_dim,
            feature_dim=-1,
            hidden_dim=hidden_dim,
            meta_dim=sf_dim,
            lr=lr,
            discount=0.99,
            knn_rms=True,
            knn_k=12,
            knn_avg=True,
            knn_clip=0.0001,
            obs_type='state',
            obs_shape=(state_dim,),
            action_shape=(action_dim,),
            device=device,
            critic_target_tau=0.01,
            stddev_schedule=0.2,
            stddev_clip=0.3,
        )
        self.meta = self.aps.init_meta()

        self.warm_up_cycles = warm_up_cycles
        self.max_steps_per_episode = max_steps_per_episode
        self.update_every = update_every
        self.bs = bs

        self.live_buffer = ReplayBuffer(state_dim+sf_dim, action_dim, track_reward=False, track_terminal=False)
        
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project='cs330-aps', name=uuid.uuid4().hex)
        self.device = device
        self.step = 0

    def on_cycle_start(self, cycle_idx):
        logging.info(f'Train cycle {cycle_idx}')
        self.cycle_step = 0
    
    def run_episode(self, episode_idx, cycle_idx):
        logging.info(f'Episode {episode_idx}')
        self.epi_step = 0
        self.env.set_task(self.ml.train_tasks[np.random.randint(0, 50)])
        # self.env.set_task(self.ml.train_tasks[0])
        self.env.max_path_length = self.max_steps_per_episode
        self.env._partially_observable = False
        state, _ = self.env.reset()
        
        for i in range(self.max_steps_per_episode):
            self.epi_step += 1
            self.cycle_step += 1
            self.step += 1
            self.meta = self.aps.update_meta(self.meta, self.step, None)

            state_task = np.concatenate((state, self.meta['task']), axis=0).astype(np.float32)
            
            with torch.no_grad(), url_utils.eval_mode(self.aps):
                if cycle_idx < self.warm_up_cycles or random.random() < 0.1:
                    action = env.action_space.sample()
                elif random.random() < 1.0: # by default, always noisy
                    action = self.aps.act(state_task, step=self.step, eval_mode=False)
                else:
                    action = self.aps.act(state_task, step=self.step, eval_mode=True)
            next_state, reward, terminal, truncate, info = env.step(action)
            if info['success'] > 0 and truncate:
                terminal = 1
            else:
                terminal = 0

            next_state_task = np.concatenate((next_state, self.meta['task']), axis=0).astype(np.float32)
            self.live_buffer.add(state_task, action, next_state_task)
            self.data.add(state, action, next_state, reward, terminal)
            
            state_change = np.sqrt(np.sum((next_state - state)**2))
            if self.use_video and cycle_idx % self.video_freq == 0 and i % 5 == 0:
                self.vid.add_frame(env, f'Cycle {cycle_idx} Ep {episode_idx} Step {i}\nExtrinsic reward (unused): {reward:.4f}\nObject to target: {info["obj_to_target"]}')
            if self.use_wandb:
                log_dict = {
                    'step': self.step,
                    'episode': self.episodes_per_cycle * cycle_idx + episode_idx + i / self.max_steps_per_episode,
                    'cycle': cycle_idx + episode_idx / self.episodes_per_cycle + i / self.max_steps_per_episode / self.episodes_per_cycle,
                    'extr_reward': reward,
                    'success': info['success'],
                    'near_object': info['near_object'],
                    'obj_to_target': info['obj_to_target'],
                    'grasp_reward': info['grasp_reward'],
                    'in_place_reward': info['in_place_reward'],
                    'arm_x': state[0].item(),
                    'arm_y': state[1].item(),
                    'arm_z': state[2].item(),
                    'state_change': state_change,
                }
                wandb.log(log_dict)
            if i % 100 == 0:
                with np.printoptions(precision=3, suppress=True):
                    logging.debug(f'Location: {state[:3]}, Action: {action}, Reward: {reward:.3E}, State Change: {state_change:.3E}')
            if truncate:
                break
            state = next_state

            if (i+1) % self.update_every == 0 and cycle_idx >= self.warm_up_cycles:
                # TODO: hindsight part?    
                
                # Concurrent part
                cycle_size = self.max_steps_per_episode * self.episodes_per_cycle
                c_states, c_actions, c_next_states = self.live_buffer.sample_last_n(cycle_size)
                idx = random.sample(range(cycle_size), self.bs)
                c_states, c_actions, c_next_states = c_states[idx], c_actions[idx], c_next_states[idx]

                metrics = self.aps.update((c_states, c_actions, c_next_states), step=self.step)
                if self.use_wandb:
                    log_dict = {
                        'step': self.step,
                        'episode': self.episodes_per_cycle * cycle_idx + episode_idx + i / self.max_steps_per_episode,
                        'cycle': cycle_idx + episode_idx / self.episodes_per_cycle + i / self.max_steps_per_episode / self.episodes_per_cycle,
                        'critic_loss': metrics['critic_loss'],
                        'actor_loss': metrics['actor_loss'],
                        'aps_loss': metrics['aps_loss'],
                        'actor_logprob': metrics['actor_logprob'],
                        'actor_ent': metrics['actor_ent'],
                        'intr_reward': metrics['intr_reward'],
                        'intr_ent_reward': metrics['intr_ent_reward'],
                        'intr_sf_reward': metrics['intr_sf_reward'],
                    }
                    wandb.log(log_dict)
                if (i+1) % (self.update_every * 10) == 0:
                    logging.debug(f"Step {i} Critic loss: {metrics['critic_loss']:.3E}, Actor loss: {metrics['actor_loss']:.3E}")
    
    def on_cycle_end(self, cycle_idx):
        pass
            

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('--goal_sample_size', type=int, default=5000)
    parser.add_argument('--save_path', type=str, default='data/aps.npz')
    parser.add_argument('--env', type=str)
    parser.add_argument('--use_wandb', action='store_true', default=False)

    args = parser.parse_args()

    PATH_LENGTH = 500
    STATE_DIM = 39
    # STATE_DIM = 18
    ACTION_DIM = 4

    ml1 = metaworld.ML1(args.env, seed=0)
    env = ml1.train_classes[args.env]()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    aps_col = APSDataCollection(
        ml=ml1,
        env=env,
        out_path=args.save_path,
        state_dim=STATE_DIM,
        sf_dim=10,
        action_dim=ACTION_DIM,
        hidden_dim=512,
        update_every=16,
        update_task_every=5,
        bs=64,
        lr=3e-5,
        num_cycles=200,
        episodes_per_cycle=10,
        warm_up_cycles=1,
        max_steps_per_episode=PATH_LENGTH,
        device=device,
        use_video=False,
        use_wandb=args.use_wandb,
        
    )
    aps_col.run()

