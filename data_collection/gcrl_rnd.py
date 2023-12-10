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
from data_collection.rnd import RNDModel
from data_collection.td3 import TD3
from data_collection.sac import SAC
from data_collection.base import DataCollectionAlgorithm

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class GCRLRNDDataCollection(DataCollectionAlgorithm):

    def __init__(
        self,
        ml,
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
        goal_sample_size: int = 5000,
        rnd_lr: float = 0.0001,
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
        self.rl_alg = rl_alg

        self.warm_up_cycles = warm_up_cycles
        self.max_steps_per_episode = max_steps_per_episode
        self.goal_sample_size = goal_sample_size

        self.rnd = rnd
        self.rnd_optimizer = torch.optim.AdamW(rnd.predictor.parameters(), lr=rnd_lr)
        self.live_buffer = ReplayBuffer(state_dim, action_dim, track_reward=False, track_terminal=False, track_goal=True)
        # self.live_buffer = ReplayBuffer(18, action_dim, track_reward=False, track_terminal=False, track_goal=True)
        
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project='cs330-gcrl-rnd', name=uuid.uuid4().hex)
        self.device = device
        self.step = 0

    def on_cycle_start(self, cycle_idx):
        logging.info(f'Train cycle {cycle_idx}')
        self.cycle_step = 0
    
    def run_episode(self, episode_idx, cycle_idx):
        logging.info(f'Episode {episode_idx}')
        self.epi_step = 0
        self.env.set_task(self.ml.train_tasks[np.random.randint(0, 50)])
        self.env.max_path_length = self.max_steps_per_episode
        self.env._partially_observable = False
        state, _ = self.env.reset()
        state_short = state[:39].copy() #
        state_short[-3:] = 0

        if cycle_idx >= self.warm_up_cycles:
            goal_candidates = self.live_buffer.sample(
                self.goal_sample_size)[0]
            pred_out, target_out = self.rnd(goal_candidates) # batch this if needed
            goal_idx = torch.argmax(torch.sum((pred_out - target_out)**2, dim=-1))
            goal_rnd = torch.sum((pred_out - target_out)**2, dim=-1)[goal_idx].item()
            goal = goal_candidates[goal_idx]
            with np.printoptions(precision=3, suppress=True):
                logging.debug(f'Goal rnd: {goal_rnd}, Goal location: \n{goal.cpu().numpy()}')
            if self.use_wandb:
                # goal_img = wandb.Image(goal.cpu().numpy()[None, :]) # horizontal grayscale
                log_dict = {
                    'step': self.step,
                    'episode': episode_idx + self.episodes_per_cycle * cycle_idx,
                    'cycle': cycle_idx + episode_idx / self.episodes_per_cycle,
                    # 'goal': goal_img,
                }
                if cycle_idx > self.warm_up_cycles: # since need 1 extra epoch to start training rnd
                    log_dict['goal_rnd'] = goal_rnd
                wandb.log(log_dict)
        else:
            goal = self.env.observation_space.sample()

            # make initial goals just arm movement, copy others from initial state_short
            copy_idxs = list(range(4, 18)) + list(range(22, 39))
            for idx in copy_idxs:
                goal[idx] = state_short[idx]
            # goal[7:18] = 0 # no obj rotation
            # goal[10] = 1
            # goal[25:36] = 0
            # goal[28] = 1
            # goal[-3:] = state_short[-3:] # target loc
            # goal[4:7] = state_short[4:7] # obj location
            # goal[22:25] = state_short[22:25]

            goal = goal[:39]
            goal = torch.tensor(goal, dtype=torch.float32, device=self.device)
            with np.printoptions(precision=3, suppress=True):
                logging.debug(f'Goal location: \n{goal.cpu().numpy()}')
        
        for i in range(self.max_steps_per_episode):
            self.epi_step += 1
            self.cycle_step += 1
            self.step += 1
            if random.random() < 0.1:
                action = env.action_space.sample()
            elif random.random() < 0.4:
                action = self.rl_alg.select_action_with_noise(
                    torch.cat((torch.tensor(state_short, device=device, dtype=torch.float32), goal), dim=0))
            else:
                action = self.rl_alg.select_action(
                    torch.cat((torch.tensor(state_short, device=device, dtype=torch.float32), goal), dim=0))
            next_state, reward, terminal, truncate, info = env.step(action)
            next_state_short = next_state[:39].copy() #
            next_state_short[-3:] = 0
            if info['success'] > 0 and truncate:
                terminal = 1
            else:
                terminal = 0
            self.live_buffer.add(state_short, action, next_state_short, goal=goal.cpu().numpy())
            self.data.add(state, action, next_state, reward, terminal)
            gcrl_goal_dist = np.sum((next_state_short - goal.cpu().numpy())**2)
            state_change = np.sqrt(np.sum((next_state - state)**2))
            if self.use_video and cycle_idx % self.video_freq == 0 and i % 5 == 0:
                self.vid.add_frame(env, f'Cycle {cycle_idx} Ep {episode_idx} Step {i}\nExtrinsic reward (unused): {reward:.4f}\nObject to target: {info["obj_to_target"]}')
            if self.use_wandb:
                log_dict = {
                    'step': self.step,
                    'episode': self.episodes_per_cycle * cycle_idx + episode_idx + i / self.max_steps_per_episode,
                    'cycle': cycle_idx + episode_idx / self.episodes_per_cycle + i / self.max_steps_per_episode / self.episodes_per_cycle,
                    'reward': reward,
                    'success': info['success'],
                    'near_object': info['near_object'],
                    'obj_to_target': info['obj_to_target'],
                    'grasp_reward': info['grasp_reward'],
                    'in_place_reward': info['in_place_reward'],
                    'arm_x': state[0].item(),
                    'arm_y': state[1].item(),
                    'arm_z': state[2].item(),
                }
                if cycle_idx >= self.warm_up_cycles:
                    log_dict = {**log_dict,
                        'gcrl_goal_dist': gcrl_goal_dist,
                        'state_change': state_change,
                    }
                wandb.log(log_dict)
            if i % 100 == 0:
                with np.printoptions(precision=3, suppress=True):
                    logging.debug(f'Location: {state[:3]}, Action: {action}, Reward: {reward:.3E}, GCRL Goal Dist: {gcrl_goal_dist:.3E}, State Change: {state_change:.3E}')
            if truncate: # maybe too high?
                break
            state = next_state
            state_short = next_state_short

        if cycle_idx >= self.warm_up_cycles:
            h_bs = 32
            c_bs = 16
            r_shape_max = 0

            # sample last for concurrent part
            last_states, last_actions, last_next_states, last_goals = self.live_buffer.sample_last_n(self.epi_step)

            # for i in range(self.max_steps_per_episode):
            n_updates = self.max_steps_per_episode // 5
            for i in range(n_updates):
                
                # Hindsight part
                h_states, h_actions, h_next_states, h_goals = self.live_buffer.sample(h_bs)
                h_goal_samples = h_states[torch.randint(high=h_bs, size=(h_bs,)), :]
                h_goals_dup = torch.cat((h_goals, h_goal_samples), dim=0)
                h_states_dup = torch.cat((h_states, h_states), dim=0)
                h_actions_dup = torch.cat((h_actions, h_actions), dim=0)
                h_next_states_dup = torch.cat((h_next_states, h_next_states), dim=0)
                h_rewards = -(~(torch.all(torch.isclose(h_states_dup, h_goals_dup), dim=-1))).float()
                h_rewards /= self.max_steps_per_episode
                # h_r_shape = torch.clamp(r_shape_max / 100 / torch.sum((h_next_states - h_goals)**2, dim=-1), max=r_shape_max)
                # h_r_shape = r_shape_max - torch.sum((h_next_states - h_goals)**2, dim=-1) / 10
                h_r_shape = r_shape_max - torch.sum((h_next_states_dup - h_goals_dup)**2, dim=-1)**(0.25) / 20
                h_r_shape_mean = torch.mean(h_r_shape).item()
                # h_rewards[:h_bs] += h_r_shape
                h_rewards += h_r_shape
                h_terminals = torch.logical_or(
                    torch.all(torch.isclose(h_states_dup, h_goals_dup), dim=-1),
                    torch.all(torch.isclose(h_next_states_dup, h_goals_dup), dim=-1)).float()
                h_train_states = torch.cat((h_states_dup, h_goals_dup), dim=-1)
                h_train_next_states = torch.cat((h_next_states_dup, h_goals_dup), dim=-1)
                
                # Concurrent part
                b_range = random.sample(range(self.epi_step), c_bs)
                # range_start = np.random.randint(0, self.epi_step - c_bs)
                # b_range = slice(range_start, range_start + c_bs)
                c_states, c_actions, c_next_states, c_goals = last_states[b_range], last_actions[b_range], last_next_states[b_range], last_goals[b_range]
                c_goal_samples = c_states[torch.randint(high=c_bs, size=(c_bs,)), :]
                c_goals_dup = torch.cat((c_goals, c_goal_samples), dim=0)
                c_states_dup = torch.cat((c_states, c_states), dim=0)
                c_actions_dup = torch.cat((c_actions, c_actions), dim=0)
                c_next_states_dup = torch.cat((c_next_states, c_next_states), dim=0)
                c_rewards = -(~(torch.all(torch.isclose(c_states_dup, c_goals_dup), dim=-1))).float()
                c_rewards /= self.max_steps_per_episode
                # c_r_shape = torch.clamp(r_shape_max / 100 / torch.sum((c_next_states - c_goals)**2, dim=-1), max=r_shape_max)
                # c_r_shape = r_shape_max - torch.sum((c_next_states - c_goals)**2, dim=-1) / 10
                c_r_shape = r_shape_max - torch.sum((c_next_states_dup - c_goals_dup)**2, dim=-1)**(0.25) / 20
                c_r_shape_mean = torch.mean(c_r_shape).item()
                # c_rewards[:c_bs] += c_r_shape
                c_rewards += c_r_shape
                c_terminals = torch.logical_or(
                    torch.all(torch.isclose(c_states_dup, c_goals_dup), dim=-1),
                    torch.all(torch.isclose(c_next_states_dup, c_goals_dup), dim=-1)).float()
                c_train_states = torch.cat((c_states_dup, c_goals_dup), dim=-1)
                c_train_next_states = torch.cat((c_next_states_dup, c_goals_dup), dim=-1)

                full_states = torch.cat((h_train_states, c_train_states), dim=0)
                full_actions = torch.cat((h_actions_dup, c_actions_dup), dim=0)
                full_next_states = torch.cat((h_train_next_states, c_train_next_states), dim=0)
                full_rewards = torch.cat((h_rewards, c_rewards), dim=0)
                full_terminals = torch.cat((h_terminals, c_terminals), dim=0)
                # full_states = c_train_states_dup
                # full_actions = c_actions_dup
                # full_next_states = c_train_next_states_dup
                # full_rewards = c_rewards
                # full_terminals = c_terminals_dup

                critic_loss, actor_loss, pi_info = self.rl_alg.train_step((full_states, full_actions, full_next_states, full_rewards, full_terminals))
                avg_p_pi = np.mean(np.exp(pi_info['LogPi'][-c_bs:]))
                if self.use_wandb:
                    log_dict = {
                        'step': self.step,
                        # 'episode': self.episodes_per_cycle * cycle_idx + episode_idx + i / self.max_steps_per_episode,
                        # 'cycle': cycle_idx + episode_idx / self.episodes_per_cycle + i / self.max_steps_per_episode / self.episodes_per_cycle,
                        'episode': self.episodes_per_cycle * cycle_idx + episode_idx + i / n_updates,
                        'cycle': cycle_idx + episode_idx / self.episodes_per_cycle + i / n_updates / self.episodes_per_cycle,
                        'hindsight_reward_shape_sum': h_r_shape_mean,
                        'curr_reward_shape_sum': c_r_shape_mean,
                        'avg_p_pi': avg_p_pi,
                        'critic_loss': critic_loss,
                    }
                    if actor_loss is not None:
                        log_dict['actor_loss'] = actor_loss
                    wandb.log(log_dict)
                if (i+1) % 20 == 0:
                    if actor_loss is not None:
                        logging.debug(f'Step {i} Critic loss: {critic_loss:.3E}, Actor loss: {actor_loss:.3E}, Hind rshape mean: {h_r_shape_mean:.3E}, Curr rshape mean: {c_r_shape_mean:.3E}, Avg p_pi: {avg_p_pi:.3E}')
                    else:
                        logging.debug(f'Step {i} Critic loss: {critic_loss:.3E}, Hind rshape mean: {h_r_shape_mean:.3E}, Curr rshape mean: {c_r_shape_mean:.3E}, Avg p_pi: {avg_p_pi:.3E}')
                    # print(c_next_states[:, :3])
                    # print(c_goals[:, :3])
                    # print(c_r_shape)
                    # if actor_loss is not None:
                    #     logging.debug(f'Step {i} Critic loss: {critic_loss:.3E}, Actor loss: {actor_loss:.3E}, Curr reward shape sum: {c_r_shape_mean:.3E}')
                    # else:
                    #     logging.debug(f'Step {i} Critic loss: {critic_loss:.3E}, Curr reward shape sum: {c_r_shape_mean:.3E}')
    
    def on_cycle_end(self, cycle_idx):
        # if cycle_idx < self.warm_up_cycles:
        #     return
        cycle_state, _, _, _ = self.live_buffer.sample_last_n(self.cycle_step)
        for k in range(2):
            idx_order = torch.randperm(self.cycle_step).to(device)
            avg_epoch_loss = 0
            for i in range(0, self.cycle_step, 128): # batch size?
                batch_state = cycle_state[idx_order[i:i+128]]
                pred_out, target_out = self.rnd(batch_state)
                loss = torch.mean((pred_out - target_out)**2)
                avg_epoch_loss += loss.item() * len(batch_state)

                loss.backward()
                self.rnd_optimizer.step()
                self.rnd_optimizer.zero_grad()

                if self.use_wandb:
                    wandb.log({
                        'cycle': cycle_idx + k/4 + i/self.cycle_step/4,
                        'rnd_loss': loss,
                    })
            avg_epoch_loss /= self.cycle_step
            logging.debug(f'Avg RND Epoch {k} loss: {avg_epoch_loss}')
            

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--goal_sample_size', type=int, default=5000)
    parser.add_argument('--save_path', type=str, default='data/gcrl.npz')
    parser.add_argument('--env', type=str)

    args = parser.parse_args()

    PATH_LENGTH = 500
    STATE_DIM = 39
    # STATE_DIM = 18
    ACTION_DIM = 4

    ml1 = metaworld.ML1(args.env, seed=0)
    env = ml1.train_classes[args.env](render_mode='rgb_array')
    # env.set_task(ml1.train_tasks[0])
    # env.max_path_length = PATH_LENGTH
    # env._partially_observable = False
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    gcrl_rnd_col = GCRLRNDDataCollection(
        ml=ml1,
        env=env,
        out_path=args.save_path,
        # rl_alg=TD3(
        #     state_dim=STATE_DIM*2,
        #     action_dim=ACTION_DIM,
        #     max_action=1,
        #     device=device,
        #     actor_lr=3e-7,
        #     critic_lr=1e-5,
        # ),
        rl_alg=SAC(
            state_dim=STATE_DIM*2,
            action_dim=ACTION_DIM,
            max_action=1,
            hidden_dims=(300, 300),
            actor_lr=1e-4,
            critic_lr=1e-4,
            alpha=0.05,
        ).to(device),
        rnd=RNDModel(
            input_size=STATE_DIM,
            output_size=512,
        ).to(device),
        # state_dim=STATE_DIM,
        state_dim=39,
        action_dim=ACTION_DIM,
        num_cycles=200,
        episodes_per_cycle=10,
        warm_up_cycles=20,
        max_steps_per_episode=PATH_LENGTH,
        goal_sample_size=args.goal_sample_size,
        device=device,
        use_video=False,
        use_wandb=True,
    )
    gcrl_rnd_col.run()

