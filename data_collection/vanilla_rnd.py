import torch
import numpy as np
import metaworld
from tqdm import tqdm
from pathlib import Path
import os

from replay import ReplayBuffer
from rnd import RNDModel
from td3 import TD3
from render_video import MetaWorldVideo
from normalizer import RunningNormalizer

OUT_DIR = './rnd_data14'



def run_vanilla_rnd(
    state_dim,
    action_dim,
    rl_alg,
    rnd,
    env,
    device,
    num_train_cycles=150,
    num_epi_per_cycle=10,
    num_steps_per_epi=500,
    num_rand_actions=100,
    num_opt_iters=10,
) -> ReplayBuffer:
    
    max_samples = num_train_cycles * num_epi_per_cycle * num_steps_per_epi
    rnd_optimizer = torch.optim.Adam(rnd.predictor.parameters(), lr=1e-5)
    rnd_reward_normalizer = RunningNormalizer(device=device)
    rnd_obs_normalizer = RunningNormalizer(device=device, clamp=(-5,5))
    out_data = ReplayBuffer(state_dim, action_dim, max_size=max_samples)

    max_reward = 0
    max_reward_iter = (0,0,0)

    vid = MetaWorldVideo()
    for c in range(num_train_cycles):
        #vid = MetaWorldVideo() if c % 1 == 0 else None
        
        state, _ = env.reset()
        rnd_reward_normalizer.reset()
        rnd_obs_normalizer.reset()
        cycle_done = False

        # Initial random actions
        for r in range(num_rand_actions):
            action = env.action_space.sample()
            obs, reward, terminal, truncate, info = env.step(action)
            if reward > max_reward:
                max_reward = reward
                max_reward_iter = f'Initial random action {r}'
            rnd_obs_normalizer.update(obs)


        # Training episodes
        for e in tqdm(range(num_epi_per_cycle), desc=f'Cycle {c} episodes'):
            rnd_rewards = []

            for t in range(num_steps_per_epi):

                action = rl_alg.select_action(state)
                
                next_state, reward, terminal, truncate, info = env.step(action)
                if reward > max_reward:
                    max_reward = reward
                    max_reward_iter = (c,e,t)

                pred_out, target_out = rnd(rnd_obs_normalizer(state))
                rnd_reward = torch.sum(torch.square(pred_out - target_out))
                rnd_reward_normalizer.update(rnd_reward)

                rnd_rewards.append(rnd_reward)
                out_data.add(state, action, next_state, reward, terminal)
                if terminal or truncate:
                    cycle_done = True
                    break
                state = next_state

                # TODO: Update reward normalization parameters

                if vid and t % 10 == 0:
                    vid.add_frame(env, f'Cycle {c} Ep {e}\nRND reward: {rnd_reward:.8f}\nExtrinsic reward (unused): {reward:.4f}')

            ep_step_count = len(rnd_rewards)
            rnd_rewards = torch.stack(rnd_rewards)
            rnd_rewards = rnd_reward_normalizer(rnd_rewards, do_center=False, batched=True)
            #rnd_rewards = torch.nn.functional.normalize(rnd_rewards, dim=0)
            #rnd_rewards = (rnd_rewards - rnd_rewards.mean()) / rnd_rewards.std()
            # TODO: Update observation normalization parameters

            states, actions, next_states, _, terminals = out_data.sample_last_n(ep_step_count)
            for o in range(num_opt_iters):
                idx_order = torch.randperm(ep_step_count).to(device)
                rl_alg.train_step((states[idx_order],
                                   actions[idx_order],
                                   next_states[idx_order],
                                   rnd_rewards[idx_order],
                                   terminals[idx_order].flatten()))
                # RND opt step:
                rnd_obs_normalizer.update(states[idx_order], batched=True)
                pred_out, target_out = rnd(rnd_obs_normalizer(states[idx_order], batched=True))
                rnd_losses = torch.square(pred_out - target_out).sum(1)
                rnd_losses = rnd_reward_normalizer(rnd_losses, do_center=False, batched=True)
                rnd_losses.mean().backward()
                rnd_optimizer.step()
                rnd_optimizer.zero_grad()

            if cycle_done:
                break

        if vid and c % 10 == 9:
            vid.save(f'{OUT_DIR}/{c-9}.mp4')
            vid = MetaWorldVideo()

    print(f'Max reward: {max_reward}')
    print(f'Max reward iter: {max_reward_iter}')

    return out_data



if __name__ == '__main__':
    EP_LENGTH = 50
    EPS_PER_CYCLE = 25
    OPT_ITERS = 32
    NUM_RAND_ACTS = 2048
    STATE_DIM = 39
    ACTION_DIM = 4

    if not Path(OUT_DIR).exists():
        os.makedirs(OUT_DIR)

    ml1 = metaworld.ML1("pick-place-v2", seed=0)
    env = ml1.train_classes["pick-place-v2"](render_mode="rgb_array")
    env.set_task(ml1.train_tasks[0])
    env.max_path_length = EP_LENGTH * EPS_PER_CYCLE + NUM_RAND_ACTS + 1
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    out_buffer = run_vanilla_rnd(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        rl_alg=TD3(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            max_action=1,
            tau=0.4,
            policy_noise=3.0,
            noise_clip=10.0,
            actor_lr=1e-4,
            critic_lr=1e-4,
            device=device,
        ),
        rnd=RNDModel(
            input_size=STATE_DIM,
            output_size=512,
        ).to(device),
        env=env,
        device=device,
        num_steps_per_epi=EP_LENGTH,
        num_epi_per_cycle=EPS_PER_CYCLE,
        num_opt_iters=OPT_ITERS,
        num_rand_actions=NUM_RAND_ACTS,
        num_train_cycles=600,
    )

    #print(out_buffer.size, out_buffer.ptr)
    #print(out_buffer.sample(5))

    out_buffer.save(f'{OUT_DIR}/vanilla_rnd.npz')
