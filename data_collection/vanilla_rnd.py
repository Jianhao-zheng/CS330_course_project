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

OUT_DIR = './rnd_data1'



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
    num_opt_iters=100,
) -> ReplayBuffer:
    
    max_samples = num_train_cycles * num_epi_per_cycle * num_steps_per_epi
    rnd_optimizer = torch.optim.Adam(rnd.predictor.parameters(), lr=0.001)
    out_data = ReplayBuffer(state_dim, action_dim, max_size=max_samples)

    max_reward = 0
    max_reward_iter = (0,0,0)

    # TODO: initialize observation normalization parameters

    for c in range(num_train_cycles):
        vid = MetaWorldVideo() if c % 1 == 0 else None
        
        state, _ = env.reset()

        for e in tqdm(range(num_epi_per_cycle), desc='Training episodes'):
            rnd_rewards = []

            for t in range(num_steps_per_epi):

                action = rl_alg.select_action(state)
                
                next_state, reward, terminal, truncate, info = env.step(action)
                if reward > max_reward:
                    max_reward = reward
                    max_reward_iter = (c,e,t)


                pred_out, target_out = rnd(torch.tensor(state).float().to(device))
                rnd_reward = torch.sum(torch.square(pred_out - target_out))

                rnd_rewards.append(rnd_reward)
                out_data.add(state, action, next_state, reward, terminal)
                if terminal or truncate:
                    break
                state = next_state

                # TODO: Update reward normalization parameters

                if vid and t % 10 == 0:
                    vid.add_frame(env)

            ep_step_count = len(rnd_rewards)
            rnd_rewards = torch.stack(rnd_rewards)
            rnd_rewards = torch.nn.functional.normalize(rnd_rewards, dim=0)
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
                pred_out, target_out = rnd(states[idx_order])
                rnd_loss = torch.square(pred_out - target_out).sum(1).mean()
                rnd_loss.backward()
                rnd_optimizer.step()
                rnd_optimizer.zero_grad()

        if vid:
            vid.save(f'{OUT_DIR}/{c}.mp4')

    print(f'Max reward: {max_reward}')
    print(f'Max reward iter: {max_reward_iter}')

    return out_data



if __name__ == '__main__':
    EP_LENGTH = 500
    EPS_PER_CYCLE = 5
    STATE_DIM = 39
    ACTION_DIM = 4

    if not Path(OUT_DIR).exists():
        os.makedirs(OUT_DIR)

    ml1 = metaworld.ML1("pick-place-v2", seed=0)
    env = ml1.train_classes["pick-place-v2"](render_mode="rgb_array")
    env.set_task(ml1.train_tasks[0])
    env.max_path_length = EP_LENGTH * EPS_PER_CYCLE
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    out_buffer = run_vanilla_rnd(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        rl_alg=TD3(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            max_action=1,
            actor_lr=3e-4,
            critic_lr=3e-4,
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
        num_train_cycles=5,
    )

    print(out_buffer.size, out_buffer.ptr)
    print(out_buffer.sample(5))

    out_buffer.save(f'{OUT_DIR}/vanilla_rnd.npz')
