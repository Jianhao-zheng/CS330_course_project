import torch
import metaworld
from tqdm import tqdm
import numpy as np

from copy import deepcopy

from data_collection.replay import ReplayBuffer
from data_collection.rnd import RNDModel
from data_collection.td3 import TD3


# quick and dirty implementation of the data collection algorithm
# from the paper
def run_gcrl_with_rnd(
    state_dim,
    action_dim,
    rl_alg,
    rnd,
    env,
    device,
    num_train_cycles=150,
    warm_up_cycles=2, # must be >=1
    num_epi_per_cycle=10,
    num_steps_per_epi=500,
) -> ReplayBuffer:
    rnd_optimizer = torch.optim.AdamW(rnd.predictor.parameters(), lr=0.001)

    live_buffer = ReplayBuffer(state_dim, action_dim, track_reward=False, track_terminal=False, track_truncate=False)
    out_data = ReplayBuffer(state_dim, action_dim, max_size=int(num_train_cycles*num_epi_per_cycle*num_steps_per_epi),
        track_reward=True, track_terminal=True, track_truncate=True)

    for c in range(num_train_cycles):
        print(f'Train cycle {c}')
        step_ct = 0
        for e in tqdm(range(num_epi_per_cycle)):
            state, _ = env.reset()
            
            if c >= warm_up_cycles:
                goal_candidates = live_buffer.sample(num_epi_per_cycle*num_steps_per_epi)[2]
                pred_out, target_out = rnd(goal_candidates) # batch this if needed
                goal_idx = torch.argmax(torch.sum((pred_out - target_out)**2), dim=-1)
                goal = goal_candidates[goal_idx].cpu().numpy()
            else:
                goal = env.observation_space.sample()
            
            for t in range(num_steps_per_epi):
                step_ct += 1
                action = rl_alg.select_action(np.concatenate((state, goal)))
                next_state, reward, terminal, truncate, info = env.step(action)
                live_buffer.add(state, action, next_state)
                out_data.add(state, action, next_state, reward, terminal, truncate)
                if terminal or truncate:
                    if terminal or ('success' in info and info['success']):
                        print('\n\nwooooo\n\n')
                    break
                state = next_state

            for t in range(num_steps_per_epi):
                states, actions, next_states = live_buffer.sample(128) # batch size?
                goal_samples = states[torch.randint(high=len(states), size=(len(states),)), :]
                rewards = -(~(torch.all(torch.isclose(states, goal_samples), dim=-1))).float() # isclose or strict equality?
                terminals = torch.logical_or(
                    torch.all(torch.isclose(states, goal_samples), dim=-1),
                    torch.all(torch.isclose(next_states, goal_samples), dim=-1)).float()
                train_states = torch.cat((states, goal_samples), dim=-1)
                train_next_states = torch.cat((states, goal_samples), dim=-1)
                rl_alg.train_step((train_states, actions, train_next_states, rewards, terminals))
        
        cycle_state, _, cycle_next_state = live_buffer.sample_last_n(step_ct)
        for k in range(4):
            idx_order = torch.randperm(step_ct).to(device)
            for i in range(0, step_ct, 128): # batch size?
                batch_state = cycle_state[idx_order[i:i+128]]
                pred_out, target_out = rnd(batch_state)
                loss = torch.mean((pred_out - target_out)**2)

                loss.backward()
                rnd_optimizer.step()
                rnd_optimizer.zero_grad()

    return out_data

if __name__ == '__main__':
    PATH_LENGTH = 2000
    STATE_DIM = 39
    ACTION_DIM = 4

    ml1 = metaworld.ML1("pick-place-v2", seed=0)
    env = ml1.train_classes["pick-place-v2"]()
    env.set_task(ml1.train_tasks[0])
    env.max_path_length = PATH_LENGTH
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    out_buffer = run_gcrl_with_rnd(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
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
        env=env,
        device=device,
        num_train_cycles=150,
        warm_up_cycles=10,
        num_epi_per_cycle=10,
        num_steps_per_epi=PATH_LENGTH
    )
    print(out_buffer.size, out_buffer.ptr)
    print(out_buffer.sample(5))
    out_buffer.save('data/gcrl_2000.npz')
