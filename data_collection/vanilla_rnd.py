import torch
import numpy as np
import metaworld
from tqdm import tqdm

from replay import ReplayBuffer
from rnd import RNDModel
from td3 import TD3



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
) -> ReplayBuffer:
    
    rnd_optimizer = torch.optim.AdamW(rnd.predictor.parameters(), lr=0.001)
    out_data = ReplayBuffer(state_dim, action_dim, max_size=int(1e7))

    for c in tqdm(range(num_train_cycles), desc='Training cycles'):

        cycle_step_count = 0
        for e in range(num_epi_per_cycle):
            state, _ = env.reset()

            rnd_rewards = []
            ep_step_count = 0
            for t in range(num_steps_per_epi):
                ep_step_count += 1
                pred_out, target_out = rnd(torch.tensor(state).float().to(device))
                rnd_rewards.append(torch.sum(torch.square(pred_out - target_out)))

                action = rl_alg.select_action(state)
                
                next_state, reward, terminal, truncate, info = env.step(action)
                out_data.add(state, action, next_state, reward, terminal)
                if terminal or truncate:
                    break
                state = next_state

            rnd_rewards = torch.stack(rnd_rewards)
            states, actions, next_states, _, terminals = out_data.sample_last_n(ep_step_count)
            rl_alg.train_step((states, actions, next_states, rnd_rewards, terminals.flatten()))
            
            cycle_step_count += ep_step_count

        cycle_states, _, _, _, _ = out_data.sample_last_n(cycle_step_count)
        for _ in range(4):
            idx_order = torch.randperm(cycle_step_count).to(device)
            for i in range(0, cycle_step_count, 128):
                batch_state = cycle_states[idx_order[i:i+128]]
                pred_out, target_out = rnd(batch_state)
                loss = torch.mean(torch.sum(torch.square((pred_out - target_out)), dim=1))
                loss.backward()
                rnd_optimizer.step()
                rnd_optimizer.zero_grad()

    return out_data



if __name__ == '__main__':
    PATH_LENGTH = 500
    STATE_DIM = 39
    ACTION_DIM = 4

    ml1 = metaworld.ML1("pick-place-v2", seed=0)
    env = ml1.train_classes["pick-place-v2"](render_mode="rgb_array")
    env.set_task(ml1.train_tasks[0])
    env.max_path_length = PATH_LENGTH
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    out_buffer = run_vanilla_rnd(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        rl_alg=TD3(
            state_dim=STATE_DIM,
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
        num_steps_per_epi=PATH_LENGTH,
        num_train_cycles=150
    )

    print(out_buffer.size, out_buffer.ptr)
    print(out_buffer.sample(5))

    out_buffer.save('./vanilla_rnd.npz')
