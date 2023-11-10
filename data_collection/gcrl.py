import torch

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
    num_epi_per_cycle=10,
    num_steps_per_epi=500,
) -> ReplayBuffer:
    rnd_optimizer = torch.optim.AdamW(rnd.predictor.parameters(), lr=0.001)

    live_buffer = ReplayBuffer(state_dim, action_dim, track_reward=False, track_terminal=False)
    out_data = ReplayBuffer(state_dim, action_dim, max_size=1e7)
    for c in range(num_train_cycles):
        step_ct = 0
        for e in range(num_epi_per_cycle):
            state, _ = env.reset()

            goal_candidates = live_buffer.sample(num_epi_per_cycle*num_steps_per_epi)[2]
            pred_out, target_out = rnd(goal_candidates) # batch this if needed
            goal_idx = torch.argmax(torch.sum((pred_out - target_out)**2), dim=-1)
            goal = goal_candidates[goal_idx]
            for t in range(num_steps_per_epi):
                step_ct += 1
                action = rl_alg.select_action(
                    torch.cat((torch.tensor(state, device=device), goal), dim=0))
                next_state, reward, terminal, truncate, info = env.step(action)
                live_buffer.add(state, action, next_state)
                out_data.add(state, action, next_state, reward, terminal)
                if terminal or truncate:
                    break
                state = next_state

            for t in range(num_steps_per_epi):
                states, actions, next_states = live_buffer.sample(128) # batch size?
                # TODO: unsure how to sample goals
                goal_samples = states[torch.randint(high=len(states), size=(len(states),))[0]]
                rewards = -torch.all(torch.isclose(states, goal_samples), dim=-1).float() # isclose or strict equality?
                terminals = torch.logical_or(
                    torch.all(torch.isclose(states, goal_samples), dim=-1),
                    torch.all(torch.isclose(next_states, goal_samples), dim=-1)).float()
                rl_alg.train_step((states, actions, next_states, rewards, terminals))
        
        cycle_state, _, cycle_next_state = live_buffer.sample_last_n(step_ct)
        for k in range(4):
            idx_order = torch.randperm(step_ct).to(device)
            for i in range(0, step_ct, 128): # batch size?
                batch_state = cycle_state[idx_order[i:i+128]]
                pred_out, target_out = rnd(batch_state)
                loss = torch.mean((pred_out - target_out)**2)

                loss.backward()
                rnd_optimizer.zero_grad()

    return out_data

if __name__ == '__main__':
    out_buffer = run_gcrl_with_rnd() # TODO: test functionality and modularize

