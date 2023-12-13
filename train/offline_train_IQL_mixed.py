import numpy as np
import argparse
import d3rlpy
import metaworld

def main(args):
    UPSAMPLE_FACTOR = args.upsample

    # form dataset in d3rlpy
    unsup_data = np.load(args.unsup_path)

    unsup_dones = np.zeros((unsup_data["state"].shape[0]))
    unsup_timeouts = np.zeros((unsup_data["state"].shape[0]))
    for i in range(unsup_data["state"].shape[0]):
        if (i + 1) % args.max_length == 0:
            unsup_dones[i] = unsup_data["terminal"][i]
            if unsup_dones[i] == 0:
                unsup_timeouts[i] = 1

    expert_data = np.load(args.expert_path)
    k_steps = int(500 * args.k)

    expert_state = expert_data["state"][:k_steps]
    expert_action = expert_data["action"][:k_steps]
    expert_reward = expert_data["reward"][:k_steps, 0]
    expert_dones = np.zeros((expert_state.shape[0]))
    expert_timeouts = np.zeros((expert_state.shape[0]))
    for i in range(expert_state.shape[0]):
        if (i + 1) % args.max_length == 0:
            expert_dones[i] = expert_data["terminal"][i]
            if expert_dones[i] == 0:
                expert_timeouts[i] = 1
    # expert_dones = expert_data["terminal"][:k_steps, 0]
    # expert_timeouts = expert_dones

    ds_len = unsup_data["state"].shape[0] + UPSAMPLE_FACTOR * k_steps
    mix_state = np.zeros((ds_len, unsup_data["state"].shape[1]))
    mix_action = np.zeros((ds_len, unsup_data["action"].shape[1]))
    mix_reward = np.zeros(ds_len)
    mix_dones = np.zeros(ds_len)
    mix_timeouts = np.zeros(ds_len)

    interleave_unsup = unsup_data["state"].shape[0] // UPSAMPLE_FACTOR
    interleave_delta = interleave_unsup + k_steps
    print(interleave_unsup, interleave_delta)
    for i in range(UPSAMPLE_FACTOR):
        base = i * interleave_delta
        mix_state[base:base+interleave_unsup] = unsup_data["state"][i*interleave_unsup:(i+1)*interleave_unsup]
        mix_state[base+interleave_unsup:base+interleave_delta] = expert_state
        mix_action[base:base+interleave_unsup] = unsup_data["action"][i*interleave_unsup:(i+1)*interleave_unsup]
        mix_action[base+interleave_unsup:base+interleave_delta] = expert_action
        mix_reward[base:base+interleave_unsup] = unsup_data["reward"][i*interleave_unsup:(i+1)*interleave_unsup, 0]
        mix_reward[base+interleave_unsup:base+interleave_delta] = expert_reward
        mix_dones[base:base+interleave_unsup] = unsup_dones[i*interleave_unsup:(i+1)*interleave_unsup]
        mix_dones[base+interleave_unsup:base+interleave_delta] = expert_dones
        mix_timeouts[base:base+interleave_unsup] = unsup_timeouts[i*interleave_unsup:(i+1)*interleave_unsup]
        mix_timeouts[base+interleave_unsup:base+interleave_delta] = expert_timeouts

    dataset = d3rlpy.dataset.MDPDataset(
        mix_state,
        mix_action,
        mix_reward,
        mix_dones,
        mix_timeouts,
    )

    print(dataset.size())

    # scaler that do automatically normalization of the data
    observation_scaler = d3rlpy.preprocessing.StandardObservationScaler()
    action_scaler = d3rlpy.preprocessing.MinMaxActionScaler()
    reward_scaler = d3rlpy.preprocessing.StandardRewardScaler()

    iql = d3rlpy.algos.IQLConfig(
        observation_scaler=observation_scaler,
        action_scaler=action_scaler,
        reward_scaler=reward_scaler,
    ).create(device="cuda:0")

    iql.fit(
        dataset,
        n_steps=args.train_steps,
        logger_adapter=d3rlpy.logging.FileAdapterFactory(
            root_dir="d3rlpy_logs/{}".format(args.log)
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_length",
        type=int,
        default=500,
        help="Number of steps before resetting the environment. Same value as env.max_path_length when the data is collected.",
    )
    parser.add_argument(
        "--unsup_path",
        type=str,
        default="data/cic_door_open.npz",
        help="Path to the collected data",
    )
    parser.add_argument(
        "--expert_path",
        type=str,
        default="data/expert/door_open.npz",
        help="Path to the collected data",
    )
    parser.add_argument(
        "--log",
        type=str,
        default="door_open/IQL_cic_100",
        help="Path to save the trained model",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="num shots"
    )
    parser.add_argument(
        "--upsample",
        type=int,
        default=20,
        help="asdf"
    )
    parser.add_argument(
        "--train_steps",
        type=int,
        default=500000,
        help="Number of training steps"
    )

    args = parser.parse_args()
    main(args)
