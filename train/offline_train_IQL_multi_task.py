import numpy as np
import argparse
import d3rlpy
import metaworld

TASKS = [
    "reach",
    "window_open",
    "window_close",
    "button_press",
    "pick_place",
    "door_open",
    "drawer_open",
    "drawer_close",
    "peg_insert_side",
    "push",
]


def main(args):
    states = np.zeros((10000000, 40))
    actions = np.zeros((10000000, 4))
    rewards = np.zeros((10000000, 1))
    dones = np.zeros((10000000))
    timeouts = np.zeros((10000000))
    start_idx = 0
    for task_id, task in enumerate(TASKS):
        data = np.load("data/{}_{}.npz".format(args.alg, task))
        data_len = data["state"].shape[0]
        assert data_len == 1000000

        dones_temp = np.zeros((data_len))
        timeouts_temp = np.zeros((data_len))
        for i in range(data_len):
            if (i + 1) % args.max_length == 0:
                dones_temp[i] = data["terminal"][i]
                if dones_temp[i] == 0:
                    timeouts_temp[i] = 1

        states[start_idx : start_idx + data_len, :39] = data["state"].copy()
        states[start_idx : start_idx + data_len, 39] = task_id
        actions[start_idx : start_idx + data_len, :] = data["action"].copy()
        rewards[start_idx : start_idx + data_len, :] = data["reward"].copy()
        dones[start_idx : start_idx + data_len] = dones_temp.copy()
        timeouts[start_idx : start_idx + data_len] = timeouts_temp.copy()
        start_idx += data_len
    assert start_idx == 10000000
    assert np.sum(states[-1, :]) != 0

    dataset = d3rlpy.dataset.MDPDataset(
        states,
        actions,
        rewards,
        dones,
        timeouts,
    )

    print("dataset size:{}".format(dataset.size()))

    # scaler that do automatically normalization of the data
    observation_scaler = d3rlpy.preprocessing.StandardObservationScaler()
    action_scaler = d3rlpy.preprocessing.MinMaxActionScaler()
    reward_scaler = d3rlpy.preprocessing.StandardRewardScaler()

    iql = d3rlpy.algos.IQLConfig(
        observation_scaler=observation_scaler,
        action_scaler=action_scaler,
        reward_scaler=reward_scaler,
    ).create(device="cuda:0")

    # td_error_evaluator = d3rlpy.metrics.TDErrorEvaluator(episodes=dataset.episodes)
    # env_evaluator = d3rlpy.metrics.EnvironmentEvaluator(eval_env)

    # iql.fit(
    #     dataset,
    #     n_steps=500000,
    #     evaluators={
    #         "td_error": td_error_evaluator,
    #         "environment": env_evaluator,
    #     },
    # )
    iql.fit(
        dataset,
        n_steps=500000,
        logger_adapter=d3rlpy.logging.FileAdapterFactory(
            root_dir="d3rlpy_logs/IQL_multi_{}".format(args.alg)
        ),
    )

    # iql.save_model(args.save_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_length",
        type=int,
        default=500,
        help="Number of steps before resetting the environment. Same value as env.max_path_length when the data is collected.",
    )
    parser.add_argument(
        "--alg",
        type=str,
        default="random",
        help="data collected by which algorithm",
    )

    args = parser.parse_args()
    main(args)
