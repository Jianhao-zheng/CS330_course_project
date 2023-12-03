import numpy as np
import argparse
import d3rlpy
import metaworld


def main(args):
    ml1 = metaworld.ML1("pick-place-v2")
    eval_env = ml1.train_classes["pick-place-v2"](render_mode="rgb_array")
    eval_env.set_task(ml1.train_tasks[0])

    # form dataset in d3rlpy
    data = np.load(args.data_path)
    assert np.sum(data["terminal"]) == 0

    timeouts = np.zeros(data["state"].shape[0]).astype(int)
    for i in range(data["state"].shape[0]):
        if (i + 1) % args.max_length == 0:
            timeouts[i] = 1

    dataset = d3rlpy.dataset.MDPDataset(
        data["state"], data["action"], data["reward"], data["terminal"], timeouts
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

    td_error_evaluator = d3rlpy.metrics.TDErrorEvaluator(episodes=dataset.episodes)
    env_evaluator = d3rlpy.metrics.EnvironmentEvaluator(eval_env)

    iql.fit(
        dataset,
        n_steps=500000,
        evaluators={
            "td_error": td_error_evaluator,
            "environment": env_evaluator,
        },
    )

    iql.save_model(args.save_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_length",
        type=int,
        default=2000,
        help="Number of steps before resetting the environment. Same value as env.max_path_length when the data is collected.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/random_2000.npz",
        help="Path to the collected data",
    )
    parser.add_argument(
        "--save_model",
        type=str,
        default="model/iql.pt",
        help="Path to save the trained model",
    )

    args = parser.parse_args()
    main(args)
