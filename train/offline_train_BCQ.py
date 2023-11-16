import numpy as np
import d3rlpy
import metaworld

ml1 = metaworld.ML1("pick-place-v2")
env = ml1.train_classes["pick-place-v2"](render_mode="rgb_array")
env.set_task(ml1.train_tasks[0])

MAX_LENGTH = 2000

# form dataset in d3rlpy
data = np.load("data/gcrl_2000.npz")
assert np.sum(data["terminal"]) == 0

dataset = d3rlpy.dataset.MDPDataset(
    data["state"],
    data["action"],
    data["reward"],
    data["terminal"],
    data["truncate"].reshape(-1),
)
print(dataset.size())


# scaler that do automatically normalization of the data
observation_scaler = d3rlpy.preprocessing.StandardObservationScaler()
action_scaler = d3rlpy.preprocessing.MinMaxActionScaler()
reward_scaler = d3rlpy.preprocessing.StandardRewardScaler()

bcq = d3rlpy.algos.BCQConfig(
    observation_scaler=observation_scaler,
    action_scaler=action_scaler,
    reward_scaler=reward_scaler,
    batch_size=32,
    actor_learning_rate=2.5e-4,
    critic_learning_rate=2.5e-4,
    update_actor_interval=100,
).create(device="cuda:0")

td_error_evaluator = d3rlpy.metrics.TDErrorEvaluator(episodes=dataset.episodes)
env_evaluator = d3rlpy.metrics.EnvironmentEvaluator(env)

bcq.fit(
    dataset,
    n_steps=500000,
    evaluators={
        "td_error": td_error_evaluator,
        "environment": env_evaluator,
    },
)

bcq.save_model("gcrl_2000.pt")
