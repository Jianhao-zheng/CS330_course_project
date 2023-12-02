import d3rlpy
import metaworld
import matplotlib.pyplot as plt

import argparse

MAX_EPISODE_LENGTH = 500 # maybe more?

def main(args):
    num_models = len(args.model_paths)

    env = None # TODO

    completion_rates = [[] for _ in range(num_models)]
    rewards = [[] for _ in range(num_models)]

    for i, path in enumerate(args.model_paths):
        model = d3rlpy.load_learnable(path) # TODO figure out spec
        
        for j in range(args.num_samples):
            num_complete = 0
            for k in range(args.sample_size):
                curr_reward = 0
                obs, _ = env.reset()
                while True:
                    action = model.predict([obs])[0]
                    obs, reward, terminal, truncate, _ = env.step(action)
                    curr_reward += reward
                    if terminal:
                        num_complete += 1
                        break
                    if truncate:
                        break
                rewards[i].append(curr_reward) # do something with this idk
            completion_rates[i].append(num_complete / args.sample_size)
    
    plt.figure()


            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_size', type=int, default=20,
        help='Number of episodes for each bootstrapping sample.')
    parser.add_argument('--num_samples', type=int, default=20,
        help='Number of samples for boostrapping distribution')

    parser.add_argument('--model_paths', nargs='+', type=str)
    # TODO, figure out specs e.g. needing to specify tasks

    args = parser.parse_args()
    main(args)
