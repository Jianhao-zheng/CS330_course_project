import metaworld
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import time
import argparse
from tqdm import tqdm

from replay import ReplayBuffer

SEED = 0
STATE_DIM = 39
ACTION_DIM = 4
RANDOM_ACTION_KEY = " "
END_ITER_KEY = "f"
CV_WINDOW_NAME = "Robot"


def visualize(image):
    img_with_txt = image.copy()
    text_position = (10, 20)
    font = cv2.FONT_ITALIC
    font_scale = 0.6
    font_color = (1, 1, 1)  # White color
    line_type = 1
    cv2.putText(
        img_with_txt,
        "press 'space' to continue",
        text_position,
        font,
        font_scale,
        font_color,
        line_type,
        cv2.LINE_AA,
    )
    text_position = (10, 40)
    cv2.putText(
        img_with_txt,
        "press 'f' to end this iter",
        text_position,
        font,
        font_scale,
        font_color,
        line_type,
        cv2.LINE_AA,
    )

    cv2.imshow(CV_WINDOW_NAME, img_with_txt)


def main(args):
    # Construct the benchmark, sampling tasks
    ml1 = metaworld.ML1("pick-place-v2", seed=SEED)

    # Create an environment with task `pick_place`
    # arg:render_mode="rgb_array" is activated so that we can get images to monitor what's going on
    env = ml1.train_classes["pick-place-v2"](render_mode="rgb_array")

    if args.save_data:
        # set up reply buffer
        out_data = ReplayBuffer(
            STATE_DIM, ACTION_DIM, max_size=args.num_iter * args.max_path_length
        )
    for iter_idx in tqdm(range(args.num_iter)):
        # Set task (totally 50 tasks for this environment)
        # env.set_task(ml1.train_tasks[0])
        env.set_task(ml1.train_tasks[random.randint(0, 49)])
        env.max_path_length = args.max_path_length
        env._partially_observable = False
        if args.verbose:
            print("*" * 30)
            print("Environment reset")
        obs = env.reset()
        state = obs[0]

        if args.visual:
            visualize(env.render()[:, :, ::-1])
        truncate = False
        while not truncate:
            keystroke = cv2.waitKey(0)

            if args.visual:
                if keystroke == ord(RANDOM_ACTION_KEY):
                    # Randomly sample an action from the possible action space
                    # the action is an numpy array with shape 4 representing the following:
                    #           [delta(x), delta(y), delta(z), gripper_effort]
                    action = env.action_space.sample()
                elif keystroke == ord(END_ITER_KEY):
                    break
                else:
                    print("Undefined key")
                    continue
            else:
                action = env.action_space.sample()

            # run one step of action
            # The env.step function Returns:
            #            (np.ndarray): the last stable observation (39,) by concatenating [curr_obs(18,), prev_obs(18,), pos_goal(3,)]
            #                          An observation contains: [position of the end effector(3,),
            #                                                    gripper_distance_apart(1,),
            #                                                    position of the object1(3,), quaternion of the object1(4,),
            #                                                    position of the object2(3,), quaternion of the object2(4,),]
            #                          In this environment, states of obj2 are all zeros.
            #            (float): Reward
            #            (bool): termination_flag (step function will always return a false)
            #            (bool): True if the current path length == max path length
            #            (dict): all the other information
            # For more detailed information, check the function of Class SawyerXYZEnv and SawyerPickPlaceEnvV2
            obs, reward, terminal, truncate, info = env.step(action)

            next_state = obs
            out_data.add(state, action, next_state, reward, terminal)
            state = next_state

            if args.verbose:
                print(
                    "A new step is executed.\nAction: delta_x={:.6f}, delta_y={:.6f}, delta_z={:.6f}, gripper_effort={:.6f}".format(
                        action[0], action[1], action[2], action[3]
                    )
                )
                print(
                    "Task success: {}; Grasp success: {}".format(
                        info["success"], info["grasp_success"]
                    )
                )
                print(
                    "Total reward: {:.6f}; Grasp rewad: {:.6f}; In place reward: {:.6f}".format(
                        reward, info["grasp_reward"], info["in_place_reward"]
                    )
                )
                x, y, z, qw, qx, qy, qz = obs[4:11]
                print(
                    f"Current observation of object: position({x:.4f},{y:.4f},{z:.4f}), rotation({qw:.4f},{qx:.4f},{qy:.4f},{qz:.4f})"
                )
            if args.visual:
                visualize(env.render()[:, :, ::-1])
            if info["success"]:
                print("the task is successful!")
                break

        else:
            print("Maximum length of step reached!")
        cv2.destroyAllWindows()

    if args.save_data:
        out_data.save(args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("random sample")
    parser.add_argument(
        "--visual",
        default=False,
        action="store_true",
        help="whether to show the image of the current state of the robot",
    )
    parser.add_argument(
        "--save_data",
        default=False,
        action="store_true",
        help="whether to save the collected data into args.save_path",
    )
    parser.add_argument(
        "--verbose",
        default=False,
        action="store_true",
        help="whether to print information",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./collection.npz",
        help="path to save the collected data",
    )
    parser.add_argument(
        "--num_iter", type=int, default=1, help="number of iterations to run"
    )
    parser.add_argument(
        "--max_path_length",
        type=int,
        default=500,
        help="the maximum length of path, if this is reached, the iteration will end",
    )

    args = parser.parse_args()
    main(args)
