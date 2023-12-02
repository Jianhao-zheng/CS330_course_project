import metaworld
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import time
import argparse
import d3rlpy

from data_collection.replay import ReplayBuffer

SEED = 0
STATE_DIM = 39
ACTION_DIM = 4
RANDOM_ACTION_KEY = " "
END_ITER_KEY = "f"
CV_WINDOW_NAME = "Robot"
visual_id = 0


def visualize(image, dist):
    img_with_txt = image.copy()
    text_position = (10, 18)
    font = cv2.FONT_ITALIC
    font_scale = 0.5
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
    text_position = (10, 36)
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

    text_position = (10, 54)
    cv2.putText(
        img_with_txt,
        "Object distance to target: {:.6f}".format(dist),
        text_position,
        font,
        font_scale,
        font_color,
        line_type,
        cv2.LINE_AA,
    )

    cv2.imshow(CV_WINDOW_NAME, img_with_txt)
    global visual_id
    cv2.imwrite("imgs/frame_{}.png".format(visual_id), img_with_txt)
    visual_id += 1


def main(args):
    # Construct the benchmark, sampling tasks
    ml1 = metaworld.ML1("pick-place-v2", seed=SEED)

    # Create an environment with task `pick_place`
    # arg:render_mode="rgb_array" is activated so that we can get images to monitor what's going on
    env = ml1.train_classes["pick-place-v2"](render_mode="rgb_array")
    eval_env_orders = list(range(50))
    random.shuffle(eval_env_orders)

    # load model from the checkpoint
    model = d3rlpy.load_learnable(args.ckpt, device="cuda:0")

    # count the number of successful environment
    count_success = 0

    assert args.num_eval_tasks <= 50
    for eval_env_idx in range(args.num_eval_tasks):
        # Set task (totally 50 tasks for this environment)
        env.set_task(ml1.train_tasks[eval_env_orders[eval_env_idx]])
        env.max_path_length = args.max_path_length
        env._partially_observable = False
        if args.verbose:
            print("*" * 30)
            print("Environment reset")
        obs = env.reset()
        state = obs[0]
        obs = obs[0]

        if args.visual:
            visualize(env.render()[:, :, ::-1], np.linalg.norm(state[4:7] - state[-3:]))
        truncate = False
        while not truncate:
            keystroke = cv2.waitKey(0)
            if args.verbose:
                print("predict action")

            if args.visual:
                if keystroke == ord(RANDOM_ACTION_KEY):
                    action = model.predict(obs.reshape((1, -1)))[0]
                elif keystroke == ord(END_ITER_KEY):
                    break
                else:
                    if args.verbose:
                        print("Undefined key")
                    continue
            else:
                action = model.predict(obs.reshape((1, -1)))[0]

            obs, reward, terminal, truncate, info = env.step(action)

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
                print("Object distance to target: {:.6f}".format(info["obj_to_target"]))

            if args.visual:
                visualize(env.render()[:, :, ::-1], info["obj_to_target"])
            if info["success"]:
                count_success += 1
                if args.verbose:
                    print("the task is successful!")
                break
        else:
            if args.verbose:
                print("Maximum length of step reached!")
        cv2.destroyAllWindows()

    print(
        "The overall success rate is : {:.2f}".format(
            count_success / args.num_eval_tasks
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("evaluate the trained model")
    parser.add_argument(
        "--visual",
        default=False,
        action="store_true",
        help="whether to show the image of the current state of the robot",
    )
    parser.add_argument(
        "--verbose",
        default=False,
        action="store_true",
        help="whether to print information",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="./model/expert_iql.d3",
        help="path to the saved model",
    )
    parser.add_argument(
        "--max_path_length",
        type=int,
        default=500,
        help="the maximum length of path, if this is reached, the iteration will end",
    )
    parser.add_argument(
        "--num_eval_tasks",
        type=int,
        default=50,
        help="number of evaluated tasks, should be in the range of [1,50]",
    )

    args = parser.parse_args()
    main(args)
