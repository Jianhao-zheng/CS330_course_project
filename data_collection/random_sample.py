import metaworld
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import time
import argparse

SEED = 0
RANDOM_ACTION_KEY = " "
END_ITER_KEY = "f"
CV_WINDOW_NAME = "Robot"

def visualize(image):
    img_with_txt = image.copy()
    text_position = (10, 20)
    font = cv2.FONT_ITALIC
    font_scale = 0.6
    font_color = (1, 1, 1) # White color
    line_type = 1
    cv2.putText(img_with_txt, "press 'space' to continue", text_position, font, font_scale, font_color, line_type,cv2.LINE_AA)
    text_position = (10, 40)
    cv2.putText(img_with_txt, "press 'f' to end this iter", text_position, font, font_scale, font_color, line_type,cv2.LINE_AA)

    cv2.imshow(CV_WINDOW_NAME, img_with_txt)

def main(args):
    # Construct the benchmark, sampling tasks
    ml1 = metaworld.ML1("pick-place-v2", seed=SEED)

    # Create an environment with task `pick_place`
    # arg:render_mode="rgb_array" is activated so that we can get images to monitor what's going on
    env = ml1.train_classes["pick-place-v2"](render_mode="rgb_array")
    # Set task (totally 50 tasks for this environment)
    env.set_task(ml1.train_tasks[0])
    env.max_path_length = args.max_path_length

    if args.save_data:
        collected_data = {'action':[[] for _ in range(args.num_iter)],'obs':[[] for _ in range(args.num_iter)],'info':[[] for _ in range(args.num_iter)],}
    for iter_idx in range(args.num_iter):
        print('*'*30)
        print('Environment reset')
        obs = env.reset()
        time.sleep(2)

        if args.visual:
            visualize(env.render()[:,:,::-1])
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
            obs, reward, done, truncate, info = env.step(action)

            print(
                "A new step is executed.\nAction: delta_x={:.6f}, delta_y={:.6f}, delta_z={:.6f}, gripper_effort={:.6f}".format(
                    action[0], action[1], action[2], action[3]
                )
            )
            print(
                "Task success: {}; Grasp success: {}".format(info["success"], info["grasp_success"])
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
            print("Object distance to target: {:.6f}".format(info['obj_to_target']))
            if info['obj_to_target'] < 0.2:
                time.sleep(10)
            if args.visual:
                visualize(env.render()[:,:,::-1])
            if args.save_data:
                collected_data['action'][iter_idx].append(action)
                collected_data['obs'][iter_idx].append(obs)
                collected_data['info'][iter_idx].append(info)
            if info['success']:
                print('the task is successful!')
                break
        else:
            print('Maximum length of step reached!')
        cv2.destroyAllWindows()

    if args.save_data:
        np.savez(args.save_path,collected_data=collected_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train a ProtoNet!')
    parser.add_argument('--visual', default=False, action='store_true',
                        help='whether to show the image of the current state of the robot')
    parser.add_argument('--save_data', default=False, action='store_true',
                        help='whether to save the collected data into args.save_path')
    parser.add_argument('--save_path', type=str, default='./collection.npz',
                        help='path to save the collected data')
    parser.add_argument('--num_iter', type=int, default=1,
                        help='number of iterations to run')
    parser.add_argument('--max_path_length', type=int, default=500,
                        help='the maximum length of path, if this is reached, the iteration will end')
    
    args = parser.parse_args()
    main(args)