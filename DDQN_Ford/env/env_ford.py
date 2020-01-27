import matlab.engine
import numpy as np

import gym
import argparse
import configparser
import time
import random
from collections import deque
from gym.utils import seeding
from utils import *


discrete_resolution = 10


def parse_args():
    default_base_dir = 'C:\Users\Dong\PycharmProjects\Python2Simulink\RL2MAT\Data'
    default_config_dir = '.config/config.ford.ini'
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str, required=False,
                        default=default_base_dir, help="experiment base dir")
    parser.add_argument('--config-dir', type=str, required=False,
                        default=default_config_dir, help="experiment config dir")
    parser.add_argument('--is_training', type=str, required=False,
                        default=True, help="True=train, False=evaluation")
    parser.add_argument('--test-mode', type=str, required=False,
                    default='no_test',
                    help="test mode during training",
                    choices=['no_test', 'in_train_test', 'after_train_test', 'all_test'])

    args = parser.parse_args()
    if not args.option:
        parser.print_help()
        exit(1)
    return args


class FordEnv(gym.Env):
    """
    This is the environment for ford project which is built on Matlab and python.

    Observation:
    Type: Box(7)
    Num	Observation                 Min         Max
    0	VehicleSpd_mph               0          100
    1	Engine_Spd_c__radps         -1e4        1e4
    2	MG1_Spd_radps               -1e4        1e4
    3	MG2_Spd_radps               -1e4        1e4
    4   Acc_pad                      0           1
    5   Dec_pad                      0           1
    6   WheelTqDemand_Nm           -1e4         1e4

    Actions:
        Type: Discrete(discrete_resolution)
        Num	Action
        0	Push cart to the left
        1	Push cart to the right
    """

    def __init__(self, config, modelName='tracking', discrete=True, render=True, time_step=765):
        # Setup gym environment

        self.modelName = config.get('modelName')
        self.model_address = config.get('modelAddress')
        # file name of parameters, we need to run it first
        self.render = int(config.get('render'))
        self.episode_length = int(config.get('episode_length'))
        self.seed(66)

        low = np.array([0, -1e4, -1e4, -1e4, 0, 0, -1e4])
        high = np.array([100, 1e4, 1e4, 1e4, 1, 1, 1e4])

        if discrete is True:
            self.action_space = gym.spaces.Discrete(discrete_resolution)
            self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)
        else:
            self.action_space = gym.spaces.Box(np.array([-1, 0]), np.array([1, 1]),
                                               dtype=np.float32)  # steer, throttle
            self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

        try:
            print("Starting matlab")
            self.eng = matlab.engine.start_matlab()

            # initialize matlab and env
            self.obs = connectToMatlab(self.eng, modelName)

        except Exception as e:
            self.close()
            raise e

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, ):
        self.step = 0
        # initialize plot
        initialize_plot(tHist, x1Hist, xd1Hist)
        # reset the matlab model
        self.obs = reset_env(self.eng, self.modelName)

    def close(self):
        disconnect(self.eng, self.modelName)
        self.closed = True

    def render(self, ):
        updateFig(fig1, fig2, tHist, x1Hist, xd1Hist)

    def step(self, action):
        if self.closed:
            raise Exception("FordEnv.step() called after the environment was closed." +
                            "Check for info[\"closed\"] == True in the learning loop.")
        if action is not None:
            self.obs_new = run_step(self.eng, self.modelName, action)

        self.last_reward = reward_fn()
        if self.render:
            self.render()

        if self.step >= self.episode_length:
            self.terminal_state = True

        self.step += 1

        return self.obs_new, self.last_reward, self.terminal_state, {
            "closed": self.closed}


if __name__ == "__main__":
    args = parse_args()
    base_dir = args.base_dir
    config_dir = args.config_dir
    config = configparser.ConfigParser()
    config.read(config_dir)
    # Example of using FordEnv with fixed controls
    env = FordEnv(config['ENV_CONFIG'])
    a_size = env.action_space.shape[0]
    while True:
        env.reset()
        while True:
            action = np.random.randint(a_size, size=1)
            # Take action
            obs, _, done, info = env.step(action[0])
            if info["closed"]:  # Check if closed
                exit(0)
            env.render()  # Render
            if done:
                break
        env.close()
