import matlab.engine
import numpy as np

import sys
sys.path.append("../")

import gym
import argparse
import configparser
import time
import random
from collections import deque
from gym.utils import seeding
from env.utils import *


discrete_resolution = 10


def parse_args():
    default_base_dir = '/home/derek/PycharmProjects/Python2Simulink/DDQN_Ford/Data'
    default_config_dir = 'DDQN_Ford\config\config_ford.ini'
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

    def __init__(self, config, modelName='tracking', discrete=True, time_step=765):
        # Setup gym environment
        self.modelName = config.get('modelName')
        self.model_address = config.get('modelAddress')
        # file name of parameters, we need to run it first
        self.rendering = int(config.getfloat('rendering'))
        self.sample_time = config.getfloat('sample_time')
        self.episode_length = int(config.getfloat('episode_length'))
        self.seed(66)

        low = np.array([0, -1e4, -1e4, -1e4, 0, 0, -1e4])
        high = np.array([100, 1e4, 1e4, 1e4, 1, 1, 1e4])

        if discrete is True:
            self.action_space = gym.spaces.Discrete(discrete_resolution)
            self.observation_space = gym.spaces.Box(
                low, high, dtype=np.float32)
        else:
            self.action_space = gym.spaces.Box(np.array([-1, 0]), np.array([1, 1]),
                                               dtype=np.float32)
            self.observation_space = gym.spaces.Box(
                -high, high, dtype=np.float32)

        try:

            # initialize matlab and env
            self.engMAT = MatEng()

        except Exception as e:
            self.close()
            raise e

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, ):
        self.steps = 0
        # reset the matlab model
        self.obs = self.engMAT.reset_env(self.rendering)

    def close(self):
        self.engMAT.disconnect()

    def render(self, ):
        self.engMAT.updateFig()

    def step(self, action):
        if action is not None:
            obs_new, self.last_reward, self.terminal_state, _ = self.engMAT.run_step(
                action)

        if self.rendering:
            self.render()

        if self.steps >= int(self.episode_length / self.sample_time) - 1:
            self.terminal_state = True

        self.steps += 1

        return obs_new, self.last_reward, self.terminal_state, _


if __name__ == "__main__":
    args = parse_args()
    base_dir = args.base_dir
    config_dir = args.config_dir
    config = configparser.ConfigParser()
    config.read(config_dir)
    epoch = 0
    # Example of using FordEnv with sample controller
    env = FordEnv(config['ENV_CONFIG'])
    action_size = env.action_space.n
    print('--------------')
    print("Simulation starting...")
    while True:
        env.reset()
        rewards = 0
        last_reward = 0
        while True:
            # print('--------------')
            # print("steps = ", env.steps)
            # print("rewards = ", last_reward)
            action = np.random.randint(action_size, size=1)
            # Take an action
            obs, last_reward, done, _ = env.step(4)  # action[0], 4
            rewards += last_reward
            if done:
                break
        print('--------------')
        print("steps = ", env.steps)
        print("rewards = ", rewards)
        epoch += 1
    env.close()
