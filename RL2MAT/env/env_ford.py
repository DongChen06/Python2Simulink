import matlab.engine
import numpy as np

import gym
import time
import random
from collections import deque
from gym.utils import seeding
from .utils import *


DISCRETE_ACTIONS = {
    0: [0.3, 0.0],  #
    1: [0.5, -0.5],  #
    2: [0.5, 0.5],  #
    3: [1.0, 0.0],  #
}


class FordEnv(gym.Env):
    """
        This is the environment for ford project which is built on Matlab and python.
    """

    def __init__(self, modelName='tracking', discrete=True, render=True, time_step=765):
        # Setup gym environment
        self.modelName = modelName
        self.render = render
        self.time_step = time_step
        self.seed(66)
        if discrete is True:
            self.action_space = gym.spaces.Discrete(len(DISCRETE_ACTIONS))  # steer, throttle
            self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(*obs_res, 3), dtype=np.float32)
        else:
            high = np.array([100, 20, 100])
            self.action_space = gym.spaces.Box(np.array([-1, 0]), np.array([1, 1]), dtype=np.float32)  # steer, throttle
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

    def close(self):
        disconnect(self.eng, self.modelName)
        self.closed = True

    def render(self, ):
        updateGraph(fig1, fig2, tHist, x1Hist, xd1Hist)

    def step(self, action):
        if self.closed:
            raise Exception("FordEnv.step() called after the environment was closed." +
                            "Check for info[\"closed\"] == True in the learning loop.")
        if action is not None:
            self.obs_new = run_step(self.eng, self.modelName, action)

        self.last_reward = reward_fn()
        if self.render:
            self.render()

        if self.step >= self.time_step:
            self.terminal_state = True

        self.step += 1

        return self.obs_new, self.last_reward, self.terminal_state, {
            "closed": self.closed}


if __name__ == "__main__":
    # Example of using FordEnv with fixed controls
    env = FordEnv()
    action = np.zeros(env.action_space.shape[0])
    while True:
        env.reset()
        while True:
            action[0] = -0.5
            action[0] = 0.5
            action[0] = np.clip(action[0], -1, 1)

            # Take action
            obs, _, done, info = env.step(action)
            if info["closed"]:  # Check if closed
                exit(0)
            env.render()  # Render
            if done:
                break
        env.close()
