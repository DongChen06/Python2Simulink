from __future__ import print_function, division
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import gym
from deepq import deepq
import tensorflow as tf
import tensorflow.contrib.layers as layers


def model(inpt, num_actions, scope, reuse=False):
    """This model takes as input an observation and returns values of all actions."""
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        out = layers.fully_connected(out, num_outputs=128, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out


def main():
    # Initialize environment
    print("Initializing environment")
    env = CarlaEnv(action_smoothing=0.2,
                  synchronous=True, fps=30)
   #  env = gym.make("CartPole-v0")

    act = deepq.learn(
        env,
        network='mlp',
        lr=1e-4,
        total_timesteps=2000000,
        buffer_size=500000,
        exploration_fraction=0.5,
        exploration_final_eps=0.1,
        print_freq=20,
        callback=callback
    )
    print("Saving model to cartpole_model.pkl")
    act.save("carla_model.pkl")


if __name__ == '__main__':
    main()