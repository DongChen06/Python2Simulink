from __future__ import print_function, division
import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import configparser
import gym
import tensorflow as tf
import tensorflow.contrib.layers as layers
from .env.env_ford import FordEnv
from .utils import *
from .trainer import *
from .agents.models import IQL


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


def train_fn(args):
    base_dir = args.base_dir
    dirs = init_dir(base_dir)
    init_log(dirs['log'])
    config_dir = args.config_dir
    copy_file(config_dir, dirs['data'])
    config = configparser.ConfigParser()
    config.read(config_dir)
    in_test, post_test = init_test_flag(args.test_mode)  # test during training, test after training

    # Initialize environment
    print("Initializing environment")
    env = FordEnv(config['ENV_CONFIG'])
    #  env = gym.make("CartPole-v0")

    #logging.info('Training: s dim: %d, a dim %d, s dim ls: %r, a dim ls: %r' %
    #             (env.n_s, env.n_a, env.n_s_ls, env.n_a_ls))

    # init step counter
    total_step = int(config.getfloat('TRAIN_CONFIG', 'total_step'))
    test_step = int(config.getfloat('TRAIN_CONFIG', 'test_interval'))
    log_step = int(config.getfloat('TRAIN_CONFIG', 'log_interval'))
    global_counter = Counter(total_step, test_step, log_step)

    seed = config.getint('ENV_CONFIG', 'seed')

    model = IQL()

    summary_writer = tf.summary.FileWriter(dirs['log'])
    trainer = Trainer(env, model, global_counter, summary_writer, in_test, output_path=dirs['data'])
    trainer.run()

    # post-training test
    if post_test:
        tester = Tester(env, model, global_counter, summary_writer, dirs['data'])
        tester.run_offline(dirs['data'])

    # save model
    final_step = global_counter.cur_step
    logging.info('Training: save final model at step %d ...' % final_step)
    model.save(dirs['model'], final_step)

def evaluate_fn(args):
    pass


if __name__ == '__main__':
    args = parse_args()
    if args.is_training is True:
        train_fn(args)
    else:
        evaluate_fn(args)
