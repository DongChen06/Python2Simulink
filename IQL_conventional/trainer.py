import logging
import numpy as np
import tensorflow as tf
import pandas as pd
import time


class Trainer():
    def __init__(self, env, model, global_counter, summary_writer, run_test, output_path=None, rendering=False):
        self.cur_step = 0
        self.rendering = rendering
        self.global_counter = global_counter
        self.env = env
        self.agent = 'iql'  # TODO
        self.model = model
        self.sess = self.model.sess
        self.n_step = self.model.n_step  # bacth size
        self.summary_writer = summary_writer
        self.run_test = run_test  # ToDo 
        # assert self.env.T % self.n_step == 0
        self.data = []
        self.output_path = output_path
        if run_test:
            self.test_num = self.env.test_num
            logging.info('Testing: total test num: %d' % self.test_num)
        self._init_summary()

    def _init_summary(self):
        self.train_reward = tf.placeholder(tf.float32, [])
        self.train_summary = tf.summary.scalar(
            'train_reward', self.train_reward)
        self.test_reward = tf.placeholder(tf.float32, [])
        self.test_summary = tf.summary.scalar('test_reward', self.test_reward)

    def _add_summary(self, reward, global_step, is_train=True):
        if is_train:
            summ = self.sess.run(self.train_summary, {
                                 self.train_reward: reward})
        else:
            summ = self.sess.run(self.test_summary, {self.test_reward: reward})
        self.summary_writer.add_summary(summ, global_step=global_step)

    def take_action(self, prev_ob, prev_done):
        #  take actions for a batch size
        ob = prev_ob
        done = prev_done
        rewards = 0  # ori = []
        for _ in range(self.n_step):
            if self.agent.endswith('a2c'):
                policy, value = self.model.forward(ob, done)
                action = []
                for pi in policy:
                    action.append(np.random.choice(np.arange(len(pi)), p=pi))
            else:
                action, policy = self.model.forward(ob, mode='explore')
            next_ob, reward, done, _ = self.env.step(action[0])  # ori = action, global_reward
            if self.rendering:
                self.env.render()
            rewards += reward
            global_step = self.global_counter.next()
            self.cur_step += 1
            self.model.add_transition(ob, action, reward, next_ob, done)
            if done:
                break
            ob = next_ob
        return ob, done, _, rewards

    def evaluate(self, test_ind, demo=False, policy_type='default'):
        # test function
        ob = self.env.reset(gui=demo, test_ind=test_ind)
        # note this done is pre-decision to reset LSTM states!
        done = True
        self.model.reset()
        rewards = []
        while True:
            if self.agent == 'greedy':
                action = self.model.forward(ob)
            elif self.agent.endswith('a2c'):
                # policy-based on-poicy learning
                policy = self.model.forward(ob, done, 'p')
                if self.agent == 'ma2c':
                    self.env.update_fingerprint(policy)
                if self.agent == 'a2c':
                    if policy_type != 'deterministic':
                        action = np.random.choice(
                            np.arange(len(policy)), p=policy)
                    else:
                        action = np.argmax(np.array(policy))
                else:
                    action = []
                    for pi in policy:
                        if policy_type != 'deterministic':
                            action.append(np.random.choice(
                                np.arange(len(pi)), p=pi))
                        else:
                            action.append(np.argmax(np.array(pi)))
            else:
                # value-based off-policy learning
                if policy_type != 'stochastic':
                    action, _ = self.model.forward(ob)
                else:
                    action, _ = self.model.forward(ob, stochastic=True)
            next_ob, reward, done, global_reward = self.env.step(action)
            rewards.append(global_reward)
            if done:
                break
            ob = next_ob
        mean_reward = np.mean(np.array(rewards))
        std_reward = np.std(np.array(rewards))
        return mean_reward, std_reward

    def run(self):
        while not self.global_counter.should_stop():
            # test or not
            if self.run_test and self.global_counter.should_test():
                rewards = []
                global_step = self.global_counter.cur_step
                for test_ind in range(self.test_num):
                    mean_reward, std_reward = self.evaluate(test_ind)
                    self.env.terminate()
                    rewards.append(mean_reward)
                    log = {'agent': self.agent,
                           'step': global_step,
                           'test_id': test_ind,
                           'avg_reward': mean_reward,
                           'std_reward': std_reward}
                    self.data.append(log)
                avg_reward = np.mean(np.array(rewards))
                self._add_summary(avg_reward, global_step, is_train=False)
                logging.info('Testing: global step %d, avg R: %.2f' %
                             (global_step, avg_reward))

            # train
            ob = self.env.reset()
            done = True
            self.model.reset()
            self.cur_step = 0
            rewards = []
            while True:
                ob, done, _, cur_rewards = self.take_action(ob, done)
                rewards.append(cur_rewards)  # ori
                global_step = self.global_counter.cur_step
                # update network for each bach size steps
                self.model.backward(self.summary_writer, global_step)
                # termination
                if done:
                    break
            rewards = np.array(rewards)  # reward for one epoch
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            log = {'agent': self.agent,
                   'step': global_step,
                   'test_id': -1,
                   'avg_reward': mean_reward,
                   'std_reward': std_reward}
            self.data.append(log)
            self._add_summary(mean_reward, global_step)
            self.summary_writer.flush()
        df = pd.DataFrame(self.data) # data: dictionary
        df.to_csv(self.output_path + 'train_reward.csv')


class Tester(Trainer):
    def __init__(self, env, model, global_counter, summary_writer, output_path):
        super().__init__(env, model, global_counter, summary_writer)
        self.env.train_mode = False
        self.test_num = self.env.test_num
        self.output_path = output_path
        self.data = []
        logging.info('Testing: total test num: %d' % self.test_num)

    def _init_summary(self):
        self.reward = tf.placeholder(tf.float32, [])
        self.summary = tf.summary.scalar('test_reward', self.reward)

    def run_offline(self):
        # enable traffic measurments for offline test
        is_record = True
        record_stats = False
        self.env.cur_episode = 0
        self.env.init_data(is_record, record_stats, self.output_path)
        rewards = []
        for test_ind in range(self.test_num):
            rewards.append(self.evaluate(test_ind))
            self.env.terminate()
            time.sleep(2)
            self.env.collect_tripinfo()
        avg_reward = np.mean(np.array(rewards))
        logging.info('Offline testing: avg R: %.2f' % avg_reward)
        self.env.output_data()

    def run_online(self, coord):
        self.env.cur_episode = 0
        while not coord.should_stop():
            time.sleep(30)
            if self.global_counter.should_test():
                rewards = []
                global_step = self.global_counter.cur_step
                for test_ind in range(self.test_num):
                    cur_reward = self.evaluate(test_ind)
                    self.env.terminate()
                    rewards.append(cur_reward)
                    log = {'agent': self.agent,
                           'step': global_step,
                           'test_id': test_ind,
                           'reward': cur_reward}
                    self.data.append(log)
                avg_reward = np.mean(np.array(rewards))
                self._add_summary(avg_reward, global_step)
                logging.info('Testing: global step %d, avg R: %.2f' %
                             (global_step, avg_reward))
                # self.global_counter.update_test(avg_reward)
        df = pd.DataFrame(self.data)
        df.to_csv(self.output_path + 'train_reward.csv')


class Evaluator(Tester):
    def __init__(self, env, model, output_path, demo=False, policy_type='default'):
        self.env = env
        self.model = model
        self.agent = self.env.agent
        self.env.train_mode = False
        self.test_num = self.env.test_num
        self.output_path = output_path
        self.demo = demo
        self.policy_type = policy_type

    def run(self):
        is_record = True
        record_stats = False
        self.env.cur_episode = 0
        self.env.init_data(is_record, record_stats, self.output_path)
        time.sleep(1)
        for test_ind in range(self.test_num):
            reward, _ = self.evaluate(
                test_ind, demo=self.demo, policy_type=self.policy_type)
            self.env.terminate()
            logging.info('test %i, avg reward %.2f' % (test_ind, reward))
            time.sleep(2)
            self.env.collect_tripinfo()
        self.env.output_data()