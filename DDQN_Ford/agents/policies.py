import numpy as np
import tensorflow as tf
from .utils import *


class QPolicy:
    def __init__(self, n_a, n_s, n_step, policy_name, agent_name):
        self.name = policy_name
        if agent_name is not None:
            # for multi-agent system
            self.name += '_' + str(agent_name)
        self.n_a = n_a
        self.n_s = n_s
        self.n_step = n_step

    def forward(self, ob, *_args, **_kwargs):
        raise NotImplementedError()

    def _build_fc_net(self, h, n_fc_ls):
        for i, n_fc in enumerate(n_fc_ls):
            h = fc(h, 'q_fc_%d' % i, n_fc)
        q = fc(h, 'q', self.n_a, act=lambda x: x)
        return tf.squeeze(q)

    def _build_net(self):
        raise NotImplementedError()

    def prepare_loss(self, max_grad_norm, gamma):
        self.A = tf.placeholder(tf.int32, [self.n_step])
        self.S1 = tf.placeholder(
            tf.float32, [self.n_step, self.n_s])
        self.R = tf.placeholder(tf.float32, [self.n_step])
        self.DONE = tf.placeholder(tf.bool, [self.n_step])
        A_sparse = tf.one_hot(self.A, self.n_a)

        # backward, calculate loss
        with tf.variable_scope(self.name + '_q', reuse=True):
            q0s = self._build_net(self.S)
            q0 = tf.reduce_sum(q0s * A_sparse, axis=1)
        with tf.variable_scope(self.name + '_q', reuse=True):
            q1s = self._build_net(self.S1)
            q1 = tf.reduce_max(q1s, axis=1)
        tq = tf.stop_gradient(tf.where(self.DONE, self.R, self.R + gamma * q1))
        self.loss = tf.reduce_mean(tf.square(q0 - tq))

        wts = tf.trainable_variables(scope=self.name)
        grads = tf.gradients(self.loss, wts)
        if max_grad_norm > 0:
            grads, self.grad_norm = tf.clip_by_global_norm(
                grads, max_grad_norm)
        self.lr = tf.placeholder(tf.float32, [])
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self._train = self.optimizer.apply_gradients(list(zip(grads, wts)))
        # monitor training
        if self.name.endswith('_0a'):
            summaries = []
            summaries.append(tf.summary.scalar(
                'train/%s_loss' % self.name, self.loss))
            summaries.append(tf.summary.scalar('train/%s_q' %
                                               self.name, tf.reduce_mean(q0)))
            summaries.append(tf.summary.scalar('train/%s_tq' %
                                               self.name, tf.reduce_mean(tq)))
            summaries.append(tf.summary.scalar(
                'train/%s_gradnorm' % self.name, self.grad_norm))
            self.summary = tf.summary.merge(summaries)


class DeepQPolicy(QPolicy):
    def __init__(self, n_s, n_a, n_step, n_fc0=128, n_fc=64, name=None):
        super().__init__(n_a, n_s, n_step, 'dqn', name)
        self.n_fc = n_fc
        self.n_fc0 = n_fc0
        self.S = tf.placeholder(tf.float32, [None, n_s])
        with tf.variable_scope(self.name + '_q'):
            self.qvalues = self._build_net(self.S)

    def _build_net(self, S):
        h0 = fc(S[:, :self.n_s], 'q_fcw', self.n_fc0)
        h1 = fc(S[:, self.n_s:], 'q_fct', self.n_fc0 / 4)
        h = tf.concat([h0, h1], 1)
        return self._build_fc_net(h, [self.n_fc])

    def forward(self, sess, ob):
        return sess.run(self.qvalues, {self.S: np.array([ob])})

    def backward(self, sess, obs, acts, next_obs, dones, rs, cur_lr,
                 summary_writer=None, global_step=None):
        # update networks
        if summary_writer is None:
            ops = self._train
        else:
            ops = [self.summary, self._train]
        outs = sess.run(ops,
                        {self.S: obs,
                         self.A: acts,
                         self.S1: next_obs,
                         self.DONE: dones,
                         self.R: rs,
                         self.lr: cur_lr})
        if summary_writer is not None:
            summary_writer.add_summary(outs[0], global_step=global_step)
