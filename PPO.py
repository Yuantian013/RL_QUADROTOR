import matplotlib
matplotlib.use('TkAgg')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym
from Q_ENV import QUADROTOR as QUADROTOR
from mpl_toolkits.mplot3d import Axes3D
MAX_EPISODES = 20000
MAX_EP_STEPS =2500
GAMMA = 0.99
A_LR = 0.0001
C_LR = 0.0002
BATCH = 128
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
S_DIM, A_DIM = 13, 4

EWMA_p=0.95
EWMA_step=np.zeros((1,MAX_EPISODES+1))
EWMA_reward=np.zeros((1,MAX_EPISODES+1))
iteration=np.zeros((1,MAX_EPISODES+1))

METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective, find this is better
][1]        # choose the method for optimization


class PPO(object):

    def __init__(self,high):
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')
        self.a_high = high
        # critic
        with tf.variable_scope('critic'):
            n_l1 = 256  # 30
            w1_s = tf.get_variable('w1_s', [S_DIM, n_l1])
            b1 = tf.get_variable('b1', [1, n_l1])
            net_0 = tf.nn.relu(tf.matmul(self.tfs, w1_s) + b1)
            net_1 = tf.layers.dense(net_0, 256, activation=tf.nn.relu, name='l2')  # 原始是30
            net_2 = tf.layers.dense(net_1, 128, activation=tf.nn.relu, name='l3')  # 原始是30
            self.v=tf.layers.dense(net_2, 1)
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.advantage = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # actor
        pi, pi_params,self.mu,self.sigma = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params,_,_ = self._build_anet('oldpi', trainable=False)
        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)       # choosing action
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
                ratio = pi.prob(self.tfa) / oldpi.prob(self.tfa)
                surr = ratio * self.tfadv
            if METHOD['name'] == 'kl_pen':
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(oldpi, pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))
            else:   # clipping method, find this is better
                self.aloss = -tf.reduce_mean(tf.minimum(
                    surr,
                    tf.clip_by_value(ratio, 1.-METHOD['epsilon'], 1.+METHOD['epsilon'])*self.tfadv))

        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)

        tf.summary.FileWriter("log/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def update(self, s, a, r):
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
        # adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful

        # update actor
        if METHOD['name'] == 'kl_pen':
            for _ in range(A_UPDATE_STEPS):
                _, kl = self.sess.run(
                    [self.atrain_op, self.kl_mean],
                    {self.tfs: s, self.tfa: a, self.tfadv: adv, self.tflam: METHOD['lam']})
                if kl > 4*METHOD['kl_target']:  # this in in google's paper
                    break
            if kl < METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                METHOD['lam'] /= 2
            elif kl > METHOD['kl_target'] * 1.5:
                METHOD['lam'] *= 2
            METHOD['lam'] = np.clip(METHOD['lam'], 1e-4, 10)    # sometimes explode, this clipping is my solution
        else:   # clipping method, find this is better (OpenAI's paper)
            [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(A_UPDATE_STEPS)]

        # update critic
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(C_UPDATE_STEPS)]

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            net_0 = tf.layers.dense(self.tfs, 256, activation=tf.nn.relu, name='l1', trainable=trainable)  # 原始是30
            net_1 = tf.layers.dense(net_0, 256, activation=tf.nn.relu, name='l2', trainable=trainable)  # 原始是30
            net_2 = tf.layers.dense(net_1, 256, activation=tf.nn.relu, name='l3', trainable=trainable)  # 原始是30
            net_3 = tf.layers.dense(net_2, 128, activation=tf.nn.relu, name='l4', trainable=trainable)  # 原始是30

            mu = tf.layers.dense(net_3, A_DIM, activation=tf.nn.tanh, name='a', trainable=trainable)
            # mu = tf.multiply(mu,self.a_high, name='scaled_a')

            sigma = tf.layers.dense(net_3, A_DIM, tf.nn.softplus, trainable=trainable)
            # sigma = tf.multiply(sigma, 0, name='scaled_sigma')
            norm_dist= tf.distributions.Normal(loc=mu, scale=sigma)

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params,mu,mu

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        # print(self.sess.run(self.mu, {self.tfs: s})[0])
        return a#np.clip(a, a_low, a_high)

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]

# ENV_NAME = 'CartPole-v2'
env = QUADROTOR()
# env = gym.make(ENV_NAME)
env = env.unwrapped
a_high = env.force_space.high
a_low = env.force_space.low
ppo = PPO(a_high)
all_ep_r = []
fig = plt.figure()
ax = plt.subplot(projection='3d')  # 创建一个三维的绘图工程
plot_x = []
plot_y = []
plot_z = []
fig = plt.figure()
var=0.5*a_high
for i in range(MAX_EPISODES):
    ax = plt.subplot(projection='3d')  # 创建一个三维的绘图工程
    ax.scatter(plot_x, plot_y, plot_z, c='r')  # 绘制数据点,颜色是红色
    ax.set_zlabel('Z')  # 坐标轴
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.draw()
    plt.pause(0.1)
    plt.close(fig)
    plot_x = []
    plot_y = []
    plot_z = []
    # des_start = env.trajectory('circle', 0)
    # s = env.reset_traj(des_start[0], 0)
    s=env.reset()
    buffer_s, buffer_a, buffer_r = [], [], []
    ep_reward = 0
    for j in range(MAX_EP_STEPS):    # in one episode
        # print(s)
        # if i>10:
        print(s)
        a = ppo.choose_action(s)

        print(a)
        a = np.clip(a, a_low, a_high)
        a[0]=a[0]+np.random.normal(a[0], a_high[0])
        a = np.clip(np.random.normal(a, var), a_low, a_high)
        # print(a)

        # if i<2:
        #     desired_state = env.trajectory('circle', t)
        #     F, M = env.controller(desired_state, env.state)
        #     a = [F, M[0], M[1], M[2]]
        s_, r, done= env.ppo_step(a[0],a[1:])
        buffer_s.append(s)
        buffer_a.append(a)
        buffer_r.append(r/10)    # normalize reward, find to be useful
        s = s_
        ep_reward += r
        plot_x.append(s_[0])
        plot_y.append(s_[1])
        plot_z.append(s_[2])
        # update ppo
        if (j+1) % BATCH == 0 or j == MAX_EP_STEPS-1:
            v_s_ = ppo.get_v(s_)
            discounted_r = []
            for r in buffer_r[::-1]:
                v_s_ = r + GAMMA * v_s_
                discounted_r.append(v_s_)
            discounted_r.reverse()

            bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
            buffer_s, buffer_a, buffer_r = [], [], []
            ppo.update(bs, ba, br)
            # print("train")

        if j == MAX_EP_STEPS - 1:
            EWMA_step[0,i+1]=EWMA_p*EWMA_step[0,i]+(1-EWMA_p)*j
            EWMA_reward[0,i+1]=EWMA_p*EWMA_reward[0,i]+(1-EWMA_p)*ep_reward
            # print('Episode:', i, 'step', j, ' Reward: %i' % int(ep_reward), "EWMA_step = ", EWMA_step[0, i + 1],
            #       "EWMA_reward = ", EWMA_reward[0, i + 1], s_[0], s_[1], s_[2])
            break
        #
        elif done:
            EWMA_step[0,i+1]=EWMA_p*EWMA_step[0,i]+(1-EWMA_p)*j
            EWMA_reward[0,i+1]=EWMA_p*EWMA_reward[0,i]+(1-EWMA_p)*ep_reward
            #
            # print('Episode:', i, 'step', j, ' Reward: %i' % int(ep_reward), "EWMA_step = ", EWMA_step[0, i + 1],
            #           "EWMA_reward = ", EWMA_reward[0, i + 1], s_[0], s_[1], s_[2])
            break

    if i == 0: all_ep_r.append(ep_reward)
    else: all_ep_r.append(all_ep_r[-1]*0.9 + ep_reward*0.1)
    print('Episode:', i,'Stop at :',s_[0], s_[1], s_[2],
          'step',j,
        'Ep: %i' % i,
        "|Ep_r: %i" % ep_reward,
        ("|Lam: %.4f" % METHOD['lam']) if METHOD['name'] == 'kl_pen' else '',
    )

plt.plot(np.arange(len(all_ep_r)), all_ep_r)
plt.xlabel('Episode');plt.ylabel('Moving averaged episode reward');plt.show()