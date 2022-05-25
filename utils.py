import tensorflow as tf
from random import sample, choices
import numpy as np
from collections import deque
from gym import Wrapper
import mahotas


def preprocess(im):
    return mahotas.imresize(mahotas.colors.rgb2grey(im), (84, 84)) / 255.0


@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


class ReplayBuffer:
    def __init__(self, maxlen):
        self._queue = deque(maxlen=maxlen)
        self.maxlen = maxlen

    def store(self, state, length, action, reward, next_state, next_length):
        self._queue.append((state, length, action, reward, next_state, next_length))

    def sample(self, n):
        if len(self._queue) >= n:
            minibatch = sample(self._queue, k=n)
        else:
            minibatch = choices(self._queue, k=n)

        states = np.array([transition[0] for transition in minibatch])
        lengths = np.array([transition[1] for transition in minibatch])
        actions = np.array([transition[2] for transition in minibatch])
        rewards = np.array([transition[3] for transition in minibatch])
        next_states = np.array([transition[4] for transition in minibatch])
        next_lengths = np.array([transition[5] for transition in minibatch])

        return states, lengths, actions, rewards, next_states, next_lengths


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class FrameStack(Wrapper):
    def __init__(self, env):
        super(FrameStack, self).__init__(env)
        self._obs_buffer = deque(maxlen=4)

    def reset(self):
        for i in range(3):
            self._obs_buffer.append(np.zeros((84, 84)))
        self._obs_buffer.append(preprocess(self.env.reset()))
        return self.observe()

    def observe(self):
        return np.stack(self._obs_buffer, axis=-1)

    def step(self, action):
        observation, reward, terminal, info = self.env.step(action)
        self._obs_buffer.append(preprocess(observation))
        return self.observe(), reward, terminal, info
