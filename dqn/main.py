# coding: utf-8
import random
import gym
import numpy as np
import chainer.functions as F
import chainer.links as L
import matplotlib.pyplot as plt
import chainer
import cupy
from collections import deque
from chainer import Chain, optimizers
from chainer import cuda

gpu = True

if gpu:
    xp = cupy
else:
    xp = np



class QFunction(Chain):

    def __init__(self, n_action, n_obs):
        super(QFunction, self).__init__(
            l1=L.Linear(n_obs, 100),
            l2=L.Linear(100, n_action),
        )

    def forward(self, obs, train=True):
        h1 = F.relu(self.l1(obs))
        h2 = self.l2(h1)
        return h2

    def __call__(self, states, actions, y):
        q_value = F.select_item(self.forward(states), actions)

        loss = F.mean_squared_error(y, q_value)
        return loss


class ReplayMemory(object):

    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.memory = deque()

    def store(self, state, action, reward, next_state, done):
        '''
        Parameters:
            state: [ndarray[np.float]]
            action: [int]
            reward: [float]
            next_state: [ndarray[np.float]]
            done: [bool]
        '''
        self.memory.append([state, action, reward, next_state, done])
        if len(self.memory) <= self.capacity:
            return
        self.memory.popleft()

    def sample(self, size=512):
        '''
        Returns:
            states: [ndarray[np.float32]]
            actions: [ndarray[np.integer]]
            rewards: [ndarray[np.float32]]
            next_states: [ndarray[np.float32]]
            dones: [ndarray[np.bool]]
        '''
        size = min(size, len(self.memory))
        states, actions, rewards, next_states, dones = map(
            np.array, zip(*random.sample(self.memory, size)))
        states = states.astype(np.float32)
        rewards = rewards.astype(np.float32)
        next_states = next_states.astype(np.float32)
        return states, actions, rewards, next_states, dones


class DQNAgent(object):

    def __init__(self, action_space, q_function, gamma=0.9, eps=0.05, gpu=False):
        self.action_space = action_space
        self.q_function = q_function
        self.replay_memory = ReplayMemory()
        self.gamma = gamma
        self.eps = eps
        self.optimizer = optimizers.Adam()
        self.optimizer.use_cleargrads()
        self.optimizer.setup(self.q_function)
        self.gpu = gpu

    def action(self, obs):
        '''
        obs: [Observation]
        '''
        if random.random() < self.eps:
            return self.action_space.sample()
        obs = xp.array([obs], dtype=xp.float32)
        q_value = self.q_function.forward(obs, train=False)
        act = F.argmax(q_value)
        return int(act.data)

    def store_experience_replay(self, state, action, reward, next_state, done):
        self.replay_memory.store(state, action, reward, next_state, done)

    def update(self):
        states, actions, rewards, next_states, dones = self.replay_memory.sample()
        if self.gpu:
            actions = cuda.to_gpu(actions)
            states = cuda.to_gpu(states)
            rewards = cuda.to_gpu(rewards)
            next_states = cuda.to_gpu(next_states)
            dones = cuda.to_gpu(dones)

        q_value = self.q_function.forward(next_states, train=False)
        max_q_value = F.max(q_value, axis=1)
        terminal = xp.ones(dones.shape, dtype=xp.float32)
        terminal[dones] = 0
        y = rewards + self.gamma * terminal * max_q_value.data

        self.optimizer.zero_grads()
        loss = self.q_function(states, actions, y)
        loss.backward()
        self.optimizer.update()


env = gym.make('CartPole-v0')
n_action = env.action_space.n
n_obs = env.observation_space.shape[0]
q_function = QFunction(n_action, n_obs)
if gpu:
    q_function.to_gpu(0)
agent = DQNAgent(env.action_space, q_function, gpu=gpu)
for j in range(10000):
    state = env.reset()
    sum_reward = 0
    while True:
        act = agent.action(state)
        next_state , reward, done, info = env.step(act)
        agent.store_experience_replay(state, act, reward, next_state, done)
        agent.update()
        sum_reward += reward
        state = next_state
        if done:
            if j % 10 == 0:
                print('reward: {}'.format(sum_reward))
            break
