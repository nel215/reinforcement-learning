# coding: utf-8
import random
import gym
import numpy as np
import chainer.functions as F
import chainer.links as L
import matplotlib.pyplot as plt
import chainer
from chainer import Chain, optimizers


class QFunction(Chain):

    def __init__(self, n_action, n_obs):
        super(QFunction, self).__init__(
            l=L.Linear(n_obs, n_action),
        )

    def forward(self, obs, train=True):
        h = F.dropout(self.l(obs), train=train)
        return F.softmax(h)

    def __call__(self, states, actions, y):
        q_value = F.select_item(self.forward(states), actions)

        loss = F.sum(F.squared_error(y, q_value))
        return loss


class Transitions(object):

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def _shift(self):
        if len(self.states) <= 100000:
            return
        self.states = self.states[1:]
        self.actions = self.actions[1:]
        self.rewards = self.rewards[1:]
        self.next_states = self.next_states[1:]
        self.dones = self.dones[1:]

    def store(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self._shift()

    def sample(self, size=32):
        n = len(self.states)
        size = min(n, size)
        perm = np.random.permutation(n)[:size]
        return (
            np.array(self.states)[perm],
            np.array(self.actions)[perm],
            np.array(self.rewards)[perm],
            np.array(self.next_states)[perm],
            np.array(self.dones)[perm],
        )


class DQNAgent(object):

    def __init__(self, action_space, q_function, gamma=0.9, eps=0.05):
        self.action_space = action_space
        self.q_function = q_function
        self.transitions = Transitions()
        self.gamma = gamma
        self.eps = eps
        self.optimizer = optimizers.Adam()
        self.optimizer.use_cleargrads()
        self.optimizer.setup(self.q_function)

    def action(self, obs):
        '''
        obs: [Observation]
        '''
        if random.random() < self.eps:
            return self.action_space.sample()
        obs = np.array([obs], dtype=np.float32)
        q_value = self.q_function.forward(obs, train=False)
        act = F.argmax(q_value)
        return act.data

    def store_transition(self, state, action, reward, next_state, done):
        self.transitions.store(state, action, reward, next_state, done)

    def update(self):
        states, actions, rewards, next_states, dones = self.transitions.sample()
        max_q_value = F.max(
            self.q_function.forward(next_states.astype(np.float32), train=False), axis=1)
        terminal = np.ones(dones.shape, dtype=np.float32)
        terminal[dones] = 0
        y = rewards + self.gamma * terminal * max_q_value.data
        y = y.astype(np.float32)

        states = states.astype(np.float32)
        actions = actions.astype(np.int32)
        self.optimizer.update(self.q_function, states, actions, y)


env = gym.make('CartPole-v0')
n_action = env.action_space.n
n_obs = env.observation_space.shape[0]
q_function = QFunction(n_action, n_obs)
agent = DQNAgent(env.action_space, q_function)
for j in range(1000):
    state = env.reset()
    sum_reward = 0
    while True:
        act = agent.action(state)
        next_state , reward, done, info = env.step(act)
        agent.store_transition(state, act, reward, next_state, done)
        agent.update()
        sum_reward += reward
        state = next_state
        if done:
            print('reward: {}'.format(sum_reward))
            break