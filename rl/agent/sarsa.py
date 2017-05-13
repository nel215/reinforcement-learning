# coding: utf-8
import random
import gym
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # noqa
env = gym.make('FrozenLake-v0')


class SarsaAgent(object):

    def __init__(self, action_space, eps=0.01, alpha=0.1, gamma=0.9):
        self.action_space = action_space
        self.eps = eps
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = {}
        self.prev_obs = None
        self.obs = None
        self.reward = None
        self.act = None

    def q_function(self, obs, act):
        key = (obs, act)
        if key not in self.q_table:
            self.q_table[key] = 100
        return self.q_table[key]

    def get_action_list(self):
        return range(self.action_space.n)

    def _get_best_action(self, obs):
        best = (-1e9, None)
        for act in self.get_action_list():
            score = self.q_function(obs, act)
            if best[0] < score:
                best = (score, act)
        return best[1]

    def store(self, obs, act, reward, next_obs):
        self.prev_obs = self.obs
        self.obs = obs
        self.act = act
        self.reward = reward

    def action(self, obs):
        if random.random() < self.eps:
            return self.action_space.sample()
        return self._get_best_action(obs)

    def update(self):
        if self.prev_obs is None:
            return
        q = 0
        q += (1.0 - self.alpha) * self.q_function(self.prev_obs, self.act)
        q += self.alpha * (
            self.reward + self.gamma * self.q_function(self.obs, self.act))
        self.q_table[(self.obs, self.act)] = q


def run_episode(agent):
    prev_obs, prev_action = None, None
    obs = env.reset()
    cnt = 0
    while True:
        cnt += 1
        action = agent.action(obs)
        if prev_obs is not None and prev_action is not None:
            agent.update(prev_obs, prev_action, reward, obs, action)
        prev_obs, prev_action = obs, action
        obs, reward, done, info = env.step(action)
        if done:
            break
    return cnt
