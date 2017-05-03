# coding: utf-8
import random
import gym
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # noqa
env = gym.make('FrozenLake-v0')


class Agent(object):

    def __init__(self, action_space, eps=0.01, alpha=0.1, gamma=0.9):
        self.action_space = action_space
        self.eps = eps
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = {}

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

    def action(self, obs):
        if random.random() < self.eps:
            return self.action_space.sample()
        return self._get_best_action(obs)

    def update(self, obs, act, reward, next_obs):
        next_act = self._get_best_action(next_obs)
        q = 0
        q += (1.0 - self.alpha) * self.q_function(obs, act)
        q += self.alpha * (
            reward + self.gamma * self.q_function(next_obs, next_act))
        self.q_table[(obs, act)] = q


def run_episode(agent):
    obs = env.reset()
    cnt = 0
    while True:
        cnt += 1
        action = agent.action(obs)
        next_obs, reward, done, info = env.step(action)
        if done:
            break
        agent.update(obs, action, reward, next_obs)
        obs = next_obs
    return cnt


trial = 50
n_episode = 2000
result = []
for i in range(trial):
    print('trial %i start.' % (i))
    agent = Agent(env.action_space)
    for j in range(n_episode):
        cnt = run_episode(agent)
        result.append([j, cnt])

df = pd.DataFrame(result, columns=['episode', 'n_move'])
df.plot.scatter(x='episode', y='n_move')
plt.savefig('result.png')
