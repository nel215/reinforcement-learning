# coding: utf-8
import gym
import numpy as np
import chainer.functions as F
import chainer.links as L
import matplotlib.pyplot as plt
from chainer import Chain, optimizers


class Policy(Chain):

    def __init__(self, n_action, n_obs):
        super(Policy, self).__init__(
            l=L.Linear(n_obs, n_action),
        )

    def forward(self, obs):
        h = self.l(obs)
        return F.softmax(h)

    def __call__(self, actions, observations, rewards):
        n = len(actions)
        actions = np.array(actions).astype(np.int32)
        observations = np.array(observations).astype(np.float32)
        rewards = np.array(rewards).astype(np.float32)
        baseline = np.mean(rewards)
        action_prob = self.forward(observations)
        actual_action_prob = F.select_item(action_prob, actions)

        loss = -F.sum(F.log(actual_action_prob) * (rewards - baseline)) / n
        return loss


class ReinforceAgent(object):

    def __init__(self, n_action, n_obs):
        self.n_action = n_action
        self.actions = []
        self.observations = []
        self.rewards = []
        self.policy = Policy(n_action, n_obs)
        self.optimizer = optimizers.Adam(alpha=0.01)
        self.optimizer.use_cleargrads()
        self.optimizer.setup(self.policy)

    def action(self, obs):
        '''
        obs: [Observation]
        '''
        action_prob = self.policy.forward(
            np.reshape(obs, (1, -1)).astype(np.float32)).data[0]
        next_action = np.random.choice(self.n_action, p=action_prob)
        return next_action

    def store_reward(self, obs, act, reward):
        self.observations.append(obs)
        self.actions.append(act)
        self.rewards.append(reward)

    def update(self):
        self.optimizer.update(self.policy, self.actions, self.observations, self.rewards)


env = gym.make('CartPole-v0')
n_action = env.action_space.n
n_obs = env.observation_space.shape[0]
agent = ReinforceAgent(n_action, n_obs)
rewards = []
for j in range(1000):
    obs = env.reset()
    acc_reward = 0
    while True:
        act = agent.action(obs)
        next_obs, reward, done, info = env.step(act)
        acc_reward += reward
        agent.store_reward(obs, act, reward)
        obs = next_obs
        if done:
            if j % 10 == 0:
                print(acc_reward)
                rewards.append(acc_reward)
            agent.update()
            break


plt.plot(rewards)
plt.savefig('./result.png')
