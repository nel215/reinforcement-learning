# coding: utf-8


class TDLambda(object):

    def __init__(self, action_space, alpha=0.1, gamma=0.9, ld=0.5):
        self.action_space = action_space
        self.V = {}
        self.E = {}  # eligibility trace
        self.alpha = alpha
        self.gamma = gamma
        self.ld = ld
        self.obs = None
        self.reward = None
        self.next_obs = None

    def v_function(self, obs):
        if obs not in self.V:
            self.V[obs] = 10
        return self.V[obs]

    def action(self, obs):
        # random policy
        return self.action_space.sample()

    def store(self, obs, act, reward, next_obs):
        self.obs = obs
        self.reward = reward
        self.next_obs = next_obs

    def update(self):
        et = 1 + self.gamma * self.ld * self.E.get(self.obs, 1)
        self.E[self.obs] = et
        loss = self.reward + self.gamma * self.v_function(self.next_obs) - self.v_function(self.obs)
        self.V[self.obs] += self.alpha * loss * et

    def reset(self):
        self.E = {}
