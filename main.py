# coding: utf-8
import matplotlib
matplotlib.use('Agg')
from rl.agent import QLearningAgent, SarsaAgent, TDLambda
import gym
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # noqa
env = gym.make('FrozenLake-v0')


def run_episode(agent):
    agent.reset()
    obs = env.reset()
    cnt = 0
    while True:
        cnt += 1
        action = agent.action(obs)
        next_obs, reward, done, info = env.step(action)
        agent.store(obs, action, reward, next_obs)
        agent.update()
        if done:
            break
        obs = next_obs
    return cnt


if __name__ == '__main__':
    trial = 50
    n_episode = 2000
    result = []
    for i in range(trial):
        print('trial %i start.' % (i))
        agent = TDLambda(env.action_space)
        for j in range(n_episode):
            cnt = run_episode(agent)
            result.append([j, cnt])

    df = pd.DataFrame(result, columns=['episode', 'n_move'])
    df.plot.scatter(x='episode', y='n_move')
    plt.savefig('result.png')
