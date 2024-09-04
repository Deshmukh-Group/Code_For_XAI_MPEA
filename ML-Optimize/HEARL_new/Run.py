# !/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2021-03-11 17:59:16
LastEditor: Toby
LastEditTime: 2022-04-22 13:49:54
Discription:
Environment:
'''
import os
import sys
import torch

curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)  # add current terminal path to sys.path

import datetime
from agent import Sarsa
from utils import plot_rewards
from utils import save_results, make_dir
from Composition_env import CompositionEnv

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # obtain current time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SarsaConfig:
    ''' parameters for Sarsa
    '''

    def __init__(self):
        self.algo_name = 'Sarsa'
        self.env_name = 'Composition_env'  # 8 actions
        self.result_path = curr_path + "/outputs/" + self.env_name + '/' + curr_time + '/results/'  # path to save results
        self.model_path = curr_path + "/outputs/" + self.env_name + '/' + curr_time + '/models/'  # path to save models
        self.train_eps = 200
        self.test_eps = 50
        self.epsilon = 0.15  # epsilon: The probability to select a random action
        self.gamma = 0.9  # gamma: Gamma discount factor.
        self.lr = 0.2  # learning rate: step size parameter
        self.n_steps = 2000
        self.device = device  # cpu or gpu
        self.save = True


def env_agent_config(cfg=None, start_config=None, goal_config=None, max_steps=None):
    env = CompositionEnv(start_config=start_config, goal_config=goal_config, max_steps=max_steps)
    action_dim = 8
    agent = Sarsa(action_dim, cfg)
    return env, agent


def train(cfg, env, agent):
    rewards = []
    ma_rewards = []
    for i_episode in range(cfg.train_eps):
        # Print out which episode we're on, useful for debugging.
        # Generate an episode.
        # An episode is an array of (state, action, reward) tuples
        state = env.reset()
        ep_reward = 0
        while True:
            # for t in range(cfg.n_steps):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            # print(i_episode, next_state, reward)
            ep_reward += reward
            next_action = agent.choose_action(next_state)
            agent.update(state, action, reward, next_state, next_action, done)
            state = next_state
            if done:
                break
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1] * 0.9 + ep_reward * 0.1)
        else:
            ma_rewards.append(ep_reward)
        rewards.append(ep_reward)
        if (i_episode + 1) % 10 == 0:
            print("Episode:{}/{}: Reward:{}".format(i_episode + 1, cfg.train_eps, ep_reward))
    return rewards, ma_rewards


'''
def eval(cfg, env, agent):
    rewards = []
    ma_rewards = []
    for i_episode in range(cfg.test_eps):
        # Print out which episode we're on, useful for debugging.
        # Generate an episode.
        # An episode is an array of (state, action, reward) tuples
        state = env.reset()
        ep_reward = 0
        while True:
            # for t in range(cfg.n_steps):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            ep_reward += reward
            state = next_state
            if done:
                break
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1] * 0.9 + ep_reward * 0.1)
        else:
            ma_rewards.append(ep_reward)
        rewards.append(ep_reward)
        if (i_episode + 1) % 10 == 0:
            print("Episode:{}/{}: Reward:{}".format(i_episode + 1, cfg.test_eps, ep_reward))
    print('Complete evaling！')
    return rewards, ma_rewards
'''

if __name__ == "__main__":

    # remove output.csv if it exists
    try:
        os.remove("output.csv")
    except OSError:
        pass

    cfg = SarsaConfig()
    start_config = (5.0, 5.0, 31, 31)
    goal_config = 1000
    max_steps = 128

    env, agent = env_agent_config(cfg, start_config=start_config, goal_config=goal_config, max_steps=max_steps)
    rewards, ma_rewards = train(cfg, env, agent)
    make_dir(cfg.result_path, cfg.model_path)
    agent.save(path=cfg.model_path)
    save_results(rewards, ma_rewards, tag='train', path=cfg.result_path)
    plot_rewards(rewards, ma_rewards, tag="train", plot_cfg=cfg)

    # env, agent = env_agent_config(cfg, start_config=start_config, goal_config=goal_config)
    # agent.load(path=cfg.model_path)
    # rewards, ma_rewards = eval(cfg, env, agent)
    # save_results(rewards, ma_rewards, tag='eval', path=cfg.result_path)
    # plot_rewards(rewards, ma_rewards, tag="eval", plot_cfg=cfg)
