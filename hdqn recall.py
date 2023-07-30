import numpy as np
from collections import defaultdict
from itertools import count
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from agents.hdqn_mdp import hDQN
from utils.replay_memory import ReplayMemory
from utils import plotting

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)


def one_hot_state(state):
    vector = np.zeros(6)
    vector[state - 1] = 1.0
    return np.expand_dims(vector, axis=0)


def one_hot_goal(goal):
    vector = np.zeros(6)
    vector[goal - 1] = 1.0
    return np.expand_dims(vector, axis=0)


def hdqn_learning(
    env,
    agent:hDQN,
    num_episodes,
    exploration_schedule,
    gamma=1.0,
):
    """The h-DQN learning algorithm.
    All schedules are w.r.t. total number of steps taken in the environment.
    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    agent:
        a h-DQN agent consists of a meta-controller and controller.
    num_episodes:
        Number (can be divided by 1000) of episodes to run for. Ex: 12000
    exploration_schedule: Schedule (defined in utils.schedule)
        schedule for probability of chosing random action.
    gamma: float
        Discount Factor
    """
    ###############
    # RUN ENV     #
    ###############
    # Keep track of useful statistics
    stats = plotting.EpisodeStats(np.zeros(num_episodes),
                                  np.zeros(num_episodes))
    num_thousand_episodes = int(np.floor(num_episodes/1000))
    visits = np.zeros((num_thousand_episodes, env.n))
    total_timestep = 0
    meta_timestep = 0
    ctrl_timestep = 0

    for i_thousand_episode in range(num_thousand_episodes):
        for i_episode in range(num_episodes):
            current_state = env.reset()
            encoded_current_state = one_hot_state()
            meta_epsilon = exploration_schedule.value(total_timestep)
            goal = agent.select_goal(encoded_current_state, meta_epsilon)
            encoded_goal = one_hot_goal(goal)
            episode_length = 0


            done = False
            while not done:
                extrinsic_reward = 0
                goal_reached = False
                while not done and not goal_reached:
                    total_timestep += 1
                    episode_length += 1
                    joint_state_goal = np.concatenate([encoded_current_state, encoded_goal], axis=1)
                    ctrl_epsilon = exploration_schedule.value(total_timestep)
                    action = agent.select_action(joint_state_goal, ctrl_epsilon)

                    next_state, reward, done, _ =env.step(action)
                    extrinsic_reward += reward

                    visits[i_thousand_episode][next_state-1] += 1
                    stats.episode_lengths[i_thousand_episode*1000 + i_episode] += 1
                    stats.episode_rewards[i_thousand_episode*1000 + i_episode] = episode_length

                    encoded_next_state = one_hot_state(next_state)
                    joint_next_state_goal = np.concatenate([encoded_next_state, encoded_current_state])

                    agent.ctrl_replay_memory.push(joint_state_goal, action, reward, joint_next_state_goal, done)

                    agent.update_meta_controller()
                    agent.update_controller()

                    current_state = next_state
                    encoded_current_state = encoded_next_state

                agent.meta_replay_memory.push(encoded_current_state, encoded_goal, extrinsic_reward, encoded_next_state)
                if not done:
                    meta_epsilon = exploration_schedule.value(total_timestep)
                    goal = agent.select_goal(encoded_current_state, meta_epsilon)

