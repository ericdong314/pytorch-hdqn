import numpy as np
from collections import defaultdict
from itertools import count
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

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
    return np.expand_dims(state, axis=0)

def one_hot_goal(goal):
    num_rows, num_cols = 4, 4
    goal = np.zeros(num_rows * num_cols, dtype=np.int8)  ### empty board
    return np.expand_dims(goal, axis=0)

def hdqn_learning(
    env,
    agent,
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
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))
    n_thousand_episode = int(np.floor(num_episodes / 1000))
    # visits = np.zeros((n_thousand_episode, env.nS))
    total_timestep = 0
    meta_timestep = 0
    # ctrl_timestep = defaultdict(int)

    for i_thousand_episode in range(n_thousand_episode):
        for i_episode in range(1000):
            episode_length = 0
            current_state, _ = env.reset()
            # visits[i_thousand_episode][current_state-1] += 1
            encoded_current_state = one_hot_state(current_state)

            done = False
            while not done:
                meta_timestep += 1
                # Get annealing exploration rate (epislon) from exploration_schedule
                meta_epsilon = exploration_schedule.value(total_timestep)
                goal = agent.select_goal(encoded_current_state, meta_epsilon)   ### only 4 4 are used
                encoded_goal = one_hot_goal(goal)

                total_extrinsic_reward = 0
                goal_reached = False
                while not done and not goal_reached:
                    total_timestep += 1
                    episode_length += 1
                    # ctrl_timestep[goal] += 1
                    # Get annealing exploration rate (epislon) from exploration_schedule
                    ctrl_epsilon = exploration_schedule.value(total_timestep)
                    joint_state_goal = np.concatenate([encoded_current_state, encoded_goal], axis=1)
                    action = agent.select_action(joint_state_goal, ctrl_epsilon)[0]
                    ### Step the env and store the transition
                    next_state, extrinsic_reward, done, _, _ = env.step(action)
                    # Update statistics
                    stats.episode_rewards[i_thousand_episode*1000 + i_episode] += extrinsic_reward
                    stats.episode_lengths[i_thousand_episode*1000 + i_episode] = episode_length
                    # visits[i_thousand_episode][next_state-1] += 1

                    encoded_next_state = one_hot_state(next_state)
                    intrinsic_reward = agent.get_intrinsic_reward(goal, next_state)
                    if intrinsic_reward:
                        print(f'intrinsic_reward: {intrinsic_reward}')
                    # goal_reached = agent.check_goal_lines(goal, extrinsic_reward)
                    goal_reached = agent.check_goal(goal, current_state)

                    joint_next_state_goal = np.concatenate([encoded_next_state, encoded_goal], axis=1)
                    agent.ctrl_replay_memory.push(joint_state_goal, action, joint_next_state_goal, intrinsic_reward, done)
                    # Update Both meta-controller and controller
                    agent.update_meta_controller(gamma)
                    agent.update_controller(gamma)

                    total_extrinsic_reward += extrinsic_reward
                    current_state = next_state
                    encoded_current_state = encoded_next_state
                # Goal Finished
                agent.meta_replay_memory.push(encoded_current_state, goal, encoded_next_state, total_extrinsic_reward, done)
            episode = i_thousand_episode*1000 + i_episode
            print(f'steps: {total_timestep} episode: {episode + 1} reward: {stats.episode_rewards[episode]} average: {sum(stats.episode_rewards[:episode + 1])/episode + 1}')
    return agent, stats, {}
