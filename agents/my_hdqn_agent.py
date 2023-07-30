import numpy as np
import random
from collections import namedtuple
import my_hdqn_learning
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from utils.replay_memory import ReplayMemory, Transition

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)

class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.fc2 = nn.Linear(256, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


"""
    OptimizerSpec containing following attributes
        constructor: The optimizer constructor ex: RMSprop
        kwargs: {Dict} arguments for constructing optimizer
"""
OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])

class hDQN():
    def __init__(self,
                 optimizer_spec,
                 num_goal,
                 num_action,
                 num_meta_in,
                 replay_memory_size,
                 batch_size):
        ###############
        # BUILD MODEL #
        ###############
        self.num_goal = num_goal
        self.num_action = num_action
        self.batch_size = batch_size
        # Construct meta-controller and controller
        self.meta_controller = Model(num_meta_in, num_goal).type(dtype)
        self.target_meta_controller = Model(num_meta_in, num_goal).type(dtype)
        self.controller = Model(num_meta_in + num_goal, num_action).type(dtype)
        self.target_controller = Model(num_meta_in + num_goal, num_action).type(dtype)
        # Construct the optimizers for meta-controller and controller
        self.meta_optimizer = optimizer_spec.constructor(self.meta_controller.parameters(), **optimizer_spec.kwargs)
        self.ctrl_optimizer = optimizer_spec.constructor(self.controller.parameters(), **optimizer_spec.kwargs)
        # Construct the replay memory for meta-controller and controller
        self.meta_replay_memory = ReplayMemory(replay_memory_size)
        self.ctrl_replay_memory = ReplayMemory(replay_memory_size)
    
    def check_goal(self, goal, extrinsic_reward):
        reward_list = [40, 100, 300, 1200]
        return reward_list[goal] == extrinsic_reward

    def get_intrinsic_reward(self, goal, extrinsic_reward):
        return 1.0 if self.check_goal(goal, extrinsic_reward) else 0.0
         
    def select_action(self, state, epilson, goal=False):
        num, controller = [self.num_goal, self.meta_controller] if goal else [self.num_action, self.controller]
        sample = random.random()
        if sample > epilson:
            state = torch.from_numpy(state).type(dtype)
            # Use volatile = True if variable is only used in inference mode, i.e. don’t save the history
            return controller(Variable(state, volatile=True)).data.max(1)[1].cpu()
        else:
            return torch.IntTensor([random.randrange(num)])

    def update_controller(self, gamma, meta=False):
        memory, model, target, optimizer = (
            [self.meta_replay_memory, self.meta_controller, self.target_meta_controller, self.meta_optimizer] if meta 
            else [self.ctrl_replay_memory, self.controller, self.target_controller, self.ctrl_optimizer])
        
        if len(memory) < self.batch_size:
            return
        state_batch, goal_batch, next_state_batch, ex_reward_batch, done_mask = \
            memory.sample(self.batch_size)
        state_batch = Variable(torch.from_numpy(state_batch).type(dtype))
        goal_batch = Variable(torch.from_numpy(goal_batch).long())
        next_state_batch = Variable(torch.from_numpy(next_state_batch).type(dtype))
        ex_reward_batch = Variable(torch.from_numpy(ex_reward_batch).type(dtype))
        not_done_mask = Variable(torch.from_numpy(1 - done_mask)).type(dtype)
        if USE_CUDA:
            goal_batch = goal_batch.cuda()
        # Compute current Q value, meta_controller takes only state and output value for every state-goal pair
        # We choose Q based on goal chosen.
        current_Q_values = model(state_batch).gather(1, goal_batch.unsqueeze(1))
        # Compute next Q value based on which goal gives max Q values
        # Detach variable from the current graph since we don't want gradients for next Q to propagated
        next_max_q = target(next_state_batch).detach().max(1)[0]
        next_Q_values = not_done_mask * next_max_q
        # Compute the target of the current Q values
        target_Q_values = ex_reward_batch + (gamma * next_Q_values)
        # Compute Bellman error (using Huber loss)
        criterion = nn.SmoothL1Loss()
        loss = criterion(current_Q_values, target_Q_values)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 100)
        optimizer.step()

        soft_update_target(model, target)


def soft_update_target(model, target, TAU=0.005):
    target_state_dict = target.state_dict()
    model_state_dict = model.state_dict()
    for k in model_state_dict:
        target_state_dict[k] = model_state_dict[k] * TAU + target_state_dict[k] * (1-TAU)
    target.load_state_dict(target_state_dict)