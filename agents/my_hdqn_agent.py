import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple


from utils.replay_memory import ReplayMemory, Transition

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
        self.meta_controller = Model(num_meta_in, num_goal).to(device)
        self.target_meta_controller = Model(num_meta_in, num_goal).to(device)
        self.controller = Model(num_meta_in + num_goal, num_action).to(device)
        self.target_controller = Model(num_meta_in + num_goal, num_action).to(device)
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
            state = torch.tensor(state, device=device, dtype=torch.float32)
            with torch.no_grad():
                return controller(state).data.max(1)[1]
        else:
            return torch.tensor([random.randrange(num)], device=device, dtype=torch.long)

    def update_controller(self, gamma, meta=False):
        memory, model, target, optimizer = (
            [self.meta_replay_memory, self.meta_controller, self.target_meta_controller, self.meta_optimizer] if meta 
            else [self.ctrl_replay_memory, self.controller, self.target_controller, self.ctrl_optimizer])
        
        if len(memory) < self.batch_size:
            return
        state_batch, goal_batch, next_state_batch, ex_reward_batch, done_mask = memory.sample(self.batch_size)
        state_batch = torch.tensor(state_batch, device=device, dtype=torch.float32)
        goal_batch = torch.tensor(goal_batch, device=device)
        next_state_batch = torch.tensor(next_state_batch, device=device, dtype=torch.float32)
        ex_reward_batch = torch.tensor(ex_reward_batch, device=device)
        not_done_mask = torch.tensor(1 - done_mask, device=device)

        # Compute current Q value, meta_controller takes only state and output value for every state-goal pair
        # We choose Q based on goal chosen.
        current_Q_values = model(state_batch).gather(1, goal_batch.unsqueeze(1))
        # Compute next Q value based on which goal gives max Q values
        # Detach variable from the current graph since we don't want gradients for next Q to propagated
        with torch.no_grad():
            next_max_q = target(next_state_batch).max(1)[0]
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