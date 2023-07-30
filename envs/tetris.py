from typing import Any
from gymnasium.spaces.space import MaskNDArray
import numpy as np
import random
import gymnasium as gym
import jumanji
import envs.wrapper
from gymnasium import spaces
import jax

class MyDiscrete(spaces.Discrete):
    def __init__(self, n, state):
        super().__init__(n)
        self.state = state


class Tetris(gym.Env):
    """Custom Environment that follows gym interface"""

    # metadata = {'render.modes': ['human']}

    def __init__(self, num_rows, num_cols):
        super(Tetris, self).__init__()
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.jumanji_env = jumanji.make(
            "Tetris-v0", num_rows=num_rows, num_cols=num_cols
        )
        self.env = envs.wrapper.JumanjiToGymWrapper(self.jumanji_env)
        state = self.env.reset()

        self.observation_space = spaces.MultiBinary(num_cols * num_cols + 4 * 4)
        self.action_space = MyDiscrete(4 * self.num_cols, state)

    def extract_obs(self, state):
        return np.array(
            [x for row in state["grid"] for x in row]
            + [x for row in state["tetromino"] for x in row],
            dtype=np.int8,
        )

    def step(self, action, add_jumanji_state=False):
        action = np.unravel_index(action, (4, self.num_cols))
        state, reward, done, extra, jumanji_state = self.env.step(action)
        self.action_space.state = state
        observation = self.extract_obs(state)
        info = state
        if add_jumanji_state:
          return observation, reward, done, False, info, jumanji_state
        else:
          return observation, reward, done, False, info
    
    def reset(self, seed=None):
        state, jumanji_state = self.env.reset(seed=seed)
        self.action_space.state = state
        observation = self.extract_obs(state)
        return observation, jumanji_state  # reward, done, info can't be included

    def render(self, mode="human"):
        self.env.render()

    def animate(
        self,
        states,
        interval=100,
        save_path=None,
    ):
        return self.jumanji_env.animate(states, interval, save_path)

    def close(self):
        self.close()


