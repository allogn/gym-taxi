import os, subprocess, time, signal
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvasAgg
import matplotlib
matplotlib.use('Agg')

import logging
logger = logging.getLogger(__name__)

class TaxiEnvDummy(gym.Env):
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self, gridsize):
        super(TaxiEnvDummy, self).__init__()
        self.cells = gridsize
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(low=0, high=1, shape=(3,))
        # Example for using image as input:
        self.observation_space = spaces.MultiDiscrete([2]*self.cells)
        # one-hot of position of current car



    def step(self, action):
        norm_action = action / np.sum(action)
        if np.sum(action) == 0:
            norm_action = np.ones(action.shape)/action.shape[0]
        move = np.random.choice([-1, 0, 1], p=norm_action)
        self.position += move
        self.last_move = move
        if self.position < 0:
            self.position = 0

        reward = 0
        done = False
        if self.position == self.cells-1:
            reward = 100
            done = True
        observation = np.zeros(self.cells)
        observation[self.position] = 1
        info = {}
        return observation, reward, done, info

    def reset(self):
        self.position = 0
        self.last_move = 0
        observation = np.zeros(self.cells)
        observation[self.position] = 1
        return observation

    def render(self, mode='rgb_array'):
        fig = plt.figure()
        ax = fig.gca()
        ax.axis('off')
        plt.scatter(list(range(self.cells)),[0]*self.cells)
        plt.arrow(self.position, 0, self.last_move*0.5, 0, length_includes_head=True, head_width=0.003, head_length=0.2) #x,y,dx,dy

        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        s, (width, height) = canvas.print_to_buffer()

        # Option 2a: Convert to a NumPy array
        X = np.frombuffer(s, np.uint8).reshape((height, width, 4))
        plt.close(fig)
        return X
