import gym
from typing import Tuple, Dict, List
from nptyping import Array
from gym import spaces
import numpy as np
import networkx as nx
import math
from collections import Counter
import imageio

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvasAgg
import matplotlib
import seaborn as sns
matplotlib.use('Agg')

import logging
logger = logging.getLogger(__name__)

from gym_taxi.envs.taxi_env import TaxiEnv

class TaxiEnvBatch(gym.Env):
    '''
    This class is a wrapper over taxi_env, providing an interface for cA2C,
    that requires processing drivers in batches + some additional context information
    '''
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self,
                 world: nx.Graph,
                 orders: Tuple[int, int, int, int, float],
                 order_sampling_rate: float,
                 drivers_per_node: Array[int],
                 n_intervals: List,
                 wc: float,
                 count_neighbors: bool = False,
                 weight_poorest: bool = False,
                 normalize_rewards: bool = True,
                 minimum_reward: bool = False,
                 reward_bound: float = None,
                 include_income_to_observation: bool = False,
                 poorest_first: bool = False,
                 idle_reward: bool = False) -> None:

        self.itEnv = TaxiEnv(world, orders, order_sampling_rate, drivers_per_node, n_intervals, wc, count_neighbors,
                                weight_poorest, normalize_rewards, minimum_reward, reward_bound,
                                include_income_to_observation, poorest_first, idle_reward)
        self.world = self.itEnv.world
        self.n_intervals = n_intervals
        self.n_drivers = self.itEnv.n_drivers
        self.time = 0
        self.include_income_to_observation = include_income_to_observation
        self.one_cell_action_space = self.itEnv.max_degree+1
        self.action_space = spaces.Box(low=0, high=1, shape=(self.one_cell_action_space*self.itEnv.world_size,))

        if include_income_to_observation:
            assert self.itEnv.observation_space_shape[0] == 3*len(self.world) + self.itEnv.n_intervals + 3
            self.observation_space_shape = (self.itEnv.observation_space_shape[0] + 2*len(self.world) - 3,)
        else:
            assert self.itEnv.observation_space_shape[0] == 3*len(self.world) + self.itEnv.n_intervals
            self.observation_space_shape = (self.itEnv.observation_space_shape[0] - len(self.world),)
        self.observation_space = spaces.Box(low=0, high=1, shape=self.observation_space_shape)

    def reset(self) -> Array[int]:
        self.time = 0
        if self.itEnv.include_income_to_observation:
            t = self.itEnv.world_size + 3
            observation = self.itEnv.reset()[:-t]
            # assuming all incomes are zero
            return np.concatenate((observation, np.zeros(3*self.itEnv.world_size)))
        else:
            t = self.itEnv.world_size
            return self.itEnv.reset()[:-t]

    def get_reset_info(self):
        '''
        Currently used only to get max_orders and max_drivers, that should current_cell independent
        '''
        return self.itEnv.get_reset_info()

    def step(self, action: Array[float]) -> Tuple[Array[int], float, bool, Dict]:
        cells_with_nonzero_drivers = np.sum([1 for n in self.itEnv.world.nodes(data=True) if n[1]['info'].get_driver_num() > 0])
        nodes_with_orders = np.sum([1 for n in self.itEnv.world.nodes(data=True) if n[1]['info'].get_order_num() > 0])
        total_orders = np.sum([n[1]['info'].get_order_num() for n in self.itEnv.world.nodes(data=True)])
        global_observation = np.zeros(5*self.itEnv.world_size + self.itEnv.n_intervals)
        global_done = False
        global_reward = 0
        reward_per_node = np.zeros(self.itEnv.world_size)
        init_t = self.itEnv.time
        self.last_action_for_drawing = action

        total_served_orders = 0
        max_driver = None
        max_order = None

        for i in range(cells_with_nonzero_drivers):
            current_cell = self.itEnv.current_node_id
            a = current_cell*self.one_cell_action_space
            action_per_cell = action[a:a+self.one_cell_action_space]

            observation, reward, done, info = self.itEnv.step(action_per_cell)

            reward_per_node[current_cell] = reward
            global_done = done
            global_reward += reward
            total_served_orders += info['served_orders']

            # updated at each step, but the final should be corrent
            max_driver = info["driver normalization constant"]
            max_order = info["order normalization constant"]

            if self.itEnv.include_income_to_observation:
                assert observation.shape[0] == 3*self.itEnv.world_size+self.itEnv.n_intervals+3
                size_without_income = 2*self.itEnv.world_size+self.itEnv.n_intervals
                ws = self.itEnv.world_size
                offset = current_cell
                global_observation[:size_without_income] = observation[:size_without_income]
                global_observation[size_without_income+3*offset:size_without_income+3*offset+3] = observation[-3:]
            else:
                global_observation = observation[:-self.itEnv.world_size]

        # if cells_with_nonzero_drivers == 0:
        #     observation, reward, done, info = self.itEnv.step(action_per_cell)
        #
        #     reward_per_node[current_cell] = reward
        #     global_done = done
        #     global_reward += reward
        #     total_served_orders += info['served_orders']
        #
        #     # updated at each step, but the final should be corrent
        #     max_driver = info["driver normalization constant"]
        #     max_order = info["order normalization constant"]
        #
        #     if self.itEnv.include_income_to_observation:
        #         assert observation.shape[0] == 3*self.itEnv.world_size+self.itEnv.n_intervals+3
        #         size_without_income = 2*self.itEnv.world_size+self.itEnv.n_intervals
        #         ws = self.itEnv.world_size
        #         offset = current_cell
        #         global_observation[:size_without_income] = observation[:size_without_income]
        #         global_observation[size_without_income+3*offset:size_without_income+3*offset+3] = observation[-3:]
        #     else:
        #         global_observation = observation[:-self.itEnv.world_size]

        assert not global_done or init_t + 1 == self.itEnv.n_intervals
        assert self.itEnv.time == init_t + 1
        self.time += 1

        global_info = {"reward_per_node": reward_per_node,
                        "served_orders": total_served_orders,
                        "nodes_with_drivers": cells_with_nonzero_drivers,
                        "nodes_with_orders": nodes_with_orders,
                        "driver normalization constant": max_driver,
                        "order normalization constant": max_order,
                        "total_orders": total_orders,
                        "idle_reward": float(np.mean([d.get_idle_period() for d in self.itEnv.all_driver_list])),
                        "min_idle": float(np.min([d.get_idle_period() for d in self.itEnv.all_driver_list]))}
        return global_observation, global_reward, global_done, global_info

    def seed(self, seed):
        self.itEnv.seed(seed)

    def get_min_revenue(self):
        return self.itEnv.get_min_revenue()

    def get_total_revenue(self):
        return self.itEnv.get_total_revenue()

    def compute_remaining_drivers_and_orders(self, state):
        return self.itEnv.compute_remaining_drivers_and_orders(state)

    def set_income_bound(self, bound):
        self.itEnv.set_income_bound(bound)

    def render(self, mode='rgb_array'):
        fig = plt.figure(figsize=(20, 20))
        ax = fig.gca()
        ax.axis('off')

        pos = nx.get_node_attributes(self.world, 'coords')
        G = nx.DiGraph(self.world)
        nodelist = []
        edgelist = []
        action = self.last_action_for_drawing
        act = self.itEnv.action_space_shape[0]
        node_colors = []
        edge_colors = []
        for n in self.world.nodes():
            node_action = action[act*n:act*(n+1)]
            nodelist.append(n)
            node_colors.append(node_action[-1])
            j = 0
            added = 0
            for nn in self.world.neighbors(n):
                if node_action[j] > 0:
                    edgelist.append((n,nn))
                    edge_colors.append(node_action[j])
                    added += 1
                j += 1
            assert abs(np.sum(node_action) - 1) < 0.00001, node_action
            assert node_action[-1] != 0 or added > 0, (node_action, n)

        nx.draw_networkx(G, edgelist=edgelist, edge_color=edge_colors, vmin=-1, vmax=1, node_shape='.', edge_vmax=1.1,
                            cmap=matplotlib.cm.get_cmap("Blues"), edge_cmap=matplotlib.cm.get_cmap("Blues"),
                            node_color=node_colors, nodelist=nodelist, pos=pos, arrows=True, with_labels=False, ax=ax)

        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        s, (width, height) = canvas.print_to_buffer()

        # Option 2a: Convert to a NumPy array
        X = np.frombuffer(s, np.uint8).reshape((height, width, 4))
        plt.close(fig)
        return X
