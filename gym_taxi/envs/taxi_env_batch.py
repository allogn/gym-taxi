import gym
from typing import Tuple, Dict, List
from nptyping import Array
from gym import spaces
import numpy as np
import networkx as nx
import math
from collections import Counter

import logging
logger = logging.getLogger(__name__)

from gym_taxi.envs.taxi_env import TaxiEnv

class TaxiEnvBatch(TaxiEnv):
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
        super(TaxiEnvBatch, self).__init__(world, orders, order_sampling_rate, drivers_per_node,
                                n_intervals, wc, count_neighbors,
                                weight_poorest, normalize_rewards, minimum_reward, reward_bound,
                                include_income_to_observation, poorest_first, idle_reward)
        # update action and observation spaces: now covering all network
        # do not override single-cell variables, as they are used to make steps in the parent class
        self.global_action_space_shape = (self.action_space_shape[0]*self.world_size,)
        self.global_action_space = spaces.Box(low=0, high=1, shape=self.global_action_space_shape)

        if include_income_to_observation:
            # new space = old space - incomes - current_cell_onehot + 3*incomes (of a size of the world)
            self.global_observation_space_shape = (self.observation_space_shape[0] + 2*len(self.world) - 3,)
        else:
            # new space = old space - current_cell_onehot
            self.global_observation_space_shape = (self.observation_space_shape[0] - len(self.world),)
        self.observation_space = spaces.Box(low=0, high=1, shape=self.global_observation_space_shape)

    def reset(self) -> Array[int]:
        obs = super(TaxiEnvBatch, self).reset()
        return self.get_global_observation()

    def get_global_observation(self):
        observation, _, _ = self.get_observation()
        global_observation = np.zeros(self.global_observation_space_shape)
        # observation: <driver distr, order distr, one-hot time, 
        #        one-hot location, mean/min/max income (optional)>
        # global_observation: <driver distr, order distr, one-hot time, 
        #        mean/min/max income (optional)>
        size_without_income = 2*self.world_size+self.n_intervals
        global_observation[:size_without_income] = observation[:size_without_income]
        if self.include_income_to_observation:
            for i in range(self.world_size):
                offset = 3*i
                v = self.get_income_per_node(i)
                global_observation[size_without_income+offset:size_without_income+offset+3] = v
        return global_observation

    def step(self, action: Array[float]) -> Tuple[Array[int], float, bool, Dict]:
        if self.done:
            raise Exception("Trying to step terminated environment. Call reset first.")
        global_reward = 0
        reward_per_node = np.zeros(self.world_size)
        init_t = self.time

        total_served_orders = 0
        max_driver = None
        max_order = None
        cells_with_nonzero_drivers = np.sum([1 for n in self.world.nodes(data=True) if n[1]['info'].get_driver_num() > 0])
        last_info = {} # info of the last step should be the correct one
        for i in range(cells_with_nonzero_drivers):
            a = self.current_node_id*self.action_space_shape[0]
            action_per_cell = action[a:a+self.action_space_shape[0]]
            _, reward, done, info = super(TaxiEnvBatch, self).step(action_per_cell)
            last_info = info

            reward_per_node[self.current_node_id] = reward
            global_reward += reward
            total_served_orders += info['served_orders']

            # updated at each step, but the final should be corrent
            max_driver = info["driver normalization constant"]
            max_order = info["order normalization constant"]
            assert done == self.done
            assert i == cells_with_nonzero_drivers-1 or self.done == False

        global_observation = self.get_global_observation()
        assert (not self.done) or (self.time == self.n_intervals)
        assert self.time > init_t # has to be incremented, but might have several steps passed (if no drivers to control/dispatch)

        global_info = {"reward_per_node": reward_per_node,
                        "served_orders": total_served_orders,
                        "driver normalization constant": max_driver,
                        "order normalization constant": max_order
                        }
        global_info.update(last_info)
        return global_observation, global_reward, self.done, global_info

    def get_action_space_shape(self):
        return self.global_action_space_shape

    def print_observation(self):
        global_observation = self.get_global_observation()
        print("Driver distribution:", global_observation[:self.world_size])
        print("Order distribution:", global_observation[self.world_size:self.world_size*2])
        t = 2*self.world_size+self.n_intervals
        print("One-Hot Time:", global_observation[self.world_size*2:t])
        if self.include_income_to_observation:
            print("Incomes:", global_observation[t:])