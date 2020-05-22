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
    """
    This class merges steps per all cells as a single step.
    """
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self,
                 world: nx.Graph,
                 orders: Tuple[int, int, int, int, float],
                 order_sampling_rate: float,
                 drivers_per_node: Array[int],
                 n_intervals: List,
                 wc: float = 0,
                 count_neighbors: bool = False,
                 weight_poorest: bool = False,
                 normalize_rewards: bool = True,
                 minimum_reward: bool = False,
                 reward_bound: float = None,
                 include_income_to_observation: bool = False,
                 poorest_first: bool = False,
                 idle_reward: bool = False,
                 seed: int = 0,
                 debug: bool = True) -> None:
        super(TaxiEnvBatch, self).__init__(world, orders, order_sampling_rate, drivers_per_node,
                                n_intervals, wc, count_neighbors,
                                weight_poorest, normalize_rewards, minimum_reward, reward_bound,
                                include_income_to_observation, poorest_first, idle_reward, seed, True, debug)

    def set_action_and_observation_space(self, max_degree, world_size, n_intervals):
        super(TaxiEnvBatch, self).set_action_and_observation_space(max_degree, world_size, n_intervals)

        # do not override single-cell variables (e.g. action_space_shape), 
        # as they are used to make steps in the parent class
        self.global_action_space_shape = (self.action_space_shape[0]*world_size,)
        self.global_action_space = spaces.Box(low=0, high=1, shape=self.global_action_space_shape)

        # current node_id is dropped from observation
        self.global_observation_space_shape = (self.observation_space_shape[0] - world_size,)
        assert self.global_observation_space_shape[0] == 3*world_size + n_intervals or \
                    self.global_observation_space_shape[0] == 4*world_size + n_intervals
        self.global_observation_space = spaces.Box(low=0, high=1, shape=self.global_observation_space_shape)

    def reset(self) -> Array[int]:
        obs = super(TaxiEnvBatch, self).reset()
        return self.get_global_observation()

    def get_reset_info(self) -> Dict:
        obs, driver_max, order_max = self.get_observation()
        return {"served_orders": 0,
                "driver normalization constant": self.episode_logs["driver normalization constant"], # required for cA2C
                "order normalization constant": self.episode_logs["order normalization constant"]
                }

    def get_global_observation(self):
        """
        Returns observation from env without node_id.
        Observation:
            <driver distr, order distr, idle_drivers, one-hot time, min income (or idle times) (optional)>
            which is of (3/4)*world_size + n_intervals
        """
        observation, _, _ = self.get_observation()
        global_observation = np.zeros(self.global_observation_space_shape)
        # global_obs is obs without cell_id
        world_size = len(self.full_to_view_ind)
        global_observation[:3*world_size+self.n_intervals] = observation[:3*world_size+self.n_intervals]
        global_observation[3*world_size+self.n_intervals:] = observation[4*world_size+self.n_intervals:]
        assert (global_observation >= 0).all() and (global_observation <= 1).all()
        return global_observation

    def step(self, action: Array[float]) -> Tuple[Array[int], float, bool, Dict]:
        if self.done:
            raise Exception("Trying to step terminated environment. Call reset first.")
        global_reward = 0
        world_size = len(self.full_to_view_ind)
        reward_per_node = np.zeros(world_size)
        init_t = self.time

        total_served_orders = 0
        max_driver = None
        max_order = None
        cells_with_nonzero_drivers = np.sum([1 for n, _ in self.full_to_view_ind.items() \
                                                if self.world.nodes[n]['info'].get_driver_num() > 0])
        last_info = {} # info of the last step should be the correct one
        for i in range(cells_with_nonzero_drivers):
            a = self.full_to_view_ind[self.current_node_id]*self.action_space_shape[0]
            action_per_cell = action[a:a+self.action_space_shape[0]]
            _, reward, done, info = super(TaxiEnvBatch, self).step(action_per_cell)
            last_info = info

            reward_per_node[self.full_to_view_ind[self.current_node_id]] = reward
            global_reward += reward
            total_served_orders += info['served_orders']

            assert done == self.done
            assert i == cells_with_nonzero_drivers-1 or self.done == False

        global_observation = self.get_global_observation()
        assert (not self.done) or (self.time == self.n_intervals)
        assert self.time > init_t # has to be incremented, but might have several steps passed (if no drivers to control/dispatch)

        global_info = {"reward_per_node": reward_per_node,
                        "served_orders": total_served_orders,
                        "driver normalization constant": self.episode_logs["driver normalization constant"], # required for cA2C
                        "order normalization constant": self.episode_logs["order normalization constant"] # required for cA2C
                        }
        global_info.update(last_info)
        return global_observation, global_reward, self.done, global_info

    def get_action_space_shape(self):
        return self.global_action_space_shape

    def get_observation_space_shape(self):
        return self.global_observation_space_shape

    def print_observation(self):
        global_observation = self.get_global_observation()
        world_size = len(self.full_to_view_ind)
        print("Driver distribution:", global_observation[:world_size])
        print("Order distribution:", global_observation[world_size:world_size*2])
        print("Idle distribution:", global_observation[2*world_size:world_size*3])
        t = 3*world_size+self.n_intervals
        print("One-Hot Time:", global_observation[world_size*3:t])
        if self.include_income_to_observation:
            print("Incomes:", global_observation[t:])