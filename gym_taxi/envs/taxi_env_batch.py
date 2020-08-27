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
                 penalty_for_invalid_action: float = 1000,
                 driver_automatic_return: bool = True,
                 include_action_mask: bool = False,
                 hold_observation: bool = True,
                 discrete: bool = False,
                 fully_collaborative: bool = False,
                 bounded_income: bool = False,
                 waiting_period: int = 1,
                 randomize_drivers: bool = False,
                 debug: bool = True) -> None:
        """
        :param fully_collaborative: global actions are defined per each car, not per cell
        """
        self.fully_collaborative = fully_collaborative
        self.cool_start = True # a bit that is used to ignore a very first action, because it does not use the action mask
        super(TaxiEnvBatch, self).__init__(world, orders, order_sampling_rate, drivers_per_node,
                                n_intervals, wc, count_neighbors,
                                weight_poorest, normalize_rewards, minimum_reward, reward_bound,
                                include_income_to_observation, poorest_first, idle_reward, seed, True, 
                                penalty_for_invalid_action, driver_automatic_return, include_action_mask, 
                                discrete, bounded_income, waiting_period, randomize_drivers, debug)


    def set_action_and_observation_space(self, max_degree, world_size, n_intervals):
        super(TaxiEnvBatch, self).set_action_and_observation_space(max_degree, world_size, n_intervals)

        # do not override single-cell variables (e.g. action_space_shape), 
        # as they are used to make steps in the parent class\
        if self.fully_collaborative:
            # note that the shape is exponential (!!!) in number of drivers
            self.global_action_space_shape = (self.action_space_shape[0]**self.n_drivers,)
            self.action_space = spaces.Discrete(self.global_action_space_shape[0])
        else:
            self.global_action_space_shape = (self.action_space_shape[0]*world_size,)
            self.action_space = spaces.Box(low=0, high=1, shape=self.global_action_space_shape) # not "global"! this var is used by Gym

        # current node_id is dropped from observation, and cruising bit per car is added if fully_collaborative
        if self.fully_collaborative:
            self.global_observation_space_shape = (self.observation_space_shape[0] - world_size + self.n_drivers,)
        else:
            self.global_observation_space_shape = (self.observation_space_shape[0] - world_size,)
        assert self.global_observation_space_shape[0] in [3*world_size + n_intervals, \
                        4*world_size + n_intervals, 3*world_size + n_intervals + self.n_drivers, 3*world_size + n_intervals + 2*self.n_drivers]
        self.observation_space = spaces.Box(low=0, high=1, shape=self.global_observation_space_shape)

    def reset(self) -> Array[int]:
        self.cool_start = True
        obs = super(TaxiEnvBatch, self).reset()
        return self.get_global_observation()

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
        if self.fully_collaborative:
            global_observation[3*world_size+self.n_intervals:-self.n_drivers] = observation[4*world_size+self.n_intervals:]
            global_observation[-self.n_drivers:] = np.array([d.status for d in self.all_driver_list])
        else:    
            global_observation[3*world_size+self.n_intervals:] = observation[4*world_size+self.n_intervals:]
        assert (global_observation >= 0).all() and (global_observation <= 1).all()
        return global_observation

    def get_individual_action(self, action):
        if self.fully_collaborative:
            assert type(action) == int or type(action) == np.int64, (action, type(action))
            d = self.world.nodes[self.current_node_id]['info'].drivers
            if len(d) == 0:
                return None # the rest of the drivers are travelling
            next_driver_id = d[0].driver_id
            ind = np.unravel_index(action, self.get_global_action_mask_shape())
            agent_action_id = ind[next_driver_id]
            individual_action = int(agent_action_id)
        else:
            a = self.full_to_view_ind[self.current_node_id]*self.action_space_shape[0]
            individual_action = action[a:a+self.action_space_shape[0]]
        return individual_action

    def get_global_action_mask_shape(self):
        n_actions = self.action_space_shape[0]
        tensor_mask_shape = [int(n_actions) for a in range(self.n_drivers)]
        return tuple(tensor_mask_shape)

    def get_global_action_mask(self):
        if not self.discrete or not self.fully_collaborative:
            return None
        assert self.include_action_mask
        n_actions = self.action_space_shape[0]
        tensor_mask_shape = self.get_global_action_mask_shape()
        tensor_mask = np.ones(tensor_mask_shape, dtype=int)
        for a_id in range(self.n_drivers):
            node = self.all_driver_list[a_id].position
            agent_mask = np.where(np.array(self.get_action_mask(node)) == 0) # index ids!
            # print("agent {}, mask {}, ind {}, pos {}".format(a_id, self.get_action_mask(node), agent_mask, node))
            index = [slice(0,n_actions) for i in range(self.n_drivers)]
            index[a_id] = agent_mask
            index = tuple(index)
            tensor_mask[index] = 0 # we do logical and for all agents, so for each agent_mask=0, the resulting value is also 0
        return tensor_mask.flatten()

    def step(self, action):
        """
        :param action: action can be either a vector (e.g. cA2C), or an id (fully-collaborative)
        """
        if self.done:
            raise Exception("Trying to step terminated environment. Call reset first.")
        
        world_size = len(self.full_to_view_ind)
        global_reward = 0        
        init_t = self.time
        reward_per_node = np.zeros(world_size)
        total_served_orders = 0
        last_info = {} # info of the last step should be the correct one

        if self.cool_start:
            self.cool_start = False
        else:

            if self.fully_collaborative:
                individual_iterations = self.n_drivers
            else:
                cells_with_nonzero_drivers = np.sum([1 for n, _ in self.full_to_view_ind.items() \
                                                        if self.world.nodes[n]['info'].get_driver_num() > 0])
                individual_iterations = cells_with_nonzero_drivers

            start_time = self.time
            for i in range(individual_iterations):
                individual_action = self.get_individual_action(action)

                if individual_action is None or self.time > start_time:
                    # the rest of the drivers are travelling

                    # we don't know what number of cars we need to manage, but if time was updated - then drivers were put back in the nodes
                    # so we need to stop this iterations
                    break

                cell_id_that_brings_reward = self.current_node_id
                observation_per_cell, reward, done, last_info = super(TaxiEnvBatch, self).step(individual_action)

                reward_per_node[self.full_to_view_ind[cell_id_that_brings_reward]] += reward
                if self.minimum_reward:
                    if global_reward > reward:
                        global_reward = reward
                else:
                    global_reward += reward
                total_served_orders += last_info['served_orders']

                assert done == self.done
                assert i == individual_iterations-1 or self.done == False or self.time > start_time

            assert (not self.done) or (self.time == self.n_intervals)
            assert self.time > init_t # has to be incremented, but might have several steps passed (if no drivers to control/dispatch)
            if not self.minimum_reward:
                assert np.abs(np.sum(reward_per_node) - global_reward) < 0.0001

        global_observation = self.get_global_observation()
        global_info = {
            "reward_per_node": reward_per_node,
            "served_orders": total_served_orders,
            "driver normalization constant": self.episode_logs["driver normalization constant"], # required for cA2C
            "order normalization constant": self.episode_logs["order normalization constant"] # required for cA2C
        }
        global_info.update(last_info)
        global_info["action_mask"] = self.get_global_action_mask()
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

    def print_observation_per_cell(self, obs):
        world_size = len(self.full_to_view_ind)
        print("Drivers distribution:", obs[:world_size])
        print("Order distribution:", obs[world_size:world_size*2])
        print("Idle distribution:", obs[2*world_size:world_size*3])
        t = self.n_intervals
        print("One-Hot Time:", obs[world_size*3:world_size*3+t])
        print("One-Hot Node:", obs[world_size*3+t:world_size*4+t])
        if self.include_income_to_observation:
            print("Incomes:", obs[t:])