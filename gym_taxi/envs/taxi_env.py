import gym
from typing import Tuple, Dict, List
from nptyping import Array
from gym import spaces
import numpy as np
import networkx as nx
import uuid
import copy
import math
import time
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvasAgg
import matplotlib
matplotlib.use('Agg')

import logging
logger = logging.getLogger(__name__)

from gym_taxi.envs.node import Node
from gym_taxi.envs.driver import Driver
from gym_taxi.envs.helpers import *

ActionList = List[Tuple[int, int, float, int]]

class TaxiEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array', 'fig']}

    def __init__(self,
                 world: nx.Graph,
                 orders: Tuple[int, int, int, int, float],
                 order_sampling_rate: float,
                 drivers_per_node: Array[int],
                 n_intervals: int,
                 wc: float = 0,
                 count_neighbors: bool = False,
                 weight_poorest: bool = False,
                 normalize_rewards: bool = True,
                 minimum_reward: bool = False,
                 reward_bound: float = None,
                 include_income_to_observation: bool = False,
                 poorest_first: bool = False,
                 idle_reward: bool = False, 
                 seed: int = None,
                 hold_observation: bool = True,
                 penalty_for_invalid_action: float = 1000,
                 driver_automatic_return: bool = True,
                 include_action_mask: bool = False,
                 discrete: bool = False,
                 bounded_income: bool = False,
                 waiting_period: int = 1,
                 randomize_drivers: bool = False,
                 debug: bool = True) -> None: 
        '''
        :param world: undirected networkx graph that represents spatial cells for drivers to travel
                        nodes should be enumerated sequntially
        :param orders: a list of *all* orders during a month or any long period
                        format: tuple <source, destination, time, length, price>
                        source and destination should belong to the world.
                        time can be greater than n_intervals, random sampling will proceed by the time % n_intervals
        :param order_sampling_rate: fraction of orders to sample per time step. 
                        For example, if n_intervals = intervals per day, orders are per 30 days, then sampling_rate=1/30
                        means orders will be sampled with average density per time step
        :param drivers_per_node: a distribution of drivers per node
        :param n_intervals: number of time intervals per day
        :param wc: walk cost, a cost of taxi relocation for 1 hop
        :param count_neighbors: an option that allows drivers to match with neighbour cells
        :param weight_poorest: multiply reward of each car by softmax of their income so far
        :param normalize_rewards: divide a reward per cell by number of cars in a cell
        :param minimum_reward: return reward as minimum income per driver per cell
        :param reward_bound: return reward such that no car can earn more than this bound 
                            (in fact, it is the income bound, not very good naming. reward might be different)
        :param include_income_to_observation: observation includes car income distribution
        :param poorest_first: assignment strategy such that poorest drivers get assigned first
        :param idle_reward: reward by the time a driver is idle rather than by income
        :param seed: seed for a random state
        :param hold_observation: at each step return observation as like in the beginning of the time interval (true)
                                    or like after applying changes to the previous node (false)
        :param penalty_for_invalid_action: reward is decreased by this coef * sum(invalid action)
        :param driver_automatic_return: if a view is used and the driver is sent outside the view, then setting 
                                        this to true makes the driver to appear in the closest node within view
                                        assuming that it drives back as soon as the customer is delivered
        :param discrete: action space is discrete, applied per each car
        :param randomize_drivers: drivers have random initial position and income in the beginning of episode
        :param debug: extra consistency checks

        Change self.DEBUG to False in __init__() to disable consistency checks per iteration.

        Usage:
        A policy decides where for an idle (free) driver to go.
        step() function returns observation for the next state, reward for parforming the step, 
        done=True if the next state is the final one (time of the next state = n_intervals)
        and a dict with additional information on the new state and transition.

        reset() returns initial observation. This function is required for Gym.
        get_reset_info() returns additional info about the initial step. 
        Note that done might be equal to True right after initialization, so the external check is required.

        Simulator maintains a time counter. Time is increased once there are no idle drivers to decide upon.
        Time can possibly increase from 0 to n_intervals. Calling step() further results in error.

        If there is no drivers to decide upon up to the time=n_intervals, then done=True is returned immediately.
        This means that number of step()'s is might not be equal to n_intevals*number of nodes

        Important note: although a policy influences only idle drivers, rewards and steps are defined for any 
        available drivers. That is, if all drivers are dispatched, we still count the step as valid.
        '''
        self.seed(seed)
        self.env_id = str(uuid.uuid1())
        self.DEBUG = debug
        if self.DEBUG:
            logging.warning("DEBUG mode is active for taxi_env")
        self.number_of_resets = 0
        self.best_income_so_far = 0

        # Setting simulator parameters
        super(TaxiEnv, self).__init__()
        self.world = nx.Graph(world)
        self.world_size = len(self.world)
        for i in range(self.world_size):
            assert self.world.has_node(i) # check world node ids are sequential
            self.world.nodes[i]['info'] = Node(i) # initiating basic information per node

        self.n_intervals = n_intervals
        assert n_intervals > 0

        self.order_sampling_rate = order_sampling_rate
        assert order_sampling_rate > 0 and order_sampling_rate <= 1

        self.n_drivers = np.sum(drivers_per_node)
        assert self.n_drivers >= 0

        self.wc = wc
        assert self.wc >= 0

        self.poorest_first = poorest_first
        self.idle_reward = idle_reward
        self.minimum_reward = minimum_reward 
        self.income_bound = reward_bound
        self.weight_poorest = weight_poorest 
        self.count_neighbors = count_neighbors
        self.normalize_rewards = normalize_rewards
        self.include_income_to_observation = include_income_to_observation
        self.hold_observation = hold_observation
        self.penalty_for_invalid_action = penalty_for_invalid_action
        self.driver_automatic_return = driver_automatic_return
        self.include_action_mask = include_action_mask
        self.bounded_income = bounded_income
        self.discrete = discrete
        self.waiting_period = waiting_period
        self.randomize_drivers = randomize_drivers
        if discrete:
            assert self.include_action_mask

        self.drivers_per_node = np.array(drivers_per_node)
        assert self.drivers_per_node.dtype == int
        assert self.drivers_per_node.shape == (self.world_size,)

        self.all_driver_list = []
        self.driver_dict = {}
        self.set_orders_per_time_interval(orders)

        max_degree = np.max([d[1] for d in nx.degree(self.world)])
        self.set_action_and_observation_space(max_degree, self.world_size, self.n_intervals)

        self.episode_logs = {}
        self.set_view([n for n in self.world.nodes()]) # set default view to all nodes, call init inside
        self.last_episode_logs = None
    
    def set_action_and_observation_space(self, max_degree, world_size, n_intervals):
        # set action space
        self.action_space_shape = (max_degree+1,)
        if self.discrete:
            self.action_space = spaces.Discrete(max_degree+1)
            self.max_action_id = max_degree # starting from 0
        else:
            self.action_space = spaces.Box(low=0, high=1, shape=self.action_space_shape)

        # set observation space: distribution of drivers, orders, current cell and current time
        self.observation_space_shape = 4*world_size + n_intervals
        if self.include_income_to_observation:
            # optionally with mean,avg,min incomes
            if self.discrete:
                self.observation_space_shape += self.n_drivers
            else:
                self.observation_space_shape += world_size

        self.observation_space_shape = (self.observation_space_shape,)

        self.observation_space = spaces.Box(low=0, high=1, shape=self.observation_space_shape)


    def reset_episode_logs(self):
        # updating values that accumulate over steps. other values stay as they are, 
        # so that max_driver and max_order are saved from previous observation update
        self.episode_logs.update({
            'order_response_rates': [],
            'number_of_idle_drivers': [],
            'number_of_served_orders': [],
            'nodes_with_drivers': [],
            'nodes_with_orders': [],
            'rewards': [],
            'gini': 0.,
            "total_steps": 0., # must be float for type check in callbacks
            'last_time_step': 0.
        })

    def init(self):
        # this function is separated from reset() because taxi_env_batch overrides reset
        # and initialization of taxi_env_batch otherwise produce error due to wrong call to wrong reset
        self.time = 0
        self.total_steps = 0
        self.done = False
        self.traveling_pool = {} # lists of drivers in traveling state, indexed by arrival time
        for i in range(self.n_intervals+1): # the last element contains all drivers arriving out of the episode
            self.traveling_pool[i] = []
        self.last_total_orders = self.bootstrap_orders()
        self.bootstrap_drivers()
        self.find_non_empty_nodes()
        updated = not self.update_current_node_id()
        assert updated, "Initial state does not have drivers to manage"
        self.last_timestep_dispatch = {} # container for dispatch actions, used in plotting
        self.this_timestep_dispatch = {} # extra container so that each time we can plot for t-1, and save for this for plotting
        self.overall_timestep_dispatch = {} # this is to plot edge usage over an episode

        self.reset_episode_logs()
        self.time_profile = np.zeros(5)
        obs, max_driver, max_order = self.get_observation()
        self.update_episode_logs_once_per_obs_update(max_driver, max_order)
        self.last_time_step_obs = obs
        if self.bounded_income:
            self.set_income_bound(self.best_income_so_far)

    def reset(self) -> Array[int]:
        self.auto_update_income_bound()
        self.number_of_resets += 1
        self.init()
        obs, max_driver, max_order = self.get_observation()
        self.update_episode_logs_once_per_obs_update(max_driver, max_order)
        self.last_time_step_obs = obs
        return obs

    def get_number_of_resets(self) -> int:
        return self.number_of_resets

    def step(self, action) -> Tuple[Array[int], float, bool, Dict]:
        """
        Applies the action, and returns observation.
        If self.hold_observation parameter is true (default), then
        the observation is always returned as it is in the beginning of the time step.

        Otherwise, observation is updated at each step (after each node update). 
        In this case, a significant time overhead for updating the observation is expected.

        :returns: observation, reward, done, info
        """
        if self.done:
            raise Exception("Trying to step terminated environment. Call reset first.")
        assert self.time < self.n_intervals

        if self.discrete:
            assert action >= 0 and action <= self.max_action_id
        else:
            assert action.shape == self.action_space_shape, (action.shape, self.action_space_shape)
        
        if self.episode_logs['last_time_step'] == 0:
            assert self.time == 0

        t1 = time.time()
        info = {
            "total_orders": self.last_total_orders,
            "left_orders": 0,
            "nodes_with_orders": 0,
            "nodes_with_drivers": 0
        }
        if self.DEBUG:
            # saving time on large scale
            for n, _ in self.full_to_view_ind.items():
                node = self.world.nodes[n]['info']
                info["left_orders"] += node.get_order_num()
                info["nodes_with_orders"] += 1 if node.get_order_num() > 0 else 0
                info["nodes_with_drivers"] += 1 if node.get_driver_num() > 0 else 0

        dispatch_actions, unmasked_sum = self.get_dispatch_actions_from_action(action)
        if self.include_action_mask:
            assert unmasked_sum == 0 # action_mask==1 can only hold for PPO, and if so, then no illegal actions expected

        if self.DEBUG and not self.discrete:
            # number of actions should be equal to number of idle drivers, since
            # idle operation is a special case of dispatch operation
            assert sum([a[1] for a in dispatch_actions]) == self.world.nodes[self.current_node_id]['info'].get_driver_num(), \
                        (dispatch_actions, self.world.nodes[self.current_node_id]['info'].get_driver_num())

        dispatch_actions_with_drivers = self.dispatch_drivers(dispatch_actions)

        t2 = time.time()
        for d in dispatch_actions_with_drivers:
            k = (self.current_node_id, d[0])
            self.this_timestep_dispatch[k] = self.this_timestep_dispatch.get(k,0) + d[1] 

        reward = self.calculate_reward(dispatch_actions_with_drivers)
        penalty = unmasked_sum * self.penalty_for_invalid_action
        reward -= penalty
        info['unmasked_penalty'] = penalty

        t3 = time.time()
        time_updated = False
        driver_time = 0
        order_time = 0
        while self.update_current_node_id() and not self.done: # while there is no drivers to manage at current iteration
            self.time += 1
            self.last_timestep_dispatch = self.this_timestep_dispatch
            for k, val in self.this_timestep_dispatch.items():
                self.overall_timestep_dispatch[k] = self.overall_timestep_dispatch.get(k,0) + val
            self.this_timestep_dispatch = {}
            time_updated = True
            self.done = self.time == self.n_intervals
            # before we update status, every car in the view should be travelling
            if self.DEBUG:
                for d in self.all_driver_list:
                    assert d.status == 0 or d.position not in self.full_to_view_ind, d
            t = time.time()
            self.driver_status_control()  # drivers that finish an order become available again.
            driver_time += time.time() - t
            t = time.time()
            if self.done:
                break
            self.last_total_orders = self.bootstrap_orders()
            self.find_non_empty_nodes()
            order_time += time.time() - t

        t4 = time.time()
        info2 = {"served_orders": self.served_orders}
        info.update(info2)
        self.update_episode_logs_each_step(reward, dispatch_actions, info)
        observation = None
        if (self.hold_observation and time_updated) or (not self.hold_observation):
            observation, driver_max, order_max = self.get_observation()
            self.update_episode_logs_once_per_obs_update(driver_max, order_max)
        else:
            self.set_onehot_grid_id(self.last_time_step_obs)

        if time_updated:
            self.last_time_step_obs = observation
            if self.DEBUG:
                logging.info("New time step: {}".format(self.time))

        if self.done:
            self.last_episode_logs = copy.deepcopy(self.episode_logs)
            self.reset_episode_logs()
        t5 = time.time()

        if self.DEBUG:
            # time increased, some drivers must avait for instructions or it is done
            self.check_consistency(time_updated) 

        info['action_mask'] = self.get_action_mask(self.current_node_id)
        info['time_updated'] = time_updated
        
        t6 = time.time()

        self.time_profile += np.array([t6-t5, t5-t4, t4-t3, t3-t2, t2-t1])
        return self.last_time_step_obs if self.hold_observation else observation, reward, self.done, info

    def get_action_mask(self, node_id):
        n = self.max_action_id+1 if self.discrete else self.action_space_shape[0]
        mask = np.ones((n), dtype=int)
        if self.include_action_mask:
            neighbors = self.get_node_neigbors_in_view(node_id)
            node_degree = len(neighbors)
            mask[node_degree:-1] = 0
        return mask.tolist()

    def update_episode_logs_once_per_obs_update(self, driver_max, order_max):
        self.episode_logs.update({
            "idle_periods": [float(d.get_not_idle_periods()) for d in self.all_driver_list],
            "env_runtime": float(np.sum(self.time_profile)),
            "last_time_step": float(self.time),
            "driver_income": [float(d.income) for d in self.all_driver_list],
            "driver_income_bounded": [float(d.get_income()) for d in self.all_driver_list], 
            "driver normalization constant": float(driver_max), 
            "order normalization constant": float(order_max)
        })
        self.episode_logs["gini"] = gini(self.episode_logs["driver_income"])

    def update_episode_logs_each_step(self, reward, dispatch_actions, info):
        self.episode_logs["number_of_idle_drivers"].append(sum([a[1] for a in dispatch_actions if a[2] <= 0]))
        assert self.served_orders == sum([a[1] for a in dispatch_actions if a[2] > 0])
        self.episode_logs["number_of_served_orders"].append(self.served_orders)
        assert info['total_orders'] >= info['served_orders']
        self.episode_logs["order_response_rates"].append(float(info['served_orders']/(info['total_orders']+0.0001)))
        self.episode_logs["nodes_with_drivers"].append(int(info['nodes_with_drivers']))
        self.episode_logs["nodes_with_orders"].append(int(info['nodes_with_orders']))
        self.episode_logs["rewards"].append(float(reward))
        self.episode_logs["total_steps"] += 1. # total calls to "step" function
        if not self.discrete:
            assert self.episode_logs["total_steps"] <= self.world_size * self.n_intervals

    def get_episode_info(self):
        if self.last_episode_logs is None:
            logging.error("No last_episode_logs available, check Gym config per_rollout. Current logs: {}".format(self.episode_logs))
        return self.last_episode_logs

    def set_orders_per_time_interval(self, orders: Tuple[int, int, int, int, float]) -> None:
        assert len(orders) > 0, "Orders can not be empty"
        self.orders_per_time_interval = {}
        max_reward = 0
        for i in range(self.n_intervals+1): # last element is not filled and used to store final results
            self.orders_per_time_interval[i] = []

        non_paying_customers = 0
        
        s = int(self.order_sampling_rate * len(orders))
        sampled_orders_index = self.random.choice(len(orders), size=s, replace=False)
        sampled_orders = list(np.array(orders)[sampled_orders_index,:])
        for order in sampled_orders:
            # all order prices are assumed to be positive (when calculating statistics on dispatching)
            assert order[4] >= 0, order
            if order[4] == 0:
                non_paying_customers += 1
            else:
                max_reward = max(max_reward, order[4])
                t = order[2] % self.n_intervals
                self.orders_per_time_interval[t].append(tuple(order))
        if non_paying_customers > 0:
            logging.warning("{} (out of {}) customers do not pay.".format(non_paying_customers, len(orders)))
        self.max_reward = max_reward

    def driver_status_control(self) -> None:
        '''
        Return drivers from traveling pool.
        Drivers are rewarded instantly after dispatching, so here we do not reward.
        '''
        assert len(self.traveling_pool[self.time - 1]) == 0, self.time - 1
        for driver in self.traveling_pool[self.time]:
            driver.set_active()
            self.world.nodes[driver.position]['info'].add_driver(driver)
        self.traveling_pool[self.time] = []

    def bootstrap_orders(self) -> int:
        '''
        Remove remaining orders and add new orders.
        Orders wait only one time interval.
        Orders are dispatched only in the view, to save time.
        
        :return: total_orders
        '''

        ## we need to go through all nodes in the network because the env can be initialized with all nodes in the view
        ## and then orders are drivers are bootstrapped over all the network.
        # for n in self.full_to_view_ind:
        if self.time % self.waiting_period == 0:
            for n in self.world.nodes(data=True):
                n[1]['info'].clear_orders()
                # self.world.nodes[n]['info'].clear_orders()

        for r in self.orders_per_time_interval[self.time]:
            if r[0] in self.full_to_view_ind:
                self.world.nodes[r[0]]['info'].add_order(r)

        return len(self.orders_per_time_interval[self.time])

    def bootstrap_drivers(self) -> None:
        """
        Assign initial distribution of drivers to nodes, in all the world, to preserve consistency.
        """
        self.all_driver_list = []

        if self.randomize_drivers:
            drivers_per_node_dict = Counter(self.random.choice(len(self.full_to_view_ind), self.n_drivers))
            drivers_per_node = [drivers_per_node_dict[i] for i in range(len(self.drivers_per_node))]
        else:
            drivers_per_node = self.drivers_per_node

        for n in self.world.nodes(data=True):
            driver_num = drivers_per_node[n[0]]
            assert driver_num >= 0
            n[1]['info'].clear_drivers()
            for i in range(driver_num):
                driver = Driver(len(self.all_driver_list), self, self.income_bound)
                if self.randomize_drivers:
                    driver.income = np.random.randint(100)
                self.all_driver_list.append(driver)
                self.driver_dict[driver.driver_id] = driver
                n[1]['info'].add_driver(driver)

    def calculate_reward(self, dispatch_actions_with_drivers: ActionList) -> float:
        '''
        Getting reward per current node, for all actions applied for that node
        :param dispatch_actions_with_drivers:
                    action list of <destination, number_of_drivers, reward, time_length, a list of driver ids assigned>

        :return: float showing the reward of the step
        '''
        if self.weight_poorest:
            assert not self.minimum_reward
            driver_ids = []
            their_income = []
            for d in self.all_driver_list:
                driver_ids.append(d.driver_id)
                their_income.append(d.get_income())
            softmax_income_list = self.softmax(their_income)
            softmax_income = dict(list(zip(driver_ids, softmax_income_list)))
            reward = 0
            ll = []
            for destination, number_of_drivers, driver_reward, time_length, driver_list in dispatch_actions_with_drivers:
                assert len(driver_list) == number_of_drivers
                reward += np.sum([driver_reward * (1-softmax_income[driver_id]) for driver_id in driver_list])
        else:
            if self.minimum_reward:
                reward = np.min([a[2] for a in dispatch_actions_with_drivers])
            else:
                reward = np.sum([a[1]*a[2] for a in dispatch_actions_with_drivers])
        if self.normalize_rewards:
            reward = reward/np.sum([a[1] for a in dispatch_actions_with_drivers])
        return reward

    def get_dispatch_actions_from_action(self, action):
        '''
        Number of dispatch actions should be equal to number of drivers in a cells.
        Dispatch actions may have destination repeated, because of possibly different price.

        :return: a list of <destination, number_of_drivers, reward, time_length>
        '''

        driver_to_order_list = self.make_order_dispatch_list_and_remove_orders()
        self.served_orders = len(driver_to_order_list)
        node = self.current_node_id
        idle_drivers = self.world.nodes[node]['info'].get_driver_num() - len(driver_to_order_list)

        actionlist = []

        neighbors = self.get_node_neigbors_in_view(node)
        if self.discrete:
            # send a single car to the destination defined by the action
            if idle_drivers > 0:

                if self.time == 0 and action >= len(neighbors) and action != self.max_action_id:
                    # for very first action mask is not passed properly in stable_baseline framework
                    # and all the actions assumed to be feasible
                    # so we hack it just by assuming that any illegal action means just staying at the same node
                    action = self.max_action_id

                assert action < len(neighbors) or action == self.max_action_id, \
                    "Received illegal action-id: {}, neigh {}, max action {}, action_mask {}, cur node {}".format(action, 
                        neighbors, self.max_action_id, self.get_action_mask(self.current_node_id), self.current_node_id)
                target_node_id = neighbors[action] if action < len(neighbors) else node # last action_id is the node itself
                actionlist = [(target_node_id, 1, -self.wc, 1)] # action is the id in action_array, i.e. id of neighbor, not a neighbor node_id
            unmasked_sum = 0 # asserting that we don't send cars outside the view (action is in neigbors)
        else:
            node_degree = len(neighbors)
            missing_dimentions = len(action) - node_degree - 1
            neighbors += [0]*missing_dimentions + [node]

            masked_action = np.copy(action)
            masked_action[node_degree:-1] = 0
            s = np.sum(masked_action)
            if s == 0:
                masked_action.fill(1. / len(masked_action))
            else:
                masked_action /= s
            unmasked_sum = np.sum(action[node_degree:-1])

            actionlist = []
            if idle_drivers > 0:
                targets = Counter(self.random.choice(neighbors, idle_drivers, p=masked_action))
                for k in targets:
                    actionlist.append((k, targets[k], -self.wc, 1))

        return driver_to_order_list + actionlist, unmasked_sum

    def get_node_neigbors_in_view(self, node):
        return [n for n in self.world.neighbors(node) if n in self.full_to_view_ind]

    def make_order_dispatch_list_and_remove_orders(self):
        '''
        Create dispatch list from orders if drivers are available,
        and remove orders from nodes.
        
        In discrete version, all the orders are dispatched anyway, because its only the idle_drivers that are managed individually
        '''
        node = self.world.nodes[self.current_node_id]['info']
        orders_to_dispatch = min([node.get_driver_num(), node.get_order_num()])
        
        dispatch_list = []
        for order in node.select_and_remove_orders(orders_to_dispatch, self.random):
            assert order[0] == node.node_id
            assert self.time == order[2] % self.n_intervals
            if self.driver_automatic_return:
                target, length = self.calculate_target_and_length_of_trip(order[1], order[3])
            else:
                target = order[1]
                length = order[3]
            assert(length > 0)
            price = order[4]
            dispatch_list.append((target, 1, price, length))

        if self.count_neighbors and (orders_to_dispatch == 0 or not self.discrete): # if discrete and there are no orders in the node, 
                                                                                    # assign to a neighbour
            if self.discrete:
                leftover_drivers = 1
            else:
                leftover_drivers = node.get_driver_num() - len(dispatch_list)

            if leftover_drivers > 0:
                neighbor_list = self.get_node_neigbors_in_view(self.current_node_id)
                self.random.shuffle(neighbor_list)
                # assign orders from neighbors if neighbors have not enough drivers
                for n in neighbor_list:
                    nnode = self.world.nodes[n]['info']
                    available_orders = max(0, nnode.get_order_num() - nnode.get_driver_num())
                    orders_to_dispatch = min(available_orders, leftover_drivers)
                    for order in nnode.select_and_remove_orders(orders_to_dispatch, self.random):
                        assert order[0] == nnode.node_id
                        assert self.time == order[2] % self.n_intervals
                        dispatch_list.append((order[1], 1, order[4], order[3]))
                        leftover_drivers -= 1
                    if leftover_drivers == 0:
                        break

        return dispatch_list

    def calculate_target_and_length_of_trip(self, target, length):
        '''
        A car is sent to target, taking length time steps. The target can be outside the current view.
        If so, find a node that is closest to the target, and assume the car "automatically" returns there
        after serving the customer.

        Length for the return trip costs one hop per iteration.

        Assuming that the source node is the current_node_id

        :return: (new target, new length), where new length is the one for the whole trip
        '''
        if target in self.full_to_view_ind:
            return target, length

        # find a path from the target to the source node, and the closest node within the view in the path
        path = nx.shortest_path(self.world, target, self.current_node_id)
        assert path[0] not in self.full_to_view_ind
        first_node_in_view = None
        checked_nodes = 0
        for n in path:
            if n in self.full_to_view_ind:
                first_node_in_view = n
            if first_node_in_view is not None:
                break
            checked_nodes += 1
        assert first_node_in_view is not None
        left_hops = len(path) - checked_nodes
        return first_node_in_view, left_hops + length

    def dispatch_drivers(self, dispatch_action_list: ActionList) -> Tuple:
        '''
        Assigns drivers from the current cell randomly to destinations according to dispatch_action_list.
        Drivers get reward instantly, as we need to have a reward per one step, not per one time interval.
        Orders are special case of dispatch actions, as well as static drivers.

        :param dispatch_action_list: a list of <destination, number_of_drivers, reward, time_length>
        :return: <destination, number_of_drivers, reward, time_length, a list of driver ids assigned>
        '''
        dispatch_actions_with_drivers = []
        node = self.world.nodes[self.current_node_id]['info']
        if self.DEBUG:
            total_cars = np.sum([a[1] for a in dispatch_action_list])    
            if not self.discrete:
                assert node.get_driver_num(), (total_cars, node.get_driver_num())
            else:
                total_cruising_cars = np.sum([a[1] for a in dispatch_action_list if a[2] <= 0])
                assert total_cruising_cars <= 1, total_cruising_cars  # the rest are orders 
        drivers = node.drivers

        i = 0
        for a in dispatch_action_list:
            assert len(a) == 4
            added_drivers = []

            for j in range(a[1]):
                d = drivers[i]
                d.update_position(a[0])
                d.set_inactive()
                arrival_time = self.time + a[3]
                assert(a[3] > 0)
                if arrival_time < self.n_intervals:
                    self.traveling_pool[arrival_time].append(d)
                else:
                    self.traveling_pool[self.n_intervals].append(d)
                i += 1

                if a[2] > 0: # if income is positive, then a customer is served
                    # otherwise it was moved as a part of cruising
                    d.inc_not_idle() # so increase non-idle time periods
                non_idle_times_increase = d.get_not_idle_periods()
                driver_total_income_increase = d.add_income(a[2])

                if self.idle_reward:
                    driver_reward = non_idle_times_increase
                else:
                    driver_reward = driver_total_income_increase

                if self.income_bound is not None:
                    # if we bound the max income of a driver, then we add each driver independently,
                    # since their income might differ depending on their history
                    dispatch_actions_with_drivers.append([a[0], 1, driver_reward, a[3], [d.driver_id]])
                added_drivers.append(d.driver_id)

            if self.income_bound is None:
                # if we don't bound ther income of a driver, then we return the action and the list of assigned drivers
                dispatch_actions_with_drivers.append([ai for ai in a] + [added_drivers])
        
        node.drivers = drivers[i:]
        if not self.discrete:
            assert node.get_driver_num() == 0
        
        return dispatch_actions_with_drivers

    def compute_remaining_drivers_and_orders(self, driver_customer_distr: Array[int, int]) -> Array[int, int]:
        '''
        Helper function
        :param driver_customer_distr: an array of shape <2, world_size>, where first column is driver distr, second - order distr
        :return: an array of shape <2, world_size> with remaining drivers, customers
        '''

        remain_drivers = driver_customer_distr[0] - driver_customer_distr[1]
        remain_drivers[remain_drivers < 0] = 0

        remain_orders = driver_customer_distr[1] - driver_customer_distr[0]
        remain_orders[remain_orders < 0] = 0

        if np.sum(remain_orders) == 0 or np.sum(remain_drivers) == 0:
            context = np.array([remain_drivers, remain_orders])
            return context

        remain_orders_1d = remain_orders.flatten()
        remain_drivers_1d = remain_drivers.flatten()

        if self.count_neighbors:
            for curr_node_id, curr_node_id_view in self.full_to_view_ind.items():
                for neighbor_id in self.get_node_neigbors_in_view(curr_node_id):
                    neighbor_id_view = self.full_to_view_ind[neighbor_id]
                    a = remain_orders_1d[curr_node_id_view]
                    b = remain_drivers_1d[neighbor_id_view]
                    remain_orders_1d[curr_node_id_view] = max(a-b, 0)
                    remain_drivers_1d[neighbor_id_view] = max(b-a, 0)
                    if remain_orders_1d[curr_node_id_view] == 0:
                        break

        context = np.array([remain_drivers_1d, remain_orders_1d])
        return context

    def update_current_node_id(self) -> bool: 
        '''
        Update node for dispatching: pick a random node with non-zero drivers.
        Since considered cells must have zero drivers - simply select random cell out of non-empty
        if no such cell available now - increase the time

        return true if we need to increase time
        '''
        if self.discrete and hasattr(self, 'current_node_id') and self.world.nodes[self.current_node_id]['info'].get_driver_num() > 0:
            # allow to try to update node when using discrete, serving a node does not clear all the drivers
            # if not all the drivers are dispatched in the current node, then don't update the current_node_id
            return False

        while len(self.non_empty_nodes) > 0:
            self.current_node_id = self.non_empty_nodes.pop()
            # can be false positive due to neighbourhood matches
            node = self.world.nodes[self.current_node_id]['info']
            if node.get_driver_num() > 0:
                # accept the choice and return a signal that there are non-served nodes

                # sort drivers in the order they will be served (must be well-defined at any time, since it is used in taxi_env_batch)
                if self.poorest_first:
                    drivers_with_income = [(d, d.get_income()) for d in node.drivers]
                    drivers_with_income.sort(key=lambda x: x[1])
                    node.drivers = [x[0] for x in drivers_with_income]
                else:
                    self.random.shuffle(node.drivers)

                return False
        # return that it is time to update the time step
        return True

    def find_non_empty_nodes(self) -> None:
        self.non_empty_nodes = [n for n, _ in self.full_to_view_ind.items() if self.world.nodes[n]['info'].get_driver_num() > 0]
        self.random.shuffle(self.non_empty_nodes)

    def get_driver_and_order_distr(self) -> Array[int]:
        next_state = np.empty((2, len(self.full_to_view_ind)))
        for n, _ in self.full_to_view_ind.items():
            node = self.world.nodes[n]['info']
            next_state[0, self.full_to_view_ind[node.node_id]] = node.get_driver_num()
            next_state[1, self.full_to_view_ind[node.node_id]] = node.get_order_num()
        return next_state

    def get_observation(self):
        '''
        Returns observation (state) of size (4*world_size or 5*world_size) + n_intervals
        :return: observation, driver_max, order_max (for recovering full driver distribution)
            Observation:
            <driver distr, order distr, idle_drivers, one-hot time, 
                one-hot location, min income (or idle times) (optional)>
        '''
        view_size = len(self.full_to_view_ind)
        next_state = self.get_driver_and_order_distr()
        m = np.max(next_state, axis=1)
        driver_max = m[0]
        order_max = m[1]

        time_and_grid_one_hot = np.zeros((self.n_intervals + view_size))
        time_and_grid_one_hot[self.time % self.n_intervals] = 1 # the very last moment (when its "done") is the first time interval of the next epoch

        idle_drivers_per_node = next_state[0, :] - next_state[1, :]
        idle_drivers_per_node[idle_drivers_per_node < 0] = 0

        max_idle_driver = np.max(idle_drivers_per_node)
        if max_idle_driver > 0:
            idle_drivers_per_node_normalized = idle_drivers_per_node / max_idle_driver
        else:
            idle_drivers_per_node_normalized = np.array(idle_drivers_per_node)

        if self.DEBUG:
            assert (idle_drivers_per_node_normalized >= 0).all()
            assert np.max(idle_drivers_per_node_normalized) == 1 or np.max(idle_drivers_per_node_normalized) == 0

        obs_components = [
            next_state[0, :], 
            next_state[1, :], 
            idle_drivers_per_node_normalized, 
            time_and_grid_one_hot
        ]

        if self.include_income_to_observation:
            obs_components.append(self.get_income_per_node(idle_drivers_per_node))

        observation = np.concatenate(obs_components)
        self.set_onehot_grid_id(observation)

        observation[:view_size] /= max(driver_max, 1)
        observation[view_size:2*view_size] /= max(order_max, 1)
        
        if self.DEBUG:
            assert (observation >= 0).all() and (observation <= 1).all()
        return observation, driver_max, order_max

    def set_onehot_grid_id(self, observation) -> None:
        view_size = len(self.full_to_view_ind)
        offset = 3*view_size+self.n_intervals
        ind = self.full_to_view_ind[self.current_node_id]
        observation[offset: offset + view_size] = 0
        observation[offset + ind] = 1

    def get_income_per_node(self, idle_drivers_per_node_in_view):
        """
        :returns: a normalized vector of incomes of idle drivers in the whole world
        """
        if self.discrete:
            income = np.array([d.income for d in self.all_driver_list])
        else:
            view_size = len(self.full_to_view_ind)
            income = np.zeros(view_size)
            for n, view_ind in self.full_to_view_ind.items():
                node = self.world.nodes[n]['info']
                if self.idle_reward == False:
                    if self.income_bound is not None:
                        driver_incomes = [max(0,self.income_bound - d.income) for d in node.drivers]
                    else:
                        driver_incomes = [d.income for d in node.drivers]
                else:
                    driver_incomes = [d.get_not_idle_periods() for d in node.drivers]

                if self.poorest_first:
                    driver_incomes = sorted(driver_incomes)[-int(idle_drivers_per_node_in_view[view_ind]):] # idle_drivers_per_node is of a size of view
                    
                income[view_ind] = 0 if len(driver_incomes) == 0 else np.min(driver_incomes) # alternatively, np.mean. for discrete min is important because it shows who's next to go

        # normalization. Note that income might be negative
        income -= np.min(income)
        if np.sum(income) > 0:
            income = income / np.max(income)

        assert np.min(income) == 0
        assert np.max(income) == 1 or np.sum(income) == 0
        return income

    def seed(self, seed):
        self.random = np.random.RandomState(seed)

    @staticmethod
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def get_min_revenue(self):
        return np.min([d.get_income() for d in self.all_driver_list])

    def get_total_revenue(self):
        return np.sum([d.get_income() for d in self.all_driver_list])

    def get_min_idle(self):
        return np.min([d.get_not_idle_periods() for d in self.all_driver_list])

    def get_total_idle(self):
        return np.sum([d.get_not_idle_periods() for d in self.all_driver_list])

    def get_time(self):
        return self.time
    
    def get_resets(self):
        return self.number_of_resets

    def get_action_space_shape(self):
        return self.action_space_shape

    def get_observation_space_shape(self):
        return self.observation_space_shape

    def get_view_size(self):
        return len(self.full_to_view_ind)

    def get_n_intervals(self):
        return self.n_intervals

    def set_income_bound(self, bound):
        self.income_bound = bound
        for d in self.all_driver_list:
            d.income_bound = bound

    def check_consistency(self, time_updated: bool):
        # this is run at the end of a step
        assert len(self.all_driver_list) == self.n_drivers
        free_drivers = sum([n[1]['info'].get_driver_num() for n in self.world.nodes(data=True)])
        busy_drivers = sum([len(self.traveling_pool[i]) for i in range(self.n_intervals+1)])
        assert sum([len(self.traveling_pool[i]) for i in range(self.time)]) == 0
        assert self.n_drivers == free_drivers + busy_drivers, (self.n_drivers, free_drivers, busy_drivers)
        for i in range(self.n_intervals+1):
            for d in self.traveling_pool[i]:
                assert d.status == 0
        for n in self.world.nodes(data=True):
            for d in n[1]['info'].drivers:
                assert d.status == 1
                assert d.position == n[0]
                if self.income_bound is not None:
                    assert d.income_bound == self.income_bound

        # all orders should be bootstraped, except for the last time step
        if time_updated and self.time < self.n_intervals:
            expected_orders = len(self.orders_per_time_interval[self.time])
            for n in self.world.nodes():
                if n not in self.full_to_view_ind:
                    assert self.world.nodes[n]['info'].get_order_num() == 0
            number_of_orders = sum([n[1]['info'].get_order_num() for n in self.world.nodes(data=True)])
            # some orders in orders_per_time_interval may not be assigned to the nodes if out of the view
            assert expected_orders >= number_of_orders, (expected_orders, number_of_orders)

        # Some nodes might have been considered in this time step already, some might not be.
        # Since for each considered node we update status to inactive, then
        # Those who are active have not been processed in the current time step yet
        for d in self.all_driver_list:
            if d.status == 1: # driver hasn't moved in this time interval (their node hasn't been processed)
                # Their income should be consistent to the current time interval
                # float error added (if negative balance accumulates)
                assert d.income - (-self.wc*(self.time - d.get_not_idle_periods())) >= -0.00001, "Driver's income is not consistent: {}".format(d)

    def render(self, mode='fig'):
        if self.time == self.n_intervals-1:
            return self.render_dispatch(mode, self.overall_timestep_dispatch)
        else:
            return self.render_dispatch(mode, self.last_timestep_dispatch)

    def render_dispatch(self, mode, dispatch_info):
        '''
        Return a single image
        A mode where an image is plotted in a popup window is not implemented
        '''
        fig = plt.figure(figsize=(10,10),dpi=150)
        ax = fig.gca()
        ax.axis('off')

        # if coords are not available
        for n in self.world.nodes(data=True):
            if 'coords' not in n[1]:
                NotImplementedError("Implement adding coords if missing")

        view_size = len(self.full_to_view_ind)
        x = np.zeros((2, view_size))

        observation, driver_max, order_max = self.get_observation()

        c = []
        c_border = []
        for i, _ in self.full_to_view_ind.items():
            x[0, i] = self.world.nodes[i]['coords'][0]
            x[1, i] = self.world.nodes[i]['coords'][1]
            # dots are number of orders, borders around dots are drivers
            c.append(observation[i+view_size])
            c_border.append(observation[i])

        cmap = matplotlib.cm.get_cmap('Greens')
        cmap_e = matplotlib.cm.get_cmap('cool')
        c_border = [cmap(j) for j in c_border]

        plt.scatter(x[0,:], x[1,:], c=c, s=200, linewidth=3, edgecolors=c_border, cmap="Reds")
        if len(dispatch_info) == 0:
            edge_norm = 1
        else:
            edge_norm = max([val for k, val in dispatch_info.items()])
        for e in self.world.edges():
            if e[0] not in self.full_to_view_ind or e[1] not in self.full_to_view_ind:
                continue

            flow = dispatch_info.get((e[0], e[1]),0) - dispatch_info.get((e[1], e[0]),0)
            if flow >= 0:
                c1 = self.world.nodes[e[0]]['coords']
                c2 = self.world.nodes[e[1]]['coords']
            else:
                c1 = self.world.nodes[e[1]]['coords']
                c2 = self.world.nodes[e[0]]['coords']
                flow *= -1

            edge_w = flow / edge_norm
            if flow == 0:
                plt.plot([c1[0],c2[0]],[c1[1],c2[1]],color="grey")
            else:
                plt.arrow(c1[0], c1[1], c2[0]-c1[0], c2[1]-c1[1], color=cmap_e(edge_w), lw=0.7,
                            length_includes_head=True, head_width=0.07, head_length=0.4) #x,y,dx,dy

        # plt.title("t={}".format(self.time)) -- do not plot any titles, so that to add them in a peper

        if mode == "fig":
            return fig
       
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        s, (width, height) = canvas.print_to_buffer()

        # Option 2a: Convert to a NumPy array
        X = np.frombuffer(s, np.uint8).reshape((height, width, 4))
        plt.close(fig)
        return X

    def sync(self, another_env):
        '''
        Syncronize status of this environment with another environment build on the same data.
        Take care only about nodes that are in the view
        '''
        # update each driver
        for d in self.all_driver_list:
            d.sync(another_env.driver_dict[d.driver_id])

        # update traveling pool
        for time, driver_list in another_env.traveling_pool.items():
            self.traveling_pool[time] = [self.driver_dict[d.driver_id] for d in driver_list]

        self.time = another_env.time
        self.episode_logs = copy.deepcopy(another_env.episode_logs)
        self.non_empty_nodes = another_env.non_empty_nodes
        self.discrete = another_env.discrete
        self.current_node_id = another_env.current_node_id
        self.done = another_env.done
        self.last_episode_logs = another_env.last_episode_logs
        self.number_of_resets = another_env.number_of_resets
        self.last_timestep_dispatch = copy.deepcopy(another_env.last_timestep_dispatch)
        for n in self.world.nodes(data=True):
            n[1]['info'].sync(another_env.world.nodes[n[0]]['info'], self.driver_dict)

        # syncronize sampled orders (because they might have been sampled differently)
        self.orders_per_time_interval = copy.deepcopy(another_env.orders_per_time_interval)

    def set_view(self, nodes):
        """
        This set a view of this environment so that only a subset of nodes are visible from outside of this class.
        This is used in SplitSolver to build subsolvers.

        :param nodes: a subset of nodes to preserve in the world.
        """
        self.full_to_view_ind = {}
        nodes = list(nodes)
        self.view_to_full_ind = list(nodes)
        for i in range(len(nodes)):
            self.full_to_view_ind[nodes[i]] = i

        # create subgraph in order to calculate node degrees correctly
        g = self.world.subgraph([k for k, _ in self.full_to_view_ind.items()])
        max_degree = np.max([d[1] for d in nx.degree(g) if d[0] in self.full_to_view_ind])
        self.set_action_and_observation_space(max_degree, len(self.full_to_view_ind), self.n_intervals)
        self.init()

    def get_reset_info(self) -> Dict:
        obs, driver_max, order_max = self.get_observation()
        info = {"served_orders": 0,
                "driver normalization constant": self.episode_logs["driver normalization constant"], # required for cA2C
                "order normalization constant": self.episode_logs["order normalization constant"],
                }
        if self.include_action_mask:
            info['action_mask'] = self.get_action_mask(self.current_node_id)
        return info

    def auto_update_income_bound(self):
        if self.bounded_income:
            b = np.mean([d.income for d in self.all_driver_list])
            if b > self.best_income_so_far:
                logging.info("Updated best income to {}".format(b))
                self.best_income_so_far = b
