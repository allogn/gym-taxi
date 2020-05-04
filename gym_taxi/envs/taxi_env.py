import gym
from typing import Tuple, Dict, List
from nptyping import Array
from gym import spaces
import numpy as np
import networkx as nx
import copy
import math
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvasAgg
import matplotlib
matplotlib.use('Agg')

import logging
logger = logging.getLogger(__name__)

from gym_taxi.envs.node import Node
from gym_taxi.envs.driver import Driver

ActionList = List[Tuple[int, int, float, int]]

class TaxiEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array', 'fig']}

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
                 include_prematch: bool = False,
                 seed: int = None) -> None: 
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
        :param include_income_to_observation: observation includes car income distribution
        :param poorest_first: assignment strategy such that poorest drivers get assigned first
        :param idle_reward: reward by the time a driver is idle rather than by income
        :param include_prematch: include the number of idle drivers into observations (together with the total number)
        :param seed: seed for a random state

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
        self.DEBUG = True
        self.number_of_resets = 0

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
        self.reward_bound = reward_bound
        self.weight_poorest = weight_poorest 
        self.count_neighbors = count_neighbors
        self.normalize_rewards = normalize_rewards
        self.include_income_to_observation = include_income_to_observation

        self.drivers_per_node = np.array(drivers_per_node)
        assert self.drivers_per_node.dtype == int
        assert self.drivers_per_node.shape == (self.world_size,)

        self.all_driver_list = []
        self.driver_dict = {}
        self.set_orders_per_time_interval(orders)

        # set action space
        self.max_degree = np.max([d[1] for d in nx.degree(self.world)])
        self.action_space_shape = (self.max_degree+1,)
        self.action_space = spaces.Box(low=0, high=1, shape=self.action_space_shape)

        # set observation space: distribution of drivers, orders, current cell and current time
        self.observation_space_shape = 3*self.world_size + self.n_intervals
        if self.include_income_to_observation:
            # optionally with mean,avg,min incomes
            self.observation_space_shape += 3
        self.observation_space_shape = (self.observation_space_shape,)

        self.observation_space = spaces.Box(low=0, high=1, shape=self.observation_space_shape)
        self.reset_episode_logs()
        self.last_episode_logs = None
        self.init()

    def reset_episode_logs(self):
        self.episode_logs = {
            'order_response_rates': [],
            'number_of_idle_drivers': [],
            'number_of_served_orders': [],
            'nodes_with_drivers': [],
            'nodes_with_orders': [],
            'rewards': [],
            "total_steps": 0.
        }

    def init(self):
        # this function is separated from reset() because taxi_env_batch overrides reset
        # and initialization of taxi_env_batch otherwise produce error due to wrong call to wrong reset
        self.time = 0
        self.total_steps = 0
        self.done = False
        self.traveling_pool = {} # lists of drivers in traveling state, indexed by arrival time
        for i in range(self.n_intervals+1): # the last element contains all drivers arriving out of the episode
            self.traveling_pool[i] = []
        self.bootstrap_orders()
        self.bootstrap_drivers()
        self.update_current_node_id()
        self.last_timestep_dispatch = {} # container for dispatch actions, used in plotting
        self.this_timestep_dispatch = {} # extra container so that each time we can plot for t-1, and save for this t

        self.reset_episode_logs()

    def reset(self) -> Array[int]:
        self.number_of_resets += 1
        self.init()
        obs, _, _ = self.get_observation()
        return obs

    def get_number_of_resets(self):
        return self.number_of_resets

    def get_reset_info(self):
        obs, driver_max, order_max = self.get_observation()
        return {"served_orders": 0, 
                "driver normalization constant": driver_max, 
                "order normalization constant": order_max}

    def step(self, action: Array[float]) -> Tuple[Array[int], float, bool, Dict]:
        """
        :returns: observation, reward, done, info
        """
        if self.done:
            raise Exception("Trying to step terminated environment. Call reset first.")
        assert self.time < self.n_intervals
        assert action.shape == self.action_space.shape, (action.shape, self.action_space.shape)

        info = {}
        info["total_orders"] = np.sum([n[1]['info'].get_order_num() for n in self.world.nodes(data=True)])
        info["nodes_with_orders"] = np.sum([1 for n in self.world.nodes(data=True) if n[1]['info'].get_order_num() > 0])
        info["nodes_with_drivers"] = np.sum([1 for n in self.world.nodes(data=True) if n[1]['info'].get_driver_num() > 0])

        dispatch_actions = self.get_dispatch_actions_from_action(action)

        # number of actions should be equal to number of idle drivers, since
        # idle operation is a special case of dispatch operation
        assert sum([a[1] for a in dispatch_actions]) == self.world.nodes[self.current_node_id]['info'].get_driver_num(), \
                    (dispatch_actions, self.world.nodes[self.current_node_id]['info'].get_driver_num())

        dispatch_actions_with_drivers = self.dispatch_drivers(dispatch_actions)

        for d in dispatch_actions_with_drivers:
            k = (self.current_node_id, d[0])
            self.this_timestep_dispatch[k] = self.this_timestep_dispatch.get(k,0) + d[1] 

        reward = self.calculate_reward(dispatch_actions_with_drivers)

        time_updated = False
        while self.update_current_node_id() and not self.done: # while there is no drivers to manage at current iteration
            self.time += 1
            self.last_timestep_dispatch = self.this_timestep_dispatch
            self.this_timestep_dispatch = {}
            time_updated = True
            self.done = self.time == self.n_intervals
            # before we update status, every car should be travelling
            if self.DEBUG:
                for d in self.all_driver_list:
                    assert d.status == 0, d
            self.driver_status_control()  # drivers that finish an order become available again.
            if self.done:
                break
            self.bootstrap_orders()
        

        observation, driver_max, order_max = self.get_observation()
        non_idle_periods = [float(d.get_not_idle_periods()) for d in self.all_driver_list]
        
        info2 = {"served_orders": self.served_orders, 
                "driver normalization constant": driver_max, 
                "order normalization constant": order_max,
                "idle_reward": float(np.mean(non_idle_periods)),
                "min_idle": float(np.min(non_idle_periods))}
        info.update(info2)

        self.episode_logs["number_of_idle_drivers"].append(sum([a[1] for a in dispatch_actions if a[2] <= 0]))
        assert self.served_orders == sum([a[1] for a in dispatch_actions if a[2] > 0])
        self.episode_logs["number_of_served_orders"].append(self.served_orders)
        assert info['total_orders'] >= info['served_orders']
        self.episode_logs["order_response_rates"].append(float(info['served_orders']/(info['total_orders']+0.0001)))
        self.episode_logs["nodes_with_drivers"].append(int(info['nodes_with_drivers']))
        self.episode_logs["nodes_with_orders"].append(int(info['nodes_with_orders']))
        self.episode_logs["driver_income"] = [float(d.income) for d in self.all_driver_list]
        self.episode_logs["driver_income_bounded"] = [float(d.get_income()) for d in self.all_driver_list]
        self.episode_logs["rewards"].append(float(reward))
        self.episode_logs["total_steps"] += 1. # total calls to "step" function
        self.episode_logs["idle_periods"] = non_idle_periods # distribution of non-idle-periods over drivers at the last iteration

        if self.done:
            self.last_episode_logs = copy.deepcopy(self.episode_logs)

        if self.DEBUG:
            # time increased, some drivers must avait for instructions or it is done
            self.check_consistency(time_updated) 
        return observation, reward, self.done, info

    def get_episode_info(self):
        return self.last_episode_logs

    def set_orders_per_time_interval(self, orders: Tuple[int, int, int, int, float]) -> None:
        self.orders_per_time_interval = {}
        max_reward = 0
        for i in range(self.n_intervals+1): # last element is not filled and used to store final results
            self.orders_per_time_interval[i] = []
        for order in orders:
            max_reward = max(max_reward, order[4])
            t = order[2] % self.n_intervals
            self.orders_per_time_interval[t].append(tuple(order))
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

    def bootstrap_orders(self) -> None:
        '''
        Remove remaining orders and add new orders.
        In the future possibly for orders to "wait" longer #IDEA
        '''
        for n in self.world.nodes(data=True):
            n[1]['info'].clear_orders()
            l = [r for r in self.orders_per_time_interval[self.time] if r[0] == n[0]]
            s = int(self.order_sampling_rate * len(l))
            random_orders_ind = self.random.choice(np.arange(len(l)), size=s)
            random_orders = np.array(l)[random_orders_ind]
            n[1]['info'].add_orders(random_orders)

    def bootstrap_drivers(self) -> None:
        self.all_driver_list = []
        for n in self.world.nodes(data=True):
            driver_num = self.drivers_per_node[n[0]]
            assert driver_num >= 0
            n[1]['info'].clear_drivers()
            for i in range(driver_num):
                driver = Driver(len(self.all_driver_list), self, self.reward_bound)
                self.all_driver_list.append(driver)
                self.driver_dict[driver.driver_id] = driver
                n[1]['info'].add_driver(driver)

    def calculate_reward(self, dispatch_actions_with_drivers: ActionList) -> float:
        '''
        Getting reward per one node, for all actions applied for that node
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

    def get_dispatch_actions_from_action(self, action: Array[float]) -> ActionList:
        '''
        Number of dispatch actions should be equal to number of drivers in a cells.
        Dispatch actions may have destination repeated, because of possibly different price.

        :return: a list of <destination, number_of_drivers, reward, time_length>
        '''

        driver_to_order_list = self.make_order_dispatch_list_and_remove_orders()
        node = self.current_node_id
        idle_drivers = self.world.nodes[node]['info'].get_driver_num() - len(driver_to_order_list)

        self.served_orders = len(driver_to_order_list)

        node_degree = self.world.degree(node)
        neighbors = list(self.world.neighbors(node))
        missing_dimentions = len(action) - node_degree - 1
        neighbors += [0]*missing_dimentions + [node]

        masked_action = np.copy(action)
        masked_action[node_degree:-1] = 0
        if np.sum(masked_action) == 0:
            masked_action = np.ones(masked_action.shape)
        masked_action /= np.sum(masked_action)

        actionlist = []
        if idle_drivers > 0:
            targets = Counter(self.random.choice(neighbors, idle_drivers, p=masked_action))
            for k in targets:
                actionlist.append((k, targets[k], -self.wc, 1))

        return driver_to_order_list + actionlist

    def make_order_dispatch_list_and_remove_orders(self):
        '''
        Create dispatch list from orders if drivers are available,
        and remove orders from nodes.
        '''
        node = self.world.nodes[self.current_node_id]['info']
        orders_to_dispatch = min([node.get_driver_num(), node.get_order_num()])
        dispatch_list = []
        for order in node.select_and_remove_orders(orders_to_dispatch):
            assert order[0] == node.node_id
            assert self.time == order[2] % self.n_intervals
            target = order[1]
            length = order[3]
            assert(length > 0)
            price = order[4]
            dispatch_list.append((target, 1, price, length))

        if self.count_neighbors:
            leftover_drivers = node.get_driver_num() - len(dispatch_list)
            if leftover_drivers > 0:
                neighbor_list = list(self.world.neighbors(self.current_node_id))
                self.random.shuffle(neighbor_list)
                # assign orders from neighbors if neighbors have not enough drivers
                for n in neighbor_list:
                    nnode = self.world.nodes[n]['info']
                    available_orders = max(0, nnode.get_order_num() - nnode.get_driver_num())
                    orders_to_dispatch = min(available_orders, leftover_drivers)
                    for order in nnode.select_and_remove_orders(orders_to_dispatch):
                        assert order[0] == nnode.node_id
                        assert self.time == order[2] % self.n_intervals
                        dispatch_list.append((order[1], 1, order[4], order[3]))
                        leftover_drivers -= 1
                    if leftover_drivers == 0:
                        break

        return dispatch_list

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
        assert np.sum([a[1] for a in dispatch_action_list]) == node.get_driver_num(), node.get_driver_num()
        drivers = list(node.drivers)

        if self.poorest_first:
            drivers_with_income = [(d, d.get_income()) for d in drivers]
            drivers_with_income.sort(key=lambda x: x[1])
            drivers = [x[0] for x in drivers_with_income]
        else:
            self.random.shuffle(drivers)

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

                if self.reward_bound is not None:
                    # if we bound the max income of a driver, then we add each driver independently,
                    # since their income might differ depending on their history
                    dispatch_actions_with_drivers.append([a[0], 1, driver_reward, a[3], [d.driver_id]])
                added_drivers.append(d.driver_id)

            if self.reward_bound is None:
                # if we don't bound ther income of a driver, then we return the action and the list of assigned drivers
                dispatch_actions_with_drivers.append([ai for ai in a] + [added_drivers])
        node.clear_drivers()
        return dispatch_actions_with_drivers

    def compute_remaining_drivers_and_orders(self, driver_customer_distr: Array[int]) -> Array[int]:
        '''
        former step_pre_order_assigin from CityReal
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
            for curr_node_id in self.world.nodes():
                for neighbor_id in self.world.neighbors(curr_node_id):
                    a = remain_orders_1d[curr_node_id]
                    b = remain_drivers_1d[neighbor_id]
                    remain_orders_1d[curr_node_id] = max(a-b, 0)
                    remain_drivers_1d[neighbor_id] = max(b-a, 0)
                    if remain_orders_1d[curr_node_id] == 0:
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
        non_empty_nodes = [n[0] for n in self.world.nodes(data=True) if n[1]['info'].get_driver_num() > 0]
        if len(non_empty_nodes) == 0:
            return True
        self.current_node_id = self.random.choice(non_empty_nodes)
        return False

    def get_driver_and_order_distr(self) -> Array[int]:
        next_state = np.zeros((2, self.world_size))
        for n in self.world.nodes(data=True):
            node = n[1]['info']
            next_state[0, node.node_id] = node.get_driver_num()
            next_state[1, node.node_id] = node.get_order_num()
        return next_state

    def get_observation(self):
        '''
        :return: feature vector consisting of
            <driver distr, order distr, one-hot time, 
                one-hot location, mean/min/max income/idle (optional)>
            driver_max and order_max (for recovering full driver distribution)
        '''
        next_state = self.get_driver_and_order_distr()
        # context = self.compute_remaining_drivers_and_orders(next_state)

        time_one_hot = np.zeros((self.n_intervals))
        time_one_hot[self.time % self.n_intervals] = 1 # the very last moment (when its "done") is the first time interval of the next epoch

        onehot_grid_id = np.zeros((self.world_size))
        onehot_grid_id[self.current_node_id] = 1

        observation = np.zeros(self.observation_space_shape)
        observation[:self.world_size] = next_state[0, :]
        observation[self.world_size:2*self.world_size] = next_state[1, :]
        #observation[2*self.world_size:3*self.world_size] = context[0,:]
        #observation[3*self.world_size:4*self.world_size] = context[1,:]
        observation[2*self.world_size:2*self.world_size+self.n_intervals] = time_one_hot
        observation[2*self.world_size+self.n_intervals:3*self.world_size+self.n_intervals] = onehot_grid_id
        if self.include_income_to_observation:
            observation[-3:] = self.get_income_per_node(self.current_node_id)

        driver_max = np.max(observation[:self.world_size])
        order_max =  np.max(observation[self.world_size:2*self.world_size])
        if driver_max == 0:
            driver_max = 1
        if order_max == 0:
            order_max = 1
        observation[:self.world_size] /= driver_max
        observation[self.world_size:2*self.world_size] /= order_max
        
        assert (observation >= 0).all() and (observation <= 1).all()
        return observation, driver_max, order_max

    def get_income_per_node(self, node_id):
        income = np.zeros(3)
        if self.idle_reward == False:
            driver_incomes = [d.get_income()/(self.time+1) for d in self.world.nodes[node_id]['info'].drivers]
            # normalization by time+1 because, e.g., during first steps time is not increased, but drivers earn already
            min_income = -self.wc
            max_income = self.max_reward
            d = max_income - min_income

            if self.world.nodes[node_id]['info'].get_driver_num() == 0:
                driver_incomes = [0]

            income[-3] = (np.mean(driver_incomes) - min_income)/d
            income[-2] = (np.min(driver_incomes) - min_income)/d
            income[-1] = (np.max(driver_incomes) - min_income)/d
        else:
            driver_idle = [d.get_not_idle_periods()/(self.time+1) for d in self.world.nodes[node_id]['info'].drivers]

            if self.world.nodes[node_id]['info'].get_driver_num() == 0:
                driver_incomes = [0]

            income[-3] = (np.mean(driver_idle))
            income[-2] = (np.min(driver_idle))
            income[-1] = (np.max(driver_idle))
        assert np.min(income[-3:]) >= 0, income[-3:]
        assert np.max(income[-3:]) <= 1, income[-3:]
        return income

    def seed(self, seed = None):
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

    def get_action_space_shape(self):
        return self.action_space_shape

    def get_observation_space_shape(self):
        return self.observation_space_shape

    def set_income_bound(self, bound):
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
                if self.reward_bound is not None:
                    assert d.income_bound == self.reward_bound

         # all orders should be bootstraped, except for the last time step
            expected_orders = 0
        if time_updated and self.time < self.n_intervals:
            for n in self.world.nodes(data=True):
                l = [r for r in self.orders_per_time_interval[self.time] if r[0] == n[0]]
                expected_orders += int(self.order_sampling_rate * len(l))
            number_of_orders = sum([n[1]['info'].get_order_num() for n in self.world.nodes(data=True)])
            assert expected_orders == number_of_orders, (expected_orders, number_of_orders)

        # Some nodes might have been considered in this time step already, some might not be.
        # Since for each considered node we update status to inactive, then
        # Those who are active have not been processed in the current time step yet
        for d in self.all_driver_list:
            if d.status == 1: # driver hasn't moved in this time interval (their node hasn't been processed)
                # Their income should be consistent to the current time interval
                assert d.income >= -self.wc*(self.time - d.get_not_idle_periods()), d

    def render(self, mode='rgb_array'):
        '''
        Return a single image
        A mode where an image is plotted in a popup window is not implemented
        '''
        fig = plt.figure()
        ax = fig.gca()
        ax.axis('off')

        # if coords are not available
        for n in self.world.nodes(data=True):
            if 'coords' not in n[1]:
                NotImplementedError("Implement adding coords if missing")

        x = np.zeros((2, self.world_size))

        observation, driver_max, order_max = self.get_observation()

        c = []
        c_border = []
        for i in range(self.world_size):
            x[0, i] = self.world.nodes[i]['coords'][0]
            x[1, i] = self.world.nodes[i]['coords'][1]
            # dots are number of orders, borders around dots are drivers
            c.append(observation[i+self.world_size])
            c_border.append(observation[i])

        cmap = matplotlib.cm.get_cmap('Greens')
        cmap_e = matplotlib.cm.get_cmap('cool')
        c_border = [cmap(j) for j in c_border]

        plt.scatter(x[0,:], x[1,:], c=c, s=200, linewidth=3, edgecolors=c_border, cmap="Reds")
        if len(self.last_timestep_dispatch) == 0:
            edge_norm = 1
        else:
            edge_norm = max([val for k, val in self.last_timestep_dispatch.items()])
        for e in self.world.edges():

            flow = self.last_timestep_dispatch.get((e[0], e[1]),0) - self.last_timestep_dispatch.get((e[1], e[0]),0)
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
                            length_includes_head=True, head_width=0.04, head_length=0.4) #x,y,dx,dy

        plt.title("t={}".format(self.time))

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
        Syncronize status of this environment with another environment build on the same data
        '''
        # update each driver
        for d in self.all_driver_list:
            d.sync(another_env.driver_dict[d.driver_id])

        # update traveling pool
        for time, driver_list in another_env.traveling_pool.items():
            self.traveling_pool[time] = [self.driver_dict[d.driver_id] for d in driver_list]

        self.time = another_env.time
        self.episode_logs = copy.deepcopy(another_env.episode_logs)
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
        self.view = set(nodes)
        # self.subgraph_ind_old_to_new = {}
        # self.subgraph_ind_new_to_old = nodes
        # for i in range(len(nodes)):
        #     self.subgraph_node_index[nodes[i]] = i