import pytest
import imageio
import os, sys
import networkx as nx
import numpy as np
from gym_taxi.envs.taxi_env_batch import TaxiEnvBatch
from copy import deepcopy

class TestTaxiEnvBatch:

    def test_strict_policy(self):
        g = nx.Graph()
        g.add_edges_from([(0,1),(1,2),(2,3),(2,0)])
        orders = [(0,3,0,2,0.75)]
        drivers = [1,2,0,4]
        env = TaxiEnvBatch(g, orders, 1, drivers, 4, 0.5, 
                            normalize_rewards=False, 
                            include_income_to_observation=True)

        # first iteration: one car from 0 sent to 4; two cars from 1 sent to 0; 4 cars stay in 4
        action = np.zeros(4*4) # maxdegree = 3, action space = 4
        action[4] = 1
        action[-1] = 1
        observation, reward, done, info = env.step(action)
        observation, reward, done, info = env.step(action) # first step is a cold start

        assert done == False
        assert env.world.nodes[0]['info'].get_driver_num() == 2
        assert env.world.nodes[1]['info'].get_driver_num() == 0
        assert env.world.nodes[2]['info'].get_driver_num() == 0
        assert env.world.nodes[3]['info'].get_driver_num() == 4
        assert len(env.traveling_pool[2]) == 1
        assert env.traveling_pool[2][0].income == 0.75
        assert env.traveling_pool[2][0].status == 0

        observation, reward, done, info = env.step(action)
        assert done == False

        observation, reward, done, info = env.step(action)
        assert done == False

        observation, reward, done, info = env.step(action)
        assert done == True

    
    def test_view(self):
        g = nx.Graph()
        g.add_edges_from([(0,1),(1,2),(2,3),(3,4),(1,5)])
        nx.set_node_attributes(g, {0: (0,1), 1: (0,2), 2: (0,3), 3: (0,4), 4: (0,5), 5: (1,1)}, name="coords")
        orders = [(3,2,2,3,3)]
        drivers = np.array([1,1,1,1,1,1])
        action = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0], dtype=float)

        env = TaxiEnvBatch(g, orders, 1, drivers, 10)
        env.set_view([2,3,4])
        obs = env.get_global_observation()

        # check observation space and content
        assert env.global_observation_space_shape == obs.shape
        view_size = 3
        assert env.global_observation_space_shape == (view_size*3 + 10,) # default income is not included, so its <driver, order, idle, time_id>
        # degree of 1 is 3, but of the rest is 2. So it should be 2 + 1 (staying action), multiplies by number of nodes
        assert env.global_action_space_shape == (9,) 
        assert env.current_node_id in [2,3,4]

        obs, rew, done, info = env.step(action)
        obs, rew, done, info = env.step(action) # first step is a cold start
        assert env.current_node_id in [2,3]
        assert (obs[:view_size] == np.array([0.5,1,0])).all() # 0.5 because action takes into consideration only nodes in view, so there is no left-step from the 2nd (most left) node
        assert (obs[view_size:2*view_size] == np.array([0,0,0])).all()
        assert (obs[2*view_size:3*view_size] == np.array([0.5,1,0])).all()
        # next time iteration should happen
        assert env.time == 1
        assert (obs[2*view_size:3*view_size] == np.array([0.5,1,0])).all()
        assert (obs[3*view_size:] == np.array([0,1,0,0,0,0,0,0,0,0])).all()
        assert [d.position for d in env.all_driver_list] == [0, 1, 3, 2, 3, 5]

    def test_fully_collaborative(self):
        g = nx.Graph()
        g.add_edges_from([(0,1),(1,2),(2,3),(3,4)])
        nx.set_node_attributes(g, {0: (0,1), 1: (0,2), 2: (0,3), 3: (0,4), 4: (0,5)}, name="coords")
        orders = [(2,0,10,3,3)] # <source, destination, time, length, price>
        drivers = np.array([1,0,0,0,1])
        env = TaxiEnvBatch(g, orders, 1, drivers, 12, \
                            discrete=True, fully_collaborative=True, include_action_mask=True) # 1 is order sampling rate
        _, _, _, info = env.step(0)

        # try all possible actions, and check all outcomes
        mask = info["action_mask"]
        possible_outcomes = set()
        for i in range(len(mask)):
            if mask[i] == 0:
                continue
            env2 = deepcopy(env)
            obs, _, _, _ = env2.step(i)
            drivers = obs[:5]
            possible_outcomes.add(tuple(drivers.tolist()))

        assert possible_outcomes == set([(1,0,0,0,1), (0,1,0,0,1), (1,0,0,1,0), (0,1,0,1,0)])