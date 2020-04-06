import pytest
import imageio
import os, sys
import networkx as nx
import numpy as np
from gym_taxi.envs.taxi_env_batch import TaxiEnvBatch

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

        assert done == False
        assert env.world.nodes[0]['info'].get_driver_num() == 2
        assert env.world.nodes[1]['info'].get_driver_num() == 0
        assert env.world.nodes[2]['info'].get_driver_num() == 0
        assert env.world.nodes[3]['info'].get_driver_num() == 4
        assert len(env.traveling_pool[2]) == 1
        assert env.traveling_pool[2][0].income == 0.75
        assert env.traveling_pool[2][0].status == 0

        # driver distr, order distr, one-hot time, mean/min/max income
        incomes = np.array([-0.5,-0.5,-0.5, 0,0,0, 0,0,0, -0.5,-0.5,-0.5])/2 # time normalization
        incomes = (incomes+0.5)/(0.75 + 0.5)
        true = np.concatenate((np.array([2/4,0,0,4/4, 0,0,0,0, 0,1,0,0]), incomes))
        assert (observation == np.array(true)).all()

        observation, reward, done, info = env.step(action)
        assert done == False

        observation, reward, done, info = env.step(action)
        assert done == False

        observation, reward, done, info = env.step(action)
        assert done == True

