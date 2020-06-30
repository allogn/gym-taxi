import pytest
import networkx as nx
import numpy as np

from gym_taxi.envs.taxi_env import TaxiEnv

class TestTaxiEnv:

    def testInit(self):
        '''
        Test initialization of driver and order distributions
        '''
        # initialize input parameters
        g = nx.Graph()
        g.add_edges_from([(0,1),(1,2),(0,2),(2,3)])
        orders = [(1,1,0,1,0.5), (1,1,1,1,0.7)] # <source, destination, time, length, price>
        drivers = [0,1,2,3]
        order_sampling_rate = 1
        n_intervals = 3
        env = TaxiEnv(g, orders, order_sampling_rate, drivers, n_intervals)
        assert env.n_drivers == 6
        assert len(env.all_driver_list) == env.n_drivers
        assert env.max_reward == 0.7
        assert len(env.orders_per_time_interval) == n_intervals+1
        assert env.action_space_shape == (4,)
        assert env.done == False
        assert env.time == 0
        assert len(env.world) == 4
        assert sum([env.drivers_per_node[i] for i in range(4)]) == 6
        assert sum([env.world.nodes[i]['info'].get_driver_num() for i in range(4)]) == 6
        assert sum([env.world.nodes[i]['info'].get_order_num() for i in range(4)]) == 1

    def testMove3CarsStrictly(self): 
        g = nx.Graph()
        N = 9
        g.add_edges_from([(0,1), (1,2), (0,3), (1,4), (2,5), (3,4), (4,5), (3,6), (4,7), (5,8), (6,7), (7,8)])
        nx.set_node_attributes(g, {0: (0,0), 1: (0,1), 2: (0,2), 3: (1,0),
                                     4: (1,1), 5: (1,2), 6: (2,0), 7: (2,1), 8: (2,2)}, "coords")
        orders = [(7,3,0,2,999.4)]
        drivers = np.zeros(N, dtype=int)
        drivers[0] = 1
        drivers[2] = 1
        drivers[7] = 2
        env = TaxiEnv(g, orders, 1, drivers, 3, 0.5, hold_observation=False)
        observation = env.reset()

        # observation should be [drivers] + [customers] + [onehot time] + [onehot cell]
        order_distr = np.zeros(N)
        order_distr[7] = 1
        assert (observation[:N] == drivers / np.max(drivers)).all()
        assert (observation[N:2*N] == order_distr).all()
        onehot_time = np.zeros(3)
        onehot_time[0] = 1
        assert (observation[3*N:3*N+3] == onehot_time).all()
        node_id =  np.argmax(observation[3*N+3:4*N+3])
        assert node_id in [0, 2, 7]

        env.current_node_id = 0
        env.find_non_empty_nodes()
        action = np.zeros(5)
        action[0] = 1
        observation, reward, done, info = env.step(action)
        assert env.time == 0
        assert reward == -0.5
        assert done == False
        new_drivers = np.copy(drivers)
        new_drivers[0] = 0
        new_drivers[1] = 0
        assert (observation[:N] == new_drivers/np.max(new_drivers)).all()

        env.current_node_id = 2
        env.find_non_empty_nodes()
        action = np.zeros(5)
        action[-1] = 1
        observation, reward, done, info = env.step(action)
        assert done == False
        assert reward == -0.5

        assert env.current_node_id == 7
        env.find_non_empty_nodes()
        assert env.time == 0
        assert (observation[N:2*N] == order_distr).all()
        new_drivers[2] = 0
        assert (observation[:N] == new_drivers/np.max(new_drivers)).all()
        assert (observation[3*N:3*N+3] == onehot_time).all()
        assert np.argmax(observation[3*N+3:4*N+3]) == 7
        action = np.zeros(5)
        action[1] = 1
        observation, reward, done, info = env.step(action)

        # should have completed the step
        assert done == False
        assert reward == (999.4 - 0.5)/2 # averaging by cars in the node by default
        assert env.time == 1
        assert (observation[N:2*N] == np.zeros(N)).all()
        new_drivers = np.zeros(N)
        new_drivers[2] = 1
        new_drivers[1] = 1
        new_drivers[6] = 1 # assuming second edge is to 6th node
        # one driver still travelling
        assert (observation[:N] == new_drivers/np.max(new_drivers)).all()
        assert len(env.traveling_pool[2]) == 1
        d = env.traveling_pool[2][0]
        assert d.income == 999.4
        assert d.status == 0
        assert d.position == 3

        action = np.zeros(5)
        action[-1] = 1

        observation, reward, done, info = env.step(action)
        assert done == False
        assert env.time == 1
        observation, reward, done, info = env.step(action)
        assert done == False
        assert env.time == 1
        observation, reward, done, info = env.step(action)
        assert env.time == 2
        assert done == False # True when time = n_intervals (3)

        # check status of nodes and drivers
        assert env.world.nodes[0]['info'].get_driver_num() == 0
        assert env.world.nodes[1]['info'].get_driver_num() == 1
        assert env.world.nodes[2]['info'].get_driver_num() == 1
        assert env.world.nodes[6]['info'].get_driver_num() == 1
        assert env.world.nodes[3]['info'].get_driver_num() == 1
        assert len(env.all_driver_list) == 4
        total_income = 0
        for d in env.all_driver_list:
            total_income += d.income
            assert d.status == 1
        assert total_income == -0.5 * 3 * 2 + 999.4

    def test_count_neighbors(self):
        g = nx.Graph()
        g.add_edges_from([(0,1),(1,2),(2,0)])
        nx.set_node_attributes(g, {0: (0,1), 1: (0,2), 2: (1,1)}, name="coords")
        orders = [(0,1,0,1,1), (1,2,0,2,2), (2,0,0,3,3)]
        drivers = np.array([4,0,1])

        env = TaxiEnv(g, orders, 1, drivers, 3, 0.5, count_neighbors = True)
        observation1 = env.reset()
        env.current_node_id = 0
        env.non_empty_nodes = [1,2]
        dispatch_list = env.make_order_dispatch_list_and_remove_orders()
        assert len(dispatch_list) == 2

        env.reset() # order_dispatch_list can be run only single time
        env.current_node_id = 0
        env.non_empty_nodes = [1,2]
        observation2, reward, done, info = env.step(np.zeros(3))
        assert reward == (1 + 2 - 0.5 - 0.5) / 4

    def test_reward_options(self):
        '''
        Test these:
            weight_poorest: bool = False,
            normalize_rewards: bool = True,
            minimum_reward: bool = False,
            reward_bound: float = None,
            include_income_to_observation: int = 0
        '''
        g = nx.Graph()
        g.add_edges_from([(0,1),(1,2),(2,3)])
        nx.set_node_attributes(g, {0: (0,1), 1: (0,2), 2: (1,1), 3: (1,2)}, name="coords")
        orders = [(0,1,0,1,1), (1,1,0,2,2), (2,2,0,3,3), (3,2,0,3,3)]
        drivers = np.array([1,0,0,5])
        action = np.array([1, 0, 0], dtype=float)

        env = TaxiEnv(g, orders, 1, drivers, 3, 0.5, count_neighbors = True, normalize_rewards = False)
        observation = env.reset()
        env.current_node_id = 3
        env.non_empty_nodes = [0,1,2]
        observation, reward, done, info = env.step(action)
        assert reward == (3 + 3 - 0.5 - 0.5 - 0.5)

        env = TaxiEnv(g, orders, 1, drivers, 3, 0.5, count_neighbors = True, weight_poorest=True)
        observation = env.reset()
        env.current_node_id = 3
        env.non_empty_nodes = [0,1,2]
        observation, reward, done, info = env.step(action)
        # reward is softmax of the reard multiplied by reward
        r = np.array([0, 3, 3, -0.5, -.5, -.5]) # 0 is because there is a guy in the node 0 that does not move
        mult = 1-env.softmax(r)
        rew = mult*r
        rew /= 5
        assert reward == pytest.approx(np.sum(rew))

        env = TaxiEnv(g, orders, 1, drivers, 3, 0.5, count_neighbors = True, minimum_reward = True)
        observation = env.reset()
        env.current_node_id = 3
        env.non_empty_nodes = [0,1,2]
        observation, reward, done, info = env.step(action)
        assert reward == -0.5 / 5 # returns a single value of a minimum reward, normalized

        env = TaxiEnv(g, orders, 1, drivers, 3, 0.5, count_neighbors = True, normalize_rewards = False, minimum_reward = True)
        observation = env.reset()
        env.current_node_id = 3
        env.non_empty_nodes = [0,1,2]
        observation, reward, done, info = env.step(action)
        assert reward == -0.5 # returns a single value of a minimum reward, non-normalized

        env = TaxiEnv(g, orders, 1, drivers, 3, 0.5, count_neighbors = True, reward_bound = 1)
        observation = env.reset()
        env.current_node_id = 3
        env.non_empty_nodes = [0,1,2]
        observation, reward, done, info = env.step(action)
        assert reward == (1 + 1 - 0.5 - 0.5 - 0.5) / 5

        drivers = np.array([2,0,0,5])
        env = TaxiEnv(g, orders, 1, drivers, 3, 0.5, count_neighbors = True, reward_bound = 1, include_income_to_observation = True)
        observation = env.reset()
        env.world.nodes[0]['info'].drivers[0].add_income(0.9)
        env.current_node_id = 3
        env.non_empty_nodes = [0,1,2]
        # all drivers from 3rd node are moved but haven't arrived, so observation should show only the driver at 0's node
        observation, reward, done, info = env.step(action)
        assert env.current_node_id == 0
        assert observation.shape[0] == 5*env.world_size + env.n_intervals

    def test_seeding(self):
        g = nx.Graph()
        g.add_edges_from([(0,1),(1,2),(2,3)])
        nx.set_node_attributes(g, {0: (0,1), 1: (0,2), 2: (1,1), 3: (1,2)}, name="coords")
        orders = [(0,1,0,1,1), (1,1,1,2,2), (2,2,1,3,3), (3,2,2,3,3)]
        drivers = np.array([1,0,0,5])
        action = np.array([0.3, 0.4, 0.3], dtype=float)

        obs, rew, done, info = None, None, None, None 
        for i in range(100):
            env = TaxiEnv(g, orders, 1, drivers, 10, seed=123)
            env.step(action)
            obs2, rew2, done2, info2 = env.step(action)
            if i > 0:
                assert (obs == obs2).all()
                assert rew == rew2
                assert done == done2
                assert info == info2
            obs, rew, done, info = obs2, rew2, done2, info2

    def test_sync(self):
        g = nx.Graph()
        g.add_edges_from([(0,1),(1,2),(2,3)])
        nx.set_node_attributes(g, {0: (0,1), 1: (0,2), 2: (1,1), 3: (1,2)}, name="coords")
        orders = [(0,1,0,1,1), (1,1,1,2,2), (2,2,1,3,3), (3,2,2,3,3)]
        drivers = np.array([1,0,0,5])
        action = np.array([0.3, 0.4, 0.3], dtype=float)

        env = TaxiEnv(g, orders, 1, drivers, 10)
        env.step(action)

        env2 = TaxiEnv(g, orders, 1, drivers, 10)
        env2.sync(env)

        o1, _, _ = env.get_observation() 
        o2, _, _ = env2.get_observation()
        assert (o1 == o2).all()
        
        env.seed(1)
        env2.seed(1)

        while not env.done:
            obs, rew, done, info  = env.step(action)
            obs2, rew2, done2, info2 = env2.step(action)
            
            assert (obs == obs2).all()
            assert rew == rew2
            assert done == done2
            assert info == info2

    def test_view(self):
        g = nx.Graph()
        g.add_edges_from([(0,1),(1,2),(2,3),(3,4),(1,5)])
        nx.set_node_attributes(g, {0: (0,1), 1: (0,2), 2: (0,3), 3: (0,4), 4: (0,5), 5: (1,1)}, name="coords")
        orders = [(3,2,2,3,3)] # <source, destination, time, length, price>
        drivers = np.array([1,1,1,1,1,1])
        action = np.array([1, 0, 0], dtype=float)

        env = TaxiEnv(g, orders, 1, drivers, 10)
        env.seed(123)
        env.set_view([2,3,4])
        obs, _, _ = env.get_observation()

        # check observation space and content
        assert env.observation_space_shape == obs.shape
        view_size = 3
        assert env.observation_space_shape == (view_size*4 + 10,) # default income is not included, so its <driver, order, idle, time_id, node_id>
        assert env.action_space_shape == (3,) # degree of 1 is 3, but of the rest is 2. So it should be 2 + 1 (staying action)
        assert env.current_node_id in [2,3,4]

        # an action [1, 0, 0] for the node 2 means to go to node 3, because its the only neighbor in the view
        env.step(action)
        assert env.current_node_id in [2,3,4]
        env.step(action)
        assert env.current_node_id in [2,3,4]
        obs, rew, done, info = env.step(action)
        assert (obs[:view_size] == np.array([0.5,1,0])).all() # there are 2 drivers in the node 3 at the end, one from node 2, one from node 4.
        assert (obs[view_size:2*view_size] == np.array([0,0,0])).all()
        assert (obs[2*view_size:3*view_size] == np.array([0.5,1,0])).all()
        # next time iteration should happen
        assert env.time == 1
        assert env.current_node_id in [2,3]
        assert (obs[2*view_size:3*view_size] == np.array([0.5,1,0])).all()
        assert (obs[3*view_size:3*view_size+10] == np.array([0,1,0,0,0,0,0,0,0,0])).all()
        assert obs[3*view_size+10:].shape == (3,)
        assert (obs[3*view_size+10:] == np.array([1,0,0])).all() or  (obs[3*view_size+10:] == np.array([0,1,0])).all()
        assert [d.position for d in env.all_driver_list] == [0, 1, 3, 2, 3, 5]

    def test_automatic_return(self):
        """
        Check linear graph: half of the graph is outside the view, and the car that was sent outside
        is returning automatically to the nearest node in the view.
        """
        g = nx.Graph()
        g.add_edges_from([(0,1),(1,2),(2,3),(3,4),(4,5)])
        nx.set_node_attributes(g, {0: (0,1), 1: (0,2), 2: (0,3), 3: (0,4), 4: (0,5), 5: (0,6)}, name="coords")
        orders = [(1,4,0,10,80)]
        drivers = np.array([0,1,0,0,0,0])
        action = np.array([0, 0, 0], dtype=float)

        env = TaxiEnv(g, orders, 1, drivers, 30)
        env.set_view([0,1,2])
        env.step(action)
        # check that the final destination of the car is the node 2, in (10+2) intervals
        # after performing step(), the env should set current_node_id to 2, and time to 12
        assert env.time == 12
        assert env.current_node_id == 2
        d = env.all_driver_list[0]
        d.status = 1
        d.income = 80
        d.position = 2

    def test_discrete(self):
        g = nx.Graph()
        g.add_edges_from([(0,1),(1,2),(0,2),(0,3),(1,3),(2,3)])
        nx.set_node_attributes(g, {0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)}, name="coords")
        orders = [(3,2,20,3,3)] # one fake order in a very long time <source, destination, time, length, price>
        drivers = np.array([3,0,0,0])
        actions = [1,2,3]

        env = TaxiEnv(g, orders, 1, drivers, 30, discrete=True, poorest_first=True)
        # update incomes of drivers
        env.all_driver_list[0].income = 5
        env.all_driver_list[1].income = 4
        env.all_driver_list[2].income = 3

        # grid network 2x2; 3 cars in top-left, with different income. Step actions to 3 other corners, then check positions.
        for a in actions:
            env.step(a)
        
        assert env.time == 1
        assert env.current_node_id in [1,2,3]
        d = env.all_driver_list[0]
        assert d.income == 5
        assert d.position == 3 # first action is 1, but 0th driver should be selected last
        
        d = env.all_driver_list[1]
        assert d.income == 4
        assert d.position == 2

        d = env.all_driver_list[2]
        assert d.income == 3
        assert d.position == 1