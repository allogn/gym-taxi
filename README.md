# Gym Taxi Environemt

This envirnoment simulates taxi cabs picking and delivering customers. A policy defines an idle taxi strategy.

# Parameters of the environment

There are 2 modes of the environment: a batch mode, and a single mode. A batch mode updates all nodes as a single step. A single mode updates one node per step, and updates internal time when all nodes are updated.

# Usage

```python
g = nx.Graph()
g.add_edges_from([(0,1)])
nx.set_node_attributes(g, {0: (0,1), 1: (1,2)}, "coords")
orders = [(1,1,1,1,0.5)]
drivers = np.ones((2), dtype=int)

env_id = "TaxiEnvBatchTest-v01"
gym.envs.register(
    id=env_id,
    entry_point='gym_taxi.envs:TaxiEnvBatch', # or gym_taxi.envs:TaxiEnv
    kwargs={
        'world': g,
        'orders': orders,
        'order_sampling_rate': 1,
        'drivers_per_node': drivers,
        'n_intervals': 10,
        'wc': 0.5
    }
)

DATA_PATH = ... # place data path here
if os.path.isdir(DATA_PATH):
    shutil.rmtree(DATA_PATH)
os.makedirs(DATA_PATH)

def make_env():
    env = gym.make(env_id)
    env.seed(1)
    return env
env = DummyVecEnv([make_env])

model = A2C(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=10)

obs = env.reset()
images = []
img = env.render(mode="rgb_array")
images.append(img)
for _ in range(10):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    images.append(env.render(mode="rgb_array"))
imageio.mimwrite(os.path.join(DATA_PATH, 'taxi_batch_a2c.gif'), [np.array(img) for i, img in enumerate(images)], format="GIF-PIL", fps=5)
```