"""
Script for rolling out a hand-designed rmp2 policy on the franka robot
rmp parameters are given in rmp2/configs/franka_config.yaml
"""

from rmp2.rmpgraph import RobotRMPGraph
from rmp2.envs import FrankaEnvSF
from rmp2.utils.env_wrappers import FrankaFullRMPWrapper
import tensorflow as tf

n_trials = 15
seed = 15
dtype = "float32"

env_wrapper = FrankaFullRMPWrapper(dtype=dtype)
rmp_graph = RobotRMPGraph(robot_name="franka", dtype=dtype, timed=True)

config = {
    "goal": [0.5, -0.5, 0.5],
    "horizon": 1800,
    "action_repeat": 3, # Repeat the action for this many time steps
    "q_init": [ 0.0000, -0.7854,  0.0000, -2.4435,  0.0000,  1.6581,  0.75],
    "render": True,
    "max_obstacle_num": 4,
    "min_obstacle_num": 4,    
    "min_obstacle_radius": 0.02,    
    "max_obstacle_radius": 0.05,
}

goal = tf.convert_to_tensor([config['goal']])

def policy(state,env):
    ts_state = tf.convert_to_tensor([state])
    policy_input = env_wrapper.obs_to_policy_input(ts_state)
    policy_input['goal'] =  tf.convert_to_tensor(env.current_goal, dtype=dtype) #goal
    ts_action = rmp_graph(**policy_input)
    action = ts_action[0].numpy()
    return action

env = FrankaEnvSF(config)
env.seed(seed)
# state = env.reset()
# action = policy(state)

for n in range(n_trials):
    print("Resetting state")
    state = env.reset()
    print(f"Starting sim run {n}")
    while True:
        action = policy(state,env)
        state, reward, done, _ = env.step(action)
        if done:
            break

