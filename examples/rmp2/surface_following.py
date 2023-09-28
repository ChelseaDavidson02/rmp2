"""
Script for rolling out a hand-designed rmp2 policy on the franka robot
rmp parameters are given in rmp2/configs/franka_config.yaml
"""

from rmp2.rmpgraph import RobotRMPGraph
from rmp2.envs import FrankaEnvSF
from rmp2.utils.env_wrappers import FrankaFullRMPWrapper
import tensorflow as tf
from math import pi

n_trials = 15
seed = 15
dtype = "float32"

env_wrapper = FrankaFullRMPWrapper(dtype=dtype)
rmp_graph = RobotRMPGraph(robot_name="franka", dtype=dtype, timed=True)

config = {
    "goal": [0.5, -0.5, 0.5],
    "horizon": 6000, # 1800
    "action_repeat": 3, # Repeat the action for this many time steps
    "q_init": [ 0.0000, -pi/5,  0.0000, -1*pi/2,  0.0000,  3*pi/4,  pi/4],
    "render": True,
    "max_obstacle_num": 8,  # 4
    "min_obstacle_num": 8,  # 4  
    "min_obstacle_radius": 0.05,    # 0.02
    "max_obstacle_radius": 0.08,    # 0.05
    "waypoints": [[0.3, -0.2, 0.7], [0.3, -0.225, 0.7],[0.3, -0.25, 0.7], [0.3, -0.275, 0.7],[0.3, -0.3, 0.7], [0.3, -0.325, 0.7],[0.3, -0.35, 0.7], [0.3, -0.375, 0.7],[0.3, -0.4, 0.7], [0.3, -0.425, 0.7],[0.3, -0.45, 0.7],[0.3, -0.475, 0.7],[0.3, -0.5, 0.7]],
    "waypoint_reaching": False,
    "dynamic_env": True,
    "monorail_vel": [0,0.9,0],
    "simulating_point_cloud": True,
    "plotting_point_cloud": True,
    "point_cloud_radius": 0.015,
    "goal_distance_from_surface": 0.3,
}
if config['waypoint_reaching']:
    goal = config['waypoints'][0]
else:
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

