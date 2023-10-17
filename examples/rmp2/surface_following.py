"""
Script for rolling out a hand-designed rmp2 policy on the franka robot
rmp parameters are given in rmp2/configs/franka_config.yaml
"""

from rmp2.rmpgraph import RobotRMPGraph
from rmp2.envs import FrankaEnvSF
from rmp2.utils.env_wrappers import FrankaFullRMPWrapper
import tensorflow as tf
from math import pi
import time


n_trials = 1
seed = 15
dtype = "float32"

env_wrapper = FrankaFullRMPWrapper(dtype=dtype)
rmp_graph = RobotRMPGraph(robot_name="franka", dtype=dtype, timed=True)

config = {
    "goal": [0.5, -0.5, 0.5],
    "horizon": 15000, # 1800
    "action_repeat": 3, # Repeat the action for this many time steps
    "q_init": [0.0000, -1*pi/4,  0.0000, -7*pi/12,  0.0000,  10*pi/12,  pi/4],
    "render": True,
    "max_obstacle_num": 8,  # 4
    "min_obstacle_num": 8,  # 4  
    "min_obstacle_radius": 0.05,    # 0.02
    "max_obstacle_radius": 0.08,    # 0.05
    "waypoints": [[0.3, -0.2, 0.7], [0.3, -0.225, 0.7],[0.3, -0.25, 0.7], [0.3, -0.275, 0.7],[0.3, -0.3, 0.7], [0.3, -0.325, 0.7],[0.3, -0.35, 0.7], [0.3, -0.375, 0.7],[0.3, -0.4, 0.7], [0.3, -0.425, 0.7],[0.3, -0.45, 0.7],[0.3, -0.475, 0.7],[0.3, -0.5, 0.7]],
    "waypoint_reaching": False,
    "dynamic_env": True,
    "monorail_vel": [0,1.0,0],
    "simulating_point_cloud": True,
    "plotting_point_cloud": False,
    "plotting_point_cloud_results": True,
    "point_cloud_radius": 0.03,
    "goal_distance_from_surface": 0.20,
    "env_mode": 'cylinder_combo',
    "initial_collision_buffer": 0.0,
    "initial_joint_limit_buffer": 0.0,
    "initial_goal_distance_min": 0.0, 
    # pybullet gravity
    "gravity": 9.8, 
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

try:
    env = FrankaEnvSF(config)
    env.seed(seed)
    # state = env.reset()
    # action = policy(state)

    for n in range(n_trials):
        print("Resetting state")
        state = env.reset()
        print(f"Starting sim run {n}")
        env.camera.activate_sim()
        while True:
            # print("--------------------- START TIME STEP -----------------------")
            t_start = time.time()
            action = policy(state,env)
            t_end = time.time()
            # print("Time taken to get the policy", t_end-t_start)
            # print("Doing env.step(action)")
            state, reward, done, _ = env.step(action)
            if done:
                break
            t_end2 = time.time()
            # print("Time step takes", t_end2-t_start)
except KeyboardInterrupt:
    print("\nInterupted")
    env.camera.plot_error()

env.camera.plot_error()

