"""
Script for rolling out a hand-designed rmp2 policy on the franka robot for a surface following task.
rmp parameters are given in rmp2/configs/franka_config.yaml
Uncomment code if you wish to store the computation time or error of a simulation run
To end a simulation run, enter ctrl+c
"""

from rmp2.rmpgraph import RobotRMPGraph
from rmp2.envs import FrankaEnvSF
from rmp2.utils.env_wrappers import FrankaFullRMPWrapper
import tensorflow as tf
from math import pi
import time
import sys

n_trials = 1 # number of repeat trials of the environment
seed = 15 # random seed for initialising the robot's initial configuration
dtype = "float32"
# policy_calc_times = [] # used to store computation time required to evaluate the policy

# Initialise variables for when the algorithm is being run on it's own
voxel_size=0.05
monorail_velocity_y = 0.2
goal_distance = 0.2
filename_suffix = 'demo'
output_folder='data'

# Following code allows for .sh files to be created to handle automated testing
if(len(sys.argv)==5):
    voxel_size =  float(sys.argv[1])
    filename_suffix = sys.argv[2]
    monorail_velocity_y = float(sys.argv[3])
    goal_distance = float(sys.argv[4])
    print(f"Running with voxel_size={voxel_size}, filename={filename_suffix}, monorail_velocity=[0,{monorail_velocity_y}, 0], goal_distance={goal_distance}")


env_wrapper = FrankaFullRMPWrapper(dtype=dtype)
rmp_graph = RobotRMPGraph(robot_name="franka", dtype=dtype, timed=(True,output_folder,filename_suffix))

config = {
    "goal": [0.5, -0.5, 0.5],
    "horizon": 15000,
    "action_repeat": 3, # Repeat the action for this many time steps
    "q_init": [0.0000, -1*pi/4,  0.0000, -7*pi/12,  0.0000,  10*pi/12,  pi/4],
    "render": True,
    "max_obstacle_num": None,  # Values labelled none are not used for surface following
    "min_obstacle_num": None,   
    "min_obstacle_radius": None,   
    "max_obstacle_radius": None,   
    "waypoints": None, # To be used if waypoint_reaching is enabled. Example use: [[0.3, -0.475, 0.7],[0.3, -0.5, 0.7]]
    "waypoint_reaching": False,
    "dynamic_env": True,
    "monorail_vel": [0,monorail_velocity_y,0],
    "simulating_point_cloud": True,
    "plotting_point_cloud": True,
    "plotting_point_cloud_results": True,
    "point_cloud_radius": voxel_size/2,
    "goal_distance_from_surface": goal_distance,
    "env_mode": 'cylinder_combo',
    "initial_collision_buffer": 0.0, # Set to 0 so that the eef is able to go close to the surface
    "initial_joint_limit_buffer": 0.0,
    "initial_goal_distance_min": 0.0, 
    
    # pybullet gravity - modelled as positive to account for the robot being simulated right way up when it will be upside down in the LHC env
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
    # env.camera.figure_title=f'Voxel size of {voxel_size}m'
    # env.camera.filename_suffix=filename_suffix
    # env.camera.output_folder=output_folder
    env.seed(seed)

    # first = True

    for n in range(n_trials):
        print("Resetting state")
        state = env.reset()
        print(f"Starting sim run {n}")
        env.camera.activate_sim()
        while True:
            t_start = time.time()
            action = policy(state,env)
            t_end = time.time()
            # if not first:
            #     policy_calc_times.append(t_end-t_start)s
            # else:
            #     first_policy_eval_time = t_end-t_start
            state, reward, done, _ = env.step(action)
            if done:
                break
            t_end2 = time.time()
            # print("Time step takes", t_end2-t_start)
            # print("step_action_times", env.comp_times)
            # print("policy_calc_times", policy_calc_times)
            first = False
except KeyboardInterrupt:
    print("\nInterupted")

# env.camera.plot_error(env.comp_times, policy_calc_times, first_policy_eval_time)

