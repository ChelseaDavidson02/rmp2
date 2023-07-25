"""
Base gym environment for franka robot
"""

from rmp2.envs.robot_env import RobotEnv
from rmp2.utils.np_utils import sample_from_torus_3d
from rmp2.utils.python_utils import merge_dicts
from rmp2.utils.bullet_utils import add_goal, add_obstacle_cylinder, add_obstacle_ball, add_obstacle_cuboid
import numpy as np
import random

DEFAULT_CONFIG = {
    # parameters for randomly generated goals
    "goal_torus_angle_center": 0., 
    "goal_torus_angle_range": np.pi,
    "goal_torus_major_radius": 0.5,
    "goal_torus_minor_radius": 0.3,
    "goal_torus_height": 0.5,
    # parameters for randomly generated obstacles
    "obs_torus_angle_center": np.pi, # 0.
    "obs_torus_angle_range": 2*np.pi,# np.pi
    "obs_torus_major_radius": 1, # 0.5
    "obs_torus_minor_radius": 0.5, # 0.3
    "obs_torus_height": 0.5,
    # obstacle size
    "max_obstacle_radius": 0.1,
    "min_obstacle_radius": 0.05,
    # init min goal distance
    "initial_goal_distance_min": 0.5, 

}

class FrankaEnvSF(RobotEnv):
    """
    Base gym environment for franka robot
    """
    def __init__(self, config=None):
        if config is not None:
            config = merge_dicts(DEFAULT_CONFIG, config)
        else:
            config = DEFAULT_CONFIG.copy()

        # random goal config
        self._goal_torus_angle_center = config["goal_torus_angle_center"]
        self._goal_torus_angle_range = config["goal_torus_angle_range"]
        self._goal_torus_major_radius = config["goal_torus_major_radius"]
        self._goal_torus_minor_radius = config["goal_torus_minor_radius"]
        self._goal_torus_height = config["goal_torus_height"]
        # random obstacle config
        self._obs_torus_angle_center = config["obs_torus_angle_center"]
        self._obs_torus_angle_range = config["obs_torus_angle_range"]
        self._obs_torus_major_radius = config["obs_torus_major_radius"]
        self._obs_torus_minor_radius = config["obs_torus_minor_radius"]
        self._obs_torus_height = config["obs_torus_height"]

        super().__init__(
            robot_name="franka",
            workspace_dim=3,
            config=config)

    def _generate_random_goal(self):
        # if goal is not given, sample a random goal with the specified parameters
        if self.goal is None:
            current_goal = sample_from_torus_3d(
                self.np_random,
                self._goal_torus_angle_center, 
                self._goal_torus_angle_range,
                self._goal_torus_major_radius,
                self._goal_torus_minor_radius,
                self._goal_torus_height)
        # if doing waypoint reaching, use the first waypoint as the first goal
        elif self.waypoint_reaching:
            current_goal = self.waypoints[self.waypoint_indx]
        # otherwise, use the given goal
        else:
            current_goal = self.goal
        # generate goal object within pybullet
        print(f"Current goal= {current_goal}")
        goal_uid = add_goal(self._p, current_goal)
        return current_goal, goal_uid

    # def _generate_random_obstacles(self):
    #     current_obstacles = []
    #     obstacle_uids = []

    #     # if obstacle config list is given, sample one config from the list
    #     if self.obstacle_cofigs is not None:
    #         config = self.obstacle_cofigs[self.np_random.integers(1,len(self.obstacle_cofigs))]
    #         for (i, obstacle) in enumerate(config):
    #             obstacle_uids.append(
    #                 add_obstacle_ball(self._p, obstacle['center'], obstacle['radius'])
    #             )
    #             current_obstacles.append(np.append(obstacle['center'], obstacle['radius']))
    #         for i in range(len(config), self.max_obstacle_num):
    #             current_obstacles.append(np.append(np.zeros(self.workspace_dim), -1.))
    #     # otherwise, sample random obstacles with the specified parameters
    #     else:
    #         num_obstacles = self.np_random.integers(self.min_obstacle_num, self.max_obstacle_num + 1)
    #         for i in range(self.max_obstacle_num):
    #             if i < num_obstacles:
    #                 radius = self.np_random.uniform(low=self.min_obstacle_radius, high=self.max_obstacle_radius)
    #                 center = sample_from_torus_3d(
    #                     self.np_random,
    #                     self._obs_torus_angle_center, 
    #                     self._obs_torus_angle_range,
    #                     self._obs_torus_major_radius,
    #                     self._obs_torus_minor_radius,
    #                     self._obs_torus_height)
    #                 obstacle_uids.append(
    #                     add_obstacle_ball(self._p, center, radius)
    #                 )
    #                 current_obstacles.append(np.append(center, radius))
    #             else:
    #                 current_obstacles.append(np.append(np.zeros(self.workspace_dim), -1.))
    #     # generate obstacle objects within pybullet
    #     current_obstacles = np.array(current_obstacles).flatten()
    #     return current_obstacles, obstacle_uids
    
    def _generate_random_obstacles(self):
        current_obstacles = []
        obstacle_uids = []
        
        # Adding big tunnel
        obstacle_uids.append(add_obstacle_cylinder(self._p, [1.5,0,0], radius=1))
        
        # TODO - this centre and radius value seen as centre, radius below, is hard coded
        current_obstacles.append(np.append([1.5,0,0], 1))

        # Adding cubes
    
        for i in range(20):
            num1 = random.randint(-4, 4)
            num2 = random.randint(-100, 100)
            center = [0.5,0.1*num2, 0.1*num1]
            obstacle_uids.append(add_obstacle_cuboid(self._p, center))
            
            # TODO - this centre and radius value seen as centre, radius below, is hard coded
            # current_obstacles.append(np.append([0.5,-0.8,1.25], 0.03))
            current_obstacles.append(np.append(center, 0.03))
        

        current_obstacles = np.array(current_obstacles).flatten()
        return current_obstacles, obstacle_uids
        
        

    # def _generate_random_obstacles(self):
    #     current_obstacles = []
    #     obstacle_uids = []

    #     # Adding big tunnel
    #     obstacle_uids.append(add_big_cylinder())
    #     # TODO - this centre and radius value seen as centre, radius below, is hard coded
    #     current_obstacles.append(np.append([5.,-1.75,1], 1))

    #     # Adding cube
    #     obstacle_uids.append(add_cuboid())
    #     # TODO - this centre and radius value seen as centre, radius below, is hard coded
    #     # current_obstacles.append(np.append([0.5,-0.8,1.25], 0.03))
    #     current_obstacles.append(np.append([0.2,-0.5,0.25], 0.03))

    #     current_obstacles = np.array(current_obstacles).flatten()
    #     return current_obstacles, obstacle_uids
