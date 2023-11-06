"""
Base gym environment for franka robot for surface following task.
Default config is not currently used in surface following task - env_mode is set to cylinder_combo in surface_following.py.
"""

from rmp2.envs.robot_env import RobotEnv
from rmp2.utils.np_utils import sample_from_torus_3d
from rmp2.utils.python_utils import merge_dicts
from rmp2.utils.bullet_utils import add_goal, add_obstacle_cylinder, add_obstacle_ball, add_obstacle_cuboid
import numpy as np
import random
from math import pi

DEFAULT_CONFIG = {
    # parameters for randomly generated goals
    "goal_torus_angle_center": 0., 
    "goal_torus_angle_range": np.pi,
    "goal_torus_major_radius": 0.5,
    "goal_torus_minor_radius": 0.3,
    "goal_torus_height": 0.5,
    # parameters for randomly generated obstacles
    "obs_torus_angle_center": 0, 
    "obs_torus_angle_range": np.pi, 
    "obs_torus_major_radius": 0.5, 
    "obs_torus_minor_radius": 0.3,
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
        self.simulating_point_cloud = config["simulating_point_cloud"]
        self.env_mode = config["env_mode"]

        super().__init__(
            robot_name="franka",
            workspace_dim=3,
            config=config)

    def _generate_random_goal(self):
        """Generates a random goal for the algorithm"""
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

    def _generate_random_obstacles(self):
        """
        Creates a set of obstacles to be used in the environment depending on the env_mode parameter set in surface_following.py
        Function will not generate 'random' obstacles unless a random environment mode is selected. 
        Function naming is to keep robot_env.py integratable with all examples.
        """
        obstacle_colour = [0.5, 0.5, 0.5, 1]
        if self.env_mode == 'single_body':
            return self.generate_single_body_obst(obstacle_colour)
        elif self.env_mode == 'single_eef':
            return self.generate_single_eef_obst(obstacle_colour)
        elif self.env_mode == 'cylinder_sphere':
            return self.generate_cylinder_with_spherical_obst(obstacle_colour)
        elif self.env_mode == 'cylinder_combo':
            return self.generate_cylinder_combination_obst(obstacle_colour)
        elif self.env_mode == 'cylinder_cluttered':
            return self.generate_cluttered_combination_obst(obstacle_colour)
        elif self.env_mode == 'cylinder_overlap':
            return self.generate_overlapping_combination_obst(obstacle_colour)
        elif self.env_mode == 'cylinder_random':
            return self.generate_cylinder_random_obst(obstacle_colour)
        elif self.env_mode == 'surface':
            return self.generate_surface()
        elif self.env_mode == 'random_spheres':
            return self.generate_obs_random()
    
    
            
    
    
    def generate_obs_random(self):
        """Generates a random obstacles based on the config. parameters set in this function and surface_following.py"""
        current_obstacles = []
        obstacle_uids = []

        # if obstacle config list is given, sample one config from the list
        if self.obstacle_cofigs is not None:
            config = self.obstacle_cofigs[self.np_random.integers(1,len(self.obstacle_cofigs))]
            for (i, obstacle) in enumerate(config):
                obstacle_uids.append(
                    add_obstacle_ball(self._p, obstacle['center'], obstacle['radius'])
                )
                current_obstacles.append(np.append(obstacle['center'], obstacle['radius']))
            for i in range(len(config), self.max_obstacle_num):
                current_obstacles.append(np.append(np.zeros(self.workspace_dim), -1.))
        # otherwise, sample random obstacles with the specified parameters
        else:
            num_obstacles = self.np_random.integers(self.min_obstacle_num, self.max_obstacle_num + 1)
            for i in range(self.max_obstacle_num):
                if i < num_obstacles:
                    radius = self.np_random.uniform(low=self.min_obstacle_radius, high=self.max_obstacle_radius)
                    center = sample_from_torus_3d(
                        self.np_random,
                        self._obs_torus_angle_center, 
                        self._obs_torus_angle_range,
                        self._obs_torus_major_radius,
                        self._obs_torus_minor_radius,
                        self._obs_torus_height)
                    obstacle_uids.append(
                        add_obstacle_ball(self._p, center, radius)
                    )
                    current_obstacles.append(np.append(center, radius))
                else:
                    current_obstacles.append(np.append(np.zeros(self.workspace_dim), -1.))
        # generate obstacle objects within pybullet
        current_obstacles = np.array(current_obstacles).flatten()
        return current_obstacles, obstacle_uids        
    
    def generate_single_eef_obst(self, obstacle_colour):
        """Generates a single obstacle in the scene which is on path to collide with the robot's eef."""
        current_obstacles = []
        obstacle_uids = []
    
        # Adding big tunnel
        obstacle_uids.append(add_obstacle_cuboid(self._p, center=[1.6,0,1], size=[1, 20, 1]))
        center = [0.5, -0.5, 1.0]
        s = 0.2
        obstacle_uids.append(add_obstacle_cuboid(self._p, center=center, size=[s, s, s], color=obstacle_colour))
        
        return current_obstacles, obstacle_uids

    def generate_single_body_obst(self, obstacle_colour):
        """Generates a single obstacle in the scene which is on path to collide with the robot's body."""
        current_obstacles = []
        obstacle_uids = []
    
        # Adding big tunnel
        obstacle_uids.append(add_obstacle_cuboid(self._p, center=[1.6,0,0], size=[1, 20, 2]))
        center = [0.5, -0.5, 0.7]
        obstacle_uids.append(add_obstacle_cuboid(self._p, center=center, size=[0.4, 0.2, 0.1], color=obstacle_colour))
        
        return current_obstacles, obstacle_uids
    
    def generate_surface(self):
        """Generates a single cuboid in the scene to imitate a wall."""
        current_obstacles = []
        obstacle_uids = []
    
        # Adding big wall
        obstacle_uids.append(add_obstacle_cuboid(self._p, center=[1.6,0,0], size=[1, 20, 2]))
        
        return current_obstacles, obstacle_uids
    
    def generate_cylinder_with_spherical_obst(self, obstacle_colour):
        """Generates a cylinder with a single spherical obstacle"""
        current_obstacles = []
        obstacle_uids = []
    
        # Adding big tunnel
        obstacle_uids.append(add_obstacle_cylinder(self._p, center=[1.6,0,0.5]))
        center = [0.5, -0.5, 0.7]
        obstacle_uids.append(add_obstacle_ball(self._p, center=center, radius=0.2, color=obstacle_colour))
        
        return current_obstacles, obstacle_uids
    
    def generate_cylinder_combination_obst(self, obstacle_colour):
        """Generates a cylinder with a combonation of obstacles along its path. This environment is intended to model the LHC environment."""
        current_obstacles = []
        obstacle_uids = []
        
        # Adding big tunnel
        obstacle_uids.append(add_obstacle_cylinder(self._p, center=[0.95,0,0.95], radius = 0.5))
        
        # Adding line along tunnel
        obstacle_uids.append(add_obstacle_cylinder(self._p, center=[0.55,0,0.65], radius = 0.04, color=obstacle_colour))
        
        # Adding ball
        obstacle_uids.append(add_obstacle_ball(self._p, center=[0.5, -0.5, 0.75], radius=0.2, color=obstacle_colour))
        
        # Adding short cylinder beside body obstacle
        obstacle_uids.append(add_obstacle_cylinder(self._p, center=[0.95, -0.7, 0.95], radius=0.6, length = 0.1, color=obstacle_colour))
        
        # Adding body obstacle
        obstacle_uids.append(add_obstacle_cuboid(self._p, center=[0.5, -1.4, 0.85], size=[0.3, 0.2, 0.1], color=obstacle_colour))
        
        
        # Adding eef obstacle
        s = 0.2
        obstacle_uids.append(add_obstacle_cuboid(self._p, center=[0.55, -1.8, 0.95], size=[s, s, s], color=obstacle_colour))

        # Adding short cylinder beside eef obstacle
        obstacle_uids.append(add_obstacle_cylinder(self._p, center=[0.95, -2.6, 0.95], radius=0.6, length = 0.1, color=obstacle_colour))
        
        # Adding small cylinder
        obstacle_uids.append(add_obstacle_cylinder(self._p, center=[0.35, -3.4, 0.85], radius=0.1, length = 0.4, color=obstacle_colour))
        
        # Adding large object beside small cylinder
        obstacle_uids.append(add_obstacle_cuboid(self._p, center=[0.4, -4.4, 1.05], size=[0.2, 0.2, 0.6], color=obstacle_colour))
        
        return current_obstacles, obstacle_uids
    
    def generate_overlapping_combination_obst(self, obstacle_colour):
        """Generates a cylinder with a combonation of obstacles along its path that overlap with eachother"""
        current_obstacles = []
        obstacle_uids = []
        
        # Adding big tunnel
        obstacle_uids.append(add_obstacle_cylinder(self._p, center=[1.6,0,1.1], radius = 1.1))
        
        # Adding ball
        obstacle_uids.append(add_obstacle_ball(self._p, center=[0.5, -0.5, 1.0], radius=0.2, color=obstacle_colour))
        # Adding smaller balls and surrounding obstacles
        obstacle_uids.append(add_obstacle_cuboid(self._p, center=[0.5, -0.4, 1.2], size=[0.1, 0.1, 0.1], color=obstacle_colour))
        obstacle_uids.append(add_obstacle_cylinder(self._p, center=[0.4, -0.8, 1.0], radius=0.1, length=0.5, angle_rotation_rads=[0,0,0], color=obstacle_colour))
        obstacle_uids.append(add_obstacle_cuboid(self._p, center=[0.6, -0.5, 0.8], size=[0.15, 0.15, 0.15], color=obstacle_colour))
        obstacle_uids.append(add_obstacle_ball(self._p, center=[0.4, -0.9, 1.2], radius=0.1, color=obstacle_colour))
        
        # Adding body obstacle
        obstacle_uids.append(add_obstacle_cuboid(self._p, center=[0.5, -1.2, 0.8], size=[0.3, 0.2, 0.1], color=obstacle_colour))
        
        # Adding small cylinder above body obstacle
        obstacle_uids.append(add_obstacle_cylinder(self._p, center=[0.3, -1.2, 1.2], radius=0.2, length = 0.4, color=obstacle_colour))
        
        # Adding eef obstacle
        s = 0.2
        obstacle_uids.append(add_obstacle_cuboid(self._p, center=[0.5, -1.5, 1.0], size=[s, s, s], color=obstacle_colour))
        
        # Adding small cylinder
        obstacle_uids.append(add_obstacle_cylinder(self._p, center=[0.4, -2.0, 1.0], radius=0.2, length = 0.4, color=obstacle_colour))
        
        return current_obstacles, obstacle_uids
    
    def generate_cluttered_combination_obst(self, obstacle_colour):
        """Generates a cylinder with a many different obstacles along its path"""
        current_obstacles = []
        obstacle_uids = []
        
        # Adding big tunnel
        obstacle_uids.append(add_obstacle_cylinder(self._p, center=[1.1,0,1.1], radius = 0.6))
        
        # Adding ball
        obstacle_uids.append(add_obstacle_ball(self._p, center=[0.5, -0.5, 1.0], radius=0.2, color=obstacle_colour))
        
        # Adding small cylinder
        obstacle_uids.append(add_obstacle_cylinder(self._p, center=[0.5, -0.8, 1.0], radius=0.05, length = 0.4, angle_rotation_rads=[0,0,0], color=obstacle_colour))
        
        # Adding small ball
        obstacle_uids.append(add_obstacle_ball(self._p, center=[0.5, -1.0, 0.8], radius=0.1, color=obstacle_colour))
        
        # Adding small box
        obstacle_uids.append(add_obstacle_cuboid(self._p, center=[0.5, -1.1, 1.2], size=[0.05, 0.05, 0.05], color=obstacle_colour))
        
        # Adding body obstacle
        obstacle_uids.append(add_obstacle_cuboid(self._p, center=[0.5, -1.5, 0.9], size=[0.3, 0.2, 0.1], color=obstacle_colour))
        
        # Adding small cylinder above body obstacle
        obstacle_uids.append(add_obstacle_cylinder(self._p, center=[0.3, -1.5, 1.2], radius=0.1, length = 0.4, color=obstacle_colour))
        
        # Adding small ball
        obstacle_uids.append(add_obstacle_ball(self._p, center=[0.5, -1.9, 1.0], radius=0.08, color=obstacle_colour))
        
        # Adding thin cylinder 
        obstacle_uids.append(add_obstacle_cylinder(self._p, center=[0.5, -2.1, 1.2], radius=0.05, length = 0.4, angle_rotation_rads=[0,0,0], color=obstacle_colour))
        
        # Adding small box
        obstacle_uids.append(add_obstacle_cuboid(self._p, center=[0.5, -2.15, 0.8], size=[0.05, 0.05, 0.05], color=obstacle_colour))
        
        # Adding eef obstacle
        s = 0.2
        obstacle_uids.append(add_obstacle_cuboid(self._p, center=[0.5, -2.5, 1.0], size=[s, s, s], color=obstacle_colour))
        
        # Adding thin cylinder
        obstacle_uids.append(add_obstacle_cylinder(self._p, center=[0.5, -2.9, 1.0], radius=0.05, length = 0.4, angle_rotation_rads=[0,0,0], color=obstacle_colour))
        
        # Adding small ball
        obstacle_uids.append(add_obstacle_ball(self._p, center=[0.5, -3.1, 0.8], radius=0.1, color=obstacle_colour))
        
        # Adding small box
        obstacle_uids.append(add_obstacle_cuboid(self._p, center=[0.5, -3.2, 1.2], size=[0.05, 0.05, 0.05], color=obstacle_colour))
        
        # Adding object cylinder
        obstacle_uids.append(add_obstacle_cylinder(self._p, center=[0.4, -3.5, 1.0], radius=0.2, length = 0.4, color=obstacle_colour))
        
        return current_obstacles, obstacle_uids
    
    def generate_cylinder_random_obst(self, obstacle_colour):
        """Generates a cylinder with a combonation of randomly sized and placed obstacles along its path. 
        This environment is intended to model a randomised version of the LHC environment."""
        current_obstacles = []
        obstacle_uids = []
        
        # Adding big tunnel
        obstacle_uids.append(add_obstacle_cylinder(self._p, center=[0.95,0,1.1], radius = 0.5))

        # Adding welding
        obstacle_uids.append(add_obstacle_cylinder(self._p, center=[0.55,0,0.8], radius = 0.04, color=obstacle_colour))

        obstacle_uids.append(add_obstacle_cylinder(self._p, center=[0.95, -0.7, 1.1], radius=0.6, length = 0.1, color=obstacle_colour))
        obstacle_uids.append(add_obstacle_cylinder(self._p, center=[0.95, -2.6, 1.1], radius=0.6, length = 0.1, color=obstacle_colour))

        # Adding cubes
        for i in range(5):
            # randomly scatter rectangles along the tunnel wall
            center = [random.uniform(0.35,0.45),random.uniform(-6, -0.5), random.uniform(0.8, 1.1)]
            s1 = random.uniform(0.1, 0.2) # half side length
            s2 = random.uniform(0.1, 0.2) # half side length
            obstacle_uids.append(add_obstacle_cuboid(self._p, center, size=[s1, s2, s1], color=obstacle_colour))
        
        for i in range(5):
            # randomly scatter cubes along the tunnel wall
            center = [random.uniform(0.35,0.45),random.uniform(-6, -0.5), random.uniform(0.8, 1.1)]
            s = random.uniform(0.1, 0.2) # half side length
            obstacle_uids.append(add_obstacle_ball(self._p, center, radius=s, color=obstacle_colour))

        for i in range(5):
            # randomly scatter cubes along the tunnel wall
            center = [random.uniform(0.35,0.45),random.uniform(-6, -0.5), random.uniform(0.8, 1.1)]
            s = random.uniform(0.05, 0.2) # half side length
            l = random.uniform(0.1, 0.4) # half side lengths
            obstacle_uids.append(add_obstacle_cylinder(self._p, center, radius=s, length=l, color=obstacle_colour))
        

        # current_obstacles = np.array(current_obstacles).flatten()
        return current_obstacles, obstacle_uids
    

        
    