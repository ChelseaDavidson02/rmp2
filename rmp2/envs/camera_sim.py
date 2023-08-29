"""
Class for simulating camera 
"""

import math
from math import pi
import numpy as np
import matplotlib.pyplot as plt
# from bullet_utils import add_obstacle_cuboid, add_obstacle_cylinder, add_obstacle_ball
import time

class Camera():
    def __init__(self, bullet_client, IMG_W=128, IMG_H=128, CAM_DISTANCE=100000):
        
        # simulated camera configs
        self.CAM_DISTANCE = CAM_DISTANCE
        self.IMG_W, self.IMG_H = IMG_W, IMG_H
        self.ax = None
        self.figure = None
        self.bullet_client = bullet_client
        self.scatter = None
        self.extrinsic_matrix = None
        self.intrinsic_matrix = None
        
    
    def setup_plot(self):
        """Sets up the plot used to display the point cloud"""
        # to run GUI event loop
        plt.close('all')
        plt.ion()
        
        self.figure = plt.figure()
        
        self.ax = self.figure.add_subplot(111, projection='3d')
        self.scatter = self.ax.scatter([], [], [], s=15)

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        
        # setting the original view
        self.ax.view_init(0, 90)
        
        self.ax.axes.set_xlim3d(left=0, right=1) 
        self.ax.axes.set_ylim3d(bottom=-0.5, top=0.5) 
        self.ax.axes.set_zlim3d(bottom=0, top=0.5) 

        # setting title
        plt.title("Point Cloud", fontsize=20)
        
    def setup_extrinsic_matrix(self, robot, cam_yaw, cam_pitch, cam_x_dist=0.1):
        """
        Computes the extrinsic view matrix using a given camera position and view angle
        robot: the robot in the pybullet simulation - the camera will be situated at the base of this robot
        cam_yaw: the camera's rotation around the z axis 
        cam_pitch: the camera's rotation around the y axis
        cam_x_dist: how far foward the camera is positioned from the base of the robot
        
        """
        # adapted from https://ai.aioz.io/guides/robotics/2021-05-19-visual-obs-pybullet/ 
        # finding position of the camera
        base_pos, base_orn = self.bullet_client.getBasePositionAndOrientation(robot.robot_uid)

        # Set the camera to be at the base of the robot
        xA, yA, zA = base_pos
        xA = xA + cam_x_dist # make the camera a little further forward than the robot - stops it from being in frame

        # Apply pitch and yaw 
        xB = xA + math.cos(cam_yaw) * math.cos(cam_pitch) * self.CAM_DISTANCE
        yB = yA + math.sin(cam_yaw) * math.cos(cam_pitch) * self.CAM_DISTANCE
        zB = zA + math.sin(cam_pitch) * self.CAM_DISTANCE

        view_matrix = self.bullet_client.computeViewMatrix(
                            cameraEyePosition=[xA, yA, zA],
                            cameraTargetPosition=[xB, yB, zB],
                            cameraUpVector=[0, 0, 1.0]
                        )
        self.extrinsic_matrix = view_matrix
        
        #show where the camera is pointing
        line_ID = self.bullet_client.addUserDebugLine([xA, yA, zA], [xB, yB, zB], lineColorRGB=[0, 0, 1])
    
    def setup_intrinsic_matrix(self, fov=60, nearVal=0.1, farVal=5.0):
        """
        Computes the intrinsic projection matrix for a given field of view, far value, and near value
        fov: the field of view (int)
        nearVal: the closest distance that the camera will register
        farVal: the farthest distance that the camera will register
        """
        projection_matrix = self.bullet_client.computeProjectionMatrixFOV(
                                fov=fov, aspect=float(self.IMG_W/self.IMG_H), nearVal=nearVal, farVal=farVal)
        self.intrinsic_matrix = projection_matrix
    
    def get_intrinsic_matrix(self):
        """
        Returns the intrinsic camera matrix
        """
        return self.intrinsic_matrix
    
    def get_extrinsic_matrix(self):
        """
        Returns the extrinsic camera matrix
        """
        return self.extrinsic_matrix
                
    def setup_pointcloud(self, robot, cam_yaw, cam_pitch, cam_x_dist=0.1):
        """Initialises the intrinsic and extrinsic matrices used to simulate the depth images"""
        self.setup_plot()
        self.setup_intrinsic_matrix()
        self.setup_extrinsic_matrix(robot, cam_yaw, cam_pitch, cam_x_dist)
    
    def update_camera(self):
        """Returns the simulated images for the current state of the pybullet environment."""
        # Take images
        imgs = self.bullet_client.getCameraImage(self.IMG_W, self.IMG_H,
                                self.extrinsic_matrix,
                                self.intrinsic_matrix, shadow=True,
                                renderer=self.bullet_client.ER_BULLET_HARDWARE_OPENGL)
        
        return imgs
    
    def get_point_cloud(self, im):
        """
        Returns points from a depth image in world coordinates.
        """
        # adapted from https://github.com/bulletphysics/bullet3/issues/1924
        # get a depth image
        # "infinite" depths will have a value close to 1
        depth = im[3]

        # create a 4x4 transform matrix that goes from pixel coordinates (and depth values) to world coordinates
        proj_matrix = np.asarray(self.intrinsic_matrix).reshape([4, 4], order="F")
        view_matrix = np.asarray(self.extrinsic_matrix).reshape([4, 4], order="F")
        tran_pix_world = np.linalg.inv(np.matmul(proj_matrix, view_matrix))

        # create a grid with pixel coordinates and depth values
        y, x = np.mgrid[-1:1:2 / self.IMG_H, -1:1:2 / self.IMG_W]
        y *= -1.
        x, y, z = x.reshape(-1), y.reshape(-1), depth.reshape(-1)
        h = np.ones_like(z)

        pixels = np.stack([x, y, z, h], axis=1)
        # filter out "infinite" depths
        pixels = pixels[z < 0.99]
        pixels[:, 2] = 2 * pixels[:, 2] - 1

        # turn pixels to world coordinates
        points = np.matmul(tran_pix_world, pixels.T).T
        points /= points[:, 3: 4]
        points = points[:, :3]

        return points

    
    def plot_point_cloud_dynamic(self, points):
        """
        Updates the 3D scatter plot to plot the given points. Assumes that a scatter plot figure has already been created. 
        """
        # Main functions found from https://stackoverflow.com/questions/5179589/continuous-3d-plotting-i-e-figure-update 
        x_coords, y_coords, z_coords = points[:, 0], points[:, 1], points[:, 2]
        self.scatter._offsets3d = (x_coords, y_coords, z_coords) # override the previous scatter plot
        plt.pause(0.001) #TODO
    
    def step_sensing(self):
        """
        Completes the sensing involved in a simulation step - captures camera data in the current pybullet environment and returns points 
        representing the point cloud. 
        """
        imgs = self.update_camera()
        points = self.get_point_cloud(im = imgs)
        
        # combined_points.append(points_i)
        # combined_points_stack = np.vstack(combined_points)
        # unique_combined_points = np.unique(combined_points_stack, axis=0) # removing any points which were found in multiple camera angles
        
        
        return points