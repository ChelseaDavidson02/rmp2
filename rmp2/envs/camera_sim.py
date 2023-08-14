"""
helper functions for simulating camera 
"""

import pybullet as p
import pybullet_data
import math
from math import pi
import random
import robot_sim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from bullet_utils import add_obstacle_cuboid, add_obstacle_cylinder, add_obstacle_ball
import time

class Camera():
    def __init__(self):
        
        # simulated camera configs
        self.CAM_DISTANCE = 100000
        self.IMG_W, self.IMG_H = 128, 128
        self.ax = None
        self.figure = None
    
        
    def get_point_cloud(self, im, view_matrix, proj_matrix):
        # based on https://stackoverflow.com/questions/59128880/getting-world-coordinates-from-opengl-depth-buffer

        # get a depth image
        # "infinite" depths will have a value close to 1
        depth = im[3]

        # create a 4x4 transform matrix that goes from pixel coordinates (and depth values) to world coordinates
        proj_matrix = np.asarray(proj_matrix).reshape([4, 4], order="F")
        view_matrix = np.asarray(view_matrix).reshape([4, 4], order="F")
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
        x_coords, y_coords, z_coords = points[:, 0], points[:, 1], points[:, 2]
        self.ax.plot(x_coords, y_coords, z_coords)
        plt.draw()
        plt.pause(0.02)
        self.ax.cla()
    
    def setup_plot(self):
        # to run GUI event loop
        plt.ion()
        
        self.figure = plt.figure()
        
        self.ax = self.figure.add_subplot(111, projection='3d')

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        
        self.ax.axes.set_xlim3d(left=0, right=2) 
        self.ax.axes.set_ylim3d(bottom=-1, top=1) 
        self.ax.axes.set_zlim3d(bottom=0, top=2) 


        # setting title
        plt.title("Point Cloud", fontsize=20)
        
    
    def update_camera(self, robot, view, height):
        # finding position of the camera
        agent_pos, agent_orn =p.getBasePositionAndOrientation(robot.robot_uid)

        yaw = p.getEulerFromQuaternion(agent_orn)[-1] + view #- pi/6
        xA, yA, zA = agent_pos
        zA = zA + height # make the camera a little higher than the robot
        # print(xA, yA, zA)

        # compute focusing point of the camera
        xB = xA + math.cos(yaw) * self.CAM_DISTANCE
        yB = yA + math.sin(yaw) * self.CAM_DISTANCE
        zB = zA

        view_matrix = p.computeViewMatrix(
                            cameraEyePosition=[xA, yA, zA],
                            cameraTargetPosition=[xB, yB, zB],
                            cameraUpVector=[0, 0, 1.0]
                        )

        projection_matrix = p.computeProjectionMatrixFOV(
                                fov=80, aspect=float(self.IMG_W/self.IMG_H), nearVal=0.05, farVal=100.0)
        #show where the camera is pointing
        line_ID = p.addUserDebugLine([xA, yA, zA], [xB, yB, zB], lineColorRGB=[0, 0, 1])

        # Making images
        imgs = p.getCameraImage(self.IMG_W, self.IMG_H,
                                view_matrix,
                                projection_matrix, shadow=True,
                                renderer=p.ER_BULLET_HARDWARE_OPENGL)
        
        return imgs, view_matrix, projection_matrix


# def main():
    
#     # Initialize the simulation
#     time_step = 1/240.
#     iterations = 100
#     cam = Camera()

#     physicsClient = p.connect(p.GUI)
#     p.setAdditionalSearchPath(pybullet_data.getDataPath())

#     # plt.switch_backend('TkAgg')
#     cam.setup_plot()

#     # Set gravity
#     p.setGravity(0, 0, -9.81)

#     # # Add objects
#     # # Add cylinder
#     # radius=1 
#     # length=20
#     # color=[0.4, 0.4, 0.4, 1]
#     # center = [1.4,0,1.2]
#     # add_obstacle_cylinder(p, center=center, radius=radius, length=length, color=color)

#     # # Add cubes
#     # for i in range(30):
#     #     s = random.uniform(0.05, 0.2)
#     #     size=[s, s, s]
#     #     center = [random.uniform(0.3,0.5),random.uniform(-10, 10), random.uniform(0.5, 1.7)]
#     #     add_obstacle_cuboid(p, center=center, size=size, color=color)

#     # Simple point cloud test
#     add_obstacle_cuboid(p, center=[1.5, 0.0, 0.25], size=[0.5, 0.5, 0.25], color=[0.4, 0.4, 0.4, 1])
#     add_obstacle_ball(p, center=[1, 0, 0], radius=0.1)

#     # robot
#     robot = robot_sim.create_robot_sim("franka", p, time_step, mode=robot_sim.VEL_OPEN_LOOP)
#     # changing the initial position of the robot
#     initial_config = [ 0.0000, -pi/5,  0.0000, -1*pi/2,  0.0000,  3*pi/4,  pi/4]
#     initial_config = initial_config + [0,0,0,0,0,0,0,0,0] # Was testing above numbers not whole config
#     initial_vel = [0.00172073,-0.00127602,-0.00037006,-0.00094746,-0.00245997,-0.00352159,-0.00484516,-0.00352159,-0.00484516]
#     initial_vel = initial_vel + [0.0,0.0,0.0,0.0,0.0,0.0,0.0,-0.00352159,-0.00484516]
#     robot.reset(initial_config, initial_vel)


#     # Simulation loop
#     for i in range(iterations):
#         p.stepSimulation()

#         imgs_1, view_matrix_1, projection_matrix_1 = cam.update_camera(robot, pi/6, 0.6)
#         points_1 = cam.get_point_cloud(im = imgs_1 ,view_matrix=view_matrix_1, proj_matrix=projection_matrix_1)

#         imgs_2, view_matrix_2, projection_matrix_2 = cam.update_camera(robot, -pi/6, 0.6)
#         points_2 = cam.get_point_cloud(im = imgs_2, view_matrix=view_matrix_2, proj_matrix=projection_matrix_2)
        
#         imgs_3, view_matrix_3, projection_matrix_3 = cam.update_camera(robot, 0, 0.6)
#         points_3 = cam.get_point_cloud(im = imgs_3, view_matrix=view_matrix_3, proj_matrix=projection_matrix_3)
        
#         combined_points = np.vstack([points_1,points_2, points_3])
#         unique_combined_points = np.unique(combined_points, axis=0)
#         # print(len(combined_points))
#         cam.plot_point_cloud_dynamic(points=unique_combined_points)

#     plt.close()


#     # Disconnect from the simulation
#     p.disconnect()
    
# if __name__ == "__main__":
#     main()