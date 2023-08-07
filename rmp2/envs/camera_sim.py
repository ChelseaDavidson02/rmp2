"""
helper functions for simulating camera 
"""

import pybullet as p
import math
from math import pi
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# simulated camera configs
CAM_DISTANCE = 100000
IMG_W, IMG_H = 128, 128


def get_point_cloud(im, width, height, view_matrix, proj_matrix):
    # based on https://stackoverflow.com/questions/59128880/getting-world-coordinates-from-opengl-depth-buffer

    # get a depth image
    # "infinite" depths will have a value close to 1
    depth = im[3]

    # create a 4x4 transform matrix that goes from pixel coordinates (and depth values) to world coordinates
    proj_matrix = np.asarray(proj_matrix).reshape([4, 4], order="F")
    view_matrix = np.asarray(view_matrix).reshape([4, 4], order="F")
    tran_pix_world = np.linalg.inv(np.matmul(proj_matrix, view_matrix))

    # create a grid with pixel coordinates and depth values
    y, x = np.mgrid[-1:1:2 / height, -1:1:2 / width]
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

def plot_point_cloud(points, fig):
    x_coords, y_coords, z_coords = points[:, 0], points[:, 1], points[:, 2]
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x_coords, y_coords, z_coords, s=1)  # s is the marker size

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.show()
    # plt.pause(0.01)
    
def update_camera(robot):
    # finding position of the camera
    agent_pos, agent_orn =p.getBasePositionAndOrientation(robot.robot_uid)

    yaw = p.getEulerFromQuaternion(agent_orn)[-1] #- pi/5
    xA, yA, zA = agent_pos
    zA = zA + 0.5 # make the camera a little higher than the robot

    # compute focusing point of the camera
    xB = xA + math.cos(yaw) * CAM_DISTANCE
    yB = yA + math.sin(yaw) * CAM_DISTANCE
    zB = zA

    view_matrix = p.computeViewMatrix(
                        cameraEyePosition=[xA, yA, zA],
                        cameraTargetPosition=[xB, yB, zB],
                        cameraUpVector=[0, 0, 1.0]
                    )

    projection_matrix = p.computeProjectionMatrixFOV(
                            fov=60, aspect=float(IMG_W/IMG_H), nearVal=0.1, farVal=100.0)
    #show where the camera is pointing
    line_ID = p.addUserDebugLine([xA, yA, zA], [xB, yB, zB], lineColorRGB=[0, 0, 1])

    # adding a line to see if it shows up in the depth image 
    #test_ID = p.addUserDebugLine([0, 0, 0], [10, -10, 10], lineColorRGB=[1, 0, 0])

    # Making images
    imgs = p.getCameraImage(IMG_W, IMG_H,
                            view_matrix,
                            projection_matrix, shadow=True,
                            renderer=p.ER_BULLET_HARDWARE_OPENGL)
    
    return imgs, view_matrix, projection_matrix


###         SIMULATION      ###
'''
Example usage:

plt.switch_backend('TkAgg')
fig = plt.figure()
imgs, view_matrix, projection_matrix = update_camera(robot)
points = get_point_cloud(im = imgs, width=IMG_W, height=IMG_H,view_matrix=view_matrix, proj_matrix=projection_matrix)
plot_point_cloud(points=points)

while (i<100000000000):
    i = i+1
plt.close()

'''


