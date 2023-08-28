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
    
        
    def get_point_cloud(self, im, view_matrix, proj_matrix):
        # adapted from https://github.com/bulletphysics/bullet3/issues/1924
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
        # Main functions found from https://stackoverflow.com/questions/5179589/continuous-3d-plotting-i-e-figure-update 
        x_coords, y_coords, z_coords = points[:, 0], points[:, 1], points[:, 2]
        self.scatter._offsets3d = (x_coords, y_coords, z_coords) # override the previous scatter plot
        plt.pause(0.001) #TODO
        
    
    def setup_plot(self):
        # to run GUI event loop
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
        
    
    def update_camera(self, robot, cam_yaw, cam_pitch, cam_x_dist=0.1):
        # adapted from https://ai.aioz.io/guides/robotics/2021-05-19-visual-obs-pybullet/ 
        # finding position of the camera
        base_pos, base_orn = self.bullet_client.getBasePositionAndOrientation(robot.robot_uid)

        # # set the camera at the end-effector link
        # end_effector_link_index = 9

        # # Get the end-effector's position and orientation
        # eef_pos, eef_orient, _, _, _, _ = self.bullet_client.getLinkState(robot.robot_uid, end_effector_link_index)
        # yaw = cam_yaw #self.bullet_client.getEulerFromQuaternion(eef_orient)[-1] + cam_yaw #- pi/6
        xA, yA, zA = base_pos
        xA = xA + cam_x_dist # make the camera a little further forward than the robot - stops it from being in frame

        xB = xA + math.cos(cam_yaw) * math.cos(cam_pitch) * self.CAM_DISTANCE
        yB = yA + math.sin(cam_yaw) * math.cos(cam_pitch) * self.CAM_DISTANCE
        zB = zA + math.sin(cam_pitch) * self.CAM_DISTANCE

        view_matrix = self.bullet_client.computeViewMatrix(
                            cameraEyePosition=[xA, yA, zA],
                            cameraTargetPosition=[xB, yB, zB],
                            cameraUpVector=[0, 0, 1.0]
                        )

        projection_matrix = self.bullet_client.computeProjectionMatrixFOV(
                                fov=60, aspect=float(self.IMG_W/self.IMG_H), nearVal=0.1, farVal=5.0)
        #show where the camera is pointing
        line_ID = self.bullet_client.addUserDebugLine([xA, yA, zA], [xB, yB, zB], lineColorRGB=[0, 0, 1])

        # Making images
        imgs = self.bullet_client.getCameraImage(self.IMG_W, self.IMG_H,
                                view_matrix,
                                projection_matrix, shadow=True,
                                renderer=self.bullet_client.ER_BULLET_HARDWARE_OPENGL)
        
        return imgs, view_matrix, projection_matrix
    
    def step_sensing(self, robot, cam_yaws, cam_pitches):
        """
        Captures camera data in the current pybullet environment, plots the point cloud and returns points 
        representing the point cloud of the current environment. 
        """
        combined_points = []
        for i in range(len(cam_yaws)):
            imgs_i, view_matrix_i, projection_matrix_i = self.update_camera(robot, cam_yaw=cam_yaws[i], cam_pitch = cam_pitches[i])
            points_i = self.get_point_cloud(im = imgs_i ,view_matrix=view_matrix_i, proj_matrix=projection_matrix_i)
            combined_points.append(points_i)
            # self.bullet_client.removeUserDebugItem(line_ID_i) #removing the line which represents the current camera line of sight
        combined_points_stack = np.vstack(combined_points)
        unique_combined_points = np.unique(combined_points_stack, axis=0) # removing any points which were found in multiple camera angles
        
        
        return unique_combined_points

        
        
    


# def main():
    
#     # Initialize the simulation
#     time_step = 1/240.
#     iterations = 100
#     cam = Camera(p)

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