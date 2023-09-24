"""
Uses the config file to find intrinsic parameters.
Calculates point cloud manually and plots on matplotlib.
Environemnt is dynamic.
Increased speed by using voxel sampling.
Correct."""

import pybullet as p
import math
from math import pi
import numpy as np
import matplotlib.pyplot as plt
import yaml
from sklearn.neighbors import KDTree



class CameraIntrinsic():
    """
    Intrinsic model of a pinhole camera.

    Attributes:
        name (str): Name of the camera (must match the format of the corresponding config file)
            eg. name = 'pybullet_sim_cam' for config file: 'pybullet_sim_cam_config.yaml' (must be located in configs folder)
        width (int): The width in pixels of the image from the camera.
        height(int): The height in pixels of the imgage from the camera.
        K: The intrinsic camera matrix.
    """

    def __init__(self, cam_name):
        self.name = cam_name
        self.width = None
        self.height = None
        self.fx =  None
        self.fy =  None
        self.cx =  None
        self.cy =  None
        self.fov = None
        self.K = None
        self.fov: None
        self.nearVal: None
        self.farVal: None
        
        self.read_from_config()

    
    def read_from_config(self):
        """
        Setting the intrinsic parameters based on the the yaml config file.
        """
        file_name = "./rmp2/configs/" + self.name + "_config.yaml"
        with open(file_name, "r") as yamlfile:
            data = yaml.safe_load(yamlfile)
        
        self.width = data['intrinsic_params']['img_width']
        self.height = data['intrinsic_params']['img_height']
        
        self.fx =  data['intrinsic_params']['fx']
        self.fy =  data['intrinsic_params']['fy']
        self.cx =  data['intrinsic_params']['cx']
        self.cy =  data['intrinsic_params']['cy']
        self.fov = data['intrinsic_params']['fov']
        self.nearVal = data['intrinsic_params']['nearVal']
        self.farVal = data['intrinsic_params']['farVal']       
        
        self.aspect = float(self.width/self.height)
        self.fx = self.width / (2 * np.tan(np.deg2rad(self.fov) / 2))
        self.fy = self.fx/ self.aspect
        self.cx, self.cy = self.width / 2.0, self.height / 2.0
        self.K = np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])
            
    def get_projection_matrix(self):
        """
        Returns the projection matrix calculated using the camera's intrinsic parameters.
        """
        projection_matrix = p.computeProjectionMatrixFOV(fov=self.fov, aspect=self.aspect, nearVal=self.nearVal, farVal=self.farVal)
        return projection_matrix
        
class Camera_Extrinsic():
    """Extrinsic model of a camera in the world.

    Attributes:
        name (str): Name of the camera (must match the format of the corresponding config file)
            eg. name = 'pybullet_sim_cam' for config file: 'pybullet_sim_cam_config.yaml' (must be located in configs folder)
        dist_x (float): How far forward from the robot's base the camera is positioned.
        dist_y(float): How far left from the robot's base the camera is positioned.
        dist_z (float): How far up from the robot's base the camera is positioned.
        pitch (float): The camera's rotation in place about the y-axis (only changes view not position).
        yaw (rads): The camera's rotation in place about the z-axis (only changes view not position).
        
        Note that there is no roll as that would mean the camera is on an angle.
    """
    def __init__(self, cam_name):
        self.name = cam_name
        self.dist_x = None
        self.dist_y = None
        self.dist_z = None
        self.pitch = None
        self.yaw = None
        self.cam_dist = None
        
        self.read_from_config()
        
    def read_from_config(self):
        """
        Setting the extinsic parameters based on the the yaml config file.
        """
        file_name = "./rmp2/configs/" + self.name + "_config.yaml"
        with open(file_name, "r") as yamlfile:
            data = yaml.safe_load(yamlfile)
        
        self.dist_x = data['extrinsic_params']['dist_x']
        self.dist_y = data['extrinsic_params']['dist_y']
        self.dist_z = data['extrinsic_params']['dist_z']
        self.pitch = data['extrinsic_params']['pitch']
        self.yaw = data['extrinsic_params']['yaw']
        self.cam_dist = data['extrinsic_params']['cam_dist']
    
    def get_view_matrix(self, base_position):
        """
        Returns the view matrix which is calculated using the extrinsic parameters of the camera. 
        Uses world coordinates but assumes the camera is the initial reference frame for translations. 
        """
        # Set the camera to be at the base of the robot then move it accordinly
        xA, yA, zA = base_position
        xA = xA + self.dist_x 
        yA = yA + self.dist_y
        zA = zA + self.dist_z  

        # Apply pitch and yaw 
        xB = xA + math.cos(self.yaw) * math.cos(self.pitch) * self.cam_dist
        yB = yA + math.sin(self.yaw) * math.cos(self.pitch) * self.cam_dist
        zB = zA + math.sin(self.pitch) * self.cam_dist

        view_matrix = p.computeViewMatrix(
                            cameraEyePosition=[xA, yA, zA],
                            cameraTargetPosition=[xB, yB, zB],
                            cameraUpVector=[0, 0, 1.0]
                        )
        
        #show where the camera is pointing
        line_ID = p.addUserDebugLine([xA, yA, zA], [xB, yB, zB], lineColorRGB=[0, 0, 1])
        
        return view_matrix
    
class Camera():
    """
    Model of a simulated camera in pybullet that has both intrinsic and extrinsic properties.
    """
    def __init__(self, bullet_client):
        
        # pybullet variables
        self.bullet_client = bullet_client
        
        # simulated camera configs
        self.name = 'pybullet_sim_cam'
        self.cam_intrinsic = CameraIntrinsic(self.name)
        self.cam_extrinsic = Camera_Extrinsic(self.name)
        
        # initialise the view and projection matrix
        self.view_matrix = None
        self.projection_matrix = None
        
        # matplotlib variables
        self.ax = None
        self.figure = None
        self.scatter = None
    
    def setup_point_cloud(self, robot):
        """
        Sets up the plot used to display the point cloud
        """
        plt.close('all')
        # to run GUI event loop
        plt.ion()
        
        self.figure = plt.figure()
        
        self.ax = self.figure.add_subplot(111, projection='3d')
        self.scatter = self.ax.scatter([], [], [], s=36)

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        
        self.ax.view_init(0, 90)
        
        self.ax.axes.set_xlim3d(left=0, right=1.2) 
        self.ax.axes.set_ylim3d(bottom=-0.5, top=0.5) 
        self.ax.axes.set_zlim3d(bottom=0, top=0.5) 

        # setting title
        plt.title("Point Cloud", fontsize=20)
        
        # calculate view and projection matrix - assumes robot doesn't move during sim
        base_pos, base_orn = self.bullet_client.getBasePositionAndOrientation(robot.robot_uid)

        self.view_matrix = self.cam_extrinsic.get_view_matrix(base_pos)
        self.projection_matrix = self.cam_intrinsic.get_projection_matrix()
    
        
    def step_sensing(self, voxel_size=0.05):
        """
        Captures camera data in the current pybullet environment, plots the point cloud and returns points 
        representing the point cloud of the current environment. 
        """
        imgs = self.update_camera()
        points = self.get_point_cloud(imgs, voxel_size)
        return points
    
    def update_camera(self):
        """
        Returns the simulated images for the current state of the pybullet environment.
        
        Adapted from https://ai.aioz.io/guides/robotics/2021-05-19-visual-obs-pybullet/ 
        """
        # Making images
        imgs = self.bullet_client.getCameraImage(self.cam_intrinsic.width, self.cam_intrinsic.height,
                                self.view_matrix,
                                self.projection_matrix, shadow=True,
                                renderer=self.bullet_client.ER_TINY_RENDERER)
        
        return imgs
    
        
    def get_point_cloud(self, im, voxel_size):
        """
        Uses the depth image from the given images to return a set of points in a point cloud. 
        Uses voxel downsampling to reduce the number of points in the point cloud.
        
        Adapted from https://github.com/bulletphysics/bullet3/issues/1924
        """
        # get a depth image
        # "infinite" depths will have a value close to 1
        depth = im[3]

        # create a 4x4 transform matrix that goes from pixel coordinates (and depth values) to world coordinates
        proj_matrix = np.asarray(self.projection_matrix).reshape([4, 4], order="F")
        view_matrix = np.asarray(self.view_matrix).reshape([4, 4], order="F")
        tran_pix_world = np.linalg.inv(np.matmul(proj_matrix, view_matrix))

        # create a grid with pixel coordinates and depth values
        y, x = np.mgrid[-1:1:2 / self.cam_intrinsic.height, -1:1:2 / self.cam_intrinsic.width]
        y *= -1.
        x, y, z = x.reshape(-1), y.reshape(-1), depth.reshape(-1)
        h = np.ones_like(z)

        pixels = np.stack([x, y, z, h], axis=1)
        # filter out "infinite" depths
        pixels = pixels[z < 1.00]
        pixels[:, 2] = 2 * pixels[:, 2] - 1

        # turn pixels to world coordinates
        points = np.matmul(tran_pix_world, pixels.T).T
        points /= points[:, 3: 4]
        points = points[:, :3]
        
        
        # Do voxel downsampling - equations recieved from chatGPT - "I have a set of 21,000 points representing a point cloud as an numpy array in python and I want to apply voxel downsampling on these. How do I do that?"
        # print("OG size", len(points))

        # Calculate the minimum and maximum bounds of your point cloud
        min_bound = np.min(points, axis=0)
        max_bound = np.max(points, axis=0)

        # Calculate voxel indices for each point
        voxel_indices = ((points - min_bound) / voxel_size).astype(int)

        # Create a KD-Tree from the original point cloud
        kdtree = KDTree(points)

        # Hash the voxel indices to identify unique voxels
        voxel_hash = voxel_indices[:, 0] + voxel_indices[:, 1] * (max_bound[0] / voxel_size) + voxel_indices[:, 2] * (max_bound[0] / voxel_size) * (max_bound[1] / voxel_size)

        # Use unique voxel hashes to select one point per voxel
        unique_voxel_hashes, downsampled_indices = np.unique(voxel_hash, return_index=True)

        # Get the downsampled point cloud
        downsampled_point_cloud = points[downsampled_indices]
        
        return downsampled_point_cloud

    
    def plot_point_cloud_dynamic(self, points):
        """
        Updates the plot with the given points.
        """
        x_coords, y_coords, z_coords = points[:, 0], points[:, 1], points[:, 2]
        self.scatter._offsets3d = (x_coords, y_coords, z_coords)
        plt.pause(0.001) #TODO