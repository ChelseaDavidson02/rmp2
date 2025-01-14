"""
Contains an intrinsic and extrinsic model of a camera with a camera object that uses these to capture simulated images. 
Uses the pybullet_sim__cam_config.yaml config file to obtain extrinsic and intrinsic parameters.
Calculates point cloud and plots it using matplotlib - this is updated at each time step to allow for dynamic environments.
Increased speed by using voxel sampling.
"""
import pybullet as p
import math
from math import pi
import numpy as np
import matplotlib.pyplot as plt
import yaml
from sklearn.neighbors import KDTree
from scipy.spatial import cKDTree
import time

from PIL import Image
import cv2





class CameraIntrinsic():
    """
    Intrinsic model of a pinhole camera.

    Attributes:
        name (str): Name of the camera (must match the format of the corresponding config file)
            eg. name = 'pybullet_sim_cam' for config file: 'pybullet_sim_cam_config.yaml' (must be located in configs folder)
        width (int): The width in pixels of the image from the camera.
        height(int): The height in pixels of the imgage from the camera.
        fov(int): Field of view in degrees (along x plane)
        nearVal (int): Position of near plane (camera cannot see points closer than this plane)
        farVal (int): Position of far plane (camera cannot see points beyond this plane)
    """

    def __init__(self, cam_name):
        self.name = cam_name
        self.width = None
        self.height = None
        self.fx =  None
        self.fy =  None
        self.cx =  None
        self.cy =  None
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
        # line_ID = p.addUserDebugLine([xA, yA, zA], [xB, yB, zB], lineColorRGB=[0, 0, 1])
        
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
        self.scatter1 = None
        self.scatter2 = None
        
        # sytem variables 
        self.ideal_pose = None
        self.voxel_size = None
        self.sim_running = False
        self.goal_distance = 0
        self.max_obs_num = None

        # pybullet sim variables
        self.robot = None
        self.goal_uid = None
        self.goal_line_ID = None
        
        # Performance variables
        self.error_values = []
        self.y_distances = []
        self.distances_from_surface = []
        self.camera_updates_time = []
        

        # Deafault filenames for saving results
        self.filename_suffix = "default"
        self.output_folder="data"
        self.figure_title="Title"
    
    def setup_point_cloud(self, robot, goal_uid, goal_distance, voxel_size, time_step):
        """
        Sets up the plot used to display the point cloud
        """
        # Store system parameters
        self.robot = robot
        self.goal_uid = goal_uid
        self.goal_distance = goal_distance
        self.voxel_size = voxel_size
        self.time_step = time_step
        
        plt.close('all')
        # to run GUI event loop
        plt.ion()
        
        self.figure = plt.figure()
        
        # Initialse scatters which are used to display point cloud
        self.ax = self.figure.add_subplot(111, projection='3d')
        self.scatter1 = self.ax.scatter([], [], [], s=66)
        self.scatter2 = self.ax.scatter([], [], [],  c='g', marker='o', s=50, label='Specific Point')

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        
        self.ax.view_init(0, 90)
        
        self.ax.axes.set_xlim3d(left=0, right=1.2) 
        self.ax.axes.set_ylim3d(bottom=-0.5, top=0.5) 
        self.ax.axes.set_zlim3d(bottom=0, top=1.0) 

        # setting title
        plt.title("Point Cloud", fontsize=20)
        
        # calculate view and projection matrix - assumes robot's base doesn't move during sim
        base_pos, base_orn = self.bullet_client.getBasePositionAndOrientation(robot.robot_uid)

        self.view_matrix = self.cam_extrinsic.get_view_matrix(base_pos)
        self.projection_matrix = self.cam_intrinsic.get_projection_matrix()

    def activate_sim(self):
        """Informs the camera when the RMP graph has been constructed and the algorithm is online"""
        self.sim_running = True
        self.camera_updates_time = []
        
    def step_sensing(self, current_y_distance, max_obstacle_num = None):
        """
        Captures camera data in the current pybullet environment, plots the point cloud and returns points 
        representing the point cloud of the current environment. Stores computation times required to complete various processes.
        """
        self.max_obs_num = max_obstacle_num
        t0 = time.time()
        imgs = self.update_camera()
        t1 = time.time()
        all_points = self.get_point_cloud(imgs)
        t2 = time.time()
        points = self.restrict_ROI(all_points)
        downsampled_points = self.downsample_point_cloud(points, self.voxel_size)
        final_points = self.restrict_point_num(max_obstacle_num, downsampled_points)
        t3 = time.time()
        eef_info = p.getLinkState(self.robot.robot_uid, self.robot.eef_uid, computeLinkVelocity=1, computeForwardKinematics=1)
        t4 = time.time()
        goal_position = self.get_goal_point(points, eef_info[0], current_y_distance)
        t5 = time.time()

        
        # print("Time taken updating the camera: ", t1-t0)
        # print("Time taken getting point cloud: ", t2-t1)
        # print("Time taken downsampling: ", t3-t2)
        # print("Time taken getting goal point: ", t5-t4)
        # print("Total time: ", (t5-t4) + (t3-t2))
        self.camera_updates_time.append((t5-t4) + (t3-t2))

        # print("Total points", len(all_points))
        # print("ROI points", len(points))
        # print("downsampled points", len(downsampled_points))
        
        return final_points, goal_position
            

    def update_camera(self):
        """
        Returns the simulated images for the current state of the pybullet environment.
        Makes the target sphere transparent while the image is being taken so it does not appear in the image.
        
        Adapted from https://ai.aioz.io/guides/robotics/2021-05-19-visual-obs-pybullet/ 
        """
        # Making images
        self.bullet_client.changeVisualShape(self.goal_uid, -1, rgbaColor=[0, 0, 0, 0])
        imgs = self.bullet_client.getCameraImage(self.cam_intrinsic.width, self.cam_intrinsic.height,
                                self.view_matrix,
                                self.projection_matrix, shadow=True,
                                renderer=self.bullet_client.ER_BULLET_HARDWARE_OPENGL)
        self.bullet_client.changeVisualShape(self.goal_uid, -1, rgbaColor=[0, 1, 0, 1])
        
        return imgs
    
        
    def get_point_cloud(self, im):
        """
        Uses the depth image from the given images to return a set of points in a point cloud. 
        Uses voxel downsampling to reduce the number of points in the point cloud.
        
        Adapted from https://github.com/bulletphysics/bullet3/issues/1924
        """
        # get a depth image
        # "infinite" depths will have a value close to 1
        depth_image = im[3]
        
        # get a segmentation mask image
        segmentation_mask = im[4]

        # Create a 4x4 transform matrix that goes from pixel coordinates (and depth values) to world coordinates
        proj_matrix = np.asarray(self.projection_matrix).reshape([4, 4], order="F")
        view_matrix = np.asarray(self.view_matrix).reshape([4, 4], order="F")
        tran_pix_world = np.linalg.inv(np.matmul(proj_matrix, view_matrix))

        # Create a grid with pixel coordinates, depth values, and segmentation mask values
        y, x = np.mgrid[-1:1:2 / self.cam_intrinsic.height, -1:1:2 / self.cam_intrinsic.width]
        y *= -1.
        x, y, z = x.reshape(-1), y.reshape(-1), depth_image.reshape(-1)
        segmentation_mask = segmentation_mask.reshape(-1) # reshape the segmentation mask

        # Filter out "infinite" depths and pixels belonging to the robot or goal sphere (segmentation_mask == robot_uid when the pixels correspond to the robot)
        pixels_w_obs_only = np.logical_and(segmentation_mask != self.goal_uid, segmentation_mask != self.robot.robot_uid)
        valid_pixels = np.logical_and(pixels_w_obs_only, z < 1.00)
        # valid_pixels = np.logical_and(z < 1.00, segmentation_mask != self.robot.robot_uid)
        x, y, z = x[valid_pixels], y[valid_pixels], z[valid_pixels]
        h = np.ones_like(z)

        pixels = np.stack([x, y, z, h], axis=1)
        pixels[:, 2] = 2 * pixels[:, 2] - 1

        # Transform pixels to world coordinates
        points = np.matmul(tran_pix_world, pixels.T).T
        points /= points[:, 3:4]
        points = points[:, :3]
        
        return points
    
    def restrict_ROI(self, point_cloud, max_x_distance = 1.0, max_y_distance = 0.5, max_z_distance = 1.1):
        """
        Filters out point cloud data that falls outside of the task space.
        """
        mask = (np.abs(point_cloud[:, 0]) <= max_x_distance) & \
           (np.abs(point_cloud[:, 1]) <= max_y_distance) & \
           (np.abs(point_cloud[:, 2]) <= max_z_distance)
        
        return point_cloud[mask]
    
    def downsample_point_cloud(self, points, voxel_size):
        """
        Applies voxel downsampling to the point cloud.
        
        params:
            points (array): points contained in the point cloud
            voxel_size (float): size of voxel when applying voxel downsampling

        """

        # Do voxel downsampling - equations recieved from chatGPT - "I have a set of 21,000 points representing a point cloud as an numpy array in python and I want to apply voxel downsampling on these. How do I do that?"

        grid_indices = (points / voxel_size).astype(int)

        # Combine grid indices into a single unique identifier for each voxel
        voxel_ids = grid_indices[:, 0] + grid_indices[:, 1] * 1000 + grid_indices[:, 2] * 1000000

        # Sort points based on voxel IDs
        sorted_indices = np.argsort(voxel_ids)

        # Group points by voxel IDs
        voxel_groups = np.split(points[sorted_indices], np.where(np.diff(voxel_ids[sorted_indices]))[0] + 1)

        # Calculate the centroid of each voxel
        centroids = np.array([group.mean(axis=0) for group in voxel_groups])
        return centroids

    
    def restrict_point_num(self, obstacle_num, points):
        """
        Randomly samples or duplicates points in the point cloud to ensure that the number of points are exactly equal to the obstacle number.
        params:
            points (array): points contained in the point cloud
            voxel_size (float): size of voxel when applying voxel downsampling
        """
        if obstacle_num is not None:
            if len(points) > obstacle_num: # if we have too many points, randomly sample
                randomly_sampled_indices = np.random.choice(points.shape[0], size=obstacle_num, replace=False)
                points = points[randomly_sampled_indices] 
            elif len(points) < obstacle_num:# if we have too little points, randomly duplicate
                random_indices = np.random.choice(len(points), size=obstacle_num-len(points), replace=True)
                duplicated_points = points[random_indices]

                # Stack the duplicated pointss
                points = np.vstack((points, duplicated_points))
        return points
    
    def plot_point_cloud_dynamic(self, points, goal_point):
        """
        Updates the plot with the given points.

        params:
            points (array): points contained in the point cloud
            goal_point (float): current goal point
        """
        t0 = time.time()
        x_coords, y_coords, z_coords = points[:, 0], points[:, 1], points[:, 2]
        self.scatter1._offsets3d = (x_coords, y_coords, z_coords)
        
        x_cp, y_cp, z_cp = goal_point[0], goal_point[1], goal_point[2]
        self.scatter2._offsets3d = (np.array([x_cp]), np.array([y_cp]), np.array([z_cp]))

        plt.pause(0.0001)
        t1 = time.time()
        # print("Time taken plotting: ", t1-t0)
        
    def get_goal_point(self, points, eef_pos, current_y_distance):
        """
        Finds the current goal point in the environment.

        params:
            points (array): points contained in the point cloud
            eef_pos (float): current position of the robot's end effector
            current_y_distance (float): current change in distance from the start of the simulation (how far the obstacles have moved)
        """
        # Remove previous line ID of goal
        if self.goal_line_ID is not None:
            self.bullet_client.removeUserDebugItem(self.goal_line_ID)
            
        # Create a kd-tree from your point cloud data
        kdtree = cKDTree(points)
        eef_location = np.array(eef_pos)
        
        # Query the kd-tree to find the index of the closest point
        closest_point_index = kdtree.query(eef_location)[1]
        
        # Get the actual coordinates of the closest point
        closest_point = points[closest_point_index]
        
        # Calculate the vector between the two points - this goes from object towards eef
        vector = closest_point - eef_location

        # Calculate the length (magnitude) of the vector
        vector_length = np.linalg.norm(vector)

        # Calculate the unit vector by dividing the vector by its length
        unit_vector = vector / vector_length
        
        goal_point = closest_point - (unit_vector*self.goal_distance) 

        # Calculate and store the error:
        if self.sim_running:
            distance_from_surface = np.linalg.norm(closest_point - eef_location)
            self.distances_from_surface.append(distance_from_surface)
            # print("Distance:", distance_from_surface)
            error = float(abs(abs(distance_from_surface) - self.goal_distance))
            # print("error:", error)
            self.error_values.append(error)
            self.y_distances.append(current_y_distance)
    
        if self.ideal_pose is not None:
            # To mitigate drift from the obstacle RMP, choose a goal position (there are infinitely many) which would move the robot towards its initial eef position and maintain a smooth trajectory between time steps
            # Find deviation between current eef pos and initial in terms of x and y  
            delta_y = float(eef_location[1] - self.ideal_pose[1])
            
            angle_deg = 40 * delta_y
            angle_radians = np.deg2rad(angle_deg)
                
            # Create the rotation matrix
            R_z = np.array([[np.cos(angle_radians), -np.sin(angle_radians), 0],
               [np.sin(angle_radians), np.cos(angle_radians), 0],
               [0, 0, 1]])

            # Perform the rotation
            unit_vector = np.dot(R_z, unit_vector)

            # Apply in the vertical direction as well - not in the x as the robot must be able to avoid the obstacles
            delta_z = float(self.ideal_pose[2]- eef_location[2])

            angle_deg_z = 40 * delta_z
            angle_radians_z = np.deg2rad(angle_deg_z)
                
            # Create the rotation matrix
            R_y = np.array([
                [np.cos(angle_radians_z), 0, np.sin(angle_radians_z)],
                [0, 1, 0],
                [-np.sin(angle_radians_z), 0, np.cos(angle_radians_z)]
            ])
            # Perform the rotation
            unit_vector = np.dot(R_y, unit_vector)
            
        goal_point = closest_point - (unit_vector*self.goal_distance) 

        # Show the current goal vector
        self.goal_line_ID = p.addUserDebugLine(goal_point, closest_point, lineColorRGB=[0, 1, 0])
        
        if self.ideal_pose is None:
            self.ideal_pose = goal_point
        
        # Return the goal point
        return goal_point
    
    def save_last_img(self):
        """
        Function used to save an image of the environment from a birds eye view once the simulation has completed.
        """
        view_matrix = self.bullet_client.computeViewMatrix(
                            cameraEyePosition=[-1.0, 3, 3.5],
                            cameraTargetPosition=[0.9, 3, 1.6],
                            cameraUpVector=[0, 0, 1.0]
                        )
        proj_matrix = self.bullet_client.computeProjectionMatrixFOV(fov=90,
                                                        aspect=float(960) / 720,
                                                        nearVal=0.1,
                                                        farVal=100.0)
        imgs = self.bullet_client.getCameraImage(width=960,
                                                    height=720,
                                                    viewMatrix=view_matrix,
                                                    projectionMatrix=proj_matrix,
                                                    renderer=self.bullet_client.ER_BULLET_HARDWARE_OPENGL)
        rgb_image = imgs[2]

        pil_image = Image.fromarray(rgb_image)

        # Save the image to a file
        pil_image.save(f"data/random_trials/output_image_{self.filename_suffix}.png")

    def plot_error(self, step_comp_times, policy_calc_times, first_policy_eval_time):
        """
        Plots the percentage error over the trial and saves computation time data, graph building time data, error data, and distance from surface data.
        """
        plt.figure()
        # Store error data
        distance_array = np.array(self.y_distances)
        error_values = np.array(self.error_values)
        camera_time_values = np.array(self.camera_updates_time)
        policy_time_values = np.array(policy_calc_times)
        step_time_values = np.array(step_comp_times)
        data = np.column_stack((distance_array, error_values))
        # Store data in case want to change plots later
        np.savetxt(f'{self.output_folder}/error_data_{self.filename_suffix}.csv', data, delimiter=",", header="distance,error", comments="")

        # Store distance data
        surface_dis_data = np.array(self.distances_from_surface)
        data = np.column_stack((distance_array, surface_dis_data))
        np.savetxt(f'{self.output_folder}/surface_dist_data_{self.filename_suffix}.csv', data, delimiter=",", header="distance,dist_from_surface", comments="")

        cam_time_av = np.average(camera_time_values)
        policy_time_av = np.average(policy_time_values)
        step_time_av = np.average(step_time_values)
        total_average_time = cam_time_av + policy_time_av + step_time_av

        time_data = np.column_stack((self.max_obs_num , first_policy_eval_time, cam_time_av, policy_time_av, step_time_av, total_average_time))
        np.savetxt(f'{self.output_folder}/time_data_{self.filename_suffix}.csv', time_data, delimiter=",", header="obstacle_num, first_policy_eval_time, camera_time_av, policy_time_av, step_time_av, total_time_step_average", comments="")
        
        percent_array = np.full(error_values.shape, 100/self.goal_distance)
        percentage_error_values =  error_values * percent_array
        
        plt.plot(distance_array, percentage_error_values)
        plt.xlabel('Y distance')
        plt.ylabel('Percentage error')
        plt.title(f'Error Over Time - {self.figure_title}')
        plt.grid(True, 'both')
        # Add text to the bottom center
        average_error = np.average(self.error_values)
        average_percentage_error = (average_error / self.goal_distance) *100
        text = "Average percentage error: %.3f%%" % (average_percentage_error)
        plt.text(0.88, -0.1, text, horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

        plt.savefig(f'{self.output_folder}/error_plot_{self.filename_suffix}.png')

        self.save_last_img()

        print("-------------------END OF TEST LOG-------------------")
        print("Obstacle number:", self.max_obs_num)
        print("First policy evaluation time - need to build RMP-tree:", first_policy_eval_time)
        print("Average camera update time:",  cam_time_av)
        print("Average policy eval time:",  policy_time_av)
        print("Average step action time:",  step_time_av)
        print("Total average time", total_average_time)
        print("-----------------------------------------------------")


        
        