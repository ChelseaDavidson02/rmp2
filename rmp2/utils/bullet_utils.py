"""
helper functions for pybullet
"""

import pybullet as p
import math

def add_goal(bullet_client, position, radius=0.05, color=[0.0, 1.0, 0.0, 1]):
    collision = -1
    visual = bullet_client.createVisualShape(p.GEOM_SPHERE, radius=radius,
                                             rgbaColor=color)
    goal = bullet_client.createMultiBody(baseMass=0,
                                    baseCollisionShapeIndex=collision,
                                    baseVisualShapeIndex=visual,
                                    basePosition=position)
    return goal

def add_collision_goal(bullet_client, position, radius=0.05, color=[0.0, 1.0, 0.0, 1]):
    collision = bullet_client.createCollisionShape(p.GEOM_SPHERE, radius=0.01)
    visual = -1
    goal = bullet_client.createMultiBody(baseMass=0,
                                    baseCollisionShapeIndex=collision,
                                    baseVisualShapeIndex=visual,
                                    basePosition=position)
    return goal


def add_obstacle_ball(bullet_client, center, radius=0.1, color=[0.4, 0.4, 0.4, 1]):
    collision = bullet_client.createCollisionShape(bullet_client.GEOM_SPHERE, radius=radius)
    visual = bullet_client.createVisualShape(bullet_client.GEOM_SPHERE, radius=radius,
                                             rgbaColor=color)
    obstacle = bullet_client.createMultiBody(baseMass=0,
                                    baseCollisionShapeIndex=collision,
                                    baseVisualShapeIndex=visual,
                                    basePosition=center)
    return obstacle


def add_obstacle_cylinder(bullet_client, center, radius=1, length=20, color=[0.4, 0.4, 0.4, 1], angle_rotation_rads = [math.pi/2, 0, 0]):
    collision = bullet_client.createCollisionShape(bullet_client.GEOM_CYLINDER, radius=radius, height=length,
                                                flags=bullet_client.GEOM_FORCE_CONCAVE_TRIMESH)
    visual = bullet_client.createVisualShape(bullet_client.GEOM_CYLINDER, radius=radius, length=length,
                                                rgbaColor=color)
    rotation = bullet_client.getQuaternionFromEuler(angle_rotation_rads)  # Rotate the cylinder by 90 degrees around the x-axis if angle_rotation_rads unchanged
    obstacle = bullet_client.createMultiBody(baseMass=0,
                                                baseCollisionShapeIndex=collision,
                                                baseVisualShapeIndex=visual,
                                                basePosition=center,
                                                baseOrientation=rotation)
    return obstacle

def add_obstacle_cuboid(bullet_client, center, size=[0.1, 0.1, 0.1], color=[0.4, 0.4, 0.4, 1], angle_rotation_rads = [0,0,0]):
    collision = bullet_client.createCollisionShape(bullet_client.GEOM_BOX, halfExtents=size)
    visual = bullet_client.createVisualShape(bullet_client.GEOM_BOX, halfExtents=size, rgbaColor=color)
    rotation = bullet_client.getQuaternionFromEuler(angle_rotation_rads)  # Rotate the cylinder by 90 degrees around the x-axis if angle_rotation_rads unchanged
    obstacle = bullet_client.createMultiBody(baseMass=0,
                                 baseCollisionShapeIndex=collision,
                                 baseVisualShapeIndex=visual,
                                 basePosition=center,
                                 baseOrientation=rotation)
    return obstacle
