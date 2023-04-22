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
    collision = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
    visual = p.createVisualShape(p.GEOM_SPHERE, radius=radius,
                                             rgbaColor=color)
    obstacle = p.createMultiBody(baseMass=0,
                                    baseCollisionShapeIndex=collision,
                                    baseVisualShapeIndex=visual,
                                    basePosition=center)
    return obstacle


def add_obstacle_cylinder(bullet_client, center, radius=0.1, length=0.1, color=[0.4, 0.4, 0.4, 1]):
    collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=length)
    visual = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=length, 
                                             rgbaColor=color)
    obstacle = p.createMultiBody(baseMass=0,
                                    baseCollisionShapeIndex=collision,
                                    baseVisualShapeIndex=visual,
                                    basePosition=center)
    return obstacle


def add_big_cylinder():
    baseOrientationCylinder = p.getQuaternionFromEuler([math.pi/2, 0, math.pi/2])
    cylinderMainUid = p.loadURDF("rmp2/utils/blocks/cylinder_main.urdf", \
            basePosition=[5.,-1.75,1], baseOrientation=baseOrientationCylinder, globalScaling=1)
    return cylinderMainUid

def add_cuboid():
    # Cuboid_Uid = p.loadURDF("rmp2/utils/blocks/cuboid_main.urdf", basePosition=[0.5,-0.8,1.25], \
    #                        globalScaling=10)
    Cuboid_Uid = p.loadURDF("rmp2/utils/blocks/cuboid_main.urdf", basePosition=[0.2,-0.5,0.25], \
                            globalScaling=10)
    return Cuboid_Uid