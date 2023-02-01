#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 16:15:21 2020

@author: hp
"""

import rospy
import roslib
import tf
import numpy as np

import urx
import math3d as m3d
#import open3d as o3d

def listen_tf(src, target):
    rospy.init_node('tf_listener', anonymous=True)
    tf_listener = tf.TransformListener()
    tf_listener.waitForTransform(target, src, rospy.Time(), rospy.Duration(5))
    t, q = tf_listener.lookupTransform(target, src, rospy.Time())
    
    o_m3d = m3d.Orientation()
    q_m3d = m3d.UnitQuaternion(q[3], q[0], q[1], q[2])
    o_m3d.set_quaternion(q_m3d)

    t_m3d = m3d.Vector(t[0], t[1], t[2])
    T_m3d = m3d.Transform(o_m3d, t_m3d)
    
    return T_m3d



#robot_ip = "158.132.172.193"
robot_ip = "192.168.0.2"

#robot_port = 
rob = urx.Robot(robot_ip, use_rt=True)

#rob.set_tcp((0, 0, 0, 0, 0, 0))
# rob.set_tcp((0, -0.08, 0, 0, 0, 0))
rob.set_tcp((0,-0.15, 0, 0, 3.14, 0))

# pose=np.array([0.25,0.20,0.20,-1.57,0,0])
# rob.movel(pose,0.01,0.01)


#base_to_marker = rob.get_pose()
marker_to_base = rob.get_pose()
print("marker_to_base")
print(marker_to_base)


#camera_to_marker = listen_tf("camera_color_optical_frame", "ar_marker_avg")
camera_to_marker = listen_tf("camera_rgb_optical_frame", "ar_marker_avg")


print('camera_to_marker')
print('quaternion')
print(camera_to_marker)

#base_to_camera = base_to_marker * camera_to_marker.inverse()
camera_to_base = marker_to_base * camera_to_marker
print("camera_to_base")
print(camera_to_base)
rob.close()






