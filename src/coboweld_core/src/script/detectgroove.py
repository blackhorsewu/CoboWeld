#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
*  Chinese National Engineering Research Centre for Steel Structure
*  (Hong Kong Branch)
*
*  Hong Kong Polytechnic University
*
*  Author: Victor W H Wu
*  Date: 27 October 2022.
*
*  Revision 1: 14 November, 2022.
*     To import Open3d ros helper for open3d and ros point cloud conversions.
*     The original conversion does not work anymore in ROS Noetic. Details of this
*     package, refer to README.md.
*
*  Revision 2: 16 November, 2022.
*     The function select_down_sample was replaced by select_by_index in Open3d
*     This script tries to detect groove in the workpiece.
*
*  Revision 3: 3 February, 2023.
*     Moved the code from cncweld_core to CoboWeld.
*
*  Revision 4: 9 February, 2023.
*     Introduce URx into CoboWeld.
*
*  Revision 5: 16 March, 2023.
*     Parameters adjusted for the Y-tube joint.
*
*  Revision 6: 30 March, 2023.
*     Use tf for coordinate transformations.
*
*  Revision 7: 6 April, 2023.
*     Use ArUCo marker to locate the groove.
*
*  Copyright (c) 2022-2023 Victor W H Wu
*
*  Description:
*    This script is based on Jeffery Zhou's work.
*    It uses Open3D to handle point clouds and visions, ROS and RViz to visualize
*    the operations. URx is used to actually move the UR5 manipulator.
*
*  It requires:
*  rospy, 
*  numpy, 
*  open3d, 
*  math,
*  open3d_ros_helper,
*  urx,
*  tf,
*  OpenCV,
*  cv_bridge
*
'''
# Imports for ROS
import roslib
import rospy
import tf
import sys
import os

import cv2
from cv_bridge import CvBridge, CvBridgeError

import numpy as np
import open3d as o3d
import copy
import math

from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField, Image, CameraInfo
from geometry_msgs.msg import PoseStamped, Pose, PoseArray, Point
from visualization_msgs.msg import Marker, MarkerArray
import sensor_msgs.point_cloud2 as pc2

import vg # Vector Geometry
from scipy.spatial.transform import Rotation as R
from scipy import interpolate
import scipy.spatial as spatial

import time
import urx

import csv

#*************************#
# Define Parameter values #
#*************************#
# 1. Feature value neighbours
feature_neighbours = 6
# 2. Maximum distance between cluster neighbours
cluster_neighbour_distance = 0.007 # m or 8mm
# 3. Minimum cluster members
min_cluster_memb = 6
# 4. Point cloud thickness in thin_line
thickness = 0.007
# 5. Voxel size
voxelsize = 0.001 # m or 1mm
# 6. Normal estimation neighbourhood
# radius
my_radius = 0.01 # m or 12mm
# maximum nearest neighbours
maxnn = 350
# 7. Delete percentage of feature values
percentage = 0.98

execute = False

#########################################################
#                                                       #
#                     ArUCo Marker                      #
#                                                       #
#########################################################


# This is for conversion from Open3d point cloud to ROS point cloud
# Note: Add `.ravel()` to the end of line 261 in the `open3d_ros_helper.py` before it can work
# Refer to README.md 
from open3d_ros_helper import open3d_ros_helper as orh

ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

aruco_type = "DICT_5X5_100"

# Call back function to receive a ROS point cloud published by the RealSense D435 camera
def callback_roscloud(ros_cloud):
    global received_ros_cloud

    received_ros_cloud = ros_cloud

def getColourCamInfo():
  global intrinsic_camera, distortion
  info = rospy.wait_for_message('/d435/color/camera_info', CameraInfo, timeout=10)
  intrinsic_camera = np.array(info.K)
  intrinsic_camera = intrinsic_camera.reshape(3, 3)
  # print('intrinsic_camera: ', intrinsic_camera)
  distortion = np.array(info.D)
  # print('distortion: ', distortion)

bridge = CvBridge()

def urxPose_to_rosPose(inPose, inFrame):
  outPose = PoseStamped()
  outPose.header.frame_id = inFrame
  outPose.pose.position.x = inPose[0]
  outPose.pose.position.y = inPose[1]
  outPose.pose.position.z = inPose[2]
  r = R.from_rotvec(inPose[3:])
  orient = r.as_quat()
  outPose.pose.orientation.x = orient[0]
  outPose.pose.orientation.y = orient[1]
  outPose.pose.orientation.z = orient[2]
  outPose.pose.orientation.w = orient[3]
  outPose.header.stamp = rospy.Time.now()
  return outPose

# Aruco Marker pose estimation
def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):

  # print('pose_estimation.')
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  cv2.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
  parameters = cv2.aruco.DetectorParameters_create()

  corners, ids, rejected_img_points = cv2.aruco.detectMarkers(
      gray,
      cv2.aruco_dict,
      parameters=parameters
    )

  if len(corners) > 0:
    for i in range(0, len(ids)):
      rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(
          corners[i], 
          0.0443, # the marker is 45mm square
          matrix_coefficients,
          distortion_coefficients
        )

    tvec = tvec[0,0]
    # print('tvec: ', tvec.shape)
    rvec = rvec[0,0]
    # rvec = rvec * [-pi, 0.0, 0.0]
    # print('rvec: ', rvec.shape)
    
    # The marker is pointing towards the camera
    # It is necessary for it to point away from the camera
    # This can be done by rotating about the X-axis by pi
    r = R.from_rotvec(rvec) # get the r from rotation vector
    ar_mat = r.as_matrix()  # get the rotation matrix from r
    # multiply it with this matrix to rotate it about X-axis
    ar_mat = ar_mat * [[1, 0, 0],[0, -1, 0],[0, 0, -1]]

    ar_r = R.from_matrix(ar_mat)  # get a new r from the new matrix
    ar_rvec = ar_r.as_rotvec()    # get the new rotvec

    out_pose = np.hstack((tvec[:3], ar_rvec))

    pub_pose.publish(urxPose_to_rosPose(out_pose, 'd435_color_optical_frame'))

  return out_pose # URx pose

def getMarkerPose():
  # get the colour image from camera
  # print('getMarkerPose.')
  image = rospy.wait_for_message('/d435/color/image_raw', Image, timeout=10)
  global cv_image
  p_listener = tf.TransformListener()
  try:
    cv_image = bridge.imgmsg_to_cv2(image, "bgr8")
    # Estimate the ArUCo marker pose, URx pose
    urx_pose = pose_estimation(
                cv_image,
                ARUCO_DICT[aruco_type], 
                intrinsic_camera, 
                distortion
              )
    # print('pose estimated.')
    ros_pose = urxPose_to_rosPose(urx_pose, 'd435_color_optical_frame')
    try:
      now = rospy.Time.now()
      # Wait for transform to base
      p_listener.waitForTransform("base", "/d435_color_optical_frame", now, rospy.Duration(4.0))
      # Do the actual transform
      pose = p_listener.transformPose("base", ros_pose)
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
      print("tf error!")
  except ChildProcessError as e:
    print(e)
  return pose # ROS pose

# The input, marker_pose, is a ROS pose with time stamp and frame 'base'.
# Outputs target for depth camera, in URx format and publish the pose in RViz.
#
# this function will transform the pose to one suitable for the
# wrist_3_frame to use without the need to set the tcp for urx.
# This function will also publish the pose in RViz
# It will return a URx pose for move
#
def setDepthCameraPose(marker_pose, x, y, z):
  # print('setDepthCameraPose.')
  quat = marker_pose.pose.orientation
  quat = [quat.x, quat.y, quat.z, quat.w]
  r = R.from_quat(quat)
  rotvec = r.as_rotvec()
  matrix = r.as_matrix()
  # X-axis is the 1st column of the rotation matrix
  x_axis = matrix[:, 0]
  # Y-axis is the 2nd column of the rotation matrix
  y_axis = matrix[:, 1]
  # Z-axis is the 3rd column of the rotation matrix
  z_axis = matrix[:, 2]

  # Find the original position first
  posit = [marker_pose.pose.position.x,
           marker_pose.pose.position.y,
           marker_pose.pose.position.z]

  # Find the out pose position
  outPosit = posit + (x_axis * x)
  outPosit = outPosit + (y_axis * y)
  outPosit = outPosit - (z_axis * z)

  outPose = [outPosit[0], outPosit[1], outPosit[2], rotvec[0], rotvec[1], rotvec[2]]

  return outPose

# Given the target frame, say the camera frame and its pose in URx format
# return the pose in URx of tcp (wrist_3_link) and publish it in RViz.
# This is done by requesting the transformation from tf but then apply it
# in the reverse direction. This can only be done by hand coding it, not by 
# applying tf.
def setTcpPose(sourceFrame, inPose):
  # convert the inPose into ROS pose first
  inROSpose = PoseStamped()
  inROSpose.pose.position.x = inPose[0]
  inROSpose.pose.position.y = inPose[1]
  inROSpose.pose.position.z = inPose[2]
  rotvec = inPose[3:]
  r = R.from_rotvec(rotvec)
  quat = r.as_quat()
  rotvec = r.as_rotvec()
  matrix = r.as_matrix()
  # X-axis is the 1st column of the rotation matrix
  x_axis = matrix[:, 0]
  # Y-axis is the 2nd column of the rotation matrix
  y_axis = matrix[:, 1]
  # Z-axis is the 3rd column of the rotation matrix
  z_axis = matrix[:, 2]
  inROSpose.pose.orientation.x = quat[0]
  inROSpose.pose.orientation.y = quat[1]
  inROSpose.pose.orientation.z = quat[2]
  inROSpose.pose.orientation.w = quat[3]
  inROSpose.header.frame_id = 'base'
  inROSpose.header.stamp = rospy.Time.now()
  pub_campose.publish(inROSpose)
  now = rospy.Time.now()
  d_listener.waitForTransform(sourceFrame, 'wrist_3_link', 
                               now, rospy.Duration(4.0))
  (trans, rot) = d_listener.lookupTransform(sourceFrame, 'wrist_3_link', now)
  
#  print('trans: ', trans)
#  print('rot: ', rot)
#  tcpROSpose = PoseStamped()

  tcpROSpose = PoseStamped()
  inPosit = [inROSpose.pose.position.x,
             inROSpose.pose.position.y,
             inROSpose.pose.position.z]
  
  outPosit = inPosit + (x_axis * trans[0])
  outPosit = outPosit + (y_axis * trans[1])
  outPosit = outPosit + (z_axis * trans[2])

  tcpPose = [outPosit[0], outPosit[1], outPosit[2], rotvec[0], rotvec[1], rotvec[2]]

  # print('publish tcp pose.')
  tcpROSpose = urxPose_to_rosPose(tcpPose, 'base')
  pub_tcpPose.publish(tcpROSpose)
  
  
  return tcpPose

# Input poses are the ur_poses returned from "find_orientation"
# that is used by URx and UR5, [x, y, z, rx, ry, rz]
# The torch tip is Leung Sir drawing [x, y, z] = [0.0, -0.105, -0.365]
# Since the orientation is not changed, then [rx, ry, rz] are the same
def setTorchPose(poses):
  torchPoses = []
  for pose in poses:
    # print('pose: ', pose)
    r = R.from_rotvec(pose[3:])
    position = pose[:3]
    # print('position: ',position)
    matrix = r.as_matrix()
    x_axis = matrix[:, 0]
    y_axis = matrix[:, 1]
    z_axis = matrix[:, 2]
    torchPosition = position                         # X position is not changed
    # print('torchPosition: ', torchPosition)
    torchPosition = torchPosition + (y_axis * 0.097) # Y position +ve 0.105m
    # make the torch 10mm farther away for welding wire
    torchPosition = torchPosition - (z_axis * 0.375) # Z position -ve 0.375m
    torchPose = np.hstack((torchPosition, pose[3:]))
    torchPoses.append(torchPose)

  # publish_path_poses(torchPoses)
  return torchPoses


def transform_cam_wrt_base(pcd):

  # Updated on 30 March 2023 by Victor Wu.
  # Cannot use tf.transformPointCloud because pcd is not a ROS pointcloud2!
  try:
    now = rospy.Time.now()
    listener.waitForTransform("base", "/d435_depth_optical_frame", now, rospy.Duration(4.0))
    (trans, rot) = listener.lookupTransform("base", "/d435_depth_optical_frame", now)
    r = R.from_quat(rot)
    transformation = np.vstack((np.hstack((r.as_matrix(), np.vstack((trans))
                                          )
                                         ),
                                [0, 0, 0, 1]
                               )
                              )
    pcd_copy = copy.deepcopy(pcd).transform(transformation)
  except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
    print("tf error!")

  return pcd_copy

#########################################################
#                                                       #
#                    Feature detection                  #
#                                                       #
#########################################################
def normalize_feature(feature_value_list):
    
    max_value = feature_value_list.max()
    min_value = feature_value_list.min()
    feature_value_range = max_value - min_value
    normalized_feature_value_list = (feature_value_list - min_value)/ feature_value_range

    return np.array(normalized_feature_value_list)


# find feature value of each point of the point cloud and put them into a list
def find_feature_value(pcd):

  # Build a KD (k-dimensional) Tree for Flann
  # Fast Library for Approximate Nearest Neighbor
  pcd_tree = o3d.geometry.KDTreeFlann(pcd)

  # Treat pcd.points as an numpy array of n points by m attributes of a point
  # The first dimension, shape[0], of this array is the number of points in this point cloud.
  pc_number = np.asarray(pcd.points).shape[0]

  feature_value_list = []
  
  # This is very important. It specifies the attribute that we are using to find the feature
  # when it is pcd.normals, it is using the normals to find the feature,
  # n_list is an array of normals of all the points
  n_list = np.asarray(pcd.normals)

  # a // b = integer quotient of a divided by b
  # so neighbor (number of neighbors) whichever is smaller of 30 or the quotient 
  # of dividing the number of points by 100
  # neighbour = min(pc_number//100, 30)
  neighbour = feature_neighbours
  print("Feature value neighbour: ", neighbour)
  # for every point of the point cloud
  for index in range(pc_number):
      
      # Search the k nearest neighbour for each point of the point cloud.
      # The pcd.points[index] is the (query) point to find the nearest neighbour for.
      # 'neighbour', found above, is the number of neighbours to be searched.
      [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[index], neighbour)

      # get rid of the query point in the neighbourhood
      idx = idx[1:]

      # n_list, computed above, is an array of normals of every point.
      # 'vector' is then a vector with its components the arithmetic mean of every
      # element of all the k neighbours of that (query) point
      # This can be called the CENTROID of the NORMALs of its neighbours
      vector = np.mean(n_list[idx, :], axis=0)
      
      # the bigger the feature value, meaning the normal of that point is more 
      # different from its neighbours
      feature_value = np.linalg.norm(
          vector - (n_list[index, :] * np.dot(vector, n_list[index, :])))
      feature_value_list.append(feature_value)

  return np.array(feature_value_list)

def cluster_groove_from_point_cloud(pcd):

    # eps - the maximum distance between neighbours in a cluster, originally 0.005,
    # at least min_points to form a cluster.
    # returns an array of labels, each label is a number. 
    # labels of the same cluster have their labels the same number.
    # if this number is -1, that is this cluster is noise.
    # DBSCAN - a Density-Based Algorithm for discovering Clusters in large spacial Databases
    # labels = np.array(pcd.cluster_dbscan(eps=0.005, min_points=3, print_progress=True))
    # eps (float) - Density parameter that is used to find neighbouring points
    # the EPSilon radius for all points.
    # min_points (int) Minimum number of points to form a cluster
    labels = np.array(pcd.cluster_dbscan(eps=cluster_neighbour_distance, 
                                         min_points=min_cluster_memb, 
                                         print_progress=False))

    # np.unique returns unique labels, label_counts is an array of number of that label
    label, label_counts = np.unique(labels, return_counts=True)

    # Find the largest cluster
    ## [-1] is the last element of the array, minus means counting backward.
    ## So, after sorting ascending the labels with the cluster with largest number of
    ## members at the end. That is the largest cluster.
    label_number1 = label[np.argsort(label_counts)[-1]]

    if label_number1 == -1:
        if label.shape[0]>1:
            label_number = label[np.argsort(label_counts)[-2]]
        elif label.shape[0]==1:
            # sys.exit("can not find a valid groove cluster")
            print("can not find a valid groove cluster")
    
    # Pick the points belong to the largest cluster
    groove_index = np.where(labels == label_number1)
    groove1 = pcd.select_by_index(groove_index[0])

    return groove1

#########################################################
#                                                       #
#                    Line extraction                    #
#                                                       #
#########################################################
def thin_line(points, point_cloud_thickness=thickness, iterations=1, sample_points=0):

    if sample_points != 0:
        points = points[:sample_points]

    # Sort points into KDTree for nearest neighbours computations later
    point_tree = spatial.cKDTree(points)

    # Initially, the array for transformed points is empty
    new_points = []
    
    # Initially, the array for regression lines corresponding ^^ points is empty
    regression_lines = []

    for point in point_tree.data:

        # Get list of points within specified radius {point_cloud_thickness}
        points_in_radius = point_tree.data[point_tree.query_ball_point(point, point_cloud_thickness)]

        # Get mean of points within radius
        data_mean = points_in_radius.mean(axis=0)

        # Calculate 3D regression line/principal component in point form with 2 coordinates
        uu, dd, vv = np.linalg.svd(points_in_radius - data_mean)
        linepts = vv[0] * np.mgrid[-1:1:2j][:, np.newaxis]
        linepts += data_mean
        regression_lines.append(list(linepts))

        # Project original point onto 3D regression line
        ap = point - linepts[0]
        ab = linepts[1] - linepts[0]
        point_moved = linepts[0] + np.dot(ap, ab) / np.dot(ab, ab) * ab
        
        new_points.append(list(point_moved))

    return np.array(new_points), regression_lines

def sort_points(points, regression_lines, sorted_point_distance=0.01):

    # Index of point to be sorted
    index = 0

    # sorted points array for left and right of initial point to be sorted
    sort_points_left = [points[index]]
    sort_points_right = []

    # Regression line of previously sorted point
    regression_line_prev = regression_lines[index][1] - regression_lines[index][0]

    # Sort points into KDTree for nearest neighbours computation later
    point_tree = spatial.cKDTree(points)

    # Iterative add points sequentially to the sort_points_left array
    while 1:
        # Calculate regression line vector, makes sure line vector is similar direction as
        # previous regression line
        v = regression_lines[index][1] - regression_lines[index][0]
        if np.dot(regression_line_prev, v) / (np.linalg.norm(regression_line_prev) * np.linalg.norm(v)) < 0:
            v = regression_lines[index][0] - regression_lines[index][1]
        regression_line_prev = v

        # Find point {distR_point} on regression line distance {sorted_point_distance} 
        # from  original point 
        distR_point = points[index] + ((v / np.linalg.norm(v)) * sorted_point_distance)

        # Search nearest neighbours of distR_point within radius {sorted_point_distance / 3}
        points_in_radius = point_tree.data[point_tree.query_ball_point(distR_point, sorted_point_distance / 1.5)]
        if len(points_in_radius) < 1:
            break

        # Neighbour of distR_point with smallest angle to regression line vector is selected
        # as next point in order
        nearest_point = points_in_radius[0]
        distR_point_vector = distR_point - points[index]
        nearest_point_vector = nearest_point - points[index]
        for x in points_in_radius:
            x_vector = x - points[index]
            if vg.angle(distR_point_vector, x_vector) < vg.angle(distR_point_vector, nearest_point_vector):
                nearest_point_vector = nearest_point - points[index]
                nearest_point = x
        index = np.where(points == nearest_point)[0][0]

        # Add nearest point to 'sort_points_left' array
        sort_points_left.append(nearest_point)

    # Do it again but in the other direction of initial starting point
    index = 0
    regression_line_prev = regression_lines[index][1] - regression_lines[index][0]
    while 1:
        # Calculate regression line vector, makes sure line vector is similar direction as
        # previous regression line
        v = regression_lines[index][1] - regression_lines[index][0]
        if np.dot(regression_line_prev, v) / (np.linalg.norm(regression_line_prev) * np.linalg.norm(v)) < 0:
            v = regression_lines[index][0] - regression_lines[index][1]
        regression_line_prev = v

        # Find point {distR_point} on regression line distance {sorted_point_distance} 
        # from  original point 
        #
        # Now vector is SUBTRACTED instead of ADDED from the point to go in other direction
        #               ==========            =====
        distR_point = points[index] - ((v / np.linalg.norm(v)) * sorted_point_distance)

        # Search nearest neighbours of distR_point within radius {sorted_point_distance / 3}
        points_in_radius = point_tree.data[point_tree.query_ball_point(distR_point, sorted_point_distance / 1.5)]
        if len(points_in_radius) < 1:
            break

        # Neighbour of distR_point with smallest angle to regression line vector is selected
        # as next point in order
        nearest_point = points_in_radius[0]
        distR_point_vector = distR_point - points[index]
        nearest_point_vector = nearest_point - points[index]
        for x in points_in_radius:
            x_vector = x - points[index]
            if vg.angle(distR_point_vector, x_vector) < vg.angle(distR_point_vector, nearest_point_vector):
                nearest_point_vector = nearest_point - points[index]
                nearest_point = x
        index = np.where(points == nearest_point)[0][0]

        # Add nearest point to 'sort_points_left' array
        sort_points_right.append(nearest_point)

    # Combine 'sort_points_right' and 'sort_points_left'
    sort_points_right = sort_points_right[: : -1]
    sort_points_right.extend(sort_points_left)
    sort_points_right = np.flip(sort_points_right, 0)

    return np.array(sort_points_right)

#########################################################
#                                                       #
#                    Path processing                    #
#                                                       #
#########################################################
# To generate a welding path for the torch. This is only a path and should not be called a trajectory!
def generate_path(groove):

    points = np.asarray(groove.points)

    # Thin & sort points
    thinned_points, regression_lines = thin_line(points)
    sorted_points = sort_points(thinned_points, regression_lines)

    x = sorted_points[:, 0]
    y = sorted_points[:, 1]
    z = sorted_points[:, 2]

    try:
       (tck, u), fp, ier, msg = interpolate.splprep([x, y, z], s=float("inf"), full_output=1)
    except TypeError:
      print("\n ************* End ************* ")
      robot.stop()
      # close the communication, otherwise python will not shutdown properly
      robot.close()
      print('UR5 closed')
      rospy.signal_shutdown("Finished shutting down")

    u_fine = np.linspace(0, 1, x.size*2)

    # Evaluate points on B-spline
    try:
      x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
    except TypeError:
      print("\n ************* End ************* ")
      robot.stop()
      # close the communication, otherwise python will not shutdown properly
      robot.close()
      print('UR5 closed')
      rospy.signal_shutdown("Finished shutting down")

    sorted_points = np.vstack((x_fine, y_fine, z_fine)).T

    path_pcd = o3d.geometry.PointCloud()
    path_pcd.points = o3d.utility.Vector3dVector(sorted_points)

    return path_pcd


def publish_path_poses(poses):

  PoseList_torch_rviz = PoseArray()

  for pose in poses:

    r = R.from_rotvec(pose[3:])
    orientation = r.as_quat()

    #publish to torch_pose
    torch_pose_rviz = Pose()
    #tip position
    torch_pose_rviz.position.x = pose[0]
    torch_pose_rviz.position.y = pose[1]
    torch_pose_rviz.position.z = pose[2]
    #tip orientation
    torch_pose_rviz.orientation.x = orientation[0]
    torch_pose_rviz.orientation.y = orientation[1]
    torch_pose_rviz.orientation.z = orientation[2]
    torch_pose_rviz.orientation.w = orientation[3]
    #publish torch tip pose trajectory
    PoseList_torch_rviz.poses.append(torch_pose_rviz)

  # PoseList_torch_rviz.header.frame_id = 'd435_depth_optical_frame'
  PoseList_torch_rviz.header.frame_id = 'base'
  PoseList_torch_rviz.header.stamp = rospy.Time.now()
  pub_poses.publish(PoseList_torch_rviz)

def publish_camposes(poses):
  camposelist = PoseArray()
  for pose in poses:
    r = R.from_rotvec(pose[3:])
    orientation = r.as_quat()
    campose = Pose()
    campose.position.x = pose[0]
    campose.position.y = pose[1]
    campose.position.z = pose[2]
    campose.orientation.x = orientation[0]
    campose.orientation.y = orientation[1]
    campose.orientation.z = orientation[2]
    campose.orientation.w = orientation[3]
    camposelist.poses.append(campose)
  camposelist.header.frame_id = 'base'
  camposelist.header.stamp = rospy.Time.now()
  pub_camposes.publish(camposelist)

def publish_tcpPoses(poses):
  tcpPoselist = PoseArray()
  for pose in poses:
    r = R.from_rotvec(pose[3:])
    orientation = r.as_quat()
    tcpPose = Pose()
    tcpPose.position.x = pose[0]
    tcpPose.position.y = pose[1]
    tcpPose.position.z = pose[2]
    tcpPose.orientation.x = orientation[0]
    tcpPose.orientation.y = orientation[1]
    tcpPose.orientation.z = orientation[2]
    tcpPose.orientation.w = orientation[3]
    tcpPoselist.poses.append(tcpPose)
  tcpPoselist.header.frame_id = 'base'
  tcpPoselist.header.stamp = rospy.Time.now()
  pub_tcpPoses.publish(tcpPoselist)

# After a welding path is generated, it is necessary to find the orientation of the 
# welding torch before a pose for each point can be sent to the robot for execution.
# groove is the detected groove with respect to the camera frame.
def find_orientation(path): 

  #path = transform_cam_wrt_base(groove)
  path = np.asarray(path.points)

  # A list of all the Rotation Vectors used to specify the orientation in UR format
  rotvecs = []

  # for each of the points on the path
  for i in range(path.shape[0]):
    # Find the vector pointing from one point to the next, diff_x,
    # if this is the last point, subtract the point before
    if i == path.shape[0] - 1:
      diff_x = path[i] - path[i - 1]
    # otherwise subtract from the next point
    else:
      diff_x = path[i + 1] - path[i]

    # use the horizontal line as the Y-axis
    # *** remember, the groove, as it is now, is still in the camera frame
    # therefore the horizontal line is the camera's X-axis
    y_axis = np.array([0.0, 0.0, -1.0]) # the negative X-axis of the camera
    y_axis = y_axis/np.linalg.norm(y_axis, axis=0) # normalize it
    # The diff_x cross the Y-axis (the horizontal line) gives the Z-axis
    # pointing into the workpiece (the tube)
    z_axis = np.cross(diff_x, y_axis)
    z_axis = z_axis/np.linalg.norm(z_axis, axis=0) # normalize it
    # The Y-axis cross the Z-axis gives the X-axis
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis/np.linalg.norm(x_axis, axis=0)

    '''
    # diff_x cross a vertical line gives the Y-axis
    y_axis = np.cross(diff_x, np.array([0.0, 0.0, 1.0]))
    y_axis = y_axis/np.linalg.norm(y_axis, axis=0) # normalize it
    # The diff_x cross the Y-axis gives the Z-axis
    z_axis = np.cross(diff_x, y_axis)
    z_axis = z_axis/np.linalg.norm(z_axis, axis=0) # normalize it
    # The Y-axis cross the Z-axis gives the X-axis
    x_axis = np.cross(y_axis, z_axis)
    # since both Y-axis and Z-axis are normalized therefore no need to normalize it

    # if move along the Y-axis, then it should be called diff_y
    # diff_y cross a vertical line gives the X-axis
    x_axis = np.cross(diff_x, np.array([0.0, 0.0, -1.0]))
    x_axis = x_axis/np.linalg.norm(x_axis, axis=0) # normalize it
    # The diff_y cross the X-axis gives the Z-axis
    z_axis = np.cross(x_axis, diff_x)
    z_axis = z_axis/np.linalg.norm(z_axis, axis=0) # normalize it
    # The Z-axis cross the X-axis gives the Y-axis
    y_axis = np.cross(z_axis, x_axis)
    # since both Z-axis and X-axis are normalized therefore no need to normalize it
    '''

    # Use the scipy.spatial.transform library Rotation to find the Rotation Vector
    # from the X, Y, Z axis
    r = R.from_matrix(np.vstack((x_axis, y_axis, z_axis)).T)
    rotvec = r.as_rotvec()

    # if this is the first point use it to work out the approach point 
    if i == 0:
      # Needs the Quaternion to publish its pose
      orientation = r.as_quat()
      # Needs the Rotation Vector to send to URx (UR5)
      first_rotvec = r.as_rotvec()
      # The approach point is set to 50mm from the first point along the Z axis
      init_pos = z_axis * 0.075

    # if this is the last point use it to work out the leaving point 
    if i == (path.shape[0]-1):
      # Needs the Quaternion to publish its pose
      orientation = r.as_quat()
      # Needs the Rotation Vector to send to URx (UR5)
      last_rotvec = r.as_rotvec()
      # The approach point is set to 50mm from the first point along the Z axis
      last_pos = z_axis * 0.05

    rotvecs.append(rotvec)
  # End for loop

  # Construct the Approach point
  # path[0] is the position of the first point
  # path[0] - init_pos takes the approach from first point by init_pos 
  # in negative Z direction, because the minus operator.
  approach = path[0] - init_pos
  approach = np.hstack((approach, first_rotvec))

  # Construct the Leaving point
  # path[-1] is the last point
  # path[-1] - last_pos takes the leaving from last point by last_pos in Z direction
  leaving = path[-1] - last_pos
  leaving = np.hstack((leaving, last_rotvec))

  # Put approach, path, and leaving all in a tuple and then stack them vertically.
  ur_poses = np.vstack((approach, np.hstack((path, np.array(rotvecs))), leaving))

  return ur_poses

#########################################################
#                                                       #
#                 Point cloud processing                #
#                                                       #
#########################################################
'''
# marker_pose is the pose of the ArUCo marker in ROS format
def setSidePose(marker_pose, x, y, z):
  # First, find the target from marker_pose
  # the target is about 30mm below the marker_pose
  target = [marker_pose.pose.position.x,
            marker_pose.pose.position.y + 0.03,
            marker_pose.pose.position.z - 0.3] 
  
  quat = marker_pose.pose.orientation
  quat = [quat.x, quat.y, quat.z, quat.w]
  r = R.from_quat(quat)
  rotvec = r.as_rotvec()
  matrix = r.as_matrix()
  # X-axis is the 1st column of the rotation matrix
  x_axis = matrix[:, 0]
  # Y-axis is the 2nd column of the rotation matrix
  y_axis = matrix[:, 1]
  # Z-axis is the 3rd column of the rotation matrix
  z_axis = matrix[:, 2]

  outPos = [targetX + x, targetY + y, targetZ - z]

  depthCamPose = []
  hypot = math.sqrt(x**2 + y**2 + z**2) # hypotenuse
  sin = y / hypot
  cos = x / hypot
  rx = [[ 1.0,  0.0,  0.0],
        [ 0.0,  cos, -sin],
        [ 0.0,  sin,  cos]]
  ry = [[ cos,  0.0,  sin],
        [ 0.0,  1.0,  0.0],
        [-sin,  0.0,  cos]]
  rz = [[ cos, -sin,  0.0],
        [ sin,  cos,  0.0],
        [ 0.0,  0.0,  1.0]]
  print('centre: ', centre)
  r = R.from_rotvec(centre[3:])
  matrix = r.as_matrix()
  print('matrix: ', matrix)
  fst_mat = np.matmul(rz, ry)
  snd_mat = np.matmul(fst_mat, rx)
  outR = np.matmul(matrix, snd_mat)
  # outR = np.matmul(np.matmul(np.matmul(matrix, rz), ry), rx)
  out_r = R.from_matrix(outR)
  outRvec = out_r.as_rotvec()
  outOrient = out_r.as_quat()
  depthCamPose = np.hstack((outPos, outRvec))
  ''''''
  camPose = PoseStamped()
  camPose.header.frame_id = 'base'
  camPose.pose.position.x = centre[0]
  camPose.pose.position.y = centre[1]
  camPose.pose.position.z = centre[2]
  camPose.pose.orientation.x = outOrient[0]
  camPose.pose.orientation.y = outOrient[1]
  camPose.pose.orientation.z = outOrient[2]
  camPose.pose.orientation.w = outOrient[3]
  camPose.header.stamp = rospy.Time.now()
  pub_campose.publish(camPose)
  
  return depthCamPose
  '''
#########################################################
#                                                       #
#                        Work flow                      #
#                                                       #
#########################################################
def detect_groove_workflow(pcd):

  original_pcd = pcd

  global delete_percentage

  # 1. Down sample the point cloud
  ## a. Define a bounding box for cropping
  bbox = o3d.geometry.AxisAlignedBoundingBox(
      # x right, y down, z forward; for the camera
      # min_bound = (-0.025, -0.25, 0.2), 
      # max_bound = (0.05, 0.1, 0.5)  
      # 50mm x 50mm plane with 0.5m depth
      #min_bound = (-0.015, -0.025, 0.2), 
      #max_bound = (0.035, 0.025, 0.5)  
      min_bound = (-0.5, -0.10, 0.25), 
      max_bound = (0.8, 0.10, 0.34)  
  )

  ## b. Define voxel size
  # Declared at the top as 1mm cube for each voxel

#    print("\n ************* Before cropping ************* ")
#    rviz_cloud = orh.o3dpc_to_rospc(pcd, frame_id="d435_depth_optical_frame")
#    pub_captured.publish(rviz_cloud)

  # print('voxel size: no downsample')
  print('voxel size: ', voxelsize)
  # Actually down sampling the point cloud captured
  pcd = pcd.voxel_down_sample(voxel_size = voxelsize)
  
  # Cropping the down sampled point cloud
  pcd = pcd.crop(bbox)

  ### it was 'remove_none_finite_points' in Open3D version 0.8.0... but
  ### it is  'remove_non_finite_points'  in Open3D version 0.15.1...
  pcd.remove_non_finite_points()
  print("Point cloud cropped.")
  rviz_cloud = orh.o3dpc_to_rospc(pcd, frame_id="d435_depth_optical_frame")
  pub_captured.publish(rviz_cloud)
  
  '''
  if first_round == True:
    print("Do you want to save the new point cloud?")
    reply = input("Y for yes: ")
    if (reply == "Y") or (reply == "y"):
      filename = input("Please input filename: ")
      o3d.io.write_point_cloud(filename, pcd)
    # else do nothing
  # else do nothing
  '''

  ## c. Count the number of points afterwards
  pc_number = np.asarray(pcd.points).shape[0]
  print('Total number of points: ', pc_number)

  # 2. Estimate normal toward camera location and normalize it.
  pcd.estimate_normals(
      search_param = o3d.geometry.KDTreeSearchParamHybrid(
          radius = my_radius, max_nn = maxnn
      )
  )

  print('normal estimation neighbours: radius: ', my_radius, 'max_nn: ', maxnn)

  pcd.normalize_normals()
  pcd.orient_normals_towards_camera_location(camera_location = [0.0, 0.0, 0.0])

  # 3. Use different geometry features to find groove
  #    Use asymmetric normals as a feature to find groove
  feature_value_list = find_feature_value(pcd)
  normalized_feature_value_list = normalize_feature(feature_value_list)

  # 4. Delete low value points and cluster
  delete_points = int(pc_number * delete_percentage)

#    pcd_selected = pcd.select_down_sample(
  pcd_selected = pcd.select_by_index(
      ## np.argsort performs an indirect sort
      ## and returns an array of indices of the same shape
      ## that index data along the sorting axis
      ## in ascending order by default. So the smaller value first
      ## and the largest value at the end
      np.argsort(normalized_feature_value_list)[delete_points:]
      ## therefore this is a list of indices of the point cloud
      ## with the top 5 percent feature value
  )

  # define an inner bounding box for border removing
  # 5mm less on each side
  ibbox = o3d.geometry.AxisAlignedBoundingBox(
     min_bound = ( -1.0, -0.094, 0.255), 
     max_bound = ( 0.54, 0.094, 0.34)  
  )

  pcd_selected = pcd_selected.crop(ibbox)

  print('Selected: ', pcd_selected)

  reply = input("Featured points selected.\nc to continue others to quit.")
  if (reply == "c"):
    pcd_selected.paint_uniform_color([0, 1, 0])
    rviz_cloud = orh.o3dpc_to_rospc(pcd_selected, frame_id="d435_depth_optical_frame")
    pub_selected.publish(rviz_cloud)

    groove = cluster_groove_from_point_cloud(pcd_selected)
    print('Groove: ',np.asarray(groove))
  else:
    rospy.signal_shutdown("Finished shutting down")
    return

  groove = groove.paint_uniform_color([1, 0, 0])
  groove = transform_cam_wrt_base(groove) # *************************************************
  reply = input("Going to cluster selected points.\nc to continue others to quit.")
  if (reply == "c"):
    # rviz_cloud = orh.o3dpc_to_rospc(groove, frame_id="d435_depth_optical_frame")
    rviz_cloud = orh.o3dpc_to_rospc(groove, frame_id="base")
    pub_clustered.publish(rviz_cloud)

    # 5. Generate a path from the clustered Groove

    reply = input("Press 'c' to show path, any other to quit.")
    if (reply == "c"):
      generated_path = generate_path(groove)
      generated_path = generated_path.paint_uniform_color([0, 0, 1])

      # rviz_cloud = orh.o3dpc_to_rospc(generated_path, frame_id="d435_depth_optical_frame")
      rviz_cloud = orh.o3dpc_to_rospc(generated_path, frame_id="base")
      pub_path.publish(rviz_cloud)
    else:
      rospy.signal_shutdown("Finished shutting down")
      return
  else:
    rospy.signal_shutdown("Finished shutting down")
    return

  ur_poses = find_orientation(generated_path)
  # transform_cam_wrt_base(ur_poses)
  publish_path_poses(ur_poses)

  return(ur_poses)

#########################################################
#                                                       #
#                    Main function                      #
#                                                       #
#########################################################
if __name__ == "__main__":

  global received_ros_cloud, cv_image
  global intrinsic_camera, distortion

  if len(sys.argv) > 1:
    if sys.argv[1] == 'execute':
      execute = True
    else:
      execute = False

  # Initialize the node and name it.
  rospy.init_node('coboweld_core', anonymous=True)

  listener = tf.TransformListener()
  aruco_listener = tf.TransformListener()
  d_listener = tf.TransformListener()

  # Must have __init__(self) function for a class, similar to a C++ class constructor.

  # delete_percentage = 0.95 ORIGINAL VALUE
  delete_percentage = percentage

  received_ros_cloud = None

  # Setup subscriber for point cloud
  rospy.Subscriber('/d435/depth/color/points', PointCloud2, 
                    callback_roscloud, queue_size=1
                  )

  # Setup publishers
  pub_captured = rospy.Publisher("captured", PointCloud2, queue_size=1)
  pub_selected = rospy.Publisher("selected", PointCloud2, queue_size=1)
  pub_clustered = rospy.Publisher("clustered", PointCloud2, queue_size=1)
  pub_path = rospy.Publisher("path", PointCloud2, queue_size=1)
  pub_poses = rospy.Publisher('poses', PoseArray, queue_size=1)
  pub_pose = rospy.Publisher('/ArUCo', PoseStamped, queue_size=1)
  pub_campose = rospy.Publisher("campose", PoseStamped, queue_size=1)
  pub_camposes = rospy.Publisher("camposes", PoseArray, queue_size=1)
  pub_tcpPose = rospy.Publisher('tcp_pose', PoseStamped, queue_size=1)
  pub_tcpPoses = rospy.Publisher('tcpPoses', PoseArray, queue_size=1)

  aruco_type = "DICT_5X5_100"

  arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[aruco_type])
  arucoParams = cv2.aruco.DetectorParameters_create()

  print("\n ************* Start *************")

  # Start URx only when execute is True
  
  # Do not start URx when testing software
  

  # home1 is when it faces to the right
  home1j = [0.0001, -1.1454, -2.7596, 0.7290, 0.0000, 0.0000]
  # home2 is when it faces the CHS
  home2j = [0.6496, -1.1454, -2.7596, 0.7289, 0.0000, 0.0000]
  startG1j = [0.2173, -1.8616, -0.2579, -2.6004, 1.5741, 0.2147]
  startchsj = [0.6792, -0.4243, -2.5662, -0.1751, 0.9010, 0.0188]
  # startchs1 is the target for point cloud capturing
  startchs1j = [-0.4060, -1.4229, -2.2255, 0.5201, 1.0502, -0.0131]

  listener.waitForTransform("base", "/d435_depth_optical_frame", rospy.Time(), rospy.Duration(4.0))
  # listener.waitForTransform("/world", "/d435_depth_optical_frame", rospy.Time(), rospy.Duration(4.0))

  # Acquire colour camera information include intrinsic matrix and distortion
  # before pose of ArUCo marker can be found
  getColourCamInfo()
  # print('intrinsic_camera: ', intrinsic_camera)
  # print('distortion: ', distortion)

  robot = urx.Robot('192.168.0.103')

  # first_round = True
  while not rospy.is_shutdown():

    # move UR5 to starting point
    robot.movej(home2j, 0.4, 0.4, wait=True)
    time.sleep(0.2)

    # robot.set_tcp((0, 0, 0, 0, 0, 0))
    # time.sleep(0.3)

    # find the ArUCo marker, in the 'world' frame
    marker_pose = getMarkerPose() # ROS pose
    # print('marker_pose', marker_pose)
    # print('publish marker pose.')
    pub_pose.publish(marker_pose) # Visualize marker pose in RViz

    # move the depth camera to somewhere near the ArUCo marker.
    # Since the Marker is 45mm square, the centre is about 22.5mm away
    # from the groove edge. Assuming the groove is about 15mm wide, so
    # the d435_depth_optical_frame should be about 30mm below the marker.
    # It should also be 300mm away from the groove, along the Z-axis.
    # camPose in URx format.
    #
    # This is the CENTRE of 5 positions that the camera will capture a
    # complete point cloud of the workpiece.
    #
    # camPose0 is the CENTRE
    camPose0 = setDepthCameraPose(marker_pose,  0.00,  0.03, 0.3)
    # print('camera pose 0: ', camPose0)
    '''
    camPose1 = setDepthCameraPose(marker_pose, -0.05, -0.02, 0.3)
    # print('camera pose 1: ', camPose1)
    camPose2 = setDepthCameraPose(marker_pose,  0.05, -0.02, 0.3)
    # print('camera pose 2: ', camPose2)
    camPose3 = setDepthCameraPose(marker_pose,  0.05,  0.08, 0.3)
    # print('camera pose 3: ', camPose3)
    camPose4 = setDepthCameraPose(marker_pose, -0.05,  0.08, 0.3)
    # print('camera pose 4: ', camPose4)
    
    camPoses = [camPose0, camPose1, camPose2, camPose3, camPose4]
    # publish_camposes(camPoses)

    tcpPose0 = setTcpPose('d435_depth_optical_frame' , camPose0) # URx pose
    # print('tcpPose0: ', tcpPose0)
    tcpPose1 = setTcpPose('d435_depth_optical_frame' , camPose1) # URx pose
    # print('tcpPose1: ', tcpPose1)
    tcpPose2 = setTcpPose('d435_depth_optical_frame' , camPose2) # URx pose
    # print('tcpPose2: ', tcpPose2)
    tcpPose3 = setTcpPose('d435_depth_optical_frame' , camPose3) # URx pose
    # print('tcpPose3: ', tcpPose3)
    tcpPose4 = setTcpPose('d435_depth_optical_frame' , camPose4) # URx pose
    # print('tcpPose4: ', tcpPose4)

    tcpPoses = [tcpPose0, tcpPose1, tcpPose2, tcpPose3, tcpPose4]
    # publish_tcpPoses(tcpPoses)

    pcd = o3d.geometry.PointCloud()
    pcds = []
    for tcppose in tcpPoses:
      robot.movej_to_pose(tcppose, 0.5, 0.5, wait=True)
      # convert the received ROS pc into Open3D pc

      received_open3d_cloud = orh.rospc_to_o3dpc(received_ros_cloud)
      o3dpc = transform_cam_wrt_base(received_open3d_cloud)
      # pcd += o3dpc
      # pcd = pcd.voxel_down_sample(voxel_size=0.001)
      pcds.append(o3dpc)
      rviz_cloud = orh.o3dpc_to_rospc(o3dpc, 
                                      frame_id='base')
      # pcd.append(received_open3d_cloud)
      pub_captured.publish(rviz_cloud)
      reply = input('Hit "c" to continue, "t" to use this one :')
      if reply == 't':
        ur_poses = detect_groove_workflow(received_open3d_cloud)
        break
    '''
    tcpPose0 = setTcpPose('d435_depth_optical_frame' , camPose0) # URx pose
    robot.movej_to_pose(tcpPose0, 0.5, 0.5, wait=True)
    '''
    reply = input('Which pcd do you prefer: ')
    rviz_cloud = orh.o3dpc_to_rospc(pcds[int(reply)], frame_id='base')
    pub_captured.publish(rviz_cloud)

    ur_poses = detect_groove_workflow(pcds[int(reply)])
    '''




    # Use the marker_pose to set the Depth Camera Pose
    # Do the transformation to the wrist_3_link as well to avoid 
    # converting to and from ros pose and urx pose
    # print(tcpPose)
    # robot.movel(tcpPose, 0.1, 0.1, wait=True)

    #robot.movej_to_pose(tcpPose, 0.4, 0.4, wait=True)

    ##########################################################################

    # setTcpPose(marker_pose, -0.05, -0.08, -0.3)

    ''''''
    if not received_ros_cloud is None:
      received_open3d_cloud = orh.rospc_to_o3dpc(received_ros_cloud)

      rviz_cloud = orh.o3dpc_to_rospc(received_open3d_cloud, 
                                      frame_id="d435_depth_optical_frame")
      pub_captured.publish(rviz_cloud)

      # These ur_poses are with respect to "base" FOR "wrist_3_link"
      ur_poses = detect_groove_workflow(received_open3d_cloud)

      if execute:
        reply = input('Do you want to move to the Approaching Point? Y for yes: ')
        if (reply == "y"):
          # Before these poses can be used to move the welding torch
          # they must be transformed with respect to "base" FOR "torch"
          torchPoses = setTorchPose(ur_poses)

          # Move to the Approach point
          robot.movel(torchPoses[0], acc=0.1, vel=0.1, wait=True)

          input('\nPress any to continue')
          robot.movel(torchPoses[1], acc =0.1, vel=0.1, wait=True)
          time.sleep(0.5)
          # robot.set_digital_out(0, True)
          # time.sleep(0.5)
          # robot.movels(ur_poses[1:-2], acc=0.1, vel=0.1, wait=True)
          robot.execute_ls(torchPoses[1:-2], output=0, acc=0.1, vel=0.1, wait=True)
          # time.sleep(0.5)
          # robot.set_digital_out(0, False)
          robot.movel(torchPoses[-1], acc=0.1, vel=0.1, wait=True)

          robot.movej(home1j, 0.4, 0.4, wait=True)
    
    reply = input('Do you want to do it again? :')
    if (reply == 'n'):
      break
    #first_round = False

  print("\n ************* End ************* ")

  robot.stop()
  # close the communication, otherwise python will not shutdown properly
  robot.close()
  print('UR5 closed')
  rospy.signal_shutdown("Finished shutting down")

