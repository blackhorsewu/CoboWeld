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

# Aruco Marker pose estimation
def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):

  aruco_pose = PoseStamped()
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
          0.0447, # the marker is 45mm square
          matrix_coefficients,
          distortion_coefficients
        )
      tvec = tvec[0,0]
      # print('tvec: ', tvec.shape)
      rvec = rvec[0,0]
      # rvec = rvec * [-pi, 0.0, 0.0]
      # print('rvec: ', rvec.shape)
      
      # The marker is pointing towards the camera
      # It is necessary to use it to point away from the camera
      # This can be done by rotating about the X-axis
      r = R.from_rotvec(rvec) # get the r from rotation vector
      ar_mat = r.as_matrix()  # get the rotation matrix from r
      # multiply it with this matrix to rotate it about X-axis
      ar_mat = ar_mat * [[1, 0, 0],[0, -1, 0],[0, 0, -1]]

      ar_r = R.from_matrix(ar_mat)  # get a new r from the new matrix
      ar_quat = ar_r.as_quat()      # get the quaternion for publishing

      # Construct a pose to be published in RViz
      # A pose, to be published in RViz, consists of:
      #  (a) position x, y, z, 
      #  (b) orientation in quaternion x, y, z, w
      #
      # position
      aruco_pose.pose.position.x = tvec[0]
      aruco_pose.pose.position.y = tvec[1]
      aruco_pose.pose.position.z = tvec[2]
      # orientation
      aruco_pose.pose.orientation.x = ar_quat[0]
      aruco_pose.pose.orientation.y = ar_quat[1]
      aruco_pose.pose.orientation.z = ar_quat[2]
      aruco_pose.pose.orientation.w = ar_quat[3]

      aruco_pose.header.frame_id = 'd435_color_optical_frame'
      aruco_pose.header.stamp = rospy.Time.now()
      # pub_pose.publish(aruco_pose)

  return aruco_pose

def getMarkerPose():
  # get the colour image from camera
  image = rospy.wait_for_message('/d435/color/image_raw', Image, timeout=10)
  global cv_image
  p_listener = tf.TransformListener()
  try:
    cv_image = bridge.imgmsg_to_cv2(image, "bgr8")
    # Estimate the ArUCo marker pose
    pose = pose_estimation(
                cv_image,
                ARUCO_DICT[aruco_type], 
                intrinsic_camera, 
                distortion
              )
    try:
      now = rospy.Time.now()
      # Wait for transform to world
      p_listener.waitForTransform("world", "/d435_color_optical_frame", now, rospy.Duration(4.0))
      # Do the actual transform
      pose = p_listener.transformPose("world", pose)
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
      print("tf error!")
  except ChildProcessError as e:
    print(e)
  return pose

def setDepthCameraPose(marker_pose):
  camPose = PoseStamped()
  # Extract the Z-axis from the orientation
  quat = marker_pose.pose.orientation
  quat = [quat.x, quat.y, quat.z, quat.w]
  r = R.from_quat(quat)
  matrix = r.as_matrix()
  # Z-axis is the the 3rd column of the rotation matrix
  z_axis = matrix[:, 2]

  # Same frame_id and same orientation
  camPose.header.frame_id = marker_pose.header.frame_id
  camPose.pose.orientation = marker_pose.pose.orientation

  # Find the original position first
  posit = marker_pose.pose.position
  posit = [posit.x, posit.y, posit.z]
  # The position is then shifted towards the camera by 0.3m (300mm)
  camPosit = posit - (z_axis * 0.3)
  camPose.pose.position.x = camPosit[0]
  camPose.pose.position.y = camPosit[1]
  camPose.pose.position.z = camPosit[2]

  camPose.header.stamp = rospy.Time.now()

  return camPose

# The in_pose is a ROS PoseStamped
# The out_pose is a URx pose or UR pose, ie first 3 are position,
# last 3 are Rotation Vector.
def ros_to_urx(in_pose):
  quat = in_pose.pose.orientation
  quat = [quat.x, quat.y, quat.z, quat.w]
  r = R.from_quat(quat)
  rvec = r.as_rotvec()
  pos = in_pose.pose.position
  out_pose = [pos.x, pos.y, pos.z, rvec[0], rvec[1], rvec[2]]
  return out_pose

def transform_cam_wrt_base(pcd):

  # Updated on 30 March 2023 by Victor Wu.
  # Cannot use tf.transformPointCloud because pcd is not a ROS pointcloud2!
  try:
    now = rospy.Time.now()
    listener.waitForTransform("/base", "/d435_depth_optical_frame", now, rospy.Duration(4.0))
    (trans, rot) = listener.lookupTransform("/base", "/d435_depth_optical_frame", now)
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
      init_pos = z_axis * 0.05

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

  aruco_type = "DICT_5X5_100"

  arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[aruco_type])
  arucoParams = cv2.aruco.DetectorParameters_create()

  print("\n ************* Start *************")

  # Start URx only when execute is True
  
  # Do not start URx when testing software
  
  robot = urx.Robot('192.168.0.103')

  # home1 is when it faces to the right
  home1j = [0.0001, -1.1454, -2.7596, 0.7290, 0.0000, 0.0000]
  # home2 is when it faces the CHS
  home2j = [0.6496, -1.1454, -2.7596, 0.7289, 0.0000, 0.0000]
  startG1j = [0.2173, -1.8616, -0.2579, -2.6004, 1.5741, 0.2147]
  startchsj = [0.6792, -0.4243, -2.5662, -0.1751, 0.9010, 0.0188]
  # startchs1 is the target for point cloud capturing
  startchs1j = [-0.4060, -1.4229, -2.2255, 0.5201, 1.0502, -0.0131]

  listener.waitForTransform("/base", "/d435_depth_optical_frame", rospy.Time(), rospy.Duration(4.0))

  # Acquire colour camera information include intrinsic matrix and distortion
  # before pose of ArUCo marker can be found
  getColourCamInfo()
  # print('intrinsic_camera: ', intrinsic_camera)
  # print('distortion: ', distortion)

  # first_round = True
  while not rospy.is_shutdown():

    # move UR5 to starting point
    robot.movej(home2j, 0.4, 0.4, wait=True)
    time.sleep(0.2)

    robot.set_tcp((0, 0, 0, 0, 0, 0))
    time.sleep(0.3)

    # find the ArUCo marker
    marker_pose = getMarkerPose()
    # print('marker_pose', marker_pose)
    pub_pose.publish(marker_pose) # Visualize marker pose in RViz

    # robot.movej(startchs1j, 0.4, 0.4, wait=True)

    # move the depth camera to somewhere near the ArUCo marker.
    # Since the Marker is 45mm square, the centre is about 22.5mm away
    # from the groove edge. Assuming the groove is about 15mm wide, so
    # the d435_depth_optical_frame should be about 30mm below the marker.
    # It should also be 300mm away from the groove, along the Z-axis.

    # Use the marker_pose to set the Depth Camera Pose
    camera_pose = setDepthCameraPose(marker_pose)
    pub_campose.publish(camera_pose)
    # convert ROS pose to URx pose
    camera_pose = ros_to_urx(camera_pose)

    print(camera_pose)

    # The relative position of the depth_optical_frame from wrist_3_link is
    # [-0.0175, -0.035, 0.20245]
    camera_tcp = [-0.0175, -0.035, 0.20245, 0.0, 0.0, 0.0]
    robot.set_tcp(camera_tcp)
    time.sleep(0.2)
    robot.movel(camera_pose, 0.1, 0.1, wait=True)
    '''
    if not received_ros_cloud is None:
      received_open3d_cloud = orh.rospc_to_o3dpc(received_ros_cloud)

      rviz_cloud = orh.o3dpc_to_rospc(received_open3d_cloud, 
                                      frame_id="d435_depth_optical_frame")
      pub_captured.publish(rviz_cloud)

      # tcp_pose = robot.get_pose()
      ur_poses = detect_groove_workflow(received_open3d_cloud)

      if execute:
        reply = input('Do you want to move to the Approaching Point? Y for yes: ')
        if (reply == "y"):
          # torch_tcp = [0.0, -0.105, 0.365, 0.0, 0.0, 0.0]
          torch_tcp = [0.0, -0.095, 0.385, 0.0, 0.0, 0.0]
          robot.set_tcp(torch_tcp)
          # pause is essential for tcp to take effect, min time is 0.1s
          time.sleep(0.2)

          # Move to the Approach point
          robot.movel(ur_poses[0], acc=0.1, vel=0.1, wait=True)

          input('\nPress any to continue')
          robot.movel(ur_poses[1], acc =0.1, vel=0.1, wait=True)
          time.sleep(0.5)
          # robot.set_digital_out(0, True)
          # time.sleep(0.5)
          # robot.movels(ur_poses[1:-2], acc=0.1, vel=0.1, wait=True)
          robot.execute_ls(ur_poses[1:-2], output=0, acc=0.1, vel=0.1, wait=True)
          # time.sleep(0.5)
          # robot.set_digital_out(0, False)
          robot.movel(ur_poses[-1], acc=0.1, vel=0.1, wait=True)

          robot.movej(home1j, 0.4, 0.4, wait=True)
    '''
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

