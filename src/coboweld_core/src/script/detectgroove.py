#!/usr/bin/env python
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
*  Revision 5: 16 March, 2323.
*     Parameters adjusted for the Y-tube joint.
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
*  urx
*
'''
# Imports for ROS
import roslib
import rospy
import sys

import numpy as np
import open3d as o3d
import math

from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2

import vg # Vector Geometry
from scipy import interpolate
import scipy.spatial as spatial

import urx

import csv

#
# Define Parameter values
#
# 1. Feature value neighbours
feature_neighbours = 7
# 2. Distance between cluster neighbours
cluster_neighbour_distance = 0.01 # m or 10mm
# 3. Minimum cluster members
min_cluster_memb = 7
# 4. Point cloud thickness in thin_line
thickness = 0.0143
# 5. Voxel size
voxelsize = 0.001 # m or 1mm
# 6. Normal estimation neighbourhood
# radius
my_radius = 0.012 # m or 12mm
# maximum nearest neighbours
maxnn = 452
# 7. Delete percentage of feature values
percentage = 0.96

# This is for conversion from Open3d point cloud to ROS point cloud
# Note: Add `.ravel()` to the end of line 261 in the `open3d_ros_helper.py` before it can work
# Refer to README.md 
from open3d_ros_helper import open3d_ros_helper as orh

# Call back function to receive a ROS point cloud published by the RealSense D435 camera
def callback_roscloud(ros_cloud):
    global received_ros_cloud

    received_ros_cloud = ros_cloud

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
          vector - (n_list[index, :] * np.dot(vector,n_list[index, :])))
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
      #robot.stop()
      # close the communication, otherwise python will not shutdown properly
      #robot.close()
      rospy.signal_shutdown("Finished shutting down")

    u_fine = np.linspace(0, 1, x.size*2)

    # Evaluate points on B-spline
    try:
      x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
    except TypeError:
      print("\n ************* End ************* ")
      #robot.stop()
      # close the communication, otherwise python will not shutdown properly
      #robot.close()
      rospy.signal_shutdown("Finished shutting down")

    sorted_points = np.vstack((x_fine, y_fine, z_fine)).T

    path_pcd = o3d.geometry.PointCloud()
    path_pcd.points = o3d.utility.Vector3dVector(sorted_points)

    return path_pcd

def detect_groove_workflow(pcd, first_round):

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
      min_bound = (-0.100, -0.07, 0.25), 
      max_bound = (0.100, 0.03, 0.35)  
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
  if first_round == True:
    print("Do you want to save the new point cloud?")
    reply = input("Y for yes: ")
    if (reply == "Y") or (reply == "y"):
      filename = input("Please input filename: ")
      o3d.io.write_point_cloud(filename, pcd)
    # else do nothing
  # else do nothing

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
     min_bound = (-0.095, -0.066, 0.255), 
     max_bound = (0.095, 0.026, 0.345)  
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

    # print("\n ************* Groove ************* ")
  groove = groove.paint_uniform_color([1, 0, 0])
  reply = input("Going to cluster selected points.\nc to continue others to quit.")
  if (reply == "c"):
    rviz_cloud = orh.o3dpc_to_rospc(groove, frame_id="d435_depth_optical_frame")
    pub_clustered.publish(rviz_cloud)

    # 5. Generate a path from the clustered Groove

    reply = input("Press 'c' to show path, any other to quit.")
    if (reply == "c"):
      generated_path = generate_path(groove)
      generated_path = generated_path.paint_uniform_color([0, 0, 1])

      rviz_cloud = orh.o3dpc_to_rospc(generated_path, frame_id="d435_depth_optical_frame")
      pub_path.publish(rviz_cloud)
    else:
      rospy.signal_shutdown("Finished shutting down")
      return
  else:
    rospy.signal_shutdown("Finished shutting down")
    return

# Main function.
if __name__ == "__main__":
  # Initialize the node and name it.
  rospy.init_node('coboweld_core', anonymous=True)

  # Start URx
  
  # Do not start URx when testing software
  # robot = urx.Robot("192.168.0.103")

  # Must have __init__(self) function for a class, similar to a C++ class constructor.
  global received_ros_cloud, delete_percentage

  # delete_percentage = 0.95 ORIGINAL VALUE
  delete_percentage = percentage

  received_ros_cloud = None

  # Setup subscriber
  rospy.Subscriber('/d435/depth/color/points', PointCloud2, 
                    callback_roscloud, queue_size=1
                  )

  # Setup publishers
  pub_captured = rospy.Publisher("captured", PointCloud2, queue_size=1)
  pub_selected = rospy.Publisher("selected", PointCloud2, queue_size=1)
  pub_clustered = rospy.Publisher("clustered", PointCloud2, queue_size=1)
  pub_neighbours = rospy.Publisher("neighbours", PointCloud2, queue_size=1)
  pub_path = rospy.Publisher("path", PointCloud2, queue_size=1)

  print("\n ************* Start *************")

  first_round = True
  while not rospy.is_shutdown():

    if not received_ros_cloud is None:
      received_open3d_cloud = orh.rospc_to_o3dpc(received_ros_cloud)

      rviz_cloud = orh.o3dpc_to_rospc(received_open3d_cloud, 
                                      frame_id="d435_depth_optical_frame")
      pub_captured.publish(rviz_cloud)

      detect_groove_workflow(received_open3d_cloud, first_round)

      first_round = False

  print("\n ************* End ************* ")
  #robot.stop()
  # close the communication, otherwise python will not shutdown properly
  #robot.close()
  rospy.signal_shutdown("Finished shutting down")

