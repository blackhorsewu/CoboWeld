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


# This is for conversion from Open3d point cloud to ROS point cloud
from open3d_ros_helper import open3d_ros_helper as orh

def callback_roscloud(ros_cloud):
    global received_ros_cloud

    received_ros_cloud = ros_cloud

def normalize_feature(feature_value_list):
    
    normalized_feature_value_list = (feature_value_list-feature_value_list.min())/(feature_value_list.max()-feature_value_list.min())

    return np.array(normalized_feature_value_list)

'''
# find feature value of each point of the point cloud and put them into a list
def find_feature_value(pcd, voxel_size):

    # Build a KD (k-dimensional) Tree for Flann
    # Fast Library for Approximate Nearest Neighbor
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    # Treat pcd.points as an numpy array of n points by m attributes of a point
    # The first dimension, shape[0], of this array is the number of points in this point cloud.
    pc_number = np.asarray(pcd.points).shape[0]

    feature_value_list = []
    
    # This is very important. It specifies the attribute that we are using to find the feature
    # when it is pcd.normals, it is using the normals to find the feature,
    # when it is pcd.colors, it is using the colours to find the feature.
    # n_list is an array of normals of all the points
    n_list = np.asarray(pcd.normals)
    # n_list = np.asarray(pcd.colors)

    # a // b = integer quotient of a divided by b
    # so neighbor (number of neighbors) whichever is smaller of 30 or the quotient 
    # of dividing the number of points by 100
    # neighbour = min(pc_number//80, 65)
    # neighbour = min(pc_number//50, 100)
    neighbour = min(pc_number//100, 30)
    # neighbour = 30
    
    # for every point of the point cloud
    for index in range(pc_number):
        
        # Search the k nearest neighbour for each point of the point cloud
        # The pcd.points[index] is the point to find the nearest neighbour for
        # neighbour, found above, is the neighbours to be searched
        [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[index], neighbour)

        # n_list, computed above, is an array of normals of every point
        # vector is then a vector with its components the arithmetic mean of every
        # element of all the k neighbours of that point
        # This can be called the CENTROID of the NORMALs of its neighbours
        vector = np.mean(n_list[idx, :], axis=0)
        
        # the bigger the feature value, meaning the normal of that point is more 
        # different from its neighbours
        feature_value = np.linalg.norm(
            vector - n_list[index, :] * np.dot(vector,n_list[index, :]) / np.linalg.norm(n_list[index, :]))
        feature_value_list.append(feature_value)

    return np.array(feature_value_list)
'''

def find_feature_value(pcd):
  # Build a KD (k-dimensional) Tree using Flann
  # that is Fast Library for Approximate Nearest Neighbour
  pcd_tree = o3d.geometry.KDTreeFlann(pcd)

  pointcloud = np.asarray(pcd.points)
  pointcolour = np.asarray(pcd.colors)

  # Convert the pcd.points to a numpy array
  # The first dimension of this array, shape[0] is the number of points in this cloud
  pc_number = np.asarray(pcd.points).shape[0]

  # Declare the feature value list
  feature_value_list = []

  # define neighbour
  #neighbour = min(pc_number//100, 30)

  neighbour = int(input('Please enter the number of neighbours you want: '))

  # show the neighbours one by one
  reply = input('Do you want to show neighbourhoods one by one? (Y/n): ')
  if (reply == 'Y') or (reply == 'y'):
    for index in range(pc_number):
      [k, idx, _] = pcd_tree.search_knn_vector_3d(pointcloud[index], neighbour)
      for coldex in idx:
        pointcolour[coldex] = [0.0, 1.0, 0.0] # set the neighbourhood to Green
      pointcolour[index] = [1.0, 0.0, 0.0]    # set the query point to Red
      rviz_cloud = orh.o3dpc_to_rospc(pcd, frame_id="d435_depth_optical_frame")
      pub_neighbours.publish(rviz_cloud)

      # Find the Geometric Centroid of the neighbourhood
      centroid = np.mean(pointcloud[idx, :], axis = 0)

      pointcolour[coldex] = [0.0, 0.0, 0.0] # reset the neighbourhood to black after displayed
      print("Point number: ", index)
      if input("Hit ENTER for next neighbourhood: ") == "q" : break
  else:
    reply = input('Input the point number for the neighbourhood: ')
    while (reply.isdigit()):
      index = int(reply)
      print('Coordinates of this point: ', pointcloud[index, 0], pointcloud[index, 1], pointcloud[index, 2])
      [k, idx, _] = pcd_tree.search_knn_vector_3d(pointcloud[index], neighbour)
      for coldex in idx:
        pointcolour[coldex] = [0.0, 1.0, 0.0] # set the neighbourhood to Green
      filename = input('Input the file name for coordinates of the neighbourhood: ')
      with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["X", "Y", "Z"])
        for posdex in idx:
          writer.writerow([pointcloud[posdex, 0], pointcloud[posdex, 1], pointcloud[posdex, 2]])

      pointcolour[index] = [1.0, 0.0, 0.0]    # set the query point to Red
      rviz_cloud = orh.o3dpc_to_rospc(pcd, frame_id="d435_depth_optical_frame")
      pub_neighbours.publish(rviz_cloud)
      reply = input('Input the point number for the next neighbourhood: ')

    rospy.signal_shutdown("Finished shutting down")
    return


'''
  # for every point of the point cloud
  for index in range(pc_number):

    # Search the k nearest neighbours for each point
    # [k, idx, _] = pcd_tree.search_knn_vector_3d(pointcloud[index], neighbour)

    # Find the resolution
    resolution = (pointcloud[index, :] - pointcloud[idx, :]).min()

    # Find the Geometric Centroid of the neighbourhood
    centroid = np.mean(pointcloud[idx, :], axis = 0)

    # Find the Mean Shift as the feature value
    #feature_value = np.linalg.norm(centroid - pointcloud[idx])/resolution

#    if ((feature_value < 0.3) or (feature_value > 0.45)): feature_value = 0.0

    # Put the feature value into the feature value list
    #feature_value_list.append(feature_value)

  feature_value_list = np.array(feature_value_list)
  min_value = feature_value_list.min()
  max_value = feature_value_list.max()
  denom = max_value - min_value
  # feature_value_list[:] = (feature_value_list[:] - min_value)/denom

  for index in range(pc_number):
    if (feature_value_list[index] < 0.3) or (feature_value_list[index] > 0.45): 
      feature_value_list[index] = 0

  return feature_value_list
'''

def cluster_groove_from_point_cloud(pcd, voxel_size, verbose=False):

    global neighbor

    # eps - the maximum distance between neighbours in a cluster, originally 0.005,
    # at least min_points to form a cluster.
    # returns an array of labels, each label is a number. 
    # labels of the same cluster have their labels the same number.
    # if this number is -1, that is this cluster is noise.
    # labels = np.array(pcd.cluster_dbscan(eps=0.005, min_points=3, print_progress=True))
    labels = np.array(pcd.cluster_dbscan(eps=0.005, min_points=3, print_progress=False))

    # np.unique returns unique labels, label_counts is an array of number of that label
    label, label_counts = np.unique(labels, return_counts=True)

    # Find the largest cluster
    ## [-1] is the last element of the array, minus means counting backward.
    ## So, after sorting ascending the labels with the cluster with largest number of
    ## members at the end. That is the largest cluster.
    label_number1 = label[np.argsort(label_counts)[-1]]
#    label_number2 = label[np.argsort(label_counts)[-2]]
#    label_number3 = label[np.argsort(label_counts)[-3]]

    if label_number1 == -1:
        if label.shape[0]>1:
            label_number = label[np.argsort(label_counts)[-2]]
        elif label.shape[0]==1:
            # sys.exit("can not find a valid groove cluster")
            print("can not find a valid groove cluster")
    
    # Pick the points belong to the largest cluster
    groove_index = np.where(labels == label_number1)
#    groove = pcd.select_down_sample(groove_index[0])
    groove1 = pcd.select_by_index(groove_index[0])
    groove1.paint_uniform_color([1, 0, 0])
#    groove_index = np.where(labels == label_number2)
#    groove2 = pcd.select_by_index(groove_index[0])
#    groove2.paint_uniform_color([0, 1, 0])
#    groove_index = np.where(labels == label_number3)
#    groove3 = pcd.select_by_index(groove_index[0])
#    groove3.paint_uniform_color([0, 0, 1])

    return groove1 #+groove2   #+groove3

def thin_line(points, point_cloud_thickness=0.01, iterations=1, sample_points=0):
                    # point_cloud_thickness=0.015
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

# To generate a welding path for the torch. This is only a path and should be called a trajectory!
def generate_path(groove):

    points = np.asarray(groove.points)

    # Thin & sort points
    thinned_points, regression_lines = thin_line(points)
    sorted_points = sort_points(thinned_points, regression_lines)

    x = sorted_points[:, 0]
    y = sorted_points[:, 1]
    z = sorted_points[:, 2]

    (tck, u), fp, ier, msg = interpolate.splprep([x, y, z], s=float("inf"), full_output=1)

    u_fine = np.linspace(0, 1, x.size*2)

    # Evaluate points on B-spline
    x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)

    sorted_points = np.vstack((x_fine, y_fine, z_fine)).T

    path_pcd = o3d.geometry.PointCloud()
    path_pcd.points = o3d.utility.Vector3dVector(sorted_points)

    return path_pcd

def detect_groove_workflow(pcd):

    original_pcd = pcd

    global delete_percentage

    # 1. Down sample the point cloud
    ## a. Define a bounding box for cropping
    bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound = (-0.025, -0.25, 0.2), # x right, y down, z forward; for the camera
        max_bound = (0.05, 0.1, 0.5)  # 50mm x 50mm plane with 0.5m depth
    )

    ## b. Define voxel size
    voxelsize = 0.001 # 1mm cube for each voxel

#    print("\n ************* Before cropping ************* ")
#    rviz_cloud = orh.o3dpc_to_rospc(pcd, frame_id="d435_depth_optical_frame")
#    pub_captured.publish(rviz_cloud)

    pcd = pcd.voxel_down_sample(voxel_size = voxelsize)
    pcd = pcd.crop(bbox)

    ### it was remove_none_finite_points in Open3D version 0.8.0... but
    ### it is  remove_non_finite_points  in Open3D version 0.15.1...
    pcd.remove_non_finite_points()
    reply = input("Point cloud cropped.\nc to continue,\ns to save, other to quit.")
    print("Do you want to use saved point cloud?")
    reply = input("N for no, Y for yes: ")
    if (reply == "N") or (reply == "n"):
      rviz_cloud = orh.o3dpc_to_rospc(pcd, frame_id="d435_depth_optical_frame")
      pub_captured.publish(rviz_cloud)
      print("Do you want to save the new point cloud?")
      reply = input("Y for yes: ")
      if (reply == "Y") or (reply == "y"):
         filename = input("Please input filename: ")
         o3d.io.write_point_cloud(filename, pcd)
    elif (reply == "Y") or (reply == 'y'):
      filename = input("Please enter file name to load old saved point cloud: ")
      pcd = o3d.io.read_point_cloud(filename)
    else:
      rospy.signal_shutdown("Finished shutting down")
      return

    ## c. Count the number of points afterwards
    pc_number = np.asarray(pcd.points).shape[0]
    rospy.loginfo("Total number of points {}".format(pc_number))

    # 2. Estimate normal toward camera location and normalize it.
    pcd.estimate_normals(
        search_param = o3d.geometry.KDTreeSearchParamHybrid(
            radius = 0.01, max_nn = 30
        )
    )

    pcd.normalize_normals()
    pcd.orient_normals_towards_camera_location(camera_location = [0.0, 0.0, 0.0])

    # 3. Use different geometry features to find groove
    #    Use asymmetric normals as a feature to find groove
    feature_value_list = find_feature_value(pcd)
    # normalized_feature_value_list = normalize_feature(feature_value_list)

    # 4. Delete low value points and cluster
#    delete_points = int(pc_number * delete_percentage)

#    pcd_selected = pcd.select_down_sample(
#    pcd_selected = pcd.select_by_index(
        ## np.argsort performs an indirect sort
        ## and returns an array of indices of the same shape
        ## that index data along the sorting axis
        ## in ascending order by default. So the smaller value first
        ## and the largest value at the end
#        np.argsort(feature_value_list)[delete_points:]
        ## therefore this is a list of indices of the point cloud
        ## with the top 5 percent feature value
#    )

'''
    reply = input("Featured points selected.\nc to continue others to quit.")
    if (reply == "c"):
      # pcd_selected.paint_uniform_color([0, 1, 0])
      rviz_cloud = orh.o3dpc_to_rospc(pcd_selected, frame_id="d435_depth_optical_frame")
      pub_selected.publish(rviz_cloud)

      groove = cluster_groove_from_point_cloud(pcd_selected, voxelsize)
    else:
      rospy.signal_shutdown("Finished shutting down")
      return

      # print("\n ************* Groove ************* ")
    # groove = groove.paint_uniform_color([0, 1, 0])
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
'''

# Main function.
if __name__ == "__main__":
  # Initialize the node and name it.
  rospy.init_node('coboweld_core', anonymous=True)

  # Start URx
  robot = urx.Robot("192.168.0.103")

  # Must have __init__(self) function for a class, similar to a C++ class constructor.
  global received_ros_cloud, delete_percentage

  # delete_percentage = 0.95 ORIGINAL VALUE
  delete_percentage = 0.98

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

  while not rospy.is_shutdown():

    if not received_ros_cloud is None:
      received_open3d_cloud = orh.rospc_to_o3dpc(received_ros_cloud)

      rviz_cloud = orh.o3dpc_to_rospc(received_open3d_cloud, 
                                      frame_id="d435_depth_optical_frame")
      pub_captured.publish(rviz_cloud)

      detect_groove_workflow(received_open3d_cloud)

  print("\n ************* End ************* ")
  robot.stop()
  # close the communication, otherwise python will not shutdown properly
  robot.close()
  rospy.signal_shutdown("Finished shutting down")

