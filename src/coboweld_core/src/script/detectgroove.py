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
import copy
import math

from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
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

#
# Define Parameter values
#
# 1. Feature value neighbours
feature_neighbours = 6
# 2. Distance between cluster neighbours
cluster_neighbour_distance = 0.005 # m or 10mm
# 3. Minimum cluster members
min_cluster_memb = 6
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

def transform_cam_wrt_base(pcd):

  # Added by Victor Wu on 25 July 2022 for Realsense D435i on UR5
  # Updated on 29 July 2022. Needs calibration later.
  # T_cam_wrt_end_effector = np.array( [[ 1.0000000,  0.0000000,  0.0000000, -0.01270],
  #                                     [ 0.0000000,  1.0000000,  0.0000000, -0.04000],
  #                                     [ 0.0000000,  0.0000000,  1.0000000,  0.18265],
  #                                     [ 0.0000000,  0.0000000,  0.0000000,  1.00000]] )
  # Z in translation added 0.015m because base is 0.015m above table
  T_cam_wrt_end_effector = np.array( [[ 1.0000000,  0.0000000,  0.0000000, -0.01750],
                                      [ 0.0000000,  1.0000000,  0.0000000, -0.03800],
                                      [ 0.0000000,  0.0000000,  1.0000000,  0.18400],
                                      [ 0.0000000,  0.0000000,  0.0000000,  1.00000]] )

  pcd_copy1 = copy.deepcopy(pcd).transform(T_cam_wrt_end_effector)
  # pcd_copy1.paint_uniform_color([0.5, 0.5, 1]) 
  # Do not change the colour, commented out by Victor Wu on 26 July 2022.

  pcd_copy2 = copy.deepcopy(pcd_copy1).transform(tcp_pose.array)
  # pcd_copy2.paint_uniform_color([1, 0, 0])
  # Do not change the colour, commented out by Victor Wu on 26 July 2022.
  # o3d.visualization.draw_geometries([pcd, pcd_copy1, pcd_copy1, pcd_copy2])
  return pcd_copy2

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

    '''
    # use the horizontal line as the Y-axis
    # *** remember, the groove, as it is now, is still in the camera frame
    # therefore the horizontal line is the camera's X-axis
    y_axis = np.array([-1.0, 0.0, 0.0]) # the negative X-axis of the camera
    y_axis = y_axis/np.linalg.norm(y_axis, axis=0) # normalize it
    # The diff_x cross the Y-axis (the horizontal line) gives the Z-axis
    # pointing into the workpiece (the tube)
    z_axis = np.cross(diff_x, y_axis)
    z_axis = z_axis/np.linalg.norm(z_axis, axis=0) # normalize it
    # The Y-axis cross the Z-axis gives the X-axis
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis/np.linalg.norm(x_axis, axis=0)

    # diff_x cross a vertical line gives the Y-axis
    y_axis = np.cross(diff_x, np.array([0.0, 0.0, 1.0]))
    y_axis = y_axis/np.linalg.norm(y_axis, axis=0) # normalize it
    # The diff_x cross the Y-axis gives the Z-axis
    z_axis = np.cross(diff_x, y_axis)
    z_axis = z_axis/np.linalg.norm(z_axis, axis=0) # normalize it
    # The Y-axis cross the Z-axis gives the X-axis
    x_axis = np.cross(y_axis, z_axis)
    # since both Y-axis and Z-axis are normalized therefore no need to normalize it
    '''

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

    # Use the scipy.spatial.transform library Rotation to find the Rotation Vector
    # from the X, Y, Z axis
    r = R.from_matrix(np.vstack((x_axis, y_axis, z_axis)).T)
    rotvec = r.as_rotvec()

    # if this is the first point use it to work out the approach point 
    if i == 0:
      # Needs the Quaternion to publish its pose
      orientation = r.as_quat()
      # Needs the Rotation Vector to send to URx (UR5)
      app_rotvec = r.as_rotvec()
      # The approach point is set to 50mm from the first point along the Z axis
      init_pos = z_axis * 0.05
    rotvecs.append(rotvec)
  # End for loop

  # Construct the Approach point
  approach = path[0] - init_pos
  approach = np.hstack((approach, app_rotvec))

  ur_poses = np.vstack((approach, np.hstack((path, np.array(rotvecs)))))

  return ur_poses

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
      min_bound = (-0.062, -0.10, 0.25), 
      max_bound = (0.028, 0.10, 0.35)  
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
     min_bound = ( -0.06, -0.094, 0.255), 
     max_bound = ( 0.024, 0.094, 0.345)  
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
  groove = transform_cam_wrt_base(groove)
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

# Main function.
if __name__ == "__main__":
  # Initialize the node and name it.
  rospy.init_node('coboweld_core', anonymous=True)

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
  pub_poses = rospy.Publisher('poses', PoseArray, queue_size=1)
  # pub_my_pose = rospy.Publisher("my_pose", PoseArray, queue_size=1)


  print("\n ************* Start *************")

  # Start URx
  
  # Do not start URx when testing software
  robot = urx.Robot('192.168.0.103')

  home1j = [0.0001, -1.1454, -2.7596, 0.7290, 0.0000, 0.0000]
  startG1j = [0.2173, -1.8616, -0.2579, -2.6004, 1.5741, 0.2147]

  robot.movej(home1j, 0.4, 0.4, wait=True)
  time.sleep(0.2)

  robot.movej(startG1j, 0.4, 0.4, wait=True)
  time.sleep(0.2)

  first_round = True
  while not rospy.is_shutdown():

    if not received_ros_cloud is None:
      received_open3d_cloud = orh.rospc_to_o3dpc(received_ros_cloud)

      rviz_cloud = orh.o3dpc_to_rospc(received_open3d_cloud, 
                                      frame_id="d435_depth_optical_frame")
      pub_captured.publish(rviz_cloud)

      tcp_pose = robot.get_pose()
      ur_poses = detect_groove_workflow(received_open3d_cloud, first_round)

      reply = input('Do you want to move to the Approaching Point? Y for yes: ')
      if (reply == "y"):
        torch_tcp = [0.0, -0.111, 0.366, 0.0, 0.0, 0.0]
        robot.set_tcp(torch_tcp)
        time.sleep(0.2)

        robot.movel(ur_poses[0], acc=0.1, vel=0.1, wait=True)

        input('\nPress any to continue')
        robot.movel(ur_poses[1], acc=0.1, vel=0.1, wait=True)

      first_round = False

  print("\n ************* End ************* ")
  robot.stop()
  # close the communication, otherwise python will not shutdown properly
  robot.close()
  print('UR5 closed')
  rospy.signal_shutdown("Finished shutting down")

