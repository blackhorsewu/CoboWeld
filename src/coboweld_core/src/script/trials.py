import open3d as o3d
import numpy as np
import random

# Get the cropped point cloud file name
fileName = input("File name of point cloud: ")

# Define the window size of the visualization in pixels
windwd = input("Width of window in pixels: ")
winht = input("Height of window in pixels: ")

# Read the Point Cloud from file
pc = o3d.io.read_point_cloud(fileName)

# Show the Point Cloud
def show(pcd):
  o3d.visualization.draw_geometries([pcd],
    window_name=fileName, width=int(windwd), height=int(winht),
    point_show_normal=True)

# Build the KD Tree using the Fast Approximate Nearest Neighbour algorithm
pc_kdtree = o3d.geometry.KDTreeFlann(pc)

# Find the number of points in the Point Cloud
pc_number = np.asarray(pc.points).shape[0]

# Define the number of nearest neighbours
# neighbour = min(pc_number//100, 50)

# First reset, or clear, the point cloud to no colour or all black [0,0,0]
def clear_pc_color(pcd):
  for index in range(pc_number):
    np.asarray(pcd.colors)[index, :] = [0, 0, 0]

# Generate n randomly placed groups of nearest neighbours
def random_neighbour(n):
  neighbour = input("Number of neighbours: ")
  for count in range(n):
    index = random.randint(0, pc_number)
    [k, idx, _] = pc_kdtree.search_knn_vector_3d(pc.points[index], int(neighbour))
    np.asarray(pc.colors)[idx, :] = [1, 0, 0] # set them into RED in colour

# Generate a group of nearest neighbours at the centre
def center_neighbour():
  neighbour = input("Number of neighbours: ")
  index = pc_number//2
  [k, idx, _] = pc_kdtree.search_knn_vector_3d(pc.points[index], int(neighbour))
  np.asarray(pc.colors)[idx, :] = [1, 0, 0]

