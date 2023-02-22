import open3d as o3d
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

pc = o3d.io.read_point_cloud('50x50.pcd')
pointcloud = np.asarray(pc.points)

x = pointcloud[:, 0]
y = pointcloud[:, 1]
z = pointcloud[:, 2]

# Number of points in this "patch"
count = np.asarray(pc.points).shape[0]

# Build the KD Tree using the Fast Library for Approximate Nearest Neighbour
pc_kdtree = o3d.geometry.KDTreeFlann(pc)

# Estimate normals
pc.estimate_normals(
  search_param=o3d.geometry.KDTreeSearchParamHybrid(
    radius=0.01, max_nn=30
  )
)
pc.normalize_normals()

# make a 'feature' cloud
feature_cloud = np.ndarray(shape=(count, 3))

# list of normals
n_list = np.asarray(pc.normals)

# try to find neighbours along the center line where x = 0.0
# all the points within 1cm of the center line
neighbour = 30

#########################################################
#for index in range(count):
  #feature_cloud[index, 2] = 0.0
#  if ((pointcloud[index, 0] >= -0.01) and (pointcloud[index, 0] <= 0.01)):
#  [k, idx, _] = pc_kdtree.search_knn_vector_3d(pc.points[index], neighbour)
#  centroid = np.mean(n_list[idx, :], axis=0)
  # Make the feature cloud
  # feature_cloud[index, 0] = pointcloud[index, 0].copy()
  # feature_cloud[index, 1] = pointcloud[index, 1].copy()
#  feature_cloud[index, 2] = np.linalg.norm(
#    centroid - n_list[index, :] * np.dot(centroid, n_list[index, :])/np.linalg.norm(n_list[index, :]))

# xf = feature_cloud[:, 0]
# yf = feature_cloud[:, 1]
#zf = feature_cloud[:, 2]
#############################################################

# Try to find the edge intensity according to Jeffery's paper
for index in range(count):
  feature_cloud[index, 2] = 0.0
  #if ((pointcloud[index, 0] >= -0.02) and (pointcloud[index, 0 ] <= 0.02)):
  # find all the neighbours of the query point
  [k, idx, _] = pc_kdtree.search_knn_vector_3d(pc.points[index], neighbour)
  # Normals Centroid of the neighbours 
  centroid = np.mean(n_list[idx, :], axis=0)
  # Edge Intensity using normals instead of geometric position 
  feature_cloud[index, 2] = np.linalg.norm(
    centroid - n_list[index, :] * np.dot(centroid, n_list[index, :])/np.linalg.norm(n_list[index, :]))

min_val = feature_cloud[:, 2].min()
max_val = feature_cloud[:, 2].max()

#print("max: ", max_val)
#print("\nmin: ", min_val)
den = max_val - min_val
feature_cloud[:, 2] = (feature_cloud[:, 2] - min_val)/den

for index in range(count):
  if (feature_cloud[index, 2] < 0.3): feature_cloud[index, 2] = 0

zf = feature_cloud[:, 2]*(0.3 - 0.275) + 0.25
#zf = feature_cloud[:, 2]

fig = plt.figure(figsize=(10,10))
ax = plt.axes(projection='3d')
ax.grid()
ax.set_title('The "patch" point cloud')
# c='r' ; colour is RED, s=10 ; size is 10
ax.scatter(x, y, z, s=10)
ax.scatter(x, y, zf, c='r', s=10)
ax.set_xlabel('x', labelpad=20)
ax.set_ylabel('y', labelpad=20)
ax.set_zlabel('z', labelpad=20)
# ax.set_zlim3d(-0.0001, 0.0001)
ax.set_zlabel('zf', labelpad=20)

plt.show()

