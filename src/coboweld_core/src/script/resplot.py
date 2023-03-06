'''
Plot resolution instead of point density.
'''
import open3d as o3d
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

#pc = o3d.io.read_point_cloud('50x50.pcd')
pc = o3d.io.read_point_cloud('patch.pcd')
pointcloud = np.asarray(pc.points)

x = pointcloud[:, 0]
y = pointcloud[:, 1]
z = pointcloud[:, 2]

xd = pointcloud[:, 0]
yd = pointcloud[:, 1]

# Number of points in this "patch"
count = np.asarray(pc.points).shape[0]
print("Count: ", count)

# Build the KD Tree using the Fast Library for Approximate Nearest Neighbour
pc_kdtree = o3d.geometry.KDTreeFlann(pc)

# try to find neighbours along the center line where x = 0.0
# all the points within 1cm of the center line
neighbour = 50

resolution = []
shift_list = []

for index in range(count):

  # find all the neighbours of the query point
  [k, idx, _] = pc_kdtree.search_knn_vector_3d(pointcloud[index], neighbour)

  d = 0

  # Query point
  q_pt = pointcloud[index]

  distance = []

  # find the Geometric Centroid
  centroid = np.mean(pointcloud[idx], axis=0)

  # first element of idx is the query point, therefore start from 1
  idx = idx[1:]
  
  # find the resolution and the mean shift
  for cnt in idx:
    dist = np.linalg.norm(q_pt - pointcloud[cnt])
    distance.append(dist)

  min_distance = np.min(distance)
  print("min_distance[index]: ", index, min_distance)

  resolution.append(min_distance)

  shift = np.linalg.norm(centroid - pointcloud[index]) / min_distance
  shift_list.append(shift)

max_cloud = np.max(pointcloud[:, 2])
min_cloud = np.min(pointcloud[:, 2])
cloud_range = max_cloud - min_cloud

max_resolut = np.max(resolution)
print('max resolution: ', max_resolut)
min_resolut = np.min(resolution)
print('min resolution: ', min_resolut)
resolution_range = max_resolut - min_resolut

resolution = ((resolution - min_resolut) / resolution_range) * cloud_range + 0.27

#zd = resolution
fig = plt.figure(figsize=(10,10))
ax = plt.axes(projection='3d')
ax.grid()
ax.set_title('The "patch1" point cloud and its point resolution')
# c='r' ; colour is RED, s=10 ; size is 10
ax.scatter(x, y, z, c='g', s=1)
ax.scatter(x, y, resolution, c='r', s=5)
ax.scatter(x, y, shift_list, c='b', s=10)
ax.set_xlabel('x', labelpad=20)
ax.set_ylabel('y', labelpad=20)
ax.set_zlabel('z', labelpad=20)
ax.set_zlabel('point resolution', labelpad=20)

plt.show()

