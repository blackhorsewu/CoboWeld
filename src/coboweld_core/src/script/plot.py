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

density = []

for index in range(count):

  # find all the neighbours of the query point
  [k, idx, _] = pc_kdtree.search_knn_vector_3d(pointcloud[index], neighbour)

  d = 0

  # Query point
  q_pt = pointcloud[index]

  for cnt in idx:
    #print('pointcloud[cnt]: ', pointcloud[cnt])
    dist = np.linalg.norm(q_pt - pointcloud[cnt])
    #print('distance: ', dist)
    if dist < 0.0007021: d += 1

  print('density: ', d)

  density.append(d)

max_cloud = np.max(pointcloud[:, 2])
min_cloud = np.min(pointcloud[:, 2])
cloud_range = max_cloud - min_cloud

max = np.max(density)
min = np.min(density)
range = max - min
density = ((density - min) / range) * cloud_range + 0.27

zd = density
fig = plt.figure(figsize=(10,10))
ax = plt.axes(projection='3d')
ax.grid()
ax.set_title('The "patch1" point cloud and its point density')
# c='r' ; colour is RED, s=10 ; size is 10
ax.scatter(x, y, z, c='g', s=1)
ax.scatter(x, y, density, c='r', s=5)
ax.set_xlabel('x', labelpad=20)
ax.set_ylabel('y', labelpad=20)
ax.set_zlabel('z', labelpad=20)
ax.set_zlabel('point density', labelpad=20)

plt.show()

