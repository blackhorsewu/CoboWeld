import open3d as o3d
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

#pc = o3d.io.read_point_cloud('50x50.pcd')
pc = o3d.io.read_point_cloud('patch.pcd')
pointcloud = np.asarray(pc.points)
normalcloud = np.asarray(pc.normals)

x = pointcloud[:, 0]
y = pointcloud[:, 1]
z = pointcloud[:, 2]

# Number of points in this "patch"
count = np.asarray(pc.points).shape[0]
print("Count: ", count)

# Estimate the normals towards the camera
pc.estimate_normals(
  search_param = o3d.geometry.KDTreeSearchParamHybrid(
    radius = 0.01, max_nn = 50 # the radius of search is 0.01m or 1cm or 10mm
  )
)

# make a list of normals
n_list = np.asarray(pc.normals)

# Build the KD Tree using the Fast Library for Approximate Nearest Neighbour
pc_kdtree = o3d.geometry.KDTreeFlann(pc)

# make a 'Centroid' cloud of normal displacements
centroid_cloud = np.ndarray(shape=(count, 3))
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(centroid_cloud[:])

shift_cloud = np.ndarray(shape=(count, 3))

# try to find neighbours along the center line where x = 0.0
# all the points within 1cm of the center line
neighbour = 30

# Build the 'Centroid' cloud
for index in range(count):

  # find all the neighbours of the query point
  [k, idx, _] = pc_kdtree.search_knn_vector_3d(pointcloud[index], neighbour)

  # Find the resolution
  # resolution = (pointcloud[index, :] - pointcloud[idx, :]).min()
  # print(resolution)

  # Normals Centroid of the neighbourhood
  normal_centroid = np.mean(n_list[idx, :], axis=0)

  # Displacement of the Geometric Centroid from the query point
  shift_cloud[index, 0] = pointcloud[index, 0]
  shift_cloud[index, 1] = pointcloud[index, 1]
  shift_cloud[index, 2] = np.linalg.norm(normal_centroid - n_list[index, :])

pointnumber = int(input("Input the point number: "))
xf = pointcloud[pointnumber, 0]
yf = pointcloud[pointnumber, 1]
zf = pointcloud[pointnumber, 2]
print("Point number: ", pointnumber)
print("X: ", pointcloud[pointnumber, 0])
print("Y: ", pointcloud[pointnumber, 1])
print("Z: ", pointcloud[pointnumber, 2])

[k, idx, _] = pc_kdtree.search_knn_vector_3d(pointcloud[pointnumber], neighbour)
print("Neighbour point numbers: ", idx)

xn = pointcloud[idx, 0]
yn = pointcloud[idx, 1]
zn = pointcloud[idx, 2]

fig = plt.figure(figsize=(10,10))
ax = plt.axes(projection='3d')
ax.grid()
ax.set_title('The "patch" point cloud')
# c='r' ; colour is RED, s=10 ; size is 10
#ax.scatter(x, y, z, c='g', s=1)
ax.scatter(xf, yf, zf, c='r', s=15)
ax.scatter(xn, yn, zn, c='b', s=5)
ax.set_xlabel('x', labelpad=20)
ax.set_ylabel('y', labelpad=20)
ax.set_zlabel('z', labelpad=20)
# ax.set_zlim3d(-0.0001, 0.0001)
ax.set_zlabel('zf', labelpad=20)

plt.show()

