import open3d as o3d
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

#pc = o3d.io.read_point_cloud('50x50.pcd')
#pc = o3d.io.read_point_cloud('parallel.pcd')
filename = input('file name: ')
# filename = '2mm.pcd'
pcd = o3d.io.read_point_cloud(filename)
pointcloud = np.asarray(pcd.points)

x = pointcloud[:, 0]
y = pointcloud[:, 1]
z = pointcloud[:, 2]

# Number of points in this "patch"
count = np.asarray(pcd.points).shape[0]
print("Count: ", count)

pcd.estimate_normals(
  search_param = o3d.geometry.KDTreeSearchParamHybrid(
      radius = 0.005, max_nn = 100
      #radius = 0.027, max_nn = 270
  )
)

pcd.normalize_normals()
pcd.orient_normals_towards_camera_location(camera_location = [0.0, 0.0, 0.0])

# Build the KD Tree using the Fast Library for Approximate Nearest Neighbour
pc_kdtree = o3d.geometry.KDTreeFlann(pcd)

neighbour = 270 # min(count//100, 30)

n_list = np.asarray(pcd.normals)

density = []
shift_list = []

for index in range(count):

  # find all the neighbours of the query point
  [k, idx, _] = pc_kdtree.search_knn_vector_3d(pointcloud[index], neighbour)

  idx = idx[1:]

  vector = np.mean(n_list[idx, :], axis=0)
  
  d = 0

  # Query point
  q_pt = pointcloud[index]

  # the query point is always the first point in idx and should not be used
  # therefore must start from 1
  idx = idx[1:]

  '''
  # find the normal Centroid
  centroid = np.mean(pointcloud[idx], axis=0)

  # find the point density and the mean shift
  for cnt in idx:
    # number of points less than
    dist = np.linalg.norm(q_pt - pointcloud[cnt])
    #print('distance: ', dist)
    if dist < 0.002: d += 1
  '''

  # shift = np.linalg.norm(centroid - pointcloud[index])
  shift = np.linalg.norm(
    vector - n_list[index, :] * np.dot(vector,n_list[index, :]) / np.linalg.norm(n_list[index, :])
  )

  # print('shift: ', shift)
  # print('density: ', d)

  # shift = shift * d
  shift_list.append(shift)
  density.append(d)

max_cloud = np.max(pointcloud[:, 2])
min_cloud = np.min(pointcloud[:, 2])
cloud_range = max_cloud - min_cloud

max_density = np.max(density)
min_density = np.min(density)
density_range = max_density - min_density

#shift_list = shift_list * density
density = ((density - min_density) / density_range) * cloud_range + min_cloud

max_shift = np.max(shift_list)
min_shift = np.min(shift_list)
shift_range = max_shift - min_shift

shift_list = ((shift_list - min_shift) / shift_range) * cloud_range + min_cloud

fig = plt.figure(figsize=(10,10))
ax = plt.axes(projection='3d')
ax.grid()
ax.set_title('The point cloud and its feature values')
# c='r' ; colour is RED, s=10 ; size is 10
ax.scatter(x, y, z, c='g', s=1)
#ax.scatter(x, y, density, c='r', s=5)
ax.scatter(x, y, shift_list, c='b', s=10)
ax.set_xlabel('x', labelpad=20)
ax.set_ylabel('y', labelpad=20)
ax.set_zlabel('z', labelpad=20)
ax.set_zlabel('feature values', labelpad=20)

plt.show()

