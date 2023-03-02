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

'''
# Number of points in this "patch"
count = np.asarray(pc.points).shape[0]
print("Count: ", count)

# Build the KD Tree using the Fast Library for Approximate Nearest Neighbour
pc_kdtree = o3d.geometry.KDTreeFlann(pc)

# try to find neighbours along the center line where x = 0.0
# all the points within 1cm of the center line
neighbour = 30

for index in range(count):

  # find all the neighbours of the query point
  [k, idx, _] = pc_kdtree.search_knn_vector_3d(pointcloud[index], neighbour)

  # Find the resolution
  # resolution = (pointcloud[index, :] - pointcloud[idx, :]).min()
  # print(resolution)

  # Geometric Centroid of the neighbourhood
  centroid = np.mean(np.asarray(pc.points)[idx[0:]], axis=0)

  # Displacement of the Geometric Centroid from the query point
  mean_shift = np.linalg.norm(centroid - pointcloud[index])

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
'''

edge_points = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 21,
               23, 24, 25, 26, 27, 28, 31, 32, 33, 37, 44, 49, 50, 51, 53,
               60, 61, 62, 63, 70, 71, 86, 94, 95, 96, 97, 99, 100, 101, 
               102, 103, 126, 128, 129, 135, 136, 137, 138, 139, 140, 143, 
               144, 145, 146, 169, 170, 172, 175, 176, 177, 178, 184, 185, 
               186, 187, 188, 211, 212, 215, 216, 217, 218, 230, 247, 248, 
               249, 251, 651, 1015, 1016, 1027, 1157, 1171, 1251, 1253, 1258, 
               1316, 1401, 1426, 1474, 1504, 1566, 1732, 1751, 1767, 1771, 
               1784, 1883, 1935, 1974, 2146, 2236, 2270, 2301, 2380, 2405, 
               2418, 2471, 2497, 2531, 2538, 2540, 2557, 2647, 2649, 2678, 
               2693, 2765, 2833, 2975, 3203, 3308, 3366, 3524, 3551, 3615, 
               3620, 3644, 3678, 3682, 3800, 3815, 3841, 3855, 3925, 4089, 
               4120, 4129, 4133, 4135, 4264, 4358, 4416, 4509, 4514, 4525, 
               4553, 4684, 4767, 4777, 4779, 4780, 4782, 4801, 4803, 4804, 
               4807, 4808, 4809, 4810, 4830, 4831, 4832, 4836, 4838, 4839, 
               4840, 4861, 4868, 4871, 4873, 4875, 4876, 4877, 4878, 4903, 
               4904, 4905, 4906, 4908, 4909, 4910, 4911, 4912, 4944, 4945, 
               4946, 4947, 4949, 4950, 4953, 4954, 4955, 4977, 4978, 4979, 
               4980, 5001, 5002, 5003, 5006, 5007, 5008, 5023, 5031, 5034, 
               5087, 5137, 5225, 5275, 5388, 5549, 5926, 6042, 6754, 6783, 
               7134, 7549, 7559, 7627, 7632, 8976, 9020, 9096, 9176, 9441, 
               9516, 9690, 9719, 9765, 9802, 10052, 10127, 10229, 10324, 
               10333, 10482, 10522, 10618, 10785, 11125, 11399]

fig = plt.figure(figsize=(10,10))
ax = plt.axes(projection='3d')
ax.grid()
ax.set_title('The "patch" point cloud')
# c='r' ; colour is RED, s=10 ; size is 10
ax.scatter(x, y, z, c='g', s=1)
#ax.scatter(xf, yf, zf, c='r', s=15)
ax.scatter(xn, yn, zn, c='b', s=5)
ax.set_xlabel('x', labelpad=20)
ax.set_ylabel('y', labelpad=20)
ax.set_zlabel('z', labelpad=20)
# ax.set_zlim3d(-0.0001, 0.0001)
ax.set_zlabel('zf', labelpad=20)

plt.show()

