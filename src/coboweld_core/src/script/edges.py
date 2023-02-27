import open3d as o3d
import numpy as np
import pyransac3d as pyrsc
import math

pc = o3d.io.read_point_cloud('patch.pcd')

pc_kdtree = o3d.geometry.KDTreeFlann(pc)
count = np.asarray(pc.points).shape[0]

edge_points = []
neighbours = min(count//100, 30) # 30 neighbours

for index in range(count):
  # Find neighbours
  [k, idx, _] = pc_kdtree.search_knn_vector_3d(pc.points[index], neighbours)
  # Fit a plane
  plane1 = pyrsc.Plane()
  eq, liers = plane1.fit(np.asarray(pc.points)[idx[0:]], 0.002) # within 2mm
  # Use the plane equation to form the RANSAC-normal
  local_normal = eq[0:3] # Coefficients of the plane equation
  inliers = liers.shape[0]
  # If the fitted plane has less than 3 points, it is not a plane
  if (pc.points[index] not in np.asarray(pc.points)[liers[0:]]) or (inliers < 3):
    continue
  # Construct coordinate frame u, v
  # Let the first inlier be the origin, the 2nd inlier - the origin gives the first vector (u)
  # The normal cross the first vector becomes the second vector (v)
  # Find all the Angular Gap theta.
  u = pc.points[index] - np.asarray(pc.points)[liers[0]]
  v = np.cross(local_normal, u)
  theta = []
  for ilr in range(inliers):
    if (liers[ilr] != index) :
      opi = pc.points[liers[ilr]] - pc.points[index]
      diu = np.dot(opi, u)
      div = np.dot(opi, v)
      theta.append(math.atan(diu/div))
  g_theta = []
  for ilr in range(inliers-1):
    g_theta.append(theta[ilr+1] - theta[ilr])
  g_theta = np.asarray(g_theta)
  g_theta = max(g_theta[:])
  if (g_theta >= math.pi/2):
    edge_points.append(index)

print(edge_points)

