'''
Try to visualize normals of a point cloud using open3d
'''
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

pcd = o3d.io.read_point_cloud('testnormals.pcd')

pcd_tree = o3d.geometry.KDTreeFlann(pcd)
pc_number = np.asarray(pcd.points).shape[0]

'''
pcd.estimate_normals(
  search_param = o3d.geometry.KDTreeSearchParamHybrid(
  radius = 0.01, max_nn = 50
  )
)
pcd.normalize_normals()
pcd.orient_normals_towards_camera_location(camera_location = [0.0, 0.0, 0.0])

n_list = np.asarray(pcd.normals)


feature_value_list = []

#neighbours = min(pc_number//100, 30)

'''
neighbours = 30

for index in range(pc_number):
  [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[index], neighbours)

  # vector = np.mean(n_list[idx, :], axis=0)
  '''
  feature_value = np.linalg.norm(
    vector - (n_list[index, :] * np.dot(vector, n_list[index, :]))
  )
  '''
  # find the number of points within 5mm radius
  
  q_pt = np.asarray(pcd.points)[index] # query point position
  nr_pts = 0 # number of neighbouring points
  for cnt in idx:
    cnt_pt = np.asarray(pcd.points)[cnt]
    if (np.linalg.norm(q_pt - cnt_pt) < 0.005): 
      nr_pts += 1
  
  # np.asarray(pcd.points)[index, 2]=
  #feature_value_list.append(feature_value)


o3d.visualization.draw_geometries([pcd], width = 600, height = 600, point_show_normal=True)