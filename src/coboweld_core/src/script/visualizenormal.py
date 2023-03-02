'''
Try to visualize normals of a point cloud using open3d
'''
import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud('50x50new.pcd')

pcd_tree = o3d.geometry.KDTreeFlann(pcd)
pc_number = np.asarray(pcd.points).shape[0]

pcd.estimate_normals(
  search_param = o3d.geometry.KDTreeSearchParamHybrid(
  radius = 0.05, max_nn = 30
  )
)
pcd.normalize_normals()
pcd.orient_normals_towards_camera_location(camera_location = [0.0, 0.0, 0.0])

neighbours = min(pc_number//100, 30)

for index in range(pc_number):
  [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[index], neighbours)

  pcd.points[index][]

  n_on_line_list = 

o3d.visualization.draw_geometries([pcd], point_show_normal=True)